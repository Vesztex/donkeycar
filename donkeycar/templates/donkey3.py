#!/usr/bin/env python3
"""
Script to drive a donkey 2 car using the RC controller instead of the web
controller and to do a calibration of the RC throttle and steering triggers.

Usage:
    manage.py (drive) [--pid] [--no_cam] [--model=<path_to_pilot>] [--verbose]
    manage.py (calibrate)

Options:
    -h --help        Show this screen.
"""

from docopt import docopt
from simple_pid import PID

import donkeycar as dk
from donkeycar.parts.camera import PiCamera
from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle, \
    RCReceiver, ModeSwitch
from donkeycar.parts.datastore import TubWiper, TubHandler
from donkeycar.parts.clock import Timestamp
from donkeycar.parts.transform import PIDController
from donkeycar.parts.sensor import Odometer, LapTimer
from donkeycar.utils import normalize_and_crop


class TypePrinter:
    def __init__(self, type_name):
        self.type_name = type_name

    def run(self, in_type):
        print("Type of", self.type_name, type(in_type))


def drive(cfg, use_pid=False, no_cam=False, model_path=None, verbose=False):
    """
    Construct a working robotic vehicle from many parts. Each part runs as a job
    in the Vehicle loop, calling either its run or run_threaded method depending
    on the constructor flag `threaded`. All parts are updated one after another
    at the frame rate given in cfg.DRIVE_LOOP_HZ assuming each part finishes
    processing in a timely manner. Parts may have named outputs and inputs. The
    framework handles passing named outputs to parts requesting the same named
    input.
    """
    if no_cam:
        assert model_path is None, "Can't drive with pilot but w/o camera"

    car = dk.vehicle.Vehicle()

    clock = Timestamp()
    car.add(clock, outputs=['timestamp'])

    if not no_cam:
        cam = PiCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H,
                       image_d=cfg.IMAGE_DEPTH, framerate=cfg.CAMERA_FRAMERATE)
        car.add(cam, outputs=['cam/image_array'], threaded=True)

    odo = Odometer(gpio=cfg.ODOMETER_GPIO,
                   tick_per_meter=cfg.TICK_PER_M,
                   weight=1.0,
                   debug=verbose)
    car.add(odo, outputs=['car/speed'])
    lap = LapTimer(gpio=cfg.LAP_TIMER_GPIO)
    car.add(lap, outputs=['car/lap'], threaded=True)

    # create the RC receiver with 3 channels
    rc_steering = RCReceiver(cfg.STEERING_RC_GPIO, invert=True)
    rc_throttle = RCReceiver(cfg.THROTTLE_RC_GPIO)
    rc_wiper = RCReceiver(cfg.DATA_WIPER_RC_GPIO, jitter=0.05, no_action=0)
    car.add(rc_steering, outputs=['user/angle', 'user/steering_on'])
    car.add(rc_throttle, outputs=['user/throttle', 'user/throttle_on'])
    car.add(rc_wiper, outputs=['user/wiper', 'user/wiper_on'])

    # if pid we want to convert throttle to speed
    if use_pid:
        class Rescaler:
            def run(self, controller_input):
                return controller_input * cfg.MAX_SPEED

        car.add(Rescaler(), inputs=['user/throttle'], outputs=['user/speed'])

    # load model if present
    if model_path is not None:
        print("Using auto-pilot")
        model_type = 'tflite_linear' if '.tflite' in model_path else 'linear'
        kl = dk.utils.get_model_by_type(model_type, cfg)
        kl.load(model_path)

        class ImgPrecondition:
            def __init__(self, cfg):
                self.cfg = cfg

            def run(self, img_arr):
                return normalize_and_crop(img_arr, self.cfg)

        car.add(ImgPrecondition(cfg), inputs=['cam/image_array'],
                outputs=['cam/normalized/cropped'])

        outputs = ['pilot/angle', 'pilot/speed' if use_pid else 'pilot/throttle']
        car.add(kl, inputs=['cam/normalized/cropped'], outputs=outputs)
        mode_switch = ModeSwitch(num_modes=2)
        car.add(mode_switch, inputs=['user/wiper_on'], outputs=['user/mode'])

        # This part dispatches between user or ai depending on the switch state
        class PilotCondition:
            def run(self, user_mode, user_var, pilot_var):
                if user_mode == 0:
                    return user_var
                else:
                    return pilot_var

        # switch between user or pilot speed (if pid) or throttle (if no pid)
        var = 'speed' if use_pid else 'throttle'
        car.add(PilotCondition(),
                inputs=['user/mode', 'user/' + var, 'pilot/' + var],
                outputs=[var])

    # use pid either for rc control output or for ai output
    speed = 'speed' if model_path is not None else 'user/speed'
    # drive by pid w/ speed
    if use_pid:
        # add pid controller to convert throttle value into speed
        # pid = PIDController(p=cfg.PID_P, i=cfg.PID_I, d=cfg.PID_D, debug=False)
        class PidController:
            def __init__(self):
                self.pid = PID(Kp=cfg.PID_P, Ki=cfg.PID_I, Kd=cfg.PID_D)
                self.pid.output_limits = (0, None)

            def run(self, set_point, feedback):
                self.pid.setpoint = set_point
                if verbose:
                    print('setpoint{0:4.2f} feedback{1:4.2f}'.format(set_point,feedback))
                return self.pid(feedback)

        pid = PidController()
        car.add(pid, inputs=[speed, 'car/speed'], outputs=['throttle'])

    # create the PWM throttle controller for esc
    throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL)
    throttle = PWMThrottle(controller=throttle_controller,
                           max_pulse=cfg.THROTTLE_FORWARD_PWM,
                           zero_pulse=cfg.THROTTLE_STOPPED_PWM,
                           min_pulse=cfg.THROTTLE_REVERSE_PWM)
    # feed signal which is either rc (user) or ai
    input_field = 'user/throttle' if not use_pid and model_path is None \
        else 'throttle'
    car.add(throttle, inputs=[input_field], threaded=True)
    # create the PWM steering controller
    steering_controller = PCA9685(cfg.STEERING_CHANNEL)
    steering = PWMSteering(controller=steering_controller,
                           left_pulse=cfg.STEERING_LEFT_PWM,
                           right_pulse=cfg.STEERING_RIGHT_PWM)
    # feed signal which is either rc (user) or ai
    input_field = 'user/angle' if model_path is None else 'pilot/angle'
    car.add(steering, inputs=[input_field], threaded=True)

    # only record if cam is on and no auto-pilot
    record_on_ai = cfg.RECORD_DURING_AI if hasattr(cfg, 'RECORD_DURING_AI') \
        else False

    if not no_cam and (model_path is None or record_on_ai):
        class RecordingCondition:
            def run(self, throttle_on, throttle_val):
                return throttle_on and throttle_val > 0

        car.add(RecordingCondition(),
                inputs=['user/throttle_on', 'user/throttle'],
                outputs=['user/recording'])

        # add tub to save data
        inputs = ['cam/image_array', 'user/angle', 'user/throttle',
                  'car/speed', 'car/lap', 'timestamp']
        types = ['image_array', 'float', 'float', 'float', 'int', 'str']
        # add debug output to tub
        if verbose and use_pid:
            inputs += [speed, 'throttle']
            types += ['float', 'float']
        # multiple tubs
        tubh = TubHandler(path=cfg.DATA_PATH)
        tub = tubh.new_tub_writer(inputs=inputs,
                                  types=types,
                                  allow_reverse=False)
        car.add(tub,
                inputs=inputs,
                outputs=["tub/num_records"],
                run_condition='user/recording')

        # add a tub wiper that is triggered by channel 3 on the RC
        tub_wipe = TubWiper(tub, num_records=cfg.DRIVE_LOOP_HZ)
        car.add(tub_wipe, inputs=['user/wiper_on'])

    # run the vehicle
    car.start(rate_hz=cfg.DRIVE_LOOP_HZ, max_loop_count=cfg.MAX_LOOPS,
              verbose=verbose)


def calibrate(cfg):
    """
    Construct an auxiliary robotic vehicle from only the RC controllers and
    prints their values. The RC remote usually has a tuning pot for the throttle
    and steering channel. In this loop we run the controllers and simply print
    their values in order to allow centering the RC pwm signals. If there is a
    third channel on the remote we can use it for wiping bad data while
    recording, so we print its values here, too.
    """
    donkey_car = dk.vehicle.Vehicle()

    clock = Timestamp()
    donkey_car.add(clock, outputs=['timestamp'])

    # create the RC receiver
    rc_steering = RCReceiver(cfg.STEERING_RC_GPIO, invert=True)
    rc_throttle = RCReceiver(cfg.THROTTLE_RC_GPIO)
    rc_wiper = RCReceiver(cfg.DATA_WIPER_RC_GPIO, jitter=0.05, no_action=0)
    donkey_car.add(rc_steering, outputs=['user/angle', 'user/steering_on'])
    donkey_car.add(rc_throttle, outputs=['user/throttle', 'user/throttle_on'])
    donkey_car.add(rc_wiper, outputs=['user/wiper', 'user/wiper_on'])

    # create plotter part for printing into the shell
    class Plotter:
        def run(self, angle, steering_on, throttle, throttle_on, wiper, wiper_on):
            print('angle=%+5.4f, steering_on=%1d, throttle=%+5.4f, '
                  'throttle_on=%1d wiper=%+5.4f, wiper_on=%1d' %
                  (angle, steering_on, throttle, throttle_on, wiper, wiper_on))

    # add plotter part
    donkey_car.add(Plotter(), inputs=['user/angle', 'user/steering_on',
                                      'user/throttle', 'user/throttle_on',
                                      'user/wiper', 'user/wiper_on'])
    # run the vehicle at 5Hz to keep network traffic down
    donkey_car.start(rate_hz=10, max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':
    args = docopt(__doc__)
    config = dk.load_config()
    if args['drive']:
        drive(config,
              use_pid=args['--pid'],
              no_cam=args['--no_cam'],
              model_path=args['--model'],
              verbose=args['--verbose'])
    elif args['calibrate']:
        calibrate(config)
