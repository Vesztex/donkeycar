#!/usr/bin/env python3
"""
Script to drive a donkey 2 car using the RC controller instead of the web
controller. Also provide a calibration of the RC throttle and
steering triggers.

Usage:
    manage.py (drive) [--pid] [--no_cam] [--model=<path_to_pilot>] [--web]\
 [--fpv] [--no_tub] [--verbose] [--type=<model_type>]
    manage.py (calibrate)
    manage.py (stream)

Options:
    -h --help        Show this screen.
"""
from docopt import docopt
import logging
import donkeycar as dk
from donkeycar.parts.camera import PiCamera, FrameStreamer
from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle, \
    RCReceiver, ModeSwitch
from donkeycar.parts.tub_v2 import TubWiper, TubWriter
from donkeycar.parts.clock import Timestamp
from donkeycar.parts.file_watcher import FileWatcher
from donkeycar.parts.keras_2 import ModelLoader
from donkeycar.parts.transform import SimplePidController, \
    ImuCombinerNormaliser, ThrottleSwitch, \
    SpeedRescaler, RecordingCondition
from donkeycar.parts.sensor import Odometer, LapTimer
from donkeycar.parts.controller import WebFpv
from donkeycar.parts.imu import Mpu6050Ada
from donkeycar.pipeline.augmentations import ImageAugmentation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Renamer:
    def run(self, data):
        return data


# define some strings that are used in the vehicle data flow
CAM_IMG = 'cam/image_array'


def drive(cfg, use_pid=False, no_cam=False, model_path=None, model_type=None,
          web=False, fpv=False, no_tub=False, verbose=False):
    """
    Construct a working robotic vehicle from many parts. Each part runs as a job
    in the Vehicle loop, calling either its run or run_threaded method depending
    on the constructor flag `threaded`. All parts are updated one after another
    at the frame rate given in cfg.DRIVE_LOOP_HZ assuming each part finishes
    processing in a timely manner. Parts may have named outputs and inputs. The
    framework handles passing named outputs to parts requesting the same named
    input.
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG, force=True)

    if no_cam:
        assert model_path is None, "Can't drive with pilot but w/o camera"

    if model_path is not None:
        use_pid = True

    car = dk.vehicle.Vehicle()
    clock = Timestamp()
    car.add(clock, outputs=['timestamp'])

    # handle record on ai: only record if cam is on and no auto-pilot ----------
    record_on_ai = getattr(cfg, 'RECORD_DURING_AI', False)
    # reduce car and camera frequency if we record on AI driving
    car_frequency = cfg.DRIVE_LOOP_HZ
    frame_rate = cfg.CAMERA_FRAMERATE
    if model_path is not None and record_on_ai \
            and hasattr(cfg, 'FREQ_REDUCTION_WITH_AI'):
        car_frequency = int(cfg.FREQ_REDUCTION_WITH_AI * car_frequency)
        frame_rate = int(cfg.FREQ_REDUCTION_WITH_AI * frame_rate)

    # add camera ---------------------------------------------------------------
    if not no_cam:
        cam = PiCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H,
                       image_d=cfg.IMAGE_DEPTH, framerate=frame_rate)
        car.add(cam, outputs=[CAM_IMG], threaded=True)

    # add odometer -------------------------------------------------------------
    odo = Odometer(gpio=cfg.ODOMETER_GPIO,
                   tick_per_meter=cfg.TICK_PER_M,
                   weight=0.025)
    car.add(odo, outputs=['car/speed', 'car/inst_speed', 'car/distance'])

    # add lap timer ------------------------------------------------------------
    lap = LapTimer(gpio=cfg.LAP_TIMER_GPIO, trigger=4)
    car.add(lap, inputs=['car/distance'], outputs=['car/lap', 'car/m_in_lap'],
            threaded=True)

    # add mpu ------------------------------------------------------------------
    mpu = Mpu6050Ada()
    car.add(mpu, outputs=['car/accel', 'car/gyro'], threaded=True)

    # add fpv parts ------------------------------------------------------------
    if web:
        car.add(WebFpv(), inputs=[CAM_IMG], threaded=True)
    if fpv:
        streamer = FrameStreamer(cfg.PC_HOSTNAME, cfg.FPV_PORT)
        car.add(streamer, inputs=[CAM_IMG], threaded=True)

    # create the RC receiver with 3 channels------------------------------------
    rc_steering = RCReceiver(cfg.STEERING_RC_GPIO, invert=True)
    rc_throttle = RCReceiver(cfg.THROTTLE_RC_GPIO)
    rc_wiper = RCReceiver(cfg.DATA_WIPER_RC_GPIO, jitter=0.05, no_action=0)
    car.add(rc_steering, outputs=['user/angle', 'user/angle_on'])
    car.add(rc_throttle, outputs=['user/throttle', 'user/throttle_on'])
    car.add(rc_wiper, outputs=['user/wiper', 'user/wiper_on'])

    # load model if present ----------------------------------------------------
    if model_path is not None:
        logger.info("Using auto-pilot")
        if not model_type:
            model_type = 'tflite_linear'

        kl = dk.utils.get_model_by_type(model_type, cfg)
        kl.load(model_path)
        kl_inputs = [CAM_IMG]
        # Add image transformations like crop or trapezoidal mask
        if hasattr(cfg, 'TRANSFORMATIONS') and cfg.TRANSFORMATIONS:
            outputs = ['cam/image_array_aug']
            car.add(ImageAugmentation(cfg, 'TRANSFORMATIONS'),
                    inputs=kl_inputs, outputs=outputs)
            kl_inputs = outputs
        # imu transformation and addition AI input -----------------------------
        use_imu = 'imu' in model_path
        if use_imu:
            logger.info('Using IMU in pilot')
            imu_prep = ImuCombinerNormaliser(cfg)
            car.add(imu_prep, inputs=['car/accel', 'car/gyro'],
                    outputs=['car/imu'])
            kl_inputs.append('car/imu')

        kl_outputs = ['pilot/angle', 'pilot/throttle']
        car.add(kl, inputs=kl_inputs, outputs=kl_outputs)
        # add file watcher and model loader so model can be reloaded
        f = FileWatcher(model_path)
        car.add(f, outputs=['model/update'])
        ml = ModelLoader(kl, model_path=model_path)
        car.add(ml, inputs=['model/update'], outputs=['model/loaded'],
                threaded=True)

        # if driving w/ ai switch between user throttle or pilot throttle by
        # pressing channel 3 on the remote control we have 3 modes,
        # user/steering + user/speed, pilot/steering + user/speed,
        # pilot/steering + pilot/speed
        mode_switch = ModeSwitch(num_modes=2)
        car.add(mode_switch, inputs=['user/wiper_on'], outputs=['user/mode'])

        # This part dispatches between user or ai depending on the switch state
        switch = ThrottleSwitch(cfg)
        car.add(switch, inputs=['user/mode', 'user/throttle', 'pilot/throttle'],
                outputs=['throttle'])
    else:
        # rename the usr throttle
        car.add(Renamer(), inputs=['user/throttle'], outputs=['throttle'])
    # drive by pid w/ speed
    if use_pid:
        # use pid either for rc control output or for ai output
        # convert throttle to speed
        car.add(SpeedRescaler(cfg), inputs=['throttle'], outputs=['speed'])
        # add pid controller to convert throttle value into speed
        pid = SimplePidController(p=cfg.PID_P, i=cfg.PID_I, d=cfg.PID_D)
        car.add(pid, inputs=['speed', 'car/inst_speed'],
                outputs=['pid/throttle'])

    # create and add the PWM steering controller
    steering_controller = PCA9685(cfg.STEERING_CHANNEL)
    steering = PWMSteering(controller=steering_controller,
                           left_pulse=cfg.STEERING_LEFT_PWM,
                           right_pulse=cfg.STEERING_RIGHT_PWM)

    steering_in = 'pilot/angle' if model_path else 'user/angle'
    car.add(steering, inputs=[steering_in], threaded=True)

    # create and add the PWM throttle controller for esc
    throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL)
    throttle = PWMThrottle(controller=throttle_controller,
                           max_pulse=cfg.THROTTLE_FORWARD_PWM,
                           zero_pulse=cfg.THROTTLE_STOPPED_PWM,
                           min_pulse=cfg.THROTTLE_REVERSE_PWM)
    # feed signal which is either rc (user) or ai
    throttle_input = 'pid/throttle' if use_pid else 'throttle'
    car.add(throttle, inputs=[throttle_input], threaded=True)

    # if we want to record a tub -----------------------------------------------
    if not no_cam and (model_path is None or record_on_ai) and not no_tub:
        static_condition = None if model_path is None else record_on_ai
        rec_cond = RecordingCondition(static_condition=static_condition)
        rec_inputs = ['user/throttle_on', 'user/throttle'] \
            if model_path is None else ['dummy', 'pilot_or_user/speed']
        car.add(rec_cond, inputs=rec_inputs, outputs=['recording'])

        # add tub to save data
        inputs = [CAM_IMG,
                  'user/angle', 'user/throttle', 'pilot/angle',
                  'pilot/throttle',
                  'user/mode', 'car/speed', 'car/inst_speed', 'car/distance',
                  'car/m_in_lap', 'car/lap', 'car/accel', 'car/gyro',
                  'timestamp']
        types = ['image_array', 'float', 'float', 'float', 'float',
                 'int', 'float', 'float', 'float',
                 'float', 'int', 'vector', 'vector',
                 'str']

        # multiple tubs
        tub_writer = TubWriter(base_path=cfg.DATA_PATH, inputs=inputs,
                               types=types)
        car.add(tub_writer, inputs=inputs, outputs=["tub/num_records"],
                run_condition='recording')

        # add a tub wiper that is triggered by channel 3 on the RC, but only
        # if we don't use channel 3 for switching between ai & manual
        if model_path is None:
            tub_wiper = TubWiper(tub_writer.tub, num_records=car_frequency)
            car.add(tub_wiper, inputs=['user/wiper_on'])
        elif record_on_ai:
            class Combiner:
                def run(self, signal, mode):
                    """ Clean at switching back from auto pilot b/c there was a
                    mistake """
                    return signal and mode == 0
            car.add(Combiner(), inputs=['user/wiper_on', 'user/mode'],
                    outputs=['user/clean'])
            # delete last two seconds when we go from pilot speed to manual
            tub_wiper = TubWiper(tub_writer.tub, num_records=2 * car_frequency)
            car.add(tub_wiper, inputs=['user/clean'])

    # run the vehicle
    car.start(rate_hz=car_frequency, max_loop_count=cfg.MAX_LOOPS,
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
    # run the vehicle at 10Hz to keep network traffic down
    donkey_car.start(rate_hz=10, max_loop_count=cfg.MAX_LOOPS)


def stream(cfg):
    car = dk.vehicle.Vehicle()
    clock = Timestamp()
    car.add(clock, outputs=['timestamp'])
    hz = 20
    cam = PiCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H,
                   image_d=cfg.IMAGE_DEPTH, framerate=hz)
    car.add(cam, outputs=['cam/image_array'], threaded=True)
    streamer = FrameStreamer(cfg.PC_HOSTNAME, cfg.FPV_PORT)
    car.add(streamer, inputs=['cam/image_array'], threaded=True)
    car.start(rate_hz=hz, max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':
    args = docopt(__doc__)
    config = dk.load_config()
    if args['drive']:
        drive(cfg=config,
              use_pid=args['--pid'],
              no_cam=args['--no_cam'],
              model_path=args['--model'],
              web=args['--web'],
              fpv=args['--fpv'],
              no_tub=args['--no_tub'],
              verbose=args['--verbose'],
              model_type=args['--type'])
    elif args['calibrate']:
        calibrate(config)
    elif args['stream']:
        stream(config)
