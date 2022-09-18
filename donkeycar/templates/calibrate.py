#!/usr/bin/env python3
"""
Calibrate steering / throttle using the webserver

Usage:
    calibrate.py

Options:
    -h --help          Show this screen.
"""

from docopt import docopt

import donkeycar as dk
from donkeycar.parts.web_controller.web import LocalWebController
from donkeycar.parts.throttle_filter import ThrottleFilter
from donkeycar.parts import pins, PWMSteeringCalibrator, PWMThrottleCalibrator
from socket import gethostname


def main(cfg):
    """
    Construct a working robotic vehicle from many parts. Each part runs as a
    job in the Vehicle loop, calling either it's run or run_threaded method
    depending on the constructor flag `threaded`. All parts are updated one
    after another at the frame rate given in cfg.DRIVE_LOOP_HZ assuming each
    part finishes processing in a timely manner. Parts may have named outputs
    and inputs. The framework handles passing named outputs to parts
    requesting the same named input.
    """

    # Initialize car
    V = dk.vehicle.Vehicle()

    ctr = LocalWebController(port=cfg.WEB_CONTROL_PORT)
    V.add(ctr,
          inputs=['cam/image_array', 'tub/num_records'],
          outputs=['angle', 'throttle', 'user/mode', 'recording', 'config'],
          threaded=True)

    # this throttle filter will allow one tap back for esc reverse
    th_filter = ThrottleFilter()
    V.add(th_filter, inputs=['throttle'], outputs=['throttle'])

    # Drive train setup
    if cfg.DRIVE_TRAIN_TYPE == "PWM_STEERING_THROTTLE":
        #
        # drivetrain for RC car with servo and ESC.
        # using a PwmPin for steering (servo)
        # and as second PwmPin for throttle (ESC)
        #
        from donkeycar.parts.actuator \
            import PWMSteering, PWMThrottle, PulseController
        dt = cfg.PWM_STEERING_THROTTLE
        steering_controller = PulseController(
            pwm_pin=pins.pwm_pin_by_id(dt["PWM_STEERING_PIN"]),
            pwm_scale=dt["PWM_STEERING_SCALE"],
            pwm_inverted=dt["PWM_STEERING_INVERTED"])
        steering = PWMSteering(controller=steering_controller,
                               left_pulse=dt["STEERING_LEFT_PWM"],
                               right_pulse=dt["STEERING_RIGHT_PWM"])

        throttle_controller = PulseController(
            pwm_pin=pins.pwm_pin_by_id(dt["PWM_THROTTLE_PIN"]),
            pwm_scale=dt["PWM_THROTTLE_SCALE"],
            pwm_inverted=dt['PWM_THROTTLE_INVERTED'])
        throttle = PWMThrottle(controller=throttle_controller,
                               max_pulse=dt['THROTTLE_FORWARD_PWM'],
                               zero_pulse=dt['THROTTLE_STOPPED_PWM'],
                               min_pulse=dt['THROTTLE_REVERSE_PWM'])

        steering_calib = PWMSteeringCalibrator(pwm_steering=steering)
        throttle_calib = PWMThrottleCalibrator(pwm_throttle=throttle)

        V.add(steering, inputs=['angle'], threaded=True)
        V.add(throttle, inputs=['throttle'], threaded=True)
        V.add(steering_calib, inputs=['config'])
        V.add(throttle_calib, inputs=['config'])

    elif cfg.DRIVE_TRAIN_TYPE == "I2C_SERVO":
        from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle
        steering_controller = PCA9685(cfg.STEERING_CHANNEL,
                                      cfg.PCA9685_I2C_ADDR,
                                      busnum=cfg.PCA9685_I2C_BUSNUM)
        steering = PWMSteering(controller=steering_controller,
                               left_pulse=cfg.STEERING_LEFT_PWM,
                               right_pulse=cfg.STEERING_RIGHT_PWM)

        throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL,
                                      cfg.PCA9685_I2C_ADDR,
                                      busnum=cfg.PCA9685_I2C_BUSNUM)
        throttle = PWMThrottle(controller=throttle_controller,
                               max_pulse=cfg.THROTTLE_FORWARD_PWM,
                               zero_pulse=cfg.THROTTLE_STOPPED_PWM,
                               min_pulse=cfg.THROTTLE_REVERSE_PWM)

        steering_calib = PWMSteeringCalibrator(pwm_steering=steering)
        throttle_calib = PWMThrottleCalibrator(pwm_throttle=throttle)

        V.add(steering, inputs=['angle'], threaded=True)
        V.add(throttle, inputs=['throttle'], threaded=True)
        V.add(steering_calib, inputs=['config'])
        V.add(throttle_calib, inputs=['config'])

    elif cfg.DRIVE_TRAIN_TYPE == "MM1":
        from donkeycar.parts.robohat import RoboHATDriver, RoboHATDriverCalibrator
        drive_train = RoboHATDriver(cfg)
        calibrator = RoboHATDriverCalibrator(drive_train)
        V.add(drive_train, inputs=['angle', 'throttle'])
        V.add(calibrator, inputs=['config'])

    print(f"Go to http://{gethostname()}.local:{ctr.port}/calibrate to "
          f"calibrate steering and throttle")

    V.start(rate_hz=cfg.DRIVE_LOOP_HZ, max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()
    main(cfg)
