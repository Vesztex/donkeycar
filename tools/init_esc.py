#!/usr/bin/env python3

# This script is meant to running on the car. Optimally run it automatically at
# RPI startup. It is sending 0 throttle to car esc to avoid beeping
import os
import donkeycar as dk
from donkeycar.parts.actuator import PCA9685

# this needs to point to donkey's config
conf = os.path.expanduser('/home/pi/mycar/config.py')
cfg = dk.load_config(conf)
# this creates the pwm driver on the throttle channel and sends the
# initialisation pwm
channel = cfg.THROTTLE_CHANNEL
c = PCA9685(channel)
c.run(cfg.THROTTLE_STOPPED_PWM)
print('Sent zero throttle of', cfg.THROTTLE_STOPPED_PWM, 'to esc.')

