""" 
CAR CONFIG 

This file is read by the donkey commands for driving, training, etc.

"""

import os

# PATHS
CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CAR_PATH, 'data')
MODELS_PATH = os.path.join(CAR_PATH, 'models')
ASSEMBLY_PATH = os.path.join(CAR_PATH, 'assembly')
MYPARTS_PATH = os.path.join(CAR_PATH, 'myparts')

# VEHICLE
DRIVE_LOOP_HZ = 20
MAX_LOOPS = None

# CAMERA
CAMERA_TYPE = "PICAM"  # (PICAM|WEBCAM|CVCAM|CSIC|V4L|D435|MOCK|IMAGE_LIST)
IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3  # default RGB=3, make 1 for mono
CAMERA_FRAMERATE = DRIVE_LOOP_HZ

#
# PWM_STEERING_THROTTLE
#
# Drive train for RC car with a steering servo and ESC. Uses a PwmPin for
# steering (servo) and a second PwmPin for throttle (ESC) Base PWM frequency
# is presumed to be 60hz; use PWM_xxxx_SCALE to adjust pulse with for
# non-standard PWM frequencies
#
PWM_STEERING_THROTTLE = {
    "PWM_STEERING_PIN": "PCA9685.1:40.1",  # PWM output pin for steering servo
    "PWM_STEERING_SCALE": 1.0,
    # used to compensate for PWM frequency differents from 60hz; NOT for
    # adjusting steering range
    "PWM_STEERING_INVERTED": False,
    # True if hardware requires an inverted PWM pulse
    "PWM_THROTTLE_PIN": "PCA9685.1:40.0",  # PWM output pin for ESC
    "PWM_THROTTLE_SCALE": 1.0,
    # used to compensate for PWM frequence differences from 60hz; NOT for
    # increasing/limiting speed
    "PWM_THROTTLE_INVERTED": False,
    # True if hardware requires an inverted PWM pulse
    "STEERING_LEFT_PWM": 460,  # pwm value for full left steering
    "STEERING_RIGHT_PWM": 290,  # pwm value for full right steering
    "THROTTLE_FORWARD_PWM": 500,  # pwm value for max forward throttle
    "THROTTLE_STOPPED_PWM": 370,  # pwm value for no movement
    "THROTTLE_REVERSE_PWM": 220,  # pwm value for max reverse throttle
}

# LOGGING
HAVE_CONSOLE_LOGGING = True
# (Python logging level) 'NOTSET' / 'DEBUG' / 'INFO' / 'WARNING' / 'ERROR' /
# 'FATAL' / 'CRITICAL'
LOGGING_LEVEL = 'INFO'
# (Python logging format -
# https://docs.python.org/3/library/logging.html#formatter-objects
LOGGING_FORMAT = '%(message)s'

# TRAINING
# The default AI framework to use. Choose from (tensorflow|pytorch)
DEFAULT_AI_FRAMEWORK = 'tensorflow'
# (linear|categorical|rnn|imu|behavior|3d|localizer|latent)
DEFAULT_MODEL_TYPE = 'linear'
# automatically create tflite model in training
CREATE_TF_LITE = True
# automatically create tensorrt model in training
CREATE_TENSOR_RT = False
BATCH_SIZE = 128
TRAIN_TEST_SPLIT = 0.8
MAX_EPOCHS = 100
SHOW_PLOT = True
VERBOSE_TRAIN = True
USE_EARLY_STOP = True
EARLY_STOP_PATIENCE = 5
MIN_DELTA = .0005
PRINT_MODEL_SUMMARY = True  # print layers and weights to stdout
OPTIMIZER = None  # adam, sgd, rmsprop, etc.. None accepts default
LEARNING_RATE = 0.001  # only used when OPTIMIZER specified
LEARNING_RATE_DECAY = 0.0  # only used when OPTIMIZER specified

# model transfer options
FREEZE_LAYERS = False
NUM_LAST_LAYERS_TO_TRAIN = 7

# For the categorical model, this limits the upper bound of the learned
# throttle it's very IMPORTANT that this value is matched from the training
# PC config.py and the robot.py and ideally wouldn't change once set.
MODEL_CATEGORICAL_MAX_THROTTLE_RANGE = 0.8

# RNN or 3D
SEQUENCE_LENGTH = 3

# Augmentations and Transformations
AUGMENTATIONS = []
TRANSFORMATIONS = []
# Settings for brightness and blur, use 'MULTIPLY' and/or 'BLUR' in
# AUGMENTATIONS
AUG_MULTIPLY_RANGE = (0.5, 3.0)
AUG_BLUR_RANGE = (0.0, 3.0)
# Number of pixels to crop, requires 'CROP' in TRANSFORMATIONS to be set
ROI_CROP_TOP = 45
ROI_CROP_BOTTOM = 0
ROI_CROP_RIGHT = 0
ROI_CROP_LEFT = 0
# For trapezoidal see explanation in augmentations.py, requires 'TRAPEZE' in
# TRANSFORMATIONS to be set
ROI_TRAPEZE_LL = 0
ROI_TRAPEZE_LR = 160
ROI_TRAPEZE_UL = 20
ROI_TRAPEZE_UR = 140
ROI_TRAPEZE_MIN_Y = 60
ROI_TRAPEZE_MAX_Y = 120

# SOMBRERO
HAVE_SOMBRERO = False

# RECORD OPTIONS
RECORD_DURING_AI = False
# create a new tub (tub_YY_MM_DD) directory when recording or append records
# to data directory directly
AUTO_CREATE_NEW_TUB = False

# JOYSTICK -----------------------------
# when starting the manage.py, when True, will not require a --js
# option to use the joystick
USE_JOYSTICK_AS_DEFAULT = False
# this scalar is multiplied with the -1 to 1 throttle value to limit the
# maximum throttle. This can help if you drop the controller or just don't
# need the full speed available
JOYSTICK_MAX_THROTTLE = 0.5
# some people want a steering that is less sensitve. This scalar is
# multiplied with the steering -1 to 1. It can be negative to reverse dir.
JOYSTICK_STEERING_SCALE = 1.0
# if true, we will record whenever throttle is not zero. if false, you must
# manually toggle recording with some other trigger. Usually circle button on
# joystick.
AUTO_RECORD_ON_THROTTLE = True
# (ps3|ps4|xbox|nimbus|wiiu|F710|rc3|MM1|custom) custom will run the
# my_joystick.py controller written by the `donkey createjs` command
CONTROLLER_TYPE = 'ps3'
# should we listen for remote joystick control over the network?
USE_NETWORKED_JS = False
# when listening for network joystick control, which ip is serving this
# information
NETWORK_JS_SERVER_IP = "192.168.0.1"
# when non zero, this is the smallest throttle before recording triggered.
JOYSTICK_DEADZONE = 0.0
# use -1.0 to flip forward/backward, use 1.0 to use joystick's natural
# forward/backward
JOYSTICK_THROTTLE_DIR = -1.0
# send camera data to FPV webserver
USE_FPV = False
# this is the unix file use to access the joystick.
JOYSTICK_DEVICE_FILE = "/dev/input/js0"

# WEB CONTROL
# which port to listen on when making a web controller
WEB_CONTROL_PORT = int(os.getenv("WEB_CONTROL_PORT",8887))
# which control mode to start in. one of user|local_angle|local. Setting
# local will start in ai mode.
WEB_INIT_MODE = "user"

# DRIVING
# this multiplier will scale every throttle value for all output from NN models
AI_THROTTLE_MULT = 1.0
# DonkeyGym

# Only on Ubuntu linux, you can use the simulator as a virtual donkey and
# issue the same python manage.py drive command as usual, but have them
# control a virtual car. This enables that, and sets the path to the
# simulator and the environment. You will want to download the simulator
# binary from: https://github.com/tawnkramer/donkey_gym/releases/download/
# then extract that and modify DONKEY_SIM_PATH.

DONKEY_GYM = False
# For example
# "/home/tkramer/projects/sdsandbox/sdsim/build/DonkeySimLinux/donkey_sim.x86_64"
# when racing on virtual-race-league use "remote", or user "remote"
# when you want to start the sim manually first.
DONKEY_SIM_PATH = "path to sim"

# Either of "donkey-generated-track-v0", "donkey-generated-roads-v0",
# "donkey-warehouse-v0", "donkey-avc-sparkfun-v0"
DONKEY_GYM_ENV_NAME = "donkey-generated-track-v0"

GYM_CONF = {"body_style": "donkey",  # body style(donkey|bare|car01)
    "body_rgb": (128, 128, 128),  # body rgb 0-255
    "car_name": "car",
    "font_size": 100}
GYM_CONF["racer_name"] = "Your Name"
GYM_CONF["country"] = "Place"
GYM_CONF["bio"] = "I race robots."

# when racing on virtual-race-league use host "trainmydonkey.com"
SIM_HOST = "127.0.0.1"
# this is the millisecond latency in controls. Can use useful in emulating
# the delay when useing a remote server. values of 100 to 400 probably
# reasonable.
SIM_ARTIFICIAL_LATENCY = 0


