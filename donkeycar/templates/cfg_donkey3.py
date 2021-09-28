""" 
CAR CONFIG 

This file is read by your car application's manage.py script to change the car
performance. 

"""


import os

# PATHS
CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CAR_PATH, 'data')
MODELS_PATH = os.path.join(CAR_PATH, 'models')

# VEHICLE
DRIVE_LOOP_HZ = 40
MAX_LOOPS = None

# CAMERA
CAMERA_TYPE = "PICAM"   # (PICAM|WEBCAM|CVCAM|CSIC|V4L|D435|MOCK|IMAGE_LIST)
IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3         # default RGB=3, make 1 for mono
CAMERA_FRAMERATE = DRIVE_LOOP_HZ

# 9865, over rides only if needed, ie. TX2..
PCA9685_I2C_ADDR = 0x40
PCA9685_I2C_BUSNUM = None

# STEERING
STEERING_CHANNEL = 1
STEERING_LEFT_PWM = 270
STEERING_RIGHT_PWM = 480

# THROTTLE
THROTTLE_CHANNEL = 0
THROTTLE_FORWARD_PWM = 490
THROTTLE_STOPPED_PWM = 375
THROTTLE_REVERSE_PWM = 280

# TRAINING
DEFAULT_AI_FRAMEWORK = 'tensorflow'
DEFAULT_MODEL_TYPE = 'linear'
CREATE_TF_LITE = True
CREATE_TENSOR_RT = False
BATCH_SIZE = 128
TRAIN_TEST_SPLIT = 0.9
MAX_EPOCHS = 200
SHOW_PLOT = False
VERBOSE_TRAIN = True
USE_EARLY_STOP = True
EARLY_STOP_PATIENCE = 10
MIN_DELTA = .000001
PRINT_MODEL_SUMMARY = True
USE_SPEED_FOR_MODEL = True
CACHE_IMAGES = True

# model transfer options
FREEZE_LAYERS = False
NUM_LAST_LAYERS_TO_TRAIN = 7

# For the categorical model, this limits the upper bound of the learned throttle
MODEL_CATEGORICAL_MAX_THROTTLE_RANGE = 0.5

# RNN or 3D
SEQUENCE_LENGTH = 3

# For MemoryLap model
LAP_PCT = 0.25

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

# RECORD OPTIONS
RECORD_DURING_AI = False
AUTO_CREATE_NEW_TUB = False

# WEB CONTROL
WEB_CONTROL_PORT = int(os.getenv("WEB_CONTROL_PORT", 8887))
WEB_INIT_MODE = "user"

# DRIVING
AI_THROTTLE_MULT = 1.0

# RPi
PI_USERNAME = "pi"
PI_HOSTNAME = "donkeypi.local"

# FPV MONITOR
PC_HOSTNAME = "DirksMacBook.home"
FPV_PORT = 13000