# """ 
# My CAR CONFIG 

# This file is read by your car application's manage.py script to change the car
# performance

# If desired, all config overrides can be specified here. 
# The update operation will not touch this file.
# """

# VEHICLE
DRIVE_LOOP_HZ = 40
FREQ_REDUCTION_WITH_AI = 1.0

# CAMERA
IMAGE_W = 192
IMAGE_H = 144
CAMERA_FRAMERATE = DRIVE_LOOP_HZ

# STEERING
STEERING_CHANNEL = 1
STEERING_LEFT_PWM = 270
STEERING_RIGHT_PWM = 480
STEERING_RC_GPIO = 26

# THROTTLE
THROTTLE_CHANNEL = 0
THROTTLE_FORWARD_PWM = 490
THROTTLE_STOPPED_PWM = 375
THROTTLE_REVERSE_PWM = 280
THROTTLE_RC_GPIO = 20

# DATA WIPER
DATA_WIPER_RC_GPIO = 19

# PID CONTROLLER
PID_P = 0.05
PID_I = 0.3
PID_D = 0.0010  # 0.0005

# ODOMETER
MAX_SPEED = 4.4
ODOMETER_GPIO = 18
TICK_PER_M = 630

# LAP TIMER
LAP_TIMER_GPIO = 16

# IMU VALUES
IMU_ACCEL_NORM = 20
IMU_GYRO_NORM = 250

# TUB SETTINGS
RECORD_DURING_AI = False

# TRAINING
TRAIN_TEST_SPLIT = 0.9
EARLY_STOP_PATIENCE = 10
MAX_EPOCHS = 200
# MIN_DELTA = 0.000001
USE_SPEED_FOR_MODEL = True
# IMU_DIM = 2
NN_SIZE = 'R'
BATCH_SIZE = 1024

# To suppress records when standing or just starting.
def filter_record(record):
    return record.underlying['car/speed'] > 0.5
TRAIN_FILTER = filter_record

CACHE_IMAGES = False
FREEZE_LAYERS = False
NUM_LAST_LAYERS_TO_TRAIN = None  # important to overwrite default of 7
# EXCLUDE_SLOW_LAPS = 0.5
SORT_LAPS_BY = 'lap_time'  # gyro_z, accel_x
# ROI_CROP_TOP = 36
# USE_TENSORBOARD = True
SEQUENCE_LENGTH = 3
SEQUENCE_TRAIN_STEP_SIZE = 1
ENCODER_PATH = 'models/encoder.h5'
OVERWRITE_LATENT = False

# PI DATA
PI_HOSTNAME = 'donkeypi.local'
PI_USERNAME = 'pi'

# AUTOPILOT
AI_THROTTLE_MULT = 1.0

# FPV MONITOR
PC_HOSTNAME = "DirksMacBook.home"
FPV_PORT = 13000
