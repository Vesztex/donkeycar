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
LAP_PCT_L = 0.25
LAP_PCT_R = 1.0

# AUTOPILOT
AI_THROTTLE_MULT = 1.0

