# shell script to run training over a set of tubs for all 3 models

# define tubs using shell wildcard expansion
tubs='data/tub_4[6-9]*norm*/,data/tub_[5-6]*norm*/'

# define repeatedly used variables
type=square_plus_imu.tflite

train.py --type $type --model models/pilot_speed_imu_norm_s_46-62.tflite --nn_size S --tub $tubs --transfer models/pilot_speed_imu_norm_s_46-62.h5
train.py --type $type --model models/pilot_speed_imu_norm_m_46-62.tflite --nn_size M --tub $tubs --transfer models/pilot_speed_imu_norm_m_46-62.h5
train.py --type $type --model models/pilot_speed_imu_norm_l_46-62.tflite --nn_size L --tub $tubs --transfer models/pilot_speed_imu_norm_l_46-62.h5
