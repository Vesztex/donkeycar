# shell script to run training over a set of tubs for all 3 models

# define tubs using shell wildcard expansion
tub_range=46-62
tubs='data/tub_4[6-9]*norm*/,data/tub_[5-6]*norm*/'
tubs_expand=$tubs

# use transfer
use_transfer=1

# iterations
iter=2

# define repeatedly used variables
type=square_plus_imu.tflite
model_name=pilot_speed_imu_norm

for size in s m l; do
  # build the argument of the train job
  model_full_name="${model_name}_${size}_${tub_range}"
  model_arg="models/${model_full_name}.tflite"
  transfer_arg="models/${model_full_name}.h5"
  command_args="--type $type --model $model_arg --nn_size $size --tub $tubs_expand"
  if [ $use_transfer = 1 ]; then
      command_args="${command_args}  --transfer $transfer_arg";
  fi
  # run training iteratively
  for (( i=0; i<iter; i++))
  do
      train.py "$command_args"
  done


done

exit 0
