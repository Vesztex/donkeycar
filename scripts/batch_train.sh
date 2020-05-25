#!/bin/bash
# shell script to run training over a set of tubs for all 3 models

# enforces Ctrl-C to stop the asynchronous processes
trap "kill 0" EXIT

# define tubs using shell wildcard expansion
tub_range=46-62
tubs='data/tub_{42..62}_*'
#exclude='aug'

# use transfer
use_transfer=1

# iterations
iter=2

# dry run?
dry=1

# define repeatedly used variables
type=square_plus_imu.tflite
model_name=pilot_speed_imu_norm

for size in s m l; do
  # build the argument of the train job
  model_full_name="${model_name}_${size}_${tub_range}"
  model_arg="models/${model_full_name}.tflite"
  transfer_arg="models/${model_full_name}.h5"
  command_args="--type $type --model $model_arg --nn_size $size --tub $tubs"
  if [ $use_transfer = 1 ]; then
      command_args="${command_args}  --transfer $transfer_arg";
  fi
  if [ $dry = 1 ]; then
      command_args="${command_args}  --dry";
  fi
  if [ -n "$exclude" ] ; then
      command_args="${command_args}  --exclude $exclude";
  fi
  # run training iteratively
  for (( i=0; i<iter; i++))
  do
      echo "Command args $command_args"
      train.py $command_args
  done

done

exit 0
