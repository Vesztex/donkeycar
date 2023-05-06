#!/bin/bash
# shell script to run training and simulation

# enforces Ctrl-C to stop the asynchronous processes
trap "kill 0" EXIT

# iterations
iter=10

# define repeatedly used variables
model_name="pilot_recursive"
model_full="models/${model_name}.savedmodel"
model_full_tflite="models/${model_name}.tflite"

clear_tub_command="rm -rf data_recursive"
copy_tub_command="cp -r data data_recursive"
drive_command="manage.py gym --type tflite_sq_mem_lap --model $model_full_tflite --random"
train_command="donkey train --model $model_full --type sq_mem_lap --transfer $model_full"

for (( i=0; i<iter; i++)); do
  echo "%%%%%%%%%%%%%%%%%-------*-*-*--------%%%%%%%%%%%%%%%"
  echo "Iteration $i start"
  echo $clear_tub_command
  $clear_tub_command
  echo $copy_tub_command
  $copy_tub_command
  echo $drive_command
  $drive_command
  echo $train_command
  $train_command

  model_name_i="${model_name}_${i}"
  # model_full_i="models/${model_name_i}.savedmodel"
  model_full_tflite_i="models/${model_name_i}.tflite"

  # cp $model_full $model_full_i
  cp $model_full_tflite $model_full_tflite_i
  echo "Iteration $i finished"
  echo "----------------------------------------------------"
  date
done

systemctl suspend

exit 0
