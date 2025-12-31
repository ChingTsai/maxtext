#!/bin/bash
set +x
# Define the arrays to iterate over
#PER_DEVICE_BATCH_SIZES=(1 2 4)
#HF_DATA_DIRS=("data" "data-ep2")

PER_DEVICE_BATCH_SIZES=(5)
HF_DATA_DIRS=("data")
GA_STEPS=(1)
TP=(1)
LR=("5e-6")

for data_dir in "${HF_DATA_DIRS[@]}"; do
  for bs in "${PER_DEVICE_BATCH_SIZES[@]}"; do
    for lr in "${LR[@]}"; do
      for tp in "${TP[@]}"; do
        for gas in "${GA_STEPS[@]}"; do
          export RUN_NAME="$(date +%Y-%m-%d-%H-%M)-bs${bs}-${data_dir}-gs${gas}-lr${lr}-tp${tp}"
          
          # Define log file name
          LOG_FILE="log_${RUN_NAME}.txt"

          {
            echo "------------------------------------------------"
            echo "Running: $RUN_NAME"
            echo "Batch Size: $bs | Data Dir: $data_dir | gradient_accumulation_steps: $gas | LR $lr | TP $tp"
            echo "------------------------------------------------"

            python3 -m MaxText.examples.grid MaxText/configs/sft.yml \
              run_name=$RUN_NAME \
              per_device_batch_size=$bs \
              hf_data_dir=$data_dir \
              gradient_accumulation_steps=$gas \
              learning_rate=$lr \
              ici_tensor_parallelism=$tp \
              save_checkpoint_on_completion=false \
              enable_data_shuffling=false
              
          } 2>&1 | tee "$LOG_FILE"  # Captures stdout and stderr, saves to file, and prints to console
        done
      done
    done
  done
done