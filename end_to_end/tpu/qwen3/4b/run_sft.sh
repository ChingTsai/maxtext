#!/bin/bash

:'
# This script demonstrates a full end-to-end workflow for Supervised Fine-Tuning (SFT)
# a pre-trained model using MaxText. The script fine-tunes a pre-trained model using
# SFT on the `HuggingFaceH4/ultrachat_200k` dataset and produces a fine-tuned model
# in the Hugging Face format.
#
# This script supports two scenarios:
#
# 1. **Run SFT on a Hugging Face Checkpoint**
#    The script will automatically convert a Hugging Face checkpoint
#    to a MaxText checkpoint, run SFT, and then convert the fine-tuned
#    checkpoint back to the Hugging Face format.
#
# 2. **Run SFT on a MaxText Checkpoint**
#    The script will run SFT on a pre-converted MaxText checkpoint and
#    then convert the fine-tuned checkpoint back to the Hugging Face format.
#
# --- Common environment variables for both the scenarios ---
# export HF_TOKEN=<Hugging Face access token>
#
# # Output directory to store run logs
# export BASE_OUTPUT_DIRECTORY=<output directory>
#
# # Number of fine-tuning steps to run
# export STEPS=100
# export PER_DEVICE_BATCH_SIZE=1
#
# --- Scenario 1: Run SFT on a Hugging Face Checkpoint ---
# PRE_TRAINED_MODEL_CKPT_PATH should be unset for this scenario
# bash end_to_end/tpu/llama3.1/8b/run_sft.sh
#
# --- Scenario 2: Run SFT on a MaxText Checkpoint ---
# Set the GCS path to the pre-converted MaxText checkpoint
# export PRE_TRAINED_MODEL_CKPT_PATH=<gcs path for model checkpoint>
# bash end_to_end/tpu/llama3.1/8b/run_sft.sh
'

set -xe

BASE_OUTPUT_DIRECTORY="/mnt/disks/jimmy_workspace/maxtext_qwen-14b"
STEPS=100
PER_DEVICE_BATCH_SIZE=1

RUN_NAME=$(date +%Y-%m-%d-%H-%M-%S)
PRE_TRAINED_MODEL="qwen3-14b"
PRE_TRAINED_MODEL_TOKENIZER="Qwen/Qwen3-14B"
CONVERTED_CKPT_DIR="/home/jimmytsai_google_com/workspace/maxtext/src/MaxText/qwen_checkpoint-14b"

DATASET_NAME="/home/jimmytsai_google_com/workspace/tencent_sft/clid-data"
TRAIN_SPLIT="train"
HF_EVAL_SPLIT="evaluation"
HF_DATA_DIR="data"

export PRE_TRAINED_MODEL_CKPT_PATH=${CONVERTED_CKPT_DIR}/0/items
python3 -m MaxText.sft.sft_trainer "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}"/configs/sft.yml \
    run_name=${RUN_NAME} base_output_directory=${BASE_OUTPUT_DIRECTORY}/${PRE_TRAINED_MODEL} \
    model_name=${PRE_TRAINED_MODEL} load_parameters_path=${PRE_TRAINED_MODEL_CKPT_PATH} \
    hf_access_token=$HF_TOKEN tokenizer_path=${PRE_TRAINED_MODEL_TOKENIZER} \
    per_device_batch_size=${PER_DEVICE_BATCH_SIZE} steps=${STEPS} \
    train_split=${TRAIN_SPLIT} hf_eval_split=${HF_EVAL_SPLIT} hf_data_dir=${HF_DATA_DIR} \
    hf_path=${DATASET_NAME} \
    profiler=xplane max_target_length=1024 weight_dtype=bfloat16

# Get the latest fine-tuned model checkpoint
CHECKPOINTS_PATH=${BASE_OUTPUT_DIRECTORY}/${PRE_TRAINED_MODEL}/${RUN_NAME}/checkpoints
echo "Fine-tuned model checkpoint: ${CHECKPOINTS_PATH}"
