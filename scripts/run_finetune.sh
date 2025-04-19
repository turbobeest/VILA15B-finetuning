#!/bin/bash
# Shell script to run NVILA-15B fine-tuning with DeepSpeed

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vila-finetune

# Create output directory
OUTPUT_DIR="output/vila_finetuned_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Set up environment variables for DeepSpeed
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Run training script with DeepSpeed
deepspeed scripts/finetune_vila.py \
    --config configs/fine_tuning_config.yaml \
    --deepspeed configs/deepspeed/zero3_config.json \
    --output_dir $OUTPUT_DIR \
    --seed 42 \
    2>&1 | tee $OUTPUT_DIR/training.log

echo "Fine-tuning completed. Results saved to $OUTPUT_DIR" 