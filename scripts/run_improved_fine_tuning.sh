#!/bin/bash
# run_improved_fine_tuning.sh
#
# This script runs the improved video loading test and then executes
# the fine-tuning process with the fixed patches using DeepSpeed.

set -e  # Exit on error

# Setup logging
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
OUTPUT_DIR="output/improved_vila_finetuned_${TIMESTAMP}"
LOG_FILE="${OUTPUT_DIR}/training.log"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set up logging to file and console
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Starting fine-tuning process..."
echo "Using output directory: $OUTPUT_DIR"

# Ensure we're in the conda environment
echo "Activating vila-finetune conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vila-finetune || { echo "Failed to activate conda environment"; exit 1; }

# Set the current directory as the working directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"
echo "Working directory: $(pwd)"

# --- Configuration and Validation --- #
# Define original config and data paths (adjust if needed)
ORIG_CONFIG="configs/fine_tuning_config.yaml"
ORIG_DATA_JSON="data/vila-dataset/NW-test-camera-1/data_vila_format.json"
ORIG_DEEPSPEED_CONFIG="configs/deepspeed/zero3_config.json"
BASE_DATA_DIR="data/vila-dataset/"

# Define paths within the output directory
COPIED_CONFIG="$OUTPUT_DIR/fine_tuning_config.yaml" # Renamed from test_...
COPIED_DEEPSPEED_CONFIG="$OUTPUT_DIR/deepspeed_config.json" # Renamed from test_...
VALIDATED_DATA_JSON="$OUTPUT_DIR/validated_data.json"

# Copy necessary config files to output directory
echo "Setting up output directory..."
cp "$ORIG_CONFIG" "$COPIED_CONFIG"
cp "$ORIG_DEEPSPEED_CONFIG" "$COPIED_DEEPSPEED_CONFIG"

# Validate the data manifest
echo "======================================================="
echo "Validating data manifest: $ORIG_DATA_JSON"
echo "======================================================="
# Pass the original config path to the validation script
python scripts/validate_manifest.py \
    --input-json "$ORIG_DATA_JSON" \
    --data-dir "$BASE_DATA_DIR" \
    --output-json "$VALIDATED_DATA_JSON" \
    --config-path "$ORIG_CONFIG"

if [ $? -ne 0 ]; then
    echo "Data manifest validation failed. Aborting."
    exit 1
fi

# Modify the copied config to use the validated data manifest
echo "Updating config file ($COPIED_CONFIG) to use validated data ($VALIDATED_DATA_JSON) using Python..."

# Use Python to safely update the YAML file
python -c "
import yaml
import sys

config_path = '${COPIED_CONFIG}'
validated_data_path = '${VALIDATED_DATA_JSON}'

try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
except Exception as e:
    print(f'Error reading YAML file {config_path}: {e}', file=sys.stderr)
    sys.exit(1)

updated = False
if config.get('data') and isinstance(config['data'].get('train_data'), list):
    for item in config['data']['train_data']:
        if isinstance(item, dict) and 'data_vila_format.json' in item.get('path', ''):
            item['path'] = validated_data_path
            updated = True
            print(f'Updated train_data path to: {validated_data_path}')
            break # Assuming only one entry needs updating

# Also update validation data path if it exists and points to the original file
if config.get('data') and isinstance(config['data'].get('validation_data'), list):
    for item in config['data']['validation_data']:
        if isinstance(item, dict) and 'data_vila_format.json' in item.get('path', ''):
            item['path'] = validated_data_path
            updated = True
            print(f'Updated validation_data path to: {validated_data_path}')
            break # Assuming only one entry needs updating

if not updated:
    print(f'Warning: Did not find path containing \'data_vila_format.json\' to update in {config_path}', file=sys.stderr)

try:
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f'Successfully updated {config_path}')
except Exception as e:
    print(f'Error writing updated YAML file {config_path}: {e}', file=sys.stderr)
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "Python script to update YAML config failed. Aborting."
    exit 1
fi

# Create symbolic links for data and models in the output directory
# Use absolute paths for robustness
ln -sf "$PROJECT_ROOT/data" "$OUTPUT_DIR/data" # Link base data dir
ln -sf "$PROJECT_ROOT/models" "$OUTPUT_DIR/models"

# --- Video Loading Test --- #
# First, test the video loading to make sure the fixes work
echo "======================================================="
echo "Testing video loading with improved patches..."
echo "======================================================="
# Run test script from project root, referencing copied config
python scripts/test_video_loading.py --config "$COPIED_CONFIG"

# Check if the test succeeded
if [ $? -ne 0 ]; then
    echo "Video loading test failed. Aborting fine-tuning."
    exit 1
fi

echo "Video loading test passed. Proceeding with fine-tuning."

# --- Fine-tuning --- #
# Run the fine-tuning process with improved patches using DeepSpeed
echo "======================================================="
echo "Starting fine-tuning with improved patches using DeepSpeed..."
echo "======================================================="

# Define DeepSpeed arguments
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader)
DEEPSPEED_ARGS="--num_gpus=${NUM_GPUS}"

echo "Using ${NUM_GPUS} GPUs."
echo "DeepSpeed Config: ${COPIED_DEEPSPEED_CONFIG}"

# Use deepspeed launcher to run the script from the project root
# Pass the copied config paths and output dir as arguments to the script
deepspeed ${DEEPSPEED_ARGS} scripts/run_patched_finetune.py \
    --deepspeed "${COPIED_DEEPSPEED_CONFIG}" \
    --config "${COPIED_CONFIG}" \
    --output_dir "$OUTPUT_DIR" \
    --seed 42 \
    --gradient_checkpointing
    # local_rank is handled by deepspeed launcher

# Check if deepspeed command succeeded
if [ $? -ne 0 ]; then
    echo "DeepSpeed launcher failed. Check logs in $OUTPUT_DIR."
    exit 1
fi

echo "======================================================="
echo "DeepSpeed fine-tuning command executed."
echo "Check logs for actual training completion and success."
echo "======================================================="

# Summary
echo "Process completed. Check logs for details."
echo "Log file: $LOG_FILE"
echo "Output directory: $OUTPUT_DIR"

exit 0 