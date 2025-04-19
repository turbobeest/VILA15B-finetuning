# NVILA-15B Fine-tuning Implementation Guide

This document explains the implementation of the fine-tuning script for NVILA-15B on CCTV footage, following the strategy defined in `docs/fine_tuning_strategy.md`.

## Components Overview

The fine-tuning implementation consists of the following components:

1. **Configuration Files**:
   - `configs/fine_tuning_config.yaml`: Main configuration file defining all hyperparameters, data paths, and training settings
   - `configs/deepspeed/zero3_config.json`: DeepSpeed configuration for distributed training with ZeRO-3

2. **Python Scripts**:
   - `scripts/finetune_vila.py`: Primary fine-tuning script that implements the training pipeline
   - `scripts/test_finetune.py`: Test script to verify that all components load correctly before full training

3. **Shell Scripts**:
   - `scripts/run_finetune.sh`: Wrapper script to run the fine-tuning with proper environment setup and logging

## Implementation Details

### Configuration System

The configuration system uses YAML to define all parameters, making it easy to modify settings without changing code. Key configuration sections include:

- **Model Configuration**: Path to the NVILA-15B model
- **Data Configuration**: Paths to datasets, split strategy, and video frame sampling
- **Training Hyperparameters**: Learning rate, batch size, optimizer settings, etc.
- **LoRA Configuration**: Settings for parameter-efficient fine-tuning
- **Evaluation Metrics**: Metrics to track during training

### Data Handling

The implementation handles data in the following way:

1. Loads the VILA-formatted data from JSON files
2. Randomly splits the data into training and validation sets based on the configuration
3. Creates VILA-compatible datasets that handle video frame extraction
4. Manages training/validation batches during training

### Model Loading and Configuration

The script:

1. Loads the pre-trained NVILA-15B model (language model, vision tower, and projector)
2. Configures components to be fine-tuned based on settings (projector and language model)
3. Applies LoRA to the language model for parameter-efficient training
4. Sets up precision to bfloat16 for efficient training

### Training Process

The training process leverages HuggingFace's Trainer API with DeepSpeed integration:

1. Prepares training arguments from the configuration
2. Creates a LLaVATrainer instance with the model, datasets, and arguments
3. Executes training with periodic evaluation and checkpointing
4. Saves the final model and evaluation metrics

### DeepSpeed Integration

DeepSpeed is configured for ZeRO-Stage 3 optimization:

- Memory optimization without parameter offloading
- Gradient accumulation for effective batch size
- BF16 precision for performance
- Cosine learning rate schedule with warmup

## Usage Instructions

### Prerequisites

Before running the fine-tuning:

1. Ensure the conda environment is properly set up:
   ```bash
   conda activate vila-finetune
   ```

2. Verify that the NVILA-15B model is present in the specified directory (default: `./models/NVILA-15B`)

3. Confirm that the dataset files are in the expected format and location:
   ```
   data/vila-dataset/NW-test-camera-1/data_vila_format.json
   ```

### Testing the Setup

It's recommended to run the test script first to verify that all components load correctly:

```bash
./scripts/test_finetune.py --config configs/fine_tuning_config.yaml
```

This will:
- Test model loading with the specified configuration
- Test dataset loading and preprocessing
- Verify that all components are compatible without starting actual training

### Running Fine-tuning

To start the full fine-tuning process:

```bash
./scripts/run_finetune.sh
```

This script will:
1. Activate the conda environment
2. Create a timestamped output directory
3. Run the fine-tuning with DeepSpeed
4. Log all outputs to a file in the output directory

### Monitoring Training

During training, you can monitor progress through:

1. Console output showing loss, learning rate, and other metrics
2. TensorBoard logs in the output directory
3. Checkpoint files saved according to the configured frequency

### Customizing Training

To modify the training behavior:

1. Edit `configs/fine_tuning_config.yaml` to change hyperparameters, data paths, etc.
2. Edit `configs/deepspeed/zero3_config.json` to adjust DeepSpeed settings if needed
3. For more advanced changes, modify the `scripts/finetune_vila.py` file directly

## Testing Procedure

The test script (`scripts/test_finetune.py`) performs the following validation:

1. **Configuration Loading**: Verifies that the YAML configuration can be parsed
2. **Model Loading**:
   - Checks if the model path exists
   - Loads the tokenizer and verifies it works
   - Loads the model configuration and applies custom settings
   - Initializes the model and applies LoRA if configured
3. **Dataset Loading**:
   - Verifies that dataset files exist and can be loaded
   - Creates a test dataset with a small subset of samples
   - Attempts to access an item from the dataset to verify processing
   - Checks if video frames are properly loaded

This testing ensures that the full training can start without immediate failures.

## Common Issues and Solutions

1. **Out of Memory Errors**:
   - Reduce batch size in the configuration
   - Increase gradient accumulation steps
   - Enable CPU offloading in DeepSpeed config (but be aware of performance impact)

2. **Data Loading Issues**:
   - Verify dataset paths in the configuration
   - Check that the data is in the correct VILA format
   - Ensure video files are accessible and in the expected format

3. **Model Loading Failures**:
   - Confirm that all model components are present in the specified directory
   - Check for any version incompatibilities between VILA and the model

4. **Training Instability**:
   - Reduce learning rate in the configuration
   - Increase warmup steps
   - Adjust gradient clipping value in DeepSpeed config

## Next Steps

After the fine-tuning script is implemented and validated (Task 5), the next steps in the project are:

1. **Run Initial Fine-tuning** (Task 6): Execute the script on the NW-test-camera-1 dataset
2. **Evaluate Fine-tuned Model** (Task 7): Assess performance on validation data
3. **End-to-End Testing** (Task 9): Test on the PI-test-camera-1 dataset for generalization

## Conclusion

This implementation provides a robust foundation for fine-tuning NVILA-15B on CCTV footage. The modular design allows for easy adaptation to different datasets or settings in the future. The focus on testing and validation ensures that potential issues are caught early before committing to full training runs. 