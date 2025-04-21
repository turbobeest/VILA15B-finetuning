# VILA-15B Production Environment

This directory contains the production environment for VILA-15B fine-tuning system.

## Directory Structure

- `data/`: Contains all data used by the system
  - `data/raw/`: Upload point for new datasets (CCTV footage, etc.)
  - `data/processed/`: Preprocessed data ready for fine-tuning
  - `data/validation/`: Validation datasets

- `models/`: Contains all model files
  - `models/base/`: Base NVILA-15B model files
  - `models/finetuned/`: Latest fine-tuned models
  - `models/versioned/`: Version-controlled model storage

- `config/`: Configuration files
  - Training parameters
  - System configuration
  - Environment settings

- `logs/`: System logs
  - Training logs
  - Validation logs
  - System operation logs

- `scripts/`: Production scripts
  - Data processing
  - Training
  - Deployment

- `services/`: Service definitions
  - Data ingestion service
  - Fine-tuning service
  - Model serving

## Data Flow

1. New data is uploaded to `data/raw/`
2. Validation scripts verify data integrity
3. Preprocessing converts raw data to `data/processed/`
4. Fine-tuning uses processed data to update models
5. New models are versioned and stored in `models/versioned/`
6. Latest models are symlinked to `models/finetuned/`

## Setup

This directory structure was created by running `setup_production_structure.sh`.
