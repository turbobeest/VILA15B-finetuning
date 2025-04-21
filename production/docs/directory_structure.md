# VILA-15B Production Directory Structure

## Directory Hierarchy

```
/production/
├── data/
│   ├── raw/              # Upload point for new datasets (CCTV footage, etc.)
│   ├── processed/        # Preprocessed data ready for fine-tuning
│   └── validation/       # Validation datasets
│
├── models/
│   ├── base/             # Base NVILA-15B model files
│   │   ├── llm/          # Language model component
│   │   ├── mm_projector/ # Multimodal projector component
│   │   └── vision_tower/ # Vision encoder component
│   │
│   ├── finetuned/        # Latest fine-tuned models (symlinks to versioned)
│   │   ├── latest -> ../versioned/vila-v1.2.3/
│   │   └── stable -> ../versioned/vila-v1.2.0/
│   │
│   └── versioned/        # Version-controlled model storage
│       ├── vila-v1.0.0/  # Initial fine-tuned version
│       ├── vila-v1.1.0/  # Feature update version
│       ├── vila-v1.2.0/  # Stable version
│       └── vila-v1.2.3/  # Latest version
│
├── config/
│   ├── production.yaml           # Main configuration file
│   ├── production-environment.yml # Conda environment specification
│   └── model-cards/              # Documentation for each model version
│
├── logs/
│   ├── training/         # Training logs
│   ├── validation/       # Validation logs
│   └── system/           # System operation logs
│
├── scripts/
│   ├── activate_prod_env.sh     # Environment activation script
│   ├── data_processing/         # Data processing scripts
│   ├── training/                # Training scripts
│   └── deployment/              # Deployment scripts
│
└── services/
    ├── data_ingestion/   # Data ingestion service
    ├── fine_tuning/      # Fine-tuning service
    ├── model_serving/    # Model serving
    └── vila/             # VILA package installation
```

## Directory Responsibilities

### 1. Data Directory

The `data` directory contains all datasets used by the system, organized into three categories:

- **Raw Data**: Original, unprocessed data uploaded to the system
  - Supported formats: MP4 videos, JPG/PNG images
  - Directory watched by data ingestion service
  
- **Processed Data**: Data that has been validated and preprocessed
  - Properly formatted for VILA fine-tuning
  - Indexed and registered in the data catalog
  
- **Validation Data**: Separate datasets used to evaluate models
  - Fixed validation sets for consistent evaluation
  - Benchmarking datasets for comparative analysis

### 2. Models Directory

The `models` directory stores all model files with versioning:

- **Base Models**: Original NVILA-15B model components
  - Separate directories for each component (LLM, projector, vision)
  - Read-only, never modified directly
  
- **Fine-tuned Models**: Latest production models
  - Symbolic links to specific versions
  - "latest" points to the most recent version
  - "stable" points to the last known-good version
  
- **Versioned Models**: Storage for all model versions
  - Semantic versioning (vMAJOR.MINOR.PATCH)
  - Each version contains full model weights
  - Includes metadata files with training information

### 3. Config Directory

The `config` directory contains all configuration files:

- **Production Config**: Main YAML configuration for the system
- **Environment Specification**: Conda environment definition
- **Model Cards**: Documentation for each model version

### 4. Logs Directory

The `logs` directory stores all system logs:

- **Training Logs**: Outputs from training runs
- **Validation Logs**: Model evaluation results
- **System Logs**: Operational logs from services

### 5. Scripts Directory

The `scripts` directory contains all operational scripts:

- **Environment Scripts**: Setup and activation scripts
- **Data Processing**: Scripts for dataset preparation
- **Training**: Fine-tuning and evaluation scripts
- **Deployment**: Scripts for model deployment

### 6. Services Directory

The `services` directory contains service implementations:

- **Data Ingestion**: Service for processing new data
- **Fine-tuning**: Service for training models
- **Model Serving**: Service for model inference
- **VILA Package**: Local installation of VILA code

## Data Flow

1. New data uploaded to `data/raw/`
2. Data ingestion service validates data and moves to `data/processed/`
3. Fine-tuning service detects new processed data
4. Fine-tuning job initiated with appropriate configuration
5. New model version created in `models/versioned/`
6. Evaluation run against validation datasets
7. If validation passes, "latest" symlink updated
8. After additional testing, "stable" symlink may be updated 