#!/bin/bash
# setup_production_structure.sh
# Creates a standardized production directory structure for VILA-15B fine-tuning

# Set up base directories
PROD_ROOT="./production"
DATA_DIR="${PROD_ROOT}/data"
MODELS_DIR="${PROD_ROOT}/models"
CONFIG_DIR="${PROD_ROOT}/config"
LOGS_DIR="${PROD_ROOT}/logs"
SCRIPTS_DIR="${PROD_ROOT}/scripts"
SERVICES_DIR="${PROD_ROOT}/services"

# Create production root directory
mkdir -p ${PROD_ROOT}

# Create data directory structure
mkdir -p ${DATA_DIR}/raw
mkdir -p ${DATA_DIR}/processed
mkdir -p ${DATA_DIR}/validation

# Create models directory structure
mkdir -p ${MODELS_DIR}/base
mkdir -p ${MODELS_DIR}/finetuned
mkdir -p ${MODELS_DIR}/versioned

# Create other necessary directories
mkdir -p ${CONFIG_DIR}
mkdir -p ${LOGS_DIR}
mkdir -p ${SCRIPTS_DIR}
mkdir -p ${SERVICES_DIR}

# Create a README file to document the structure
cat > ${PROD_ROOT}/README.md << EOL
# VILA-15B Production Environment

This directory contains the production environment for VILA-15B fine-tuning system.

## Directory Structure

- \`data/\`: Contains all data used by the system
  - \`data/raw/\`: Upload point for new datasets (CCTV footage, etc.)
  - \`data/processed/\`: Preprocessed data ready for fine-tuning
  - \`data/validation/\`: Validation datasets

- \`models/\`: Contains all model files
  - \`models/base/\`: Base NVILA-15B model files
  - \`models/finetuned/\`: Latest fine-tuned models
  - \`models/versioned/\`: Version-controlled model storage

- \`config/\`: Configuration files
  - Training parameters
  - System configuration
  - Environment settings

- \`logs/\`: System logs
  - Training logs
  - Validation logs
  - System operation logs

- \`scripts/\`: Production scripts
  - Data processing
  - Training
  - Deployment

- \`services/\`: Service definitions
  - Data ingestion service
  - Fine-tuning service
  - Model serving

## Data Flow

1. New data is uploaded to \`data/raw/\`
2. Validation scripts verify data integrity
3. Preprocessing converts raw data to \`data/processed/\`
4. Fine-tuning uses processed data to update models
5. New models are versioned and stored in \`models/versioned/\`
6. Latest models are symlinked to \`models/finetuned/\`

## Setup

This directory structure was created by running \`setup_production_structure.sh\`.
EOL

# Create a dummy config file
cat > ${CONFIG_DIR}/production.yaml << EOL
# Production configuration for VILA-15B fine-tuning

# Data settings
data:
  raw_path: "../data/raw"
  processed_path: "../data/processed"
  validation_path: "../data/validation"
  valid_formats: ["mp4", "jpg", "png"]
  
# Model settings
model:
  base_path: "../models/base"
  finetuned_path: "../models/finetuned"
  versioned_path: "../models/versioned"
  
# Training settings
training:
  batch_size: 4
  learning_rate: 2.0e-5
  epochs: 3
  
# System settings
system:
  log_level: "info"
  log_path: "../logs"
EOL

# Add an initialization script for the conda environment
cat > ${SCRIPTS_DIR}/activate_prod_env.sh << EOL
#!/bin/bash
# Activate the production conda environment for VILA-15B

# Source conda for bash shell
source \$(conda info --base)/etc/profile.d/conda.sh

# Activate the environment
conda activate vila-production

# Print confirmation
echo "Production environment activated!"
echo "Working directory: \$(pwd)"
echo "Python path: \$(which python)"
echo "Environment: \$(conda info --envs | grep '*')"
EOL

chmod +x ${SCRIPTS_DIR}/activate_prod_env.sh

echo "Production directory structure created successfully at ${PROD_ROOT}"
echo "See ${PROD_ROOT}/README.md for details on the structure and usage" 