#!/bin/bash
# Activate the production conda environment for VILA-15B

# Source conda for bash shell
source $(conda info --base)/etc/profile.d/conda.sh

# Activate the environment
conda activate vila-production

# Print confirmation
echo "Production environment activated!"
echo "Working directory: $(pwd)"
echo "Python path: $(which python)"
echo "Environment: $(conda info --envs | grep '*')"
