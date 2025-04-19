# Project CodeMap

## Directory Structure and Purposes

### Root Directories
- `.taskmaster/` - Project management and documentation
  - `tasks/` - Task definitions, status, and results
  - `docs/` - Project documentation and guides
  - `scripts/` - Project management scripts
- `data/` - Dataset storage and processing
  - `vila-dataset/` - VILA-formatted datasets
    - `{model-name}/` - Model-specific datasets
      - `videos/` - Video files
      - `data.json` - Dataset metadata and annotations
- `scripts/` - Project utility scripts
  - `parse_dataset.py` - Dataset conversion and processing
- `configs/` - Configuration files
- `models/` - Model checkpoints and weights
- `src/` - Source code for the project
- `notebooks/` - Jupyter notebooks for analysis and experimentation
- `external/` - External dependencies and tools

### Key Files
- `environment.yml` - Conda environment specification
- `taskmaster.config.json` - Taskmaster configuration
- `.gitignore` - Git ignore rules

## File Purposes

### Configuration Files
- `environment.yml` - Defines the Python environment with all required dependencies
- `taskmaster.config.json` - Configures the Taskmaster project management system

### Dataset Files
- `data.json` - Contains video metadata and annotations in VILA format
- Video files (`.mp4`) - Raw video data for training

### Script Files
- `parse_dataset.py` - Converts raw dataset format to VILA training format

### Documentation
- Task status files - Track progress of each task
- Critical lessons - Document important findings and lessons learned
- CodeMap - Project structure documentation 