#!/usr/bin/env python
# validate_dataset.py
# Validates datasets uploaded to the raw data directory

import os
import sys
import yaml
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs/system/validation.log'))
    ]
)
logger = logging.getLogger("dataset_validator")

def load_config():
    """Load the production configuration"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config/production.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def validate_file_format(file_path, valid_formats):
    """Check if the file has a valid format"""
    ext = file_path.suffix.lower().lstrip('.')
    if ext not in valid_formats:
        return False, f"Invalid file format: {ext}. Expected one of: {', '.join(valid_formats)}"
    return True, "File format is valid"

def validate_video_file(file_path):
    """Validate a video file for content and integrity"""
    # This would normally include more thorough validation, such as:
    # - Checking if the video is corrupted
    # - Verifying frame rate, duration, resolution
    # - Checking if the content matches expected patterns
    
    # For this example, we'll just check if the file exists and has non-zero size
    if not file_path.exists():
        return False, f"File does not exist: {file_path}"
    
    if file_path.stat().st_size == 0:
        return False, f"File is empty: {file_path}"
    
    return True, "Video file appears valid"

def validate_image_file(file_path):
    """Validate an image file for content and integrity"""
    # Similar to video validation, this would include more thorough checks
    # For this example, we'll just check existence and size
    
    if not file_path.exists():
        return False, f"File does not exist: {file_path}"
    
    if file_path.stat().st_size == 0:
        return False, f"File is empty: {file_path}"
    
    return True, "Image file appears valid"

def validate_dataset_structure(dataset_path):
    """Validate the structure of the dataset"""
    # Check that the dataset directory exists
    if not dataset_path.exists() or not dataset_path.is_dir():
        return False, f"Dataset path does not exist or is not a directory: {dataset_path}"
    
    # Check if it has any files
    files = list(dataset_path.glob('*'))
    if not files:
        return False, f"Dataset is empty: {dataset_path}"
    
    # For more complex validation, we would check:
    # - Required directory structure
    # - Presence of metadata files
    # - Consistency of naming conventions
    
    return True, f"Dataset structure appears valid with {len(files)} files"

def process_dataset(dataset_path, config):
    """Process and validate a dataset directory"""
    logger.info(f"Processing dataset: {dataset_path}")
    
    # Step 1: Validate dataset structure
    valid, message = validate_dataset_structure(dataset_path)
    if not valid:
        logger.error(message)
        return False
    
    logger.info(message)
    
    # Step 2: Validate individual files
    valid_formats = config['data']['valid_formats']
    invalid_files = []
    valid_files = []
    
    for file_path in dataset_path.glob('**/*'):
        if file_path.is_file():
            # Check file format
            format_valid, format_message = validate_file_format(file_path, valid_formats)
            if not format_valid:
                logger.warning(format_message)
                invalid_files.append({"path": str(file_path), "error": format_message})
                continue
            
            # Validate based on file type
            ext = file_path.suffix.lower().lstrip('.')
            if ext in ['mp4']:
                valid, message = validate_video_file(file_path)
            elif ext in ['jpg', 'png']:
                valid, message = validate_image_file(file_path)
            else:
                valid, message = False, f"Unsupported file type for validation: {ext}"
            
            if not valid:
                logger.warning(f"Invalid file {file_path}: {message}")
                invalid_files.append({"path": str(file_path), "error": message})
            else:
                logger.debug(f"Valid file: {file_path}")
                valid_files.append(str(file_path))
    
    # Step 3: Generate validation report
    validation_result = {
        "dataset": str(dataset_path),
        "timestamp": datetime.now().isoformat(),
        "valid_files_count": len(valid_files),
        "invalid_files_count": len(invalid_files),
        "invalid_files": invalid_files
    }
    
    # Determine overall validation result
    if invalid_files:
        validation_result["status"] = "FAILED"
        validation_result["message"] = f"Found {len(invalid_files)} invalid files"
        logger.error(f"Dataset validation failed: {validation_result['message']}")
    else:
        validation_result["status"] = "PASSED"
        validation_result["message"] = f"All {len(valid_files)} files are valid"
        logger.info(f"Dataset validation passed: {validation_result['message']}")
    
    # Step 4: Save validation report
    processed_path = Path(config['data']['processed_path'])
    dataset_name = dataset_path.name
    validation_dir = processed_path / dataset_name
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = validation_dir / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(validation_result, f, indent=2)
    
    logger.info(f"Validation report saved to: {report_path}")
    
    return validation_result["status"] == "PASSED"

def move_validated_dataset(dataset_path, config):
    """Move validated dataset to processed directory"""
    if not process_dataset(dataset_path, config):
        logger.error(f"Dataset failed validation, not moving to processed directory: {dataset_path}")
        return False
    
    # In a real implementation, we would:
    # 1. Copy files to processed directory
    # 2. Apply any necessary preprocessing
    # 3. Update the data registry
    
    logger.info(f"Dataset passed validation, ready for processing: {dataset_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Validate datasets for VILA-15B fine-tuning')
    parser.add_argument('dataset_path', help='Path to the dataset directory to validate')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Process the dataset
    dataset_path = Path(args.dataset_path)
    result = move_validated_dataset(dataset_path, config)
    
    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main()) 