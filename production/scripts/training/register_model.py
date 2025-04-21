#!/usr/bin/env python
# register_model.py
# Registers a newly fine-tuned model in the versioned storage

import os
import sys
import yaml
import json
import shutil
import hashlib
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs/system/model_registration.log'))
    ]
)
logger = logging.getLogger("model_registrar")

def load_config():
    """Load the production configuration"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config/production.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def calculate_checksum(file_path):
    """Calculate SHA-256 checksum for a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash in chunks for large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def calculate_directory_size(directory_path):
    """Calculate the total size of a directory in GB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    
    # Convert bytes to GB
    return total_size / (1024 * 1024 * 1024)

def parse_semantic_version(version_str):
    """Parse a semantic version string into its components"""
    if version_str.startswith('v'):
        version_str = version_str[1:]
    
    try:
        major, minor, patch = map(int, version_str.split('.'))
        return major, minor, patch
    except ValueError:
        logger.error(f"Invalid semantic version: {version_str}")
        return None

def increment_version(version_str, increment_type='patch'):
    """Increment a semantic version based on increment_type"""
    parsed = parse_semantic_version(version_str)
    if not parsed:
        return None
    
    major, minor, patch = parsed
    
    if increment_type == 'major':
        return f"{major + 1}.0.0"
    elif increment_type == 'minor':
        return f"{major}.{minor + 1}.0"
    elif increment_type == 'patch':
        return f"{major}.{minor}.{patch + 1}"
    else:
        logger.error(f"Unknown increment type: {increment_type}")
        return None

def get_latest_model_version(versions_dir):
    """Find the latest model version in the versioned directory"""
    versions = []
    
    for item in versions_dir.glob('vila-v*'):
        if item.is_dir():
            version_str = item.name.replace('vila-v', '')
            parsed = parse_semantic_version(version_str)
            if parsed:
                versions.append((parsed, version_str))
    
    if not versions:
        return "0.0.0"  # No versions found
    
    # Sort by version components (major, minor, patch)
    versions.sort(reverse=True)
    return versions[0][1]

def generate_model_metadata(source_model_path, target_model_dir, model_version, training_info, performance_metrics):
    """Generate metadata file for the model"""
    # Load the metadata template
    template_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                'models/versioned/model_metadata_template.json')
    
    with open(template_path, 'r') as f:
        metadata = json.load(f)
    
    # Update metadata with provided information
    metadata["model_info"]["name"] = f"vila-v{model_version}"
    metadata["model_info"]["version"] = model_version
    metadata["model_info"]["creation_date"] = datetime.now().isoformat()
    
    # Update training info
    if training_info:
        metadata["training_info"].update(training_info)
    
    # Update performance metrics
    if performance_metrics:
        metadata["performance_metrics"].update(performance_metrics)
    
    # Update storage information
    metadata["storage"]["model_location"] = str(target_model_dir)
    metadata["storage"]["total_size_gb"] = round(calculate_directory_size(target_model_dir), 2)
    
    # Calculate checksum for a representative file
    # In a real implementation, we'd calculate checksums for all important model files
    main_model_file = list(target_model_dir.glob('*.safetensors'))
    if main_model_file:
        metadata["storage"]["checksum"] = f"sha256:{calculate_checksum(main_model_file[0])}"
    
    # Update lineage information
    # Try to find parent model info
    parent_model_metadata = list(source_model_path.glob('metadata.json'))
    if parent_model_metadata:
        with open(parent_model_metadata[0], 'r') as f:
            parent_data = json.load(f)
            metadata["lineage"]["parent_model"] = parent_data["model_info"]["name"]
            metadata["lineage"]["training_iterations"] = parent_data["lineage"]["training_iterations"] + 1
    
    return metadata

def register_model(source_model_path, version_bump='patch', training_info=None, performance_metrics=None):
    """Register a model in the versioned storage system"""
    logger.info(f"Registering model from {source_model_path}")
    
    # Load configuration
    config = load_config()
    
    # Determine the target version
    versions_dir = Path(config['model']['versioned_path'])
    latest_version = get_latest_model_version(versions_dir)
    logger.info(f"Latest model version: vila-v{latest_version}")
    
    # Increment version based on bump type
    new_version = increment_version(latest_version, version_bump)
    if not new_version:
        logger.error("Failed to generate new version number")
        return False
    
    logger.info(f"Registering new model version: vila-v{new_version}")
    
    # Create target directory
    target_model_dir = versions_dir / f"vila-v{new_version}"
    if target_model_dir.exists():
        logger.error(f"Target directory already exists: {target_model_dir}")
        return False
    
    target_model_dir.mkdir(parents=True)
    
    # Copy model files
    try:
        # In a real implementation, we might use more advanced file handling
        # or specific model saving mechanisms
        for item in source_model_path.glob('*'):
            if item.is_file():
                shutil.copy2(item, target_model_dir)
        
        logger.info(f"Copied model files to {target_model_dir}")
    except Exception as e:
        logger.error(f"Error copying model files: {e}")
        return False
    
    # Generate metadata
    metadata = generate_model_metadata(source_model_path, target_model_dir, new_version, training_info, performance_metrics)
    
    # Save metadata
    metadata_path = target_model_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Generated metadata file: {metadata_path}")
    
    # Update symlinks
    try:
        finetuned_dir = Path(config['model']['finetuned_path'])
        latest_link = finetuned_dir / "latest"
        
        # Remove existing symlink if it exists
        if latest_link.is_symlink():
            latest_link.unlink()
        
        # Create relative symlink
        os.symlink(
            os.path.relpath(target_model_dir, finetuned_dir),
            latest_link
        )
        
        logger.info(f"Updated 'latest' symlink to point to vila-v{new_version}")
    except Exception as e:
        logger.error(f"Error updating symlinks: {e}")
        # Continue anyway, this is not critical
    
    # Create model card
    try:
        model_card_dir = Path(config['config_dir']) / "model-cards"
        model_card_path = model_card_dir / f"vila-v{new_version}.md"
        
        with open(model_card_path, 'w') as f:
            f.write(f"# Model Card: vila-v{new_version}\n\n")
            f.write(f"**Version:** {new_version}\n")
            f.write(f"**Created:** {metadata['model_info']['creation_date']}\n")
            f.write(f"**Base Model:** {metadata['model_info']['base_model']}\n\n")
            
            f.write("## Description\n\n")
            f.write(f"{metadata['model_info']['description']}\n\n")
            
            f.write("## Training Information\n\n")
            f.write(f"- **Dataset:** {metadata['training_info']['dataset']['name']}\n")
            f.write(f"- **Samples:** {metadata['training_info']['dataset']['sample_count']}\n")
            f.write(f"- **Learning Rate:** {metadata['training_info']['hyperparameters']['learning_rate']}\n")
            f.write(f"- **Batch Size:** {metadata['training_info']['hyperparameters']['batch_size']}\n")
            f.write(f"- **Epochs:** {metadata['training_info']['hyperparameters']['epochs']}\n\n")
            
            f.write("## Performance\n\n")
            f.write(f"- **Validation Loss:** {metadata['performance_metrics']['validation']['loss']}\n")
            f.write(f"- **Improvement:** {metadata['performance_metrics']['comparison']['improvement_percentage']}%\n\n")
            
            f.write("## Recommended Use Cases\n\n")
            for use_case in metadata['usage']['recommended_use_cases']:
                f.write(f"- {use_case}\n")
            
            f.write("\n## Limitations\n\n")
            for limitation in metadata['usage']['limitations']:
                f.write(f"- {limitation}\n")
            
        logger.info(f"Created model card: {model_card_path}")
    except Exception as e:
        logger.error(f"Error creating model card: {e}")
        # Continue anyway, this is not critical
    
    logger.info(f"Successfully registered model vila-v{new_version}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Register a fine-tuned model in the versioned storage')
    parser.add_argument('model_path', help='Path to the model directory to register')
    parser.add_argument('--version-bump', choices=['patch', 'minor', 'major'], default='patch',
                        help='Type of version increment to apply (default: patch)')
    args = parser.parse_args()
    
    # Register the model
    model_path = Path(args.model_path)
    success = register_model(model_path, args.version_bump)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 