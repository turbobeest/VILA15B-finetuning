#!/usr/bin/env python3
"""
Prepare Validation Dataset for NW-test-camera-1

This script prepares the validation dataset for evaluating the fine-tuned
NVILA-15B model. It loads the dataset and applies appropriate preprocessing,
ensuring that the data is properly structured for evaluation.
"""

import os
import sys
import json
import yaml
import torch
import logging
import argparse
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset, random_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add the VILA library to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
external_vila_path = os.path.join(os.path.dirname(script_dir), "external/VILA")
sys.path.append(external_vila_path)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def apply_improved_patches():
    """Apply improved patches to VILA codebase."""
    # Import improved_patch_functions module
    sys.path.append(script_dir)
    from improved_patch_functions import apply_improved_patches
    
    if apply_improved_patches():
        logger.info("Successfully applied improved patches to VILA codebase")
        return True
    else:
        logger.error("Failed to apply patches")
        return False

def prepare_validation_dataset(config, output_dir=None):
    """
    Prepare the validation dataset for model evaluation.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save processed data (optional)
        
    Returns:
        validation_dataset: The prepared validation dataset
        validation_dataloader: DataLoader for the validation dataset
    """
    from transformers import AutoTokenizer, AutoConfig
    from llava.train.args import DataArguments
    from llava.train.args import TrainingArguments as LlavaTrainingArguments
    from llava.data.dataset import LazySupervisedDataset
    
    logger.info("Preparing validation dataset...")
    
    # Get the data path from config
    data_path = config["data"]["validation_data"][0]["path"]
    image_folder = config["data"]["image_folder"]
    
    # Verify that the data path exists
    if not os.path.exists(data_path):
        logger.error(f"Data path does not exist: {data_path}")
        return None, None
    
    # Load tokenizer from base model
    model_path = config["model"]["base_path"]
    tokenizer_path = os.path.join(model_path, "llm")
    
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )
        logger.info("Successfully loaded tokenizer")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return None, None
    
    # Create data arguments
    data_args = DataArguments(
        data_path=data_path,
        image_folder=image_folder,
        image_aspect_ratio=config["data"]["image_aspect_ratio"],
        num_video_frames=config["data"]["num_video_frames"],
        is_multimodal=True,
    )
    
    # Add model_path as an attribute after initialization
    data_args.model_path = model_path
    
    # Create training args for dataset loading
    training_args = LlavaTrainingArguments(
        output_dir=output_dir or "./output/validation",
        model_max_length=config["training"]["model_max_length"],
    )
    
    # Load the dataset
    try:
        logger.info(f"Loading dataset from {data_path}")
        full_dataset = LazySupervisedDataset(
            data_path=data_path,
            image_folder=image_folder,
            tokenizer=tokenizer,
            data_args=data_args,
            training_args=training_args,
        )
        
        logger.info(f"Successfully loaded dataset with {len(full_dataset)} samples")
        
        # Apply train/validation split using the same strategy from fine-tuning
        train_ratio = config["data"]["split_strategy"]["train_ratio"]
        random_seed = config["data"]["split_strategy"]["random_seed"]
        
        # Calculate split sizes
        dataset_size = len(full_dataset)
        train_size = int(dataset_size * train_ratio)
        val_size = dataset_size - train_size
        
        # Create the splits
        torch.manual_seed(random_seed)
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size]
        )
        
        logger.info(f"Created validation split with {len(val_dataset)} samples")
        
        # Create dataloader for validation set
        val_batch_size = config["training"].get("batch_size", 1)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        
        logger.info(f"Created validation dataloader with batch size {val_batch_size}")
        
        # Test loading a sample to verify dataset is valid
        try:
            logger.info("Testing sample loading...")
            sample_idx = 0
            sample = val_dataset[sample_idx]
            
            # Log sample keys and shapes
            logger.info(f"Sample keys: {list(sample.keys())}")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    logger.info(f"  {key}: type={type(value)}")
            
            logger.info("Sample loaded successfully - validation dataset is ready")
        except Exception as e:
            logger.error(f"Error loading sample from validation dataset: {e}")
            import traceback
            traceback.print_exc()
        
        # Return the validation dataset and dataloader
        return val_dataset, val_dataloader
    
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def save_dataset_stats(val_dataset, output_dir):
    """Save validation dataset statistics to JSON file."""
    if val_dataset is None:
        return
    
    stats = {
        "dataset_size": len(val_dataset),
        "sample_keys": [],
    }
    
    # Get sample keys from first item if available
    if len(val_dataset) > 0:
        try:
            sample = val_dataset[0]
            stats["sample_keys"] = list(sample.keys())
            
            # Get tensor shapes for each key
            stats["tensor_shapes"] = {}
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    stats["tensor_shapes"][key] = list(value.shape)
        except Exception as e:
            logger.error(f"Error getting sample stats: {e}")
    
    # Save stats to JSON file
    os.makedirs(output_dir, exist_ok=True)
    stats_file = os.path.join(output_dir, "validation_dataset_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved validation dataset statistics to {stats_file}")

def main():
    parser = argparse.ArgumentParser(description="Prepare validation dataset for NVILA-15B evaluation")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/fine_tuning_config.yaml",
        help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output/validation",
        help="Directory to save processed data and statistics"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    if config is None:
        return 1
    
    # Apply improved patches to VILA codebase
    if not apply_improved_patches():
        return 1
    
    # Prepare validation dataset
    val_dataset, val_dataloader = prepare_validation_dataset(config, args.output_dir)
    
    # Save dataset statistics
    if val_dataset is not None:
        save_dataset_stats(val_dataset, args.output_dir)
        logger.info(f"Successfully prepared validation dataset with {len(val_dataset)} samples")
        return 0
    else:
        logger.error("Failed to prepare validation dataset")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 