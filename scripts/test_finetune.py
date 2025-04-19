#!/usr/bin/env python3
"""
Test script for verifying the fine-tuning setup without actually training.
This script loads the model, tokenizer, and datasets, and validates that
all components can be initialized correctly before committing to full training.
"""

import os
import sys
import yaml
import json
import logging
import argparse
import torch
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the VILA library to the path
external_vila_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external/VILA")
sys.path.append(external_vila_path)

# Conditionally import VILA modules
try:
    from llava.model import LlavaLlamaConfig, LlavaLlamaModel
    from llava.data.dataset import VideoQADataset
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure VILA is properly installed and accessible.")
    sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test NVILA-15B fine-tuning setup")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fine_tuning_config.yaml",
        help="Path to the YAML configuration file"
    )
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        sys.exit(1)

def test_model_loading(config: dict) -> bool:
    """Test loading the VILA model and tokenizer."""
    try:
        logger.info("Testing model loading...")
        model_path = config["model"]["base_path"]
        
        # Check if model path exists
        if not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}")
            return False
        
        # Load tokenizer
        try:
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False,
            )
            logger.info(f"Tokenizer loaded successfully. Vocab size: {tokenizer.vocab_size}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            return False
        
        # Load model configuration
        try:
            logger.info("Loading model configuration...")
            model_config = LlavaLlamaConfig.from_pretrained(model_path)
            model_config.tune_mm_projector = config["training"]["tune_mm_projector"]
            logger.info(f"Model configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model configuration: {e}")
            return False
        
        # Initialize model
        try:
            logger.info("Loading model...")
            model = LlavaLlamaModel.from_pretrained(
                model_path,
                config=model_config,
                torch_dtype=eval(config["training"]["model_dtype"]),
            )
            logger.info(f"Model loaded successfully. Architecture: {model_config.architectures}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
        
        # Configure LoRA if enabled
        if config["training"]["lora"]["enable"]:
            try:
                logger.info("Setting up LoRA...")
                lora_config = LoraConfig(
                    r=config["training"]["lora"]["r"],
                    lora_alpha=config["training"]["lora"]["alpha"],
                    lora_dropout=config["training"]["lora"]["dropout"],
                    bias=config["training"]["lora"]["bias"],
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                )
                model = get_peft_model(model, lora_config)
                logger.info(f"LoRA configured successfully with rank {config['training']['lora']['r']}")
            except Exception as e:
                logger.error(f"Failed to configure LoRA: {e}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"Unexpected error during model loading: {e}")
        return False

def test_dataset_loading(config: dict) -> bool:
    """Test loading and preprocessing the datasets."""
    try:
        logger.info("Testing dataset loading...")
        
        # Load the training dataset
        train_data_path = config["data"]["train_data"][0]["path"]
        if not os.path.exists(train_data_path):
            logger.error(f"Data path does not exist: {train_data_path}")
            return False
        
        try:
            with open(train_data_path, "r") as f:
                data = json.load(f)
            logger.info(f"Dataset loaded from {train_data_path}. Found {len(data)} samples.")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return False
        
        # Test a small subset of the data (5 samples) to avoid memory issues
        data_subset = data[:5]
        logger.info(f"Testing dataset creation with {len(data_subset)} samples")
        
        # Load tokenizer for dataset testing
        model_path = config["model"]["base_path"]
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        
        # Create a test dataset
        try:
            logger.info("Creating test dataset...")
            image_folder = config["data"]["image_folder"]
            num_video_frames = config["data"]["num_video_frames"]
            
            test_dataset = VideoQADataset(
                data=data_subset,
                tokenizer=tokenizer,
                image_folder=image_folder,
                model_max_length=config["training"]["model_max_length"],
                num_video_frames=num_video_frames,
                image_aspect_ratio=config["data"]["image_aspect_ratio"],
            )
            logger.info(f"Test dataset created successfully.")
            
            # Check if we can access an item
            if len(test_dataset) > 0:
                logger.info("Retrieving first item from dataset...")
                try:
                    item = test_dataset[0]
                    logger.info("Successfully retrieved an item from the dataset.")
                    
                    # Log item keys to verify structure
                    logger.info(f"Item keys: {list(item.keys())}")
                    
                    # Check if 'pixel_values' is present (indicates video frames were loaded)
                    if 'pixel_values' in item:
                        logger.info(f"Video frames loaded successfully. Shape: {item['pixel_values'].shape}")
                    else:
                        logger.warning("No 'pixel_values' found in the dataset item. Check video loading.")
                except Exception as e:
                    logger.error(f"Failed to retrieve item from dataset: {e}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Failed to create test dataset: {e}")
            return False
        
    except Exception as e:
        logger.error(f"Unexpected error during dataset testing: {e}")
        return False

def main():
    """Main function to run the tests."""
    args = parse_args()
    config = load_config(args.config)
    
    # Run the tests
    logger.info("Starting tests...")
    
    # Test model loading
    model_ok = test_model_loading(config)
    logger.info(f"Model loading test: {'PASSED' if model_ok else 'FAILED'}")
    
    # Test dataset loading
    dataset_ok = test_dataset_loading(config)
    logger.info(f"Dataset loading test: {'PASSED' if dataset_ok else 'FAILED'}")
    
    # Overall result
    if model_ok and dataset_ok:
        logger.info("All tests PASSED. Fine-tuning setup is ready.")
        return 0
    else:
        logger.error("Some tests FAILED. See logs above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 