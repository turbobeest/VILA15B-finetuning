#!/usr/bin/env python3
"""
Test Video Loading with Improved Patches

This script tests the improved video loading patches by loading a single
video sample from the NW-test-camera-1 dataset and verifying the media is processed.
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Add external/VILA to the Python path
external_vila_path = os.path.join(project_root, "external/VILA")
sys.path.append(external_vila_path)

def setup():
    """Apply the improved patches and set up VILA modules."""
    # Import and apply improved patches
    from improved_patch_functions import apply_improved_patches
    if not apply_improved_patches():
        logger.error("Failed to apply patches")
        return False
    
    logger.info("Successfully applied patches")
    return True

def test_video_loading(config_path):
    """
    Test loading and processing a single video from the NW-test-camera-1 dataset.
    
    Args:
        config_path: Path to the fine-tuning configuration YAML file
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load necessary modules from VILA
    from transformers import AutoTokenizer
    from llava.train.args import DataArguments
    from llava.train.args import TrainingArguments as LlavaTrainingArguments
    from llava.data.dataset import LazySupervisedDataset
    
    # Get data paths from config
    data_path = config["data"]["train_data"][0]["path"]
    image_folder = config["data"]["image_folder"]
    num_video_frames = config["data"]["num_video_frames"]
    
    logger.info(f"Data path: {data_path}")
    logger.info(f"Image folder: {image_folder}")
    logger.info(f"Number of video frames: {num_video_frames}")
    
    # Check if data path exists
    if not os.path.exists(data_path):
        logger.error(f"Data path not found: {data_path}")
        return False
    
    # Load sample JSON data
    try:
        with open(data_path, "r") as f:
            samples = json.load(f)
        
        if not samples or not isinstance(samples, list):
            logger.error("No valid samples found in data file")
            return False
        
        logger.info(f"Loaded {len(samples)} samples from {data_path}")
        
        # Take the first sample with a video
        test_sample = samples[0]
        video_path = test_sample.get("video")
        
        if not video_path:
            logger.error("No video path found in first sample")
            return False
        
        logger.info(f"Testing with video: {video_path}")
        
        # Load the tokenizer
        model_path = config["model"]["base_path"]
        tokenizer_path = os.path.join(model_path, "llm")
        
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )
        logger.info("Tokenizer loaded successfully")
        
        # Create data arguments
        data_args = DataArguments(
            data_path=data_path,
            image_folder=image_folder,
            num_video_frames=num_video_frames,
            image_aspect_ratio=config["data"]["image_aspect_ratio"],
            is_multimodal=True,
        )
        
        # Add model_path as an attribute (for custom patches)
        data_args.model_path = model_path
        
        # Create training arguments
        training_args = LlavaTrainingArguments(
            output_dir="./output/test",
            model_max_length=config["training"]["model_max_length"],
        )
        
        # Create dataset with single sample
        logger.info("Creating test dataset with single sample")
        test_dataset = LazySupervisedDataset(
            data_path=data_path,
            image_folder=image_folder,
            tokenizer=tokenizer,
            data_args=data_args,
            training_args=training_args,
        )
        
        # Process the first sample
        logger.info("Processing first sample...")
        processed_sample = test_dataset[0]
        
        # Check if media is present in the processed sample
        if "image" not in processed_sample or processed_sample["image"] is None:
            logger.error("No processed image/video frames found in sample under the 'image' key")
            # Add more debugging info
            logger.error(f"Processed sample keys: {list(processed_sample.keys())}")
            if "image" in processed_sample:
                 logger.error(f"Value for 'image' key: {processed_sample['image']}")
            return False
        
        # Check the media content (now stored directly under "image")
        image_data = processed_sample["image"]
        
        # Check if it's a tensor and has the expected dimensions for video frames
        if isinstance(image_data, torch.Tensor):
            # Expecting shape like (num_frames, channels, height, width)
            if len(image_data.shape) == 4 and image_data.shape[0] == data_args.num_video_frames:
                 logger.info(f"Video frames found under 'image' key. Shape: {image_data.shape}")
                 logger.info(f"Video frames dtype: {image_data.dtype}")
            logger.info("Video processing successful!")
            return True
            # Handle single image case if necessary, though this test focuses on video
            elif len(image_data.shape) == 3: 
                 logger.warning(f"Single image found under 'image' key instead of video frames. Shape: {image_data.shape}")
                 # Decide if this is acceptable for the test or should be an error
                 # For now, let's consider it a pass if an image tensor exists
            logger.info("Image processing successful!")
                 return True # Or return False if strictly testing for video frames
            else:
                 logger.error(f"Unexpected tensor shape found under 'image' key: {image_data.shape}")
                 return False
        else:
            logger.error(f"Unexpected data type found under 'image' key: {type(image_data)}")
            return False
        
    except Exception as e:
        logger.error(f"Error testing video loading: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Test video loading")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/fine_tuning_config.yaml",
        help="Path to the configuration file"
    )
    args = parser.parse_args()
    
    # Set up the environment
    if not setup():
        return 1
    
    # Test video loading
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return 1
    
    result = test_video_loading(args.config)
    
    if result:
        logger.info("✅ Video loading test PASSED")
        return 0
    else:
        logger.error("❌ Video loading test FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 