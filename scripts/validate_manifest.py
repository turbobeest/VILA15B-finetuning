#!/usr/bin/env python3
"""
Validates a VILA-style JSON manifest file to ensure referenced video/image files 
exist AND can be processed. Writes a new JSON file containing only the entries 
that pass validation.
"""

import os
import sys
import json
import argparse
import logging
import yaml # For loading config
from pathlib import Path
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root and VILA path for necessary imports
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
external_vila_path = project_root / "external/VILA"
sys.path.insert(0, str(external_vila_path))

# Try importing necessary VILA/Transformers components
try:
    from llava.mm_utils import opencv_extract_frames, process_image
    from llava.train.args import DataArguments
    from transformers import AutoImageProcessor, CLIPImageProcessor
    from llava.data.dataset import LazySupervisedDataset
except ImportError as e:
    logger.error(f"Failed to import necessary VILA/Transformers components: {e}")
    logger.error("Ensure the VILA submodule and dependencies are correctly installed.")
    sys.exit(1)

# Global placeholder for image processor (initialized later)
_image_processor = None

def initialize_image_processor(config: dict):
    """Initializes the image processor based on config."""
    global _image_processor
    if _image_processor is not None:
        return _image_processor

    logger.info("Initializing Image Processor for validation...")
    # Use settings similar to improved_patch_functions
    # Default to 448x448 based on latest findings, but allow override?
    img_size = config.get("validation_image_size", 448) # Allow override if needed
    logger.info(f"Using image size: {img_size}x{img_size}")
    
    # Try loading from vision tower first
    model_base_path = config.get("model", {}).get("base_path", "./models/NVILA-15B")
    vision_tower_path = Path(model_base_path) / "vision_tower"

    try:
        if vision_tower_path.exists():
             logger.info(f"Loading processor from local vision tower: {vision_tower_path}")
             _image_processor = AutoImageProcessor.from_pretrained(
                 vision_tower_path,
                 trust_remote_code=True,
                 local_files_only=True
             )
             # Override size if needed? Models often handle different sizes.
             # _image_processor.size = {"height": img_size, "width": img_size} 
        else:
             logger.warning(f"Vision tower not found at {vision_tower_path}. Falling back to default CLIP processor.")
             _image_processor = None
    except Exception as e:
        logger.warning(f"Error loading processor from vision tower: {e}. Falling back to default CLIP processor.")
        _image_processor = None

    if _image_processor is None:
        logger.info("Using default CLIP processor.")
        _image_processor = CLIPImageProcessor(
            size={"height": img_size, "width": img_size}, 
            image_mean=[0.48145466, 0.4578275, 0.40821073], 
            image_std=[0.26862954, 0.26130258, 0.27577711],
            do_normalize=True,
            do_center_crop=True, 
            do_resize=True,
        )
    logger.info("Image Processor Initialized.")
    return _image_processor

def validate_manifest(input_json_path: str, data_dir: str, output_json_path: str, config_path: str):
    """
    Reads an input JSON manifest, checks for file existence and processability,
    and writes valid entries to an output JSON file.
    """
    logger.info(f"Validating manifest: {input_json_path}")
    logger.info(f"Using config: {config_path}")
    logger.info(f"Base data directory: {data_dir}")
    logger.info(f"Outputting validated manifest to: {output_json_path}")

    # Load main config to get necessary parameters
    if not Path(config_path).is_file():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error reading config file {config_path}: {e}")
        sys.exit(1)

    # Initialize Image Processor using config
    try:
        image_processor = initialize_image_processor(config)
    except Exception as e:
        logger.error(f"Failed to initialize image processor: {e}")
        sys.exit(1)

    # Create DataArguments using config (without image_processor kwarg)
    try:
        data_args_config = config.get("data", {})
        data_args = DataArguments(
            image_folder=data_args_config.get("image_folder", data_dir),
            image_aspect_ratio=data_args_config.get("image_aspect_ratio", "pad"),
            num_video_frames=data_args_config.get("num_video_frames", 4),
            is_multimodal=True,
        )
        # Add the image processor as an attribute manually
        data_args.image_processor = image_processor 
        logger.info("Manually attached image processor to data_args.")
    except Exception as e:
        logger.error(f"Failed to create DataArguments: {e}")
        sys.exit(1)

    if not Path(input_json_path).is_file():
        logger.error(f"Input JSON file not found: {input_json_path}")
        sys.exit(1)
        
    if not Path(data_dir).is_dir():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    try:
        with open(input_json_path, 'r') as f:
            manifest_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {input_json_path}: {e}")
        sys.exit(1)
    except Exception as e:
         logger.error(f"Error reading {input_json_path}: {e}")
         sys.exit(1)

    if not isinstance(manifest_data, list):
        logger.error(f"Expected input JSON to contain a list, but got {type(manifest_data)}")
        sys.exit(1)

    valid_entries = []
    invalid_count = 0
    total_count = len(manifest_data)

    for i, entry in enumerate(manifest_data):
        if not isinstance(entry, dict):
            logger.warning(f"Skipping non-dictionary entry at index {i}")
            invalid_count += 1
            continue

        if "video" in entry and isinstance(entry["video"], str):
            video_relative_path = entry["video"]
            video_full_path = Path(data_dir) / video_relative_path
            
            if video_full_path.is_file():
                # --- Start Processing Attempt ---
                try:
                    # Attempt to load and process frames (mimic dataset logic)
                    num_frames = data_args.num_video_frames
                    # Use static method _load_video from LazySupervisedDataset if accessible
                    # Or reimplement basic frame extraction here
                    # NOTE: This assumes _load_video doesn't require dataset instance state
                    # If it does, we might need a simpler check
                    pil_imgs, frames_loaded = LazySupervisedDataset._load_video(
                         str(video_full_path), num_frames, 0.0, data_args
                    )
                    if frames_loaded == 0:
                        raise ValueError("No frames loaded by _load_video")
                    
                    # Attempt to process the loaded frames
                    # process_image expects processor on data_args
                    processed_frames = [process_image(img, data_args, None) for img in pil_imgs]
                    valid_frames = [frame for frame in processed_frames if isinstance(frame, torch.Tensor)]
                    if len(valid_frames) != len(pil_imgs):
                         raise ValueError(f"Failed to process all loaded frames ({len(valid_frames)}/{len(pil_imgs)} valid)")
                    
                    # Attempt to stack (simulates collator step roughly)
                    _ = torch.stack(valid_frames) 
                    
                    # If all checks pass, add to valid entries
                    valid_entries.append(entry)
                except FileNotFoundError: # Should be caught earlier, but belts and suspenders
                     logger.warning(f"Video file not found during processing: {video_full_path}")
                     invalid_count += 1
                except Exception as e:
                    logger.warning(f"Failed to process video for entry {i} ({video_full_path}): {e}")
                    invalid_count += 1
                # --- End Processing Attempt ---
            else:
                logger.warning(f"Video file not found for entry {i}: {video_full_path}")
                invalid_count += 1
        elif "image" in entry:
             # === Start Image Processing Check ===
             try:
                 image_relative_path = entry["image"]
                 image_full_path = Path(data_dir) / image_relative_path
                 if not image_full_path.is_file():
                     raise FileNotFoundError(f"Image file not found: {image_full_path}")
                 
                 # Attempt to process the image
                 # process_image expects processor on data_args
                 img_tensor = process_image(str(image_full_path), data_args, None)
                 if not isinstance(img_tensor, torch.Tensor):
                     raise ValueError(f"process_image did not return a tensor for {image_full_path}")
                 
                 # If processing succeeds, add the entry (Correctly indented)
                 valid_entries.append(entry) 
             except FileNotFoundError as e: # Correctly indented except
                 logger.warning(f"Image file not found for entry {i}: {e}")
                 invalid_count += 1
             except Exception as e: # Correctly indented except
                 logger.warning(f"Failed to process image for entry {i}: {e}")
                 invalid_count += 1
             # === End Image Processing Check ===
        else:
            # Entry has neither 'video' nor 'image', consider it invalid for multimodal training
            logger.warning(f"Skipping entry {i} with no 'video' or 'image' key.")
            invalid_count += 1
            
    logger.info(f"Validation complete. Total entries: {total_count}")
    logger.info(f"Valid entries found: {len(valid_entries)}")
    logger.info(f"Invalid/skipped entries: {invalid_count}")

    # Ensure output directory exists
    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)

    # Write the valid entries to the output file
    try:
        with open(output_json_path, 'w') as f:
            json.dump(valid_entries, f, indent=2)
        logger.info(f"Successfully wrote validated manifest to {output_json_path}")
    except Exception as e:
        logger.error(f"Error writing validated manifest to {output_json_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate VILA JSON manifest for video file existence.")
    parser.add_argument("--input-json", required=True, help="Path to the input JSON manifest file.")
    parser.add_argument("--data-dir", required=True, help="Path to the base directory containing video files.")
    parser.add_argument("--output-json", required=True, help="Path to write the filtered output JSON manifest.")
    parser.add_argument("--config-path", required=True, help="Path to the config file.")
    
    args = parser.parse_args()
    
    validate_manifest(args.input_json, args.data_dir, args.output_json, args.config_path) 