import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatasetStats:
    total_videos: int = 0
    total_metadata: int = 0
    missing_metadata: int = 0
    invalid_metadata: int = 0
    video_formats: Dict[str, int] = None
    metadata_errors: List[str] = None
    
    def __post_init__(self):
        self.video_formats = defaultdict(int)
        self.metadata_errors = []

class DatasetValidator:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.stats = DatasetStats()
        
    def validate_video_file(self, video_path: Path) -> bool:
        """Validate a video file exists and has a supported format."""
        if not video_path.exists():
            # Log an error but don't stop validation for other files
            logger.warning(f"Video file referenced in metadata but not found: {video_path}")
            return False
            
        ext = video_path.suffix.lower()
        # Allow only mp4 for VILA compatibility
        if ext not in ['.mp4']: 
            logger.error(f"Unsupported video format (only .mp4 allowed): {ext} in {video_path}")
            return False
            
        self.stats.video_formats[ext] += 1
        return True
        
    def validate_metadata_file(self, metadata_path: Path) -> bool:
        """Validate the data.json file for a camera dataset."""
        if not metadata_path.exists():
            logger.error(f"Metadata file not found: {metadata_path}")
            self.stats.missing_metadata += 1
            return False

        self.stats.total_metadata += 1 # Count the file itself
        is_valid = True
        try:
            with open(metadata_path, 'r') as f:
                metadata_list = json.load(f)

            if not isinstance(metadata_list, list):
                 error_msg = f"Invalid JSON format in {metadata_path}: Expected a list of entries."
                 logger.error(error_msg)
                 self.stats.metadata_errors.append(error_msg)
                 self.stats.invalid_metadata += 1
                 return False
                
            # VILA expects specific keys, potentially including 'id', 'video', 'conversations'
            # Let's check for a common structure based on potential VILA formats
            # We'll check the first entry as a sample
            required_fields = [
                "id",       # Common identifier
                "video",    # Path to video file
                "conversations" # LLaVA/VILA style interaction data
                # Add other essential fields if known for VILA format
            ]
            
            if not metadata_list:
                 logger.warning(f"Metadata file is empty: {metadata_path}")
                 return True # Empty list might be valid, but log it

            # Check structure of the first entry
            first_entry = metadata_list[0]
            if not isinstance(first_entry, dict):
                 error_msg = f"Invalid entry format in {metadata_path}: Expected dictionaries inside the list."
                 logger.error(error_msg)
                 self.stats.metadata_errors.append(error_msg)
                 self.stats.invalid_metadata += 1
                 return False


            for field in required_fields:
                if field not in first_entry:
                    error_msg = f"Missing required field '{field}' in first entry of {metadata_path}"
                    logger.warning(error_msg) # Warning as it's a sample check
                    self.stats.metadata_errors.append(error_msg)
                    # Don't mark as invalid just for missing fields in first entry, but log it
                    # is_valid = False # Uncomment if structure must be strictly enforced on first entry
                    # break # Stop checking fields for this file if one is missing


            # --- Optional: Add more checks specific to VILA's expected conversation format ---
            # if is_valid and 'conversations' in first_entry:
            #     if not isinstance(first_entry['conversations'], list):
            #         # Log error about conversation format
            #         is_valid = False
            #     elif first_entry['conversations']:
            #          conv_entry = first_entry['conversations'][0]
            #          if not isinstance(conv_entry, dict) or 'from' not in conv_entry or 'value' not in conv_entry:
            #              # Log error about conversation entry format
            #              is_valid = False
            # -----------------------------------------------------------------------------


        except json.JSONDecodeError:
            error_msg = f"Invalid JSON format in {metadata_path}"
            logger.error(error_msg)
            self.stats.metadata_errors.append(error_msg)
            self.stats.invalid_metadata += 1
            is_valid = False
        except Exception as e:
            error_msg = f"Error reading or parsing {metadata_path}: {str(e)}"
            logger.error(error_msg)
            self.stats.metadata_errors.append(error_msg)
            self.stats.invalid_metadata += 1
            is_valid = False
            
        return is_valid

    def validate_dataset(self) -> DatasetStats:
        """Validate the dataset structure by checking each camera subdirectory."""
        logger.info(f"Validating datasets within: {self.dataset_dir}")

        if not self.dataset_dir.exists() or not self.dataset_dir.is_dir():
             logger.error(f"Base dataset directory not found: {self.dataset_dir}")
             return self.stats

        # Iterate through subdirectories (each camera dataset)
        for camera_dir in self.dataset_dir.iterdir():
            if camera_dir.is_dir():
                logger.info(f"--- Validating Camera Directory: {camera_dir.name} ---")
                metadata_path = camera_dir / "data.json"
                
                if not metadata_path.exists():
                     logger.error(f"Missing data.json in {camera_dir}")
                     self.stats.missing_metadata += 1
                     continue # Skip processing videos if metadata is missing

                # Validate the metadata file structure first
                if self.validate_metadata_file(metadata_path):
                    # If metadata is valid, check the video files within this camera dir
                    found_videos_in_dir = False
                    for item_path in camera_dir.iterdir():
                         # Only check files directly within the camera directory
                         if item_path.is_file() and item_path.suffix.lower() == '.mp4':
                              found_videos_in_dir = True
                              if self.validate_video_file(item_path):
                                   self.stats.total_videos += 1
                    
                    if not found_videos_in_dir:
                         logger.warning(f"No .mp4 video files found directly in {camera_dir}")

                else:
                     logger.warning(f"Skipping video validation for {camera_dir.name} due to invalid metadata.")

        return self.stats
        
    def generate_report(self) -> str:
        """Generate a validation report."""
        report = [
            f"\nDataset Validation Report for {self.dataset_dir}",
            "=" * 50,
            f"Total Videos: {self.stats.total_videos}",
            f"Total Metadata Files: {self.stats.total_metadata}",
            f"Missing Metadata: {self.stats.missing_metadata}",
            f"Invalid Metadata: {self.stats.invalid_metadata}",
            "\nVideo Formats:",
        ]
        
        for fmt, count in self.stats.video_formats.items():
            report.append(f"  {fmt}: {count}")
            
        if self.stats.metadata_errors:
            report.append("\nMetadata Errors:")
            for error in self.stats.metadata_errors:
                report.append(f"  - {error}")
                
        return "\n".join(report)

def main():
    # Example usage
    dataset_dir = "./data/vila-dataset"
    validator = DatasetValidator(dataset_dir)
    stats = validator.validate_dataset()
    report = validator.generate_report()
    print(report)

if __name__ == "__main__":
    main() 