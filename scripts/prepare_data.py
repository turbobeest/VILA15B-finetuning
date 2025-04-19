#!/usr/bin/env python3
"""
NVILA-15B Data Preparation Script

This script handles the preparation of video and JSON data for VILA fine-tuning
in an airgapped environment. It processes video snippets and their corresponding
JSON metadata files according to VILA's requirements.
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreparator:
    def __init__(
        self,
        input_dir: str = "data/raw",
        output_dir: str = "data/processed",
        video_extensions: List[str] = [".mp4", ".avi", ".mov"],
        frame_rate: int = 30,
        target_resolution: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize the data preparator.
        
        Args:
            input_dir: Directory containing raw video and JSON files
            output_dir: Directory for processed data
            video_extensions: Supported video file extensions
            frame_rate: Target frame rate for video processing
            target_resolution: Target resolution for video frames
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.video_extensions = video_extensions
        self.frame_rate = frame_rate
        self.target_resolution = target_resolution
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "videos").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
    def validate_json(self, json_path: Path) -> bool:
        """
        Validate the structure of a JSON metadata file.
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            bool: True if JSON is valid
        """
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
                
            required_fields = [
                "video_id",
                "description",
                "timestamp",
                "scene_type",
                "objects",
                "actions"
            ]
            
            for field in required_fields:
                if field not in metadata:
                    logger.error(f"Missing required field '{field}' in {json_path}")
                    return False
                    
            return True
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in {json_path}")
            return False
        except Exception as e:
            logger.error(f"Error validating JSON {json_path}: {e}")
            return False

    def process_video(
        self,
        video_path: Path,
        output_path: Path
    ) -> bool:
        """
        Process a video file to ensure it meets VILA's requirements.
        
        Args:
            video_path: Path to input video
            output_path: Path to save processed video
            
        Returns:
            bool: True if processing successful
        """
        try:
            # Open video file
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Could not open video {video_path}")
                return False
                
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self.frame_rate,
                self.target_resolution
            )
            
            # Process frames
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Resize frame
                frame = cv2.resize(frame, self.target_resolution)
                
                # Write processed frame
                out.write(frame)
                
            # Clean up
            cap.release()
            out.release()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            return False

    def prepare_data(self) -> Dict:
        """
        Prepare all data files in the input directory.
        
        Returns:
            Dict: Statistics about the preparation process
        """
        stats = {
            "total_files": 0,
            "processed_videos": 0,
            "processed_metadata": 0,
            "errors": 0
        }
        
        try:
            # Process all video files
            for video_path in self.input_dir.glob("**/*"):
                if video_path.suffix.lower() not in self.video_extensions:
                    continue
                    
                stats["total_files"] += 1
                logger.info(f"Processing video: {video_path}")
                
                # Create output path
                rel_path = video_path.relative_to(self.input_dir)
                output_path = self.output_dir / "videos" / rel_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Process video
                if self.process_video(video_path, output_path):
                    stats["processed_videos"] += 1
                else:
                    stats["errors"] += 1
                    
                # Process corresponding JSON file
                json_path = video_path.with_suffix('.json')
                if json_path.exists():
                    if self.validate_json(json_path):
                        # Copy JSON to output directory
                        output_json = self.output_dir / "metadata" / rel_path.with_suffix('.json')
                        output_json.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(json_path, output_json)
                        stats["processed_metadata"] += 1
                    else:
                        stats["errors"] += 1
                        
            return stats
            
        except Exception as e:
            logger.error(f"Error during data preparation: {e}")
            raise

def main():
    """Main function to prepare the data."""
    try:
        # Initialize preparator
        preparator = DataPreparator()
        
        # Prepare data
        logger.info("Starting data preparation...")
        stats = preparator.prepare_data()
        
        # Log statistics
        logger.info("Data preparation completed")
        logger.info(f"Total files processed: {stats['total_files']}")
        logger.info(f"Videos processed: {stats['processed_videos']}")
        logger.info(f"Metadata files processed: {stats['processed_metadata']}")
        logger.info(f"Errors encountered: {stats['errors']}")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 