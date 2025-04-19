import shutil
import os
from pathlib import Path
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_dataset_structure():
    """Set up the NW dataset directory structure."""
    base_dir = Path("vila_workspace/datasets/NW")
    splits = ["20", "80"]
    
    # Create base directory
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create split directories
    for split in splits:
        split_dir = base_dir / split
        split_dir.mkdir(exist_ok=True)
        
        # Create required subdirectories
        (split_dir / "videos").mkdir(exist_ok=True)
        (split_dir / "metadata").mkdir(exist_ok=True)
    
    return base_dir

def copy_dataset_files(source_dir: Path, dest_dir: Path):
    """Copy dataset files from source to destination."""
    try:
        # Copy data.json
        data_json = source_dir / "data.json"
        if data_json.exists():
            shutil.copy2(data_json, dest_dir / "data.json")
            logger.info(f"Copied data.json to {dest_dir}")
        
        # Copy eval files if they exist
        eval_files = ["eval_answers.json", "eval_questions.json"]
        for file in eval_files:
            src_file = source_dir / file
            if src_file.exists():
                shutil.copy2(src_file, dest_dir / file)
                logger.info(f"Copied {file} to {dest_dir}")
        
        # Copy videos
        videos_dir = source_dir / "videos"
        if videos_dir.exists():
            dest_videos = dest_dir / "videos"
            dest_videos.mkdir(exist_ok=True)
            for video in videos_dir.glob("*.mp4"):
                shutil.copy2(video, dest_videos / video.name)
                logger.info(f"Copied video {video.name} to {dest_videos}")
    
    except Exception as e:
        logger.error(f"Error copying files: {str(e)}")
        raise

def main():
    # Source directory containing NW dataset
    source_base = Path("/home/jamie/vila_workspace/datasets/NW")
    
    # Set up destination structure
    dest_base = setup_dataset_structure()
    
    # Copy files for each split
    splits = ["20", "80"]
    for split in splits:
        source_dir = source_base / split
        dest_dir = dest_base / split
        
        if source_dir.exists():
            logger.info(f"Processing split {split}...")
            copy_dataset_files(source_dir, dest_dir)
        else:
            logger.warning(f"Source directory {source_dir} does not exist")

if __name__ == "__main__":
    main() 