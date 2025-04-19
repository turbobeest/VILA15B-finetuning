import json
import os
from pathlib import Path
import shutil
import logging
from typing import Dict, List
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NWDatasetProcessor:
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create required directories
        (self.output_dir / "videos").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
    def load_data_json(self, split_dir: str) -> List[Dict]:
        """Load and validate data.json file."""
        data_path = self.source_dir / split_dir / "data.json"
        logger.info(f"Looking for data.json at: {data_path}")
        
        if not data_path.exists():
            logger.error(f"data.json not found in {split_dir}")
            return []
            
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    logger.error(f"Invalid data format in {data_path} - expected list")
                    return []
                logger.info(f"Loaded {len(data)} items from {data_path}")
                return data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in {data_path}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error loading {data_path}: {str(e)}")
            return []
            
    def process_video(self, video_path: Path, metadata: Dict) -> Dict:
        """Process a single video and its metadata."""
        try:
            # Copy video file
            dest_video = self.output_dir / "videos" / video_path.name
            shutil.copy2(video_path, dest_video)
            
            # Create VILA format metadata
            vila_metadata = {
                "id": metadata.get("uuid", str(uuid.uuid4())),
                "video": f"videos/{video_path.name}",
                "conversations": [
                    {
                        "from": "human",
                        "value": "Describe what is happening in this video."
                    },
                    {
                        "from": "gpt",
                        "value": metadata.get("description", "No description available.")
                    }
                ],
                "metadata": {
                    "type": metadata.get("type", "unknown"),
                    "risk": metadata.get("risk", 0),
                    "sensor": metadata.get("sensor", {}),
                    "start_timestamp": metadata.get("start_timestamp", {}),
                    "end_timestamp": metadata.get("end_timestamp", {})
                }
            }
            
            # Save metadata
            metadata_path = self.output_dir / "metadata" / f"{video_path.stem}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(vila_metadata, f, indent=2)
                
            return vila_metadata
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            return None
        
    def process_split(self, split_dir: str) -> List[Dict]:
        """Process all videos in a split directory."""
        logger.info(f"Processing split: {split_dir}")
        
        # Load data.json
        data = self.load_data_json(split_dir)
        if not data:
            return []
            
        processed_items = []
        split_path = self.source_dir / split_dir
        
        # Create media_id to metadata mapping
        media_map = {item.get("media_id"): item for item in data if item.get("media_id")}
        logger.info(f"Found metadata for {len(media_map)} videos")
        
        # Process each video file
        for video_path in split_path.glob("*.mp4"):
            video_id = video_path.stem
            metadata = media_map.get(video_id, {})
            
            if metadata:
                processed_item = self.process_video(video_path, metadata)
                if processed_item:
                    processed_items.append(processed_item)
                    logger.info(f"Processed video: {video_id}")
            else:
                logger.warning(f"No metadata found for video: {video_id}")
                
        return processed_items
        
    def process_dataset(self) -> List[Dict]:
        """Process the entire NW dataset."""
        all_processed_items = []
        
        # Process each split (20 and 80)
        for split in ["20", "80"]:
            processed_items = self.process_split(split)
            all_processed_items.extend(processed_items)
            
        # Save dataset registry
        registry = {
            "dataset_name": "NW",
            "data_path": str(self.output_dir),
            "meta_path": str(self.output_dir / "metadata"),
            "image_path": str(self.output_dir / "videos"),
            "description": "NW Camera Dataset processed in VILA format",
            "maintainer": "Project Team",
            "total_videos": len(all_processed_items)
        }
        
        with open(self.output_dir / "registry.json", 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2)
            
        logger.info(f"Processed {len(all_processed_items)} videos")
        return all_processed_items

def main():
    source_dir = Path("vila_workspace/datasets/NW").resolve()
    output_dir = Path("vila_workspace/datasets/NW-processed").resolve()
    
    logger.info(f"Processing dataset from {source_dir} to {output_dir}")
    processor = NWDatasetProcessor(str(source_dir), str(output_dir))
    processor.process_dataset()

if __name__ == "__main__":
    main() 