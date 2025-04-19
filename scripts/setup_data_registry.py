import json
import os
from pathlib import Path
from typing import Dict, List
import shutil
import logging
from dataclasses import dataclass
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    name: str
    description: str
    maintainer: str
    data_path: str
    meta_path: str
    image_path: str
    test_script: str = None

class DataRegistrySetup:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)  # Create base directory if it doesn't exist
        self.vila_format = {
            "wids_version": 1,
            "datasets": []
        }
        
    def create_dataset_structure(self, dataset_name: str) -> Path:
        """Create the required directory structure for a dataset."""
        dataset_dir = self.base_dir / dataset_name
        for subdir in ["videos", "metadata"]:
            (dataset_dir / subdir).mkdir(parents=True, exist_ok=True)
        return dataset_dir
        
    def process_dataset(self, dataset_name: str, source_dir: str) -> Dict:
        """Process a dataset and convert it to VILA format."""
        source_path = Path(source_dir)
        if not source_path.exists():
            logger.warning(f"Source directory not found: {source_dir}")
            return None
            
        dataset_dir = self.create_dataset_structure(dataset_name)
        
        # Process all video files
        for video_path in source_path.glob("**/*"):
            if video_path.is_file() and video_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
                # Copy video to videos directory
                dest_video = dataset_dir / "videos" / video_path.name
                shutil.copy2(video_path, dest_video)
                
                # Process corresponding metadata
                metadata_path = video_path.with_suffix('.json')
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        
                    # Convert to VILA format
                    vila_item = {
                        "id": str(uuid.uuid4()),
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
                        ]
                    }
                    
                    # Save metadata
                    dest_metadata = dataset_dir / "metadata" / metadata_path.name
                    with open(dest_metadata, 'w') as f:
                        json.dump(vila_item, f, indent=2)
                        
        return {
            "dataset_name": dataset_name,
            "data_path": str(dataset_dir),
            "meta_path": str(dataset_dir / "metadata"),
            "image_path": str(dataset_dir / "videos"),
            "description": f"Processed dataset for {dataset_name}",
            "maintainer": "Project Team"
        }
        
    def setup_registry(self, datasets: List[Dict]):
        """Set up the data registry with the given datasets."""
        for dataset in datasets:
            if dataset:  # Only add valid datasets
                self.vila_format["datasets"].append(dataset)
            
        # Save registry
        registry_path = self.base_dir / "registry.json"
        with open(registry_path, 'w') as f:
            json.dump(self.vila_format, f, indent=2)
            
        logger.info(f"Data registry created at: {registry_path}")
        
def main():
    # Example usage
    base_dir = "vila_workspace/datasets"
    setup = DataRegistrySetup(base_dir)
    
    # Process each dataset
    datasets = []
    for dataset_name in ["NW", "cctv_parking", "PI", "cctv_other", "cctv_entrance"]:
        source_dir = f"vila_workspace/datasets/{dataset_name}"
        dataset_config = setup.process_dataset(dataset_name, source_dir)
        if dataset_config:
            datasets.append(dataset_config)
            
    # Set up registry
    setup.setup_registry(datasets)
    
if __name__ == "__main__":
    main() 