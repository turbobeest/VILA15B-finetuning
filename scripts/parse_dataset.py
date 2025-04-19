import json
import os
from pathlib import Path
from typing import Dict, List, Set
import uuid

def parse_dataset(dataset_dir: str) -> List[Dict]:
    """
    Parse the dataset from a directory containing data.json and video files.
    Returns a list of dictionaries in VILA training format.
    """
    # Read the data.json file
    data_path = Path(dataset_dir) / "data.json"
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Track seen media_ids to filter duplicates
    seen_media_ids: Set[str] = set()
    vila_data: List[Dict] = []
    
    for item in data:
        media_id = item.get('media_id')
        if not media_id or media_id in seen_media_ids:
            continue
            
        seen_media_ids.add(media_id)
        
        # Construct video path
        video_path = f"videos/{media_id}.mp4"
        
        # Create VILA format conversation
        conversation = [
            {
                "from": "human",
                "value": "Describe what is happening in this video."
            },
            {
                "from": "gpt",
                "value": item.get('description', 'No description available.')
            }
        ]
        
        vila_item = {
            "id": str(uuid.uuid4()),
            "video": video_path,
            "conversations": conversation
        }
        
        vila_data.append(vila_item)
    
    return vila_data

def save_vila_dataset(vila_data: List[Dict], output_dir: str):
    """
    Save the parsed dataset in VILA format.
    """
    output_path = Path(output_dir) / "dataset.json"
    with open(output_path, 'w') as f:
        json.dump(vila_data, f, indent=2)

def main():
    # Example usage
    dataset_dir = "data/vila-dataset/PI-test-camera-1"
    output_dir = "data/vila-dataset/PI-test-camera-1"
    
    # Create videos directory if it doesn't exist
    videos_dir = Path(output_dir) / "videos"
    videos_dir.mkdir(exist_ok=True)
    
    # Parse the dataset
    vila_data = parse_dataset(dataset_dir)
    
    # Save the VILA format dataset
    save_vila_dataset(vila_data, output_dir)
    
    print(f"Processed {len(vila_data)} unique videos")
    print(f"Dataset saved to {output_dir}/dataset.json")

if __name__ == "__main__":
    main() 