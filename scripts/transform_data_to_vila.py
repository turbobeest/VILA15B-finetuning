import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Set

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def transform_entry(entry: Dict[str, Any], camera_dir_name: str) -> Dict[str, Any]:
    """Transforms a single entry from the original format to VILA format."""
    video_filename = f"{entry.get('media_id', '')}.mp4"
    
    # Construct video path relative to the base dataset directory
    # Assumes training script knows the base path ('data/vila-dataset')
    video_path = f"{camera_dir_name}/{video_filename}" 
    
    # Create the conversations structure
    description = entry.get('description', 'No description available.')
    conversations = [
        {
            "from": "human",
            "value": "<video>\nDescribe the video." 
        },
        {
            "from": "gpt",
            "value": description
        }
    ]
    
    # Map fields to the VILA format
    transformed = {
        "id": entry.get('uuid', entry.get('external_id', entry.get('_id', {}).get('$oid'))), # Use uuid or fallback
        "video": video_path,
        "conversations": conversations
    }
    return transformed

def process_camera_directory(camera_dir: Path, base_output_dir: Path):
    """Processes a single camera directory."""
    logger.info(f"--- Processing Camera Directory: {camera_dir.name} ---")
    original_data_path = camera_dir / "data.json"
    output_data_path = camera_dir / "data_vila_format.json" # Output to the same dir

    if not original_data_path.exists():
        logger.error(f"Missing data.json in {camera_dir}. Skipping.")
        return

    transformed_data: List[Dict[str, Any]] = []
    processed_media_ids: Set[str] = set()

    try:
        with open(original_data_path, 'r') as f:
            original_list = json.load(f)

        if not isinstance(original_list, list):
            logger.error(f"Invalid format in {original_data_path}: Expected a list. Skipping.")
            return

        for entry in original_list:
            if not isinstance(entry, dict):
                logger.warning(f"Skipping non-dictionary entry in {original_data_path}")
                continue

            media_id = entry.get('media_id')
            if not media_id:
                logger.warning(f"Skipping entry with missing 'media_id': {entry.get('uuid', 'N/A')}")
                continue
                
            # Handle duplicates based on media_id
            if media_id in processed_media_ids:
                logger.debug(f"Skipping duplicate media_id '{media_id}' in {camera_dir.name}")
                continue

            transformed_entry = transform_entry(entry, camera_dir.name)
            transformed_data.append(transformed_entry)
            processed_media_ids.add(media_id)
            
        # Write the transformed data to the new file
        with open(output_data_path, 'w') as f:
            json.dump(transformed_data, f, indent=2)
            
        logger.info(f"Successfully processed {len(transformed_data)} unique entries for {camera_dir.name}. Output: {output_data_path}")

    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in {original_data_path}. Skipping.")
    except Exception as e:
        logger.error(f"Error processing {camera_dir.name}: {str(e)}")

def main():
    base_dataset_dir = Path("./data/vila-dataset")
    
    if not base_dataset_dir.exists() or not base_dataset_dir.is_dir():
        logger.error(f"Base dataset directory not found: {base_dataset_dir}")
        return

    logger.info(f"Starting data transformation for directories in: {base_dataset_dir}")

    for item in base_dataset_dir.iterdir():
        if item.is_dir():
            process_camera_directory(item, base_dataset_dir) # Pass base dir for context if needed later

    logger.info("Data transformation process complete.")

if __name__ == "__main__":
    main() 