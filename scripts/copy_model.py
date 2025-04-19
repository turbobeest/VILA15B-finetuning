import shutil
import os
from pathlib import Path
import logging
import stat

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def copy_with_permissions(src: Path, dst: Path):
    """Copy a file or directory while preserving permissions."""
    if src.is_file():
        shutil.copy2(src, dst)
        # Preserve permissions
        src_stat = src.stat()
        os.chmod(dst, src_stat.st_mode)
    elif src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        # Copy directory permissions
        src_stat = src.stat()
        os.chmod(dst, src_stat.st_mode)
        # Recursively copy contents
        for item in src.iterdir():
            copy_with_permissions(item, dst / item.name)

def main():
    # Source and destination paths
    source_dir = Path("/home/jamie/vila_workspace/models/NVILA-15B")
    dest_dir = Path("./models/NVILA-15B")
    
    try:
        # Create destination directory if it doesn't exist
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy each component
        components = ["llm", "vision_tower", "mm_projector"]
        for component in components:
            src = source_dir / component
            dst = dest_dir / component
            logger.info(f"Copying {component}...")
            copy_with_permissions(src, dst)
            
        # Copy root files
        root_files = ["config.json", "trainer_state.json", "README.md", ".gitattributes"]
        for file in root_files:
            src = source_dir / file
            dst = dest_dir / file
            if src.exists():
                logger.info(f"Copying {file}...")
                copy_with_permissions(src, dst)
                
        logger.info("Model copy completed successfully")
        
    except Exception as e:
        logger.error(f"Error copying model: {str(e)}")
        raise

if __name__ == "__main__":
    main() 