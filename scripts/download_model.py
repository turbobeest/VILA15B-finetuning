#!/usr/bin/env python3
"""
NVILA-15B Model Preparation Script

This script handles the verification and preparation of the NVILA-15B model files
for an airgapped system, including verification and proper directory structure setup.
"""

import os
import json
import logging
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelPreparator:
    def __init__(
        self,
        model_dir: str = "models/VILA-15B",
        expected_files: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the model preparator.
        
        Args:
            model_dir: Directory containing the model files
            expected_files: Dictionary of expected files and their SHA256 hashes
        """
        self.model_dir = Path(model_dir)
        self.expected_files = expected_files or {
            "config.json": None,  # Will be populated during verification
            "model.safetensors": None,
            "tokenizer.json": None,
            "tokenizer_config.json": None,
            "special_tokens_map.json": None
        }
        
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def verify_model_files(self) -> bool:
        """
        Verify the model files in the specified directory.
        
        Returns:
            bool: True if verification successful
        """
        try:
            if not self.model_dir.exists():
                logger.error(f"Model directory does not exist: {self.model_dir}")
                return False
                
            # Check for required files
            missing_files = []
            for file in self.expected_files:
                file_path = self.model_dir / file
                if not file_path.exists():
                    missing_files.append(file)
                    continue
                    
                # Calculate and store hash if not already known
                if self.expected_files[file] is None:
                    self.expected_files[file] = self.calculate_file_hash(file_path)
                    logger.info(f"Calculated hash for {file}: {self.expected_files[file]}")
            
            if missing_files:
                logger.error(f"Missing required files: {missing_files}")
                return False
                
            logger.info("All required files present")
            return True
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False

    def save_metadata(self) -> None:
        """
        Save metadata about the model files.
        """
        try:
            metadata = {
                "model_name": "VILA-15B",
                "files": {
                    file: {
                        "path": str(self.model_dir / file),
                        "sha256": self.expected_files[file]
                    }
                    for file in self.expected_files
                }
            }
            
            with open(self.model_dir / "model_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
                
            logger.info("Metadata saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise

def main():
    """Main function to verify and prepare the model."""
    try:
        # Initialize preparator
        preparator = ModelPreparator()
        
        # Verify model files
        if not preparator.verify_model_files():
            raise RuntimeError("Model verification failed")
            
        # Save metadata
        preparator.save_metadata()
        
        logger.info("Model preparation completed successfully")
        logger.info(f"Model is available at: {preparator.model_dir}")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 