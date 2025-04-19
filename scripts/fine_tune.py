#!/usr/bin/env python3
"""
NVILA-15B Fine-tuning Script

This script handles the fine-tuning of the NVILA-15B model using prepared video
and metadata data. It uses DeepSpeed for efficient training and includes
checkpointing and logging capabilities.
"""

import os
import json
import logging
import torch
import deepspeed
from pathlib import Path
from typing import Dict, List, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VILAFineTuner:
    def __init__(
        self,
        model_path: str = "models/VILA-15B",
        data_path: str = "data/processed",
        output_dir: str = "output/fine_tuned",
        deepspeed_config: str = "configs/deepspeed_config.json",
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 4,
        max_seq_length: int = 2048
    ):
        """
        Initialize the VILA fine-tuner.
        
        Args:
            model_path: Path to the pre-trained model
            data_path: Path to processed data
            output_dir: Directory for saving checkpoints and final model
            deepspeed_config: Path to DeepSpeed configuration
            batch_size: Training batch size
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps
            gradient_accumulation_steps: Gradient accumulation steps
            max_seq_length: Maximum sequence length
        """
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.deepspeed_config = Path(deepspeed_config)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_seq_length = max_seq_length
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
    def load_model_and_tokenizer(self):
        """Load the model and tokenizer."""
        try:
            logger.info("Loading model and tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model and tokenizer: {e}")
            raise
            
    def load_deepspeed_config(self) -> Dict:
        """Load and validate DeepSpeed configuration."""
        try:
            with open(self.deepspeed_config, 'r') as f:
                config = json.load(f)
                
            # Validate required fields
            required_fields = [
                "train_batch_size",
                "gradient_accumulation_steps",
                "optimizer",
                "scheduler"
            ]
            
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field in DeepSpeed config: {field}")
                    
            return config
            
        except Exception as e:
            logger.error(f"Error loading DeepSpeed config: {e}")
            raise
            
    def prepare_dataset(self) -> Dataset:
        """Prepare the training dataset from processed data."""
        try:
            logger.info("Preparing dataset...")
            
            # Load video paths and metadata
            video_dir = self.data_path / "videos"
            metadata_dir = self.data_path / "metadata"
            
            # Create dataset entries
            dataset_entries = []
            for video_path in video_dir.glob("**/*.mp4"):
                metadata_path = metadata_dir / video_path.with_suffix('.json').name
                if not metadata_path.exists():
                    logger.warning(f"Missing metadata for {video_path}")
                    continue
                    
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                # Create training example
                example = {
                    "video_path": str(video_path),
                    "description": metadata["description"],
                    "objects": json.dumps(metadata["objects"]),
                    "actions": json.dumps(metadata["actions"])
                }
                dataset_entries.append(example)
                
            # Create HuggingFace dataset
            dataset = Dataset.from_list(dataset_entries)
            logger.info(f"Dataset prepared with {len(dataset)} examples")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            raise
            
    def train(self):
        """Run the fine-tuning process."""
        try:
            # Load model and tokenizer
            self.load_model_and_tokenizer()
            
            # Load DeepSpeed config
            ds_config = self.load_deepspeed_config()
            
            # Prepare dataset
            dataset = self.prepare_dataset()
            
            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                per_device_train_batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                num_train_epochs=self.num_epochs,
                warmup_steps=self.warmup_steps,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                max_seq_length=self.max_seq_length,
                save_strategy="epoch",
                save_total_limit=2,
                logging_steps=10,
                report_to="none",  # Disable external logging
                deepspeed=str(self.deepspeed_config)
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=False
                )
            )
            
            # Start training
            logger.info("Starting training...")
            trainer.train()
            
            # Save final model
            logger.info("Saving final model...")
            trainer.save_model(str(self.output_dir / "final_model"))
            self.tokenizer.save_pretrained(str(self.output_dir / "final_model"))
            
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

def main():
    """Main function to run the fine-tuning process."""
    try:
        # Initialize fine-tuner
        fine_tuner = VILAFineTuner()
        
        # Start training
        fine_tuner.train()
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 