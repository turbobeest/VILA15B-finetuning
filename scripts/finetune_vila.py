#!/usr/bin/env python3
"""
NVILA-15B Fine-tuning Script

This script implements the fine-tuning strategy for NVILA-15B on CCTV footage
using the VILA framework. It supports:
- Loading and configuring the NVILA-15B model
- Parameter-efficient fine-tuning with LoRA
- DeepSpeed integration for efficient training
- Video data loading and preprocessing
- Train/validation splitting
- Evaluation metrics tracking
- Checkpoint management
"""

import os
import sys
import json
import yaml
import logging
import random
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Subset, random_split

import transformers
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    HfArgumentParser,
    TrainingArguments,
    set_seed
)

import deepspeed
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# Add the VILA library to the path
external_vila_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external/VILA")
sys.path.append(external_vila_path)

from llava.model import LlavaLlamaConfig, LlavaLlamaModel
from llava.mm_utils import process_video
from llava.train.args import DataArguments, ModelArguments, TrainingArguments
from llava.train.llava_trainer import LLaVATrainer
from llava.data.dataset import VideoQADataset
from llava.constants import IGNORE_INDEX

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("finetune_vila.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune NVILA-15B on CCTV footage")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fine_tuning_config.yaml",
        help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default="configs/deepspeed/zero3_config.json",
        help="Path to DeepSpeed configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/vila_finetuned",
        help="Directory to save model checkpoints and logs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization"
    )
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_and_preprocess_data(config: dict, tokenizer) -> Tuple[Dataset, Dataset]:
    """
    Load and preprocess training and validation data.
    
    Args:
        config: Configuration dictionary
        tokenizer: Tokenizer for the model
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger.info("Loading and preprocessing data...")
    
    # Load the training dataset
    train_data_path = config["data"]["train_data"][0]["path"]
    with open(train_data_path, "r") as f:
        data = json.load(f)
    
    # Get split parameters
    train_ratio = config["data"]["split_strategy"]["train_ratio"]
    val_ratio = config["data"]["split_strategy"]["val_ratio"]
    random_seed = config["data"]["split_strategy"]["random_seed"]
    
    # Create a list of indices and shuffle
    random.seed(random_seed)
    indices = list(range(len(data)))
    random.shuffle(indices)
    
    # Calculate split points
    train_size = int(len(data) * train_ratio)
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    logger.info(f"Dataset split: {len(train_indices)} training samples, {len(val_indices)} validation samples")
    
    # Create VILA-compatible datasets
    image_folder = config["data"]["image_folder"]
    num_video_frames = config["data"]["num_video_frames"]
    
    # Create a custom dataset for VILA
    train_dataset = VideoQADataset(
        data=[data[i] for i in train_indices],
        tokenizer=tokenizer,
        image_folder=image_folder,
        model_max_length=config["training"]["model_max_length"],
        num_video_frames=num_video_frames,
        image_aspect_ratio=config["data"]["image_aspect_ratio"],
    )
    
    val_dataset = VideoQADataset(
        data=[data[i] for i in val_indices],
        tokenizer=tokenizer,
        image_folder=image_folder,
        model_max_length=config["training"]["model_max_length"],
        num_video_frames=num_video_frames,
        image_aspect_ratio=config["data"]["image_aspect_ratio"],
    )
    
    return train_dataset, val_dataset

def setup_model_and_tokenizer(config: dict) -> Tuple[LlavaLlamaModel, AutoTokenizer]:
    """
    Set up the VILA model and tokenizer with LoRA configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info("Setting up model and tokenizer...")
    
    # Load model paths
    model_path = config["model"]["base_path"]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    
    # Load model configuration
    model_config = LlavaLlamaConfig.from_pretrained(model_path)
    
    # Update model configuration based on our settings
    model_config.tune_mm_projector = config["training"]["tune_mm_projector"]
    
    # Initialize model
    model = LlavaLlamaModel.from_pretrained(
        model_path,
        config=model_config,
        torch_dtype=eval(config["training"]["model_dtype"]),
    )
    
    # Configure LoRA if enabled
    if config["training"]["lora"]["enable"]:
        logger.info("Setting up LoRA for fine-tuning...")
        
        lora_config = LoraConfig(
            r=config["training"]["lora"]["r"],
            lora_alpha=config["training"]["lora"]["alpha"],
            lora_dropout=config["training"]["lora"]["dropout"],
            bias=config["training"]["lora"]["bias"],
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        
        model = get_peft_model(model, lora_config)
        logger.info(f"LoRA configured with rank {config['training']['lora']['r']}")
    
    return model, tokenizer

def prepare_training_args(config: dict, output_dir: str, deepspeed_config: str) -> TrainingArguments:
    """
    Prepare HuggingFace TrainingArguments.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save model and logs
        deepspeed_config: Path to DeepSpeed configuration
        
    Returns:
        TrainingArguments object
    """
    logger.info("Preparing training arguments...")
    
    # Optimizer and learning rate settings
    optimizer = config["training"]["optimizer"]["name"]
    learning_rate = config["training"]["optimizer"]["learning_rate"]
    weight_decay = config["training"]["optimizer"]["weight_decay"]
    
    # Scheduler settings
    scheduler_type = config["training"]["scheduler"]["type"]
    warmup_ratio = config["training"]["scheduler"]["warmup_ratio"]
    
    # Training parameters
    epochs = config["training"]["epochs"]
    per_device_batch_size = config["training"]["batch_size"]
    gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
    
    # Prepare training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        optim=optimizer,
        lr_scheduler_type=scheduler_type.lower(),
        save_strategy=config["training"]["save_strategy"],
        save_steps=config["training"]["save_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        evaluation_strategy=config["training"]["evaluation_strategy"],
        eval_steps=config["training"]["eval_steps"],
        logging_steps=10,
        remove_unused_columns=False,
        label_names=["labels"],
        deepspeed=deepspeed_config,
        fp16=False,
        bf16=True,
        report_to="tensorboard",
        overwrite_output_dir=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        tune_mm_projector=config["training"]["tune_mm_projector"],
        tune_vision_tower=config["training"]["tune_vision_tower"],
        tune_language_model=config["training"]["tune_language_model"],
        model_max_length=config["training"]["model_max_length"],
    )
    
    return training_args

def main():
    """Main function to run fine-tuning."""
    # Parse command line arguments
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Load and preprocess data
    train_dataset, val_dataset = load_and_preprocess_data(config, tokenizer)
    
    # Prepare training arguments
    training_args = prepare_training_args(config, args.output_dir, args.deepspeed)
    
    # Setup trainer
    trainer = LLaVATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info("Saving final model...")
    trainer.save_model(os.path.join(args.output_dir, "final"))
    
    # Perform final evaluation
    logger.info("Performing final evaluation...")
    final_metrics = trainer.evaluate()
    
    # Log final metrics
    with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    logger.info(f"Fine-tuning completed. Final metrics: {final_metrics}")

if __name__ == "__main__":
    main() 