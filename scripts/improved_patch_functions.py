#!/usr/bin/env python3
"""
Improved patch functions for VILA fine-tuning
This script contains improved patches that fix the image processor issue
"""
import os
import sys
import logging
import torch
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def apply_improved_patches():
    """Apply all improved patches to the VILA code"""
    # Add VILA library to the path
    external_vila_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external/VILA")
    sys.path.append(external_vila_path)
    
    # Import necessary modules
    from llava.train.args import TrainingArguments, DataArguments
    from transformers import Trainer, AutoImageProcessor, CLIPImageProcessor
    import llava.mm_utils

    # 1. Patch LLaVATrainer
    logger.info("Applying patch for LLaVATrainer")
    original_trainer_init = Trainer.__init__
    
    def patched_trainer_init(self, *args, **kwargs):
        original_trainer_init(self, *args, **kwargs)
        # Add missing attributes that might be used
        if hasattr(self, 'args'):
            if not hasattr(self.args, 'sample_lens'):
                logger.info("Adding missing 'sample_lens' attribute")
                self.args.sample_lens = []
            
            if not hasattr(self.args, 'eval_sample_lens'):
                logger.info("Adding missing 'eval_sample_lens' attribute")
                self.args.eval_sample_lens = []
                
            if not hasattr(self.args, 'longvila_sampler'):
                logger.info("Adding missing 'longvila_sampler' attribute")
                self.args.longvila_sampler = False
    
    # Apply the patch to the Trainer class
    Trainer.__init__ = patched_trainer_init
    
    # 2. Create a proper image processor
    logger.info("Creating proper image processor (setting resolution to 448x448 as requested)")
    default_processor = CLIPImageProcessor(
        size={"height": 448, "width": 448},  # Set to 448x448 (Note: may increase memory usage)
        image_mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP defaults
        image_std=[0.26862954, 0.26130258, 0.27577711],
        do_normalize=True,
        do_center_crop=True, # Note: Check if center cropping is desired for 448x448
        do_resize=True,
    )
    
    # 3. Patch mm_utils.process_image function
    logger.info("Applying improved patch for image processing")
    
    # Save original function
    original_process_image = llava.mm_utils.process_image
    
    # Define patched function
    def patched_process_image(image_file, data_args, image_folder, enable_dynamic_res=False, enable_dynamic_s2=False, max_tiles=None):
        # Add missing attributes with proper processor object
        if not hasattr(data_args, 'image_processor'):
            logger.info("Adding missing 'image_processor' attribute with proper processor object")
            data_args.image_processor = default_processor
        
        # If image_processor is a string, convert it to a proper processor
        if isinstance(data_args.image_processor, str):
            logger.info(f"Converting string processor '{data_args.image_processor}' to proper processor")
            try:
                # Try to load from local model path first
                model_path = getattr(data_args, 'model_path', './models/NVILA-15B')
                vision_tower_path = os.path.join(model_path, 'vision_tower')
                
                if os.path.exists(vision_tower_path):
                    logger.info(f"Loading processor from local vision tower: {vision_tower_path}")
                    data_args.image_processor = AutoImageProcessor.from_pretrained(
                        vision_tower_path,
                        trust_remote_code=True,
                        local_files_only=True
                    )
                else:
                    # Fallback to default processor
                    logger.info("Using default CLIP processor")
                    data_args.image_processor = default_processor
            except Exception as e:
                logger.warning(f"Error loading processor: {e}. Using default processor.")
                data_args.image_processor = default_processor
        
        # Also add s2 attribute if missing
        if not hasattr(data_args, 's2'):
            data_args.s2 = False
            
        # Set other required attributes for dynamic processing
        if not hasattr(data_args, 'min_tiles'):
            data_args.min_tiles = 1
            
        if not hasattr(data_args, 'max_tiles'):
            data_args.max_tiles = 12
            
        if not hasattr(data_args, 's2_scales'):
            data_args.s2_scales = "336,672,1008"
        
        # Call original function
        return original_process_image(image_file, data_args, image_folder, enable_dynamic_res, enable_dynamic_s2, max_tiles)
    
    # Apply the patch to the process_image function
    llava.mm_utils.process_image = patched_process_image
    
    # 4. Patch DataArguments to ensure it always has the needed attributes
    logger.info("Applying improved patch for DataArguments")
    original_data_args_init = DataArguments.__init__
    
    def patched_data_args_init(self, *args, **kwargs):
        original_data_args_init(self, *args, **kwargs)
        # Add missing attributes
        if not hasattr(self, 'image_processor'):
            self.image_processor = default_processor
        if not hasattr(self, 's2'):
            self.s2 = False
        if not hasattr(self, 'min_tiles'):
            self.min_tiles = 1
        if not hasattr(self, 'max_tiles'):
            self.max_tiles = 12
        if not hasattr(self, 's2_scales'):
            self.s2_scales = "336,672,1008"
        # Add missing video-related attributes
        if not hasattr(self, 'num_video_frames'):
            self.num_video_frames = 8  # Default to 8 frames per video
            
    # Apply the patch to the DataArguments class
    DataArguments.__init__ = patched_data_args_init
    
    # === Start Edit: Reinstate necessary part of Patch #5 ===
    logger.info("Applying patch to handle potential None media input")
    
    # Import the necessary class
    from llava.model.language_model.llava_llama import LlavaLlamaModel
    
    # Get the original forward method
    original_forward = LlavaLlamaModel.forward
    
    # Define the patched forward method (Simplified)
    def patched_forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, images=None, image=None, media=None, media_config=None, return_dict=None, **kwargs):
        
        # Simplest fix: Ensure media is an empty dict if it's None right before calling original
        if media is None:
            # logger.warning("Media was None, setting to empty dict.") # Optional: Keep logging if needed
            media = {}
            
        # Call the original forward method
        return original_forward(self, input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, images=images, image=image, media=media, media_config=media_config, return_dict=return_dict, **kwargs)
    
    # Apply the patch to the forward method
    LlavaLlamaModel.forward = patched_forward
    logger.info("Applied patch to LlavaLlamaModel.forward to handle media input.")
    # === End Edit: Simplified Patch #5 ===

    # NOTE: Patches 6 and 7 removed previously as they targeted non-existent methods
    
    logger.info("Relevant patches applied successfully") # Updated log message
    
    return True

if __name__ == "__main__":
    apply_improved_patches()
    print("Improved patches applied successfully. You can now run the fine-tuning script.") 