import torch
# Explicitly import Tokenizer, Image Processor, and Model classes
from transformers import AutoTokenizer, SiglipImageProcessor, Qwen2Tokenizer
from llava.model.language_model.llava_llama import LlavaLlamaModel
import logging
from pathlib import Path
import sys

# Add VILA source directory to Python path
VILA_SOURCE_DIR = Path(__file__).parent.parent / "external/VILA"
sys.path.insert(0, str(VILA_SOURCE_DIR))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the model path relative to the script location
MODEL_PATH = Path(__file__).parent.parent / "models/NVILA-15B"

def test_load_model():
    """Attempts to load the NVILA-15B model, tokenizer, and image processor explicitly."""
    model_path_str = str(MODEL_PATH.resolve())
    llm_path_str = str((MODEL_PATH / "llm").resolve()) # Path for tokenizer and LLM
    vision_tower_path_str = str((MODEL_PATH / "vision_tower").resolve()) # Path for image processor
    logger.info(f"Attempting to load components from: {model_path_str}")

    if not MODEL_PATH.exists() or not MODEL_PATH.is_dir():
        logger.error(f"Model directory not found at: {model_path_str}")
        return False
    if not (MODEL_PATH / "vision_tower").exists():
        logger.error(f"Vision tower directory not found at: {vision_tower_path_str}")
        return False
        
    tokenizer = None
    image_processor = None
    model = None

    try:
        # Load the Tokenizer explicitly from llm subdirectory
        logger.info(f"Loading tokenizer using Qwen2Tokenizer from: {llm_path_str}...")
        tokenizer = Qwen2Tokenizer.from_pretrained(llm_path_str, trust_remote_code=True)
        logger.info("Tokenizer loaded successfully.")

        # Load the Image Processor
        logger.info(f"Loading image processor from: {vision_tower_path_str}...")
        image_processor = SiglipImageProcessor.from_pretrained(vision_tower_path_str)
        logger.info("Image processor loaded successfully.")

        # Load the Model using the specific VILA class
        logger.info("Loading model using LlavaLlamaModel...")
        model = LlavaLlamaModel.from_pretrained(
            model_path_str, 
            trust_remote_code=True 
        )
        logger.info("Model loaded successfully.")
        
        logger.info(f"Model config class: {type(model.config)}")
        logger.info(f"Model device: {model.device}")

        # Final Check
        if tokenizer and image_processor and model:
             logger.info("Basic model, tokenizer, and image processor loading test PASSED.")
             return True
        else:
             logger.error("One or more components (tokenizer, image processor, model) failed to load.")
             return False

    except ImportError as e:
        logger.error(f"ImportError: {e}. Check dependencies.")
        return False
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {e}. Check model files.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during loading: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    if test_load_model():
        logger.info("Model loading test script finished successfully.")
    else:
        logger.error("Model loading test script finished with errors.")
        # Exit with a non-zero code to indicate failure for automation
        sys.exit(1) 