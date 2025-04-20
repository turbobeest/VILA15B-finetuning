#!/usr/bin/env python3
"""
Wrapper script for running finetune_vila.py with patches applied
"""
import os
import sys
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run VILA fine-tuning with patches")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fine_tuning_config.yaml",
        help="Path to the fine-tuning configuration"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/vila_finetuned",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    # Pass through all other args to the training script
    args, unknown = parser.parse_known_args()
    return args

# Use absolute paths for importing
project_root = "/home/jamie/vila15B-fine-tuning-conda"
scripts_dir = os.path.join(project_root, "scripts")

# Make sure these directories are in sys.path
for path in [project_root, scripts_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

logger.info(f"Python path setup: {sys.path[:5]}")
logger.info(f"Current working directory: {os.getcwd()}")

# Import and apply patches
try:
    sys.path.insert(0, scripts_dir)  # Ensure scripts is first
    from improved_patch_functions import apply_improved_patches
    apply_improved_patches()
    logger.info("Successfully applied patches")
except Exception as e:
    logger.error(f"Error applying patches: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Import finetune_vila using absolute path
try:
    finetune_path = os.path.join(scripts_dir, "finetune_vila.py")
    logger.info(f"Looking for finetune_vila.py at: {finetune_path}")
    
    if os.path.exists(finetune_path):
        logger.info(f"finetune_vila.py exists at this location")
    else:
        logger.error(f"finetune_vila.py NOT FOUND at this location")
        # List files in the scripts directory
        files = os.listdir(scripts_dir)
        logger.info(f"Files in scripts directory: {files}")
    
    # Try direct import with scripts dir in path
    try:
        from finetune_vila import main as finetune_main
        logger.info("Successfully imported finetune_vila via direct import")
    except ImportError as e:
        logger.error(f"Direct import failed: {e}")
        
        # Try with explicit module path
        # Create a spec and import the module directly
        import importlib.util
        spec = importlib.util.spec_from_file_location("finetune_vila", finetune_path)
        if spec is not None:
            finetune_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(finetune_module)
            finetune_main = finetune_module.main
            logger.info("Successfully imported finetune_vila via spec loader")
        else:
            raise ImportError(f"Could not create module spec from {finetune_path}")
    
    logger.info("Successfully imported finetune_vila.main")
except Exception as e:
    logger.error(f"Error importing finetune_vila: {e}")
    logger.error(f"Current sys.path: {sys.path}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def main():
    """Main entry point for the script."""
    args = parse_args()
    logger.info(f"Command line arguments: {args}")
    logger.info("Running finetune_vila.main() with patches applied")
    
    # Override sys.argv with the parsed args to pass to finetune_vila.main
    sys.argv = [sys.argv[0]]
    if args.config:
        sys.argv.extend(["--config", args.config])
    if args.output_dir:
        sys.argv.extend(["--output_dir", args.output_dir])
    if args.seed:
        sys.argv.extend(["--seed", str(args.seed)])
    if args.local_rank >= 0:
        sys.argv.extend(["--local_rank", str(args.local_rank)])
    
    logger.info(f"Passing arguments to finetune_vila: {sys.argv}")
    
    # Run finetune_vila.main()
    try:
        finetune_main()
        return 0
    except Exception as e:
        logger.error(f"Error running finetune_vila.main(): {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
