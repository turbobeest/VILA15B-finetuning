# Task 02: Research, Analysis & Prerequisite Gathering - Results

**Deliverables:**

*   **Previous Workspace Analysis:**
    *   Identified Docker/script-based workflow, potential complexity.
    *   Extracted key dependencies (Python 3.10, PyTorch 2.3, CUDA 12.1, Transformers 4.46, Accelerate, PEFT, BitsAndBytes, DeepSpeed 0.9.5/0.10.0, OpenCV, pytorchvideo, decord, flash-attn, s2wrapper Git dep, protobuf 3.20).
    *   Noted need for Conda focus and careful air-gap planning.
*   **NVLabs/VILA Repo Analysis:**
    *   Located core logic in `llava/` (model, train, data).
    *   Identified main training script: `llava/train/train_mem.py` (calls `train.py`).
    *   Found custom trainer `LLaVATrainer` with specialized samplers, optimizer creation, and saving logic.
    *   Confirmed data format (JSON + media files) and registration process (`llava/data/registry/datasets/default.yaml`).
    *   Parsed fine-tuning arguments (`llava/train/args.py`).
    *   Found critical setup step: copying files into installed `deepspeed` package.
*   **Hugging Face Page Review:**
    *   Confirmed NVILA-15B as target model (efficiency, video support).
    *   Noted license details (Apache 2.0 code, CC-BY-NC-4.0 weights).
*   **Model File Identification:**
    *   LLM Weights & Config
    *   Tokenizer Files
    *   Vision Tower Weights & Config (e.g., PaliGemma)
    *   MM Projector Weights (likely in main checkpoint)
    *   Overall VILA/LLaVA Config
*   **Dependency List (Consolidated Draft):** Python 3.10, PyTorch 2.3.0, CUDA 12.1, Transformers 4.46.0, Accelerate 0.34.2, PEFT>=0.9.0, BitsAndBytes 0.43.2, DeepSpeed 0.9.5, OpenCV-headless 4.8.0, pytorchvideo 0.1.5, decord 0.6.0, sentencepiece 0.1.99, protobuf 3.20.*, ninja, wandb, flash-attn 2.5.8 (wheel), s2wrapper (Git), triton 3.1.0 (optional/quantization). 