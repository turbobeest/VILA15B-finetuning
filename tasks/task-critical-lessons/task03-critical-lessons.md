# Task 03 Critical Lessons

**Date:** 2025-04-19

**Lessons Learned & Troubleshooting:**

1.  **Conda/Pip Dependency Conflicts:** Installing VILA via `pip install -e` while having `opencv-python-headless` managed by conda caused conflicts. `pip` cannot reliably uninstall conda packages. 
    *   **Resolution:** Identified the conda-installed version using `pip` error messages, manually removed the package files (`rm -rf .../site-packages/cv2 .../site-packages/opencv_python_headless*`), then installed the specific version required by VILA using `pip install opencv-python-headless==4.8.0.76` (targeting the correct environment's python/pip).

2.  **DeepSpeed Patch Application:** The initial attempt to apply the DeepSpeed patch using `cp -rv source/* dest/` failed silently, leaving the target `mics.py` file empty. 
    *   **Resolution:** Re-ran the copy command explicitly targeting the source and destination file (`cp -v source/file dest/file`). Confirmed source file existed and contained expected content (`MiCS_Init`).

3.  **Model Loading (`Auto*` vs Specific Classes):** 
    *   `AutoModelForCausalLM` failed because `transformers` didn't recognize the `llava_llama` architecture type specified in `config.json`.
    *   `AutoProcessor` failed because the model's `config.json` lacked a `processor_class` entry.
    *   `AutoTokenizer` failed because it couldn't map the `LlavaLlamaConfig` class name back to a known tokenizer.
    *   **Resolution:** Inspected `config.json` and VILA source code. Explicitly imported and used the specific classes:
        *   Model: `LlavaLlamaModel` (from `llava.model.language_model.llava_llama`)
        *   Tokenizer: `Qwen2Tokenizer` (from `transformers`, based on `llm_cfg.model_type="qwen2"`, loaded from `./models/NVILA-15B/llm/`)
        *   Image Processor: `SiglipImageProcessor` (from `transformers`, based on `vision_tower_cfg.model_type="siglip_vision_model"`, loaded from `./models/NVILA-15B/vision_tower/`)

4.  **`torch_dtype` Conflict:** Providing `torch_dtype` to `LlavaLlamaModel.from_pretrained` caused a `TypeError` because the underlying VILA code also tried to set it.
    *   **Resolution:** Removed the explicit `torch_dtype` argument from the `LlavaLlamaModel.from_pretrained` call in the test script, allowing the VILA code to determine it from the config.

5.  **Editable Install (`pip install -e`) and `sys.path`:** Internal imports within the VILA package (e.g., `FloatPointQuantizeTorch`) failed initially despite the editable install.
    *   **Resolution:** Temporarily added the VILA source directory (`external/VILA`) to `sys.path` at the beginning of the test script (`scripts/test_model_load.py`) to ensure modules could be found during execution.

6.  **`.gitignore` Importance:** `git add .` was hanging because crucial large directories (`models/`, `data/`, `external/`) were not ignored. The default `.gitignore` from `task-master init` also incorrectly ignored `tasks.json` and `tasks/`.
    *   **Resolution:** Corrected `.gitignore` to ignore the large asset directories and un-ignore the essential task files/directories.

7.  **Git Lock Files:** Hanging `git` commands left behind `.git/index.lock`, preventing subsequent commands.
    *   **Resolution:** Manually removed the lock file (`rm -f .git/index.lock`).

## Dataset Processing
- **IMPORTANT**: Do not modify any files in `vila_workspace` directory
- All dataset processing should be done in proper locations within the project structure
- Always follow the taskmaster protocol when updating task status and documenting progress

## Environment Management
- All dependencies must be installed via conda as per project requirements
- Exceptions (pip installations) are only allowed for:
  - flash-attn (due to specific wheel requirements)
  - s2wrapper (due to Git installation)
  - VILA (due to editable mode requirement)

## Best Practices
- Document all actions in the appropriate taskmaster files
- Update task status after each significant step
- Always check existing code and configuration before creating new scripts
- Maintain the airgap compatibility of the codebase 