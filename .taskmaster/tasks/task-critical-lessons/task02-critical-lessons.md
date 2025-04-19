# Task 02: Research, Analysis & Prerequisite Gathering - Critical Lessons

*   **Environment Complexity:** The VILA fine-tuning process has specific and sometimes non-standard dependencies (specific flash-attn wheel, Git dependency `s2wrapper`, protobuf downgrade, potential Triton nightly build). Careful environment setup is crucial.
*   **DeepSpeed Modifications:** The need to manually copy files into the installed DeepSpeed package is a significant finding and potential source of errors if not done correctly.
*   **Code Structure:** Fine-tuning logic is embedded within the `llava` package structure, relying on custom trainers and data handling. Replicating requires understanding this structure, not just running a standalone script.
*   **Configuration:** Fine-tuning involves many parameters (Args classes) and leverages environment variables set by wrapper scripts (`sft.sh`). Direct Python calls will need to replicate this configuration.
*   **Data Format Adaptation:** The expected VILA JSON format differs from the project's input format. A conversion step will be necessary (Task 03).
*   **CUDA Version:** CUDA 12.1 seems to be the target version based on previous workspace and flash-attn wheel, update needed from initial 11.8 guess.
*   **Dataset Structure:** Each fine-tuned model should have its own dataset directory with a consistent structure (data.json + videos/). This allows for easy model-specific training and evaluation.
*   **Data Duplication:** The source data may contain duplicates that need to be filtered out during conversion to VILA format. The conversion script should handle this automatically. 