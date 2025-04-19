# NVILA-15B Fine-tuning Strategy for CCTV Footage

## Overview

This document outlines our strategy for fine-tuning the NVILA-15B model on CCTV footage datasets. The goal is to enhance the model's capability to understand and describe surveillance video content from specific cameras.

## Model

We're using the NVILA-15B model, which consists of:
- A 15B parameter multimodal LLM based on Llama 2
- A vision tower based on SigLIP
- A projector that connects the vision tower to the language model

## Data Strategy

### Available Datasets
- **NW-test-camera-1**: 491 video samples with descriptions
- **PI-test-camera-1**: 799 video samples with descriptions

### Data Splitting Approach
- **Train/Validation Split**: 80%/20% split of the NW-test-camera-1 dataset
- **Test Set**: The entire PI-test-camera-1 dataset will be used for testing
- **Rationale**: This approach allows us to train on one camera type and test on another, evaluating the model's ability to generalize to different camera perspectives and environments.

### Video Frame Sampling
- **Number of frames**: 8 frames per video
- **Rationale**: This provides a balance between capturing sufficient temporal information and computational efficiency. Eight frames should be adequate to capture key events in surveillance footage while keeping memory requirements manageable.

## Hyperparameters

### Learning Rate: 2.0e-5
- **Rationale**: A relatively low learning rate is chosen to avoid catastrophic forgetting of the pre-trained model's knowledge. The VILA model has sophisticated multimodal understanding that we want to preserve while adapting to our specific domain.

### Batch Size and Gradient Accumulation
- **Per-device batch size**: 1
- **Gradient accumulation steps**: 16
- **Effective batch size**: 16
- **Rationale**: Given memory constraints of the GPU (RTX A6000), we need to use a small per-device batch size and compensate with gradient accumulation. This balance allows for stable training without requiring excessive memory.

### Epochs: 3
- **Rationale**: With our dataset size (~400 training samples), three epochs strike a balance between sufficient learning and avoiding overfitting. This will result in approximately 75 update steps per epoch with our effective batch size.

### Parameter-Efficient Fine-tuning with LoRA
- **LoRA enabled**: Yes
- **LoRA rank (r)**: 64
- **LoRA alpha**: 16
- **Rationale**: LoRA significantly reduces the number of trainable parameters while still allowing effective adaptation. The relatively high rank (64) is chosen because our task requires adaptation to a specific visual domain (CCTV footage), which might benefit from higher expressivity.

### Components to Fine-tune
- **Vision tower**: Frozen (not fine-tuned)
- **Language model**: Fine-tuned (with LoRA)
- **MM projector**: Fine-tuned (fully)
- **Rationale**: The vision tower is kept frozen because it already performs well on general image understanding. The language model and projector are fine-tuned to adapt to the specific language patterns of CCTV descriptions and align the visual features with appropriate textual outputs.

### Precision
- **Model dtype**: torch.bfloat16
- **Rationale**: Using bfloat16 precision offers a good balance between computational efficiency and training stability, especially for large language models.

### Sequence Length
- **Maximum sequence length**: 4096
- **Rationale**: This allows for sufficiently detailed descriptions of video content, including complex scenes with multiple actors and events.

### Optimizer and Schedule
- **Optimizer**: AdamW
- **Learning rate schedule**: Cosine with warmup
- **Warmup ratio**: 0.03 (3% of steps)
- **Rationale**: AdamW with cosine schedule is a proven approach for language model fine-tuning. The small warmup helps stabilize the early phase of training.

## DeepSpeed Configuration
- **ZeRO Stage**: 3
- **Offload**: Disabled
- **Rationale**: ZeRO Stage 3 provides memory optimization without CPU offloading, as our GPU has sufficient memory for this configuration when using LoRA.

## Evaluation Strategy

### Frequency
- **Evaluation frequency**: Every 100 steps
- **Rationale**: Frequent evaluation allows for early detection of overfitting or other training issues.

### Metrics
1. **Loss**: Standard cross-entropy loss to measure prediction accuracy
2. **Perplexity**: To measure how well the model predicts the next token
3. **Accuracy**: For assessing the correctness of generated descriptions

### Qualitative Evaluation
Beyond metrics, we'll perform qualitative evaluation by:
1. Comparing generated descriptions to ground truth
2. Assessing the model's ability to identify key events and actors
3. Evaluating temporal understanding and coherence in descriptions

## Checkpoint Management
- **Save frequency**: Every 200 steps
- **Total checkpoints kept**: 3
- **Rationale**: Balances regular saving for recovery purposes against disk space usage.

## Additional Considerations

### Domain-Specific Features
- Fine-tuning will focus on improving the model's ability to recognize:
  - Common activities in CCTV footage (vehicles passing, people walking, etc.)
  - Relevant details in surveillance contexts (clothing descriptions, movement patterns)
  - Temporal relationships in video sequences

### Adaptation to Camera Characteristics
- The training process should help the model adapt to:
  - The specific angle and perspective of these cameras
  - The quality and resolution of the footage
  - The typical lighting conditions in these environments

## Next Steps

After defining this strategy (Task 4), we will:
1. Implement the fine-tuning script (Task 5)
2. Run initial fine-tuning on NW-test-camera-1 (Task 6)
3. Evaluate the fine-tuned model (Task 7)
4. Test on PI-test-camera-1 to assess generalization (Task 9) 