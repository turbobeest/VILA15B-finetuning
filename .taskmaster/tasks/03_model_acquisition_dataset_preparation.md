# Task 03: Model Acquisition & Dataset Preparation

## Objective
Acquire the NVILA-15B model and prepare the dataset for fine-tuning, ensuring all components are properly structured and validated.

## Subtasks

### 1. Model Acquisition
- [ ] Download NVILA-15B model from Hugging Face
- [ ] Verify model components:
  - [ ] LLM weights and config
  - [ ] Tokenizer files
  - [ ] Vision tower weights and config
  - [ ] MM Projector weights
  - [ ] Overall VILA/LLaVA config
- [ ] Set up model files in correct directory structure
- [ ] Validate model loading and basic functionality

### 2. Dataset Preparation
- [ ] Process remaining datasets (NW-test-camera-1)
- [ ] Validate dataset structure:
  - [ ] Video files present and accessible
  - [ ] data.json format correct
  - [ ] No duplicate entries
- [ ] Set up data registry in VILA format
- [ ] Create dataset validation script
- [ ] Document dataset statistics and characteristics

### 3. Environment Setup
- [ ] Install flash-attention wheel
- [ ] Install s2wrapper from GitHub
- [ ] Clone and install VILA in editable mode
- [ ] Apply DeepSpeed patch
- [ ] Verify all dependencies are correctly installed

## Deliverables
1. Model files properly downloaded and structured
2. All datasets processed and validated
3. Environment fully configured for training
4. Documentation of any issues or findings
5. Updated task status and critical lessons

## Success Criteria
- Model can be loaded and run basic inference
- All datasets are in correct VILA format
- Environment passes all dependency checks
- All components are properly documented 