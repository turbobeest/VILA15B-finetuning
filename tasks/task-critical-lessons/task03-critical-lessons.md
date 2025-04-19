# Task 03 Critical Lessons: Model Acquisition & Dataset Preparation

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