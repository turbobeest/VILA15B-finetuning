# Codebase File Tree - After Task 02

**Date:** $(date +%Y-%m-%d)

```
.
├── .taskmaster/
│   ├── docs/
│   │   ├── .gitkeep             # Taskmaster-specific: Placeholder for docs
│   │   ├── codebase-tree-after-task01.md  # Taskmaster-specific: Previous tree structure
│   │   └── codemap.md           # Taskmaster-specific: Current tree structure
│   ├── README.md                # Taskmaster-specific: Project procedures
│   ├── scripts/
│   │   └── .gitkeep             # Taskmaster-specific: Placeholder for TM scripts
│   └── tasks/
│       ├── task-critical-lessons/
│       │   ├── .gitkeep         # Taskmaster-specific: Placeholder
│       │   ├── task01-critical-lessons.md  # Taskmaster-specific: Task 01 Lessons
│       │   └── task02-critical-lessons.md  # Taskmaster-specific: Task 02 Lessons
│       ├── task-results/
│       │   ├── .gitkeep         # Taskmaster-specific: Placeholder
│       │   ├── task01-results.md  # Taskmaster-specific: Task 01 Results
│       │   └── task02-results.md  # Taskmaster-specific: Task 02 Results
│       ├── task-status/
│       │   ├── .gitkeep         # Taskmaster-specific: Placeholder
│       │   ├── task01-status.md   # Taskmaster-specific: Task 01 Status
│       │   └── task02-status.md   # Taskmaster-specific: Task 02 Status
│       ├── 03_model_acquisition_dataset_preparation.md  # Taskmaster-specific: Task 03 Prompt
│       └── template.md            # Taskmaster-specific: Template for new tasks
├── data/
│   ├── .gitkeep                 # Codebase-specific: Placeholder for input data
│   └── vila-dataset/            # Codebase-specific: VILA-formatted datasets
│       ├── PI-test-camera-1/    # Codebase-specific: Model-specific dataset
│       │   ├── videos/          # Codebase-specific: Video files
│       │   ├── data.json        # Codebase-specific: Dataset metadata
│       │   └── dataset.json     # Codebase-specific: VILA format dataset
│       └── NW-test-camera-1/    # Codebase-specific: Model-specific dataset
│           ├── videos/          # Codebase-specific: Video files
│           └── data.json        # Codebase-specific: Dataset metadata
├── models/
│   └── .gitkeep                 # Codebase-specific: Placeholder for model files
├── notebooks/
│   └── .gitkeep                 # Temporary: Placeholder for Jupyter notebooks
├── scripts/
│   ├── .gitkeep                 # Codebase-specific: Placeholder for runnable scripts
│   └── parse_dataset.py         # Codebase-specific: Dataset conversion script
├── src/
│   └── .gitkeep                 # Codebase-specific: Placeholder for source code
├── configs/                     # Codebase-specific: Configuration files
│   └── .gitkeep
├── external/                    # Codebase-specific: External dependencies
│   └── .gitkeep
├── .gitignore                   # Taskmaster-specific: Git ignore rules
├── environment.yml              # Libraries/Dependencies: Conda environment definition
└── taskmaster.config.json       # Taskmaster-specific: TM configuration
```

**Categorization Notes:**

*   **Removable immediately:** None
*   **Temporary:** `notebooks/` (contents will be temporary/exploratory)
*   **Taskmaster-specific:** `.taskmaster/`, `.gitignore`, `taskmaster.config.json`
*   **Cursor-specific:** None yet
*   **Codebase-specific:** `src/`, `data/`, `models/`, `scripts/`, `configs/`, `external/`
*   **Libraries/Dependencies:** `environment.yml`
*   **Redundant:** None

**Review & Actions:**

*   Structure established with new dataset and script additions
*   All directories properly categorized and documented 