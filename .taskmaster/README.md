# Task Master Project Procedures

This document outlines the standard operating procedures for managing tasks and project artifacts using Task Master in this repository.

## 1. GitHub Repository and Commit Strategy

- **Initial Setup:** A GitHub repository should be set up for this project.
- **Commit Frequency:** Changes should be committed to the repository after the completion of every 1st order task and every 2nd order subtask.
- **Commit Procedure:**
    1. The AI Assistant will identify the files changed or created during the subtask.
    2. The AI Assistant will generate an informative commit message summarizing the changes.
    3. The AI Assistant will provide the necessary Git commands (`git add .` and `git commit -m "<generated_message>"`) for the human operator to execute.
    4. The human operator executes the commands in the terminal.

## 2. Task Documentation (Status, Results, Lessons)

Upon completion of **every 1st order task and 2nd order subtask**, the following markdown documents **must** be created and saved in their respective subdirectories within `.taskmaster/tasks/`:

1.  **`<task-id>-status.md`** (in `.taskmaster/tasks/task-status/`): Briefly describe the current status of the task (e.g., completed, blocked, needs review).
2.  **`<task-id>-results.md`** (in `.taskmaster/tasks/task-results/`): Detail the specific outcomes and deliverables of the completed task.
3.  **`<task-id>-critical-lessons.md`** (in `.taskmaster/tasks/task-critical-lessons/`): Document challenges encountered, solutions implemented, dependencies resolved, code adjustments made, and any critical insights gained. **Crucially, author this as if for another LLM that might need to understand the history and reasoning behind decisions to modify or fix the codebase later.**

*(Replace `<task-id>` with the actual task identifier, e.g., `task01`, `task03b`)*

## 3. Codebase File Tree Revision

At the end of **every 1st order task**, a revised codebase file tree must be generated and placed in a markdown document within the `.taskmaster/docs/` directory. The document should be named appropriately (e.g., `codebase-tree-after-task01.md`).

The tree should categorize files and directories using the following taxonomies to aid understanding and organization:

-   **Removable immediately:** Files/directories no longer needed.
-   **Temporary:** Test files, logs, temporary scripts, debugging code.
-   **Taskmaster-specific:** Files/directories related to Task Master setup and operation (`.taskmaster/`, `taskmaster.config.json`).
-   **Cursor-specific:** Files/directories related to Cursor IDE (e.g., `.cursor/`).
-   **Codebase-specific:** Core application/library source code.
-   **Libraries/Dependencies:** Third-party libraries, vendor directories, dependency manifests (e.g., `node_modules/`, `package.json`).
-   **Redundant:** Files/directories that duplicate others or can be consolidated.

For each entry, provide a brief note on its purpose.

## 4. Post-Tree Update Review

After updating the codebase tree document (Step 3), coordinate with the human operator to review the categorized tree. Discuss and decide on any necessary actions, such as moving, deleting, or reorganizing files and directories based on the categorization.

## 5. Task Handoff & Context Preparation (End of 1st Order Task)

To ensure smooth transitions between tasks, especially when starting new chat sessions, the following steps must be performed at the conclusion of every 1st order task:

1.  **Finalize Documentation:** Review the `task-status`, `task-results`, and `task-critical-lessons` markdown files for the completed task. Ensure they are comprehensive, accurate, and clearly written, capturing all relevant outcomes, deliverables, and insights.
2.  **Generate Transition Prompt:** Create a prompt to be used when initiating the *next* chat session for the subsequent task. This prompt should contain:
    *   **Project Context:** A brief reminder of the overall project goal (e.g., "Fine-tuning NVILA-15B for CCTV...").
    *   **Last Task Summary:** State the ID of the task just completed (e.g., "Task 02") and briefly mention its main outcome (e.g., "Completed research and dependency analysis"). Reference the corresponding `.taskmaster/tasks/` files for details.
    *   **Next Task:** State the ID and the primary goal of the *next* 1st order task to be started.
    *   **Procedure Reference:** Briefly remind the AI assistant to follow the procedures outlined in `.taskmaster/README.md`.
    *   Example:
        ```
        Project: NVILA-15B Fine-tuning for CCTV
        Last Task Completed: Task 02 - Research, Analysis & Prerequisite Gathering. Key dependencies and VILA repo structure analyzed. See .taskmaster/tasks/task-results/task02-results.md and task02-critical-lessons.md for details.
        Next Task to Start: Task 03 - Model & Data Acquisition and Preparation.
        Goal: Secure model files, update environment, and develop data loading/processing scripts.
        Remember to follow project procedures in .taskmaster/README.md (commits with provided commands/messages, task docs, tree updates, handoff).
        Let's begin Task 03, starting with Subtask 3.1.
        ```
3.  **Start New Chat (Optional):** The human operator can use the generated transition prompt to start a new chat session with the AI assistant for the next task. 