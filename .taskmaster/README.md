# Task Master Project Procedures

This document outlines the standard operating procedures for managing tasks and project artifacts using Task Master in this repository.

## 1. GitHub Repository and Commit Strategy

- **Initial Setup:** A GitHub repository should be set up for this project.
- **Commit Frequency:** Changes should be committed to the repository after the completion of every 1st order task and every 2nd order subtask. This ensures a granular history and facilitates rollbacks if necessary.

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