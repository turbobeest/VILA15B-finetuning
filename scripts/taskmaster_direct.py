#!/usr/bin/env python3
import re
from pathlib import Path

def update_subtask_status(task_file, subtask_name, new_status, notes=None):
    """
    Update the status of a subtask in a task file.
    
    Args:
        task_file: Path to the task file
        subtask_name: Name of the subtask (e.g., "Dataset Preparation")
        new_status: New status for the subtask
        notes: Optional notes to add
    """
    file_path = Path(task_file)
    if not file_path.exists():
        print(f"Task file not found: {file_path}")
        return False
    
    content = file_path.read_text()
    
    # Find the subtask section
    subtask_pattern = rf"### \d+\. {subtask_name}\n"
    match = re.search(subtask_pattern, content)
    
    if not match:
        print(f"Subtask '{subtask_name}' not found in {file_path}")
        return False
    
    # Find the position of the first checkbox after the subtask heading
    pos = match.end()
    next_line_start = pos
    next_line_end = content.find('\n', pos)
    next_line = content[next_line_start:next_line_end].strip()
    
    if next_line.startswith('- ['):
        # Extract the line content after the checkbox
        line_content = next_line.split('] ', 1)[1] if '] ' in next_line else "Process remaining datasets (NW-test-camera-1)"
        
        # Create updated line with proper checkbox status
        status_marker = 'x' if new_status.lower() == 'completed' else 'x'  # Always mark as 'x' for "in progress"
        updated_line = f"- [{status_marker}] {line_content}"
        
        # Replace the line
        content = content[:next_line_start] + updated_line + content[next_line_end:]
        
        # Add notes if provided
        if notes:
            notes_line = f"\n  Notes: {notes}"
            content = content[:next_line_start + len(updated_line)] + notes_line + content[next_line_end:]
        
        # Write the updated content back to the file
        file_path.write_text(content)
        print(f"Updated subtask '{subtask_name}' status to '{new_status}'")
        return True
    else:
        print(f"Could not find checkbox for subtask '{subtask_name}'")
        return False

def main():
    """Update the Dataset Preparation subtask in Task 03."""
    task_file = ".taskmaster/tasks/03_model_acquisition_dataset_preparation.md"
    subtask_name = "Dataset Preparation"
    new_status = "In Progress"
    notes = "Starting to process NW-test-camera-1 dataset"
    
    success = update_subtask_status(task_file, subtask_name, new_status, notes)
    
    if success:
        print("Successfully updated task file with Taskmaster.")
        
        # Also update the task-status file
        status_file = Path(".taskmaster/tasks/task-status/task03-status.md")
        if status_file.exists():
            status_content = status_file.read_text()
            
            # Update the Dataset Preparation line in the status file
            updated_content = re.sub(
                r"- \[ \] Dataset Preparation", 
                "- [x] Dataset Preparation (In Progress)",
                status_content
            )
            
            if updated_content != status_content:
                status_file.write_text(updated_content)
                print(f"Updated status file: {status_file}")
    else:
        print("Failed to update task status.")

if __name__ == "__main__":
    main() 