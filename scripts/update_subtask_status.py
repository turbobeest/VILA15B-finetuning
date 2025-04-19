#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Import the TaskMaster class from the taskmaster.py file
sys.path.append(str(Path('.taskmaster/scripts')))
from taskmaster import TaskMaster

def main():
    """Update a subtask status directly using the TaskMaster class."""
    if len(sys.argv) < 4:
        print("Usage: python update_subtask_status.py <task_id> <subtask_id> <status> [notes]")
        print("Example: python update_subtask_status.py task03 'Dataset Preparation' 'In Progress' 'Processing NW-test-camera-1'")
        return
    
    task_id = sys.argv[1]
    subtask_id = sys.argv[2]
    status = sys.argv[3]
    notes = sys.argv[4] if len(sys.argv) > 4 else None
    
    taskmaster = TaskMaster()
    print(f"Updating subtask {subtask_id} in task {task_id} to status: {status}")
    taskmaster.update_subtask_status(task_id, subtask_id, status, notes)
    print("Done!")

if __name__ == "__main__":
    main() 