#!/usr/bin/env python3
import sys
from pathlib import Path

# Import the TaskMaster class from the taskmaster.py file
sys.path.append(str(Path('.taskmaster/scripts')))
from taskmaster import TaskMaster

def main():
    """Update Task 03 Dataset Preparation subtask status."""
    if len(sys.argv) < 2:
        print("Usage: python update_dataset_preparation.py <status> [notes]")
        print("Example: python update_dataset_preparation.py 'In Progress' 'Processing NW-test-camera-1'")
        return
    
    task_id = "03_model_acquisition_dataset_preparation"
    subtask_id = "2"  # Dataset Preparation is subtask 2 in Task 03
    status = sys.argv[1]
    notes = sys.argv[2] if len(sys.argv) > 2 else None
    
    taskmaster = TaskMaster()
    print(f"Updating Dataset Preparation subtask to status: {status}")
    taskmaster.update_subtask_status(task_id, subtask_id, status, notes)
    print("Done!")

if __name__ == "__main__":
    main() 