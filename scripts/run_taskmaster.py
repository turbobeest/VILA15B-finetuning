#!/usr/bin/env python3
import subprocess
import sys
import os

def run_taskmaster_command(args):
    """Run a taskmaster command and print the output."""
    cmd = ["python", ".taskmaster/scripts/taskmaster.py"] + args
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("\nOutput:")
        print(result.stdout)
        
        if result.stderr:
            print("\nErrors:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print("\nCommand failed with error:")
        print(e.stderr)

def main():
    if len(sys.argv) < 2:
        print("Available taskmaster commands:")
        print("  create-task - Create a new task")
        print("  update-task - Update task status")
        print("  create-subtask - Create a new subtask")
        print("  update-subtask - Update subtask status")
        print("\nExample: python run_taskmaster.py update-subtask task03 \"Dataset Preparation\" \"In Progress\"")
        return
    
    # Pass all arguments to the taskmaster script
    run_taskmaster_command(sys.argv[1:])

if __name__ == "__main__":
    main() 