#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

# Import the TaskMaster class from the taskmaster.py file
sys.path.append(str(Path('.taskmaster/scripts')))
from taskmaster import TaskMaster

def main():
    """Provide a comprehensive interface to the TaskMaster functionality."""
    parser = argparse.ArgumentParser(description="TaskMaster Interface")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create task command
    create_parser = subparsers.add_parser("create-task", help="Create a new task")
    create_parser.add_argument("task_id", help="Task ID (e.g., task01)")
    create_parser.add_argument("title", help="Task title")
    create_parser.add_argument("description", help="Task description")
    create_parser.add_argument("--subtasks", nargs="+", help="List of subtasks")
    
    # Update task status command
    status_parser = subparsers.add_parser("update-task", help="Update task status")
    status_parser.add_argument("task_id", help="Task ID")
    status_parser.add_argument("status", help="New status")
    status_parser.add_argument("--notes", help="Additional notes")
    
    # Create subtask command
    subtask_parser = subparsers.add_parser("create-subtask", help="Create a new subtask")
    subtask_parser.add_argument("task_id", help="Task ID")
    subtask_parser.add_argument("subtask_id", help="Subtask ID")
    subtask_parser.add_argument("title", help="Subtask title")
    subtask_parser.add_argument("description", help="Subtask description")
    
    # Update subtask status command
    subtask_status_parser = subparsers.add_parser("update-subtask", help="Update subtask status")
    subtask_status_parser.add_argument("task_id", help="Task ID")
    subtask_status_parser.add_argument("subtask_id", help="Subtask ID")
    subtask_status_parser.add_argument("status", help="New status")
    subtask_status_parser.add_argument("--notes", help="Additional notes")
    
    # Add a command to list tasks
    list_parser = subparsers.add_parser("list", help="List all tasks")
    
    args = parser.parse_args()
    taskmaster = TaskMaster()
    
    if args.command == "create-task":
        taskmaster.create_task(args.task_id, args.title, args.description, args.subtasks or [])
    elif args.command == "update-task":
        taskmaster.update_task_status(args.task_id, args.status, args.notes)
    elif args.command == "create-subtask":
        taskmaster.create_subtask(args.task_id, args.subtask_id, args.title, args.description)
    elif args.command == "update-subtask":
        taskmaster.update_subtask_status(args.task_id, args.subtask_id, args.status, args.notes)
    elif args.command == "list":
        # This functionality isn't in the original TaskMaster class, so we'll add it here
        print("Available tasks:")
        for task_file in Path(taskmaster.tasks_dir).glob("*.md"):
            if task_file.name != "template.md":
                print(f"- {task_file.stem}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 