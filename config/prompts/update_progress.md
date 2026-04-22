Read .agent-files/tasks/TASK_INDEX.md. Update .agent-files/tasks/PROGRESS.md to track task progress - completed and remaining tasks, next unblocked task.

Commit in .agent-files (git repo separate from main repo) after updating .agent-files/tasks/PROGRESS.md

Respond with either:
- file path of the next unblocked task,
- COMPLETE if you are certain that all tasks are complete.

The response must contain ONLY the file path or COMPLETE on a single line, no other text.

## Format of PROGRESS.md

Only three sections:
1. Task status for each task
    - done / blocked / unblocked (not started / in progress)
2. Next unblocked task
3. Completion log
    - ONE line per task as each task lands

Keep PROGRESS.md concise.
