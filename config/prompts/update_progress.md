Read ${TASKS_DIR}/TASK_INDEX.md. Update ${TASKS_DIR}/PROGRESS.md to track task progress - completed and remaining tasks, next unblocked task (not yet completed and not blocked by other tasks). Check against codebase sources.

Commit in ${TASKS_DIR} (git repo separate from main repo) after updating ${TASKS_DIR}/PROGRESS.md

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
