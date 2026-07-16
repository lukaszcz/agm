Read ${TASKS_DIR}/TASK_INDEX.md. Update ${TASKS_DIR}/PROGRESS.md to track task progress - completed and remaining tasks, next unblocked task (not yet completed and not blocked by other tasks). Check against codebase sources. For the most recently completed task, verify the implementation against the corresponding task file and if not fully finished re-open the task with a description of what is missing.

Commit in ${TASKS_DIR} (separate repo) after updating ${TASKS_DIR}/PROGRESS.md.

Respond with either:
- file path of the next unblocked task,
- COMPLETE if you are certain that all tasks are complete.

CRITICAL: The response must contain ONLY the file path or COMPLETE on a single line, NO other text.

IMPORTANT: A task is not done until all its acceptance criteria are satisfied. Do not mark partially done tasks as complete. Do not consider deferrals as signs of completion.

## Format of PROGRESS.md

Only three sections:
1. Task status for each task (list)
    - done / blocked / unblocked (not started / in progress)
2. Next unblocked task
3. Completion log
    - ONE line per task as each task lands (explicitly state this requirement)

Keep PROGRESS.md concise.
