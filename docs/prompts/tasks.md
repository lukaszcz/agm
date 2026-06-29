Read %{PLAN_FILE} and create implementation tasks for this plan. Make sure that each task can be implemented by an agent in 200k context widow. If the steps described in the plan are too big, split them into multiple tasks. In each task, add a brief section summarizing what the ultimate goals of the whole plan are. Make it clear that any work done must be a step toward these goals.

Save the task files to .agent-files/tasks/TASK_*.md. Create a task index in .agent-files/tasks/TASK_INDEX.md. Create .agent-files/tasks/PROGRESS.md to track task progress - list of completed and remaining tasks, next unblocked task.

## Format of PROGRESS.md

Only three sections:
1. Task status for each task (a list)
    - done / blocked / unblocked (not started / in progress)
2. Next unblocked task
3. Completion log
    - ONE line per task as each task lands (explicitly state this requirement)

Keep PROGRESS.md concise.
