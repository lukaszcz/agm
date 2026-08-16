Read %{PLAN_FILE} and create implementation tasks. Each task must fit an agent's 200k context; split larger work.

Give each task a brief `Context` section that summarizes the design goals and distinguishes those goals from the task-specific acceptance criteria.

Save task briefs to `.agent-files/tasks/TASK_*.md`, the index to `.agent-files/tasks/TASK_INDEX.md`, and the status tracker to `.agent-files/tasks/PROGRESS.md`.

## Status tracker format

Use only these sections:

1. Task status: done, blocked, or unblocked (not started or in progress).
2. Next unblocked task.
3. Completion log: exactly one line for each completed task.
