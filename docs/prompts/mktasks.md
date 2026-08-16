Read %{PLAN_FILE} and create implementation tasks. The tasks must collectively cover every plan requirement and acceptance criterion. Each task must fit an agent's 200k context; split larger work without losing coverage.

Give each task a brief `Context` section that summarizes the design goals and distinguishes those goals from the task-specific acceptance criteria.

Save task briefs to `.agent-files/tasks/TASK_*.md`, the index to `.agent-files/tasks/TASK_INDEX.md`, and the status tracker to `.agent-files/tasks/PROGRESS.md`.

## Status tracker format

Use only these sections:

1. Task status: one indexed entry per task, marked done, blocked, or unblocked (not started or in progress).
2. Next unblocked task.
3. Completion log: exactly one indexed line for each completed task.
