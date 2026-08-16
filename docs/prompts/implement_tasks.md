Implement the tasks from `.agent-files/tasks/TASK_*.md`, tracked in `.agent-files/tasks/PROGRESS.md`. The tasks implement the design in `.agent-files/PLAN.md`.

Use subagents sequentially. Choose the next unblocked task from the tracker.

A task is complete only when all its acceptance criteria are satisfied. Do not mark partial work complete or defer work. The plan is complete only after verifying that every plan requirement and acceptance criterion is satisfied.

After each implementation agent finishes, use a subagent to review correctness, completeness, maintainability, and adherence to the task brief and relevant `AGENTS.md` files.

Address every valid review finding. Resolve deeper architectural problems directly with principled, general, extensible, maintainable solutions.

After each task, commit the main repository, update the status tracker and related task material, and commit that tracking repository when it is separate.

## Status tracker format

Use only these sections:

1. Task status: one indexed entry per task, marked done, blocked, or unblocked (not started or in progress).
2. Next unblocked task.
3. Completion log: exactly one indexed line for each completed task.

Keep the tracker concise.
