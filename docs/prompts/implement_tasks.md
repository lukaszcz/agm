Implement the tasks from .agent-files/tasks/TASK_*.md, tracked in .agent-files/tasks/PROGRESS.md. The tasks describe implementation steps for the plan in .agent-files/PLAN.md

Use subagents to implement the tasks. Launch subagents sequentially one at a time, not in parallel. Choose the next task based on .agent-files/tasks/PROGRESS.md.

A task is not done until all its acceptance criteria are satisfied. Do not mark partially done tasks as complete. Never defer any work.

After each implementation agent finishes, use a subagent to review its work for correctness, completeness, maintainability, adherence to the task spec file and to relevant AGENTS.md files.

For EVERY issue identified by the reviewer, check if the issue is valid and if so, dispatch a subagent to fix it. If the review surfaces deeper architectural problems, resolve them yourself first by making reasonable design and architecture choices. All solutions must be principled, general, extensible and maintainable. EVERY issue identified by a reviewer MUST be addressed.

After completing each task:
1. commit in the main repo,
2. update .agent-files/tasks/PROGRESS.md and other relevant files in .agent-files/tasks,
3. commit in .agent-files (separate repo).

## Acceptance criteria

The goal is not complete until all acceptance criteria of every task and of the plan are met.

## Format of PROGRESS.md

Only three sections:
1. Task status for each task
    - done / blocked / unblocked (not started / in progress)
2. Next unblocked task
3. Completion log
    - ONE line per task as each task lands (explicitly state this requirement)

Keep PROGRESS.md concise.
