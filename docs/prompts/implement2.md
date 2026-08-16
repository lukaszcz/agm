Implement the design brief from `.agent-files/PLAN.md`.

Divide the work into well-scoped tasks doable by an agent in 200k context. Split work that is too large.

Use subagents sequentially. Each handoff must summarize the design goals, state the task-specific acceptance criteria, and provide the task description directly in the request.

After each implementation agent finishes, use a subagent to review correctness, completeness, maintainability, and adherence to relevant `AGENTS.md` files.

Address every valid review finding. Resolve deeper architectural problems directly with principled, general, extensible, maintainable solutions.

Commit after completing each task. The work is complete only when every acceptance criterion is met.
