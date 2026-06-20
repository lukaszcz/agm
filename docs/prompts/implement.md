Implement the plan from %{PLAN_FILE}.

Divide the implementation work into well-scoped tasks doable by an agent in 200k context widow. If the steps described in the plan are too big, split them into multiple tasks.

Use Sonnet subagents to implement the tasks. When handing off each task, include a brief summary of the ultimate goals of the whole plan. Make it clear that any work done must be a step toward these goals. Write the subagent task description as a file under .agent-files/tasks/*.md that the implementer can reference. Use up to 3 Sonnet implementation agents in parallel, as reasonable.

After each implementation agent finishes, use an Opus subagent to review its work for correctness, completeness, maintainability, adherence to the plan and to relevant AGENTS.md files.

For EVERY issue identified by the reviewer, check if the issue is valid and if so, dispatch a Sonnet subagent to fix it. If the review surfaces deeper architectural problems, resolve them yourself first by making reasonable design and architecture choices. All solutions must be principled, general, extensible and maintainable. EVERY issue identified by a reviewer MUST be addressed.

Commit after completing each task.

The goal is not complete until all acceptance criteria of the plan are met.
