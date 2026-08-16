Implement the supplied design brief.

Divide the work into well-scoped tasks doable by an agent in 200k context. Split work that is too large.

Use Sonnet subagents sequentially. Each handoff must summarize the design goals, state the task-specific acceptance criteria, and give the implementer an accessible task brief.

After each implementation agent finishes, use an Opus subagent to review correctness, completeness, maintainability, and adherence to relevant `AGENTS.md` files.

For every valid review finding, dispatch a Sonnet subagent to fix it. Resolve deeper architectural problems directly with principled, general, extensible, maintainable solutions.

Commit after completing each task. The work is complete only when every acceptance criterion is met.
