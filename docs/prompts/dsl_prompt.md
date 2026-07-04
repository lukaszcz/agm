Implement the AgL agent DSL following the current AgL implementation requirements.

Use Sonnet subagents for well-scoped implementation work that fits in the context window. After each implementation agent finishes, use an Opus subagent to review its work for correctness, completeness, maintainability, and adherence to the current requirements. If the review surfaces deeper architectural problems, resolve them yourself by making reasonable design and architecture choices. All solutions must be principled, general, extensible, and maintainable.

After each work slice is fully finished, review the implementation with a Fable subagent, then use Opus subagents to address any implementation issues. For deeper architectural issues, own the design and delegate the implementation.

The goal is not complete until all acceptance criteria are met.
