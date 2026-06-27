# AGM commands reference

AGM is an Agent Project Management CLI. A single `agm` executable manages
agent-oriented project directories — workspaces, dependencies, sandboxing, git worktrees, and tmux sessions — and runs agent review/revise/refine and loop
workflows on them, including programs written in the AgL workflow DSL.

This reference describes each command's behavior and options from an end-user
perspective.

## Global usage

```text
agm <command> [options] [args]
```

Global options:

- `--dry-run`
- `--install-completion`
- `--show-completion`

`agm help` prints the command overview, and `agm help <command>` prints detailed
help for a single command or command group. Each command also accepts `--help`.

## Chapters

| Chapter | Contents |
| ------- | -------- |
| [Workspace and project lifecycle](workspaces.md) | `agm open`/`close`, `agm workspace`/`wsp`, `agm init`, `agm sync` |
| [Agent workflows](agents.md) | `agm review`, `agm revise`, `agm refine` |
| [Loop automation](loop.md) | `agm loop` run/step/select, prompts, selectors, logging |
| [AgL workflow DSL](agl.md) | `agm exec`, `agm repl` |
| [Configuration](config.md) | `agm config` copy/env/update |
| [Dependencies](dependencies.md) | `agm dep` |
| [Sandboxing](run.md) | `agm run` |
| [Worktrees](worktrees.md) | `agm worktree`/`wt` |
| [tmux sessions](tmux.md) | `agm tmux` |
