# AGM commands reference

AGM is an Agent Project Management CLI. A single `agm` executable manages
agent-oriented project directories — workspaces, git worktrees, dependencies,
sandboxes, and tmux sessions — and runs agent review/revise/refine and loop
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

## Getting help

`agm help` prints the command overview, and `agm help <command>` prints detailed
help for a single command or command group. Each command also accepts `--help`.

Most command groups have a shorter alias (`wsp` for `workspace`, `wt` for
`worktree`), and several commands have alias forms (`rm` for `remove`, `cp` for
`copy`); each alias is listed alongside its command in the relevant chapter.

## Chapters

| Chapter | Contents |
| ------- | -------- |
| [Workspace and project lifecycle](workspaces.md) | `agm open`/`close`, `agm workspace`/`wsp`, `agm init`, `agm sync` |
| [Agent workflows](agents.md) | `agm review`, `agm revise`, `agm refine` |
| [Loop automation](loop.md) | `agm loop` run/step/select, prompts, selectors, logging |
| [AgL workflow DSL](agl.md) | `agm exec`, `agm repl` |
| [Configuration](config.md) | `agm config` copy/env/update |
| [Dependencies](dependencies.md) | `agm dep` |
| [Sandboxed command execution](run.md) | `agm run` |
| [Worktrees](worktrees.md) | `agm worktree`/`wt` |
| [tmux sessions](tmux.md) | `agm tmux` |
