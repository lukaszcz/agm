# AGM Architecture Overview

AGM is an Agent Project Management CLI. It does two largely independent jobs: it sets up and operates *agent-oriented project directories* (workspaces, git worktrees, dependencies, sandboxes, tmux sessions, agent loops), and it implements *AgL*, a statically typed workflow language whose programs orchestrate agents and shell commands. A single `agm` executable exposes both.

Start here for the system shape, then read only the subsystem documents relevant to the task.

## System Shape

AGM is layered from a thin CLI down to reusable primitives, with AgL as a self-contained subsystem hanging off two commands (`agm exec`, `agm repl`):

- **CLI layer** — a Typer command tree whose directory structure mirrors the command tree exactly. It parses arguments into typed containers and dispatches to command implementations.
- **Command layer** — one module per command/command-group; each orchestrates domain logic but holds little of its own.
- **Domain layer** — project/workspace layout, git integration, sandboxing, tmux, configuration, and agent invocation.
- **Primitive layer** — process execution, environment handling, filesystem and TOML I/O, and a cross-cutting dry-run facility, plus pure generic utility helpers.
- **AgL subsystem** — a complete language implementation (lexer → parser → AST → scope → typecheck → match compilation → lower → IR eval) plus its host runtime, lazily imported so non-AgL commands stay fast to start.

## Architecture and Design Decisions

- **The command tree is the directory tree.** `src/agm/commands/` mirrors the CLI command hierarchy one-to-one, including nested groups (`config/`, `dep/`, `loop/`, `sync/`, `tmux/`, `workspace/`, `worktree/`). Finding a command's code is a path lookup.
- **Commands orchestrate; primitives do.** Command modules wire config, project layout, git, and agents together. Reusable behavior lives in `util/`, `core/`, `project/`, `vcs/`, `config/`, and `agent/`, never copied into individual commands.
- **Configuration is layered TOML.** Settings merge across install, home, project, and workspace scopes; per-command sections override base sections; AgL source config declarations and CLI flags override the file layers for the relevant commands.
- **The filesystem is the project model.** A project is a directory layout (embedded or split) plus git worktrees and dependency checkouts. AGM detects state from disk rather than maintaining a separate database.
- **Real agents are never run in tests, and never assumed.** Agent invocation is a subprocess boundary with timeout and output capture; runners are resolved from config and always have a default floor.
- **AgL is firewalled, not isolated.** The firewall is semantic: its passes depend only on a stable AST, never on the parser, and it is reached only through the `exec`/`repl` commands and lazily imported. It still reuses the shared layers below it rather than reimplementing them.

## What To Read Next

- Read [cli.md](cli.md) for the CLI definition, command dispatch, argument containers, and shell completion.
- Read [core.md](core.md) for the shared process, environment, filesystem, TOML, and dry-run primitives, and the pure utility helpers.
- Read [config.md](config.md) for configuration loading, layering precedence, and command/sandbox config sections.
- Read [workspaces.md](workspaces.md) for project layout, git worktrees, dependencies, sync, and tmux — the project-management half of AGM.
- Read [sandbox.md](sandbox.md) for `agm run`, the SRT sandbox, and resource limits.
- Read [agents.md](agents.md) for the agent runner and the loop/review/revise/refine workflows.
- Read [agl/index.md](agl/index.md) first for any AgL language task; it links to the AgL frontend, execution, modules, and REPL documents.
- Read [testing.md](testing.md) when changing tests, coverage, or the repository quality gates.

## Code Entry Points

- `src/agm/cli.py` defines the Typer app and every command group; `src/agm/parser.py` holds help text and command-overview resolution; `src/agm/completion.py` provides shell completions. `src/agm/command_catalog.py` is the pure data-leaf source of truth for top-level command names, shared by the CLI help layer and the AgL reserved-program-name guard.
- `src/agm/commands/` contains the command implementations, one subtree per command group.
- `src/agm/cli_support/` holds the typed argument containers that bridge the CLI layer and command implementations.
- `src/agm/core/` contains the cross-cutting process, environment, filesystem, and TOML primitives plus the dry-run facility; `src/agm/util/` holds pure, `agm`-import-free generic helpers (graph algorithms, text normalization).
- `src/agm/config/` implements loading and resolving general and sandbox configuration.
- `src/agm/project/` implements project/worktree setup and layout management.
- `src/agm/sandbox/` implements sandbox runtime/template support.
- `src/agm/tmux/` implements tmux session and layout logic.
- `src/agm/vcs/` implements git integration.
- `src/agm/agl/` is the AgL language implementation and host runtime.

### Additional directories

- `tests/` for the test suite.
- `docs/` for project documentation.
- `config/` for config templates.
- `stubs/` for local typing support for third-party modules.
- `tools/` for repository tooling.
