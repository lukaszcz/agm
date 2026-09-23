# AGM Architecture Overview

AGM is an Agent Project Management CLI. It does two largely independent jobs: it sets up and operates *agent-oriented project directories* (workspaces, git worktrees, dependencies, sandboxes, tmux sessions, agent loops), and it implements *AgL*, a statically typed workflow language whose programs orchestrate agents and shell commands. A single `agm` executable exposes both.

Start here for the system shape, then read only the subsystem documents relevant to the task.

## System Shape

AGM is layered from a thin CLI down to reusable primitives, with AgL as a self-contained subsystem used by `agm exec`, `agm repl`, `agm check`, package validation, and package-registered commands:

- **CLI layer** — a Typer command tree whose directory structure mirrors the command tree exactly. It parses arguments into typed containers and dispatches to command implementations.
- **Command layer** — one module per command/command-group; each orchestrates domain logic but holds little of its own.
- **Domain layer** — project/workspace layout, packages, git integration, sandboxing, tmux, configuration, and agent invocation.
- **Primitive layer** — process execution, environment handling, filesystem and TOML I/O, outbound HTTP, and a cross-cutting dry-run facility, plus pure generic utility helpers.
- **AgL subsystem** — a complete language implementation (lexer → parser → AST → scope → typecheck → match compilation → lower → IR eval) plus its host runtime. Its public façade exports the execution stack lazily, so lower domains can use AgL leaves without loading it.

## Architecture and Design Decisions

- **The command tree is the directory tree.** `src/agm/commands/` mirrors the CLI command hierarchy one-to-one, including nested groups. Finding a command's code is a path lookup.
- **Commands orchestrate; primitives do.** Command modules wire config, project layout, git, and agents together. Reusable behavior lives in `util/`, `core/`, `project/`, `vcs/`, `config/`, and `agent/`, never copied into individual commands.
- **Configuration is layered TOML.** Settings merge across install, home, project, and workspace scopes; per-command sections override base sections; AgL source writes and CLI flags override the file layers for the relevant commands.
- **The filesystem is the project model.** A project is a directory layout (embedded or split) plus git worktrees and dependency checkouts. AGM detects state from disk rather than maintaining a separate database.
- **Real agents are never run in tests, and never assumed.** Agent invocation is a subprocess boundary with timeout and output capture; runners are resolved from config and always have a default floor.
- **AgL is firewalled, not isolated.** Its static passes depend only on a stable AST, never on the parser, and its execution façade is lazily imported by its CLI and package-domain callers. It still reuses the shared layers below it rather than reimplementing them.

## What To Read Next

- [cli.md](cli.md) — the CLI definition, command dispatch, argument containers, and shell completion.
- [core.md](core.md) — the shared process, environment, filesystem, TOML, and dry-run primitives, and the pure utility helpers.
- [config.md](config.md) — configuration loading, layering precedence, the AGM home, and standard-library location.
- [workspaces.md](workspaces.md) — project layout, git worktrees, dependencies, sync, and tmux: the project-management half of AGM.
- [packages.md](packages.md) — package manifests, identity, dependencies, validation, and registered commands; [package-store.md](package-store.md) — the store, activation, installation, and the managed `std` package.
- [sandbox.md](sandbox.md) — the sandbox preparation library, `agm run`, the SRT backend, and resource limits.
- [agents.md](agents.md) — the agent runner, sessions, and the loop/review/revise/refine workflows.
- [agl/index.md](agl/index.md) — start here for any AgL language task; it links to the frontend, execution, module, hosting, and REPL documents.
- [testing.md](testing.md) — tests, coverage, and the repository quality gates.

## Code Entry Points

- `src/agm/cli.py` defines the Typer app and every command group; `src/agm/cli_dispatch.py` is the fallback for package-registered commands; `src/agm/parser.py` holds help text; `src/agm/completion.py` provides shell completions. `src/agm/command_catalog.py` is a pure data leaf for CLI help.
- `src/agm/commands/` contains the command implementations, one subtree per command group.
- `src/agm/cli_support/` holds the typed argument containers and the shared AgL CLI support (engine seeds, program parameters, execution roots and targets).
- `src/agm/core/` holds the process, environment, filesystem, TOML, HTTP, logging, and dry-run primitives; `src/agm/util/` holds pure, `agm`-import-free helpers.
- `src/agm/config/` loads and resolves general and sandbox configuration; `src/agm/stdlib_locator.py` finds the shipped standard library.
- `src/agm/agent/` implements agent runner invocation and session backends.
- `src/agm/project/`, `src/agm/vcs/`, `src/agm/tmux/` implement project layout, git integration, and tmux sessions.
- `src/agm/packages/` implements package manifests, identity, the store, and discipline validation.
- `src/agm/sandbox/` implements the sandbox preparation library, with SRT as its shipped backend.
- `src/agm/agl/` is the AgL language implementation and host runtime; `packages/stdlib/src/` is the AgL standard library with its Python companions.
- `src/agm/version.py` defines AGM's release version; package and project metadata keep it in lockstep.
- `tests/` holds the test suite, `docs/` the documentation, `config/` the config templates and editor modes, `stubs/` local typing stubs, and `tools/` repository tooling.
