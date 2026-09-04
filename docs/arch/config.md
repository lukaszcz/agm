# Configuration

AGM is configured through layered TOML files resolved relative to a *config context* — the AGM home, the current project directory, and the invocation directory — and merged so that more specific scopes override more general ones. Commands read only the sections they need.

## Config Context

The context locates the directories that contribute configuration. The project directory comes from an environment variable when set, otherwise from walking up from the invocation directory until a recognizable project layout is found ([workspaces.md](workspaces.md)). This is what lets a command behave the same from the main workspace and from a branch worktree.

## AGM Home and the Standard Library

One AGM home holds config, prompts, sandbox settings, the AgL global library, and the package store: `AGM_HOME` when set, otherwise a populated `<install-prefix>/.agm` beside the executable, otherwise `~/.agm`. `just install` populates the prefix tree and installs the config templates from the repository's `config/`.

The AgL standard library resolves from `AGM_STDLIB` (an unchecked escape hatch), then a development `std` checkout whose module tree holds the invocation's anchor (the entry file, or the working directory) so the library is checked and run from the tree being edited, then the active managed `std` store package, which must exactly match the running AGM version, then the shipped fallback tree that `stdlib_locator.py` finds at the repository root in a checkout or bundled inside the wheel. See [package-store.md](package-store.md) for the managed package.

## Layering and Precedence

General configuration merges from least to most specific: the install prefix's `.agm/config.toml`, the AGM home's `config.toml`, the project config directory, and the workspace-local `.agm/config.toml`. Later layers override earlier ones; table-valued sections merge by key. Path-valued settings resolve relative to the file that defined them, expand `~`, and interpolate `%{VAR}` from the environment *leniently*: an unresolved hole stays verbatim, because one file serves every command and an irrelevant section must never prevent loading.

## Sections

Sections are consumed by specific features — `[loop]`, `[review]`, `[revise]`, `[run]`, `[exec]`, `[modules]`, `[packages]`. Per-command override sub-tables such as `[review.<name>]` merge over their base section. `[packages]` pins exact package versions and overrides global activation for that invocation. The `agm config` command group copies project config files into a workspace, prints the workspace environment as shell statements, and creates missing `config.toml` files.

## AgL Settings

AgL programs read two kinds of configuration, both routed through *qualified* tables named after the program's module path:

- **Engine settings** (`default-agent`, `log`, `log-file`, `strict-json`, `max-iters`, `timeout`) are `builtin var` bindings at the root of `std/config`. Precedence is source write > CLI flag > qualified program table > `[exec]` > engine default. Their catalog — name, value kind, config accessor, default, and which side of the host boundary consumes a write — is the pure data leaf `config/engine_keys.py`, shared with the AgL checker, IR validation, and evaluator.
- **Program arguments** — a `program def`'s own value parameters — resolve as CLI flag > qualified config table > signature default > required error, from the selected program's own qualified table (e.g. `[workflow.main]`, the same table an engine-key override reads), using the same `QualifiedConfigKey` shape as the engine-key mechanism. A required argument with no default reachable from the selected program is a pre-execution error.

`config/qualified_keys.py` routes a config table to a module by path suffix, keeps layer provenance, and rejects ambiguous suffixes. AGM's own top-level sections stay out of module matching, while a loose file with a reserved stem can still use its nested program table. How the hosts seed these values into a program is described in [agl/hosting.md](agl/hosting.md).

## Sandbox Configuration

`agm run` sandbox settings follow their own discovery and merge chain across the same scopes, selecting a per-command settings file with a default fallback ([sandbox.md](sandbox.md)).

## Code Entry Points

- `src/agm/config/context.py` — config context and project-directory discovery.
- `src/agm/config/general.py` — layer loading, merging, precedence retention, and per-feature readers.
- `src/agm/config/command_config.py` — per-command override sections.
- `src/agm/config/engine_keys.py` — the engine-key catalog; `src/agm/config/qualified_keys.py` — qualified AgL config-key routing.
- `src/agm/config/module_roots.py` — AgL module roots from `[modules]` and the `AGM_STDLIB` override; `src/agm/stdlib_locator.py` — the shipped stdlib tree.
- `src/agm/config/sandbox/` — SRT sandbox settings discovery and merging.
- `src/agm/commands/config/` — the `agm config` commands; `config/` (repository root) — the templates `just install` copies into the AGM home.
