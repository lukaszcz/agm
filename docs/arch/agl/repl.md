# AgL REPL and Program Hosting

`agm exec` runs a complete AgL program; `agm repl` evaluates one entry at a
time. Both use the same compilation pipeline and runtime. This document covers
the host behavior that makes the REPL incremental.

## Incremental Sessions

A REPL session retains successful declarations, bindings, types, and runtime
state, so later entries can refer to earlier entries without replaying them.
A later declaration at the same reachable path supersedes the earlier one.
Static failures do not promote their entry; completed runtime effects retain
their ordinary REPL behavior.

Imports and `use` declarations also persist after a successful entry. A later
import replaces retained declarations for the modules it names at the same
scope path. A later `use` replaces a retained use with the same resolved target
at that path, even when an import alias changes; uses and imports at other paths
remain. `:reset` clears retained declarations, imports, uses, and session runtime state. Retained
explicit `std/core` imports suppress the normal per-entry prelude; `--no-stdlib`
disables that prelude for the whole session.

## Hosts and Settings

`exec` discovers a program's parameters and selects its entry function. The
REPL obtains imported parameter values from configuration and requires defaults
for entry-local parameters. Both hosts seed engine settings and use the shared
runtime for agents, shell commands, and standard-library services.

## Code Entry Points

- `src/agm/agl/repl/` — session state, entry pipeline, console, rendering, and
  agent confirmation.
- `src/agm/agl/pipeline.py` — shared preparation and execution pipeline.
- `src/agm/commands/exec.py` and `src/agm/commands/repl.py` — CLI hosts.
- `src/agm/cli_support/` — execution parameters and host engine-setting seeds.
- Tests: `tests/test_agl_repl_*.py` and `tests/test_exec_command.py`.
