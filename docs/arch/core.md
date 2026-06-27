# Core Primitives

`core/` holds the low-level building blocks shared across every command: process execution, environment handling, filesystem and TOML/dotenv I/O, logging, and a cross-cutting dry-run facility. These modules have minimal coupling to AGM concepts and are the layer that actually touches the operating system.

## Process Execution

All subprocess work goes through a single process module rather than ad-hoc `subprocess` calls. It distinguishes running a command in the foreground (inheriting the terminal) from capturing its output, and offers "require success" variants that raise or exit on failure. It manages process groups and termination so that interrupting AGM cleanly tears down child processes — important for long-running agent and sandbox subprocesses, where a Ctrl-C must group-kill the child.

## Environment Handling

The environment module owns construction and resolution of process environments: cloning the ambient environment, resolving variable references, sourcing bash env files in a single shell to capture their effect, validating shell-safe identifiers, and locating the AGM installation prefix. Environments are passed explicitly as dictionaries through the call chain, so each command controls exactly what its subprocesses see.

## Filesystem, TOML, and Dotenv I/O

Filesystem mutations (mkdir, write, chmod, remove, glob) and TOML/dotenv reads and writes are wrapped so they participate in dry-run and present a consistent interface. TOML handling uses round-trip parsing so that updating a single key preserves the rest of a config file. Dotenv helpers upsert individual `.env` lines.

## Dry Run

Dry-run is a global, cross-cutting mode set from the `--dry-run` CLI flag. The primitives consult it: when enabled, process and filesystem operations print the action they *would* take instead of performing it. Because the check lives in the primitives, every command inherits dry-run support without implementing it individually.

## Code Entry Points

- `src/agm/core/process.py` — foreground/capture execution, success requirements, process-group termination.
- `src/agm/core/env.py` — environment cloning/resolution, env-file sourcing, shell-name validation, installation prefix.
- `src/agm/core/fs.py` — dry-run-aware filesystem operations.
- `src/agm/core/toml.py` and `src/agm/core/dotenv.py` — round-trip TOML and dotenv read/write helpers.
- `src/agm/core/dry_run.py` — global dry-run state and planned-command printing.
- `src/agm/core/log.py` — logging setup, including JSON trace logs used by AgL execution.
