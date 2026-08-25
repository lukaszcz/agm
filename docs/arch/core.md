# Core Primitives

Two foundation packages sit beneath everything else, and *both* are shared by both halves of AGM — the project-management commands and the AgL runtime alike. `core/` holds the OS-facing building blocks: process execution, environment handling, filesystem and TOML/dotenv I/O, logging, lifecycle cleanup, and a cross-cutting dry-run facility; AgL's host runtime runs shell commands and agents, clones environments, writes files, and emits JSONL trace logs through these same primitives. `util/` holds pure, stdlib-only generic helpers that import nothing from `agm`.

## Process Execution

Ordinary foreground and captured subprocess work goes through the shared process module. It distinguishes terminal-inheriting commands from captured output, offers "require success" variants, and manages process groups so interruption tears down descendants. The persistent Pi RPC session is the deliberate exception: `agent/session/rpc.py` owns a raw streaming `Popen`, nonblocking bounded writes, reader threads, and equivalent process-group teardown because the request/response process must outlive one capture call.

## Environment Handling

The environment module owns construction and resolution of process environments: cloning the ambient environment, resolving variable references, sourcing bash env files in a single shell to capture their effect, validating shell-safe identifiers, and locating the AGM installation prefix from the path AGM was invoked through, so each build resolves its own prefix rather than an unrelated AGM on PATH. Environments are passed explicitly as dictionaries through the call chain, so each command controls exactly what its subprocesses see.

## Filesystem, TOML, and Dotenv I/O

Filesystem mutations (mkdir, write, copy files or trees, chmod, remove, glob) and TOML/dotenv reads and writes are wrapped so they participate in dry-run and present a consistent interface. Copy helpers preserve file metadata; tree copies preserve descendant links rather than dereferencing them and refuse linked roots or destinations, so install-like flows do not need package-local `shutil` calls. TOML handling uses round-trip parsing so that updating a single key preserves the rest of a config file. Dotenv helpers upsert individual `.env` lines.

## Cleanup

The cleanup helper runs resource release without replacing an exception already in flight: cleanup failures remain visible on clean exits and are attached to program, cancellation, or interrupt failures. The AgL interpreter and its `exec`/`repl` command hosts use it when closing agent sessions.

## Dry Run

Dry-run is a global, cross-cutting mode set from the `--dry-run` CLI flag. The primitives consult it: when enabled, process and filesystem operations print the action they *would* take instead of performing it. Because the check lives in the primitives, every command inherits dry-run support without implementing it individually.

## Generic Utilities

`util/` is a dependency-free leaf: pure algorithms and string helpers with zero `agm` imports, deliberately usable from any layer without creating a cycle. It provides generic graph algorithms (Tarjan strongly-connected components, Kahn topological sort, and a deterministic breadth-first search for the nearest node satisfying a predicate), used by AgL module loading and the program-level passes for deterministic dependency ordering ([agl/modules.md](agl/modules.md)) and by the AgL type table to name the culprit declaration in whole-type diagnostics; universal-newline normalization shared by the AgL lexer and runtime diagnostics so both index source text identically; and the shared AgL identifier grammar and `%{name}` interpolation parser.

## Code Entry Points

- `src/agm/core/process.py` — foreground/capture execution, success requirements, process-group termination.
- `src/agm/core/env.py` — environment cloning/resolution, env-file sourcing, shell-name validation, installation prefix.
- `src/agm/core/fs.py` — dry-run-aware filesystem operations.
- `src/agm/core/path.py` — CLI path resolution, user-facing path display, and the shared safe-relative-path predicate behind archive entries, `RECORD` paths, and AgL resource paths.
- `src/agm/core/toml.py` and `src/agm/core/dotenv.py` — round-trip TOML and dotenv read/write helpers.
- `src/agm/core/cleanup.py` — primary-error-preserving lifecycle cleanup.
- `src/agm/core/dry_run.py` — global dry-run state and planned-command printing.
- `src/agm/core/log.py` — logging setup and JSONL append support. AgL trace paths use `.jsonl`; ordinary command text logs retain `.log`.
- `src/agm/util/graph.py` — generic Tarjan SCC, Kahn toposort, and nearest-hit BFS; `src/agm/util/text.py` — newline normalization; `src/agm/util/ident.py` — AgL identifier grammar; `src/agm/util/interp.py` — `%{name}` template splitting into literal/hole segments and strict, lenient, and unresolved-reporting rendering. All are pure and `agm`-import-free.
