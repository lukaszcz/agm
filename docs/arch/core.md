# Core Primitives

Two foundation packages sit beneath everything else and serve both halves of AGM. `core/` holds the OS-facing building blocks — process execution, environment handling, filesystem and TOML/dotenv I/O, outbound HTTP, logging, lifecycle cleanup, and the dry-run facility; AgL's host runtime runs shell commands and agents, writes files, makes HTTP requests, and emits trace logs through these same primitives. `util/` holds pure, stdlib-only helpers that import nothing from `agm`.

## Process Execution

Foreground and captured subprocess work goes through one process module. It distinguishes terminal-inheriting from captured runs, offers require-success variants, and manages process groups so an interruption tears down descendants. Startup defers Python termination handlers until it owns the child and its reader threads, so early interrupts run the same cleanup as interrupts while waiting without changing the child’s signal mask. A caller may register a cleanup command for a resource it owns but does not contain (`agm run` registers the stop of its transient systemd scope); while one is registered, SIGTERM and SIGHUP are delivered as `KeyboardInterrupt` so the cleanup runs before the group is killed. The persistent Pi RPC session in `agent/session/rpc.py` is the one deliberate exception that owns its own streaming process, because that process must outlive a single capture call.

## Environment Handling

The environment module clones the ambient environment, resolves variable references, sources bash env files in one shell to capture their effect, validates shell-safe names, and locates the AGM installation prefix from the path AGM was invoked through. Environments are passed explicitly as dictionaries, so each command controls exactly what its subprocesses see. Dotenv interpolation uses that explicit environment and preceding assignments across file layers without changing the process environment.

## Filesystem, TOML, and Dotenv I/O

Filesystem mutations and TOML/dotenv reads and writes are wrapped so they participate in dry-run and share one interface. TOML uses round-trip parsing so updating one key preserves the rest of a file; dotenv helpers parse complete assignments and publish updates atomically; tree copies preserve links instead of dereferencing them. Atomic writes go through one chunked byte-writer (temporary sibling, then replace) shared by text writes and any other streamed destination, such as an HTTP download.

## HTTP Transport

One module wraps `requests` for outbound HTTP: it opens a pooled session, builds and streams a request through it, classifies transport failures (URL, connection, timeout, TLS, redirect, and unsendable requests) into a small typed error independent of `requests`' own exception shape, and applies a strict declared-charset-else-UTF-8 decoding rule. The session never persists cookies or consults `~/.netrc`, and caller-supplied cookies are bound to the request's host so they never leak to a cross-host redirect target. A response saved to disk streams through the same atomic temp-and-replace helper filesystem writes use. It owns the only import of `requests` besides the package fetcher, so higher layers never see `requests` types directly.

## Dry Run and Cleanup

Dry-run is a global mode set from `--dry-run`. Because the process and filesystem primitives consult it, every command inherits dry-run support without implementing it. The cleanup helper releases resources without masking an exception already in flight; the AgL interpreter and the `exec`/`repl` hosts use it when closing agent sessions.

## Generic Utilities

`util/` is a dependency-free leaf usable from any layer: graph algorithms (Tarjan SCC, Kahn toposort, nearest-hit BFS) used by AgL module loading and type-table analyses; newline normalization shared by the lexer and diagnostics; the AgL identifier grammar; the `%{name}` interpolation parser shared by prompts, runner commands, config paths, and AgL; a `ContextVar` scoping guard; and an overlap-safe raise of the process-global recursion limit, used by the AgL interpreter, whose runs may overlap on threads.

## Code Entry Points

- `src/agm/core/process.py` — foreground/capture execution, success requirements, process-group termination, cleanup-command registration.
- `src/agm/core/env.py` — environment cloning/resolution, env-file sourcing, installation prefix.
- `src/agm/core/fs.py` — dry-run-aware filesystem operations; `src/agm/core/path.py` — path resolution, display, and the safe-relative-path predicate shared by archives, `RECORD` files, and AgL resources.
- `src/agm/core/toml.py`, `src/agm/core/dotenv.py` — round-trip TOML and dotenv helpers.
- `src/agm/core/cleanup.py` — primary-error-preserving cleanup; `src/agm/core/dry_run.py` — global dry-run state; `src/agm/core/log.py` — logging and JSONL append.
- `src/agm/core/http.py` — the `requests`-backed HTTP transport seam: session, request/response streaming, failure classification, charset decoding.
- `src/agm/util/graph.py`, `text.py`, `ident.py`, `interp.py`, `scoping.py`, `recursion.py` — the pure helpers.
