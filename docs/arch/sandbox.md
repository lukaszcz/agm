# Sandboxed Execution

A command runs inside a sandbox with an explicit filesystem/network policy and optional memory limits, giving agent-driven and untrusted commands least privilege by default. `agm run` is the CLI entry point; the preparation library below is a reusable in-process API for any caller (agent invocation, `exec`) needing a sandboxed command without a CLI subprocess round trip.

## Preparation Library

`sandbox/request.py` is the leaf: the plain data types and `cleanup_artifacts()`, importing nothing else from `agm.sandbox` so the rest of the package layers above it with no cycle. `SandboxLimits` (`LimitSpec`-valued `memory`/`swap`, an optional `settings_file`, `patch`) is the shape statable without naming a command — the shape an AgL `Sandbox` record decodes directly to; `SandboxSpec` extends it with the `profile_name`, bound by the caller that knows the command. `PreparedSandboxCommand.close()` removes the run's temp settings files and empty tracked artifacts; idempotent, never gated by dry-run.

`prepare.py::prepare()` turns a `SandboxRequest` into a `PreparedSandboxCommand`, composing the `systemd-run` resource-limit prefix, a backend's wrapper argv, and an optional PTY wrapper. `resolve_limits()` is the one place memory/swap limits resolve against `[run.<name>]` → `[run]` → the built-in floor and validate against systemd's grammar; both `prepare()` and a display-only caller (`agm run --dry-run`) call it, so the policy exists once. The library never prints or exits: failures raise `SandboxUnavailableError` or `SandboxSettingsError`, each carrying structured fields a caller formats into its own message. `SandboxContext` (home, proj_dir, cwd, run config) bundles the ambient inputs a non-CLI caller — the agent runner, the session host (`agents.md`) — needs to build a `SandboxRequest`; its `prepare()` method builds the request and calls `prepare()` in one step, so a caller never hand-maps the fields itself. `SandboxRun` pairs `SandboxLimits` (its `limits` field) with the `SandboxContext` that prepares them; the profile name binds later, from the real (post-interpolation) argv, never here. `lazy_sandbox_context()` defers a `ConfigContext` → `SandboxContext` build (and its config I/O) until first needed; `sandbox_run_for()` binds an optional `SandboxLimits` to one, returning `None` untouched when there is nothing to sandbox. Both are the shared seam every caller uses to turn a decoded call-site sandbox value into a `SandboxRun`, rather than each rebuilding the context itself.

## Backends

`backend.py` defines the `SandboxBackend` protocol (availability, settings resolution, argv wrapping, env adjustment) and a `default_backend()` registry seam for future methods. `srt.py::SrtBackend` is the shipped implementation, delegating isolation to the external `srt` tool: settings resolution/merging, git write-access patching, bwrap-artifact cleanup tracking, and a Node fetch-proxy env default.

`profile.py::profile_name()` derives a sandbox profile name from an executable path, selecting both the per-command settings file and the `[run.<name>]` limit overrides; a `run`-only alias never affects it. The config layer takes a profile name only pre-normalized this way, never deriving one itself.

## Settings Resolution

Settings are discovered and merged across the same install/home/project/workspace scopes as general configuration. A per-command file is selected by profile name, falling back to a default; policy sections merge by key, list-valued keys appending (duplicates removed), later deny lists subtracting from earlier allow lists.

## `agm run`

`commands/run.py` is a thin client: it maps CLI flags and config to a `SandboxRequest`/`SandboxSpec` (alias remapping happens here only, never for agent or `exec` argv), calls `prepare()`, and runs the result in the foreground or, under `--dry-run`, prints the same detail lines. It holds no limit, PTY, or settings logic of its own. Interactive commands run through AGM's PTY relay by default; sandbox and PTY can each be bypassed explicitly.

## Code Entry Points

- `src/agm/sandbox/request.py`, `prepare.py` — leaf data types and `cleanup_artifacts()`; `prepare()`, `resolve_limits()`, dry-run printing.
- `src/agm/sandbox/backend.py`, `profile.py`, `srt.py` — the `SandboxBackend` protocol and errors; profile-name derivation; the SRT backend.
- `src/agm/sandbox/pty.py` — controlling-terminal allocation and terminal I/O relay.
- `src/agm/commands/run.py`, `src/agm/config/sandbox/` — the `agm run` command; sandbox settings discovery and merging.
