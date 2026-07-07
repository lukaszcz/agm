# Sandboxed Execution

`agm run` executes a command inside a sandbox with an explicitly configured filesystem and network policy, and optional memory limits. It exists so that agent-driven and untrusted commands run with least privilege by default, while remaining easy to configure per project and per command.

## Sandbox Runtime

Sandboxing is delegated to SRT (the external sandbox-runtime tool); AGM does not implement isolation itself. `agm run` resolves a merged settings file and invokes SRT with it, passing the target command through. The sandbox can be bypassed explicitly for commands that need full access.

## Settings Resolution

Sandbox settings are discovered and merged across the same scopes as general configuration (install, home, project, workspace). A per-command settings file is selected by command name, falling back to a default when no command-specific file exists. Network and filesystem policy sections merge by key; list-valued policy keys are appended with duplicates removed, and later deny lists subtract from earlier allow lists. Before execution, AGM patches the merged settings to grant write access to the project-internal git directories the command legitimately needs, then runs SRT against the resulting settings. Configuration details live in [config.md](config.md).

## Resource Limits

Optional memory and swap limits are enforced by delegating to `systemd-run`, which places the sandboxed process in a scope with the configured limits. A cgroup bootstrap is exposed to the sandboxed environment so the limits apply to the whole process tree.

## Code Entry Points

- `src/agm/commands/run.py` — the `agm run` command: alias/command remapping, limit flags, sandbox invocation.
- `src/agm/sandbox/srt.py` — SRT settings resolution, the settings merge chain, project write-path patching, and artifact cleanup.
- `src/agm/config/sandbox/` — sandbox settings discovery and merging.
- `src/agm/core/process.py` — the process execution used to launch SRT and `systemd-run`.
