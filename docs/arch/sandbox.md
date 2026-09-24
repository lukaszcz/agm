# Sandboxed Execution

`agm run` executes a command inside a sandbox with an explicitly configured filesystem and network policy and optional memory limits, so agent-driven and untrusted commands run with least privilege by default while staying easy to configure per project and per command.

## Sandbox Runtime

Isolation is delegated to SRT, the external sandbox-runtime tool; AGM implements none itself. `agm run` resolves a merged settings file and invokes SRT with the target command's arguments shell-quoted, because SRT joins them into a shell command string. Direct execution bypasses both SRT and shell quoting for commands that need full access.

## Settings Resolution

Sandbox settings are discovered and merged across the same install/home/project/workspace scopes as general configuration. A per-command settings file is selected by command name, falling back to a default. Policy sections merge by key; list-valued keys append with duplicates removed, and later deny lists subtract from earlier allow lists. Before execution AGM patches in write access to the project-internal git directories the command legitimately needs.

## Resource Limits

Memory and swap limits are enforced through `systemd-run`, which places the sandboxed process in a transient scope; the process primitive registers the scope's stop as a cleanup command so an interrupted run does not leak it ([core.md](core.md)).

## Code Entry Points

- `src/agm/commands/run.py` — the `agm run` command: alias remapping, limit flags, sandbox invocation.
- `src/agm/sandbox/srt.py` — SRT settings resolution, the merge chain, project write-path patching, artifact cleanup.
- `src/agm/config/sandbox/` — sandbox settings discovery and merging.
