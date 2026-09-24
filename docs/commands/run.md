# Sandboxing

| Command | Description |
|---|---|
| `agm run [--no-sandbox] [--no-patch] [--pty\|--no-pty] [--memory LIMIT] [--swap LIMIT] [--no-memory-limit] [--no-swap-limit] [-f\|--file SETTINGS] COMMAND [ARGS...]` | Run a command directly or in an Anthropic Sandbox Runtime container |

Arguments after `COMMAND` are passed as separate values, including quoted strings with spaces (for example, `agm run cmd "AA BB CC"`). In sandbox mode AGM shell-quotes them before passing them to SRT, which runs a shell command string; without sandboxing they are passed directly to the command.

`agm run` config lookup merges all matching layers in order (later layers override earlier ones):

1. `<install-prefix>/.agm/config.toml` when AGM is installed with one
2. the selected AGM home's `config.toml` when distinct (`$AGM_HOME`, otherwise the
   populated installation home, otherwise `$HOME/.agm`)
3. `<project-config-dir>/config.toml`
4. `./.agm/config.toml`

`agm run` config keys:

- `[run].memory`: default `MemoryMax` for sandboxed runs
- `[run].swap`: default `MemorySwapMax` for sandboxed runs
- `[run].pty`: whether interactive runs receive a controlling pseudo-terminal; defaults to `true`
- `[run.<command>].memory`: per-command `MemoryMax` override
- `[run.<command>].swap`: per-command `MemorySwapMax` override
- `[run.<command>].pty`: per-command pseudo-terminal override
- `[run.<command>].alias`: replace the invoked command name before execution

`[run].memory`/`.swap` and `[run.<command>].memory`/`.swap` are shared with AgL: an agent call's `sandbox = Sandbox(...)` or `exec`'s `sandbox = Some(Sandbox(...))` (see [`sandbox`](../agl/reference/agent-calls.md#sandbox) and [Spawn parameters](../agl/reference/shell-execution.md)) resolves its limits from the same `[run.<name>]` → `[run]` → built-in-floor chain, keyed by the real command name (the agent's executable, or `exec`'s first shell word) — never a spec-name table, never `sh`. `[run.<command>].alias` and `.pty` are read only by `agm run` itself; an AgL sandboxed call never remaps its command through an alias and never allocates a pseudo-terminal.

`agm run` options:

- `--no-sandbox`: run directly without `srt`, skipping sandbox settings discovery and patching
- `--pty` / `--no-pty`: enable or disable the controlling pseudo-terminal; enabled by default and allocated only when stdin and stdout are terminals
- `-f`, `--file SETTINGS`: use one settings file directly instead of discovered settings
- `--memory LIMIT`: set `MemoryMax=LIMIT` in the delegated `systemd-run --user --scope` (the bootstrap exports `SANDBOX_CGROUP` and enables the memory controller for descendant cgroups); defaults to `32G` in sandbox mode; `0` means a zero memory limit; `unlimited` means no memory cap
- `--swap LIMIT`: set `MemorySwapMax=LIMIT` in the delegated scope; defaults to `0` in sandbox mode; `unlimited` means no swap cap
- `LIMIT` grammar (both flags, whitespace-trimmed): `infinity` or `unlimited`, a percentage (`N%`), or whitespace-separated `<number>[KMGTPE][B]` groups with case-sensitive suffixes
- `--no-memory-limit`: do not set `MemoryMax`
- `--no-swap-limit`: do not set `MemorySwapMax`
- `--no-patch`: do not append project notes, deps, and repo `.git` paths to `filesystem.allowWrite`

Sandbox settings resolution:

- per config directory: prefer `<command>.json`, else the aliased command's settings file, else `default.json`
- merge matching files in this order (later over earlier):
  1. `<AGM-home>/sandbox/`, using the same runtime-home selection described above
  2. the project sandbox config directory
  3. `./.sandbox/`
- `network` and `filesystem` are merged by key; their list-valued keys are appended and deduplicated in precedence order
- later `network.deniedDomains` entries remove matching earlier `network.allowedDomains` entries; later `filesystem.denyRead` and `filesystem.denyWrite` entries remove matching earlier `filesystem.allowRead` and `filesystem.allowWrite` entries
- `ignoreViolations` replaces the earlier value; `enabled` and `enableWeakerNestedSandbox` override when set

An AgL agent call or sandboxed `exec` resolves settings through this same discovery and merge chain; it never has an alias to fall back to, so an unmatched name goes straight to `default.json`.

The bundled `pi.json` profile sets `network.allowAllUnixSockets` so Pi extensions can create local IPC sockets. On Linux, SRT's seccomp filter cannot allow Unix sockets by path, so this permission is necessarily all-or-nothing; filesystem policy still controls which socket paths Pi can create or access.
