# Sandboxing

| Command | Description |
|---|---|
| `agm run [--no-sandbox] [--no-patch] [--memory LIMIT] [--swap LIMIT] [--no-memory-limit] [--no-swap-limit] [-f\|--file SETTINGS] COMMAND [ARGS...]` | Run a command directly or in an Anthropic Sandbox Runtime container |

`agm run` config lookup:

1. `<install-prefix>/.agm/config.toml` when present, otherwise `$HOME/.agm/config.toml`
2. `<project-config-dir>/config.toml`
3. `./.agm/config.toml`

`agm run` config keys:

- `[run].memory`: default `MemoryMax` for sandboxed runs
- `[run].swap`: default `MemorySwapMax` for sandboxed runs
- `[run.<command>].memory`: per-command `MemoryMax` override
- `[run.<command>].swap`: per-command `MemorySwapMax` override
- `[run.<command>].alias`: replace the invoked command name before execution

`agm run` options:

- `--no-sandbox`: run the command directly without `srt`; skips sandbox settings discovery and patching
- `-f`, `--file SETTINGS`: use one settings file directly instead of discovered settings
- `--memory LIMIT`: set `MemoryMax=LIMIT` in the delegated `systemd-run --user --scope`; the bootstrap exports `SANDBOX_CGROUP` and enables the memory controller for descendant cgroups; defaults to `32G` in sandbox mode; `0` means a zero memory limit; `unlimited` means no memory cap
- `--swap LIMIT`: set `MemorySwapMax=LIMIT` in the delegated scope; defaults to `0` in sandbox mode; `unlimited` means no swap cap
- `--no-memory-limit`: do not set `MemoryMax`
- `--no-swap-limit`: do not set `MemorySwapMax`
- `--no-patch`: do not append project notes, deps, and repo `.git` paths to `filesystem.allowWrite`

Sandbox settings resolution:

- for each config directory, AGM prefers `<command>.json`
- if that file does not exist there, AGM tries the aliased command name's settings file
- if neither exists, AGM falls back to `default.json`
- AGM merges matching files in this order:
  1. `$HOME/.agm/sandbox/`
  2. the project sandbox config directory
  3. `./.sandbox/`
- later files override earlier ones
- `network` and `filesystem` are merged by key
- `ignoreViolations` replaces the earlier value
- `enabled` and `enableWeakerNestedSandbox` override when set
