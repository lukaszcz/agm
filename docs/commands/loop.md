# Loop automation

| Command | Description |
|---|---|
| `agm loop [--runner COMMAND] [--selector COMMAND\|--no-selector] [--tasks-dir DIR] [--no-log\|--log-file PATH] [--prompt TEXT\|--prompt-file PATH] [--selector-prompt TEXT\|--selector-prompt-file PATH] [--extra-prompt TEXT\|--extra-prompt-file PATH] [--extra-selector-prompt TEXT\|--extra-selector-prompt-file PATH] [--timeout DURATION] CMD [RUNNER_ARGS...]` | Shorthand for `agm loop run` when `CMD` is not a built-in subcommand |
| `agm loop run [--runner COMMAND] [--selector COMMAND\|--no-selector] [--tasks-dir DIR] [--no-log\|--log-file PATH] [--prompt TEXT\|--prompt-file PATH] [--selector-prompt TEXT\|--selector-prompt-file PATH] [--extra-prompt TEXT\|--extra-prompt-file PATH] [--extra-selector-prompt TEXT\|--extra-selector-prompt-file PATH] [--timeout DURATION] [CMD [RUNNER_ARGS...]]` | Run the loop until completion |
| `agm loop step [--runner COMMAND] [--selector COMMAND\|--no-selector] [--tasks-dir DIR] [--no-log\|--log-file PATH] [--prompt TEXT\|--prompt-file PATH] [--selector-prompt TEXT\|--selector-prompt-file PATH] [--extra-prompt TEXT\|--extra-prompt-file PATH] [--extra-selector-prompt TEXT\|--extra-selector-prompt-file PATH] [--timeout DURATION] CMD [RUNNER_ARGS...]` | Perform a single loop iteration |
| `agm loop select [--runner COMMAND] [--selector COMMAND\|--no-selector] [--tasks-dir DIR] [--prompt TEXT\|--prompt-file PATH] [--selector-prompt TEXT\|--selector-prompt-file PATH] [--extra-prompt TEXT\|--extra-prompt-file PATH] [--extra-selector-prompt TEXT\|--extra-selector-prompt-file PATH] [--timeout DURATION] [CMD [RUNNER_ARGS...]]` | Run the progress-update prompt once |

Loop config is loaded from merged `config.toml` files:

- `[loop]` defines default `runner`, `selector`, `no_selector`, `tasks_dir`, `timeout`, `prompt`/`prompt_file`, `selector_prompt`/`selector_prompt_file`, `extra_prompt`/`extra_prompt_file`, and `extra_selector_prompt`/`extra_selector_prompt_file`
- `[loop.<command>]` overrides the base loop config for a specific prompt command
- `agm loop CMD` is shorthand for `agm loop run CMD` when `CMD` is not a built-in subcommand, and selects `[loop.CMD]` overrides
- CLI flags (`--runner`, `--selector`, `--no-selector`, `--tasks-dir`, `--prompt`, `--prompt-file`, `--selector-prompt`, `--selector-prompt-file`, `--extra-prompt`, `--extra-prompt-file`, `--extra-selector-prompt`, `--extra-selector-prompt-file`, `--timeout`) override config values
- `RUNNER_ARGS` are appended to the final runner command after AGM resolves `--runner`, config, or the built-in default
- bare `agm loop` prints help text

Prompt options:

- `--prompt TEXT` / `--prompt-file PATH`: override the default runner prompt (task file in selector mode, `loop.md` in no-selector mode). Mutually exclusive.
- `--selector-prompt TEXT` / `--selector-prompt-file PATH`: override the default `select.md` selector prompt. Mutually exclusive.
- `--extra-prompt TEXT` / `--extra-prompt-file PATH`: append extra content to the runner prompt, after the primary prompt. Mutually exclusive.
- `--extra-selector-prompt TEXT` / `--extra-selector-prompt-file PATH`: append extra content to the selector prompt, after the primary selector prompt. Mutually exclusive.

Prompt preprocessing:

- before a prompt file is passed to the runner or selector, AGM expands environment variable references in the prompt content using `$VAR` or `${VAR}` syntax
- unrecognized variables are left unchanged
- when expansions modify the content, AGM writes the expanded text to a temporary file; otherwise the original file path is used
- beyond the process environment, AGM provides:
  - `TASKS_DIR` — the resolved tasks directory path
  - `TASK_FILE` — the selected task file path (selector mode; set in the runner process environment at runtime)

Prompt file path:

- AGM passes the resolved prompt file path to the runner/selector command
- by default it is appended as `@<path>` to the command
- use `%%` or `%{PROMPT_FILE}` in the command to insert the path at a specific position — when either placeholder is present, it is replaced with the path and no `@<path>` suffix is appended

Timeout:

- `--timeout DURATION` sets an idle timeout that kills the runner process tree when no output is received for the given duration
- accepts seconds (plain number or `Ns`), minutes (`Nm`), or hours (`Nh`)
- disabled by default
- also configurable via `[loop] timeout` in `config.toml`

Selector mode (default):

- AGM runs the selector with `@select.md`
- if the selector returns `COMPLETE` after whitespace is removed, AGM stops
- otherwise the selector output is treated as the next task path and AGM runs the runner with that task file
- when no explicit selector command is configured, the runner command is used for the progress update

No-selector mode (`--no-selector` / `no_selector = true`):

- AGM appends the loop prompt to the runner command
- stops when the runner response is `COMPLETE` after whitespace is removed

Subcommands:

- `agm loop step` performs a single loop iteration using the same runner, selector, and logging behavior as `agm loop run`
- `agm loop select` runs `select.md` once using the resolved selector, or the resolved runner when no selector is configured — it requires selector mode; `--no-selector` is an error for `loop select`

Logging:

- by default AGM writes `loop-YYYYMMDD-HHMMSS.log` in the current directory
- `--log-file PATH` writes to a specific file
- `--no-log` disables file logging
