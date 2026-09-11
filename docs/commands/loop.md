# Loop automation

| Command | Description |
|---|---|
| `agm loop [--runner COMMAND] [--selector COMMAND\|--no-selector] [--tasks-dir DIR] [--no-log\|--log-file PATH] [--prompt TEXT\|--prompt-file PATH] [--selector-prompt TEXT\|--selector-prompt-file PATH] [--extra-prompt TEXT\|--extra-prompt-file PATH] [--extra-selector-prompt TEXT\|--extra-selector-prompt-file PATH] [--timeout DURATION] CMD [RUNNER_ARGS...]` | Shorthand for `agm loop run` when `CMD` is not a built-in subcommand |
| `agm loop run [--runner COMMAND] [--selector COMMAND\|--no-selector] [--tasks-dir DIR] [--no-log\|--log-file PATH] [--prompt TEXT\|--prompt-file PATH] [--selector-prompt TEXT\|--selector-prompt-file PATH] [--extra-prompt TEXT\|--extra-prompt-file PATH] [--extra-selector-prompt TEXT\|--extra-selector-prompt-file PATH] [--timeout DURATION] [CMD [RUNNER_ARGS...]]` | Run the loop until completion |
| `agm loop step [--runner COMMAND] [--selector COMMAND\|--no-selector] [--tasks-dir DIR] [--no-log\|--log-file PATH] [--prompt TEXT\|--prompt-file PATH] [--selector-prompt TEXT\|--selector-prompt-file PATH] [--extra-prompt TEXT\|--extra-prompt-file PATH] [--extra-selector-prompt TEXT\|--extra-selector-prompt-file PATH] [--timeout DURATION] CMD [RUNNER_ARGS...]` | Perform a single loop iteration |
| `agm loop select [--runner COMMAND] [--selector COMMAND\|--no-selector] [--tasks-dir DIR] [--prompt TEXT\|--prompt-file PATH] [--selector-prompt TEXT\|--selector-prompt-file PATH] [--extra-prompt TEXT\|--extra-prompt-file PATH] [--extra-selector-prompt TEXT\|--extra-selector-prompt-file PATH] [--timeout DURATION] [CMD [RUNNER_ARGS...]]` | Run the progress-update prompt once |

Loop config loads from merged `config.toml` files:

- `[loop]`: default `runner`, `selector`, `no_selector`, `tasks_dir`, `timeout`, `prompt`/`prompt_file`, `selector_prompt`/`selector_prompt_file`, `extra_prompt`/`extra_prompt_file`, `extra_selector_prompt`/`extra_selector_prompt_file`
- `[loop.<command>]`: overrides the base loop config for a specific prompt command
- `agm loop CMD` is shorthand for `agm loop run CMD` when `CMD` is not a built-in subcommand, and selects `[loop.CMD]` overrides
- CLI flags (`--runner`, `--selector`, `--no-selector`, `--tasks-dir`, `--prompt`, `--prompt-file`, `--selector-prompt`, `--selector-prompt-file`, `--extra-prompt`, `--extra-prompt-file`, `--extra-selector-prompt`, `--extra-selector-prompt-file`, `--timeout`) override config values
- `RUNNER_ARGS` are appended to the final runner command after AGM resolves `--runner`, config, or the built-in default
- bare `agm loop` prints help text

Prompt options (each TEXT/FILE pair is mutually exclusive):

- `--prompt TEXT` / `--prompt-file PATH`: override the default runner prompt — `prompts/implement.md` (selector mode; preprocessed after task selection with `%{TASK_FILE}`) or `loop.md` (no-selector mode)
- `--selector-prompt TEXT` / `--selector-prompt-file PATH`: override the default `select.md` selector prompt
- `--extra-prompt TEXT` / `--extra-prompt-file PATH`: append extra content to the runner prompt, after the primary prompt
- `--extra-selector-prompt TEXT` / `--extra-selector-prompt-file PATH`: append extra content to the selector prompt, after the primary selector prompt

## Prompt interpolation

AGM expands `%{name}` holes in prompt text per [Prompt interpolation](agents.md#prompt-interpolation) (strict; `\%{` escape), from the process environment overlaid with these variables, which win on conflict:

- `TASKS_DIR` — the resolved tasks directory, in every loop prompt
- `TASK_FILE` — the selected task path, only when expanding the runner prompt after selection in selector mode; selector and no-selector prompts do not receive it

When expansion changes a file's content, AGM writes a temporary prompt file; otherwise it reuses the original file.

In selector mode, dry-run does not choose a task or invoke either command. Its output marks `TASK_FILE` as unavailable and identifies the runner prompt as reprocessed after task selection.

## Prompt file path

Same prompt-file-path mechanics as [Runner command interpolation](agents.md#runner-command-interpolation) in agents.md, applied to both the runner and selector command: AGM passes the resolved prompt file path, appending `@<path>` by default; `%%` or `%{PROMPT_FILE}` places it explicitly instead (verbatim, without recursive interpolation).

Timeout:

- `--timeout DURATION`: idle timeout — kills the current agent process tree when no output is received for the duration; fails only that invocation, not the loop or command. Accepts seconds (plain number or `Ns`), minutes (`Nm`), or hours (`Nh`). Disabled by default; also configurable via `[loop] timeout`.
- When logging is enabled, all loop diagnostics — including setup failures, agent failures, timeouts, interruption, and completion — are written to stderr and the loop log.

Selector mode (default):

- AGM runs the selector with `@select.md`; when no explicit selector command is configured, the runner command is used for the progress update
- if the selector returns `COMPLETE` after whitespace is removed, AGM stops
- otherwise the selector output is treated as the next task path: AGM preprocesses `prompts/implement.md` with `%{TASK_FILE}` set to that path, then runs the runner with the resulting prompt

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
