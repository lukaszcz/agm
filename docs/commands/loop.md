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

- `--prompt TEXT` / `--prompt-file PATH`: override the default runner prompt (`prompts/implement.md`, preprocessed after task selection with `%{TASK_FILE}`, in selector mode; `loop.md` in no-selector mode). Mutually exclusive.
- `--selector-prompt TEXT` / `--selector-prompt-file PATH`: override the default `select.md` selector prompt. Mutually exclusive.
- `--extra-prompt TEXT` / `--extra-prompt-file PATH`: append extra content to the runner prompt, after the primary prompt. Mutually exclusive.
- `--extra-selector-prompt TEXT` / `--extra-selector-prompt-file PATH`: append extra content to the selector prompt, after the primary selector prompt. Mutually exclusive.

## Prompt interpolation

Before AGM passes a prompt to a runner or selector, it expands `%{name}` holes. A name is an AgL identifier, so names such as `%{log-file}` work. `\%{` writes a literal `%{`, and a bare `%` is literal. `$VAR` and `${VAR}` are plain literal text; the old shell-style interpolation syntax is not supported.

Expansion is strict: an unknown variable, invalid hole name, or unterminated `%{` is an error, not passthrough. Variables come from the full process environment, overlaid with AGM's variables (which win on conflicts):

- `TASKS_DIR` — the resolved tasks directory, in every loop prompt.
- `TASK_FILE` — the selected task path, only when expanding the runner prompt after selection in selector mode. Selector and no-selector prompts do not receive it.

When expansion changes a file's content, AGM writes a temporary prompt file; otherwise it reuses the original file.

## Prompt file path

AGM passes the resolved prompt file path to the runner or selector command. By default it appends `@<path>`. Use `%%` or `%{PROMPT_FILE}` in either command to place the path at a specific position; either prevents the suffix and inserts the path verbatim without recursively interpolating it. Command arguments use strict `%{name}` interpolation from the process environment overlaid with `PROMPT_FILE`, which wins on conflicts. Names are AgL identifiers (for example, `%{log-file}`). `%%` is an alias for `%{PROMPT_FILE}`; `\%{` writes a literal `%{`; a bare `%`, `$VAR`, and `${VAR}` are literal text, and shell-style interpolation is unsupported. Unknown, invalid, or unterminated holes are errors. Command strings are shlex-split before interpolation, so quote or otherwise protect `\%{` so its backslash reaches the argv element.

Timeout:

- `--timeout DURATION` sets an idle timeout that kills the runner process tree when no output is received for the given duration
- accepts seconds (plain number or `Ns`), minutes (`Nm`), or hours (`Nh`)
- disabled by default
- also configurable via `[loop] timeout` in `config.toml`

Selector mode (default):

- AGM runs the selector with `@select.md`
- if the selector returns `COMPLETE` after whitespace is removed, AGM stops
- otherwise the selector output is treated as the next task path; AGM preprocesses `prompts/implement.md` with `%{TASK_FILE}` set to that path, then runs the runner with the resulting prompt
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
