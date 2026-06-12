# AGM commands reference

## Global usage

```text
agm <command> [options] [args]
```

Global options:

- `--dry-run`
- `--install-completion`
- `--show-completion`

## Project session and lifecycle commands

| Command | Description |
|---|---|
| `agm open [-d\|--detach] [-n\|--num-panes PANES] [-p\|--parent PARENT] TARGET` | Open the main checkout or a branch worktree, creating or checking it out when needed |
| `agm close [-f\|--force] [-D] BRANCH` | Remove a branch worktree and close its tmux session |
| `agm init [--embedded \| --workspace] [--no-git-init \| --no-config-git \| --no-notes-git]` | Initialize the current directory without cloning a repo |
| `agm init [--embedded \| --workspace] [--no-git-init \| --no-config-git \| --no-notes-git] PROJECT_NAME` | Initialize a child project directory without cloning a repo |
| `agm init [--embedded \| --workspace] [-b\|--branch BRANCH] [--no-git-init \| --no-config-git \| --no-notes-git] [PROJECT_NAME] REPO_URL` | Initialize the current directory or named child directory and clone a repo |
| `agm init --clone [--embedded \| --workspace] [-b\|--branch BRANCH] [--no-git-init \| --no-config-git \| --no-notes-git] REPO_URL` | Initialize a URL-derived child project directory and clone a repo |
| `agm fetch` | Fetch the main repo and checked-out dependencies, then create missing tracking branches |
| `agm pull` | Run `agm fetch`, then run `git merge` in every dependency, main repo, and worktree checkout |
| `agm list [-v\|--verbose]` | List all open worktrees |
| `agm setup` | Run setup scripts for the current checkout |

`agm open` behavior:

- `repo` opens the main checkout session
- the branch currently checked out in the main checkout also opens the main checkout session
- an existing worktree target opens its tmux session
- an existing branch without a worktree is checked out into a worktree and then opened
- a missing branch is created from `--parent` or the main checkout's current branch and then opened

`agm open` options:

- `-d`, `--detach`: create the tmux session without attaching
- `-n`, `--num-panes PANES`: create the session with `PANES` panes
- `-p`, `--parent PARENT`: base a newly created branch worktree on `PARENT`

`agm close` options:

- `-f`, `--force`: force remove the worktree (even with untracked or uncommitted changes) and force delete the branch (`git branch -D`). Implies `-D`.
- `-D`: force delete the branch (`git branch -D`) instead of safe delete (`git branch -d`). The worktree is only removed if the branch deletion would succeed.

`agm close` notes:

- closes only branch worktrees
- `repo` and the main checkout branch cannot be removed with `agm close`

`agm pull` notes:

- runs the same fetch and tracking-branch sync as `agm fetch` first
- runs `git merge` in each dependency checkout/worktree, the main repo checkout, and each project worktree
- relies on each checkout's current branch upstream, matching plain `git merge`

`agm list` options:

- `-v`, `--verbose`: show the worktree directory path after each branch name

`agm list` notes:

- the main repo worktree is listed first
- the current worktree is indicated with a leading `*`

`agm init` options:

- `--embedded`: force the embedded layout with AGM data under `.agm/`
- `--workspace`: force the workspace layout with `repo/`, `deps/`, `notes/`, `worktrees/`, and `config/`
- `--clone`: initialize a child directory derived from `REPO_URL` when no `PROJECT_NAME` is provided
- `-b`, `--branch BRANCH`: clone this branch when `REPO_URL` is provided
- `--no-git-init`: do not create git repositories in `config/` and `notes/`
- `--no-config-git`: do not create a git repository in `config/`
- `--no-notes-git`: do not create a git repository in `notes/`

`agm init` layout selection:

- with `REPO_URL`, the default is the workspace layout unless `--embedded` is provided
- without `REPO_URL`, AGM chooses the embedded layout only when the target project directory is a git repo
- otherwise it chooses the workspace layout
- without `PROJECT_NAME`, AGM initializes the current directory
- with `PROJECT_NAME`, AGM initializes a child directory with that name
- with `--clone REPO_URL`, AGM initializes a child directory derived from the URL

## Loop automation

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

## Worktrees and dependencies

| Command | Description |
|---|---|
| `agm worktree new [-d\|--dir DIR] BRANCH` | Create a new branch worktree or check out an existing branch |
| `agm worktree setup` | Run configured setup scripts for the current checkout |
| `agm worktree remove [-f\|--force] BRANCH` | Remove a worktree and delete its local branch |
| `agm worktree rm [-f\|--force] BRANCH` | Alias form of `agm worktree remove` |
| `agm wt new [-d\|--dir DIR] BRANCH` | Alias form of `agm worktree new` |
| `agm wt setup` | Alias form of `agm worktree setup` |
| `agm wt rm [-f\|--force] BRANCH` | Alias form of `agm worktree remove` |
| `agm wt remove [-f\|--force] BRANCH` | Alias form of `agm worktree remove` |
| `agm dep list [-v\|--verbose] [--all]` | List dependency checkouts |
| `agm dep new [-b\|--branch BRANCH] REPO_URL` | Clone a new dependency checkout |
| `agm dep switch [-b\|--branch] DEP BRANCH` | Select or add a dependency checkout |
| `agm dep rm --all DEP` | Remove an entire dependency directory |
| `agm dep rm DEP/NAME_OR_BRANCH \| DEP/repo \| DEP/MAIN_CHECKOUT` | Remove a dependency checkout or worktree |
| `agm dep remove --all DEP` | Alias form of `agm dep rm --all` |
| `agm dep remove DEP/NAME_OR_BRANCH \| DEP/repo \| DEP/MAIN_CHECKOUT` | Alias form of `agm dep rm` |

`agm worktree new` options:

- `-d`, `--dir DIR`: use `agm worktree new --dir DIR BRANCH` to create the worktree under `DIR` instead of the project's default worktrees directory

`agm worktree setup` runs executable setup scripts, in this order:

1. project-level `config/setup.sh`
2. checkout-local `.config/setup.sh`
3. checkout-local `.setup.sh`

`agm worktree remove` options:

- `-f`, `--force`: use `agm worktree remove --force BRANCH` to force removal even when git reports uncommitted or locked state

`agm dep new` options:

- `-b`, `--branch BRANCH`: use `agm dep new --branch BRANCH REPO_URL` to clone the dependency's initial checkout from `BRANCH` instead of the dependency's default branch

`agm dep switch` options:

- `-b`, `--branch`: use `agm dep switch --branch DEP BRANCH` to create `DEP`'s `BRANCH` from the dependency's default branch before adding the new worktree; without this flag, `BRANCH` must already exist

Dependency commands track selected dependency checkout names in config `config.toml` `[deps]` tables. Environment loading turns those entries into dependency path variables, so `[deps].vyper-automation = "feat/app"` provides `VYPER_AUTOMATION=/path/to/proj/deps/vyper-automation/feat/app` before `.env` and `env.sh` are loaded.

`agm dep list` options:

- `-v`, `--verbose`: show the checkout path after each dep/branch
- `--all`: list all dependency checkouts on disk instead of only the current checkout's dependencies

`agm dep rm` targets:

- `DEP/NAME_OR_BRANCH`: remove a dependency checkout by directory name under `deps/DEP/` or by checked-out branch name
- `DEP/repo`: remove the main dependency checkout
- `DEP/MAIN_CHECKOUT`: remove the main dependency checkout by directory name

`agm dep rm` options:

- `--all DEP`: use `agm dep rm --all DEP` to remove the entire dependency directory, including the main checkout and any linked worktrees

## Agent commands

| Command | Description |
|---|---|
| `agm review [COMMAND] [--scope REVIEW_SCOPE] [--aspects REVIEW_ASPECTS] [--extra-aspects REVIEW_ASPECTS] [--runner COMMAND] [--prompt TEXT\|--prompt-file PATH] [--extra-prompt TEXT\|--extra-prompt-file PATH] [--review-file FILE\|auto\|none\|--no-review-file]` | Run the review prompt |
| `agm revise [COMMAND] [--runner COMMAND] [--prompt TEXT\|--prompt-file PATH] [--extra-prompt TEXT\|--extra-prompt-file PATH] REVIEW_FILE` | Run the revision prompt |
| `agm refine [COMMAND] [--max-steps N\|unlimited] [--no-max-steps] [--runner COMMAND] [--reviewer COMMAND] [--reviser COMMAND] [--scope REVIEW_SCOPE] [--aspects REVIEW_ASPECTS] [--review-prompt TEXT\|--review-prompt-file PATH] [--extra-review-prompt TEXT\|--extra-review-prompt-file PATH] [--revise-prompt TEXT\|--revise-prompt-file PATH] [--extra-revise-prompt TEXT\|--extra-revise-prompt-file PATH] [--save-review\|--no-save-review] [--review-file FILE\|auto\|none] [--log-file PATH\|--no-log]` | Run review/revise refinement cycles |

`agm review` runs the review prompt with `REVIEW_SCOPE` and `REVIEW_ASPECTS` available during prompt
preprocessing. The default prompt is `review.md`. Review output is saved to
`.agent-files/review-YYYYMMDD-HHMMSS-microseconds.md` by default. Use `--review-file FILE` to choose
a path, `--review-file none` or `--no-review-file` to disable saving, and `--review-file auto` to
use the default timestamped path. When `COMMAND` is provided, config from `[review.COMMAND]` is
merged over `[review]`.

`agm review` options:

- `--runner COMMAND`: review runner command. When unset, the same default runner as `agm loop` is used.
- `--scope REVIEW_SCOPE`: review scope (default: `changes on current branch`)
- `--aspects REVIEW_ASPECTS`: review aspects (default: `correctness, completeness, maintainability, adherence to AGENTS.md`)
- `--extra-aspects REVIEW_ASPECTS`: additional review aspects appended to the defaults
- `--prompt TEXT` / `--prompt-file PATH`: override the default `review.md` prompt. Mutually exclusive.
- `--extra-prompt TEXT` / `--extra-prompt-file PATH`: append extra content to the review prompt. Mutually exclusive.
- `--review-file FILE|auto|none` / `--no-review-file`: save review output to a file. `auto` uses the default timestamped path, `none` or `--no-review-file` disables saving.

`agm review` config keys in `config.toml`:

- `[review] runner`, `scope`, `aspects`, `extra_aspects`, `prompt`, `prompt_file`, `extra_prompt`, `extra_prompt_file`, `review_file`
- `[review.<command>]` overrides the base review config for a specific command

`agm revise` runs the revision prompt with `REVIEW_FILE` available during prompt preprocessing. The
default prompt is `revise.md`. When `COMMAND` is provided before `REVIEW_FILE`, config from
`[revise.COMMAND]` is merged over `[revise]`.

`agm revise` options:

- `--runner COMMAND`: revision runner command. When unset, the same default runner as `agm loop` is used.
- `--prompt TEXT` / `--prompt-file PATH`: override the default `revise.md` prompt. Mutually exclusive.
- `--extra-prompt TEXT` / `--extra-prompt-file PATH`: append extra content to the revision prompt. Mutually exclusive.

`agm revise` config keys in `config.toml`:

- `[revise] runner`, `prompt`, `prompt_file`, `extra_prompt`, `extra_prompt_file`
- `[revise.<command>]` overrides the base revise config for a specific command

`agm refine` runs review/revise cycles until the revise response is `COMPLETE`, or until the maximum
number of revision attempts is reached. A `CONTINUE` response from revise starts a fresh review;
any other non-`COMPLETE` response retries revise with the same review file. The default maximum
is 20. Review output is saved to the default timestamped review path by default.

When `COMMAND` is provided, config from `[refine.COMMAND]` is merged over `[refine]` and the same
command name is forwarded to review/revise config lookup.

`agm refine` options:

- `--max-steps N|unlimited`: maximum revision attempts (default: 20). Use `unlimited` for no limit.
- `--no-max-steps`: disable the step limit (run until COMPLETE). Mutually exclusive with `--max-steps`.
- `--runner COMMAND`: runner command for both review and revise
- `--reviewer COMMAND`: review runner command. Overrides `--runner` for the review step.
- `--reviser COMMAND`: revision runner command. Overrides `--runner` for the revise step.
- `--scope REVIEW_SCOPE`: review scope
- `--aspects REVIEW_ASPECTS`: review aspects
- `--review-prompt TEXT` / `--review-prompt-file PATH`: override the default review prompt. Mutually exclusive.
- `--extra-review-prompt TEXT` / `--extra-review-prompt-file PATH`: append extra content to the review prompt. Mutually exclusive.
- `--revise-prompt TEXT` / `--revise-prompt-file PATH`: override the default revision prompt. Mutually exclusive.
- `--extra-revise-prompt TEXT` / `--extra-revise-prompt-file PATH`: append extra content to the revision prompt. Mutually exclusive.
- `--save-review` / `--no-save-review`: save or skip saving review output (default: save)
- `--review-file FILE|auto|none`: review output file path, `auto`, or `none`
- `--log-file PATH` / `--no-log`: write command output to a log file or disable logging

`agm refine` config keys in `config.toml`:

- `[refine] max_steps`, `no_max_steps`, `runner`, `reviewer`, `reviser`, `scope`, `aspects`, `review_prompt`, `review_prompt_file`, `extra_review_prompt`, `extra_review_prompt_file`, `revise_prompt`, `revise_prompt_file`, `extra_revise_prompt`, `extra_revise_prompt_file`, `save_review`, `log_file`, `no_log`
- `[refine.<command>]` overrides the base refine config for a specific command

## Configuration, sandboxing, and tmux

| Command | Description |
|---|---|
| `agm config copy DIRNAME` | Copy known project config files into an existing target directory |
| `agm config cp DIRNAME` | Alias form of `agm config copy` |
| `agm config env` | Print shell statements for refreshing the current checkout environment |
| `agm config update` | Create missing config.toml files and commit generated changes |
| `agm run [--no-sandbox] [--no-patch] [--memory LIMIT] [--swap LIMIT] [--no-memory-limit] [--no-swap-limit] [-f\|--file SETTINGS] COMMAND [ARGS...]` | Run a command directly or in an Anthropic Sandbox Runtime container |
| `agm tmux open [-d\|--detach] [-n\|--num-panes PANES] [SESSION]` | Open a tmux session |
| `agm tmux close SESSION` | Close a tmux session |
| `agm tmux layout PANES [-w\|--window WINDOW_ID]` | Apply AGM's tmux pane layout to a window |

`agm config copy` copies dot-prefixed files and directories from the shared project
config directory into an existing target directory. When run from a branch
worktree, AGM first copies shared dot entries, then copies matching entries from
the branch config subdirectory so branch entries override shared entries.
For `.env` and `.env.local`, AGM writes merged dotenv values using the same
precedence as `agm config env`: shared `.env`, shared `.env.local`, branch
`.env`, then branch `.env.local`.

`agm config env` uses the same environment resolution as `agm open`: project and branch
`config.toml` `[deps]` tables first, then project `.env`, project `.env.local`, project
`env.sh`, and matching branch config files when the current checkout is a branch worktree.
Apply the printed shell statements with:

```bash
eval "$(agm config env)"
```

`agm config update` creates missing project and branch `config.toml` files under the project
config directory, updates dependency configuration entries, and commits any generated
changes to the config repository's git history with a `chore: update config` commit message.

When the config directory is a git repository, AGM automatically commits changes it makes to
the config directory. In addition to `agm config update`, this covers `agm init`,
`agm open`, `agm close`, `agm dep new`, `agm dep switch`, and `agm worktree new`, each of
which commits the config it adds, updates, or removes for the affected branch. Pass
`agm init --no-git-init` (or `--no-config-git`) to opt out of creating the config git
repository, which disables these automatic commits.

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

`agm tmux open` options:

- `-d`, `--detach`: create the session without attaching
- `-n`, `--num-panes PANES`: create the session with `PANES` panes

`agm tmux layout` options:

- `-w`, `--window WINDOW_ID`: apply the layout to a specific tmux window ID

## AgL workflow DSL

### `agm exec FILE`

Execute an AgL (Agent Language) workflow program.

```text
agm exec [--input KEY=VALUE]... [--strict-json|--no-strict-json]
         [--max-iters N] [--runner COMMAND]
         [--log-file PATH|--no-log]
         FILE
```

Options:

- `--input KEY=VALUE`: Provide a host input value (repeatable). Values for `text`-declared
  inputs are taken verbatim; for every other declared type (`int`/`decimal`/`bool`/`json`
  and the structured `list`/`dict`/`record`/`enum` types) the value must be **exactly one
  bare JSON value**, parsed strictly (no fence stripping or repair) and validated against the
  declared type's schema. A missing or undeclared input, or one that fails to parse/validate,
  is a host invocation error reported before any agent runs.
- `--strict-json`: Require agents to return exactly one bare JSON value (no fences, prose, or
  repair). Overridable per call site with the `strict_json` call option.
- `--no-strict-json`: Use lenient JSON recovery (the default): the runtime recovers exactly
  one JSON value from chatty output (stripping fences/prose, repairing trivially malformed
  JSON), then validates it strictly against the schema. The recovered (normalized) value is
  traced alongside the raw output.
- `--max-iters N`: Override the default `do`-loop iteration limit.
- `--runner COMMAND`: Override the default agent runner command. The full resolution chain for
  a named agent not listed in `[exec.agents]` is, in precedence order:
  `[exec.agents.<name>]` → `--runner` flag → `[exec] runner` (config) → `[loop] runner`
  (config) → `claude -p` (built-in default). Named agents listed in `[exec.agents]` always
  use their own command regardless of `--runner`.
- `--log-file PATH`: Write a structured JSONL trace log to PATH (default: auto-generated under
  `.agent-files/`).
- `--no-log`: Disable trace logging entirely.
- `--dry-run`: Run the full static pipeline, input validation, and contract
  materialization, then stop before executing any statement (static errors exit 1; a clean
  check exits 0 with no program output).  When the check succeeds and one or more agent-call
  or `exec` sites exist, the static call-site inventory is printed to stdout:

  ```
  call-sites:
    line N:C: <callee> → <target-type> [<codec>[, schema: yes][, policy: <policy>]]
  ```

  Each entry shows the 1-based source line and column (`N:C`), the callee name
  (`prompt`, `exec`, or a
  registered agent name), the target type, the selected codec (`text` or `json`), and
  optionally whether a JSON Schema is attached (`schema: yes`) and the effective
  parse-failure policy (`abort` or `retry[N]`).  When no agent calls are present, no
  inventory is printed.

Exit codes:

| Code | Meaning |
|------|---------|
| `0` | The workflow completed successfully |
| `1` | Pre-execution failure: unreadable file, static lex/parse/scope/typecheck diagnostics, host configuration error, or input validation failure |
| `2` | The workflow executed but ended with an uncaught AgL exception |

Diagnostics and warnings:

- Error-severity diagnostics (static lex/parse/scope/typecheck errors, host
  configuration errors, input validation failures) and uncaught AgL exceptions
  are printed to stderr and determine the exit code per the table above.
- Advisory **warnings** (for example a non-exhaustive `case` over an enum that
  omits some variants) are a separate channel. They are printed to stderr with a
  `warning:` prefix (`warning: line N: message`) to disambiguate them from
  errors, and they never affect the exit code — the program still runs to
  completion. Program `print` output goes to stdout, kept clean of diagnostics.

Config (`[exec]` section in `config.toml`):

```toml
[exec]
runner = "claude -p"        # default agent runner
strict_json = false         # lenient JSON recovery is the default
default_loop_limit = 5      # do[] default iteration bound
timeout = "30m"             # idle timeout

[exec.agents]
reviewer = "claude -p"      # per-agent runner commands
```

`[exec.<command>]` sub-tables provide per-command overrides of the base `[exec]`
settings. The name `agents` is reserved for the structural `[exec.agents]` map
and is never treated as a per-command override.

## Help and aliases

| Alias | Canonical form |
|---|---|
| `agm wt` | `agm worktree` |
| `agm wt rm` | `agm worktree remove` |
| `agm wt remove` | `agm worktree remove` |
| `agm worktree rm` | `agm worktree remove` |
| `agm config cp` | `agm config copy` |
| `agm dep remove` | `agm dep rm` |

Use `agm help` to show the command overview and `agm help <command>` for detailed command help.
