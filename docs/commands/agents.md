# Agent workflows

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
is 12. Review output is saved to the default timestamped review path by default.

When `COMMAND` is provided, config from `[refine.COMMAND]` is merged over `[refine]` and the same
command name is forwarded to review/revise config lookup.

`agm refine` options:

- `--max-steps N|unlimited`: maximum revision attempts (default: 12). Use `unlimited` for no limit.
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
