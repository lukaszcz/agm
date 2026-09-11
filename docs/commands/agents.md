# Agent workflows

| Command | Description |
|---|---|
| `agm review [COMMAND] [--scope REVIEW_SCOPE] [--aspects REVIEW_ASPECTS] [--extra-aspects REVIEW_ASPECTS] [--runner COMMAND] [--prompt TEXT\|--prompt-file PATH] [--extra-prompt TEXT\|--extra-prompt-file PATH] [--review-file FILE\|auto\|none\|--no-review-file]` | Run the review prompt |
| `agm revise [COMMAND] [--runner COMMAND] [--prompt TEXT\|--prompt-file PATH] [--extra-prompt TEXT\|--extra-prompt-file PATH] REVIEW_FILE` | Run the revision prompt |
| `agm refine [COMMAND] [--max-steps N\|unlimited] [--no-max-steps] [--runner COMMAND] [--reviewer COMMAND] [--reviser COMMAND] [--scope REVIEW_SCOPE] [--aspects REVIEW_ASPECTS] [--review-prompt TEXT\|--review-prompt-file PATH] [--extra-review-prompt TEXT\|--extra-review-prompt-file PATH] [--revise-prompt TEXT\|--revise-prompt-file PATH] [--extra-revise-prompt TEXT\|--extra-revise-prompt-file PATH] [--save-review\|--no-save-review] [--review-file FILE\|auto\|none] [--log-file PATH\|--no-log]` | Run review/revise refinement cycles |

## Prompt interpolation

These workflows expand `%{name}` holes in prompt content before running an agent. Names are AgL identifiers (e.g. `%{log-file}`). `\%{` escapes to a literal `%{`; a bare `%` is literal. `$VAR`/`${VAR}` are plain text — shell-style interpolation isn't supported.

Expansion is strict: an unknown variable, invalid hole name, or unterminated `%{` is an error. Variables: the process environment overlaid with each command's workflow variables below (which win on conflict):

- `review`: `REVIEW_SCOPE`, `REVIEW_ASPECTS`
- `revise`: `REVIEW_FILE`
- `refine`: `REVIEW_SCOPE`/`REVIEW_ASPECTS` for each review prompt, then `REVIEW_FILE` for the matching revise prompt

## Runner command interpolation

Runner command arguments for `review`, `revise`, and `refine` (including refine's reviewer/reviser) interpolate the same `%{name}` holes as their accompanying prompt (see [Prompt interpolation](#prompt-interpolation) above), overlaid with `PROMPT_FILE` (wins on conflict). `%%` aliases `%{PROMPT_FILE}`; either inserts the prompt path verbatim, without recursive interpolation. Commands are shlex-split before interpolation — quote or otherwise protect `\%{` so its backslash reaches the argv element. A placeholder positions the prompt path explicitly; otherwise AGM appends `@<path>`.

## Shared conventions

- Each `--*-prompt TEXT` / `--*-prompt-file PATH` pair (including `--extra-*` variants and refine's `--review-prompt`/`--revise-prompt` forms) is mutually exclusive; `--extra-*` content is appended after the primary prompt.
- `--runner COMMAND` for `review` and `revise`: when unset, uses the same default runner as `agm loop`.
- `COMMAND` merges `[x.COMMAND]` config over `[x]` (`x` = `review`, `revise`, `refine`); `revise` takes `COMMAND` before `REVIEW_FILE`, and `refine` forwards it to review/revise config lookup.
- CLI precedence: an explicit `--*-prompt` or `--*-prompt-file` flag overrides both configured forms — e.g. `--review-prompt-file` overrides a configured inline `review_prompt`.

`agm review` runs the review prompt (default `review.md`, receiving `REVIEW_SCOPE`/`REVIEW_ASPECTS`).

`agm review` options:

- `--runner COMMAND`: review runner command
- `--scope REVIEW_SCOPE`: review scope (default: `changes on current branch`)
- `--aspects REVIEW_ASPECTS`: review aspects (default: `correctness, completeness, maintainability, adherence to AGENTS.md`)
- `--extra-aspects REVIEW_ASPECTS`: additional aspects, appended to the defaults
- `--prompt TEXT` / `--prompt-file PATH`: override the default `review.md` prompt
- `--extra-prompt TEXT` / `--extra-prompt-file PATH`: append extra content to the review prompt
- `--review-file FILE|auto|none` / `--no-review-file`: save review output. Default path: `.agent-files/review-YYYYMMDD-HHMMSS-microseconds.md`; `auto` selects that default explicitly; `none`/`--no-review-file` disables saving.

`agm review` config keys in `config.toml`:

- `[review] runner`, `scope`, `aspects`, `extra_aspects`, `prompt`, `prompt_file`, `extra_prompt`, `extra_prompt_file`, `review_file`

`agm revise` runs the revision prompt (default `revise.md`, receiving `REVIEW_FILE`).

`agm revise` options:

- `--runner COMMAND`: revision runner command
- `--prompt TEXT` / `--prompt-file PATH`: override the default `revise.md` prompt
- `--extra-prompt TEXT` / `--extra-prompt-file PATH`: append extra content to the revision prompt

`agm revise` config keys in `config.toml`:

- `[revise] runner`, `prompt`, `prompt_file`, `extra_prompt`, `extra_prompt_file`

`agm refine` runs review/revise cycles until the revise response is `COMPLETE`, or the maximum number of revision attempts is reached. A `CONTINUE` response from revise starts a fresh review; any other non-`COMPLETE` response retries revise with the same review file. Default maximum: 12; an explicit `--max-steps` overrides a configured `no_max_steps = true`. Under `--dry-run`, refine prints one review/revise cycle and exits, including when the configured limit is unlimited. Review output saves to the default timestamped review path by default.

`agm refine` options:

- `--max-steps N|unlimited`: maximum revision attempts (default: 12); `unlimited` for no limit
- `--no-max-steps`: disable the step limit (run until COMPLETE); mutually exclusive with `--max-steps`
- `--runner COMMAND`: runner command for both review and revise
- `--reviewer COMMAND`: review runner command; overrides `--runner` for the review step
- `--reviser COMMAND`: revision runner command; overrides `--runner` for the revise step
- `--scope REVIEW_SCOPE`: review scope
- `--aspects REVIEW_ASPECTS`: review aspects
- `--review-prompt TEXT` / `--review-prompt-file PATH`: override the default review prompt
- `--extra-review-prompt TEXT` / `--extra-review-prompt-file PATH`: append extra content to the review prompt
- `--revise-prompt TEXT` / `--revise-prompt-file PATH`: override the default revision prompt
- `--extra-revise-prompt TEXT` / `--extra-revise-prompt-file PATH`: append extra content to the revision prompt
- `--save-review` / `--no-save-review`: save or skip saving review output (default: save)
- `--review-file FILE|auto|none`: review output file path, `auto`, or `none` — same values as `agm review`'s `--review-file` above
- `--log-file PATH` / `--no-log`: write command output to a log file or disable logging

`agm refine` config keys in `config.toml`:

- `[refine] max_steps`, `no_max_steps`, `runner`, `reviewer`, `reviser`, `scope`, `aspects`, `review_prompt`, `review_prompt_file`, `extra_review_prompt`, `extra_review_prompt_file`, `revise_prompt`, `revise_prompt_file`, `extra_revise_prompt`, `extra_revise_prompt_file`, `save_review`, `log_file`, `no_log`
