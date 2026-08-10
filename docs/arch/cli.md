# CLI and Command Dispatch

AGM's entry point is a Typer application that defines the whole command tree, parses arguments, and dispatches to command implementations. The CLI layer is deliberately thin: it validates and packages inputs, then hands off to a command module that owns the actual work.

## Command Tree

The command tree is defined once as a set of Typer apps — a root app plus one sub-app per command group — and several top-level commands. Group callbacks print help when invoked without a subcommand.

The structure of `src/agm/commands/` mirrors the command implementations: command groups have one module per subcommand, while Typer group callbacks remain in `cli.py`. For example, `agm pkg check` dispatches to `commands/pkg/check.py`. Locating a command implementation is therefore a direct path lookup from its CLI name.

When no built-in root command matches, the root group derives the effective package command index from the active or project-pinned package manifests and resolves registered command paths greedily. Its unmatched trailing words are forwarded as AgL program arguments. The index identifies a candidate only; the shared execution host verifies it against the selected manifest and module ownership before running it. Built-in commands never consult this index or load AgL; registered commands enter the shared execution host only after the fallback resolves.

## Argument Handling

CLI options and arguments are declared in `cli.py`. Rather than threading loose parameters into command functions, the CLI parses them into typed dataclass containers (one per command) before dispatch. These containers are the stable contract between the CLI surface and the command implementations, keeping signatures small and the parsing rules in one place.

Two global concerns are handled at the CLI boundary: a `--dry-run` flag, stored in the command context and consulted by the dry-run primitive (see [core.md](core.md)); and a custom help path that bypasses Click's built-in `--help` so commands can accept pass-through options (notably `agm exec`, which forwards program-defined parameters as first-class flags). Argument-validation helpers print a usage error with the relevant help text when required values are missing.

## Help and Completion

Help text and the command overview live in `parser.py`, separate from the wiring in `cli.py`, and are resolved by command path. The overview derives the effective package command index only when rendering help, appending registered command paths and descriptions. Shell completion lives in `completion.py`: it discovers dynamic completion values — branch names, dependency names, project paths, tmux sessions, registered command-path segments, and AgL program parameters — by consulting the relevant lightweight index, git, the project layout, and AgL source rather than hard-coding lists. Registered paths traverse one shell token at a time; once a path resolves to a registered program, its trailing program arguments do not fall back to root-command completions. Dynamic completion failures degrade to no suggestions. `exec` parameter help and completion use its effective module roots, including the containing development package and path-sourced dependencies.

## Code Entry Points

- `src/agm/cli.py` defines the Typer apps, options, global flags, and dispatch into command modules; `src/agm/cli_dispatch.py` supplies the lazy registered-command fallback.
- `src/agm/parser.py` holds help texts and path-based help/overview resolution.
- `src/agm/completion.py` provides dynamic shell completions.
- `src/agm/commands/` contains the implementations; `commands/exec_program.py` is the shared host for file, installed-reference, and registered AgL programs, while `commands/exec.py` adapts the explicit `exec` surface.
- `src/agm/cli_support/args.py` defines the typed per-command argument containers; `src/agm/cli_support/exec_params.py` discovers AgL program parameters for `agm exec` option wiring while `-p` selects a discovered entry declaration; `src/agm/cli_support/engine_seeds.py` resolves the CLI-over-config engine settings that `agm exec` and `agm repl` both seed the AgL engine with, returning both typed `Value` seeds and host-supplied AgL source overrides (an `--agent`/`[exec] default-agent` literal) for the pipeline's `setting_overrides` seam. Only `exec` and `repl` reach it, so non-AgL commands stay free of AgL imports.
- `docs/commands/` is the authoritative user-facing reference for command syntax and behavior (one page per command area; start at `docs/commands/index.md`).
