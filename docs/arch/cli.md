# CLI and Command Dispatch

AGM's entry point is a Typer application that defines the whole command tree, parses arguments, and dispatches to command implementations. The CLI layer is deliberately thin: it validates and packages inputs, then hands off to a command module that owns the actual work.

## Command Tree

The command tree is defined once as a set of Typer apps — a root app plus one sub-app per command group — and several top-level commands. Group callbacks print help when invoked without a subcommand.

The structure of `src/agm/commands/` mirrors this tree exactly: a command group is a directory with an `__init__.py` callback and one module per subcommand. Locating the implementation of any command is therefore a direct path lookup from its CLI name.

## Argument Handling

CLI options and arguments are declared in `cli.py`. Rather than threading loose parameters into command functions, the CLI parses them into typed dataclass containers (one per command) before dispatch. These containers are the stable contract between the CLI surface and the command implementations, keeping signatures small and the parsing rules in one place.

Two global concerns are handled at the CLI boundary: a `--dry-run` flag, stored in the command context and consulted by the dry-run primitive (see [core.md](core.md)); and a custom help path that bypasses Click's built-in `--help` so commands can accept pass-through options (notably `agm exec`, which forwards program-defined parameters as first-class flags). Argument-validation helpers print a usage error with the relevant help text when required values are missing.

## Help and Completion

Help text and the command overview live in `parser.py`, separate from the wiring in `cli.py`, and are resolved by command path. Shell completion lives in `completion.py`: it discovers dynamic completion values — branch names, dependency names, project paths, tmux sessions, and AgL program parameters — by consulting git, the project layout, and AgL source rather than hard-coding lists.

## Code Entry Points

- `src/agm/cli.py` defines the Typer apps, options, global flags, and dispatch into command modules.
- `src/agm/parser.py` holds help texts and path-based help/overview resolution.
- `src/agm/completion.py` provides dynamic shell completions.
- `src/agm/commands/` contains the implementations; its directory layout mirrors the CLI command tree.
- `src/agm/cli_support/args.py` defines the typed per-command argument containers; `src/agm/cli_support/exec_params.py` discovers AgL program parameters for `agm exec` option wiring.
- `docs/commands/` is the authoritative user-facing reference for command syntax and behavior (one page per command area; start at `docs/commands/index.md`).
