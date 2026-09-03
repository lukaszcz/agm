# CLI and Command Dispatch

AGM's entry point is a Typer application that defines the command tree, parses arguments into typed containers, and dispatches to command modules. The CLI layer is deliberately thin: it validates and packages inputs; the command module owns the work.

## Command Tree

The tree is a root app plus one sub-app per command group; group callbacks print help when invoked without a subcommand. `src/agm/commands/` mirrors the tree — `agm pkg check` dispatches to `commands/pkg/check.py` — while the Typer wiring itself stays in `cli.py`.

When no built-in command matches, a lazy fallback derives the package command index from the active or project-pinned package manifests, resolves the longest registered command path, and runs its `program def` through the same execution host as `agm exec` after verifying the manifest and module ownership (see [packages.md](packages.md)). Built-in commands never consult that index or load AgL.

## Argument Handling

Options are parsed into one typed dataclass container per command before dispatch. These containers are the stable contract between the CLI surface and the command implementations. Two global concerns live at the boundary: `--dry-run`, stored in the command context and consulted by the dry-run primitive ([core.md](core.md)); and a custom help path that bypasses Click's `--help`, which lets `agm exec` and registered programs expose both AgL `param` declarations and a selected `program def`'s own value parameters as first-class flags. `agm exec` routes each CLI token between the two mechanisms by flag recognition, so both may supply values in one invocation.

## Help and Completion

Help texts and the command overview live in `parser.py`, resolved by command path; the overview appends registered package commands. Shell completion in `completion.py` discovers dynamic values — branches, dependencies, project paths, tmux sessions, registered command paths, and AgL program parameters — from git, the project layout, the package index, and AgL source rather than hard-coded lists. Failures degrade to no suggestions. Parameter help and completion for `exec` and registered programs use the same effective module roots and the same option-name projection as execution, so the three never disagree.

## Code Entry Points

- `src/agm/cli.py` — Typer apps, options, global flags, dispatch; `src/agm/cli_dispatch.py` — the registered-command fallback.
- `src/agm/parser.py` — help texts and overview; `src/agm/completion.py` — dynamic completions.
- `src/agm/commands/` — implementations. `commands/exec_program.py` is the shared host for file, installed-reference, and registered AgL programs; `commands/exec.py` adapts the explicit `exec` surface; `commands/check.py` runs the static pipeline only.
- `src/agm/cli_support/args.py` — typed per-command argument containers; `exec_target.py` — classifies an `agm exec` argument as inline source, a file, or an installed `PACKAGE/MODULE::PROGRAM` reference; `exec_roots.py` — the effective module-root set shared by `exec` and `check`; `program_options.py` — the type-directed CLI projection (`project_option`) shared by engine-key flags and a `program def`'s own parameter surface (positional slots, options, reservation checks against both the host's own flag inventory and against each other, token parsing, usage/help, completion); it also owns `RESERVED_FLAGS`, the host's full reserved-flag set; `exec_params.py` — AgL `param`-declaration discovery and option wiring, built on `program_options.py`'s `RESERVED_FLAGS`; `program_discovery.py` — `program def` declaration discovery and entry-program selection (filtering to the entry module, matching a `-p` request), shared by `exec_program.py` and the `agm exec --help` path; `engine_seeds.py` — CLI-over-config engine settings for `exec` and `repl`, the only commands that import AgL.
- `docs/commands/` — the user-facing command reference.
