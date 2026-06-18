# Repository Guidelines

AGM is an Agent Project Management CLI tool.

## Tech stack

- Python 3.12
- Plumbum
- Typer

Use `uv run` for all Python tooling.

## Project Structure

Production code lives in `src/agm/`, organized by area:
- `agl/` for the AgL agent workflow DSL (lexer, parser, AST, scope, typecheck, eval, runtime),
- `commands/` for CLI entrypoints and command implementations; directory structure of `commands/` must reflect the CLI command tree exactly, including nested command groups (`config/`, `dep/`, `loop/`, `tmux/`, `worktree/`, etc.),
- `config/` for loading and resolving general and sandbox configuration,
- `core/` for environment and process primitives shared across features,
- `project/` for project/worktree setup and layout management,
- `sandbox/` for sandbox runtime/template support,
- `tmux/` for tmux session and layout logic,
- `vcs/` for git integration.

CLI arguments/options are defined in `src/agm/cli.py`, parser construction lives in `src/agm/parser.py`, and custom argument-value completions for CLI parameters live in `src/agm/completion.py`.

Tests live in `tests/`. Project notes and command docs are in `docs/`.

Additional directories:
- `config/` for config templates,
- `stubs/` for local typing support for third-party modules,
- `tools/` for repository tooling.

## Build, Test, and Development Commands

Use `just` for the standard workflow:

- `just setup` creates `.venv` with Python 3.12 and installs the project plus dev dependencies via `uv`
- `just lint` runs `ruff check src/ tests/`
- `just test` runs the test suite
- `just typecheck` runs strict `mypy` with `MYPYPATH=src:stubs`
- `just check` runs linting, tests, and type checking together
- `just install` installs the `agm` CLI and copies default config into `~/.agm/`

Run the CLI locally with `uv run agm ...` when iterating on a command.

## Coding Style & Naming Conventions

- Formatting: ruff (line length 100)
- Typing: strict discipline (`mypy` strict); modern union syntax (`str | None`, `dict[str, int]`, `list[str]`)
- Do NOT use `type: ignore` comments. If ignoring a type rule is necessary, ALWAYS ask the user for permission and explain why.
- Do NOT use `noqa` comment. If ignoring a lint rule is necessary, ALWAYS ask the user for permission and explain why.
- Do not use `fmt: skip` or `fmt: off` comments. If ignoring the formatter is necessary, ask the user for permission and explain why.

## Testing Guidelines

- **IMPORTANT**: Every new feature should include tests that verify its correctness at the appropriate levels (unit, integration, and possibly system level).
- **IMPORTANT**: Follow Test Driven Development (TDD). Write failing tests first, implement changes later to make the tests pass.
- **IMPORTANT**: For every bug found, add a regression test that fails because of the bug, then fix the bug and ensure the test passes.
- Avoid brittle tests. Test user workflows, not implementation details.
- Test only main app Python code under `src/agm/`, not build/install scripts, `justfile` commands or config file content. Do not test exact help messages.
- Maintain 100% test coverage of `src/`.
- Maintain 100% command coverage in e2e tests.
- Test files in `tests/` group tests by module or meaningful category.

## Commit Guidelines

- Commit format: `type: subject` in imperative lowercase (e.g., `feat: add transfer flow`).
- Keep commits focused; avoid mixing unrelated changes.

## Instructions

- Avoid code duplication. Abstract common logic into parameterized functions.
- Do NOT try to circumvent static analysis tools. Adapt the code to pass `just check` properly - do not ignore checks or suppress rules. If you absolutely need to bypass a static analysis tool, ALWAYS ask the user for approval and explain why this is necessary.
- Keep docs (README.md and docs/commands.md) and command help texts up to date with implemented command functionality. README.md is a brief description of the AGM program and should not contain overhwelming details, while docs/commands.md and the help texts are comprehensive command references.
- When finished, verify with `just check`.
