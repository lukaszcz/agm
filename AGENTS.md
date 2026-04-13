# Repository Guidelines

AGM is an Agent Project Management CLI tool.

## Tech stack

- Python 3.12
- Plumbum

## Project Structure & Module Organization

Production code lives in `src/agm/`, organized by area:
- `commands/` for subcommands,
- `utils/` for shared helpers,
- `vcs/` for git integration,
- `tmux/` for session/layout logic.

Tests live in `tests/`. Use `test_cli_parsing.py` for parser coverage, `test_project_utils.py` for focused utilities, and `test_e2e.py` for full CLI workflows.

Project notes and command docs are in `docs/`.

Sandbox config templates live in `sandbox/`, and `stubs/` provides local typing support for third-party modules.

## Build, Test, and Development Commands

Use `just` for the standard workflow:

- `just setup` creates `.venv` with Python 3.12 and installs the project plus dev dependencies via `uv`.
- `just lint` runs `ruff check src/ tests/`.
- `just test` runs `pytest tests/ -q`.
- `just typecheck` runs strict `mypy` with `MYPYPATH=src:stubs`.
- `just check` runs linting, tests, and type checking together.
- `just install` installs the `agm` CLI and copies sandbox configs into `~/.sandbox/`.

Run the CLI locally with `uv run agm ...` when iterating on a command.

## Coding Style & Naming Conventions

- Formatting: ruff (line length 100)
- Typing: strict discipline (`mypy` strict + `basedpyright`); modern union syntax (`str | None`, `dict[str, int]`, `list[str]`)
- Do NOT use `type: ignore` comments. If ignoring a type rule is necessary, ALWAYS ask the user for permission and explain why.
- Do NOT use `noqa` comment. If ignoring a lint rule is necessary, ALWAYS ask the user for permission and explain why.
- Do not use `fmt: skip` or `fmt: off` comments. If ignoring the formatter is necessary, ask the user for permission and explain why.

## Testing Guidelines

- **IMPORTANT**: Every new feature should include tests that verify its correctness at the appropriate levels (unit, integration, and possibly system level).
- **IMPORTANT**: Follow Test Driven Development (TDD). Write failing tests first, implement changes later to make the tests pass.
- **IMPORTANT**: For every bug found, add a regression test that fails because of the bug, then fix the bug and ensure the test passes.
- Avoid brittle tests. Test user workflows, not implementation details.

## Commit Guidelines

- Commit format: `type: subject` in imperative lowercase (e.g., `feat: add transfer flow`).
- Keep commits focused; avoid mixing unrelated changes.

## Security & Configuration Tips

Do not commit machine-specific sandbox files or secrets. When changing `agm run` behavior, verify both repository-local `.sandbox/` settings and the installed files copied from `sandbox/`.

## Instructions

- Avoid code duplication. Abstract common logic into parameterized functions.
- Do NOT try to circumvent static analysis tools. Adapt the code to pass `just check` properly - do not ignore checks or suppress rules. If you absolutely need to bypass a static analysis tool, ALWAYS ask the user for approval and explain why this is necessary.
- When finished, verify with `just check`.
