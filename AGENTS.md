# Repository Guidelines

AGM is an Agent Project Management CLI tool.

## Tech stack

- Python 3.12
- Plumbum
- Typer

Use `uv run` for all Python tooling.

## Architecture and project structure

Read @docs/arch/index.md to understand AGM implementation architecture.

**IMPORTANT**: Update docs/arch/*.md whenever AGM implementation architecture changes – always keep these files up-to-date with the codebase.

The primary purpose of architecture docs in docs/arch/*.md is to provide agents with a quick but comprehensive overview of the system's architecture and the codebase. Treat the docs as an onboarding guide. When updating, do not add brittle implementation details, but do include info on where to find relevant codebase references. Be succinct, not verbose. Provide architectural overview, not mechanism details. Match the existing writing style.

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
- Test only main app Python code under `src/agm/`, NOT build/install scripts, `justfile` commands or config file content. Do NOT test exact help, warning or error messages.
- Make sure tests are not flaky.
- Maintain 100% test coverage of `src/`.
- Maintain 100% command coverage in e2e tests.
- Group the tests in `tests/` by meaningful categories and name the files meaningfully.
- NEVER run real agents (claude, codex, pi, ...) in the tests - ALWAYS mock agent calls.

## Commit Guidelines

- Commit format: `type: subject` in imperative lowercase (e.g., `feat: add transfer flow`).
- Keep commits focused; avoid mixing unrelated changes.

## Documentation

- Keep docs (`README.md` and `docs/commands/*.md`) and command help texts up to date with implemented command functionality. `README.md` is a brief description of the AGM program and should not contain overhwelming details, while `docs/commands/*.md` and the help texts are comprehensive command references.
- ALWAYS keep comments and docstrings up-to-date with the codebase.
- Avoid references to plans, milestones, or unversioned files in the docs, comments and docstrings.

## Instructions

- Avoid code duplication. Abstract common logic into parameterized functions.
- Do NOT create new worktrees - edit the current worktree directly.
- Do NOT try to circumvent static analysis tools. Adapt the code to pass `just check` properly - do not ignore checks or suppress rules. If you absolutely need to bypass a static analysis tool, ALWAYS ask the user for approval and explain why this is necessary.
- When finished, verify with `just check`.
