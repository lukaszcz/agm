# Repository Guidelines

AGM is an Agent Project Management CLI tool.

## Tech stack

- Python 3.14
- Plumbum
- Typer

Use `uv run` for all Python tooling.

## Architecture and project structure

Read @docs/arch/index.md to understand AGM implementation architecture.

**IMPORTANT**: Update docs/arch/**/*.md whenever AGM implementation architecture changes – always keep these files up-to-date with the codebase.

The primary purpose of architecture docs in docs/arch/**/*.md is to provide agents with a quick but comprehensive overview of the system's architecture and the codebase. Treat the docs as an onboarding guide. When updating, do not add brittle implementation details, but do include info on where to find relevant codebase references. Be succinct, not verbose. Provide architectural overview, not mechanism details. Match the existing writing style and succinctness level.

## Build, Test, and Development Commands

Use `just` for the standard workflow:

- `just setup` creates `.venv` with Python 3.14 and installs the project plus dev dependencies via `uv`
- `just lint` runs `ruff check src/ tests/ stdlib/ tools/`, `ruff format --check src/ tests/ stdlib/ tools/`, and `just agl-style` (run `uv run ruff format src/ tests/ stdlib/ tools/` to fix formatting)
- `just agl-style` checks the layout style of every AgL source, doc snippet, and embedded test snippet; `just agl-style-fix` rewrites them into it
- `just test` runs the test suite
- `just typecheck` runs strict `mypy`
- `just check` runs linting, tests, and type checking together
- `just install` installs the `agm` CLI and copies default config into `~/.agm/`

Run the CLI locally with `uv run agm ...` when iterating on a command.

## Coding Style & Naming Conventions

- Formatting: ruff (line length 100)
- AgL layout, everywhere AgL is written (`.agl` files, ```agl fences in docs, snippets inside Python strings): indent a scope region's items under its header, leave a blank line after a region's closer and before a `program def`, and indent a function body written on its own line. `just agl-style-fix` applies it; `just lint` enforces it.
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
- Do NOT add heavy ungated validation or defensive assertions (defense-in-depth) to the code. Write appropriate tests instead. Defense-in-depth assertions are allowed ONLY if they are trivial preconditions or gated behind a test-only flag.
- Make sure tests are not flaky.
- Do not let an assertion pass on the strength of its own test's name: pytest builds `tmp_path` from the test name, so a path in an error message can contain the word being asserted. `just test-neutral-tmp` re-runs the suite with neutrally named temp directories and fails any assertion that does.
- Keep individual tests cheap. `just test` fails any test that overruns the CPU ceiling in its `check_cpu_budget`; `just test-budget` ranks tests by cost so the ceiling can be recalibrated, and `just test-budget test_cpu_budget=<seconds>` tries out a candidate number. Measure cost in CPU seconds, never wall clock — under `-n auto` a test's wall time tracks the load average rather than the test, while its CPU time varies by well under half.
- Maintain 100% test coverage of `src/` and of the standard library's Python companions in `stdlib/src/`.
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

- NEVER duplicate code. Abstract common logic into parameterized functions and separate modules.
- Do NOT create new worktrees - edit the current worktree directly.
- Do NOT try to circumvent static analysis tools. Adapt the code to pass `just check` properly - do not ignore checks or suppress rules. If you absolutely need to bypass a static analysis tool, ALWAYS ask the user for approval and explain why this is necessary.
- Be concise and precise in your responses, comments, docs, and explanations.
- When finished, verify with `just check`.
