# Testing

The test suite mirrors the architecture: the AgL pipeline is tested pass by pass, the command layer is tested through its CLI surface, and the primitives and domain packages have focused unit tests. Tests live under `tests/`, grouped by module or category.

## Strategy

- **AgL passes** are tested individually — lexer, parser, AST, scope, typecheck, lowering, IR, and evaluator each have their own suites — plus end-to-end acceptance suites that run whole programs.
- **Commands** are tested at the CLI boundary, exercising user workflows rather than internal call sequences.
- **Domain and primitives** (project layout, git, config, process/env) have unit tests for their behavior and edge cases.

The guiding rule is to test user workflows and observable behavior, not implementation details: exact help, warning, and error message text is deliberately not asserted, and tests must not be flaky.

## Invariants Enforced by Tests

Some tests guard architectural properties rather than feature behavior:

- **Package layering.** A dependency-contract test asserts the AgL package boundaries — `semantics` as the foundation leaf, the IR depending only on its own data, the evaluator never importing the frontend, the runtime staying eval-free, and the pipeline on top.
- **End-to-end acceptance.** Whole-program suites for single-file and multi-file AgL programs are part of the standing gate and must stay green.
- **Coverage.** The project maintains 100% test coverage of `src/` and 100% command coverage in end-to-end tests.

## Agents in Tests

Real agents (claude, codex, and other runners) are never invoked in tests; agent and shell boundaries are always mocked. This keeps the suite deterministic and offline. Tests are also written to survive concurrent and cross-worktree runs — no hardcoded temp paths, and interrupt tests restore default signal handling.

## Code Entry Points

- `tests/` — all tests; AgL pass suites are `tests/test_agl_*.py`, command suites are named per command.
- `tests/test_agl_dependencies.py` — the package-layering contract.
- `tests/test_agl_e2e.py` and `tests/agl/programs/` — single-file end-to-end acceptance; `tests/test_agl_multifile.py` with `tests/agl/multi_file/` — multi-file acceptance.
- `tests/conftest.py`, `tests/_agl_helpers.py`, `tests/_proc_helpers.py` — shared fixtures and helpers.
- `justfile` — the `test`, `lint`, `typecheck`, and `check` gates that run the suite.
