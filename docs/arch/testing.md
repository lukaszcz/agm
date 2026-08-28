# Testing

The test suite mirrors the architecture: the AgL pipeline is tested pass by pass, the command layer is tested through its CLI surface, and the primitives and domain packages have focused unit tests. Tests live under `tests/`, grouped by module or category.

## Strategy

- **AgL passes** are tested individually — lexer, parser, AST, scope, typecheck, match compilation, lowering, IR, and evaluator each have their own suites — plus end-to-end acceptance suites that run whole programs. Scope and typecheck have no per-module entry point (production always resolves and checks a whole program), so their unit suites build a real single-entry `ModuleGraph` through the shared helpers in `tests/agl/module_graph.py` (`resolve_entry`/`resolve_and_check_entry` for static file source, `resolve_inline_entry`/`resolve_and_check_inline_entry` for statement-oriented command source, and `resolve_program_ast`/`resolve_and_check_program_ast` for an already-parsed `Program`) rather than resolving/checking a bare AST, so no unit test exercises a configuration production does not.
- **Commands** are tested at the CLI boundary, exercising user workflows rather than internal call sequences.
- **Domain and primitives** (project layout, git, config, process/env) have unit tests for their behavior and edge cases.

The guiding rule is to test user workflows and observable behavior, not implementation details: exact help, warning, and error message text is deliberately not asserted, and tests must not be flaky.

## Invariants Enforced by Tests

Some tests guard architectural properties rather than feature behavior:

- **Package layering.** A dependency-contract test asserts the AgL package boundaries — `semantics` as the shared semantic foundation, `syntax` as an AST-only leaf, `typecheck` confined to scope's output and the frontend layers beneath it, match compilation importing no IR/lowering/evaluator/runtime code, the IR importing no frontend or match-compiler code, the evaluator never importing the frontend, the runtime staying eval-free, and the pipeline on top.
- **End-to-end acceptance.** Whole-program suites for module and multi-file AgL programs are part of the standing gate and must stay green. Their scenario harness scripts agent and shell boundaries, and can build declared filesystem fixtures in per-scenario temporary roots, so workflow coverage remains deterministic without executing real agents or shell commands or touching repository fixtures.
- **Match/IR contracts.** Match-compilation and lowering suites cover shared member-record constructor keys, singleton decomposition, record-nominal case dispatch, static JSON encode plans, and member-record equality, copying, codec, agent, and FFI crossings independently of end-to-end programs.
- **Coverage.** The project maintains 100% test coverage of `src/` and of `stdlib/std/` — the standard library's Python companions ship in the wheel, so they are gated like any other shipped code — plus 100% command coverage in end-to-end tests.

## Editor Mode Tests

The Emacs AgL mode under `config/emacs/` carries its own ERT suite, run by `just test-emacs` (batch Emacs, no display). It sits outside `just check` on purpose, so the gate stays deterministic on machines without Emacs; run it by hand when touching the mode. It tests the mode's own behavior (syntax propertization, structural font-lock, indentation, navigation, flymake, and REPL wiring), not AgL semantics. Both editor modes mirror three inventories owned by the Python sources, in full: the reserved-keyword frozenset in `src/agm/agl/keywords.py`, the raw-tail openers in `src/agm/raw_tail_catalog.py`, and the builtin call names in `src/agm/agl/scope/symbols.py`. The mirrors are maintained by hand, so a change to any of those three inventories must be carried into `config/emacs/agl-mode.el` and `config/micro/agl.yaml`. Builtins are faced by spelling wherever they appear, their own `builtin def` declaration in the stdlib included. Both modes also have to honour AgL's identifier boundary, where every character but the `IDENT_STOP` delimiters continues a name: the Emacs mode checks it directly against a mirror of `IDENT_STOP`, while the micro rules — regexp-only, with no lookaround — get there by layering, repainting any name that contains an operator character before re-facing the few spellings that legitimately contain one.

The micro rules under `config/micro/` carry a matching suite, run by `just test-micro` and likewise outside `just check`. It drives the rules through micro's own highlighter package rather than a reimplementation, so the tests see the real regexp engine, rule precedence, and string/comment regions; the first run fetches that package, and the recipe passes `-count=1` because `agl.yaml` is data the harness reads rather than a Go dependency the test cache tracks. Its centre of gravity is the identifier boundary — that every builtin and raw-tail opener is faced, and that names merely containing one (`resource-path`, `ask-request-later`, `x-resource`) are not — plus the multi-character operator names (`::`, `:=`, `|>`, any declared spelling) and the `?`/`?N` placeholders, which the delimiter rules would otherwise split across faces. A raw-tail payload is a region, so the tail of an `exec$`/`ask$` line — its opener included, since micro paints a region's start delimiter with the region's group — faces as verbatim text rather than as code.

## AgL Self-Validation

AgL carries invariant self-checks that re-verify artifacts the compiler itself just produced: the typechecker's checked-output closure boundary (no solver-local inference variable escapes a checked module, program, or the shared whole-program tables) and its per-region inference-close checks; every compiled match site's matrix, occurrence ledger, decision DAG, semantic replay, and provenance; and the structural validation of the lowered IR (`validate_ir`). These re-check already-checked source, so they are defense-in-depth rather than production behavior and are disabled by default.

One toggle (`agm.agl.self_validation`) gates all of them. `tests/conftest.py` turns it on for the whole suite, so every match site compiled and every program lowered anywhere in the tests doubles as an invariant oracle while production pays nothing. Tests that pin the production path take the `self_validation_disabled` fixture; `tests/test_agl_self_validation.py` holds the gating contract.

## Agents in Tests

Real agents (claude, codex, and other runners) are never invoked in tests; agent and shell boundaries are always mocked. This keeps the suite deterministic and offline. The command e2e harness installs AGM into a temporary venv through `uv` in offline mode, so missing cached package artifacts fail as local setup problems instead of reaching the network. Tests are also written to survive concurrent and cross-worktree runs — no hardcoded temp paths, and interrupt tests restore default signal handling.

## Hermeticity

A test run never reads or writes installed AGM files, whatever the developer's machine holds. Autouse fixtures in `tests/conftest.py` enforce this on three seams: every `AGM_*` variable inherited from a workspace shell is stripped, `AGM_STDLIB` is pinned to the in-repo `stdlib/`, and the installation prefix is reported as absent so the `<prefix>/.agm` home fallback never fires. That last one matters because the prefix comes from `sys.argv[0]`: launched as `uv run pytest` it is the project's own `.venv`, which `uv run agm pkg install` populates with a real package store. Environment pinning alone cannot cover it, since most tests pass an explicit `env` mapping that never consults the process environment. Tests asserting the fallback itself take the `installed_agm_prefix` fixture and point it at a temporary directory; `tests/test_packages_store.py` also carries the proof that a populated prefix store stays unreachable and unwritten.

## Code Entry Points

- `tests/` — all tests; AgL pass suites are `tests/test_agl_*.py`, command suites are named per command.
- `tests/test_agl_dependencies.py` — the package-layering contract.
- `tests/test_agl_self_validation.py` — the self-validation gating contract; `src/agm/agl/self_validation.py` — the toggle.
- `tests/test_agl_e2e.py` and `tests/agl/programs/` — module end-to-end acceptance; `tests/test_agl_multifile.py` with `tests/agl/multi_file/` — multi-file acceptance.
- `tests/conftest.py`, `tests/_agl_helpers.py`, `tests/_proc_helpers.py`, and `tests/_process_helpers.py` — shared fixtures and helpers; the latter provides reusable process-result builders and shell-boundary fakes.
- `config/emacs/tests/` — the Emacs mode's ERT suite; `config/micro/tests/` — the micro rules' Go suite; `config/emacs/agl-mode.el` and `config/micro/agl.yaml` — the editor keyword mirrors.
- `justfile` — the `test`, `lint`, `typecheck`, `test-emacs`, `test-micro`, and `check` gates that run the suite.
