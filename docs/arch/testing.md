# Testing

The suite mirrors the architecture: AgL is tested pass by pass plus whole-program acceptance corpora, commands are tested through their CLI surface, and primitives and domain packages have focused unit tests. Everything lives under `tests/`, grouped by module or category; `tests/AGENTS.md` holds the test-writing conventions.

## Strategy

- **AgL passes** each have their own suites, plus end-to-end corpora under `tests/agl/` (programs with multi-scenario sidecars, static rejections, multi-file programs, packages). Scope and typecheck have no per-module entry point in production, so their unit suites build a real single-entry `ModuleGraph` through `tests/agl/module_graph.py` rather than checking a bare AST. Test sources start from the production entry parse; IR evaluation helpers (`tests/agl/ir_harness.py`) run programs through `PipelineDriver` with injected agent and shell fakes, while inspection helpers stop at an intermediate artifact.
- **Commands** are tested at the CLI boundary. Most tests invoke `agm` once against a fixture; the multi-command arcs a real user follows live in `tests/test_e2e.py`, where state written by one command is proven to be the state the next one reads.
- **Domain and primitives** have unit tests for behavior and edge cases. Package command tests use real archives, manifests, and activation state; failure fixtures corrupt inputs or fail external I/O instead of replacing domain handlers.

Tests assert observable behavior, never exact help, warning, or error text. Import reuse and REPL state are checked through program output, edited modules, recovery, and redeclarations. Real agents are never invoked; agent transports, unavailable tools, and outbound HTTP requests are faked at their external boundaries, while filesystem, git, shell, and loopback HTTP workflows (the package fetcher's own tests) execute real operations. The e2e harness stages a temporary CLI entry point using the test interpreter and checkout source, with ambient Python imports disabled. It installs nothing and needs no package cache or second interpreter. Tests survive concurrent and cross-worktree runs.

## Gates and Invariants

- **Static typing** — `pyproject.toml` configures strict mypy checks and the local source and stub paths, shared by direct `uv run mypy` invocations and `just typecheck`.
- **Package layering** — `tests/test_agl_dependencies.py` asserts the AgL import contract described in [agl/index.md](agl/index.md).
- **Coverage** — 100% line and branch coverage of `src/` and of `packages/stdlib/src/` (the standard library's Python companions ship in the wheel), measured through `sys.monitoring`, which is why `.python-version` pins the development interpreter to Python 3.14.
- **Command coverage** — `tests/_command_coverage.py` walks the live Typer registry and records which leaf commands the e2e suite actually ran through the real binary, merging across xdist workers; it judges only whole-suite runs.
- **Documented examples compile** — every ```` ```agl ```` fence under `docs/agl/reference/` runs through the static pipeline (`tests/test_agl_doc_snippets.py`); an `agl-check` marker declares a deliberately incomplete or rejected block.
- **AgL layout style** — `tools/agl_style.py` is both formatter and checker for every `.agl` file, doc fence, and embedded snippet; `just agl-style` runs inside `just lint`.
- **Self-validation** — the compiler's invariant self-checks (checked-output closure, match-artifact validation, deep IR validation, FFI class shape) are gated by one toggle, `agm.agl.self_validation`; `tests/conftest.py` turns it on for the whole suite so every compilation doubles as an oracle while production pays nothing. The `self_validation_disabled` fixture pins the production path.
- **Hermeticity** — a run never reads or writes installed AGM files: autouse fixtures strip inherited `AGM_*` variables, pin `AGM_STDLIB` to the in-repo `packages/stdlib/`, and report the installation prefix as absent. Tests of the prefix fallback take the `installed_agm_prefix` fixture.
- **No real agents** — an autouse fixture refuses any process spawn that resolves an external agent or sandbox CLI (`tests/_external_agent_clis.py`) outside the test temp tree, so only fake binaries a test writes can run.
- **No real network** — an autouse fixture defaults `agm.core.http.open_session` to an empty scripted transport that refuses any request; a test that scripts HTTP (`tests/_http_helpers.py`) overrides it.
- **Dead code** — `just vulture` runs `tools/vulture_check.py`: vulture at 60% confidence over `src/agm/`, `packages/stdlib/src/`, and `tools/`, where use by tests alone does not count. Its whitelist is built each run from the grammar's rule and alias names (Lark's transformer callbacks) plus a short list of other framework and test hooks in the script; `pyproject.toml` ignores Typer- and key-binding-decorated functions.
- **Vacuous-assertion guard** — `just test-neutral-tmp` re-runs the suite with neutrally named temp directories, so an assertion cannot pass on a path that happens to contain its own test's name.

## Test Cost

Per-test cost is accounted in CPU seconds (`tests/_durations.py`), never wall clock, which swings with the load average under `-n auto`. `just test` enforces a per-test ceiling; `just test-budget` ranks tests to recalibrate it. The dominant AgL cost is compiling the modules behind a program; module precompilation and bounded in-memory artifact reuse ([agl/modules.md](agl/modules.md)) reduce repeated work. Tests isolate the disk cache, and eviction workflows use small libraries and evaluated values. Coverage’s first-use instrumentation cost is included in the CPU measurement. `just check` runs the static gates concurrently with the suite.

## Editor Modes

The Emacs mode (`config/emacs/`) and micro rules (`config/micro/`) carry their own suites, `just test-emacs` and `just test-micro`, kept outside `just check` so the gate stays deterministic without those tools. Both modes mirror two inventories owned by the Python sources — the keyword set in `src/agm/agl/keywords.py` and the builtin names in `src/agm/agl/scope/symbols.py` — by hand, so a change to either must be carried into both modes.

## Code Entry Points

- `tests/test_agl_*.py` — AgL pass suites; `tests/test_agl_e2e.py`, `test_agl_multifile.py` — acceptance suites over `tests/agl/`.
- `tests/test_e2e.py` — the command e2e suite; its `run_agm` helper invokes the staged checkout CLI, also placed on PATH for nested AGM calls.
- `tests/conftest.py`, `_agl_helpers.py`, `_process_helpers.py`, `_package_helpers.py`, `_git_helpers.py`, `_http_helpers.py` — shared fixtures and fakes.
- `tests/_command_coverage.py`, `tests/_durations.py` — the command-coverage and CPU-cost plugins.
- `justfile` — the `test`, `test-budget`, `test-neutral-tmp`, `lint`, `typecheck`, `vulture`, `test-emacs`, `test-micro`, and `check` recipes; `tools/vulture_check.py` — the dead-code gate.
