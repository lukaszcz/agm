# AgL Language Implementation

AgL is the statically typed workflow language that AGM programs are written in. Its programs orchestrate agents and shell commands as ordinary typed expressions. The implementation is a conventional compiler frontend followed by a typeless execution layer and a host runtime, exposed through two commands — `agm exec` (run a whole program) and `agm repl` (evaluate incrementally).

This document gives the shape of the AgL subsystem. Read the focused documents below for each part.

## Compilation Pipeline

Every AgL program flows through one pipeline, whether run as a whole or one REPL entry at a time:

```
source (.agl)
  → lexer        (INDENT/DEDENT, string interpolation, NAME/OP_NAME tokens)
  → parser       (Lark LALR grammar)
  → AST          (plain dataclasses — the firewall)
  → scope        (name resolution; full static pass)
  → typecheck    (full static pass; selects concrete operations)
  → match compile (exhaustiveness, redundancy, and decision artifacts)
  → lower + link (closed, typeless executable program)
  → IR eval      (interpreter over the linked program)
        ↘ host runtime: agents, shell execution, the Python FFI registry, codecs, rendering, trace store
```

The linked IR is the only execution format; checked frontend objects are never fed to the evaluator. Scope emits immutable shared pattern slots for uncertain field-directed names, typecheck selects their concrete binders or constructors, and consumers resolve them through checked-artifact accessors. No pass rewrites another pass's resolution tables.

## The Firewall

The lexer and parser are the only Lark-aware code. The AST is the firewall: every pass from scope onward depends solely on the AST dataclasses, never on the parser's types. This is what makes the front end replaceable without touching the static passes or the evaluator.

Two consequences shape the whole codebase:

- **Identifier case carries no semantic category.** Identifiers are case-sensitive (distinct spellings are distinct names), but capitalization never classifies a name. The lexer emits `NAME` for word-starting identifiers and `OP_NAME` for operator-character names; types, constructors, and variables are distinguished by their declaration and binding namespace, not by spelling style. No pass branches on the case of a name.
- **Passes never mutate the AST.** Later passes attach their results in *side tables* keyed by a stable per-node id, carried in the resolved/checked program objects rather than written back onto nodes.

## Shared AGM Layers

The firewall is *semantic*, not an I/O boundary: it isolates the static passes from the parser, not AgL from the rest of AGM. AgL reuses AGM's lower layers rather than reimplementing them — the host runtime and pipeline build on the shared primitives, while AgL-only types sit on top:

- **Agent invocation** goes through `agm.agent.runner` (the same prepare/run path as loop/review), with the default runner resolved from the shared `[loop]` config via `agm.agent.config.default_agent_runner`. AgL adds only its own dispatch/typed-request layer (`runtime/agents.py`) over that runner.
- **Primitives** come from `agm.core`: shell `exec` and agent subprocesses use `core.process`, environments use `core.env`, file and trace I/O use `core.fs`/`core.log` (so AgL participates in dry-run for free), and generic helpers (`util.text`, `util.graph`) are reused for newline normalization and SCC computation (module-cycle detection and the type-table's finiteness/schema-planning analyses).
- **Configuration** is loaded and layered by `agm.config`. The program's own engine settings are `builtin var` bindings in the standard-library module `std/config`; the evaluator reads and writes them through `IrBuiltinLoad`/`IrBuiltinStore`, backing runtime-live settings with live interpreter fields and host-consumed settings with registers that reconfigure the live host service on write (see [repl.md](agl/repl.md)). The `exec`/`repl` commands seed the initial values from the CLI and config-file layers using shared helpers (`core.log`); a source write overrides them from its program point onward.

## Expression-Oriented Design

AgL has no separate statement category. Every construct — bindings, assignment, `print`, loops, `if` without `else` — is an expression with a type, and a block yields the value of its last item. Built-ins such as `print`, `exec`, and `ask` are ordinary calls classified during resolution rather than special syntax. This uniformity is why the AST has a single call node and why the type system carries a unit type for side-effecting expressions.

## Programs and Modules

A **program** is the entry module together with its transitive imports. Unless the host disables it, every loaded entry and library module except `std/core` itself receives the automatic `std/core` open-import prelude. The production pipeline always loads that program and runs program-level scope, typecheck, match compilation, and lowering passes. A **module** is one compilation unit within the program; the corresponding module passes are workers used by those program passes and useful as white-box test seams. `ModuleGraph` remains the loader's data structure. Module loading and program passes are described in [modules.md](agl/modules.md).

## Package Map

| Stage | Package |
|---|---|
| Lexer | `src/agm/agl/lexer/` |
| Parser / grammar | `src/agm/agl/parser/`, `src/agm/agl/grammar/` |
| AST | `src/agm/agl/syntax/` |
| Scope / name resolution | `src/agm/agl/scope/` |
| Type checking | `src/agm/agl/typecheck/` |
| Pattern-match compilation, artifacts, and diagnostics | `src/agm/agl/matchcompile/` |
| Semantic foundation (values, types, exceptions) | `src/agm/agl/semantics/` |
| Lowering / linking | `src/agm/agl/lower/` |
| Execution IR (data model) | `src/agm/agl/ir/` |
| Evaluator | `src/agm/agl/eval/` |
| Host runtime services | `src/agm/agl/runtime/` |
| Module loading | `src/agm/agl/modules/` |
| REPL | `src/agm/agl/repl/` |
| Pipeline orchestrator | `src/agm/agl/pipeline.py` |

Package layering is enforced by a dependency-contract test (`tests/test_agl_dependencies.py`): `semantics` is the semantic foundation layer, the IR depends only on its own data, module ids, and the pure shared engine-key catalog, the evaluator never imports the frontend, the runtime is eval-free, and the pipeline sits on top.

## What To Read Next

- Read [frontend/index.md](agl/frontend/index.md) for the static passes — lexer, parser, AST, scope, typecheck, and match compilation.
- Read [execution/index.md](agl/execution/index.md) for lowering, the IR, the evaluator, value rendering, and the host runtime.
- Read [modules.md](agl/modules.md) for the file-based module system and the program-level passes.
- Read [repl.md](agl/repl.md) for the incremental REPL session, `agm exec` parameter/agent wiring, and engine settings.

The language grammar and surface syntax are documented from the user's perspective in the AgL reference (`docs/agl/reference/grammar.md` and `docs/agl/reference/lexical-structure.md`). The implementation-level token contract — the canonical token-type names and the lexer's merge/disambiguation passes — lives in `src/agm/agl/lexer/tokens.py` (declared the single source of truth) and the pass docstrings in `src/agm/agl/lexer/lexer.py`.
