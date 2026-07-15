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
  → deep IR validation
  → IR eval      (interpreter over the linked program)
        ↘ host runtime: agents, shell execution, the Python FFI registry, codecs, rendering, trace store
```

The linked IR is the only execution format; checked frontend objects are never fed to the evaluator.

## The Firewall

The lexer and parser are the only Lark-aware code. The AST is the firewall: every pass from scope onward depends solely on the AST dataclasses, never on the parser's types. This is what makes the front end replaceable without touching the static passes or the evaluator.

Two consequences shape the whole codebase:

- **Identifier case carries no semantic category.** Identifiers are case-sensitive (distinct spellings are distinct names), but capitalization never classifies a name. The lexer emits `NAME` for word-starting identifiers and `OP_NAME` for operator-character names; types, constructors, and variables are distinguished by their declaration and binding namespace, not by spelling style. No pass branches on the case of a name.
- **Passes never mutate the AST.** Later passes attach their results in *side tables* keyed by a stable per-node id, carried in the resolved/checked program objects rather than written back onto nodes.

## Shared AGM Layers

The firewall is *semantic*, not an I/O boundary: it isolates the static passes from the parser, not AgL from the rest of AGM. AgL reuses AGM's lower layers rather than reimplementing them — the host runtime and pipeline build on the shared primitives, while AgL-only types sit on top:

- **Agent invocation** goes through `agm.agent.runner` (the same prepare/run path as loop/review), with the default runner resolved from the shared `[loop]` config via `agm.agent.config.default_agent_runner`. AgL adds only its own dispatch/typed-request layer (`runtime/agents.py`) over that runner.
- **Primitives** come from `agm.core`: shell `exec` and agent subprocesses use `core.process`, environments use `core.env`, file and trace I/O use `core.fs`/`core.log` (so AgL participates in dry-run for free), and generic helpers (`util.text`, `util.graph`) are reused for newline normalization and SCC computation (module-cycle detection and the type-table's finiteness/schema-planning analyses).
- **Configuration** is loaded and layered by `agm.config`; the config logic inside `agl/` covers scope validation of `config` declarations, typecheck of their value expressions, and lowering into `IrConfigBind` initializers that the evaluator resolves and applies at runtime. The `exec`/`repl` commands resolve the CLI > source > config precedence using shared helpers (`core.log`).

## Expression-Oriented Design

AgL has no separate statement category. Every construct — bindings, assignment, `print`, loops, `if` without `else` — is an expression with a type, and a block yields the value of its last item. Built-ins such as `print`, `exec`, and `ask` are ordinary calls classified during resolution rather than special syntax. This uniformity is why the AST has a single call node and why the type system carries a unit type for side-effecting expressions.

## Single-File and Graph Modes

AgL supports file-based modules. A single-file program runs through the single-module passes; a program with imports runs through graph-aware variants of scope, typecheck, match compilation, and lowering that operate over the whole module graph at once. The two modes share the same AST, value model, and evaluator. Module loading and the graph passes are described in [modules.md](modules.md).

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

Package layering is enforced by a dependency-contract test (`tests/test_agl_dependencies.py`): `semantics` is the foundation leaf, the IR depends only on its own data and module ids, the evaluator never imports the frontend, the runtime is eval-free, and the pipeline sits on top.

## What To Read Next

- Read [frontend.md](frontend.md) for the lexer, parser, AST, scope, and type system.
- Read [execution.md](execution.md) for lowering, the IR, the evaluator, value rendering, and the host runtime.
- Read [modules.md](modules.md) for the file-based module system and the graph-aware passes.
- Read [repl.md](repl.md) for the incremental REPL session, `agm exec` parameter/agent wiring, and config declarations.

The language grammar and surface syntax are documented from the user's perspective in the AgL reference (`docs/agl/reference/grammar.md` and `docs/agl/reference/lexical-structure.md`). The implementation-level token contract — the canonical token-type names and the lexer's merge/disambiguation passes — lives in `src/agm/agl/lexer/tokens.py` (declared the single source of truth) and the pass docstrings in `src/agm/agl/lexer/lexer.py`.
