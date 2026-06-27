# AgL Language Implementation

AgL is the statically typed workflow language that AGM programs are written in. Its programs orchestrate agents and shell commands as ordinary typed expressions. The implementation is a conventional compiler frontend followed by a typeless execution layer and a host runtime, exposed through two commands — `agm exec` (run a whole program) and `agm repl` (evaluate incrementally).

This document gives the shape of the AgL subsystem. Read the focused documents below for each part.

## Compilation Pipeline

Every AgL program flows through one pipeline, whether run as a whole or one REPL entry at a time:

```
source (.agl)
  → lexer        (INDENT/DEDENT, string interpolation, one case-neutral NAME token)
  → parser       (Lark LALR grammar)
  → AST          (plain dataclasses — the firewall)
  → scope        (name resolution; full static pass)
  → typecheck    (full static pass; selects concrete operations)
  → lower + link (closed, typeless executable program)
  → IR eval      (interpreter over the linked program)
        ↘ host runtime: agents, shell execution, codecs, rendering, trace store
```

The linked IR is the only execution format; checked frontend objects are never fed to the evaluator.

## The Firewall

The lexer and parser are the only Lark-aware code. The AST is the firewall: every pass from scope onward depends solely on the AST dataclasses, never on the parser's types. This is what makes the front end replaceable without touching the static passes or the evaluator.

Two consequences shape the whole codebase:

- **Identifier case is meaningless.** The lexer emits a single `NAME` token for all identifiers; types, constructors, and variables are distinguished by their declaration and binding namespace, never by spelling. No pass branches on capitalization.
- **Passes never mutate the AST.** Later passes attach their results in *side tables* keyed by a stable per-node id, carried in the resolved/checked program objects rather than written back onto nodes.

## Expression-Oriented Design

AgL has no separate statement category. Every construct — bindings, assignment, `print`, loops, `if` without `else` — is an expression with a type, and a block yields the value of its last item. Built-ins such as `print`, `exec`, and `ask` are ordinary calls classified during resolution rather than special syntax. This uniformity is why the AST has a single call node and why the type system carries a unit type for side-effecting expressions.

## Single-File and Graph Modes

AgL supports file-based modules. A single-file program runs through the original single-module passes; a program with imports runs through graph-aware variants of scope, typecheck, and lowering that operate over the whole module graph at once. The two modes share the same AST, value model, and evaluator. Module loading and the graph passes are described in [modules.md](modules.md).

## Package Map

| Stage | Package |
|---|---|
| Lexer | `src/agm/agl/lexer/` |
| Parser / grammar | `src/agm/agl/parser/`, `src/agm/agl/grammar/` |
| AST | `src/agm/agl/syntax/` |
| Scope / name resolution | `src/agm/agl/scope/` |
| Type checking | `src/agm/agl/typecheck/` |
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
- Read [repl.md](repl.md) for the incremental REPL session, `agm exec` parameter/agent wiring, and config pragmas.

The language grammar and surface syntax are documented separately in `docs/agl-grammar.md`.
