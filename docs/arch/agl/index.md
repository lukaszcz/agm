# AgL Language Implementation

AgL is the statically typed workflow language AGM programs are written in; agent and shell calls are ordinary typed expressions. The implementation is a conventional compiler frontend, a typeless execution layer, and a host runtime, driven by one pipeline orchestrator that `agm exec`, `agm repl`, `agm check`, package validation, and registered package commands all share.

## Compilation Pipeline

Every AgL program flows through one pipeline, whether run whole or one REPL entry at a time:

```
source (.agl)
  → lexer        (INDENT/DEDENT, string and environment interpolation, one NAME token class)
  → parser       (Lark LALR grammar)
  → AST          (frozen dataclasses — the firewall)
  → scope        (whole-program name resolution)
  → typecheck    (whole-program checking; selects concrete operations)
  → match compile (exhaustiveness, redundancy, decision DAGs)
  → lower + link (closed, typeless executable program)
  → IR eval      (interpreter over the linked program)
        ↘ host runtime: agents and sessions, shell, Python FFI, codecs, rendering, tracing
```

The linked IR is the only execution format; checked frontend objects never reach the evaluator. Each pass wraps the previous pass's artifact and records its conclusions in side tables keyed by stable node ids. No pass mutates the AST or rewrites another pass's tables; where scope cannot decide (a field-directed pattern name, an ambiguous bare constructor), it emits a slot or candidate set that typecheck resolves and later passes read through checked-artifact accessors.

## The Firewall

The lexer and parser are the only Lark-aware code. Every pass from scope onward depends solely on the AST dataclasses, so the front end is replaceable without touching the static passes or the evaluator. One consequence shapes the whole codebase: identifier case carries no semantic category. Types, constructors, and variables are told apart by their declaration and binding namespace, never by spelling, and no pass branches on capitalization.

## Shared AGM Layers

The firewall is semantic, not an I/O boundary. AgL reuses AGM's lower layers rather than reimplementing them:

- **Agents** come from `agm.agent`: `Agent` values decode into host specs, asks dispatch through the shared runner or the session service, and `runtime/sessions.py` bridges AgL session values to it ([agents.md](../agents.md)).
- **Primitives** come from `agm.core`: shell `exec` and CLI agent subprocesses use `core.process`, environments `core.env`, files and trace logs `core.fs`/`core.log`, so AgL participates in dry-run for free. `util.graph` and `util.text` supply SCC computation and newline normalization.
- **Configuration** comes from `agm.config`: the engine-key catalog and qualified config tables feed the hosts' seeds ([hosting.md](hosting.md)).

## Expression-Oriented Design

AgL has no statement category. Bindings, assignment, loops, and `if` without `else` are expressions with a type, and a block yields its last item. Built-ins such as `print`, `exec`, and `ask` are ordinary calls classified during resolution; runtime built-in references are first-class, occurrence-specialized function values. A `$` verbatim literal is a template like a quoted string, so it reaches `exec`/`ask` through the same single-argument call sugar as any other template. Methods are selected from reachable declarations by receiver type for nominal and builtin receivers alike; lowering consumes the checker's `method_selections` unchanged. This uniformity is why the AST has a single call node and why the type system carries a unit type.

## Programs and Modules

A **program** is the entry module plus its transitive import and re-export dependencies; the pipeline assembles and validates the graph while reusing precompiled module interfaces and bodies. A `program def` is an ordinary function that is also a host-discoverable entry. Every module except `std/prelude` receives the `std/prelude` prelude unless the host disables it. Module loading, visibility, and the compilation caches are in [modules.md](modules.md).

## Package Map

| Stage | Package |
|---|---|
| Lexer | `src/agm/agl/lexer/` |
| Parser / grammar | `src/agm/agl/parser/`, `src/agm/agl/grammar/` |
| AST | `src/agm/agl/syntax/` |
| Scope / name resolution | `src/agm/agl/scope/` |
| Type checking | `src/agm/agl/typecheck/` |
| Match compilation | `src/agm/agl/matchcompile/` |
| Semantic foundation (values, types, type table, analyses, exceptions) | `src/agm/agl/semantics/` |
| Literal lexical rules and the value-syntax reader | `src/agm/agl/value_syntax/` |
| Lowering / linking | `src/agm/agl/lower/` |
| Execution IR | `src/agm/agl/ir/` |
| Evaluator | `src/agm/agl/eval/` |
| Host runtime services and FFI | `src/agm/agl/runtime/` |
| Module loading | `src/agm/agl/modules/` |
| REPL | `src/agm/agl/repl/` |
| Pipeline orchestrator and host leaves | `src/agm/agl/pipeline.py`, `capabilities.py`, `diagnostics.py`, `type_schema.py`, `artifact_cache.py`, `artifact_storage.py`, `self_validation.py` |

Layering is enforced by `tests/test_agl_dependencies.py`: `semantics` is the foundation, `syntax` is an AST-only leaf, `typecheck` reaches only scope's output and the layers beneath it, `matchcompile` imports nothing downstream, the IR depends only on its own data and the engine-key catalog, the evaluator never imports the frontend, the runtime is eval-free, and the pipeline sits on top. `agl/value_syntax/` is a leaf below the lexer, match compiler, and runtime, holding the literal scanning rules (text escapes, numbers, identifiers, environment holes) and a reader for AgL's data-only value syntax; it imports only `agm.util` and `agl/keywords.py`. `agl/zones.py`, `agl/attributes.py` (the built-in attribute catalog, which names zones and carries the host-facing shapes the `@opt-*` and `@command` attributes describe; it is the one place the language reaches out to an AGM leaf, holding a `@command` path to the CLI's own reserved-name rule) and `agl/modules/ids.py` are the vocabulary leaves below every pass, so both scope and the IR can name a parameter's zone and a module's identity without seeing each other. `artifact_storage.py` is a further leaf: the disk-cache envelope shared by the module cache, artifact serialization, and the runtime's companion bytecode cache, importing nothing under `agm`.

## What To Read Next

- [frontend/index.md](frontend/index.md) — lexer, parser, AST, scope, typecheck, match compilation.
- [execution/index.md](execution/index.md) — lowering, the IR, the evaluator, the host runtime, and the Python FFI.
- [modules.md](modules.md) — the file-based module system, the standard library's surfaces, and the compilation caches.
- [hosting.md](hosting.md) — the pipeline orchestrator, host capabilities, parameters, engine settings, diagnostics.
- [repl.md](repl.md) — the incremental REPL session and its front ends.

The language itself is documented for users in `docs/agl/reference/`; the standard library's sources under `packages/stdlib/src/` are its own reference.
