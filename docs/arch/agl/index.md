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

The linked IR is the only execution format; checked frontend objects are never fed to the evaluator. Scope emits immutable shared pattern slots for uncertain field-directed names and candidate sets for ambiguous bare `is` variants; typecheck selects their concrete binders or constructors from nominal types, and consumers resolve them through checked-artifact accessors. No pass rewrites another pass's resolution tables.

## The Firewall

The lexer and parser are the only Lark-aware code. The AST is the firewall: every pass from scope onward depends solely on the AST dataclasses, never on the parser's types. This is what makes the front end replaceable without touching the static passes or the evaluator.

Two consequences shape the whole codebase:

- **Identifier case carries no semantic category.** Identifiers are case-sensitive (distinct spellings are distinct names), but capitalization never classifies a name. The lexer emits `NAME` for word-starting identifiers and `OP_NAME` for operator-character names; types, constructors, and variables are distinguished by their declaration and binding namespace, not by spelling style. No pass branches on the case of a name.
- **Passes never mutate the AST.** Later passes attach their results in *side tables* keyed by a stable per-node id, carried in the resolved/checked program objects rather than written back onto nodes.

## Shared AGM Layers

The firewall is *semantic*, not an I/O boundary: it isolates the static passes from the parser, not AgL from the rest of AGM. AgL reuses AGM's lower layers rather than reimplementing them — the host runtime and pipeline build on the shared primitives, while AgL-only types sit on top:

- **Agent invocation** uses `agm.agent.session`: free and explicit asks own persistent or ephemeral conversation handles, while backend-neutral service operations select the CLI adapter or Pi RPC implementation. CLI asks reuse `agm.agent.runner`; `runtime/sessions.py` bridges AgL values and preserves the legacy one-shot dispatcher embedding path.
- **Primitives** come from `agm.core`: shell `exec` and ordinary agent subprocesses use `core.process`, environments use `core.env`, and file and trace I/O use `core.fs`/`core.log`. The streaming Pi RPC backend directly owns its long-lived process and equivalent process-group cleanup. Generic helpers (`util.text`, `util.graph`) are reused for newline normalization and SCC computation (module-cycle detection and the type-table's finiteness/schema-planning analyses).
- **Configuration** is loaded and layered by `agm.config`. The program's own engine settings are `builtin var` bindings in the standard-library module `std/config`; an optional constant initializer supplies the default only when no host seed is present. The checker's constant-expression predicate (`syntax/constants.py`, a syntax-layer leaf) validates a `builtin var` default without reaching outside the frontend. A host-supplied AgL literal (`--agent`/`[exec] default-agent`) is not parsed by a separate pipeline: `agm.agl.setting_overrides.SettingOverride` pairs the source text with an origin label and a `required` flag, and `PipelineDriver.prepare_parsed_entry`'s `setting_overrides` parameter splices it in as the target `builtin var`'s default before scope resolution, so it is resolved, type-checked, and constant-checked by the program's own single compilation, with a rejection diagnostic naming the origin. `pipeline.apply_setting_overrides` is the public seam both `PipelineDriver.prepare_parsed_entry` and `EntryPipeline.load_and_check_program` (REPL) call: it owns the splice-once apply condition and module-cache reconciliation, and decides whether a graph with no loaded `std/config` is silently inert (a non-`required` override, e.g. `[exec] default-agent`) or still a diagnostic (a `required` override, e.g. `--agent` — a per-run request the host cannot silently drop). The evaluator reads and writes settings through `IrBuiltinLoad`/`IrBuiltinStore`, backing runtime-live settings with live interpreter fields and host-consumed settings with registers; `default-agent` is register-only and does not reconfigure a host service (see [repl.md](repl.md)). The `exec`/`repl` commands seed the initial values from the CLI and config-file layers; a source write overrides them from its program point onward. `IrInterpreter.__init__` additionally validates the winning `default-agent` value (whichever of a host seed or a declared/spliced default wins) before any statement runs: an `AgentCommand` whose command text does not shell-split raises `HostConfigurationError`, which `PipelineDriver`/`EntryPipeline` turn into a pre-execution diagnostic rather than the `AgentCallError` an equivalent runtime source write raises only when actually dispatched. `[exec] runner`'s command text is shell-split the same way even earlier, in `cli_support/engine_seeds.py`, before any module loads.

## Expression-Oriented Design

AgL has no separate statement category. Every construct — bindings, assignment, `print`, loops, `if` without `else` — is an expression with a type, and a block yields the value of its last item. Built-ins such as `print`, `exec`, and `ask` are ordinary calls classified during resolution rather than special syntax. A selected builtin method follows the same classification and type-directed method selection, then lowers through its existing builtin route with the receiver operand. The raw-tail `exec!` and `ask!` forms desugar to calls in the parser. This uniformity is why the AST has a single call node and why the type system carries a unit type for side-effecting expressions.

## Programs and Modules

A **program** is the entry module together with its transitive import and re-export dependencies. A `program def` remains an ordinary callable function but is also a host-discoverable entry; `exec` selects an entry-module declaration after the linked modules initialize, within the evaluator's normal execution boundary. Unless the host disables it or a module explicitly imports `std/core` directly or through wildcard expansion, every loaded entry and library module except `std/core` itself receives the automatic `import std/core::*` prelude. The production pipeline always loads that program and runs program-level scope, typecheck, match compilation, and lowering passes. A **module** is one unit within the program. Scope, typecheck, match compilation, and lowering run only as whole-program passes over the graph; their per-module steps are internal workers with no standalone entry point, so no caller — production or test — can run them in a configuration the program passes do not. `ModuleGraph` remains the loader's data structure. Parameter inventories follow each selected program module's transitive import/export subgraph; their descriptors retain module identity for host CLI disambiguation. Module loading and program passes are described in [modules.md](modules.md).

## Package Map

| Stage | Package |
|---|---|
| Lexer | `src/agm/agl/lexer/` |
| Parser / grammar | `src/agm/agl/parser/`, `src/agm/agl/grammar/` |
| AST | `src/agm/agl/syntax/` |
| Scope / name resolution | `src/agm/agl/scope/` |
| Type checking | `src/agm/agl/typecheck/` |
| Pattern-match compilation, artifacts, and diagnostics | `src/agm/agl/matchcompile/` |
| Semantic foundation (values, types, exceptions, text literals, cycle detection, value copying) | `src/agm/agl/semantics/` |
| Lowering / linking | `src/agm/agl/lower/` |
| Execution IR (data model) | `src/agm/agl/ir/` |
| Evaluator | `src/agm/agl/eval/` |
| Host runtime services | `src/agm/agl/runtime/` |
| Module loading | `src/agm/agl/modules/` |
| REPL | `src/agm/agl/repl/` |
| Pipeline orchestrator | `src/agm/agl/pipeline.py` |

Package layering is enforced by a dependency-contract test (`tests/test_agl_dependencies.py`): `semantics` is the semantic foundation layer and the single owner of the AgL text-literal surface, `syntax` is a leaf over its own AST nodes, `typecheck` reaches only scope's output, the frontend layers beneath it, and the IR's id leaf (never the pipeline, parser, lowering, or evaluator), the IR depends only on its own data, module ids, and the pure shared engine-key catalog, the evaluator never imports the frontend, the runtime is eval-free, and the pipeline sits on top.

## What To Read Next

- Read [frontend/index.md](frontend/index.md) for the static passes — lexer, parser, AST, scope, typecheck, and match compilation.
- Read [execution/index.md](execution/index.md) for lowering, the IR, the evaluator, value rendering, and the host runtime.
- Read [modules.md](modules.md) for the file-based module system and the program-level passes.
- Read [repl.md](repl.md) for the incremental REPL session, `agm exec` parameter/agent wiring, and engine settings.

The language grammar and surface syntax are documented from the user's perspective in the AgL reference (`docs/agl/reference/grammar.md` and `docs/agl/reference/lexical-structure.md`). The dependency-free canonical keyword inventory lives in `src/agm/agl/keywords.py`; the remaining implementation-level token contract and the lexer's merge/disambiguation passes live in `src/agm/agl/lexer/tokens.py` and the pass docstrings in `src/agm/agl/lexer/lexer.py`.
