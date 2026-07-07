# AgL Execution

Execution is everything after type checking: lowering the checked AST into a closed, typeless program, evaluating it, and the host runtime that backs agents, shell calls, codecs, and rendering. The checked frontend objects never reach the evaluator — lowering is the boundary. See [index.md](index.md) for the surrounding pipeline.

## Lowering and Linking

Lowering consumes the checked program (a single module, or a whole module graph) and emits one linked executable program. It performs expected-type-directed translation of expressions and allocates stable program-local identities (symbols, functions, contracts, sources, nominal types), linking modules in dependency order. Top-level function closures are initialized before ordinary module initializers so forward references and hoisting work. All type arguments are erased here; nominal identity is preserved as a module-qualified identity. Built-in prelude records/enums and exceptions are registered as ordinary nominal descriptors, so values produced by host operations and values constructed or referenced directly in source share the same IR tables.

The point of lowering is to make the evaluator simple: every decision that needed type information (which built-in path, which codec, which decode schema, which conversion) is resolved now and baked into typeless descriptors, so the evaluator only interprets closed nodes. Partial call expressions are lowered here into ordinary closure descriptors with no dedicated IR node: eager callee/non-hole argument evaluation becomes temporary bindings captured by value, and the synthesized closure body reuses the normal direct, indirect, or constructor call IR.

## Execution IR

The IR is a runtime-neutral data model: program-local identities and source locations, a closed family of expression nodes (constants and construction, binding/load/assignment, arithmetic and comparison, control flow and matching, closures and calls, conversions, and host operations), and the program container holding modules, symbols, functions, sources, nominals, contracts, and the dry-run inventory. Host operations carry typeless contract requests — codec selection, format instructions, JSON schema, and a decode walk — compiled from checker types during lowering. A structural validation gate exists for the IR but runs only when explicitly requested, not in production evaluation.

### Single-Loop Primitive and Desugar

The IR has exactly one loop kind: `IrLoop(body, guarded)`, which repeats `body` unconditionally. The loop-local exits are `IrBreak` (leave the loop, yielding unit) and `IrContinue` (start the next iteration). Function-level `IrReturn` may also unwind through a loop. These signals propagate through `IrTry` bodies — which catch only `AglRaise` — so `break`/`continue`/`return` inside a `try` block are not caught by the `try`. The loop evaluator is a simple `while True:` that catches only `_BreakSignal`/`_ContinueSignal` (internal Python exceptions, not `AglRaise`).

All richer loop features are desugared by the lowerer (`lower/lowerer.py` → `_lower_loop`) before the evaluator sees them:

- **`for EXPR` clause (collection)**: before the `IrLoop`, emits `IrBind(__it, IrIterInit(kind, lower(EXPR)))` where `kind` is `IterKind.LIST`, `DICT_KEYS`, or `TEXT` (chosen from the element type recorded by the typechecker). Inside the loop body, item 1 checks `IrIterHasNext(__it)` and breaks if false; item 2 binds the loop variable to `IrIterNext(__it)`.
- **`for EXPR to/downto BOUND [by STEP]` clause (range)**: does **not** use iterator ops. Before the `IrLoop`, emits three synthetic pre-loop bindings (mutable `__cur` = start, immutable `__end` = bound, immutable `__step` = step or `1`), followed by a step guard `IrRaise(IrMakeException(RangeError, …))` when `__step ≤ 0`. Inside the loop body, item 1 is the range termination check (`__cur > __end` for `to`, `__cur < __end` for `downto`) → `IrBreak`; item 2 binds the loop variable to `__cur` then advances `__cur` by `±__step`. The range never materializes — it uses O(1) memory for any range size. No new IR node is introduced; the desugar reuses `IrBind`, `IrAssign`, `IrArith`, `IrCompare`, `IrIf`, `IrBreak`, `IrRaise`, and `IrMakeException`.
- **`while COND` guard**: emitted as item 3: `IrIf(NOT lower(COND)) => IrBreak`, after the for-variable bind (if present) and before the bound check.
- **`[n]` bound**: emitted as two synthetic pre-loop bindings (`__n` = bound expression evaluated once, `__count` = 0 mutable counter) in an enclosing `IrSequence`. Item 4 inside the loop body checks the bound and raises `MaxIterationsExceeded` or breaks (for non-positive bounds); item 5 increments `__count`. No bound → no counter, no comparison, no exception node.
- **`until E`**: appended as the final item: `IrIf(lower(E)) => IrBreak`. Absent (for `done`/omitted terminators) → no guard.

`MaxIterationsExceeded` and `RangeError` are both constructed as standard `IrMakeException` nodes, not special IR primitives, keeping `IrLoop` a pure repeat node with no exception-raising logic of its own. The three iterator IR nodes (`IrIterInit`, `IrIterHasNext`, `IrIterNext`) are also fully typeless; the `IterKind` discriminant (an enum on `IrIterInit`) is the only runtime clue about collection shape.

The `guarded` flag marks self-bounded loops — those with a `[n]` bound (which raises `MaxIterationsExceeded` itself) or a `for` clause (bounded by a finite collection). The host's global `max-iters` safety valve (resolved from `--max-iters` / `[exec] max-iters` / a source `config max-iters` declaration; off by default) applies **only to unguarded loops** (`guarded=False`), capping them at `max-iters` body executions. A self-bounded loop is never cut short by the host safety net, which exists solely to catch runaway unbounded `while`/`do…until` loops.

### AST→IR Coverage

| AST node | IR output |
|---|---|
| `Loop` (no clauses, `until E`) | `IrLoop(body=[lower(body), IrIf(E)=>IrBreak])` |
| `Loop` (bound `n`, `until E`) | `IrSequence(IrBind(__n,…), IrBind(__count,0), IrLoop(body=[bound_check, count_incr, lower(body), IrIf(E)=>IrBreak]))` |
| `Loop` (bound `n`, done/omitted) | same but no `IrIf(E)` at end |
| `Loop` (no bound, done/omitted) | `IrLoop(body=[lower(body)])` — infinite unless `break` |
| `Loop` (for clause, collection) | prefix `IrBind(__it, IrIterInit(kind, …))`, items 1–2 inside body |
| `Loop` (for clause, range `to`) | pre-loop `IrBind(__cur/end/step)` + step guard; body item 1: `IrIf(__cur > __end)=>IrBreak`; item 2: bind loop var, `__cur := __cur + __step` |
| `Loop` (for clause, range `downto`) | same shape; body item 1: `IrIf(__cur < __end)=>IrBreak`; item 2: `__cur := __cur - __step` |
| `Loop` (while clause) | item 3 inside body: `IrIf(NOT lower(while_cond))=>IrBreak` |
| `Break` | `IrBreak` |
| `Continue` | `IrContinue` |
| `Return` | `IrReturn` |

## Evaluator

The evaluator interprets the linked program and nothing else. Its frame stack holds immutable bindings by value and mutable bindings in shared cells; the base frame is module scope, and function frames hold parameters and captured lexical bindings. Programs run under a pinned decimal arithmetic context so results never depend on the host's ambient precision.

`IrReturn` evaluates its value and raises an internal `_ReturnSignal`; the function-call boundary catches that signal and uses its payload as the call result. No other evaluator site catches it, so it unwinds through loops and `try`/`catch` naturally.

`IrIterInit` materializes an `IteratorValue` — a mutable cursor over a pre-built element list — that is stored as a regular immutable binding (the cell holds the same object throughout the loop). `IrIterHasNext` tests the position, and `IrIterNext` reads the current element and advances the position in place. `IteratorValue` is intentionally not a user-visible value type: it cannot be printed, serialized, or returned from a function.

Host-backed operations are dispatched by contract identity:

- **Agents.** An `ask` call extracts the target agent value (or the default agent) and issues the call through the host agent runtime, receiving output shaped by the contract's schema and format metadata.
- **Shell.** `exec` either returns a structured result built from the subprocess output (a nonzero exit does not raise) or parses stdout into a target type (raising on nonzero exit or parse failure), as selected during checking.
- **Conversions.** Casts and `parse_json` are pre-resolved into typeless conversion recipes carrying a decode schema, executed against strict leaf parsers. Casts and `parse_json` always parse strictly; agent- and `exec`-output parsing uses the configurable strict/lenient codec pipeline.

## Value Rendering

All value display — string interpolation, `print`, `render`, `as text`, and REPL echo — goes through one recursive renderer that produces AgL-native syntax for every value kind, taking `pretty` and `quote_strings` options. Nominal fields are normalized into declaration order once, at construction, so the renderer needs no type information and every consumer (native rendering, `as json`, equality) agrees on field order. Unit values carry a small display flag: explicit `()` renders as `()`, while statement-like effects return `void`, which compares equal to `()` but lets the REPL suppress echo. The renderer depends only on the value model — no semantic types, no parser types.

## Host Runtime and Pipeline

The runtime package is the eval-free services layer: agents, codecs, parameter conversion, host environment assembly, and rendering. It imports neither the evaluator nor the pipeline, which keeps these services reusable and the dependency graph acyclic. The pipeline orchestrator sits on top, driving the full compile-lower-evaluate sequence and assembling the host environment; it is the public entry point used by `agm exec` and the REPL. Pure compile-time schema and format-instruction generation lives in its own helper so that lowering stays independent of runtime execution.

Programs are parameterized by executable `param` declarations resolved in order at evaluation time, with precedence external value > default expression > error for a required param. The pipeline can discover a program's parameter inventory before execution so a host can wire external values; this drives the `agm exec` option surface described in [repl.md](repl.md).

## Code Entry Points

- `src/agm/agl/lower/` — expected-type-directed lowering and module linking.
- `src/agm/agl/ir/` — the IR data model: identities, nodes, program container, contracts, and the validation gate.
- `src/agm/agl/eval/` — the interpreter, frame model, host dispatch, and conversion execution.
- `src/agm/agl/runtime/` — agents, codecs, parameter conversion, host-environment types, and the value renderer.
- `src/agm/agl/pipeline.py` — the top-of-stack orchestrator; `src/agm/agl/type_schema.py` — compile-time schema/format generation.
- Tests: `tests/test_agl_lower.py`, `tests/test_agl_ir_*.py`, `tests/test_agl_eval*.py` (and the IR semantics suite), `tests/test_agl_runtime.py`, `tests/test_agl_codec.py`, `tests/test_agl_convert.py`.
