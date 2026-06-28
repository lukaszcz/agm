# AgL Execution

Execution is everything after type checking: lowering the checked AST into a closed, typeless program, evaluating it, and the host runtime that backs agents, shell calls, codecs, and rendering. The checked frontend objects never reach the evaluator — lowering is the boundary. See [index.md](index.md) for the surrounding pipeline.

## Lowering and Linking

Lowering consumes the checked program (a single module, or a whole module graph) and emits one linked executable program. It performs expected-type-directed translation of expressions and allocates stable program-local identities (symbols, functions, contracts, sources, nominal types), linking modules in dependency order. Top-level function closures are initialized before ordinary module initializers so forward references and hoisting work. All type arguments are erased here; nominal identity is preserved as a module-qualified identity.

The point of lowering is to make the evaluator simple: every decision that needed type information (which built-in path, which codec, which decode schema, which conversion) is resolved now and baked into typeless descriptors, so the evaluator only interprets closed nodes.

## Execution IR

The IR is a runtime-neutral data model: program-local identities and source locations, a closed family of expression nodes (constants and construction, binding/load/assignment, arithmetic and comparison, control flow and matching, closures and calls, conversions, and host operations), and the program container holding modules, symbols, functions, sources, nominals, contracts, and the dry-run inventory. Host operations carry typeless contract requests — codec selection, format instructions, JSON schema, and a decode walk — compiled from checker types during lowering. A structural validation gate exists for the IR but runs only when explicitly requested, not in production evaluation.

### Single-Loop Primitive and Desugar

The IR has exactly one loop kind: `IrLoop(body)`, which repeats `body` unconditionally. The only loop exits are `IrBreak` (leave the loop, yielding unit) and `IrContinue` (start the next iteration). Both propagate through `IrTry` bodies — which catch only `AglRaise` — so a `break`/`continue` inside a `try` block exits the loop, not the `try`. The loop evaluator is a simple `while True:` that catches `_BreakSignal`/`_ContinueSignal` (internal Python exceptions, not `AglRaise`).

All richer loop features are desugared by the lowerer (`lower/lowerer.py` → `_lower_loop`) before the evaluator sees them:

- **`[n]` bound**: emitted as two synthetic pre-loop bindings (`__n` = bound expression evaluated once, `__count` = 0 mutable counter) in an enclosing `IrSequence`. Inside the loop body: (4) a bound-check `IrIf(__count >= __n)` that either breaks (if `__count == 0`, for zero-or-negative bounds) or raises `MaxIterationsExceeded` via a fully desugared `IrRaise(IrMakeException(...))`, then (5) `__count += 1`. No bound → no counter, no comparison, no exception node.
- **`until E`**: appended as an `IrIf(lower(E)) => IrBreak` at the end of the loop body. Absent (for `done`/omitted terminators) → no guard; the bound's raise is the only exit.
- **`for`/`while` clauses**: not yet desugared; `for`/`while` loop clause lowering is not yet implemented.

`MaxIterationsExceeded` is constructed as a standard `IrMakeException` node, not a special IR primitive, keeping `IrLoop` a pure repeat node with no exception-raising logic of its own.

### AST→IR Coverage

| AST node | IR output |
|---|---|
| `Loop` (no bound, `until E`) | `IrLoop(body=[lower(body), IrIf(E)=>IrBreak])` |
| `Loop` (bound `n`, `until E`) | `IrSequence(IrBind(__n), IrBind(__count), IrLoop(body=[bound_check, count_incr, lower(body), IrIf(E)=>IrBreak]))` |
| `Loop` (bound `n`, done/omitted) | same but no `IrIf(E)` at end |
| `Loop` (no bound, done/omitted) | `IrLoop(body=[lower(body)])` — infinite unless `break` |
| `Break` | `IrBreak` |
| `Continue` | `IrContinue` |

## Evaluator

The evaluator interprets the linked program and nothing else. Its frame stack holds immutable bindings by value and mutable bindings in shared cells; the base frame is module scope, and function frames hold parameters and captured lexical bindings. Programs run under a pinned decimal arithmetic context so results never depend on the host's ambient precision.

Host-backed operations are dispatched by contract identity:

- **Agents.** An `ask` call extracts the target agent value (or the default agent) and issues the call through the host agent runtime, receiving output shaped by the contract's schema and format metadata.
- **Shell.** `exec` either returns a structured result built from the subprocess output (a nonzero exit does not raise) or parses stdout into a target type (raising on nonzero exit or parse failure), as selected during checking.
- **Conversions.** Casts and `parse_json` are pre-resolved into typeless conversion recipes carrying a decode schema, executed against strict leaf parsers. Casts and `parse_json` always parse strictly; agent- and `exec`-output parsing uses the configurable strict/lenient codec pipeline.

## Value Rendering

All value display — string interpolation, `print`, `render`, `as text`, and REPL echo — goes through one recursive renderer that produces AgL-native syntax for every value kind, taking `pretty` and `quote_strings` options. Nominal fields are normalized into declaration order once, at construction, so the renderer needs no type information and every consumer (native rendering, `as json`, equality) agrees on field order. The renderer depends only on the value model — no semantic types, no parser types.

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
