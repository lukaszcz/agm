# AgL Execution

Execution is everything after successful match compilation: lowering the match-compiled artifact into a closed, typeless program, validating that IR, evaluating it, and the host runtime that backs agents, shell calls, codecs, and rendering. Frontend artifacts never reach the evaluator — lowering is the boundary, and no checked-only public lowering path exists. See [index.md](index.md) for the surrounding pipeline.

## Lowering and Linking

Lowering consumes a `MatchCompiledProgram` or `MatchCompiledModuleGraph` and emits one linked executable program. The artifact carries the exact checked program plus the complete decision DAG mapping required for source cases, so there is no parallel checked-only lowering route. Lowering performs expected-type-directed translation of expressions and allocates stable program-local identities (symbols, functions, contracts, sources, nominal types), linking modules in dependency order. Top-level function closures are initialized before ordinary module initializers so forward references and hoisting work. All type arguments are erased here; nominal identity is preserved as a module-qualified identity. Built-in prelude records/enums and exceptions are registered as ordinary nominal descriptors, so values produced by host operations and values constructed or referenced directly in source share the same IR tables. Record/enum field and variant shapes, and exception field shapes, are all resolved through the shared `TypeTable` built during checking (constructor construction, nominal descriptors, coercions, and contract/param schema derivation) rather than through any embedded map on the handle. Custom codec contracts are also materialized before this boundary when checker types are still available; only their resulting typeless schema/decode/instruction payload is embedded in IR. The reusable lowering entry points expose an optional validation flag for lower-level callers; the production pipeline and REPL enable it and complete deep IR validation before evaluation.

The point of lowering is to make the evaluator simple: every decision that needed type information (which built-in path, which codec, which decode schema, which conversion) is resolved now and baked into typeless descriptors, so the evaluator only interprets closed nodes. It reads the checker's concrete parameter-type and argument-routing side tables rather than inferring or binding calls again; solver-owned flexible state never crosses this boundary. Partial call expressions are lowered here into ordinary closure descriptors with no dedicated IR node: eager callee/non-hole argument evaluation becomes temporary bindings captured by value, and the synthesized closure body reuses the normal direct, indirect, or constructor call IR.

Source cases consume their match-compiled decision DAGs directly. Lowering always binds the
scrutinee once to a private immutable symbol, then translates each decision switch to a one-level
`IrCase` over an ordinary load. Enum arms bind only the immediate declaration fields demanded by
their child decision; nested tests are further one-level cases over those symbols. Leaves initialize
source binders from dominated occurrence loads before lowering the selected source body. Decision
identity is memoized per source case, so shared compiler nodes remain shared IR objects.

## Execution IR

The IR is a runtime-neutral data model: program-local identities and source locations, a closed family of expression nodes (constants and construction, binding/load/assignment, arithmetic and comparison, control flow and matching, closures and calls, conversions, and host operations), and the program container holding modules, symbols, functions, sources, nominals, contracts, and the dry-run inventory. Host operations carry contract requests — codec selection, target type label/kind, format instructions, JSON schema, and a decode walk — compiled from checker types during lowering. Requests are otherwise typeless, except for an opaque checked target type retained solely for compatibility with legacy custom codecs whose `parse(raw, target_type, ...)` hook still expects it; evaluator code does not inspect that object. For a recursive target type, the decode walk mirrors the JSON Schema's own `$defs`/`$ref` shape: a `RefDecode` node plus a `defs` table of one entry per recursive instantiation, keyed identically to the schema's `$defs` keys and built from the same instantiation-graph plan (`type_schema.py`), so the decode walk terminates on a finite value exactly where the schema closes. `validate_ir` provides cheap node-local and deep whole-program tiers. Validation runs only when a caller requests it at the lowering boundary; every production pipeline and REPL lowering call requests deep validation before handing the program to the evaluator, while lower-level API callers may opt out.

`IrCase(subject, arms, default)` is a typeless one-level switch. An arm key is either a nominal
enum variant or an IR-owned canonical scalar literal; numeric keys use the runtime's int/decimal
equality convention. Enum arms carry named immediate-field bindings, while defaults represent
compiled remainder edges. Deep validation checks key families, runtime-semantic uniqueness,
nominal and field metadata, private immutable temporary symbols, and the complete expression DAG.
It validates shared nodes by identity while rejecting cycles; nodes reached under different enum-payload bindings are also checked in each dominance context.

### Single-Loop Primitive and Desugar

The IR has exactly one loop kind: `IrLoop(body, guarded)`, which repeats `body` unconditionally. The loop-local exits are `IrBreak` (leave the loop, yielding unit) and `IrContinue` (start the next iteration). Function-level `IrReturn` may also unwind through a loop. These signals propagate through `IrTry` bodies — which catch only `AglRaise` — so `break`/`continue`/`return` inside a `try` block are not caught by the `try`. The loop evaluator is a simple `while True:` that catches only `_BreakSignal`/`_ContinueSignal` (internal Python exceptions, not `AglRaise`).

All richer loop features are desugared by the lowerer (`lower/lowerer.py` → `_lower_loop`) before the evaluator sees them:

- **`for EXPR` clause (collection)**: before the `IrLoop`, emits `IrBind(__it, IrIterInit(kind, lower(EXPR)))` where `kind` is `IterKind.LIST`, `DICT_KEYS`, or `TEXT` (chosen from the element type recorded by the typechecker). Inside the loop body, item 1 checks `IrIterHasNext(__it)` and breaks if false; item 2 binds the loop variable to `IrIterNext(__it)`.
- **`for EXPR to/downto BOUND [by STEP]` clause (range)**: does **not** use iterator ops. Before the `IrLoop`, emits three synthetic pre-loop bindings (mutable `__cur` = start, immutable `__end` = bound, immutable `__step` = step or `1`), followed by a step guard `IrRaise(IrMakeException(RangeError, …))` when `__step ≤ 0`. Inside the loop body, item 1 is the range termination check (`__cur > __end` for `to`, `__cur < __end` for `downto`) → `IrBreak`; item 2 binds the loop variable to `__cur` then advances `__cur` by `±__step`. The range never materializes — it uses O(1) memory for any range size. No new IR node is introduced; the desugar reuses `IrBind`, `IrAssign`, `IrArith`, `IrCompare`, `IrIf`, `IrBreak`, `IrRaise`, and `IrMakeException`.
- **`while COND` guard**: emitted as item 3: `IrIf(NOT lower(COND)) => IrBreak`, after the for-variable bind (if present) and before the bound check.
- **`[n]` bound**: emitted as two synthetic pre-loop bindings (`__n` = bound expression evaluated once, `__count` = 0 mutable counter) in an enclosing `IrSequence`. Item 4 inside the loop body checks the bound and raises `MaxIterationsExceeded` or breaks (for non-positive bounds); item 5 increments `__count`. No bound → no counter, no comparison, no exception node.
- **`until E`**: appended as the final item: `IrIf(lower(E)) => IrBreak`. Absent (for `done`/omitted terminators) → no guard.

`MaxIterationsExceeded` and `RangeError` are both constructed as standard `IrMakeException` nodes, not special IR primitives, keeping `IrLoop` a pure repeat node with no exception-raising logic of its own. The three iterator IR nodes (`IrIterInit`, `IrIterHasNext`, `IrIterNext`) are also fully typeless; the `IterKind` discriminant (an enum on `IrIterInit`) is the only runtime clue about collection shape.

The `guarded` flag marks self-bounded loops — those with a `[n]` bound (which raises `MaxIterationsExceeded` itself) or a `for` clause (bounded by a finite collection). The host's global `max-iters` safety valve (resolved from `--max-iters` / `[exec] max-iters` / a source `std.config::max-iters` write; off by default) applies **only to unguarded loops** (`guarded=False`), capping them at `max-iters` body executions. A self-bounded loop is never cut short by the host safety net, which exists solely to catch runaway unbounded `while`/`do…until` loops.

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
| `Case` | root `IrSequence(IrBind(scrutinee, …), decision)`; decisions are nested one-level `IrCase` nodes |

## Evaluator

The evaluator interprets the linked program and nothing else. Its frame stack holds immutable bindings by value and mutable bindings in shared cells; the base frame is module scope, and function frames hold parameters and captured lexical bindings. Programs run under a pinned decimal arithmetic context so results never depend on the host's ambient precision.

An `IrCase` evaluates its subject once, selects an enum key by nominal plus variant or a literal key
with runtime value equality, copies selected enum fields into the arm's declared symbols, and then
evaluates the arm body or default. A switch with neither a matching key nor a default is malformed
IR and raises `InvalidIrError`; case evaluation never synthesizes `MatchError`. Explicit source
construction, raising, and catching of the prelude `MatchError` remain ordinary exception behavior.

`IrReturn` evaluates its value and raises an internal `_ReturnSignal`; the function-call boundary catches that signal and uses its payload as the call result. No other evaluator site catches it, so it unwinds through loops and `try`/`catch` naturally.

`IrIterInit` materializes an `IteratorValue` — a mutable cursor over a pre-built element list — that is stored as a regular immutable binding (the cell holds the same object throughout the loop). `IrIterHasNext` tests the position, and `IrIterNext` reads the current element and advances the position in place. `IteratorValue` is intentionally not a user-visible value type: it cannot be printed, serialized, or returned from a function.

Host-backed operations are dispatched by contract identity:

- **Agents.** An `ask` call extracts the target agent value (or the default agent) and issues the call through the host agent runtime, receiving output shaped by the contract's format metadata and any codec-provided schema. Built-in codecs consume the typeless schema/decode descriptors compiled into the contract; custom host codecs consume the same typeless payload, with any target-type inspection performed before lowering.
- **Shell.** `exec` either returns a structured result built from the subprocess output (a nonzero exit does not raise) or parses stdout into a target type (raising on nonzero exit or parse failure), as selected during checking.
- **Conversions.** Casts and `parse_json` are pre-resolved into typeless conversion recipes carrying a decode schema (and its `defs` table, for a recursive target), executed against strict leaf parsers. Casts and `parse_json` always parse strictly; agent- and `exec`-output parsing uses the configurable strict/lenient codec pipeline.

## Extern (Python FFI) Dispatch

`ExecutableProgram` carries a single `functions` table for every callable. Each `FunctionDescriptor` has an `impl` variant: either an AgL body for ordinary functions or an extern implementation carrying a compiled **boundary contract** (an encode recipe per parameter and a strict decode walk for the return type), built from the checked signature during lowering while checker types are still available, with type-variable leaves compiled to seal/unseal markers. A recursive parameter or result type crosses as a finite `BoundaryRef` graph resolving into the contract's shared `defs` table, built from the same instantiation-graph plan (`type_schema.py`) that backs the codec decode walk's `$defs`/`RefDecode`; the checker rejects a type with no finite schema at the extern use site, exactly as for agent-output and cast targets. The direct- and indirect-call evaluator paths resolve one `function_id` in the table and match on the implementation, delegating to the effects layer for the extern variant rather than interpreting an AgL body.

The effects layer evaluates any unfilled AgL-side defaults, then hands the call to a registry service that mints a fresh seal per type parameter for that call, encodes the arguments per the contract, invokes the resolved Python callable positionally, and strictly decodes its result — unsealing with the same call's tokens. Every failure crossing this boundary (the callable raising, a return-contract violation, an argument-conversion failure) becomes the catchable `ExternError`, following the same host-error-to-`AglRaise` pattern as `exec`.

The registry also owns companion loading: resolving a module's companion path to its imported Python module (cached so a module's companion runs its top-level code at most once) and resolving each declared extern to a callable on it. For real execution this happens after every static pass succeeds and before evaluation starts, so a broken companion is a load-time diagnostic rather than a mid-run surprise; `agm exec --dry-run` stops before companion import to keep dry-run side-effect-free. `agm exec` evaluates the whole program in a single phase, so companion imports and module state remain single-run state. A capability flag gates the Python FFI the same way `supports_shell_exec` gates `exec`.

## Value Rendering

All value display — string interpolation, `print`, `render`, `as text`, and REPL echo — goes through one recursive renderer that produces AgL-native syntax for every value kind, taking `pretty` and `quote_strings` options. Nominal fields are normalized into declaration order once, at construction, so the renderer needs no type information and every consumer (native rendering, `as json`, equality) agrees on field order. Unit values carry a small display flag: explicit `()` renders as `()`, while statement-like effects return `void`, which compares equal to `()` but lets the REPL suppress echo. The renderer depends only on the value model — no semantic types, no parser types.

## Host Runtime and Pipeline

The runtime package is the eval-free services layer: agents, codecs, parameter conversion, host environment assembly, and rendering. It imports neither the evaluator nor the pipeline, which keeps these services reusable and the dependency graph acyclic. Built-in JSON contracts consume the typeless schema/decode data compiled by lowering; custom codecs are materialized through their own `make_contract` hook before lowering and then run from the embedded instructions, schema, and decode data, with compatibility shims for older host codecs that omit `defs` or still accept a parse-time target type. The pipeline orchestrator sits on top, driving the full compile-lower-evaluate sequence and assembling the host environment; it is the public entry point used by `agm exec` and the REPL. Pure compile-time schema and format-instruction generation lives in its own helper so that lowering stays independent of runtime execution.

Programs are parameterized by executable `param` declarations resolved in order at evaluation time, with precedence external value > default expression > error for a required param. The pipeline can discover a program's parameter inventory before execution so a host can wire external values; this drives the `agm exec` option surface described in [repl.md](repl.md).

## Code Entry Points

- `src/agm/agl/lower/` — expected-type-directed lowering and module linking.
- `src/agm/agl/ir/` — the IR data model: identities, nodes, program container, contracts, and the validation gate.
- `src/agm/agl/eval/` — the interpreter, frame model, host dispatch, and conversion execution.
- `src/agm/agl/runtime/` — agents, codecs, parameter conversion, host-environment types, the value renderer, and the extern (Python FFI) registry (`runtime/externs.py`).
- `src/agm/agl/pipeline.py` — the top-of-stack orchestrator; `src/agm/agl/type_schema.py` — compile-time schema/format generation.
- Tests: `tests/test_agl_lower.py`, `tests/test_agl_ir_*.py`, `tests/test_agl_eval*.py` (and the IR semantics suite), `tests/test_agl_runtime.py`, `tests/test_agl_codec.py`, `tests/test_agl_convert.py`, `tests/test_agl_extern_*.py`.
