# AgL Evaluator

The evaluator interprets the linked program and nothing else — it never imports the frontend. Its frame stack holds `let`/param bindings directly and `var` bindings in shared cells, but binding is never copying: an array or dict value is a reference to a mutable payload (`ArrayValue`/`DictValue` in `semantics/values.py`), so every slot, field, capture, or iterator holding that reference observes an in-place indexed-assignment mutation. Records, enums, and exceptions are themselves immutable (no field assignment) but are likewise held and passed by reference. The base frame is module scope, and function frames hold parameters and captured lexical bindings. Programs run under a pinned decimal arithmetic context so results never depend on the host's ambient precision.

Because binding is by reference, an indexed assignment can close a genuine reference cycle through an array or dict (never through a record/enum/exception alone — their fields are fixed at construction). `render_value`, `value_to_json_obj`, and the extern encoder each walk a value's containers and must not recurse forever, so they share one active-container-id guard (`semantics/cycles.py`), converted at each call site into the catchable `CyclicValueError`. The interpreter routes template, print, and explicit-render calls through one render-or-raise helper; conversion and FFI boundaries retain their distinct failure policies. Equality takes the opposite approach: `values_equal` (`semantics/values.py`) is co-inductive — a container pair already being compared is assumed equal — so `==` on a cyclic value terminates and never raises.

## Control Flow

`break`/`continue`/`return` propagate as internal Python signals caught only by their owning construct (the loop, or the function-call boundary for `return`), so they unwind naturally through `try`/`catch`, which catches only AgL-level raises. An `IrCase` evaluates its subject once and selects an arm by enum variant or literal key; a switch with no matching key and no default is malformed IR (`InvalidIrError`) — the evaluator never synthesizes `MatchError`.

Recursion is bounded by a `max_call_depth` guard that raises a catchable `RecursionError`. Because the tree-walker spans several Python frames per AgL call, `run()` raises Python's recursion limit so the AgL guard is reached first; a Python `RecursionError` that still escapes (the limit is capped) is converted to the same catchable AgL exception (`ir_interpreter.py`).

An `IrField` projection carries either exact nominal identity or a static upper-bound mode. The evaluator enforces identity only for exact projections; it does not name or implement any specific nominal hierarchy.

## Host-Backed Operations

Host operations are dispatched by contract identity:

- **Agents.** `ask` issues the call through the host agent runtime; the output is shaped by the contract's format metadata and the schema/decode descriptors compiled into it. A unit contract dispatches once and discards the response.
- **Shell.** `exec` either returns a structured result or parses stdout into a target type, as selected during checking. A structured contract exposes a nonzero exit as data, while spawn failures and timeouts raise `ExecError`; a unit contract also raises for a nonzero exit and discards successful stdout without resolving a codec. The structured result's nominal identity is carried directly on `IrExec` (not derived from the typeless contract), so a scoped `builtin record ExecResult` is tagged with its own declaration's scope path rather than the root canonical one.
- **Conversions.** Casts and `parse_json` execute pre-resolved typeless recipes and always parse strictly; agent and `exec` output parsing uses the configurable strict/lenient codec pipeline.
- **Copying.** `copy` and `shallow_copy` dispatch to `semantics/copying.py`, the single shared implementation of both walks. `shallow_copy` rebuilds one container level only. `copy` recurses with an `id()`-keyed memo, so it preserves diamond sharing and is the one deep, structure-rebuilding walk in the language that terminates on a reference cycle instead of raising `CyclicValueError` — the memo resolves a re-entrant reference to the copy already under construction.

## Extern (Python FFI) Dispatch

Every callable lives in one `functions` table; a descriptor's `impl` is either an AgL body or an extern implementation carrying a compiled boundary contract — per-parameter encode recipes and a strict return decode, with seal/unseal markers enforcing parametricity for type-variable leaves. An extern call is delegated to the runtime's extern registry, which encodes the arguments, invokes the resolved Python callable, and strictly decodes its result; every failure crossing the boundary becomes the catchable `ExternError`, except a cyclic argument (encode-time, or a companion repr'ing a sealed handle), which becomes `CyclicValueError` instead. The registry also owns companion loading — resolving and importing a module's companion `.py` after all static passes succeed and before evaluation starts, so a broken companion is a load-time diagnostic (`--dry-run` stops before import). A host capability flag gates the FFI the same way `supports_shell_exec` gates `exec`. A sealed handle's equality/hash key uses object identity for an array or dict payload (never a structural walk), so it stays cheap and terminates regardless of size or cycles.

## Code Entry Points

- `src/agm/agl/eval/` — the interpreter, frame model, host dispatch, and conversion execution.
- `src/agm/agl/semantics/copying.py` — the shared `copy`/`shallow_copy` value walks.
- `src/agm/agl/runtime/externs.py` — the extern registry and companion loading.
- Tests: `tests/test_agl_ir_*.py` (the IR semantics suite), `tests/test_agl_convert.py`, `tests/test_agl_extern_*.py`.
