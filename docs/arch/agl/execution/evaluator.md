# AgL Evaluator

The evaluator interprets the linked program and nothing else — it never imports the frontend. Its frame stack holds immutable bindings by value and mutable bindings in shared cells; the base frame is module scope, and function frames hold parameters and captured lexical bindings. Programs run under a pinned decimal arithmetic context so results never depend on the host's ambient precision.

## Control Flow

`break`/`continue`/`return` propagate as internal Python signals caught only by their owning construct (the loop, or the function-call boundary for `return`), so they unwind naturally through `try`/`catch`, which catches only AgL-level raises. An `IrCase` evaluates its subject once and selects an arm by enum variant or literal key; a switch with no matching key and no default is malformed IR (`InvalidIrError`) — the evaluator never synthesizes `MatchError`.

## Host-Backed Operations

Host operations are dispatched by contract identity:

- **Agents.** `ask` issues the call through the host agent runtime; the output is shaped by the contract's format metadata and the schema/decode descriptors compiled into it.
- **Shell.** `exec` either returns a structured result or parses stdout into a target type, as selected during checking.
- **Conversions.** Casts and `parse_json` execute pre-resolved typeless recipes and always parse strictly; agent and `exec` output parsing uses the configurable strict/lenient codec pipeline.

## Extern (Python FFI) Dispatch

Every callable lives in one `functions` table; a descriptor's `impl` is either an AgL body or an extern implementation carrying a compiled boundary contract — per-parameter encode recipes and a strict return decode, with seal/unseal markers enforcing parametricity for type-variable leaves. An extern call is delegated to the runtime's extern registry, which encodes the arguments, invokes the resolved Python callable, and strictly decodes its result; every failure crossing the boundary becomes the catchable `ExternError`. The registry also owns companion loading — resolving and importing a module's companion `.py` after all static passes succeed and before evaluation starts, so a broken companion is a load-time diagnostic (`--dry-run` stops before import). A host capability flag gates the FFI the same way `supports_shell_exec` gates `exec`.

## Code Entry Points

- `src/agm/agl/eval/` — the interpreter, frame model, host dispatch, and conversion execution.
- `src/agm/agl/runtime/externs.py` — the extern registry and companion loading.
- Tests: `tests/test_agl_ir_*.py` (the IR semantics suite), `tests/test_agl_convert.py`, `tests/test_agl_extern_*.py`.
