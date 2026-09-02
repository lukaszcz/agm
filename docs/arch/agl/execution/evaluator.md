# AgL Evaluator

The evaluator interprets the linked program and never imports the frontend. `run()` installs the parameter inventory (host-provided values first, defaults resolved on demand, dependency cycles reported), initializes modules in link order, then invokes the host-selected entry symbol — it never picks one itself — under a pinned decimal context and a recursion boundary. Frames hold `let` and param bindings directly and `var` bindings in shared cells; the base frame is module scope.

## Values and Aliasing

Binding is by reference: arrays, dicts, and records are shared, so an in-place update — an index set or a `var` field set — is observed through every alias. Text and exceptions are immutable. An enum-typed slot holds the selected member `RecordValue` directly, never a wrapper. Reference cycles are therefore possible: rendering and JSON encoding carry a cycle guard and raise catchable `CyclicValueError`; equality is co-inductive and terminates; `copy` uses a memo and terminates, while `shallow-copy` rebuilds one level (`semantics/copying.py`).

## Control Flow

`break`, `continue`, and `return` propagate as internal Python signals caught only by their owning construct, so they unwind through `try`/`catch`, which catches only AgL raises. `IrCase` dispatches on member-record identity or literal key; a switch with no matching arm is malformed IR, never a runtime match failure. Recursion is bounded by `max_call_depth`, raising a catchable `RecursionError`; the Python limit is raised so the AgL guard fires first.

## Host-Backed Operations

Effects are dispatched by contract identity through `eval/effects.py`, the seam that composes prompts and records every request and response for tracing:

- **Agents.** An explicit-agent `ask` decodes its `Agent` record to a host spec and opens an ephemeral session spanning the parse-retry loop; a free `ask` uses the snapshotted default session; session nodes drive persistent handles. Output is shaped by the contract's format and decode descriptors; a unit contract dispatches once and discards the response.
- **Shell.** `exec` returns a structured result or parses stdout per its contract. Environment, working directory, and idle timeout come from operands or standard-library bindings; spawn failures and timeouts raise `ExecError`.
- **Conversions.** Casts execute pre-resolved recipes and parse strictly; agent and `exec` output keep the configurable strict/lenient codec.
- **Resources** evaluate to their embedded absolute paths. Host-minted values (built-in exceptions, `ExecResult`, `AgentRequest`, `Option` members) take identity from the per-program builtin nominal table so they render and match like source-constructed values.
- **Externs** delegate to the FFI registry ([ffi.md](ffi.md)).

## Code Entry Points

- `src/agm/agl/eval/ir_interpreter.py` — the interpreter, frame model, entry invocation, recursion boundary.
- `src/agm/agl/eval/effects.py` — agent and shell effect handlers; `conversions.py`, `arith.py`, `indexing.py` — cast execution, decimal arithmetic, container access.
- `src/agm/agl/semantics/copying.py`, `cycles.py`, `values.py` — copying, cycle guards, value equality.
- Tests: `tests/test_agl_ir_*.py`, `test_agl_convert.py`, `test_agl_copy.py`, `test_agl_recursion_depth.py`.
