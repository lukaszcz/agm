# AgL Evaluator

The evaluator interprets the linked program and never imports the frontend. `run()` initializes modules in link order, then invokes the host-selected entry symbol with its pre-evaluated arguments — it never picks a symbol or decodes host values itself — under a pinned decimal context and a recursion boundary. A host may seed a marked static binding by its executable identity; initialization then binds that typed value in place of the binding initializer, preserving ordinary `let` and `var` frame semantics. An entry argument is either an already-evaluated value or a `UseDefault` marker, resolved against the entry function's own default expression exactly as an omitted call argument is; direct calls and the program entry share this same value-taking tail, so a defaulted parameter's default may read module bindings established before the entry runs. A `UseDefault` slot in a record/exception construction (`IrMakeRecord`/`IrMakeException.fields`) is resolved the same way, against the constructed nominal's own `NominalDescriptor.field_defaults` in place of a callee's params — one evaluation per construction, in the declaring module's call context, never cached across constructions. Frames hold `let` and function-parameter bindings directly and `var` bindings in shared cells; the base frame is module scope.

## Values and Aliasing

Binding is by reference: arrays, dicts, and records are shared, so an in-place update — an index set or a `var` field set — is observed through every alias. Text and exceptions are immutable. An enum-typed slot holds the selected member `RecordValue` directly, never a wrapper. Reference cycles are therefore possible: rendering and JSON encoding carry a cycle guard and raise catchable `CyclicValueError`; equality is co-inductive and terminates; `copy` uses a memo and terminates, while `shallow-copy` rebuilds one level (`semantics/copying.py`).

Collection iterators retain live array indexing so mutations ahead of the cursor remain visible, while capturing an entry-time length ceiling so structural growth cannot extend a loop. Shrinking an array can exhaust its iterator early. Dict-key iterators materialize their keys once. Text iterators retain the immutable string and wrap code points as they are consumed, sharing the collection cursor without allocating all characters upfront.

## Control Flow

`break`, `continue`, and `return` propagate as internal Python signals caught only by their owning construct, so they unwind through `try`/`catch`, which catches only AgL raises. A specific `catch` matches by nominal conformance (exact identity, then the value's `base` chain on a miss). `IrCase` dispatches on member-record identity or literal key; a switch with no matching arm is malformed IR, never a runtime match failure. Recursion is bounded by `max_call_depth`, raising a catchable `RecursionError`; the Python limit is raised so the AgL guard fires first.

## Host-Backed Operations

Effects are dispatched by contract identity through `eval/effects.py`, the seam that composes prompts and records every request and response for tracing:

- **Agents.** An explicit-agent `ask` decodes its `Agent` record to a host spec and opens an ephemeral session spanning the parse-retry loop; a free `ask` uses the snapshotted default session; session nodes drive persistent handles. Output is shaped by the contract's format and decode descriptors; a unit contract dispatches once and discards the response.
- **Shell.** `exec` returns a structured result or parses stdout per its contract. Environment, working directory, and idle timeout come from operands or standard-library bindings; spawn failures and timeouts raise `ExecError`. A `sandbox` operand routes the `sh -c` invocation through the shared sandbox library ([sandbox.md](../../sandbox.md)) instead, selecting its profile from the command's first shell word (never `sh`); a preparation failure raises `ExecError` exactly like a spawn failure.
- **Conversions.** Casts and `std/value::parse`/`try-parse` execute pre-resolved recipes and parse strictly (strict JSON or AgL value syntax, never lenient recovery), raising `CastError` or `ValueParseError` per their failure mode, with `try-parse` composed from an ordinary `try`/`catch`; agent and `exec` output keep the configurable strict/lenient codec. Nominal casts and `is` share that conformance check (`ir.program.nominal_conforms`), so an exception downcast or `is` accepts descendants.
- **Resources** evaluate to their embedded absolute paths. Host-minted values (built-in exceptions, `ExecResult`, `AgentRequest`, `Option` members) take identity from the per-program builtin nominal table so they render and match like source-constructed values.
- **Externs** delegate to the FFI registry ([ffi.md](ffi.md)).

## Code Entry Points

- `src/agm/agl/eval/ir_interpreter.py` — the interpreter, frame model, entry invocation, recursion boundary.
- `src/agm/agl/eval/effects.py` — agent and shell effect handlers; `conversions.py`, `arith.py`, `indexing.py` — cast execution, decimal arithmetic, container access.
- `src/agm/agl/semantics/copying.py`, `cycles.py`, `values.py` — copying, cycle guards, value equality.
- Tests: `tests/test_agl_ir_*.py`, `test_agl_convert.py`, `test_agl_copy.py`, `test_agl_recursion_depth.py`.
