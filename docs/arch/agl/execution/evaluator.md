# AgL Evaluator

The evaluator interprets the linked program and never imports the frontend. `run()` initializes modules in link order, then invokes the host-selected entry symbol with its pre-evaluated arguments — it never picks a symbol or decodes host values itself — under a pinned decimal context and a recursion boundary. A host may seed a marked static binding by its executable identity; initialization then binds that typed value in place of the binding initializer, preserving ordinary `let` and `var` frame semantics. An entry argument is either an already-evaluated value or a `UseDefault` marker, resolved against the entry function's own default expression exactly as an omitted call argument is; direct calls and the program entry share this same value-taking tail, so a defaulted parameter's default may read module bindings established before the entry runs. Frames hold `let` and function-parameter bindings directly and `var` bindings in shared cells; the base frame is module scope. The evaluator trusts the IR it is handed: it performs no runtime tag or shape checks against a checked program's static invariants, so malformed IR is caught only by `validate_ir`'s self-checks under the self-validation toggle ([testing.md](../../testing.md)), never on the production path.

## Values and Aliasing

Binding is by reference: arrays, dicts, and records are shared, so an in-place update — an index set or a `var` field set — is observed through every alias. Text and exceptions are immutable. An enum-typed slot holds the selected member `RecordValue` directly, never a wrapper. Reference cycles are therefore possible: rendering and JSON encoding carry a cycle guard and raise catchable `CyclicValueError`; equality is co-inductive and terminates; `copy` uses a memo and terminates, while `shallow-copy` rebuilds one level (`semantics/copying.py`).

`DictValue` hides its storage behind an API; no other module touches it directly. It picks one of two representations: a plain `dict[str, Value]` for `text` keys (the hot path), or a `dict[Hashable, tuple[Value, Value]]` keyed by a canonical `key_token`, for every other hashable key kind, storing the originally inserted key alongside the current value. A non-`text` first insert fixes the token representation; an empty dict stays `text`-keyed (undetermined) until then.

Collection iterators retain live array indexing so mutations ahead of the cursor remain visible, while capturing an entry-time length ceiling so structural growth cannot extend a loop. Shrinking an array can exhaust its iterator early. Dict-key iterators materialize their keys once. Text iterators retain the immutable string and wrap code points as they are consumed, sharing the collection cursor without allocating all characters upfront.

## Control Flow

`break`, `continue`, and `return` propagate as internal Python signals caught only by their owning construct, so they unwind through `try`/`catch`, which catches only AgL raises. A specific `catch` matches by nominal conformance (exact identity, then the value's `base` chain on a miss). `IrCase` dispatches on member-record identity or literal key; match compilation proves exhaustiveness statically, so a subject with no matching arm always has a `default` arm to fall through to. Recursion is bounded by `max_call_depth`, raising a catchable `RecursionError`; the Python limit is raised so the AgL guard fires first.

An `AglRaise` created without a span takes the location of the innermost IR node it unwinds through (`_eval`), including across companion callbacks.

## Host-Backed Operations

Effects are dispatched by contract identity through `eval/effects.py`, the seam that composes prompts and records every request and response for tracing:

- **Agents.** An explicit-agent `ask` decodes its `Agent` record to a host spec and opens an ephemeral session spanning the parse-retry loop; a free `ask` uses the snapshotted default session; session nodes drive persistent handles. Output is shaped by the contract's format and decode descriptors; a unit contract dispatches once and discards the response.
- **Shell.** `exec` returns a structured result or parses stdout per its contract. Environment, working directory, and idle timeout come from operands or standard-library bindings; spawn failures and timeouts raise `ExecError`.
- **Conversions.** Casts and `std/value::parse`/`try-parse` execute pre-resolved recipes and parse strictly (strict JSON or AgL value syntax, never lenient recovery), raising `CastError` or `ValueParseError` per their failure mode, with `try-parse` composed from an ordinary `try`/`catch`; agent and `exec` output keep the configurable strict/lenient codec. Nominal casts and `is` share that conformance check (`ir.program.nominal_conforms`), so an exception downcast or `is` accepts descendants. An exception encodes by its runtime nominal's entry in the program's `exception_field_encodes` table (`runtime/serialize.py::encode_value`); only uncaught-error reporting (`report_exception_fields`) degrades a non-data or cyclic field to a marker.
- **Resources** evaluate to their embedded absolute paths. Host-minted values (built-in exceptions, `ExecResult`, `AgentRequest`, `Option` members) take identity from the per-program builtin nominal table so they render and match like source-constructed values.
- **Externs** delegate to the FFI registry ([ffi.md](ffi.md)).

## Code Entry Points

- `src/agm/agl/eval/ir_interpreter.py` — the interpreter, frame model, entry invocation, recursion boundary.
- `src/agm/agl/eval/effects.py` — agent and shell effect handlers; `conversions.py`, `arith.py`, `indexing.py` — cast execution, decimal arithmetic, container access.
- `src/agm/agl/semantics/arithmetic.py` — AgL arithmetic-signal classification and the catchable `ArithmeticError` construction, built on the pinned decimal context and range invariant in `src/agm/util/decimal.py`. Mixed int/decimal comparisons are not widened by lowering: operators, `value_equal` (array membership, FFI view `index`), and `json` equality share `util/decimal.compare_numbers`, which is exact and never raises; an int needle against `decimal` dict keys (`ContainsKind.DICT_INT_NEEDLE`) widens only in range and is otherwise absent.
- `src/agm/agl/semantics/copying.py`, `cycles.py`, `values.py` — copying, cycle guards, value equality.
- Tests: `tests/test_agl_ir_*.py`, `test_agl_convert.py`, `test_agl_copy.py`, `test_agl_recursion_depth.py`.
