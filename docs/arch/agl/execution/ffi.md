# AgL Python FFI

An `extern def` declares a function implemented in a Python *companion* file that sits beside its module with a `.py` suffix. The FFI is split between `runtime/externs.py` — companion loading, the registry, call dispatch, interpreter-scoped state — and `runtime/boundary.py` — value conversion, nominal class synthesis, live views, and callable proxies. The evaluator hands each extern call to the registry; the runtime stays eval-free.

## Loading

Companions are imported after the static passes succeed and before evaluation starts, so a broken companion is a load-time diagnostic and `--dry-run` never imports one. The cache is keyed by canonical path and file identity. Companion imports compile the current source directly and neither read nor write bytecode caches because a companion may live in an immutable store tree and may change more precisely than bytecode timestamp validation records. A companion's module globals are registry-scoped and shared; its `runtime.state` accessor reaches a per-interpreter bag activated around each call, so concurrent interpreters sharing a registry never see each other's state. A host capability flag gates the FFI the way another gates shell `exec` ([hosting.md](../hosting.md)). The name a companion callable is looked up under is scope's `extern_names` fact ([frontend/scope.md](../frontend/scope.md)).

## Value Boundary

Conversion is value-directed: each `Value` kind encodes to a distinct Python representation and each concrete Python type decodes back. Arrays and dicts cross as live mutable views over their containers, so nothing is copied and cyclic values cross. Records with `var` fields likewise cross as live views that write through to declared `var` fields; immutable records and exceptions cross as snapshots. A `json` payload crosses uncopied and unchecked; the companion is trusted.

One Python class is synthesized per nominal identity, once, carrying its own descriptor, so decoding needs no call-scoped state and a class a companion captured keeps working after a REPL redeclaration mints a fresh identity. Enum members are plain record classes; the enum class is a namespace over them. Classes and the reserved helpers (`array`, `dict`, `json`, `runtime`, `nominals`, `AglException`) are exposed through a temporary `agl` module during companion import. A `nominals` address names a *loaded source declaration*, so a companion's module must import every nominal its companion addresses — the built-in exceptions it raises included.

## Callables and Exceptions

An AgL closure crosses as a callable proxy that is valid only on its owning interpreter thread while an extern call is active; the evaluator supplies the execution hook, the boundary owns conversion and the call window. A bare Python callable has no AgL representation. An AgL raise inside a callback crosses companion frames in the `AglException` carrier and resumes as the same exception if it escapes; a companion may use the carrier to initiate a typed AgL raise. Other Python `Exception`s become catchable `ExternError`; `BaseException`s such as the `SystemExit` from `std/process::exit` propagate. The declared extern signature is a companion obligation, not a runtime check.

## Code Entry Points

- `src/agm/agl/runtime/externs.py` — registry, companion loading, dispatch, state activation.
- `src/agm/agl/runtime/boundary.py` — encoding/decoding, class synthesis, views, proxies, the exception carrier.
- `stdlib/src/*.py` — the standard library's companions.
- Tests: `tests/test_agl_extern_*.py`.
