# AgL Execution

Execution is everything after successful match compilation: lowering the match-compiled artifact into a closed, typeless program, evaluating it, and the host runtime that backs agents, shell calls, the Python FFI, codecs, and rendering. Frontend artifacts never reach the evaluator; lowering is the boundary.

## Design Invariant: Typeless Downstream

Every decision that needs type information — which built-in path, which codec, which decode schema, which conversion, which nominal — is resolved during lowering and baked into typeless descriptors. The evaluator only interprets closed nodes, and the runtime services stay eval-free. This keeps the interpreter simple and the dependency graph acyclic.

## What To Read Next

- [lowering.md](lowering.md) — lowering, linking, and the execution IR.
- [evaluator.md](evaluator.md) — the interpreter, values, control flow, and host-backed operations.
- [runtime.md](runtime.md) — agents and sessions, codecs, rendering, serialization, tracing, host-backed values.
- [ffi.md](ffi.md) — the Python FFI: companion loading, the value boundary, and target contracts.
