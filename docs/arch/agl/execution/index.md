# AgL Execution

Execution is everything after successful match compilation: lowering the match-compiled artifact into a closed, typeless program, evaluating it, and the host runtime that backs agents, shell calls, codecs, and rendering. Frontend artifacts never reach the evaluator — lowering is the boundary, and no checked-only lowering path exists. See [../index.md](../index.md) for the surrounding pipeline.

## Design Invariant: Typeless Downstream

Every decision that needs type information — which built-in path, which codec, which decode schema, which conversion — is resolved during lowering and baked into typeless descriptors, so the evaluator only interprets closed nodes and the runtime services stay eval-free. This keeps the interpreter simple and the dependency graph acyclic.

## What To Read Next

- Read [lowering.md](lowering.md) for lowering/linking and the execution IR.
- Read [evaluator.md](evaluator.md) for the interpreter and extern (Python FFI) dispatch.
- Read [runtime.md](runtime.md) for the host runtime services, value rendering, and the pipeline orchestrator.
