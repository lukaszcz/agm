# AgL Host Runtime and Pipeline

The runtime package is the eval-free services layer: agents, codecs, parameter conversion, host-environment assembly, and rendering. It imports neither the evaluator nor the pipeline, which keeps the services reusable and the dependency graph acyclic. It builds on AGM's shared agent runner and core primitives rather than reimplementing them ([../index.md](../index.md)).

## Codecs

Built-in JSON contracts consume the typeless schema/decode data compiled during lowering. Custom codecs are materialized through their own `make_contract` hook before lowering, while checker types are still available, and then run from the embedded typeless payload — with compatibility shims for older host codecs.

## Value Rendering

All value display — string interpolation, `print`, `render`, `as text`, and REPL echo — goes through one recursive renderer producing AgL-native syntax. Nominal fields are normalized into declaration order at construction, so the renderer needs no type information and every consumer (rendering, `as json`, equality) agrees on field order. Unit values carry a display flag distinguishing explicit `()` from the `void` produced by statement-like effects, which lets the REPL suppress echo.

## Pipeline Orchestrator

The pipeline sits on top: it drives the compile → lower → evaluate sequence and assembles the host environment, and it is the public entry point used by `agm exec` and the REPL. Programs are parameterized by `param` declarations resolved at evaluation time (external value > default expression > error for a required param), and the pipeline can discover the parameter inventory before execution so a host can wire external values ([../repl.md](../repl.md)). Every artifact a pass produces is handed forward rather than recomputed, so however many times a host resumes the pipeline, the program compiles and lowers exactly once. Pure compile-time schema and format-instruction generation lives in its own helper so lowering stays independent of runtime execution.

## Code Entry Points

- `src/agm/agl/runtime/` — agents, codecs, parameter conversion, host-environment types, the renderer, and the extern registry.
- `src/agm/agl/pipeline.py` — the orchestrator; `src/agm/agl/type_schema.py` — compile-time schema/format generation.
- Tests: `tests/test_agl_runtime.py`, `tests/test_agl_codec.py`, `tests/test_agl_pipeline_*.py`.
