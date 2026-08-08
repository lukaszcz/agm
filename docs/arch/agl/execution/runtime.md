# AgL Host Runtime and Pipeline

The runtime package is the eval-free services layer: value-driven agents, codecs, parameter conversion, host-environment assembly, and rendering. It imports neither the evaluator nor the pipeline, which keeps the services reusable and the dependency graph acyclic. It builds on AGM's shared agent runner and core primitives rather than reimplementing them ([index.md](agl/index.md)).

## Codecs

Built-in JSON contracts consume the typeless schema/decode data compiled during lowering. Custom codecs are materialized through their own `make_contract` hook before lowering, while checker types are still available, and then run from the embedded typeless payload — with compatibility shims for older host codecs.

## Value Rendering

All value display — string interpolation, `print`, `render`, `as text`, and REPL echo — goes through one recursive renderer producing AgL-native syntax. The text-literal surface encoder is shared from `semantics/text_literal.py` with lexer decoding and match diagnostics, so interpolation escaping has one owner. Nominal fields are normalized into declaration order at construction, so the renderer needs no type information and every consumer (rendering, `as json`, equality) agrees on field order. Unit values carry a display flag distinguishing explicit `()` from the `void` produced by statement-like effects, which lets the REPL suppress echo.

Rendering and JSON serialization (`runtime/render.py`, `runtime/serialize.py`) both recurse through arrays/dicts/nominal fields and so both thread the shared cycle guard from `semantics/cycles.py`, lazily allocated so an acyclic value never pays for it; a detected cycle surfaces as a Python-level sentinel that each caller (the interpreter, casts, agent/exec prompt rendering, trace logging, error reporting, the REPL echo) converts into the catchable `CyclicValueError`, or — for trace logging and in-flight error reporting, which must never turn a working run into a failing one — degrades to a marker in place of the value.

The Python FFI is split between `runtime/externs.py`, which loads companions and dispatches calls, and `runtime/boundary.py`, which converts values at the boundary. Conversion is value-directed: AgL `Value` subclasses encode to distinct Python representations and concrete Python types decode back to AgL values. The registry synthesizes one nominal class per identity, once, and never re-shapes it — an identity's layout is fixed at its declaration — so a class a companion already captured — a module global, a closure, a default argument — keeps constructing and recognizing values of the declaration it was captured from even after a redeclaration mints a fresh identity with a class of its own. Which identity currently bears a shared name path (for a companion's bare/dotted lookup at import time) is decided by the type table's name index, threaded through lowering as a per-descriptor flag, rather than by insertion order. That an identity is never re-registered under a different shape — the premise of synthesizing once — is asserted by a self-check gated on the AgL self-validation toggle ([testing.md](../../testing.md)). Each class carries its own descriptor, so decoding consults no call-scoped state: nominal values cross correctly at companion import time, from worker threads, and after a call has returned. Fields live in one dict keyed by the original AgL field spellings. Classes are exposed with `array`, `dict`, and `json` through a temporary `agl` module during companion import; unique final names are direct `agl` attributes, and the complete module/scope identity tree is available under `agl.nominals`, where a nominal whose name is also a scope segment doubles as that scope's namespace. The registry retains no extern signature schema after lowering. Arrays and dicts cross as lazy mutable views over their containers; views hold no call scope, are not revoked, and compare/hash by their container identity. A `json` payload crosses uncopied and unchecked in both directions: the companion is trusted to hand over a JSON-shaped payload and not to retain and mutate one, so nothing walks it at the boundary.

A second sentinel, `AglNonDataValue` (`runtime/serialize.py`), covers the other way JSON serialization can fail: a value kind with no JSON representation at all (`unit`, constructor, function, iterator). Unlike the cycle sentinel it has no catchable-exception form, because every evaluator route into serialization is statically gated by `is_json_convertible` — it can only arrive via the two ungated reporters, trace logging and in-flight error reporting, which degrade it to a marker exactly as they do a cycle. A record or exception field may legitimately hold such a value even though casting its type to `json` is a static error.

## Tracing and Live Trace Settings

`runtime/trace.py` writes best-effort JSONL records. Every record starts with
`ts`, `run_id`, and `kind`; the store records only run boundaries, `print`
stdout, agent requests and responses, shell executions, and exceptions that
escape uncaught. It deliberately does not trace ordinary expression evaluation
or attach a `trace_id` to records or exception values.

`eval/effects.py` is the agent logging seam: it composes the prompt, records the
request before dispatch, and records every response path, including unit calls,
transport failures, and cancellation. `runtime/agents.py` remains the
value-driven transport boundary and sends that composed prompt verbatim.

The trace destination is the sole live host service configured by an AgL `builtin var` write. The engine-key catalog names its `log`/`log-file` register pair explicitly; either write repoints the same trace store, while other host-consumed settings remain registers read on demand. `runtime/host_settings.py` applies the command-supplied trace-path policy without importing the command layer.

## Pipeline Orchestrator

The pipeline sits on top: it drives the compile → lower → evaluate sequence and assembles the host environment, and it is the public entry point used by `agm exec` and the REPL. Programs are parameterized by `param` declarations resolved at evaluation time (external value > default expression > error for a required param), and the pipeline can discover the parameter inventory before execution so a host can wire external values ([repl.md](agl/repl.md)). Every artifact a pass produces is handed forward rather than recomputed, so however many times a host resumes the pipeline, the program compiles and lowers exactly once. Pure compile-time schema and format-instruction generation lives in its own helper so lowering stays independent of runtime execution.

## Code Entry Points

- `src/agm/agl/eval/effects.py` — the evaluator's observable-effect seam for agent request/response logging and shell execution.
- `src/agm/agl/runtime/agents.py` — decodes `Agent` enum values and runs their builder-produced argv through the shared prompt/process seam; `runtime/trace.py` writes the JSONL trace records.
- `src/agm/agl/runtime/` — codecs, parameter conversion, host-environment types, and the renderer; `runtime/externs.py` owns extern loading/dispatch and `runtime/boundary.py` owns boundary conversion and live views.
- `src/agm/agl/pipeline.py` — the orchestrator; `src/agm/agl/type_schema.py` — compile-time schema/format generation.
- Tests: `tests/test_agl_runtime.py`, `tests/test_agl_codec.py`, `tests/test_agl_pipeline_*.py`, `tests/test_agl_extern_boundary.py`, `tests/test_agl_extern_views.py`.
