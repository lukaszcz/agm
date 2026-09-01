# AgL Host Runtime

The runtime package is the eval-free services layer: agent dispatch and session bridging, codecs, parameter conversion, host-environment types, rendering, serialization, tracing, and the Python FFI registry. It imports neither the evaluator nor the pipeline, and it builds on AGM's shared agent and core layers rather than reimplementing them.

## Agents and Sessions

`runtime/agents.py` decodes an `Agent` member record into a host spec from `agent/spec.py` and runs the spec's argv through the shared prompt and process seam. `runtime/sessions.py` bridges AgL `Session` values to the agent session service ([agents.md](../../agents.md)), resolves an agent's default transport from its spec, and supplies a dispatcher-backed host for embeddings without a session service. Agent prompts and shell commands are text rendering, not JSON serialization.

## Codecs

Built-in JSON contracts consume the typeless schema and decode data compiled during lowering. `runtime/codec.py` keeps strict parsing and lenient recovery separate: agent and shell output use the configured policy, casts always parse strictly, and `std/json` exposes both explicitly.

## Rendering and Serialization

All value display — interpolation, `print`, `render`, `as text`, REPL echo — goes through one renderer producing AgL-native syntax; the text-literal encoder is shared with the lexer. Nominal fields are normalized to declaration order at construction, so rendering, JSON, and equality agree without type information. Serialization follows lowered encode plans, adding `$case` only in enum-typed slots. Both walks thread the shared cycle guard; trace logging and in-flight error reporting degrade cycles and non-data values to markers rather than turning a working run into a failing one.

## Host-Backed Values and Tracing

`builtin var` bindings are identified by module, scope path, and name. Root `std/config` bindings are engine-setting registers with live effects; all others are ambient values with declared defaults that a host may seed — `std/env::environ`, for instance, is a startup snapshot of the process environment, so AgL `setenv` stays in-process. The trace store (`runtime/trace.py`) writes best-effort JSONL for run boundaries, `print`, agent requests and responses, shell executions, and escaping exceptions — never ordinary evaluation. A write to `log` or `log-file` repoints it live through `runtime/host_settings.py`; `std/process::exit` propagates its `SystemExit` after the final trace record.

## Code Entry Points

- `src/agm/agl/runtime/agents.py`, `sessions.py`, `request.py` — agent dispatch, session bridging, request/response types.
- `src/agm/agl/runtime/codec.py`, `contract.py`, `convert.py`, `params.py` — codecs, contracts, conversion, parameter values.
- `src/agm/agl/runtime/render.py`, `serialize.py` — rendering and JSON serialization.
- `src/agm/agl/runtime/trace.py`, `host_settings.py`, `types.py`, `option.py` — tracing, live settings, host environment types, `Option` construction.
- `src/agm/agl/runtime/externs.py`, `boundary.py` — the FFI ([ffi.md](ffi.md)).
- Tests: `tests/test_agl_runtime.py`, `test_agl_codec.py`, `test_agl_trace.py`, `test_agl_session_runtime.py`, `test_agl_builtin_var_*.py`.
