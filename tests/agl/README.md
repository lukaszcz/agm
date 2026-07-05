# AgL end-to-end test corpus

Data files for `tests/test_agl_e2e.py`, the specification suite for the AgL
implementation (expression-oriented core, uniform call syntax, user-defined
functions).

Layout:

- `programs/**/*.agl` — valid AgL programs, each with a sidecar
  `<name>.scenarios.json` describing the scenarios it runs under. Every program is
  exercised under **multiple scenarios**: distinct combinations of host params and
  scripted mock agent responses, each driving a different control-flow path with its
  own expected outcome.
- `rejections/**/*.agl` — invalid programs that the static pipeline
  (lex/parse/scope/typecheck) must reject before executing anything, each with a
  sidecar `<name>.expect.json`.

## Program categories

| Directory | What it exercises |
|-----------|-------------------|
| `basics/` | `let`/`var`/`:=`, params, agent calls, print rendering |
| `calls/` | `ask` parse policies (`Retry`/`Abort`), format options |
| `canonical/` | Multi-agent review/fix workflows |
| `casts/` | `as`/`as?` casts, `CastError`/`JsonParseError`, `parse_json` |
| `control/` | `if`/`case`/`do…until`/`try…catch`/`raise` |
| `errors/` | Exception types, field access in catch, rethrow |
| `exec/` | Shell execution, `ExecResult` structured handle |
| `exprs/` | Arithmetic, comparisons, string operations |
| `functions/` | User-defined functions: recursion, default args, first-class values, lambdas, `ask(agent:)` in a `def` body |
| `generics/` | Generic types/functions: inference, explicit `::[…]` overrides, erasure, HOFs, imported generics |
| `inline/` | Single-expression programs |
| `modules/` | Multi-file module programs (via `module_roots`): imports combined with generics, casts, records/enums, pattern matching, and cross-module mutual recursion |
| `rendering/` | Console/value rendering: nesting, escaping, exception rendering |
| `templates/` | Template interpolation |
| `types/` | Records, enums, `json`, `list`, `dict` |

## `<name>.scenarios.json`

```json
{
  "scenarios": [
    {
      "name": "snake_case_scenario_id",
      "module_roots": ["program_modules/example"],
      "params": {"spec": "verbatim text", "rounds": 3},
      "agents": {
        "reviewer": ["first response", "second response"],
        "impl": {"responses": ["fix"], "repeat_last": true}
      },
      "runtime": {"default_call_depth_limit": 20, "default_strict_json": true},
      "expect": {
        "stdout": "exact full stdout",
        "stdout_contains": ["fragment"],
        "stdout_not_contains": ["fragment"],
        "calls": {"reviewer": 2},
        "prompts": [
          {"agent": "reviewer", "call": 0,
           "equals": "exact rendered prompt",
           "contains": ["fragment"], "not_contains": ["fragment"],
           "schema_contains": ["fragment"]}
        ],
        "raises": {"type": "MaxIterationsExceeded",
                   "fields": {"limit": 3},
                   "message_contains": ["fragment"]},
        "host_error": {"message_contains": ["spec"]}
      }
    }
  ]
}
```

Field notes:

- `params` — passed to `runtime.run(source, params=...)` verbatim. JSON numbers with
  a fractional part are loaded as `decimal.Decimal` (AgL has no binary floats).
- `agents` — per-agent response queues, consumed in call order **across all call
  sites** of that agent. A list is a strict queue (a call past its end fails the
  test); the object form allows `repeat_last` for loop-exhaustion scenarios. The key
  `ask` scripts the built-in default agent (passed as the runtime's
  `default_agent`, since `ask` cannot be registered by name).
- `runtime` — optional `PipelineDriver` constructor overrides
  (`default_call_depth_limit`, `default_strict_json`).
- `module_roots` — optional paths relative to `tests/agl/`. When present, the
  program runs through the multi-file module graph with these library roots.
- `expect.calls` — exact number of calls per listed agent (retries count as calls).
- `expect.prompts` — assertions on the rendered user prompt (`request.prompt`) an
  agent received on a given 0-based call index. `schema_contains` instead checks
  that call's output contract `format_instructions` (the format-instructions/JSON
  Schema channel a real runner-backed agent appends to the message; see
  `runtime/agents.py`) — the mechanism to assert a JSON target's derived schema,
  including `$defs`/`$ref` for a recursive type, reached the agent.
- `expect.raises` — the uncaught AgL exception ending the run: its type name, an
  exact-match subset of its fields, and substrings of its `message` field.
- `expect.host_error` — the run must fail pre-execution (param validation): no agent
  is called, no AgL exception is raised, and the diagnostics mention the fragments.
- Exact `stdout` is asserted only where rendering is pinned by the design (`text`
  verbatim, scalars as scalar text). Pretty-JSON console rendering and
  boundary-marked prompt rendering are asserted with `contains` fragments to avoid
  pinning incidental formatting.

## `<name>.expect.json`

```json
{"diagnostic": {"line": 2, "message_contains": ["equality"]}}
```

The program must be rejected statically: `result.ok` is false, `result.error` is
`None` (nothing executed), and at least one diagnostic exists — on `line`
(1-based, when given) and containing the `message_contains` fragments
(case-insensitive, when given).

## Rejection categories

| Directory | What it tests |
|-----------|---------------|
| `parse/` | Syntax errors |
| `scope/` | Undefined names, duplicate declarations |
| `type/` | Type mismatches, arity errors, operator type rules, opacity |
