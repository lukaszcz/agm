# AgL end-to-end test programs

Data files for `tests/test_agl_e2e.py`, the TDD specification suite for the AgL v1
implementation planned in `notes/PLAN_DSL.md` (language spec: `notes/dsl_design.md`).
The suite is **red until `agm.agl` exists**.

Layout:

- `programs/**/*.agl` — valid AgL programs, each with a sidecar
  `<name>.scenarios.json` describing the scenarios it runs under. Every program is
  exercised under **multiple scenarios**: distinct combinations of host inputs and
  scripted mock agent responses, each driving a different control-flow path with its
  own expected outcome.
- `rejections/**/*.agl` — invalid programs that the static pipeline
  (lex/parse/scope/typecheck) must reject before executing anything, each with a
  sidecar `<name>.expect.json`.

## `<name>.scenarios.json`

```json
{
  "scenarios": [
    {
      "name": "snake_case_scenario_id",
      "inputs": {"spec": "verbatim text", "rounds": 3},
      "agents": {
        "reviewer": ["first response", "second response"],
        "impl": {"responses": ["fix"], "repeat_last": true}
      },
      "runtime": {"default_loop_limit": 3, "default_strict_json": true},
      "expect": {
        "stdout": "exact full stdout",
        "stdout_contains": ["fragment"],
        "stdout_not_contains": ["fragment"],
        "calls": {"reviewer": 2},
        "prompts": [
          {"agent": "reviewer", "call": 0,
           "equals": "exact rendered prompt",
           "contains": ["fragment"], "not_contains": ["fragment"]}
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

- `inputs` — passed to `runtime.run(source, inputs=...)` verbatim. JSON numbers with
  a fractional part are loaded as `decimal.Decimal` (AgL has no binary floats).
- `agents` — per-agent response queues, consumed in call order **across all call
  sites** of that agent. A list is a strict queue (a call past its end fails the
  test); the object form allows `repeat_last` for loop-exhaustion scenarios. The key
  `ask` scripts the built-in default agent (passed as the runtime's
  `default_agent`, since `ask` cannot be registered by name).
- `runtime` — optional `WorkflowRuntime` constructor overrides
  (`default_loop_limit`, `default_strict_json`).
- `expect.calls` — exact number of calls per listed agent (retries count as calls).
- `expect.prompts` — assertions on the rendered user prompt (`request.prompt`) an
  agent received on a given 0-based call index.
- `expect.raises` — the uncaught AgL exception ending the run: its type name, an
  exact-match subset of its fields, and substrings of its `message` field.
- `expect.host_error` — the run must fail pre-execution (input validation): no agent
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
