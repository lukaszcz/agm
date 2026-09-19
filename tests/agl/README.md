# AgL end-to-end test corpus

Data files for `tests/test_agl_e2e.py`, the specification suite for the AgL
implementation (expression-oriented core, uniform call syntax, user-defined
functions).

Layout:

- `programs/**/*.agl` — valid AgL programs, each declaring an explicit
  `program def` entry (or marked `inline_entry`, see below) and carrying a sidecar
  `<name>.scenarios.json` describing the scenarios it runs under. Every program is
  exercised under **multiple scenarios**: distinct combinations of host params,
  scripted mock agent responses, and scripted shell results, each driving a different
  control-flow path with its own expected outcome.
- `rejections/**/*.agl` — invalid programs that the static pipeline
  (lex/parse/scope/typecheck) must reject before executing anything, each with a
  sidecar `<name>.expect.json`.

## Program categories

| Directory | What it exercises |
|-----------|-------------------|
| `attributes/` | Declaration attributes: `@arg-*` zones on functions, lambdas, records, exceptions, enum members, and `program def` |
| `basics/` | `let`/`var`/`:=`, params, agent calls, print rendering |
| `calls/` | `ask` parse policies (`Retry`/`Abort`), format options |
| `canonical/` | Multi-agent review/fix workflows |
| `casts/` | `as`/`as?` casts and `CastError`/`JsonParseError` handling |
| `control/` | `if`/`case`/`do…until`/`try…catch`/`raise` |
| `errors/` | Exception types, field access in catch, rethrow |
| `exec/` | Shell execution, `ExecResult` structured handle |
| `exprs/` | Arithmetic, comparisons, string operations |
| `functions/` | User-defined functions: recursion, default args, first-class values, lambdas, `Agent::ask` in a `def` body |
| `generics/` | Generic types/functions: inference, explicit `::[…]` overrides, erasure, HOFs, imported generics |
| `inline/` | Single-expression programs and host-wrapped `inline_entry` sources |
| `methods/` | Record and enum methods: direct and bound calls, generic receivers, partial application, and scope use |
| `modules/` | Multi-file module programs (via `module_roots`): imports combined with generics, casts, records/enums, pattern matching, and cross-module mutual recursion |
| `partial/` | Partial application placeholders for calls, constructors, generics, eager capture, and higher-order use |
| `rendering/` | Console/value rendering: nesting, escaping, exception rendering |
| `templates/` | Template interpolation |
| `types/` | Records, enums, `json`, `array`, `dict` |
| `values/` | `std/value::parse`/`try-parse`: explicit and contextual type arguments, `ValueParseError` |

## `<name>.scenarios.json`

```json
{
  "scenarios": [
    {
      "name": "snake_case_scenario_id",
      "module_roots": ["program_modules/example"],
      "params": {"spec": "verbatim text", "rounds": 3},
      "positional": ["verbatim text", 3],
      "agents": {
        "reviewer": ["first response", "second response"],
        "impl": {
          "responses": ["fix"],
          "repeat_last": true,
          "session": {
            "capabilities": ["ask", "compact", "fork", "stats"],
            "operations": {
              "compact": ["success"],
              "stats": [{"input_tokens": 8, "output_tokens": 3,
                         "cost": "0.02", "context_percent": "5"}]
            }
          }
        }
      },
      "shell": [{"command": "printf done", "stdout": "done"}],
      "http": [{"expect": {"method": "GET", "url": "https://x/y"},
                "status": 200, "body": "ok"}],
      "runtime": {"default_call_depth_limit": 20, "default_strict_json": true},
      "filesystem": {"directories": ["work"]},
      "expect": {
        "stdout": "exact full stdout",
        "stdout_contains": ["fragment"],
        "stdout_not_contains": ["fragment"],
        "calls": {"reviewer": 2},
        "prompts": [
          {"agent": "reviewer", "call": 0,
           "equals": "exact rendered prompt",
           "contains": ["fragment"], "not_contains": ["fragment"],
           "schema_contains": ["fragment"],
           "schema_paths": [
             {"path": ["$ref"], "equals": "#/$defs/Task"}
           ]}
        ],
        "sessions": [
          {"agent": "impl", "tag": "session-1", "opened": true, "closed": true},
          {"agent": "impl", "tag": "session-2", "parent": "session-1"}
        ],
        "session_prompts": [
          {"agent": "impl", "session": "session-1", "call": 0, "equals": "first"},
          {"agent": "impl", "session": "session-1", "call": 1,
           "follow_up": {"contains": ["correct"]}}
        ],
        "session_operations": [
          {"agent": "impl", "session": "session-1", "operation": "compact",
           "arg": "summarize"}
        ],
        "raises": {"type": "MaxIterationsExceeded",
                   "fields": {"limit": 3},
                   "message_contains": ["fragment"]},
        "exit_code": 42,
        "host_error": {"message_contains": ["spec"]}
      }
    }
  ]
}
```

Field notes:

- `params` — the selected entry's own named/standard-zone value-parameter
  arguments, verbatim. JSON numbers with a fractional part are loaded as
  `decimal.Decimal` (AgL has no binary floats).
- `positional` — an entry program's own positional-only/standard-zone
  arguments, in order.
- `agents` — response queues selected by the `Agent` value at each call site,
  consumed in call order. A list is a strict queue (a call past its end fails the
  test); the object form allows `repeat_last` for loop-exhaustion scenarios. The key
  `ask` scripts the default non-command `Agent` variants. An object may also
  carry `session.capabilities`, a list of supported protocol operations (`ask`,
  `compact`, `fork`, `set-name`, `stats`); when omitted, all are supported.
  Omit an operation to script the service's capability rejection. Separately,
  `session.operations` maps native operation names (`compact`, `reset`, `fork`,
  `stats`, `set-name`) to ordered backend outcomes. `session.ask` can similarly
  script each session prompt as `success` or a transport-failure object with a
  `cause` (and optional `exit_code`, `stderr_tail`, and `elapsed`). Each supplied
  outcome must be consumed; `success` is normal and a `stats` object may provide
  `input_tokens`, `output_tokens`, `cost`, and `context_percent`. `reset` is
  always dispatched by the service and cannot have an `unsupported` outcome.
  Legacy `unsupported` outcomes remain accepted for optional operations, but
  new scenarios should use `capabilities`. Session and ordinary asks consume the
  same response queue.
- `shell` — ordered scripted shell calls. Each object names the rendered
  `command` and may set `stdout`, `stderr`, `returncode`, `timed_out`, or
  `spawn_error`; omitted fields describe a successful command with empty output.
  The harness rejects an unexpected command and verifies the full script was used,
  so acceptance tests never execute a real shell command.
- `http` — ordered scripted HTTP exchanges, in `tests/_http_helpers.py`'s `FakeHttp`
  outcome shape: each object may assert an `expect` (method, url, header subset, body,
  `timeout`, `verify`) against the actual request and scripts its outcome — `status`,
  `headers` (a mapping, or a list of `[name, value]` pairs to script repeated headers),
  a `body` (+ optional `charset`) or `body_hex`, or a `fail`/`fail_mid_stream` kind. The
  harness installs `FakeHttp` in place of `agm.core.http.open_session` and verifies the
  full script was used, so acceptance tests never touch the network; an absent key means
  no HTTP call is allowed.
- `runtime` — optional `PipelineDriver` constructor overrides
  (`default_call_depth_limit`, `default_strict_json`).
- `filesystem` — optional fixture in a test-created temporary root. It may declare
  `directories`, UTF-8 `text_files`, hexadecimal `hex_files`, and
  `directory_symlinks`; a `"$TEMP_ROOT"` parameter value is replaced with that root.
- `module_roots` — optional paths relative to `tests/agl/`. When present, the
  program runs through the multi-file module graph with these library roots.
- `module_params` — directly supplied seed values for `@param` bindings, keyed
  by declaration path (`<entry>::name` for the entry program's own module,
  `module/path::name` for an import). Outranks every other tier, `@config`
  included.
- `inline_entry` — the program declares no `program def`: it runs through the
  same synthetic-entry transform as `agm exec -c`, which keeps root
  declarations and the bindings they read at the root.
- `expect.calls` — exact number of ordinary (non-session) calls per listed agent;
  ordinary retries count here. Session asks, including retries, are counted only
  by `expect.session_prompts`.
- `expect.prompts` — assertions on the rendered user prompt (`request.prompt`) an
  agent received on a given 0-based call index. `schema_contains` instead checks
  that call's output contract JSON Schema (the format-instructions/JSON Schema
  channel a real runner-backed agent appends to the message; see
  `runtime/agents.py`). `schema_paths` asserts exact values at dictionary-key
  paths in that schema, such as a recursive root and child `$ref` pointing to the
  same `$defs` entry.
- `expect.sessions` — exact observed set of deterministic per-agent session
  tags. `opened`, `closed`, `transport`, `single_prompt`, and (for a forked session)
  `parent` are optional exact assertions. Tags are assigned in creation order as
  `session-1`, `session-2`, and so on.
- `expect.session_prompts` — assertions on a session prompt. `session` selects
  its tag and `call` is its 0-based prompt number. For every created tag, the
  entries must enumerate every received prompt exactly once, so an extra session
  retry fails the scenario. Its prompt checks use the same `equals`,
  `starts_with`, `contains`, and `not_contains` fields as `expect.prompts`;
  `follow_up` contains those checks and requires a later call on that same
  session.
- `expect.session_operations` — assertions for native session operations;
  `operation`, optional 0-based `call`, optional `arg`, and optional `outcome`
  select and check each observation. When present, entries must enumerate every
  operation on every created session exactly once, including an `unsupported`
  capability rejection; omit the field to leave operation coverage unchecked.
- `expect.raises` — the uncaught AgL exception ending the run: its type name, an
  exact-match subset of its fields, and substrings of its `message` field. A
  dict-shaped field value (a nested record) is itself an exact-match subset of
  its keys, recursively — useful for asserting part of a nested field (e.g. a
  status and body) while ignoring a nondeterministic one (e.g. elapsed time).
- `expect.exit_code` — the program must terminate through `SystemExit` with this
  status; it is used for host-termination workflows such as `std/process::exit`.
- `expect.host_error` — the run must fail pre-execution (program argument
  binding or decode failure): no agent is called, no AgL exception is
  raised, and the diagnostics mention the fragments.
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
