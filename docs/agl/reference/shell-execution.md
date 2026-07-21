# Shell Execution: `exec`

[← Index](index.md)

`exec` runs a shell command and yields its output. It is a built-in
**function** invoked with the same uniform call syntax as user functions:

<!-- agl-check: skip -->
```agl
let res = exec "ls -la ${dir}"       # res : ExecResult (default)
let out: text = exec "cat ${path}"   # parsed form: stdout verbatim
let completed: unit = exec "make build" # unit form; raises ExecError on nonzero
```

Like `ask`, `exec` is a **contextual keyword**
([Lexical structure](lexical-structure.md)): in call position it denotes the
built-in shell executor; it cannot be declared with `let`/`var`/`param` or
as an agent name; it cannot be bound as a function value; it remains legal
as a field name. A host may statically disallow shell execution altogether,
in which case every `exec` call is a static error.

## Single-argument sugar

With no named arguments, `exec` may be called with the command template
written directly without parentheses:

```agl
exec "make build"                    # equivalent to exec("make build")
```

With named arguments, parentheses are required.

## Interpolation in shell templates

The command template uses the same uniform rendering as all other templates
([Strings and interpolation](strings-and-interpolation.md)): `text` values
interpolate verbatim; `int`, `decimal`, and `bool` as plain scalar text;
structured values (`list`, `dict`, record, enum, exception) in AgL form —
single-line, no injected newlines. To interpolate a structured value as JSON,
use an explicit cast: `${value as json}`.

Interpolated values are inserted **verbatim** into the command string —
there is **no automatic shell quoting**. The workflow author is responsible
for writing shell-safe commands. Unvalidated text (for example, model-produced
content or user input) inside a shell command is an injection hazard unless
the author explicitly handles quoting.

## The two forms of `exec`

`exec`'s behavior depends on the **target type**, determined from context
exactly as for `ask` ([Agent calls](agent-calls.md)).

### Structured form — target is `ExecResult`

When no expected type is present, or the annotation is `ExecResult`, `exec`
returns the `ExecResult` standard core record:

```text
stdout:    text
stderr:    text
exit_code: int
timed_out: bool
```

A **nonzero exit does not raise** in this form — the caller branches on
`exit_code`:

```agl
let res = exec "ls -la"             # res : ExecResult
print(res.stdout)
if res.exit_code != 0 =>
  print("command failed: ${res.stderr}")
```

A spawn failure or timeout raises `ExecError` in this form. A timeout does
not produce an `ExecResult` with `timed_out = true`.

### Parsed form — target is any non-`ExecResult` or `unit` type

When the target type is neither `ExecResult` nor `unit`, `exec` parses stdout
into that type (honouring `format`, `strict_json`, and `on_parse_error`) and
**raises `ExecError` on a nonzero exit**:

<!-- agl-check: skip -->
```agl
let out: text = exec "cat ${path}"          # stdout verbatim; raises on nonzero
let data: dict[text, int] = exec(           # JSON parsed; raises on nonzero
  "compute-stats --json",
  on_parse_error = Retry(n = 1)
)
```

A nonzero exit raises `ExecError`, and unparseable output raises
`AgentParseError` (with agent name `"exec"`).

### Unit form — target is `unit`

When context requires `unit` — for example, a non-final bare expression in a
block or a binding annotated `unit` — `exec` has target type `unit`. For this
unit contract, a nonzero exit raises `ExecError`; successful stdout is
discarded and the call returns `void`:

```agl
exec "make build"
let completed: unit = exec "make lint"
```

Because no output is parsed, `format`, `strict_json`, and `on_parse_error` are
invalid for a `unit` target.

## Execution semantics

1. The rendered command runs via the host shell (`sh -c` semantics),
   un-sandboxed, with the user's privileges. The host's configured idle
   timeout applies.
2. Standard output and standard error are captured.
3. In the **parsed form**, on success (exit status 0), trailing newlines are
   stripped from stdout — as in `$(…)` command substitution — and the result
   is bound at the call's target type. In the **structured form**, stdout
   and stderr are returned as-is.
4. Every execution is traced: command, exit code, duration, stdout, stderr.

## Named parameters

`exec` accepts the same codec-related named parameters as `ask`:

- `format` — codec name (a `text` value); normally auto-selected.
- `strict_json` — `bool`; opts the JSON codec into strict parsing.
- `on_parse_error` — `ParsePolicy`; controls retry behavior on parse
  failures in the parsed form. In the structured and unit forms, where no
  stdout parsing happens, passing this parameter is a static error.

## Retries

**Retries re-run the command.** Unlike an `ask` retry — which sends
corrective feedback to the same conversation — an `exec` retry executes the
command again; each invocation is traced separately. If every attempt fails
to parse, `AgentParseError` is raised with agent name `"exec"`.

## Exceptions

`ExecError` (a failing or timed-out command in parsed or unit form) and
`AgentParseError` (unparseable output from a succeeding command) are distinct
and independently catchable:

<!-- agl-check: skip -->
```agl
try
  let data: dict[text, int] = exec "compute-stats --json"
catch ExecError as e =>
  print "command failed (${e.exit_code}): ${e.stderr}"
catch AgentParseError as e =>
  print "not valid JSON: ${e.raw}"
```

In the structured form, `ExecError` is raised for a spawn failure (the shell
itself cannot be launched) or timeout. A nonzero exit instead surfaces in
`exit_code`.

See [Exceptions](exceptions.md) for the full field lists.
