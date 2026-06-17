# Shell Execution: `exec`

[← Index](index.md)

`exec` runs a shell command and yields its output. It is a built-in
**function** invoked with the same uniform call syntax as user functions:

```agl
let res = exec "ls -la ${dir}"       # res : ExecResult (default)
let out: text = exec "cat ${path}"   # parsed form: stdout verbatim
exec "make build"                    # for effect; result discarded
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
structured values as pretty JSON (2-space indent).

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
returns the `ExecResult` prelude record:

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

Spawn failure and timeout still set `exit_code` to `-1` and `timed_out` to
`true` respectively; they do not raise in this form.

### Parsed form — target is any non-`ExecResult` type

When the target type is any type other than `ExecResult`, `exec` parses
stdout into that type (honouring `format`, `strict_json`, and
`on_parse_error`) and **raises `ExecError` on a nonzero exit**:

```agl
let out: text = exec "cat ${path}"          # stdout verbatim; raises on nonzero
let data: dict[text, int] = exec(           # JSON parsed; raises on nonzero
  "compute-stats --json",
  on_parse_error: Retry(n: 1)
)
```

This is the same behavior as v1: a nonzero exit raises `ExecError`, and
unparseable output raises `AgentParseError` (with agent name `"exec"`).

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

- `format:` — codec name (a `text` value); normally auto-selected.
- `strict_json:` — `bool`; opts the JSON codec into strict parsing.
- `on_parse_error:` — `ParsePolicy`; controls retry behavior on parse
  failures in the parsed form. In the structured form this parameter is
  ignored.

## Retries

**Retries re-run the command.** Unlike an `ask` retry — which sends
corrective feedback to the same conversation — an `exec` retry executes the
command again; each invocation is traced separately. If every attempt fails
to parse, `AgentParseError` is raised with agent name `"exec"`.

## Exceptions

`ExecError` (a failing or timed-out command, parsed form) and
`AgentParseError` (unparseable output from a succeeding command) are distinct
and independently catchable:

```agl
try
  let data: dict[text, int] = exec "compute-stats --json"
catch ExecError as e =>
  print "command failed (${e.exit_code}): ${e.stderr}"
catch AgentParseError as e =>
  print "not valid JSON: ${e.raw}"
```

In the structured form, `ExecError` is raised only by spawn failure in
transport (the shell itself cannot be launched — a rarer condition); nonzero
exits and timeouts surface in `exit_code` / `timed_out` instead.

See [Exceptions](exceptions.md) for the full field lists.
