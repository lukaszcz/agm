# Shell Execution: `exec`

[← Index](index.md)

`exec` runs a shell command and yields its output. It is a built-in
**function** invoked with the same uniform call syntax as user functions:

<!-- agl-check: fragment -->
```agl
let res = exec "ls -la %{dir}"       # res : ExecResult (default)
let out: text = exec "cat %{path}"   # parsed form: stdout verbatim
let completed: unit = exec "make build" # unit form; raises ExecError on nonzero
```

Like `ask`, `exec` is a **contextual keyword**
([Lexical structure](lexical-structure.md)): it cannot be declared with
`let`/`var`, as a function, or as a function parameter name, but it can be
referenced as a function value. The value is an eta-expanded `text -> T`
callable using the ambient spawn defaults; `T` comes from an expected function
type or explicit `::[T]` and otherwise defaults to `ExecResult`:

```agl
program def main() -> unit =
  let run: text -> ExecResult = exec
  let read = exec::[text]
  ()
```

Use an explicit lambda around a direct call to capture non-default spawn or
parse options. `exec` remains legal as a field name. A host may statically
disallow shell execution altogether, in which case every `exec` call or value
is a static error.

## Spawn parameters

`exec` has this signature:

<!-- agl-check: fragment -->
```agl
exec(
  command: text,
  env: Environ = std/env::environ,
  cwd: Option[text] = Option[text]::None,
  timeout: Option[text] = std/config::timeout,
) -> ExecResult
```

`env` is the complete environment given to the shell; it replaces rather than
merges with the AGM process environment. The default is the startup ambient
`std/env::environ` snapshot. Use `environ.extended(overrides)` when a command
needs an explicit overlay. `cwd` is an optional working directory and `timeout`
is an optional idle timeout duration. The single-argument sugar below supplies
only the command, so it uses all three defaults.

## Single-argument sugar

With no named arguments, `exec` may be called with the command template
written directly without parentheses:

```agl
program def main() -> unit =
  exec "make build"
```

With named arguments, parentheses are required.

A `$` literal
([Strings and interpolation](strings-and-interpolation.md#the--literal)) may
supply the same single argument, inline or as a block:

```agl
program def main() -> unit =
  let directory = "."
  let listing: text = exec $ printf '%s\n' %{directory}
  let home-listing: text = exec $
    for file in "$HOME"/*; do
      printf '%s\n' "$file"
    done
```

Its payload reaches the shell **verbatim** except for `%{expr}` interpolation:
quotes, parentheses, `#`, `;`, every dollar form the shell itself recognizes
— `$HOME`, `${name}`, `$(date)`, `$1` — and ordinary backslashes all pass
through unexpanded by AgL, so it is the shell, using `exec`'s `env`, that
expands them at run time:

```agl
program def main() -> unit =
  let home: text = exec $ printf '%s' "${HOME}"
```

Juxtaposition never chains, so `print exec $ date` is a parse error; pipe
instead:

```agl
program def main() -> unit =
  print <| exec::[text] $ date
```

The only escape is `\%{`, which writes a literal `%{`. For
example, this command passes `%{literal}` to the shell:

```agl
program def main() -> unit =
  let marker: text = exec $ printf '\%{literal}'
```

The backslash in `\%{` is consumed by the escape, so a payload cannot spell a
literal backslash immediately followed by an interpolation as `\\%{expr}` —
that is a literal backslash followed by a literal `%{expr}`. Interpolate the
backslash from a text literal instead:

```agl
program def main() -> unit =
  let subdir: text = "docs"
  let path: text = exec $ printf '%s' "C:%{"\\"}%{subdir}"
```

`exec $ ...` has the same typing behavior as `exec`: without an expected type
it returns `ExecResult`; a non-`ExecResult`/non-`unit` target parses stdout;
and a `unit` target discards successful output. Use `exec(...)` instead when
the command needs named parsing options (`format`, `strict-json`, or
`on-parse-error`).

## Interpolation in shell templates

The command template uses the same uniform rendering as all other templates
([Strings and interpolation](strings-and-interpolation.md)): `text` values
interpolate verbatim; `int`, `decimal`, and `bool` as plain scalar text;
structured values (`array`, `dict`, record, enum, exception) in AgL form —
single-line, no injected newlines. To interpolate a structured value as JSON,
use an explicit cast: `%{value as json}`.

Interpolated values are inserted **verbatim** into the command string —
there is **no automatic shell quoting**. The workflow author is responsible
for writing shell-safe commands. Unvalidated text (for example, model-produced
content or user input) inside a shell command is an injection hazard unless
the author explicitly handles quoting.

## The two forms of `exec`

`exec`'s behavior depends on the **target type**, determined from context
exactly as for `ask` ([Agent calls](agent-calls.md)).

### Structured form — target is `ExecResult`

When no expected type is present — including when the propagated expectation
is itself a generic type parameter that nothing else in the enclosing
expression pins (the default applies only after sibling constraints), as in
`print <| exec "…"` — or the annotation is `ExecResult`, `exec` returns the
`ExecResult` standard-library record:

```text
stdout:    text
exit-code: int
stderr:    text
timed-out: bool
```

A **nonzero exit does not raise** in this form — the caller branches on
`exit-code`:

```agl
program def main() -> unit =
  let res = exec "ls -la"
  print(res.stdout)
  if res.exit-code != 0 =>
    print("command failed: %{res.stderr}")
```

A spawn failure or timeout raises `ExecError` in this form. A timeout does
not produce an `ExecResult` with `timed-out = true`.

### Parsed form — target is any non-`ExecResult` or `unit` type

When the target type is neither `ExecResult` nor `unit`, `exec` parses stdout
into that type (honouring `format`, `strict-json`, and `on-parse-error`) and
**raises `ExecError` on a nonzero exit**:

<!-- agl-check: fragment -->
```agl
let out: text = exec "cat %{path}"          # stdout verbatim; raises on nonzero
let data: dict[text, int] = exec(           # JSON parsed; raises on nonzero
  "compute-stats --json",
  on-parse-error = Retry(n = 1)
)
```

A nonzero exit and unparseable output both raise `ExecError`.

### Unit form — target is `unit`

When context requires `unit` — for example, a non-final bare expression in a
block or a binding annotated `unit` — `exec` has target type `unit`. For this
unit contract, a nonzero exit raises `ExecError`; successful stdout is
discarded and the call returns `()`:

```agl
program def main() -> unit =
  exec "make build"
  let completed: unit = exec "make lint"
```

Because no output is parsed, `format`, `strict-json`, and `on-parse-error` are
invalid for a `unit` target.

## Execution semantics

1. The rendered command runs via the host shell (`sh -c` semantics),
   un-sandboxed, with the user's privileges, using its `env`, `cwd`, and
   `timeout` arguments.
2. Standard output and standard error are captured.
3. In the **parsed form**, on success (exit status 0), trailing newlines are
   stripped from stdout — as in `$(…)` command substitution — and the result
   is bound at the call's target type. In the **structured form**, stdout
   and stderr are returned as-is.
4. Every execution is traced: command, exit code, duration, stdout, stderr.

## Named parameters

In addition to the spawn parameters above, `exec` accepts the same codec-related
named parameters as `ask`:

- `format` — codec name (a `text` value); normally auto-selected.
- `strict-json` — `bool`; opts the JSON codec into strict parsing.
- `on-parse-error` — `ParsePolicy`; controls retry behavior on parse
  failures in the parsed form. In the structured and unit forms, where no
  stdout parsing happens, passing this parameter is a static error.

## Retries

**Retries re-run the command.** Unlike an `ask` retry — which sends
corrective feedback in the same conversation — an `exec` retry executes the
command again with the same evaluated spawn parameters; each invocation is
traced separately. If every attempt fails to parse, `ExecError` is raised.

## Exceptions

`ExecError` covers a failing, timed-out, or unparseable shell command in the
parsed or unit form:

```agl
program def main() -> unit =
  try
    let data: dict[text, int] = exec "compute-stats --json"
  catch ExecError as e =>
    print "command failed (%{e.exit-code}): %{e.stderr}"
```

In the structured form, `ExecError` is raised for a spawn failure (the shell
itself cannot be launched) or timeout. A nonzero exit instead surfaces in
`exit-code`.

See [Exceptions](exceptions.md) for the full field lists.
