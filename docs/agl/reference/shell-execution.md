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
([Lexical structure](lexical-structure.md)): in call position it denotes the
built-in shell executor; it cannot be declared with `let`/`var`/`param` or
as a function; it cannot be bound as a function value; it remains legal
as a field name. A host may statically disallow shell execution altogether,
in which case every `exec` call is a static error.

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
is an optional idle timeout duration. `exec$` supplies only its command, so it
uses all three defaults.

## Single-argument sugar

With no named arguments, `exec` may be called with the command template
written directly without parentheses:

```agl
program def main() -> unit =
  let _ = exec "make build"
```

With named arguments, parentheses are required.

## Raw-tail `exec$`

`exec$` writes the command directly after the keyword rather than inside a
string template. The inline form takes the rest of its line; the block form
collects one dedented, newline-joined shell script. Both produce the same call
as `exec(<template>)` and accept explicit type arguments. Like every raw-tail
name, `exec$` may follow a projection: `target.exec$ command` is
`target.exec(command)` and uses ordinary member resolution. Type arguments
must touch the name (`exec$::[T]`); in `exec$ ::[T]`, the spaced `::[T]` is
command payload:

```agl
program def main() -> unit =
  let directory = "."
  let listing: text = exec$ printf '%s\n' %{directory}
  let home_listing: text = exec$
    for file in "$HOME"/*; do
      printf '%s\n' "$file"
    done
```

The payload is verbatim shell text, except that inline payloads discard
trailing spaces and tabs, and a block form drops blank lines that trail its
last content line (blank lines before and between content lines are kept).
Quotes, parentheses, `#`, `;`, every dollar form such as `$HOME`, `${name}`,
`$(date)`, and `$1`, and ordinary backslashes all reach the shell unchanged.
Only `%{expr}` interpolates; write `\%{` for a literal `%{`. A raw call needs a
nonempty inline command or a block with at least one nonblank line. For example,
this command passes `%{literal}` to the shell:

```agl
program def main() -> unit =
  let marker: text = exec$ printf '\%{literal}'
```

The backslash in `\%{` is consumed by the escape, so a payload cannot spell a
literal backslash immediately followed by an interpolation as `\\%{expr}` —
that is a literal backslash followed by a literal `%{expr}`. Interpolate the
backslash from a text literal instead:

```agl
program def main() -> unit =
  let subdir: text = "docs"
  let path: text = exec$ printf '%s' "C:%{"\\"}%{subdir}"
```

Raw-tail calls are permitted only in line-final expression positions: block
items, binding or assignment right-hand sides, inline function bodies, eligible
`return` operands, and the final juxtaposition argument (`print exec$ date`).
They cannot appear inside brackets or before more AgL syntax on the same line.
See [Grammar](grammar.md#raw-tail-calls) for the complete position rule.

`exec$` has the same typing behavior as `exec`: without an expected type it
returns `ExecResult`; a non-`ExecResult`/non-`unit` target parses stdout; and a
`unit` target discards successful output. Use `exec(...)` instead when the
command needs named parsing options (`format`, `strict_json`, or
`on_parse_error`) or must occur outside a raw-tail position.

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

When no expected type is present, or the annotation is `ExecResult`, `exec`
returns the `ExecResult` standard-library record:

```text
stdout:    text
exit_code: int
stderr:    text
timed_out: bool
```

A **nonzero exit does not raise** in this form — the caller branches on
`exit_code`:

```agl
program def main() -> unit =
  let res = exec "ls -la"
  let _ = print(res.stdout)
  let _ = if res.exit_code != 0 =>
    print("command failed: %{res.stderr}")
```

A spawn failure or timeout raises `ExecError` in this form. A timeout does
not produce an `ExecResult` with `timed_out = true`.

### Parsed form — target is any non-`ExecResult` or `unit` type

When the target type is neither `ExecResult` nor `unit`, `exec` parses stdout
into that type (honouring `format`, `strict_json`, and `on_parse_error`) and
**raises `ExecError` on a nonzero exit**:

<!-- agl-check: fragment -->
```agl
let out: text = exec "cat %{path}"          # stdout verbatim; raises on nonzero
let data: dict[text, int] = exec(           # JSON parsed; raises on nonzero
  "compute-stats --json",
  on_parse_error = Retry(n = 1)
)
```

A nonzero exit and unparseable output both raise `ExecError`.

### Unit form — target is `unit`

When context requires `unit` — for example, a non-final bare expression in a
block or a binding annotated `unit` — `exec` has target type `unit`. For this
unit contract, a nonzero exit raises `ExecError`; successful stdout is
discarded and the call returns `void`:

```agl
program def main() -> unit =
  let _ = exec "make build"
  let completed: unit = exec "make lint"
```

Because no output is parsed, `format`, `strict_json`, and `on_parse_error` are
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
- `strict_json` — `bool`; opts the JSON codec into strict parsing.
- `on_parse_error` — `ParsePolicy`; controls retry behavior on parse
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
  let _ = try
    let data: dict[text, int] = exec "compute-stats --json"
  catch ExecError as e =>
    print "command failed (%{e.exit_code}): %{e.stderr}"
```

In the structured form, `ExecError` is raised for a spawn failure (the shell
itself cannot be launched) or timeout. A nonzero exit instead surfaces in
`exit_code`.

See [Exceptions](exceptions.md) for the full field lists.
