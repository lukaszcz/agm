# Shell Execution: `exec`

[← Index](index.md)

`exec` runs a shell command and yields its output. It is **call-shaped**,
exactly like an agent call — a contextual keyword followed by optional call
options and a template:

```agl
let listing = exec "ls -la ${dir}"
let data: json = exec[on_parse_error: abort] "cat ${path}"
exec "make build"
```

Like `prompt`, `exec` is a contextual keyword
([Lexical structure](lexical-structure.md)): in call position it denotes the
built-in shell executor; it cannot be declared with `let`/`var`/`input` or
declared as an agent name; it remains legal as a field name. A host may
statically disallow shell execution altogether, in which case every `exec`
call is a static error.

## Interpolation in shell templates

The command template uses the same uniform rendering as all other templates
([Strings and interpolation](strings-and-interpolation.md)): `text` values
interpolate verbatim; `int`, `decimal`, and `bool` as plain scalar text;
structured values (`list`, `dict`, records, enums, `json`, exceptions) as
pretty JSON (2-space indent).

Interpolated values are inserted **verbatim** into the command string —
there is **no automatic shell quoting**. The workflow author is responsible
for writing shell-safe commands. Unvalidated text (for example, model-produced
content or user-provided input) inside a shell command is an injection hazard
unless the author explicitly handles quoting.

```agl
exec "grep -F ${needle} ${file}"
exec "sh -c ${script}"
```

## Execution semantics

1. The rendered command runs via the host shell (`sh -c` semantics),
   un-sandboxed, with the user's privileges. The host's configured idle
   timeout applies (no timeout if the host sets none). Standard output and
   standard error are captured.
2. On **success** (exit status 0), trailing newlines are stripped from the
   captured stdout — as in `$(…)` command substitution — and the result is
   bound at the call's target type.
3. A **nonzero exit** or a **timeout** raises `ExecError`, carrying the
   rendered command, exit code, captured stdout and stderr (trailing
   newlines stripped), and a `timed_out` flag.
4. A **spawn failure** (the shell itself cannot be launched) also raises
   `ExecError`, with `exit_code` `-1` and empty output.
5. Every execution is traced: command, exit code, duration, stdout, stderr.

An exit status of zero with empty output is a success: a `text` target binds
the empty string; a structured target proceeds to parsing.

## Typed `exec` results

The target type is determined from context exactly as for agent calls —
annotation, `set` target, propagated expectation, else `text`
([Agent calls](agent-calls.md)):

- A `text` target binds the stripped stdout verbatim.
- Any other target engages the same codec machinery as agent output: the
  stdout is parsed (leniently by default, strictly under
  `strict_json: true`), validated against the type's schema, and converted
  to a typed value. `on_parse_error` policies apply.

```agl
let stats: dict[text, int] = exec[on_parse_error: retry[1]] "compute-stats --json"
```

**Retries re-run the command.** Unlike an agent retry — which sends
corrective feedback to the same conversation — an `exec` retry simply
executes the command again; each invocation is traced separately. If every
attempt fails to parse, `AgentParseError` is raised with agent name `exec`.

`ExecError` (a failing command) and `AgentParseError` (unparseable output
from a succeeding command) are distinct and independently catchable:

```agl
try
  let data: json = exec "cat ${path}"
catch ExecError as e =>
  print "command failed (${e.exit_code}): ${e.stderr}"
catch AgentParseError as e =>
  print "not valid JSON: ${e.raw}"
```

See [Exceptions](exceptions.md) for the full field lists.
