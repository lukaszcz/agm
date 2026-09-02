# `agm check`

[← Commands](index.md)


```text
agm check [-I DIR]... [--no-stdlib] FILE...
```

Run the full **static** AgL pipeline — parse, module loading, scope resolution, type
checking, match compilation, and lowering — over each `FILE` and report GNU-style
diagnostics. `agm check` never evaluates anything and never runs an agent.

Unlike `agm exec`, no `FILE` needs to declare a `program def`: a plain library module
(the case `agm exec --dry-run` rejects) can be checked on its own. A `program def`
present in a file is validated as part of the module it lives in, but `check` never
selects one, never resolves or checks its params against configuration, and never runs
it — there are no `--<param>` options and no `-p`/`--program` selector.

Each `FILE` is checked independently, in argument order, and **every** `FILE` is checked
even when an earlier one failed. A clean `FILE` produces no output at all.

This is a different check from [`agm pkg check`](pkg.md), which validates a package
directory's manifest and module-tree discipline (dependency declarations, mounted roots,
registered command references) rather than the correctness of individual AgL programs.

### Module resolution

`agm check` resolves each `FILE`'s imports through exactly the same module-root logic as
`agm exec` — the file's own directory (and its containing development package, when
applicable), the selected standard library, the AGM home's global `lib` directory,
`[modules] roots` from config, and any `-I`/`--module-path` roots — assembled fresh for
each `FILE`, since module roots are anchored at the file being checked. See
[`agm exec` → Module resolution](exec.md#module-resolution) for the full root-assembly
rule.

### Options

- `-I DIR`, `--module-path DIR`: Add `DIR` as an additional module search root
  (repeatable), resolved relative to the invocation working directory, exactly as for
  `agm exec`.
- `--no-stdlib`: Disable automatic `std/prelude` opening throughout each checked file (the
  file itself and its library modules). Explicit `import std/prelude` is unaffected.
- `--dry-run`: Accepted for consistency with every other command, but meaningless here —
  `check` never has a side effect to skip.

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | No `FILE` produced an error-severity diagnostic |
| `1` | Some `FILE` produced an error-severity diagnostic, was unreadable or missing, or had an invalid module-root configuration |

### Diagnostics and warnings

Diagnostics print to stderr in the same compiler-style form `agm exec` uses:
a location, then `error:` or `warning:`, then the message. Most diagnostics carry a
span, so the location takes one of four shapes:

```text
path:line: error: message
path:line:col: error: message
path:line:col-endcol: error: message
path:line:col-endline:endcol: error: message
```

A related note is indented two spaces under its diagnostic. A diagnostic originating in
an imported module carries that module's own path rather than the checked `FILE`'s.
Advisory **warnings** (a TAB-indented line, for example) are printed but never affect the
exit code — only error-severity diagnostics do.

### Example

```bash
agm check workflow.agl
agm check lib/*.agl                    # library modules, no `program def` required
agm check -I vendor/ workflow.agl       # add an extra module search root
agm check --no-stdlib lib/helpers.agl
```

A clean file produces no output and exits `0`:

```bash
$ agm check lib/helpers.agl
$ echo $?
0
```

A file with a static error reports it on stderr and exits `1`:

```bash
$ agm check bad.agl
bad.agl:3:11: error: 'undefined_name' is not defined.
$ echo $?
1
```
