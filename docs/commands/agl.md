# AgL workflow DSL

AGM runs AgL workflows with [`agm exec`](exec.md) or in the interactive REPL
described here. The AgL language itself is documented in the
[AgL language reference](../agl/reference/index.md).

## `agm repl` — interactive session

```text
agm repl [--strict-json|--no-strict-json]
         [--max-iters N] [--max-call-depth N] [--agent AGL_LITERAL] [--confirm-agents]
         [--quiet] [--dry-run] [--no-stdlib] [--log|--log-file PATH|--no-log]
```

Start an interactive read-eval-print loop for AgL. Unlike `agm exec`, which runs a
whole program from a fresh environment, the REPL keeps a **persistent session**: each
entry is parsed, statically checked (including pattern coverage), and evaluated once against an environment that
accumulates bindings, types, and declarations across entries, so earlier results stay
available and agent calls fire exactly once.

The REPL reuses `[exec]` settings for `default-agent`, the max-iters valve,
call-depth limit, JSON strictness, and timeout. Imported-module params resolve from
their qualified config tables; params declared directly at the prompt use only source
defaults (or are required). Like `agm exec`, each typed
`Agent` value selects its own backend command; settings do not select it. Like
`agm exec`, `--agent` combined with `--no-stdlib` still fails — during session
initialization, before the prompt appears — if the session never loads `std/config`;
`[exec] default-agent` is simply inert in that same situation.

Like `agm exec`, the REPL supplies an automatic `import std/core::*` prelude to
each loaded program, so standard-library names such as `Option`, `Some`, and
`None` are available unqualified from a fresh prompt. An explicit import whose
expansion includes `std/core` supplies that contribution instead, so plain
`import std/core` leaves core names qualified-only. Pass `--no-stdlib` to disable
the prelude for each entry and its library modules; explicit imports still work,
including after `:reset`.
Entering a bare type name displays the type; an unapplied generic type name such as
`Option` displays its generic definition instead of being evaluated as a value.

Importing a library module that declares `extern def` (see
[Python FFI](../agl/reference/ffi.md)) works normally in the REPL; its companion
Python file imports once for the session, not once per entry. Its ordinary Python
module globals therefore last for the session; `:reset` discards that cached
companion and a later import creates new globals. A companion value obtained
through `runtime.state(...)` is different: it belongs to the fresh interpreter
that evaluates one entry and ends with that entry, even while the companion
module remains cached. A direct entry typed at the prompt (with no backing file
of its own) may not declare `extern def` itself.

For the same reason, `resource` and `resource-dir` cannot be called from a direct
entry: they anchor at the declaring module's file. An imported file-backed module
uses them normally.

### Entry editing

- Multiline editing is **AgL-aware**: pressing Enter on an unterminated block
  (`record`, `enum`, `if`, `case`, `try`, `do`, …) or a line-final raw-tail header
  such as `exec!`/`ask!` opens a continuation line (`...>`); a complete entry submits.
  Pressing Enter on a blank continuation line force-submits even an unfinished buffer
  so you can always escape.
- Syntax highlighting and tab-completion are driven from the live session.
  Highlighting colours keywords, string/number literals, operators, the builtin types
  (`text`, `int`, `decimal`, `bool`, `json`, `array`, `dict`, `unit`), and the types and
  constructors declared in the session or in the line being typed. Declaration sites
  colour by position (the name after `record`/`enum`/`type` is a type; an inline enum
  member after `|` is a constructor), so a type and a like-named constructor are distinguished
  even while you type the declaration. At a use site, a constructor call (`Box(…)`,
  `ok::[…](…)`) colours as a constructor and a type annotation as a type. Completion
  offers AgL keywords, current binding names, and meta-command names.
- Two colour themes are available: **dark** (VS Code Dark+) and **light** (VS Code
  Light+). The default is **auto**, which detects the terminal background from the
  `$COLORFGBG` environment variable (set by most terminal emulators; falls back to
  dark). Use `:theme dark|light|auto` to switch at runtime; the choice is saved to
  `~/.agm/config.toml` under `[repl] theme`. You can also set `theme = "light"`
  directly in the config file.
- Command history persists under `~/.agm/repl_history`.
- Press Ctrl-C to cancel the current entry without exiting. During a live agent call,
  Ctrl-C interrupts the call and stops the current entry; effects completed before
  cancellation remain visible, and unreached operations do not run.

### Meta-commands

Meta-commands begin with a leading `:` (which never collides with AgL syntax):

| Command | Action |
|---------|--------|
| `:help` | List the available meta-commands |
| `:quit` / `:exit` (or Ctrl-D) | Exit the REPL |
| `:reset` | Clear the whole session (bindings, types, declarations, params, imports, and uses) |
| `:type EXPR` | Type-check `EXPR` against the session and print its type (no eval) |
| `:bindings` / `:env` | List current bindings as `name : Type = value` |
| `:params` | List declared params and their resolved values |
| `:set echo on\|off` | Toggle result echoing |
| `:agent confirm\|auto` | Switch the agent-call mode (or report it with no argument) |
| `:load FILE` | Load a saved transcript by its original entries, or an ordinary `.agl` file one item per entry |
| `:save FILE` | Write the accumulated session source and entry boundaries to a transcript |
| `:theme [dark\|light\|auto]` | Show or switch the syntax-highlighting theme; saves to `~/.agm/config.toml` |

### Agent-call confirmation

- By default the REPL is in **auto** mode: agent calls fire immediately without
  prompting, matching `agm exec`.
- `--confirm-agents` (or `:agent confirm`) starts/switches to **confirm** mode: before
  every live agent call it shows the callee and the rendered prompt (truncated, with a
  `[v]iew` option to print the full text) and asks `[Y]es / [n]o / [a]lways`. `yes` runs
  the call, `no` aborts the entry (rolling its bindings back), and `always` switches the
  session to auto mode for the rest of the session.
- `exec` shell calls are **not** gated; only agent calls are confirmed.

### Options

- `--strict-json` / `--no-strict-json`: Set JSON-codec strictness for agent output
  (lenient recovery is the default), as for `agm exec`.
- `--max-iters N`, `--max-call-depth N`, `--agent AGL_LITERAL`: As for `agm exec`.
- `--confirm-agents`: Start in confirm mode, asking before each agent call (the default
  is auto; see [Agent-call confirmation](#agent-call-confirmation)).
- `--quiet`: Suppress the automatic echoing of entry results.
- `--no-stdlib`: Disable the automatic `import std/core::*` prelude for each
  loaded REPL program (its entry and library modules). Explicit standard-library imports remain available;
  `:reset` retains this launch-time choice.
- `--log` / `--log-file PATH` / `--no-log`: Control trace logging (off by default), as
  for `agm exec`. With `--log-file` each evaluated entry appends its JSONL trace records
  (one trace *run* per entry) to `PATH`. The three are mutually exclusive, and
  `--dry-run` writes no trace.
- `--dry-run`: Statically check only. Each entry runs the full static pipeline (parse /
  resolve / typecheck / match compilation) but is **never evaluated**, so no agent or `exec` calls fire and
  no bindings are persisted. The inferred type is echoed instead of a value
  (`name : Type` for a binding, `: Type` for a bare expression), making it a quick way
  to explore types interactively.

### Evaluation notes

- Blank lines and comment-only entries (everything after a `#` is a comment) are a
  no-op: pressing Enter on them simply returns a fresh prompt, with no evaluation and no
  error.
- **Declaration entries** echo the declared name followed by `declared`. A scoped
  declaration path echoes the full path (`Tools::twice declared`), and a `scope … end`
  region echoes its path (`Tools declared`). `import`, `use`, `export`, and
  fixity declarations echo nothing.
- **Bare type expressions** typed at the prompt are recognized as types rather than
  value expressions: entering `int`, a declared `enum`/`record`/`type` name, or a
  parameterized form like `array[int]` or `(int) -> bool` echoes the resolved type (e.g.
  `<type: int>`) instead of reporting ``'X' is not defined.``. This is a REPL
  convenience only — the language is unchanged, and names that are also values (a record
  constructor, a binding) keep evaluating normally.
- **Engine settings** are set at the REPL prompt by importing `std/config` and writing a
  qualified target (`std/config::max-iters := 3`). The write takes effect positionally,
  so subsequent entries in the session see the new value, including when a later
  expression in the writing entry fails; `log` or `log-file` reconfigures the
  trace destination. `:reset` clears
  the session, restoring the settings to the CLI/`[exec]` defaults set before the loop
  starts.

### Exit codes

The REPL itself only fails before the loop starts; ordinary per-entry errors are
reported inline and never exit the process. `std/process::exit(code)` is the
exception: it terminates the REPL host with its portable `0..255` status after
finalizing that entry's trace. A blank or non-string `--agent`/`[exec] default-agent`
value is one such pre-loop failure. A syntactically present but malformed literal —
unparseable, the wrong type, a non-constant expression, or an unknown engine key — is
also a pre-loop failure: session initialization loads the standard library and splices the
literal into `std/config` before the console starts, so it is resolved, type-checked,
and constant-checked (and any rejection reported, naming the flag or config key) before
the banner appears, with nothing printed and no entry accepted. An `AgentCommand(...)`
whose command text does not shell-split is deferred until the first entry reaches
interpreter construction, because session initialization does not construct an
interpreter. Construction validates the winning value before any statement executes, so
even an entry that performs no agent dispatch reports the error inline without exiting
the REPL.

| Code | Meaning |
|------|---------|
| `0` | The session ended normally (`:quit`/`:exit` or Ctrl-D) |
| `1` | Pre-loop setup failure: a blank/non-string, unparseable, wrongly typed, or non-constant `[exec] default-agent` or `--agent` value, or an unwritable `--log-file` — reported before the prompt appears |

### Examples

```bash
# Launch a session; build up state line by line.
agm repl
agl> let greeting = "hello"
greeting : text = hello
agl> :type greeting
text
agl> :bindings
greeting : text = hello
agl> :quit

# Confirm each agent call before dispatching it.
agm repl --confirm-agents

# Explore types only — no agent or exec calls fire, nothing is persisted.
agm repl --dry-run
agl> 1 + 2
: int
```
