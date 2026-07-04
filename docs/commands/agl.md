# AgL workflow DSL

| Command | Description |
|---------|-------------|
| `agm exec [--strict-json\|--no-strict-json] [--max-iters N] [--max-call-depth N] [--runner COMMAND] [--log\|--log-file PATH\|--no-log] [--no-stdlib] [-I DIR]... (FILE \| -c COMMAND) [--PARAM VALUE]...` | Execute an AgL workflow program |
| `agm repl [--strict-json\|--no-strict-json] [--max-iters N] [--max-call-depth N] [--runner COMMAND] [--confirm-agents] [--quiet] [--log\|--log-file PATH\|--no-log]` | Start an interactive AgL REPL |

AGM runs AgL (Agent Language) workflow programs two ways: `agm exec` runs a whole
program from a fresh environment, and `agm repl` evaluates entries interactively in a
persistent session that reuses the same configuration. The AgL language itself is
documented in the [AgL language reference](../agl/reference/index.md).

## `agm exec` — run a program

```text
agm exec [--strict-json|--no-strict-json]
         [--max-iters N] [--max-call-depth N] [--runner COMMAND]
         [--log|--log-file PATH|--no-log]
         [--no-stdlib]
         [-I DIR]...
         (FILE | -c COMMAND) [--PARAM VALUE]...
```

Execute an AgL workflow program, either from a source `FILE` or from inline program
text given with `-c`/`--command`. The two are mutually exclusive, and exactly one is
required.

### Module resolution

`agm exec` supports programs that import library modules. The runtime assembles an
**unordered set of search roots**:

- the directory of `FILE` (or the working directory for `-c`),
- the standard-library root (`~/.agm/stdlib` after `just install`),
- the global library root (`~/.agm/lib`, overridable via `[modules] lib_root` in config),
- any roots declared under `[modules] roots` in any config layer,
- any roots added with `-I`/`--module-path`.

A module name that resolves to exactly one file across all roots succeeds; zero files,
or two or more distinct files, are static errors (exit 1 with a diagnostic).

### Options

- `-c COMMAND`, `--command COMMAND`: Execute the AgL program given as `COMMAND`
  directly, instead of reading the program from `FILE`.
- `--PARAM VALUE`: Provide a value for a `param` declaration. Each declared param
  becomes a program-specific option; booleans use `--name` / `--no-name`. Values for
  `text` params are taken verbatim; every other scalar or structured type
  (`int`/`decimal`/`bool`/`json`/`list`/`dict`/`record`/`enum`) is parsed as exactly
  one strict JSON value and validated against the declared type. Missing required
  params or invalid values are reported before any agent runs. Run
  `agm exec FILE --help` to show the discovered param options for that program.
- `-I DIR`, `--module-path DIR`: Add `DIR` as an additional module search root
  (repeatable), resolved relative to the invocation working directory. See
  [Module resolution](#module-resolution). This is also how e2e/fixture tests point
  `agm exec` at test-specific module roots.
- `--no-stdlib`: Do not automatically open `std.core` in the entry module. Explicit
  `import std.core` still uses the normal module import semantics.
- `--strict-json`: Require agents to return exactly one bare JSON value (no fences,
  prose, or repair). Overridable per call site with the `strict_json:` named argument
  to `ask`.
- `--no-strict-json`: Use lenient JSON recovery (the default): the runtime recovers
  exactly one JSON value from chatty output (stripping fences/prose, repairing
  trivially malformed JSON), then validates it strictly against the schema. The
  recovered (normalized) value is traced alongside the raw output.
- `--max-iters N`: Override the host's `max-iters` safety valve, which caps
  **unbounded** loops (a bare `while … do … done` or `do … until E` with no
  `[n]` bound and no `for` clause) at `N` body executions, raising
  `MaxIterationsExceeded`. Self-bounded loops (`for`, `do[n]`) are never cut
  short by this valve. The valve is **off by default**; setting this option (or
  `max-iters` in config / a source `config max-iters = N` declaration) turns it
  on. See [Control flow](../agl/reference/control-flow.md).
- `--max-call-depth N`: Override the maximum recursion call depth (CLI >
  `[exec] max-call-depth` config; the canonical default is 256). Exceeding it
  raises `RecursionError`.
- `--runner COMMAND`: Override the default agent runner command (backs `ask` and any
  declared agent without its own command). See [runner precedence](#agents-and-runner-precedence).
- `--log` / `--log-file PATH` / `--no-log`: Control trace logging, which is **off by
  default**. `--log` enables it with an auto-generated timestamped path under
  `.agent-files/`; `--log-file PATH` writes a structured JSONL trace to `PATH`;
  `--no-log` disables it, overriding a `config log = true` source declaration or
  `[exec] log = true` setting. The three are mutually exclusive (at most one may be given).
- `--dry-run`: Run the full static pipeline, param validation, and contract
  materialization, then stop before evaluating any expression (static errors exit 1; a
  clean check exits 0 with no program output). When the check succeeds and one or more
  agent-call or `exec` sites exist, the static call-site inventory is printed to stdout:

  ```
  call-sites:
    line N:C: <callee> → <target-type> [<codec>[, schema: yes][, policy: <policy>]]
  ```

  Each entry shows the 1-based source line and column (`N:C`), the callee name (`ask`,
  `exec`, or a registered agent name), the target type, the selected codec (`text` or
  `json`), and optionally whether a JSON Schema is attached (`schema: yes`) and the
  effective parse-failure policy (`abort` or `retry[N]`). When no agent calls are
  present, no inventory is printed.

### Agents and runner precedence

Named agents must be **declared in the program source** with `agent NAME`, optionally
carrying a runner hint as `agent NAME = "runner"`. Calling an undeclared name is a
static binding error (exit 1). The contextual `ask` (default agent) and `exec` (shell)
are built in and need no declaration.

For each declared agent, `agm exec` resolves the command that runs it by the following
precedence (highest to lowest):

| Rung | Source |
|------|--------|
| 1 | `[exec.agents.<name>]` (config, per-agent) — backs a declared name, overriding any source hint |
| 2 | the source `agent NAME = "…"` runner string |
| 3 | `--runner COMMAND` (CLI flag) |
| 4 | `config runner = "…"` source declaration (default runner for all agents) |
| 5 | `[exec] runner` (config) |
| 6 | `[loop] runner` (config) |
| 7 | `claude -p` (built-in default) |

A `[exec.agents.<name>]` entry for a name the program never declares is a host
configuration error. Because the default runner is always the floor (rung 7), every
declared agent resolves under `agm exec` even with no config and no source hint. Runner
strings (config or source hint) support the `%%` / `%{PROMPT_FILE}` placeholders for
the rendered prompt-file path.

### Configuration

The `[exec]` section in `config.toml` supplies the engine defaults that CLI flags and
source config declarations can override:

```toml
[exec]
runner = "claude -p"        # default agent runner
strict-json = false         # lenient JSON recovery is the default
max-iters = 5               # host safety valve for unbounded loops (off when unset)
timeout = "30m"             # shell-exec idle timeout
log = false                 # trace logging off by default; set true to enable
# log-file = "trace.jsonl" # explicit trace path (omit for auto timestamped path)

[exec.agents]
reviewer = "claude -p"      # per-agent runner commands; the name must be
                            # declared in the program source (`agent reviewer`),
                            # and this entry overrides any source runner hint
```

`[exec.<command>]` sub-tables provide per-command overrides of the base `[exec]`
settings. The name `agents` is reserved for the structural `[exec.agents]` map and is
never treated as a per-command override.

#### Source-level config declarations

An AgL program may set exec options as **config declarations** at the program root.
Each declaration is also a **readable immutable binding** usable in any expression:

```agl
config log = true             # enable trace logging for this program
config log-file = "trace.log" # explicit trace path
config strict-json = true     # require bare JSON from agents
config max-iters = 10         # host safety valve cap for unbounded loops
config runner = "claude -p"   # default agent runner
config timeout = "30s"        # shell-exec idle timeout
param spec
let result = ask "Process ${spec}"
print result
```

Precedence differs by key type:

- **Engine keys** (`runner`, `log`, `strict-json`, `max-iters`, `log-file`, `timeout`):
  `CLI > source value > [<program>].X > [exec].X > engine default`
- **Param values** (`param NAME`):
  `CLI > [<program>].NAME > source default > required error`

For example, `--no-log` overrides `config log = true`, and `config max-iters = 10`
overrides `[exec] max-iters = 5`.

The three eval-consumed keys (`strict-json`, `max-iters`, `timeout`) take effect at
their declaration point (**effect-at-binding**): expressions after the declaration see
the updated setting. The remaining keys (`runner`, `log`, `log-file`) are
**start-resolved** — declare them at the top of the program so the agent factory and
trace infrastructure see the chosen values before anything runs.

Note: `config timeout` governs only the **shell-exec** timeout. The agent idle timeout
is start-resolved from CLI / `[exec]` and cannot be changed mid-program.

A bad `config timeout = "…"` source value is a runtime AgL error (exit 2), because
source config declarations are runtime-evaluated expressions. A bad `--timeout` or
`[exec].timeout` value is a pre-execution error (exit 1).

`--no-log-file` clears only the in-program `config log-file` binding; a log-file path
set via `[exec] log-file` or auto-assigned by `--log` still applies. Use `--no-log` to
disable tracing entirely.

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | The workflow completed successfully |
| `1` | Pre-execution failure: unreadable file, static lex/parse/scope/typecheck diagnostics, host configuration error, or param validation failure |
| `2` | The workflow executed but ended with an uncaught AgL exception |

### Diagnostics and warnings

- Error-severity diagnostics (static lex/parse/scope/typecheck errors, host
  configuration errors, param validation failures) and uncaught AgL exceptions are
  printed to stderr and determine the exit code per the table above.
- Advisory **warnings** (for example a non-exhaustive `case` over an enum that omits
  some variants) are a separate channel. They are printed to stderr with a `warning:`
  prefix (`warning: line N: message`) to disambiguate them from errors, and they never
  affect the exit code — the program still runs to completion. Program `print` output
  goes to stdout, kept clean of diagnostics.

## `agm repl` — interactive session

```text
agm repl [--strict-json|--no-strict-json]
         [--max-iters N] [--max-call-depth N] [--runner COMMAND] [--confirm-agents]
         [--quiet] [--log|--log-file PATH|--no-log]
```

Start an interactive read-eval-print loop for AgL. Unlike `agm exec`, which runs a
whole program from a fresh environment, the REPL keeps a **persistent session**: each
entry is parsed, type-checked, and evaluated once against an environment that
accumulates bindings, types, and declarations across entries, so earlier results stay
available and agent calls fire exactly once.

The REPL reuses the `[exec]` configuration (runner, per-agent commands,
max-iters valve, call-depth limit, JSON strictness, timeout), so an interactive
session evaluates entries with the same agent backing a batch `agm exec` run
would use.

Like `agm exec`, the REPL automatically opens `std.core`, so standard-library names
such as `Option`, `Some`, and `None` are available unqualified from a fresh prompt.
Entering a bare type name displays the type; an unapplied generic type name such as
`Option` displays its generic definition instead of being evaluated as a value.

### Entry editing

- Multiline editing is **AgL-aware**: pressing Enter on an unterminated block
  (`record`, `enum`, `if`, `case`, `try`, `do`, …) opens a continuation line (`...>`);
  a complete entry submits. Pressing Enter on a blank continuation line force-submits
  even an unfinished buffer so you can always escape.
- Syntax highlighting and tab-completion are driven from the live session.
  Highlighting colours keywords, string/number literals, operators, the builtin types
  (`text`, `int`, `decimal`, `bool`, `json`, `list`, `dict`, `unit`), and the types and
  constructors declared in the session or in the line being typed. Declaration sites
  colour by position (the name after `record`/`enum`/`type` is a type; an enum variant
  after `|` is a constructor), so a type and a like-named constructor are distinguished
  even while you type the declaration. At a use site, a constructor call (`Box(…)`,
  `ok::[…](…)`) colours as a constructor and a type annotation as a type. Completion
  offers AgL keywords, current binding names, available agent names, and meta-command
  names.
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
| `:reset` | Clear the whole session (bindings, types, declarations, params) |
| `:type EXPR` | Type-check `EXPR` against the session and print its type (no eval) |
| `:bindings` / `:env` | List current bindings as `name : Type = value` |
| `:agents` | List available agents and report the current agent-call mode |
| `:params` | List declared params and their resolved values |
| `:set echo on\|off` | Toggle result echoing |
| `:agent confirm\|auto` | Switch the agent-call mode (or report it with no argument) |
| `:load FILE` | Run an `.agl` file's items into the session, one per entry |
| `:save FILE` | Write the accumulated session source to a file |
| `:theme [dark\|light\|auto]` | Show or switch the syntax-highlighting theme; saves to `~/.agm/config.toml` |

### Agent-call confirmation

- By default the REPL is in **auto** mode: agent calls fire immediately without
  prompting, matching `agm exec`.
- `--confirm-agents` (or `:agent confirm`) starts/switches to **confirm** mode: before
  every live agent call it shows the callee and the rendered prompt (truncated, with a
  `[v]iew` option to print the full text) and asks `[Y]es / [n]o / [a]lways`. `yes` runs
  the call, `no` aborts the entry (rolling its bindings back), and `always` switches the
  session to auto mode for the rest of the session.
- `exec` shell calls are **not** gated in this version; only agent calls are confirmed.

### Options

- `--strict-json` / `--no-strict-json`: Set JSON-codec strictness for agent output
  (lenient recovery is the default), as for `agm exec`.
- `--max-iters N`, `--max-call-depth N`, `--runner COMMAND`: As for `agm exec`.
- `--confirm-agents`: Start in confirm mode, asking before each agent call (the default
  is auto; see [Agent-call confirmation](#agent-call-confirmation)).
- `--quiet`: Suppress the automatic echoing of entry results.
- `--log` / `--log-file PATH` / `--no-log`: Control trace logging (off by default), as
  for `agm exec`. With `--log-file` each evaluated entry appends its JSONL trace records
  (one trace *run* per entry) to `PATH`. The three are mutually exclusive, and
  `--dry-run` writes no trace.
- `--dry-run`: Type-check only. Each entry runs the full static pipeline (parse /
  resolve / typecheck) but is **never evaluated**, so no agent or `exec` calls fire and
  no bindings are persisted. The inferred type is echoed instead of a value
  (`name : Type` for a binding, `: Type` for a bare expression), making it a quick way
  to explore types interactively.

### Evaluation notes

- Blank lines and comment-only entries (everything after a `#` is a comment) are a
  no-op: pressing Enter on them simply returns a fresh prompt, with no evaluation and no
  error.
- **Bare type expressions** typed at the prompt are recognized as types rather than
  value expressions: entering `int`, a declared `enum`/`record`/`type` name, or a
  parameterized form like `list[int]` or `(int) -> bool` echoes the resolved type (e.g.
  `<type: int>`) instead of reporting ``'X' is not defined.``. This is a REPL
  convenience only — the language is unchanged, and names that are also values (a record
  constructor, a binding) keep evaluating normally.
- **Config declarations** (`config KEY = VALUE`) are fully supported at the REPL prompt.
  Effect-at-binding applies immediately: the eval-consumed keys (`strict-json`, `max-iters`,
  `timeout`) take effect from the next entry; the start-resolved keys (`runner`, `log`,
  `log-file`) are set for subsequent entries in the session. `:reset` clears all config
  bindings. CLI flags and `[exec]` config set session-wide defaults before the loop starts.

### Exit codes

The REPL itself only fails before the loop starts; per-entry errors are reported inline
and never exit the process.

| Code | Meaning |
|------|---------|
| `0` | The session ended normally (`:quit`/`:exit` or Ctrl-D) |
| `1` | Pre-loop setup failure: an invalid `[exec]` configuration or `--runner` command, or an unwritable `--log-file` — reported before the prompt appears |

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
