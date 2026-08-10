# `agm exec`

[← Commands](index.md)


```text
agm exec [--strict-json|--no-strict-json]
         [--max-iters N] [--max-call-depth N] [--agent AGL_LITERAL]
         [--timeout DURATION|--no-timeout] [--dry-run]
         [--log|--log-file PATH|--no-log] [--no-log-file]
         [--no-stdlib]
         [-I DIR]... [-p PATH]
         (FILE | PACKAGE/MODULE::PROGRAM | -c COMMAND) [--PARAM VALUE]...
```

Execute an AgL workflow program, either from a source `FILE` or from inline program
text given with `-c`/`--command`. The two are mutually exclusive, and exactly one is
required. `PACKAGE/MODULE::PROGRAM` resolves its module through the active package
selection, for example `agm exec review_tools/review::main`. An existing `FILE` path always
takes precedence, even when its name contains `::`.

A file must declare at least one `program def` function. `exec` initializes the
linked program and invokes its sole entry implicitly; if it declares several, select
one with `-p`/`--program` using its declaration path (for example,
`review::main`). Inline `-c` source is wrapped in a synthetic `program def main`
when it does not declare an entry itself.

### Module resolution

`agm exec` supports programs that import library modules. The runtime assembles an
**unordered set of search roots**:

- the directory of `FILE` (or the working directory for `-c`),
- when that directory is inside a development package, its containing package and the recursive closure of dependencies declared with relative `path` sources in their manifests,
- the selected standard library: the active immutable `$AGM_HOME/packages/std/<AGM_VERSION>/` package when it matches the running binary, otherwise the in-repo `stdlib/` source-checkout fallback; `AGM_STDLIB` overrides this selection without mounting the active package as an additional root,
- the global library root (`~/.agm/lib`, relocated with `AGM_HOME` and overridable via `[modules] lib_root` in config),
- any roots declared under `[modules] roots` in any config layer,
- any roots added with `-I`/`--module-path`.

The independently loaded `[modules] lib_root` and `roots` settings expand `%{VAR}` the same
leniently as other path-valued settings; see
[Path-valued settings](config.md#path-valued-settings) for the exact rule.

Set `AGM_HOME` to relocate the entire `~/.agm` directory (config, prompts, sandbox settings, global library, and managed package store); set `AGM_STDLIB` to point only the standard-library root elsewhere.

A module name that resolves to exactly one file across all roots succeeds; zero files,
or two or more distinct files, are static errors (exit 1 with a diagnostic).

A module (or a file-backed `FILE` entry program) that declares `extern def`
(see [Python FFI](../agl/reference/ffi.md)) requires a companion Python file
at the same path with a `.py` suffix; unlike a module import, this path is
derived rather than searched, so module-root ambiguity never applies to it. A
missing companion, or a companion missing the extern's declared name as a
callable attribute, is a diagnostic reported before the program runs, exactly
like any other static error.

### Options

- `-c COMMAND`, `--command COMMAND`: Execute the AgL program given as `COMMAND`
  directly, instead of reading the program from `FILE`.
- `-p PATH`, `--program PATH`: Select a `program def` entry by its declaration
  path. This accepts entry-file paths such as `main` or `review::main`; with an
  installed `PACKAGE/MODULE::PROGRAM` reference, it overrides that reference's
  program path while retaining its module.
- `--PARAM VALUE`: Provide a value for a `param` declaration. The selected
  program exposes params from its module and transitive imports; each becomes a
  program-specific option. Booleans use `--name` / `--no-name`. A param
  declared inside a named scope region uses its full path spelling, e.g.
  `--Deploy::region`. The module-qualified spelling, such as
  `--review-tools/judge::Deploy::region`, is always accepted and disambiguates
  params with the same short spelling. For an entry-file param that needs
  qualification, use `--@entry::Deploy::region`; this avoids exposing the
  pipeline's internal entry sentinel. Qualified config tables supply values
  for every parameter in the selected inventory. Values for `text` params are taken verbatim; every other
  scalar or structured type (`int`/`decimal`/`bool`/`json`/`array`/`dict`/`record`/
  `enum`) is parsed as exactly one strict JSON value and validated against the
  declared type. Missing required params or invalid values are reported before any
  agent runs. Run `agm exec FILE --help` to show the discovered param options for
  that program.
- `-I DIR`, `--module-path DIR`: Add `DIR` as an additional module search root
  (repeatable), resolved relative to the invocation working directory. See
  [Module resolution](#module-resolution). This is also how e2e/fixture tests point
  `agm exec` at test-specific module roots.
- `--no-stdlib`: Disable automatic `std/core` opening throughout the loaded
  program (the entry and its library modules). Explicit `import std/core` still
  uses the normal module import semantics.
- `--strict-json`: Require agents to return exactly one bare JSON value (no fences,
  prose, or repair). Overridable per call site with the `strict_json:` named argument
  to `ask`.
- `--no-strict-json`: Use lenient JSON recovery (the default): the runtime recovers
  exactly one JSON value from chatty output (stripping fences/prose, repairing
  trivially malformed JSON), then validates it strictly against the schema.
- `--max-iters N`: Override the host's `max-iters` safety valve with a positive
  integer, which caps
  **unbounded** loops (a bare `while … do … done` or `do … until E` with no
  `[n]` bound and no `for` clause) at `N` body executions, raising
  `MaxIterationsExceeded`. Self-bounded loops (`for`, `do[n]`) are never cut
  short by this valve. The valve is off by default; its readable setting is `0`.
  This option, config, or a positive source write enables it, while a source
  `std/config::max-iters := 0` write disables it. See
  [Control flow](../agl/reference/control-flow.md).
- `--max-call-depth N`: Override the maximum recursion call depth (CLI >
  `[exec] max-call-depth` config; the canonical default is 256). Exceeding it
  raises `RecursionError`.
- `--agent AGL_LITERAL`: Seed `std/config::default-agent` with one constant `Agent`
  expression, for example `AgentClaude("sonnet", "medium")`. The literal is parsed and
  typechecked before execution; it overrides qualified program-table/`[exec]` configuration. It
  selects the value used by `ask` calls that omit `agent`. An `AgentCommand(...)`
  literal's command text is shell-split and validated the same way as `[exec] runner`
  before execution; a malformed command (e.g. an unclosed quote) exits 1 with nothing run.
  Because `--agent` is an explicit request, it still exits 1 when combined with
  `--no-stdlib` on a program that never loads `std/config` — unlike
  qualified program-table/`[exec] default-agent`, which is simply inert (has no effect) in
  that situation.
- `--log` / `--log-file PATH` / `--no-log`: Control trace logging, which is **off by
  default**. `--log` enables it with an auto-generated timestamped path under
  `.agent-files/`; `--log-file PATH` writes a structured JSONL trace to `PATH`;
  `--no-log` disables it, providing the initial trace setting a program can still
  override with a `std/config::log := true` write, and overriding a `[exec] log = true`
  setting. The three are mutually exclusive (at most one may be given).
- `--dry-run`: Run the full static pipeline, param validation, and contract
  materialization, then stop before evaluating any expression (static errors exit 1; a
  clean check exits 0 with no program output). A program declaring `extern def`
  (see [Python FFI](../agl/reference/ffi.md)) does **not** import its companion
  Python file(s) under `--dry-run`; companion loading is skipped with evaluation
  to keep dry runs side-effect-free, so a broken companion does not fail a dry
  run. When the check succeeds and one or more agent-call, `exec`, or extern-call sites exist,
  the static call-site inventory is printed to stdout:

  ```
  call-sites:
    line N:C: <callee> → <target-type> [<codec>[, schema: yes][, policy: <policy>]]
  ```

  Each entry shows the 1-based source line and column (`N:C`), the callee name (`ask`,
  `exec`, or an extern's declared name), the target type, the
  selected codec (`text`, `json`, or `extern`), and optionally whether a JSON Schema is
  attached (`schema: yes`) and the effective parse-failure policy (`abort` or
  `retry[N]`; not applicable to extern calls). When no such call sites are present, no
  inventory is printed.

### Agents

`ask` selects an ordinary typed `Agent` value. Pass one explicitly, or omit
`agent` to read `std/config::default-agent` at that call:

```agl
let reviewer = AgentClaude("sonnet", "medium")
let review: Review = ask("Review %{artifact}", agent = reviewer)
let answer: text = ask("Summarize")
```

`AgentCommand(command)`, `AgentClaude(model, thinking)`,
`AgentCodex(model, thinking)`, and `AgentPi(provider, model, thinking)` each
build their own argv; use an `Agent` value or `default-agent` to select one.

### Agent command interpolation

The argv an `Agent` value builds — the command string of an `AgentCommand`, and the
provider variants' fixed flags — interpolates `%{name}` holes strictly from the process
environment overlaid with `PROMPT_FILE`, which wins on conflicts. Unlike `agm loop`'s
runner and selector, no workflow-specific variables are added. See
[Runner command interpolation](agents.md#runner-command-interpolation) for the shared
`%%`/`PROMPT_FILE` alias, `\%{` escape, and shlex-split rules. A prompt-file placeholder
places the rendered prompt file at that position; otherwise AGM appends `@<path>`, except
for `AgentCodex`, which pipes the prompt in on standard input instead of appending a
target. Because an AgL text literal interpolates `%{…}` itself, spell the placeholder as
`\%{PROMPT_FILE}` inside `AgentCommand("…")` so it reaches the host as literal text. An
unresolvable hole fails the call with a catchable `AgentCallError` whose `cause` is
`"spawn_failure"`.

### Configuration

The `[exec]` section in `config.toml` supplies the engine defaults that CLI flags and
source `std/config` writes can override:

```toml
[exec]
default-agent = 'AgentClaude("sonnet", "medium")' # typed default Agent value
# runner = "claude"         # bare host agent command; lower precedence than default-agent
strict-json = false         # lenient JSON recovery is the default
max-iters = 5               # opt into a safety-valve cap for unbounded loops
timeout = "30m"             # initial shell-exec and agent idle timeout
log = false                 # trace logging off by default; set true to enable
# log-file = "trace.jsonl" # explicit trace path (omit for auto timestamped path)

```

`runner` is a bare host command (like `[loop] runner`), not AgL literal syntax; when
set, it seeds `default-agent` as `AgentCommand(runner)`. It applies only when neither
`--agent` nor `default-agent` (CLI or config) supplies a value: precedence, highest
first, is `--agent` > qualified program-table/`[exec] default-agent` > `[exec] runner` > the
`std/config` declaration's own default. `runner` is shell-split and validated as soon
as configuration is read, before the module graph loads; a malformed command exits 1
with nothing run.

Qualified tables address declarations by module suffix and scope path. The entry
file's `.agl` stem is its module component. For example, a `review::main` program
in `review-tools/review` reads engine overrides from `[review-tools.review.review]`,
and a `review::max-tries` param in `review-tools/judge` reads `[judge.review]` when
that suffix is unambiguous. Use a longer suffix or an exact quoted module route
such as `["review-tools/judge".review]` to disambiguate. `runner` remains an
`[exec]`-only setting. Inline `-c` params are CLI-only.

#### Source-level engine settings (`std/config`)

An AgL program may set its own exec options by importing the standard-library
module `std/config` and writing its **engine settings** — mutable bindings backed
by the live engine. Each setting is also readable through a qualified reference:

```agl
import std/config

param spec

program def main() -> unit =
  std/config::log := true             # enable trace logging for this program
  std/config::log-file := Some("trace.jsonl")  # explicit trace path
  std/config::strict-json := true     # require bare JSON from agents
  std/config::max-iters := 10         # host safety valve cap for unbounded loops
  std/config::default-agent := AgentClaude("sonnet", "medium")
  std/config::timeout := Some("30s")  # shell-exec idle timeout

  let result = ask "Process %{spec}"
  print result
```

A qualified target (`std/config::KEY := …`) always writes a setting. After an
`open import std/config`, its names are also in scope, so a bare `KEY := …` write
is valid. The `Option[text]` settings (`log-file`, `timeout`) take a `Some("…")`
or `None` value.

Precedence differs by kind:

- **Engine settings** (`default-agent`, `log`, `strict-json`, `max-iters`, `log-file`, `timeout`):
  `source std/config::X write > CLI > qualified program table > [exec].X > engine default`.
  `default-agent` has one extra fallback below `[exec] default-agent`: `[exec] runner`.
- **Param values** (`param NAME`):
  `CLI > qualified config table > source default > required error`.

`NAME` is a scoped param's full path spelling (`Deploy::region`) when it is
  declared as a member of a named scope region. That key must be quoted in TOML, since
  `::` is not a legal bare key:

  ```toml
  [demo.Deploy]
  region = "prod"
  ```

The CLI flags and the config-file layers supply the setting's **initial** value; a
source `std/config::X := …` write overrides them from its program point onward. For
example, `--no-log` sets the initial state to off, but a later
`std/config::log := true` write turns tracing on from that point, and
`std/config::max-iters := 10` overrides `[exec] max-iters = 5`.

Every setting takes effect **positionally**, like an ordinary `var` mutation:
statements after the write see the new value, statements before it do not. Writing
`log` or `log-file` reconfigures the trace destination for subsequent calls. Assigning
`Some(path)` to `log-file` enables
logging; a later `log := false` disables it without clearing the path. Writing
`strict-json`, `max-iters`, or `timeout` changes subsequent agent-output parsing,
unbounded loops, or `exec` calls, respectively.

A CLI, qualified program table, or `[exec]` timeout initially seeds both shell execution and
agent idle timeout. A source write to the `timeout` setting changes only the
**shell-exec** timeout; agent idle timeout cannot be changed mid-program.

A bad duration in `std/config::timeout := Some("…")` is a runtime AgL error (exit 2),
because a source write is a runtime-evaluated expression. A valid assigned timeout
round-trips with its original text while the parsed duration drives shell execution.
A bad `--timeout`, qualified program-table timeout, or `[exec].timeout` value is a
pre-execution error (exit 1).

`--no-log-file` clears only the initial `log-file` value; a log-file path set via
`[exec] log-file` or auto-assigned by `--log` still applies. Use `--no-log` to
disable tracing entirely.

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | The workflow completed successfully |
| `1` | Pre-execution failure: unreadable file, static language diagnostics (including invalid `case` coverage), host configuration error, or param validation failure |
| `2` | The workflow executed but ended with an uncaught AgL exception |

### Diagnostics and warnings

- Error-severity diagnostics (static language errors, including non-exhaustive or
  redundant `case` arms, host configuration errors, param validation failures) and uncaught AgL exceptions are
  printed to stderr and determine the exit code per the table above.
- Advisory **warnings** are a separate
  channel. They are printed to stderr with a `warning:`
  prefix (`warning: line N: message`) to disambiguate them from errors, and they never
  affect the exit code — the program still runs to completion. Program `print` output
  goes to stdout, kept clean of diagnostics.
