# `agm exec`

[← Commands](index.md)


```text
agm exec [--strict-json|--no-strict-json]
         [--max-iters N] [--max-call-depth N] [--agent AGL_LITERAL]
         [--timeout DURATION|--no-timeout] [--dry-run]
         [--log|--log-file PATH|--no-log] [--no-log-file]
         [--no-stdlib]
         [-I DIR]... [-p PATH]
         (FILE | PACKAGE/MODULE::PROGRAM | -c COMMAND) [ARG]... [--NAME VALUE]...
```

Execute an AgL workflow program from a source `FILE`, an installed
`PACKAGE/MODULE::PROGRAM` reference, or inline text given with `-c`/`--command`. Exactly one
source selector is required; `-c` is mutually exclusive with the positional file/reference
selector. `PACKAGE/MODULE::PROGRAM` resolves its module through the active package
selection, for example `agm exec review_tools/review::main`. An existing `FILE` path always
takes precedence, even when its name contains `::`.

A file must declare at least one `program def` function. `exec` initializes the
linked program and invokes its sole entry implicitly; if it declares several, select
one with `-p`/`--program` using its declaration path (for example,
`review::main`). Inline `-c` source is wrapped in a synthetic `program def main`
when it does not declare an entry itself; when it declares one, the source is an
ordinary module whose root is static, so its statements and non-constant bindings
belong in the program body.

### Module resolution

`agm exec` supports programs that import library modules. The runtime assembles an
**unordered set of search roots**:

- the directory of `FILE` (or the working directory for `-c`),
- when that directory is inside a development package, its containing package and the recursive closure of dependencies declared with relative `path` sources in their manifests,
- the selected standard library: a development `std` package checkout whose own module tree holds `FILE` (or the working directory) when there is one, whatever version it declares; otherwise the active immutable `<AGM-home>/packages/std/<AGM_VERSION>/` package when it matches the running binary, where a selected active package with a different version is an error, while AGM's bundled copy (`agm/stdlib` in an installed wheel or the in-repo `stdlib/` tree in a source checkout) is used when no active package is selected or the matching store tree is absent; `AGM_STDLIB` overrides this whole selection without mounting the active package as an additional root,
- the selected AGM home's global `lib` directory (overridable via `[modules] lib_root` in config),
- any roots declared under `[modules] roots` in any config layer,
- any roots added with `-I`/`--module-path`.

The independently loaded `[modules] lib_root` and `roots` settings expand `%{VAR}` the same
leniently as other path-valued settings; see
[Path-valued settings](config.md#path-valued-settings) for the exact rule.

Set `AGM_HOME` to select the complete runtime home (config, prompts, sandbox settings, global library, and managed package store). Without it, AGM uses a populated `.agm` beside the installed executable, then falls back to `~/.agm`. Set `AGM_STDLIB` to point only the standard-library root elsewhere. A wheel installation can execute AgL with its bundled fallback even when the selected home has no active `std` package.

A module name that resolves to exactly one file across all roots succeeds; zero files,
or two or more distinct files, are static errors (exit 1 with a diagnostic).

A module (or a file-backed `FILE` entry program) that declares `extern def`
(see [Python FFI](../agl/reference/ffi.md)) requires a companion Python file
at the same path with a `.py` suffix; unlike a module import, this path is
derived rather than searched, so module-root ambiguity never applies to it. A
missing companion, or a companion missing the extern's declared name as a
callable attribute, is a diagnostic reported before the program runs, exactly
like any other static error.

`resource` and `resource-dir` (see
[Expressions](../agl/reference/expressions.md)) anchor at their declaring
module's directory or owning package root, so they need a module read from disk:
they are unavailable in inline `-c` source and in `agm repl`, where a call to
either is a static error.

### Options

- `-c COMMAND`, `--command COMMAND`: Execute the AgL program given as `COMMAND`
  directly, instead of reading the program from `FILE`.
- `-p PATH`, `--program PATH`: Select a `program def` entry by its declaration
  path. This accepts entry-file paths such as `main` or `review::main`; with an
  installed `PACKAGE/MODULE::PROGRAM` reference, it overrides that reference's
  program path while retaining its module.
- `ARG` / `--NAME VALUE`: Provide a value for one of the selected `program def`'s
  own value parameters. See [Program arguments](#program-arguments).
- `-I DIR`, `--module-path DIR`: Add `DIR` as an additional module search root
  (repeatable), resolved relative to the invocation working directory. See
  [Module resolution](#module-resolution). This is also how e2e/fixture tests point
  `agm exec` at test-specific module roots.
- `--no-stdlib`: Disable the automatic `import std/prelude::*` prelude throughout
  the loaded program (the entry and its library modules). Any explicit import whose
  expansion includes `std/prelude` supplies that module's contribution instead; plain
  `import std/prelude` leaves prelude names qualified-only.
- `--strict-json`: Require agents to return exactly one bare JSON value (no fences,
  prose, or repair). Overridable per call site with the `strict-json:` named argument
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
- `--dry-run`: Run the full static pipeline, program-argument validation, and contract
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
  inventory is printed. Standard-library methods backed by externs (`[1].size()`,
  `"a".trim()`) are listed at the call site in your own source; the standard library's
  internal calls are not inventoried unless the program imports the module explicitly.

### Program arguments

The selected `program def`'s value parameters project onto `agm exec`'s own CLI
surface, one flag or positional slot per parameter, in the same positional/standard/
named-only zones a parameter list uses everywhere else in AgL: a parameter
list with no zone attributes is entirely named-only, so bare `x: T` parameters
become `--x` options; an `@arg-pos` parameter fills an `ARG` slot in declaration
order and can never be supplied by name; an `@arg-std` parameter is standard and
accepts either a positional token or its own `--x`.

Each name-addressable parameter's declared type selects its flag form; any
value-taking flag also accepts the inline `--x=VALUE` form as an alternative
to `--x VALUE`:

| Type | Flag |
|---|---|
| `bool` | `--x` / `--no-x` — a bare flag, no value |
| `Option[T]` | `--x VALUE` (wraps `Some`) / `--no-x` (`None`); `VALUE` is taken verbatim when `T` is `text`, otherwise parsed as one strict JSON value of `T` |
| `text` | `--x VALUE`, `VALUE` taken verbatim |
| every other type | `--x VALUE`, `VALUE` parsed as one strict JSON value and validated against the declared type |

A positional slot has no `--no-x` counterpart, so it never gets the `Option[T]`
flag's special treatment: a positional token for a `text` parameter is taken
verbatim, and for every other type — `Option[T]` included — it is parsed as one
strict JSON value of the parameter's own declared type (e.g. `'{"$case": "Some",
"value": "hi"}'` for an `Option[text]` positional).

Supplying the same parameter twice — twice by flag, or once positionally and once
by `--x` for a standard parameter — is an error reported before any agent runs, as
is an unrecognized `--flag` or a positional argument beyond the program's own
positional-capable parameters.

A parameter name cannot project onto a reserved spelling. An engine-setting name
(`default-agent`, `strict-json`, `max-iters`, `timeout`, `log`, `log-file`) on any
name-addressable parameter is a static error — `agm check` reports it too — whether
or not the CLI ever supplies it, since program arguments and engine settings share
one flag and config namespace. A parameter whose projected flag would otherwise
collide with a reserved flag is instead a host-level check with no static
counterpart: it fails the program when actually selected for execution, but
`--help` and shell completion degrade silently, falling back to `agm exec`'s
own help and offering no completions for that program rather than erroring. The
reserved set is the host's own declared options (`--help`/`-h`, `--program`/`-p`, `--command`/`-c`,
`--module-path`/`-I`, `--max-call-depth`, `--no-stdlib`, `--dry-run`, `--agent`)
**union every engine-setting flag in both polarities** — `--default-agent`,
`--strict-json`/`--no-strict-json`, `--max-iters`, `--timeout`/`--no-timeout`,
`--log`/`--no-log`, `--log-file`/`--no-log-file` — so a parameter such as
`no-log: text` collides even though `no-log` itself names no engine setting. It
also includes another parameter's own projected flag, such as a `cache: bool`
parameter's negative flag colliding with a `no-cache: bool` parameter's positive
one.

That degradation also changes how an unrecognized flag beside a help flag is
answered: `agm exec FILE --nope -h` is a usage error (exit 1) when the program's
options build, because the program's own parser rejects `--nope` before reaching
the help flag, but prints `agm exec`'s own help (exit 0) when they collide and
no program parser exists to reject it.

`agm exec`'s own parser consumes a bare `--` before the program's parser sees
any tokens, so reaching the program's own end-of-options marker (to pass a
literal `--`-prefixed positional argument) takes a **doubled** `--` on the
command line — `agm exec FILE -- -- --odd-looking-value`. That first marker also
selects a flag-shaped source: `agm exec -- --odd-name.agl` runs the file
`--odd-name.agl` rather than reading it as an option.

A value-taking flag consumes whatever token follows it, flag-shaped or not:
`--msg --x` supplies the value `--x`. Spell the value inline — `--msg=--x` — when
the token after the flag is meant as a flag of its own.

The selected program's own command owns `-h` and `--help`. With exactly one
entry program — or with several and one selected via `-p` — `agm exec FILE -h`,
`agm exec -h FILE`, and `agm exec FILE -p NAME --help` all print that program's
own help: its usage line, its `@doc` prose as the description, and one entry per
visible parameter with that parameter's flags, value placeholder, and `@doc`
prose. A `-h` in a position where a value is expected (`--msg -h`, `-va -h`) is
that value, and the program runs. With several entry programs and none selected,
`agm exec`'s own help is printed followed by the declaration paths to choose
from with `-p`; an unreadable source or a parameter that cannot be projected
degrades to `agm exec`'s own help too.

Inline `-c` source with no `program def` of its own is wrapped in a synthetic,
parameterless `program def main`, so it accepts no `ARG`/`--NAME` tokens at all;
declare an explicit `program def` in the inline text to give it parameters. Even
then, an inline `-c` program cannot receive positional arguments at all: a bare
token after `-c COMMAND` is parsed as the mutually exclusive `FILE` selector, not
as `ARG` (`agm exec -c '…' hello` fails with `error: argument FILE not allowed
with -c/--command`), even though `-c … --help` still shows a positional
usage slot for a program with positional-capable parameters. Named `--x`/`--x
VALUE` options work normally; declare only standard or named-only parameters in
inline `-c` source to make every value reachable.

An omitted name-addressable argument resolves from the program's own qualified
config table (see [Configuration](#configuration)), then its signature default;
a required parameter with neither is reported before any agent runs. A
positional-only parameter has no config-table spelling at all — a config table
is a name-keyed channel, and a positional-only parameter exposes no name to key
it by — so it always falls straight through to its signature default (or is
reported as missing, if required). An `Option[T]` parameter's config table has
no spelling for `None`: a present key can only ever supply the wrapped `Some`
value, or the key can be left out entirely to fall through to the signature
default — request `None` explicitly with the CLI's `--no-x` flag instead.

#### Attributes on a parameter

Attributes on a `program def`'s own value parameters shape the CLI surface
they project onto. See [Program arguments](../agl/reference/host-environment.md#program-arguments)
in the AgL reference for the language-side definitions.

| Attribute | Effect on the CLI |
|---|---|
| `@doc("prose")` | On the `program def`, the help's description; on a parameter, that option's help entry |
| `@opt-name("flag-word")` | Renames the flag, its `--no-` negation, its config key, and its completion |
| `@opt-short("t")` | Adds `-t` beside the long flag |
| `@opt-env("VAR")` | Reads `VAR` when no CLI token supplies the parameter |
| `@opt-metavar("PATH")` | Replaces the value placeholder in usage and help |
| `@opt-hidden` | Omits the parameter from `--help` and from shell completion |

A short flag takes its value as `-t VALUE` or attached as `-tVALUE`, and
one-letter flags group: `-abc` is `-a -b -c`, and only the group's last letter
may take a value — so in `-va -h`, `-h` is the value `-a` asked for, not a help
request. Short spellings are offered by shell completion alongside the long
ones; a `@opt-hidden` parameter is offered by neither.

An `@opt-env` variable is consulted only when no CLI token supplies the
parameter, and **an empty variable counts as unset**: `VAR= agm exec FILE`
falls through to the config table and then the declared default exactly as an
unset `VAR` does. An environment fallback therefore cannot deliver an empty
`text` value — write `""` as the parameter's declared default, or pass
`--x=""` on the command line.

### Agents

`ask` selects an ordinary typed `Agent` value. Pass one explicitly, or omit
`agent` to use the lazy default session. Its first use snapshots
`std/config::default-agent`; later free asks reuse that agent and conversation:

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
provider member records' fixed flags — interpolates `%{name}` holes strictly from the
process environment overlaid with `PROMPT_FILE`, which wins on conflicts. Unlike `agm loop`'s
runner and selector, no workflow-specific variables are added. See
[Runner command interpolation](agents.md#runner-command-interpolation) for the shared
`%%`/`PROMPT_FILE` alias, `\%{` escape, and shlex-split rules. A prompt-file placeholder
places the rendered prompt file at that position; otherwise AGM appends `@<path>`, except
for `AgentCodex`, which pipes the prompt in on standard input instead of appending a
target. Because an AgL text literal interpolates `%{…}` itself, spell the placeholder as
`\%{PROMPT_FILE}` inside `AgentCommand("…")` so it reaches the host as literal text. An
unresolvable hole fails the call with a catchable `AgentCallError` whose `cause` is
`"interpolation_failure"`.

### Configuration

The `[exec]` section in `config.toml` supplies the engine defaults that CLI flags and
source `std/config` writes can override:

```toml
[exec]
default-agent = 'AgentClaude("sonnet", "medium")' # typed default Agent value
# runner = "custom-agent --session %{SESSION_ID}" # command must create/resume this id
strict-json = false         # lenient JSON recovery is the default
max-iters = 5               # opt into a safety-valve cap for unbounded loops
timeout = "30m"             # initial shell-exec and agent idle timeout
log = false                 # trace logging off by default; set true to enable
# log-file = "trace.jsonl" # explicit trace path (omit for auto timestamped path)

```

`runner` is a bare host command (like `[loop] runner`), not AgL literal syntax; when
set, it seeds `default-agent` as `AgentCommand(runner)`. Continuing free asks require the
command to consume `%{SESSION_ID}` and use that same ID to create or resume a transcript;
use a native `AgentClaude`, `AgentCodex`, or `AgentPi` value when possible. It applies only when neither
`--agent` nor `default-agent` (CLI or config) supplies a value: precedence, highest
first, is `--agent` > qualified program-table/`[exec] default-agent` > `[exec] runner` > the
`std/config` declaration's own default. `runner` is shell-split and validated as soon
as configuration is read, before the module graph loads; a malformed command exits 1
with nothing run.

Qualified tables address declarations by module suffix and scope path. A loose entry
file's `.agl` stem is its module component. A file executed directly from a package — a
development checkout, an installed store tree, or the selected standard library — instead
retains its package-qualified module route, just like an installed package reference. For
example, a `review::main` program in `review-tools/review` reads both its engine overrides
and its own value parameters from `[review-tools.review.review.main]`; a shorter
unambiguous suffix, or an exact quoted module route such as
`["review-tools/review".review.main]`, also resolves it. `runner` remains an `[exec]`-only
setting. Inline `-c` source has no file-derived route, so its program's own value parameters
are CLI-only (CLI value, then signature default).

A key in the selected program's own table that names neither one of its own
name-addressable value parameters nor an engine setting (typically a
misspelling) is reported on stderr and ignored; the program still runs on its
declared defaults. A key that instead names one of the program's
positional-only parameters is also reported — with a distinct message noting
that the parameter can only be supplied positionally — since a config table
cannot address it either way; the program still runs on that parameter's
signature default.

#### Source-level engine settings (`std/config`)

An AgL program may set its own exec options by importing the standard-library
module `std/config` and writing its **engine settings** — mutable bindings backed
by the live engine. Each setting is also readable through a qualified reference:

```agl
import std/config

program def main(spec: text) -> unit =
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
`import std/config::*`, its names are also in scope, so a bare `KEY := …` write
is valid. The `Option[text]` settings (`log-file`, `timeout`) take a `Some("…")`
or `None` value.

Precedence differs by kind:

- **Engine settings** (`default-agent`, `log`, `strict-json`, `max-iters`, `log-file`, `timeout`):
  `source std/config::X write > CLI > qualified program table > [exec].X > engine default`.
  `default-agent` has one extra fallback below `[exec] default-agent`: `[exec] runner`.
- **Program arguments** (a selected `program def`'s own value parameters):
  `CLI > qualified program table > signature default > required error`, except a
  positional-only parameter, which has no qualified-table spelling: `CLI >
  signature default > required error`.

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
| `0` | The workflow completed successfully, or `std/process::exit(0)` requested success |
| `1` | Pre-execution failure: unreadable file, static language diagnostics (including invalid `case` coverage), host configuration error, or program-argument validation failure; it can also be requested with `std/process::exit(1)` |
| `2` | The workflow executed but ended with an uncaught AgL exception; it can also be requested with `std/process::exit(2)` |
| `3`–`255` | Requested by `std/process::exit(code)` |

`std/process::exit` accepts only the portable process-status range `0..255`,
so its documented status is preserved by every supported host. An out-of-range
value is a runtime error, not a process termination.

### Diagnostics and warnings

- Error-severity diagnostics (static language errors, including non-exhaustive or
  redundant `case` arms, host configuration errors, program-argument validation failures) and uncaught AgL exceptions are
  printed to stderr and determine the exit code per the table above.
- Advisory **warnings** are a separate
  channel. They are printed to stderr with a `warning:`
  prefix (`warning: line N: message`) to disambiguate them from errors, and they never
  affect the exit code — the program still runs to completion. Program `print` output
  goes to stdout, kept clean of diagnostics.
