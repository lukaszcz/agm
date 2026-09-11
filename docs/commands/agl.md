# AgL workflow DSL

AGM runs AgL workflow programs with [`agm exec`](#agm-exec), statically checks AgL files
with [`agm check`](#agm-check), and evaluates AgL interactively with [`agm repl`](#agm-repl).
The AgL language itself is documented in the
[AgL language reference](../agl/reference/index.md).

## `agm exec`

```text
agm exec [--strict-json|--no-strict-json]
         [--max-iters N] [--max-call-depth N] [--default-agent AGENT]
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
- when that directory is inside a development package, its containing package and the recursive closure of dependencies declared with relative `path` sources in their manifests, each contributing its `src/` module tree under its own package name,
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
- `--default-agent AGENT`: Seed `std/config::default-agent` from the host Agent syntax described
  below. Canonical constructor syntax such as `AgentClaude("sonnet", "medium")` remains
  accepted. The value is typechecked before execution; it overrides qualified
  program-table/`[exec]` configuration. It
  selects the value used by `ask` calls that omit `agent`. An `AgentCommand(...)`
  literal's command text is shell-split and validated the same way as `[exec] runner`
  before execution; a malformed command (e.g. an unclosed quote) exits 1 with nothing run.
  Because `--default-agent` is an explicit request, it still exits 1 when combined with
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
| `Agent` | `--x VALUE`, using the host Agent syntax below or the canonical tagged JSON shape |
| every other type | `--x VALUE`, `VALUE` parsed as one strict JSON value and validated against the declared type |

A positional slot has no `--no-x` counterpart, so it never gets the `Option[T]`
flag's special treatment: a positional token for a `text` parameter is taken
verbatim, an `Agent` token uses the host syntax below, and every other type —
`Option[T]` included — is parsed as one strict JSON value of the parameter's own
declared type (e.g. `'{"$case": "Some", "value": "hi"}'` for an `Option[text]`
positional).

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
own help and offering no completions for that program rather than erroring. For
`agm exec`, the reserved set is the host's own declared options (`--help`/`-h`, `--program`/`-p`, `--command`/`-c`,
`--module-path`/`-I`, `--max-call-depth`, `--no-stdlib`, `--dry-run`)
**union every engine-setting flag in both polarities** — `--default-agent`,
`--strict-json`/`--no-strict-json`, `--max-iters`, `--timeout`/`--no-timeout`,
`--log`/`--no-log`, `--log-file`/`--no-log-file` — so a parameter such as
`no-log: text` collides even though `no-log` itself names no engine setting. It
also includes another parameter's own projected flag, such as a `cache: bool`
parameter's negative flag colliding with a `no-cache: bool` parameter's positive
one. A [registered package command](pkg.md#registered-commands) reserves only `--dry-run` and
`-h`/`--help`.

That degradation also changes how an unrecognized flag beside a help flag is
answered: `agm exec FILE --nope -h` is a usage error (exit 1) when the program's
options build, because the program's own parser rejects `--nope` before reaching
the help flag, but prints `agm exec`'s own help (exit 0) when they collide and
no program parser exists to reject it.

A bare `--` ends `agm exec`'s own option scanning, and the marker then reaches
the program unless it is what named the FILE, so a **single** `--` is the
program's own end-of-options marker: `agm exec FILE -- --odd-looking-value`
passes that `--`-prefixed token as a positional argument, and
`agm exec FILE -- --help` is that positional argument too rather than a help
request. The exception is a marker that selects a flag-shaped source, which
`agm exec` consumes to do so: `agm exec -- --odd-name.agl` runs the file
`--odd-name.agl` rather than reading it as an option, and reaching the
program's own marker as well then takes a second one —
`agm exec -- --odd-name.agl -- --odd-looking-value`.

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
| `@opt-hidden` | Omits the parameter's `--name` entry from `--help` and from shell completion; a positional-capable parameter keeps its usage slot |

A short flag takes its value as `-t VALUE` or attached as `-tVALUE`, and
one-letter flags group: `-abc` is `-a -b -c`, and only the group's last letter
may take a value — so in `-va -h`, `-h` is the value `-a` asked for, not a help
request. Short spellings are offered by shell completion alongside the long
ones; a `@opt-hidden` parameter's flags are offered by neither, though the
parameter still fills its positional slot when it has one.

An `@opt-env` variable is consulted only when no CLI token supplies the
parameter, and **an empty variable counts as unset**: `VAR= agm exec FILE`
falls through to the config table and then the declared default exactly as an
unset `VAR` does. An environment fallback therefore cannot deliver an empty
`text` value — write `""` as the parameter's declared default, or pass
`--x=""` on the command line.

### Host Agent syntax

Every CLI argument or TOML string whose declared type is the standard `Agent` accepts:

- `claude/MODEL-EFFORT` → `AgentClaude(MODEL, EFFORT)`
- `codex/MODEL-EFFORT` → `AgentCodex(MODEL, EFFORT)`
- `pi/PROVIDER/MODEL-EFFORT` → `AgentPi(PROVIDER, MODEL, EFFORT)`
- any other `PROVIDER/MODEL-EFFORT` → `AgentPi(PROVIDER, MODEL, EFFORT)`

The final hyphen separates the model from an opaque, non-empty effort suffix; AGM does
not restrict the suffix vocabulary. Exact lowercase `claude/` and `codex/` prefixes
select those native CLIs before the generic Pi form. Text that does not match a compact
form is a verbatim `AgentCommand`, so `--default-agent 'worker --flag'` selects that custom
command. Agent-typed program parameters also retain their canonical tagged JSON form;
`--default-agent` and the `default-agent` config key retain direct AgL constructor syntax for
compatibility.

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

### Session runners

A continuing `AgentCommand` session — including the one `[exec] runner` seeds — requires an
unescaped `%{SESSION_ID}` placeholder in its command. AGM replaces it in the command argv
with a generated id; it does not put `SESSION_ID` in the child environment.

Free `ask`, `Session::open(AgentCommand(...))`, and an `AgentCommand(...).ask(...)`
with corrective retries open continuing sessions, so their command must contain
that placeholder. A single-attempt `AgentCommand(...).ask(...)` sends exactly one
prompt and does not require it. A command without the placeholder cannot otherwise be
opened as a session and raises `SessionError`. See [Configuration](#configuration) for
`runner` precedence and [Agent calls](../agl/reference/agent-calls.md#sessions) for all
session backends.

### Configuration

The `[exec]` section in `config.toml` supplies the engine defaults that CLI flags and
source `std/config` writes can override:

```toml
[exec]
default-agent = "claude/sonnet-medium" # native shorthand or custom command
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
`--default-agent` nor a configured `default-agent` supplies a value: precedence, highest
first, is `--default-agent` > qualified program-table/`[exec] default-agent` > `[exec] runner` > the
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

Imported modules, including the standard library, are precompiled on demand under
`$XDG_CACHE_HOME/agm/agl`, or `~/.cache/agm/agl` when `XDG_CACHE_HOME` is unset.
Source and dependency edits, compiler updates, and relevant compilation settings
invalidate artifacts automatically. Module initialization and execution use the
current invocation's configuration and runtime state. The cache is disposable:
deleting it or making it unavailable does not prevent execution.

## `agm check`

```text
agm check [-I DIR]... [--no-stdlib] FILE...
```

Run the full **static** AgL pipeline — parse, module loading, scope resolution, type
checking, match compilation, and lowering — over each `FILE` and report GNU-style
diagnostics. `agm check` never evaluates anything and never runs an agent.

Unlike `agm exec`, no `FILE` needs to declare a `program def`: a plain library module
(the case `agm exec --dry-run` rejects) can be checked on its own. A `program def`
present in a file is validated as part of the module it lives in, including that each
of its own value parameters has a type that can cross the host/JSON boundary (text
verbatim, or a finite, JSON-decodable data type — for example, a function-typed
parameter is rejected) and that no name-addressable parameter spells a reserved
engine-setting name. `check` never selects an entry program, never resolves its
parameters against configuration, and never runs it — there is no CLI argument
projection and no `-p`/`--program` selector.

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
[`agm exec` → Module resolution](#module-resolution) for the full root-assembly
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
bad.agl:3:11: error: 'undefined-name' is not defined.
$ echo $?
1
```

## `agm repl`

```text
agm repl [--strict-json|--no-strict-json]
         [--max-iters N] [--max-call-depth N] [--default-agent AGENT] [--confirm-agents]
         [--quiet] [--dry-run] [--no-stdlib] [--log|--log-file PATH|--no-log] [--plain]
```

Start an interactive read-eval-print loop for AgL. Unlike `agm exec`, which runs a
whole program from a fresh environment, the REPL keeps a **persistent session**: each
entry is parsed, statically checked (including pattern coverage), and evaluated once against an environment that
accumulates bindings, types, and declarations across entries, so earlier results stay
available and agent calls fire exactly once.

### Front ends

`agm repl` has two front ends sharing the same session and evaluation behavior:

- An interactive console (prompt_toolkit) with syntax highlighting,
  tab-completion, multiline editing, command history, and colour themes — see
  [Entry editing](#entry-editing) and [Console-only editing
  features](#console-only-editing-features) below.
- A **plain** line-oriented mode with no styling, colour, or ANSI escapes: it
  prints the same `agl> ` / `...> ` prompts and reads lines from stdin,
  accumulating a multiline entry exactly as the console does, so a pasted or
  programmatically sent multi-line block still works. This is what drives the
  REPL over a pipe or from a non-terminal consumer such as an editor's comint
  buffer.

The plain front end engages automatically when stdin or stdout is not a
terminal, or when `TERM=dumb`; `--plain` forces it even on a terminal. There is
no flag to force the console front end onto a non-terminal.

The REPL reuses `[exec]` settings for `default-agent`, the max-iters valve, call-depth
limit, JSON strictness, and timeout. Free `ask` lazily opens one default agent
conversation and snapshots `default-agent` at that first use; later free calls reuse
it even if the setting changes. Explicit `Session::open` sessions also remain live
until closed or the REPL exits. `:reset` clears AgL bindings and settings but does
**not** close host sessions, including the default session used by free `ask`; a later
free `ask` continues that default conversation. Close unneeded explicit sessions
yourself. Like `agm exec`, each typed `Agent` value selects its own backend command;
settings do not select it. `--default-agent` and `[exec] default-agent` accept the shared
[host Agent syntax](#host-agent-syntax), including native shorthand and custom
command text. Like `agm exec`, `--default-agent` combined with `--no-stdlib`
still fails — during session initialization, before the prompt appears — if the
session never loads `std/config`; `[exec] default-agent` is simply inert in that same
situation.

Like `agm exec`, the REPL supplies an automatic `import std/prelude::*` prelude to
each loaded program, so standard-library names such as `Option`, `Some`, and
`None` are available unqualified from a fresh prompt. An explicit import whose
expansion includes `std/prelude` supplies that contribution instead, so plain
`import std/prelude` leaves prelude names qualified-only. Pass `--no-stdlib` to disable
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

These behaviors are shared by both front ends:

- Multiline editing is **AgL-aware**: pressing Enter on an unterminated block
  (`record`, `enum`, `if`, `case`, `try`, `do`, …) or a line-final raw-tail header
  such as `exec$`/`ask$` opens a continuation line (`...>`); a complete entry submits.
  Pressing Enter on a blank continuation line force-submits even an unfinished buffer
  so you can always escape. In the plain front end this accumulation happens as lines
  are read from stdin rather than through key bindings, and the same predicate decides
  when an entry is complete. One line at a time is less than a pasted buffer, though:
  an indented block parses after every line yet can always take one more, so there an
  entry whose latest line is indented stays open until the blank line closes it. End of
  input closes it too, so a block piped in without that blank line still runs.
- Press Ctrl-C to cancel the current entry without exiting. During a live agent call,
  Ctrl-C interrupts the call and stops the current entry; effects completed before
  cancellation remain visible, and unreached operations do not run.

### Console-only editing features

The interactive console front end (not the plain front end) additionally provides:

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
  directly in the config file. `:theme` still works and persists in the plain front
  end — it accepts the same names and saves the same way — but has no visible effect
  there, since plain output carries no colour.
- Command history persists under `~/.agm/repl_history`.

### Meta-commands

Meta-commands begin with a leading `:` (which never collides with AgL syntax):

| Command | Action |
|---------|--------|
| `:help` | List the available meta-commands |
| `:quit` / `:exit` (or Ctrl-D) | Exit the REPL |
| `:reset` | Clear the whole session (bindings, types, declarations, imports, and uses) |
| `:type EXPR` | Type-check `EXPR` against the session and print its type (no eval) |
| `:bindings` / `:env` | List current bindings as `name : Type = value` |
| `:set echo on\|off` | Toggle result echoing |
| `:agent confirm\|auto` | Switch the agent-call mode (or report it with no argument) |
| `:load FILE` | Load a saved transcript by its original entries, or an ordinary `.agl` file one item per entry |
| `:save FILE` | Write the accumulated session source and entry boundaries to a transcript |
| `:theme [dark\|light\|auto]` | Show or switch the syntax-highlighting theme; saves to `~/.agm/config.toml` |

### Agent-call confirmation

- By default the REPL is in **auto** mode: agent calls fire immediately without
  prompting, matching `agm exec`.
- `--confirm-agents` (or `:agent confirm`) starts/switches to **confirm** mode: before
  every live agent prompt, including `Session::ask` and each parse-retry follow-up, it
  shows the selected agent and rendered prompt (truncated, with a `[v]iew` option to
  print the full text) and asks `[Y]es / [n]o / [a]lways`. `yes` runs the call, `no`
  aborts the entry (rolling its bindings back), and `always` switches the session to
  auto mode for the rest of the session.
- `exec` shell calls are **not** gated; only agent calls are confirmed.

### Options

- `--strict-json` / `--no-strict-json`: Set JSON-codec strictness for agent output
  (lenient recovery is the default), as for `agm exec`.
- `--max-iters N`, `--max-call-depth N`, `--default-agent AGENT`: As for `agm exec`.
- `--confirm-agents`: Start in confirm mode, asking before each agent call (the default
  is auto; see [Agent-call confirmation](#agent-call-confirmation)).
- `--quiet`: Suppress the automatic echoing of entry results.
- `--no-stdlib`: Disable the automatic `import std/prelude::*` prelude for each
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
- `--plain`: Force the plain, non-interactive line front end (see
  [Front ends](#front-ends)) even when stdin and stdout are both terminals. There is no
  `--no-plain`; the auto-detected default already avoids the console front end whenever
  it would not work (a pipe, a redirected file, or `TERM=dumb`).

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
  trace destination. The initial `[exec] timeout` is also the idle timeout for CLI and
  Pi RPC agent sessions; a source `timeout` write changes only shell `exec`, not agent
  session timeouts. `:reset` clears
  the session, restoring the settings to the CLI/`[exec]` defaults set before the loop
  starts.

### Exit codes

The REPL itself only fails before the loop starts; ordinary per-entry errors are
reported inline and never exit the process. `std/process::exit(code)` is the
exception: it terminates the REPL host with its portable `0..255` status after
finalizing that entry's trace. A blank or non-string `--default-agent`/`[exec] default-agent`
value is one such pre-loop failure. A recognized direct Agent constructor is spliced into
`std/config` before the console starts, so wrong arguments or non-constant fields are
resolved, type-checked, and constant-checked (and any rejection names the flag or config
key) before the banner appears. Other text is custom command text rather than an AgL
parse error. An `AgentCommand(...)`
whose command text does not shell-split is deferred until the first entry reaches
interpreter construction, because session initialization does not construct an
interpreter. Construction validates the winning value before any statement executes, so
even an entry that performs no agent dispatch reports the error inline without exiting
the REPL.

| Code | Meaning |
|------|---------|
| `0` | The session ended normally (`:quit`/`:exit` or Ctrl-D) |
| `1` | Pre-loop setup failure: a blank/non-string or invalid canonical constructor in `[exec] default-agent` or `--default-agent`, or an unwritable `--log-file` — reported before the prompt appears |

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
