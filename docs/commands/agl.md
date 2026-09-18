# AgL workflow DSL

[`agm exec`](#agm-exec) runs AgL programs, [`agm check`](#agm-check) statically checks AgL
files, and [`agm repl`](#agm-repl) evaluates AgL interactively. The language is documented in
the [AgL language reference](../agl/reference/index.md).

## `agm exec`

```text
agm exec [--strict-json|--no-strict-json]
         [--max-call-depth N] [--default-agent AGENT]
         [--timeout DURATION|--no-timeout] [--dry-run]
         [--log|--log-file PATH|--no-log] [--no-log-file]
         [--no-stdlib]
         [-I DIR]... [-p PATH]
         (FILE | PACKAGE/MODULE::PROGRAM | -c COMMAND) [ARG]... [--NAME VALUE]...
```

Run an AgL program from exactly one source: a `FILE`, an installed `PACKAGE/MODULE::PROGRAM`
reference resolved through the active package selection (`agm exec review_tools/review::main`),
or inline `-c` text. An existing `FILE` path wins even if its name contains `::`.

A file must declare at least one `program def`. `exec` initializes the linked program and
invokes its sole entry, or the one selected with `-p` by declaration path (`review::main`).
Inline source without a `program def` is wrapped in a synthetic `program def main`; inline
source declaring one is an ordinary module with a static root, so its statements and
non-constant bindings belong in the program body.

### Module resolution

Imports resolve against an unordered set of search roots:

- the directory of `FILE` (the working directory for `-c`);
- if that directory is inside a development package: that package and the recursive closure of
  its relative-`path` manifest dependencies, each contributing its `src/` tree under its package
  name;
- the selected standard library (below);
- the AGM home's global `lib` directory (overridable by `[modules] lib_root`);
- `[modules] roots` from any config layer;
- `-I`/`--module-path` roots.

The standard library is the first of:

1. `AGM_STDLIB`, replacing the whole selection (the active package is not mounted as well);
2. a development `std` checkout whose module tree holds `FILE` (or the working directory),
   whatever version it declares;
3. the active immutable `<AGM-home>/packages/std/<AGM_VERSION>/` package matching the running
   binary; a selected active package of another version is an error;
4. AGM's bundled copy (`agm/stdlib` in a wheel, `packages/stdlib/` in a source checkout) when no active
   package is selected or its store tree is absent, so a wheel runs AgL even when the home has
   no active `std`.

`[modules] lib_root` and `roots` expand `%{VAR}` leniently, like other
[path-valued settings](config.md#path-valued-settings).

`AGM_HOME` selects the whole runtime home (config, prompts, sandbox settings, global library,
package store); by default AGM uses a populated `.agm` beside the executable, then `~/.agm`.
`AGM_STDLIB` relocates only the standard library.

A module name must resolve to exactly one file across all roots; zero or several distinct files
are static errors (exit 1).

A module or file-backed entry declaring `extern def` ([Python FFI](../agl/reference/ffi.md))
needs a companion `.py` at the same path. The path is derived, not searched, so root ambiguity
never applies. A missing companion, or one lacking a declared extern name as a callable
attribute, is a static diagnostic reported before the run.

`resource` and `resource-dir` ([Expressions](../agl/reference/expressions.md)) anchor at the
declaring module's directory or owning package root, so calling them from inline `-c` source or
a direct `agm repl` entry is a static error.

### Options

- `-c COMMAND`, `--command COMMAND`: Program source text, instead of `FILE`.
- `-p PATH`, `--program PATH`: Select a `program def` by declaration path (`main`,
  `review::main`). With a `PACKAGE/MODULE::PROGRAM` reference, replaces its program path and
  keeps its module.
- `ARG` / `--NAME VALUE`: The selected program's own value parameters; see
  [Program arguments](#program-arguments).
- `-I DIR`, `--module-path DIR`: Add a module search root (repeatable), relative to the working
  directory. e2e/fixture tests use it for test-specific roots.
- `--no-stdlib`: Disable the automatic `import std/prelude::*` in the entry and all library
  modules. Regardless of this flag, a module whose explicit import expansion includes
  `std/prelude` gets that contribution instead of the automatic prelude, so plain
  `import std/prelude` leaves prelude names qualified-only.
- `--strict-json`: Agents must return exactly one bare JSON value (no fences, prose, or repair).
- `--no-strict-json` (default): Recover exactly one JSON value from chatty output (stripping
  fences/prose, repairing trivially malformed JSON), then validate it strictly against the
  schema. `ask`'s `strict-json:` argument overrides either per call.
- `--max-call-depth N`: Maximum recursion depth (overrides `[exec] max-call-depth`; default 256).
  Exceeding it raises `RecursionError`.
- `--default-agent AGENT`: Seed `std/config::default-agent`, the agent of `ask` calls without
  `agent`, in [host Agent syntax](#host-agent-syntax). Decoded before execution, overriding
  qualified program-table/`[exec]` config; effective whether or not the program loads
  `std/config` or `--no-stdlib` is given. An `AgentCommand(...)` command is shell-split and
  validated before execution; a malformed one (e.g. an unclosed quote) exits 1 with nothing run.
- `--timeout DURATION` / `--no-timeout`: Override the initial shell-exec and agent idle
  timeouts, seeding `std/config::timeout` with `Some(DURATION)`, or remove configured ones,
  seeding `None`.
- `--log` / `--log-file PATH` / `--no-log` (mutually exclusive): Trace logging, **off by
  default**. `--log` writes to an auto-timestamped path under `.agent-files/`; `--log-file`
  writes a JSONL trace to `PATH`; `--no-log` disables it, overriding `[exec] log = true`. These
  set the initial state; a `std/config::log := true` write still enables tracing.
- `--no-log-file`: Clear only the initial `log-file` value; an `[exec] log-file` path or
  `--log`'s auto path still applies. Use `--no-log` to disable tracing.
- `--dry-run`: Run the static pipeline, program-argument validation, and contract
  materialization, but evaluate nothing: static errors exit 1, a clean check exits 0 with no
  program output. `extern def` companions are not imported (no side effects), so a broken
  companion does not fail a dry run. If the check succeeds and the program has agent-call,
  `exec`, or extern-call sites, their static inventory goes to stdout:

  ```
  call-sites:
    line N:C: <callee> → <target-type> [<codec>[, schema: yes][, policy: <policy>]]
  ```

  `N:C` is the 1-based line and column; `<callee>` is `ask`, `exec`, or an extern's declared
  name; `<codec>` is `text`, `json`, or `extern`; `schema: yes` marks an attached JSON Schema;
  `<policy>` is the parse-failure policy, `abort` or `retry[N]` (not for extern calls).
  Extern-backed standard-library methods (`[1].size()`, `"a".trim()`) are listed at your call
  site; the standard library's internal calls are listed only for modules the program imports
  explicitly.

### Program arguments

The selected `program def`'s value parameters project onto the CLI through AgL's
positional/standard/named-only zones. A parameter list without zone attributes is entirely
named-only, so `x: T` becomes `--x`; an `@arg-pos` parameter fills an `ARG` slot in declaration
order and is never addressable by name; an `@arg-std` parameter accepts a positional token or
`--x`.

A name-addressable parameter's type selects its flag form; every value-taking flag also accepts
`--x=VALUE`:

| Type | Flag |
|---|---|
| `bool` | `--x` / `--no-x`, no value |
| `Option[T]` | `--x VALUE` (`Some`) / `--no-x` (`None`); `VALUE` verbatim for `text`, [host Agent syntax](#host-agent-syntax) for `Agent`, else strict JSON or [value syntax](../agl/reference/host-environment.md#value-syntax) for `T` |
| `text` | `--x VALUE`, verbatim |
| `Agent` | `--x VALUE`, in [host Agent syntax](#host-agent-syntax) |
| `json` | `--x VALUE`, strict JSON or a value-syntax literal restricted to JSON-shaped data (no constructor calls) |
| other | `--x VALUE`, strict JSON or [value syntax](../agl/reference/host-environment.md#value-syntax) validated against the type |

A `path` parameter — `path`, `Option[path]`, or an alias of either — takes its value as the
matching `text` form does, with `PATH` as its default value placeholder. Shell completion offers
filesystem paths for its value (`--x <TAB>`, `-x <TAB>`, `--x=<TAB>`) and for its positional slot,
under `agm exec FILE` and a registered package command alike.

A positional slot has no `--no-x`, so `Option[T]` gets no special treatment there: `text` is
verbatim, `Agent` uses host syntax, and every other type, `Option[T]` included, is strict JSON or
value syntax of the declared type (`'{"$case": "Some", "value": "hi"}'` or `'Some("hi")'` for
`Option[text]`).

An omitted argument resolves as `CLI > @opt-env variable > qualified program table (see
[Configuration](#configuration)) > signature default > required error`; errors are reported
before any agent runs. A positional-only parameter has no name to key a config table by, so it
skips that step. A config key cannot spell `None` for `Option[T]`: a present key supplies
`Some`, an absent one falls through to the default; pass `--no-x` for `None`.

Also reported before any agent runs: a parameter supplied twice (two flags, or positional plus
`--x`), an unrecognized `--flag`, or more positionals than positional-capable parameters.

**Reserved names.** Program arguments and engine settings share one flag and config namespace,
so a name-addressable parameter named after an engine setting (`default-agent`, `strict-json`,
`timeout`, `log`, `log-file`) is a static error even if never supplied, also
reported by `agm check`.
A projected flag that collides with a reserved flag has no static check, only a host one: selecting that
program for execution fails, while `--help` and shell completion silently fall back to
`agm exec`'s own help and no completions. `agm exec` reserves:

- its own options: `--help`/`-h`, `--program`/`-p`, `--command`/`-c`, `--module-path`/`-I`,
  `--max-call-depth`, `--no-stdlib`, `--dry-run`;
- every engine-setting flag in both polarities: `--default-agent`,
  `--strict-json`/`--no-strict-json`, `--timeout`/`--no-timeout`,
  `--log`/`--no-log`, `--log-file`/`--no-log-file` (so `no-log: text` collides);
- other parameters' projected flags (`cache: bool`'s `--no-cache` vs `no-cache: bool`).

A [registered package command](pkg.md#registered-commands) reserves `--dry-run`, `-h`/`--help`,
and its run-time options: the engine-setting flags and `--max-call-depth`. Because of the help
fallback, `agm exec FILE --nope -h` prints `agm exec`'s help (exit 0) for a colliding program, but
is a usage error (exit 1) when the program's own parser exists to reject `--nope`.

**Tokens.**

- A bare `--` ends `agm exec`'s option scanning and reaches the program as its own
  end-of-options marker: in `agm exec FILE -- --odd-looking-value` and `agm exec FILE -- --help`
  the token after `--` is a positional. A `--` that introduces a flag-shaped source is consumed
  by `agm exec` instead (`agm exec -- --odd-name.agl` runs that file), so the program's marker
  needs a second one: `agm exec -- --odd-name.agl -- --odd-looking-value`.
- A value-taking flag consumes the next token even if flag-shaped (`--msg --x` supplies
  `--x`); spell the value inline (`--msg=--x`) when the next token is meant as its own flag.
- The selected program owns `-h`/`--help`. With one entry program, or one selected by `-p`,
  `agm exec FILE -h`, `agm exec -h FILE`, and `agm exec FILE -p NAME --help` print its help:
  usage line, `@doc` description, and per visible parameter its flags, value placeholder, and
  `@doc`. A `-h` where a value is expected (`--msg -h`, `-va -h`) is that value, and the
  program runs. With several entries and none selected, `agm exec`'s help is printed with the
  `-p` choices; an unreadable source or an unprojectable parameter also falls back to it.

Inline `-c` source without a `program def` gets a parameterless synthetic `main`, so it accepts
no `ARG`/`--NAME` tokens; declare a `program def` to add parameters. Even then it takes no
positionals: a bare token after `-c COMMAND` is the mutually exclusive `FILE` selector
(`agm exec -c '…' hello` fails with `error: argument FILE not allowed with -c/--command`),
although `-c … --help` shows a positional usage slot for positional-capable parameters. Named options work, so give inline
programs only standard or named-only parameters.

#### Attributes on a parameter

Language-side definitions: [Program arguments](../agl/reference/host-environment.md#program-arguments).

| Attribute | Effect on the CLI |
|---|---|
| `@doc("prose")` | On the `program def`, the help's description; on a parameter, that option's help entry |
| `@opt-name("flag-word")` | Renames the flag, its `--no-` negation, its config key, and its completion |
| `@opt-short("t")` | Adds `-t` beside the long flag |
| `@opt-env("VAR")` | Reads `VAR` when no CLI token supplies the parameter |
| `@opt-metavar("PATH")` | Replaces the value placeholder in usage and help |
| `@opt-hidden` | Omits the parameter's `--name` entry from `--help` and from shell completion; a positional-capable parameter keeps its usage slot |

A short flag takes `-t VALUE` or `-tVALUE`. One-letter flags group (`-abc` is `-a -b -c`), and
only the last may take a value, so in `-va -h`, `-h` is `-a`'s value, not a help request.
Completion offers short spellings alongside long ones, except for `@opt-hidden` parameters.

An **empty `@opt-env` variable counts as unset** (`VAR= agm exec FILE` falls through to the
config table, then the default), so it cannot deliver an empty `text`; use a `""` default or
`--x=""`.

### Module parameters

An `@param let` or `@param var` in the selected program's transitive import closure becomes a
host option. The declaring module's external parameter name is used for a bare flag when it
resolves, and for dotted qualified flags in all cases: an `A/logging` parameter `verbose` can
be `--verbose`, `--logging.verbose`, or `--A.logging.verbose`; a `scope debug` parameter is
`--logging.debug.trace` or `--A.logging.debug.trace`. `bool` and `Option[T]` also have the
corresponding `--no-...` form, and `@opt-short("v")` adds `-v`. Module and scope paths are dotted
in flags; `::` is only for declaration paths in AgL source.

Bare names are claimed by the host, the selected program's value parameters, its own module
parameters, and imported module parameters in that order. A shadowed parameter keeps its
qualified flags. Several parameters in one level can make a bare flag ambiguous; declaring them
is valid, but using that flag is an error that names the declarations. `@opt-name` supplies the
external name in flags and config; `@opt-env` is the fallback after CLI flags; `@opt-hidden`
removes the entry from help and completion without disabling it.

Selected-program help shows its usual options first, then one `Parameters of MODULE` section for
each closure module. Each visible module parameter appears under its shortest resolving spelling.
For module parameter precedence and configuration routes, see
[Module parameters](../agl/reference/host-environment.md#module-parameters).

### Host Agent syntax

Every CLI argument or TOML string of the standard `Agent` type accepts (a config value may
instead be a native TOML table, read directly as the tagged JSON object below):

- `claude/MODEL-EFFORT` → `AgentClaude(MODEL, EFFORT)`
- `codex/MODEL-EFFORT` → `AgentCodex(MODEL, EFFORT)`
- `pi/PROVIDER/MODEL-EFFORT` → `AgentPi(PROVIDER, MODEL, EFFORT)`
- any other `PROVIDER/MODEL-EFFORT` → `AgentPi(PROVIDER, MODEL, EFFORT)`

The last hyphen separates the model from an opaque, non-empty effort suffix of any vocabulary.
Exact lowercase `claude/` and `codex/` prefixes win over the generic Pi form. Failing shorthand, an
Agent-typed program parameter, `Agent`-typed flag, `--default-agent`, or the `default-agent` config
key is read, in order: as a JSON object; then as an
[AgL value syntax](../agl/reference/host-environment.md#value-syntax) `Agent` member constructor
call (`AgentCodex(model = "o3", thinking = "high")`, bare or qualified `Agent::AgentPi(...)`); text
naming no member this way, with no `(` following it, is a verbatim `AgentCommand`
(`--default-agent 'worker --flag'`). Text that does open a member call but fails to read or bind —
an unclosed `AgentClaude(model = "x"`, an unknown field, a qualifier naming anything but `Agent` —
is a host error, not a verbatim command. Whitespace-only text is always a host error, never a
verbatim empty command.

### Agents

`ask` takes a typed `Agent` value as `agent`. Without one it uses the lazy default session,
which snapshots `std/config::default-agent` at first use; later free asks reuse that agent and
conversation:

```agl
let reviewer = AgentClaude("sonnet", "medium")
let review: Review = ask("Review %{artifact}", agent = reviewer)
let answer: text = ask("Summarize")
```

`AgentCommand(command)`, `AgentClaude(model, thinking)`, `AgentCodex(model, thinking)`, and
`AgentPi(provider, model, thinking)` each build their own argv; select one with an `Agent` value
or `default-agent`.

### Agent command interpolation

The argv an `Agent` builds (an `AgentCommand`'s command string; provider records' fixed flags)
interpolates `%{name}` strictly from the process environment plus `PROMPT_FILE`, which wins on
conflicts. Unlike `agm loop`'s runner and selector, no workflow variables are added. The `%%`
alias, `\%{` escape, and shlex splitting follow
[Runner command interpolation](agents.md#runner-command-interpolation). A prompt-file
placeholder places the prompt file there; otherwise AGM appends `@<path>`, except `AgentCodex`,
which pipes the prompt on stdin. AgL text literals interpolate `%{…}` themselves, so write
`\%{PROMPT_FILE}` inside `AgentCommand("…")`. An unresolvable hole raises a catchable
`AgentCallError` with `cause` `"interpolation_failure"`.

### Session runners

A continuing `AgentCommand` session requires an unescaped `%{SESSION_ID}` in its command. AGM
substitutes a generated id into the argv (not the child environment); the command must use it to
create or resume its transcript. Free `ask` with an `AgentCommand` default agent,
`Session::open(AgentCommand(...))`, and `AgentCommand(...).ask(...)` with corrective retries open
continuing sessions; a single-attempt `AgentCommand(...).ask(...)` sends one prompt and needs no
placeholder. Opening a session from a command without it raises `SessionError`. See
[Agent calls](../agl/reference/agent-calls.md#sessions) for all session backends.

### Configuration

`[exec]` in `config.toml` supplies engine defaults, overridable by CLI flags and source
`std/config` writes:

```toml
[exec]
default-agent = "claude/sonnet-medium" # native shorthand or custom command
strict-json = false         # lenient JSON recovery is the default
timeout = "30m"             # initial shell-exec and agent idle timeout
log = false                 # trace logging off by default; set true to enable
# log-file = "trace.jsonl" # explicit trace path (omit for auto timestamped path)

```

Qualified tables address a declaration by module suffix and scope path. A loose entry file's
module component is its `.agl` stem; a file run from a package (development checkout, installed
store tree, or selected standard library) keeps its package-qualified route, like an installed
reference. A `review::main` program in `review-tools/review` reads engine overrides and its
value parameters from `[review-tools.review.review.main]`, a shorter unambiguous suffix, or the
exact quoted route `["review-tools/review".review.main]`. Inline `-c` programs have no route, so
their parameters are CLI-only (CLI value, then signature default).

Module parameters use their declaring module's **module route**. For `A/logging`, root bindings
use `[A.logging]`; bindings in `scope debug` use `[A.logging.debug]`, or the exact module anchor
`["A/logging".debug]`. Leaves use `@opt-name` when present. The selected program's table is the
**program route**: it overrides a module parameter through any resolving spelling — the bare
external name, or a dotted qualified spelling as a quoted key such as `"A.logging.verbose"`,
which reaches a parameter whose bare name a nearer declaration claims — and wins over the
program's own [`@config`](../agl/reference/attributes.md#config) entries, which in turn win over
the module route; CLI and `@opt-env` still win over all three. An inline entry has no
config-file route, but its `@config` entries still apply to its own module parameters, so
they resolve as CLI > `@opt-env` > `@config` > initializer; its imported modules retain
their module routes and may still be seeded by `@config`.

A key in the selected program's table naming neither a name-addressable parameter nor an engine
setting (typically a misspelling) is reported on stderr and ignored; one naming a
positional-only parameter is reported with a distinct message (supply it positionally). Either
way the program runs on its declared defaults.

A TOML value is already host-native, unlike a CLI token or `@opt-env` variable, which are always
text: a native TOML string for any parameter type other than `json` or `Option[json]` is read the
same way a CLI token is (verbatim for `text`, else strict JSON or value syntax); a native TOML
string for a `json`- or `Option[json]`-typed parameter is instead the parameter's own JSON
*string* value, never re-read as JSON source or value syntax, and a native TOML table or array
crosses as the matching JSON object or array directly.

#### Source-level engine settings (`std/config`)

A program sets its **engine settings**, mutable bindings backed by the live engine and readable
by qualified reference, by importing `std/config` and writing them:

```agl
import std/config

program def main(spec: text) -> unit =
  std/config::log := true             # enable trace logging for this program
  std/config::log-file := Some("trace.jsonl")  # explicit trace path
  std/config::strict-json := true     # require bare JSON from agents
  std/config::default-agent := AgentClaude("sonnet", "medium")
  std/config::timeout := Some("30s")  # shell-exec idle timeout

  let result = ask "Process %{spec}"
  print result
```

A qualified target (`std/config::KEY := …`) always works; after `import std/config::*`, so does
bare `KEY := …`. `timeout` (`Option[text]`) and `log-file` (`Option[path]`) take `Some("…")` or
`None`.

Precedence for `default-agent`, `log`, `strict-json`, `log-file`, and `timeout` is
`source write > CLI > qualified program table > @config > [exec].X > engine default`, where
`@config` is the selected program's own [`@config`](../agl/reference/attributes.md#config)
entries. CLI, config, and `@config` supply the **initial** value; a source write overrides it
from that program point on: after `--no-log`, `std/config::log := true` enables tracing from
there, and `std/config::strict-json := true` overrides `[exec] strict-json = false`.

Writes take effect **positionally**, like `var` mutation. `log`/`log-file` writes reconfigure
the trace destination for subsequent calls; `log-file := Some(path)` enables logging, and a later
`log := false` disables it without clearing the path. `strict-json` and `timeout`
writes affect subsequent agent-output parsing and `exec` calls.

A CLI, program-table, or `[exec]` timeout seeds both the shell-exec and agent idle timeouts; a
source `timeout` write changes only the **shell-exec** timeout; agent idle timeout cannot change
mid-program. A bad duration in a source write is a runtime AgL error (exit 2, as the write is
evaluated at runtime); a bad `--timeout`, program-table, or `[exec]` value exits 1
before execution. A valid written timeout keeps its original text, while the parsed duration
drives shell execution.

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | The workflow completed successfully, or `std/process::exit(0)` requested success |
| `1` | Pre-execution failure: unreadable file, static language diagnostics (including invalid `case` coverage), host configuration error, or program-argument validation failure; it can also be requested with `std/process::exit(1)` |
| `2` | The workflow executed but ended with an uncaught AgL exception; it can also be requested with `std/process::exit(2)` |
| `3`–`255` | Requested by `std/process::exit(code)` |

`std/process::exit` accepts only the portable range `0..255`, so every supported host preserves
the status; an out-of-range value is a runtime error, not a termination.

### Diagnostics and warnings

- Error diagnostics (static errors, including non-exhaustive or redundant `case` arms, host
  configuration errors, program-argument failures) and uncaught AgL exceptions go to stderr and
  set the exit code.
- Advisory **warnings** go to stderr as `warning: line N: message` and never affect the exit
  code; the program runs to completion.
- Program `print` output goes to stdout, free of diagnostics.

### Compilation cache

Imported modules, including the standard library, are precompiled on demand into
`$XDG_CACHE_HOME/agm/agl` (default `~/.cache/agm/agl`). Source and dependency edits, compiler
updates, and relevant compilation settings invalidate artifacts automatically; initialization and
execution always use the current invocation's configuration and runtime state. The cache is
disposable: deleting it or making it unavailable does not prevent execution.

## `agm check`

```text
agm check [-I DIR]... [--no-stdlib] FILE...
```

Run the full **static** pipeline (parse, module loading, scope resolution, type checking, match
compilation, lowering) on each `FILE` and print GNU-style diagnostics, never evaluating anything
or running an agent. Every `FILE` is checked independently, in argument order, even after a
failure; a clean `FILE` prints nothing.

No `program def` is required, so library modules (which `agm exec --dry-run` rejects) can be
checked. A `program def` is validated with its module, including that each value parameter's
type can cross the host/JSON boundary (`text` verbatim, or a finite, JSON-decodable data type;
a function type is rejected) and that no name-addressable parameter uses a reserved
engine-setting name. `check` never selects, configures, or runs an entry: no argument
projection, no `-p`.

[`agm pkg check`](pkg.md) is different: it validates a package's manifest and module-tree
discipline (dependency declarations, mounted roots, registered command references), not AgL
program correctness.

### Module resolution

The [`agm exec` roots](#module-resolution) (the file's directory and containing development
package, selected standard library, global `lib`, `[modules] roots`, `-I` roots), assembled
fresh per `FILE`, since they anchor at the checked file.

### Options

- `-I DIR`, `--module-path DIR`, `--no-stdlib`: As for `agm exec`, per checked file and its
  library modules.
- `--dry-run`: Accepted like on every command, but a no-op: `check` has no side effects.

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | No `FILE` produced an error-severity diagnostic |
| `1` | Some `FILE` produced an error-severity diagnostic, was unreadable or missing, or had an invalid module-root configuration |

### Diagnostics and warnings

Diagnostics go to stderr in `agm exec`'s compiler style: location, `error:` or `warning:`,
message. Most carry a span, so the location has one of four shapes:

```text
path:line: error: message
path:line:col: error: message
path:line:col-endcol: error: message
path:line:col-endline:endcol: error: message
```

Related notes are indented two spaces under their diagnostic. A diagnostic from an imported
module carries that module's path. **Warnings** (e.g. a TAB-indented line) never affect the exit
code.

### Example

```bash
agm check workflow.agl
agm check lib/*.agl                    # library modules, no `program def` required
agm check -I vendor/ workflow.agl       # add an extra module search root
agm check --no-stdlib lib/helpers.agl
```

```bash
$ agm check lib/helpers.agl            # clean: no output
$ echo $?
0
$ agm check bad.agl                    # static error on stderr
bad.agl:3:11: error: 'undefined-name' is not defined.
$ echo $?
1
```

## `agm repl`

```text
agm repl [--strict-json|--no-strict-json]
         [--max-call-depth N] [--default-agent AGENT]
         [--quiet] [--dry-run] [--no-stdlib] [--log|--log-file PATH|--no-log] [--plain]
```

Interactive AgL. Unlike `agm exec`, which runs a whole program in a fresh environment, the REPL
keeps a **persistent session**: each entry is parsed, statically checked (including pattern
coverage), and evaluated once against an environment accumulating bindings, types, and
declarations, so earlier results stay available and agent calls fire exactly once.

### Front ends

Both front ends share session and evaluation behavior:

- **Console** (prompt_toolkit): syntax highlighting, tab-completion, multiline editing,
  history, colour themes; see [Entry editing](#entry-editing) and
  [Console-only editing features](#console-only-editing-features).
- **Plain**: no styling, colour, or ANSI escapes. Same `agl> ` / `...> ` prompts; reads stdin
  lines and accumulates multiline entries like the console, so pasted or programmatically sent
  blocks work. For pipes and non-terminal consumers such as an editor's comint buffer.

Plain is used automatically when stdin or stdout is not a terminal, or `TERM=dumb`; `--plain`
forces it on a terminal. No flag forces the console onto a non-terminal.

The REPL reuses `[exec]` settings for `default-agent`, call depth, JSON strictness,
and timeout. As in `agm exec`, each typed `Agent` value selects its own backend command,
`--default-agent` and `[exec] default-agent` accept [host Agent syntax](#host-agent-syntax), and
both are effective whether or not the session loads `std/config` or `--no-stdlib` is given.

Free `ask` lazily opens one default conversation, snapshotting `default-agent` at first use;
later free calls reuse it even if the setting changes. Explicit `Session::open` sessions stay
live until closed or the REPL exits. `:reset` clears AgL bindings and settings but **not** host
sessions: a later free `ask` continues the default conversation, and unneeded explicit sessions
must be closed yourself.

Each loaded program gets the automatic prelude as in `agm exec`, so `Option`, `Some`, `None`,
etc. are unqualified from a fresh prompt.

When the REPL first loads a module with `@param` bindings, it seeds them from that module's
module route. It does not read program routes and offers no module-parameter flags. A seeded
`var` keeps later writes for the session; `:reset` restores the configured initial value when
the module is loaded again.

An imported module's `extern def` companion ([Python FFI](../agl/reference/ffi.md)) is imported
once per session, so its module globals last for the session; `:reset` discards the cached
companion and a later import creates new globals. A value from `runtime.state(...)` instead
belongs to the interpreter evaluating one entry and ends with that entry, even while the
companion stays cached. A direct entry has no backing file, so it cannot declare `extern def` or
call `resource`/`resource-dir`; imported file-backed modules use them normally.

### Entry editing

In both front ends:

- Multiline editing is **AgL-aware**: Enter on an unterminated block (`record`, `enum`, `if`,
  `case`, `try`, `do`, …) or a line-final raw-tail header (`exec$`, `ask$`) opens a `...>`
  continuation; a complete entry submits. Enter on a blank continuation line force-submits. The
  plain front end applies the same completeness test to stdin lines, except that an entry whose
  latest line is indented stays open until a blank line or end of input, since an indented
  block parses after every line yet can always take one more.
- Ctrl-C cancels the current entry without exiting. During a live agent call it interrupts the
  call and stops the entry; effects completed before cancellation remain, unreached operations
  do not run.

### Console-only editing features

- Highlighting and completion follow the live session. Highlighted: keywords, string/number
  literals, operators, builtin types (`text`, `int`, `decimal`, `bool`, `json`, `array`,
  `dict`, `unit`), and types and constructors declared in the session or the current line.
  Declaration sites colour by position (the name after `record`/`enum`/`type` is a type, an
  inline enum member after `|` a constructor), so a type and a like-named constructor differ
  even mid-declaration; at use sites a constructor call (`Box(…)`, `ok::[…](…)`) colours as a
  constructor and an annotation as a type. Completion offers keywords, binding names, and
  meta-commands.
- Themes: **dark** (VS Code Dark+), **light** (VS Code Light+), and the default **auto**, which
  reads the terminal background from `$COLORFGBG` (set by most terminals; falls back to dark).
  `:theme dark|light|auto` switches and saves to `[repl] theme` in `~/.agm/config.toml`, which
  can also be set directly. In the plain front end `:theme` works and persists but has no
  visible effect. `:set echo` and `:set echo-unit` (see [Meta-commands](#meta-commands)) persist
  the same way, to `[repl] echo` and `[repl] echo-unit`.
- History persists in `~/.agm/repl_history`.

### Meta-commands

Meta-commands start with `:`, which never collides with AgL syntax:

| Command | Action |
|---------|--------|
| `:help` | List the available meta-commands |
| `:quit` / `:exit` (or Ctrl-D) | Exit the REPL |
| `:reset` | Clear the whole session (bindings, types, declarations, imports, and uses) |
| `:type EXPR` | Type-check `EXPR` against the session and print its type (no eval) |
| `:info NAME` | Show the current binding, function, or type as concise AgL; the rich console highlights it |
| `:bindings` / `:env` | List current bindings as `name : Type = value` |
| `:set echo on\|off` | Toggle result echoing |
| `:set echo-unit on\|off` | Toggle echoing `unit`-typed entries too (off by default) |
| `:load FILE` | Load a saved transcript by its original entries, or an ordinary `.agl` file one item per entry |
| `:save FILE` | Write the accumulated session source and entry boundaries to a transcript |
| `:theme [dark\|light\|auto]` | Show or switch the syntax-highlighting theme; saves to `~/.agm/config.toml` |

### Options

- `--strict-json` / `--no-strict-json`, `--max-call-depth N`,
  `--default-agent AGENT`: As for `agm exec`.
- `--quiet`: Do not echo entry results, for this session only (does not persist and overrides a
  saved `echo = true`).
- `--no-stdlib`: Disable the automatic prelude for every loaded program (entries and library
  modules); explicit imports still work. `:reset` keeps this choice.
- `--log` / `--log-file PATH` / `--no-log`: As for `agm exec`; with `--log-file`, each evaluated
  entry appends its JSONL records to `PATH` as one trace *run*. `--dry-run` writes no trace.
- `--dry-run`: Run each entry through the static pipeline (parse, resolve, typecheck, match
  compilation) but **never evaluate** it: no agent or `exec` calls, no persisted bindings. The
  inferred type is echoed (`name : Type` for a binding, `: Type` for an expression), for
  exploring types.
- `--plain`: Force the plain [front end](#front-ends). There is no `--no-plain`.

### Evaluation notes

- Blank and comment-only entries (`#` starts a comment) are no-ops: a fresh prompt, no error.
- **Expression and binding entries** echo `: Type = value` or `name : Type = value`, except that
  by default an entry whose type is `unit` echoes nothing — this covers both an explicit `()`
  result and a statement-like expression (`print`, `:=`, an else-less `if`, a loop, a discarded
  binding), so every `unit`-typed entry is silent uniformly. `:set echo-unit on` echoes them too.
- **Declaration entries** echo `NAME declared`, with the full path for a scoped declaration
  (`Tools::twice declared`) or `scope … end` region (`Tools declared`). `import`, `use`,
  `export`, and fixity declarations echo nothing.
- **Bare type expressions** (`int`, a declared `enum`/`record`/`type` name, `array[int]`,
  `(int) -> bool`) echo the resolved type (`<type: int>`) instead of ``'X' is not defined.``; an
  unapplied generic such as `Option` shows its generic definition. A REPL convenience only; names
  that are also values (a record constructor, a binding) evaluate normally.
- **Engine settings**: import `std/config` and write a qualified target
  (`std/config::strict-json := true`). The write takes effect positionally, so subsequent entries see
  it even if a later expression in the same entry fails; `log`/`log-file` writes reconfigure the
  trace destination. The initial `[exec] timeout` is also the idle timeout for CLI and Pi RPC
  agent sessions; a source `timeout` write changes only shell `exec`. `:reset` restores the
  pre-loop CLI/`[exec]` defaults.

### Exit codes

Per-entry errors are reported inline and never exit; the REPL fails only before the loop starts.
The exception is `std/process::exit(code)`, which ends the REPL with its `0..255` status after
finalizing the entry's trace.

`--default-agent`/`[exec] default-agent` is decoded before the loop, before the session is even
built: a blank value, or text that opens a constructor call but fails to read or bind, exits with
an error naming the flag or config key. Other text is custom command text, not an AgL parse error.
An `AgentCommand(...)` whose command does not shell-split fails only when the first entry
constructs the interpreter (session initialization constructs none); construction validates the
winning value before any statement runs, so even an entry with no agent dispatch reports it inline
without exiting.

| Code | Meaning |
|------|---------|
| `0` | The session ended normally (`:quit`/`:exit` or Ctrl-D) |
| `1` | Pre-loop setup failure: a blank or invalid `[exec] default-agent` or `--default-agent`, or an unwritable `--log-file` — reported before the prompt appears |

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

# Explore types only — no agent or exec calls fire, nothing is persisted.
agm repl --dry-run
agl> 1 + 2
: int
```
