# AgL REPL and Program Hosting

`agm exec` runs a complete AgL program; `agm repl` evaluates one entry at a
time. Both use the same compilation pipeline and runtime. This document covers
the host behavior that makes the REPL incremental.

## Incremental Sessions

A REPL session retains successful declarations, bindings, types, and runtime
state, so later entries can refer to earlier entries without replaying them.
A later declaration at the same reachable path supersedes the earlier one.
Static failures do not promote their entry; completed runtime effects retain
their ordinary REPL behavior. Nominal redeclarations receive fresh declaration
identities, so retained values and methods keep the exact record, enum-member,
or exception shape against which they were checked.

Each evaluated entry still creates a fresh `IrInterpreter`. The shared extern
registry keeps a loaded companion module and its ordinary Python globals for
the session, but the companion `runtime.state` accessor reaches an
interpreter-owned bag through the extern call's `ContextVar`; that bag lives
for one entry only. `:reset` also replaces the registry, so companions and
their module globals reload on a later import.

Imports and `use` declarations also persist after a successful entry. A later
import replaces retained declarations for the modules it names at the same
scope path. A later `use` replaces a retained use with the same resolved target
at that path, even when an import alias changes. Retained replay compares prior
semantic identities only with one another and leaves current-versus-retained
replacement to scope classification before the effective entry is compiled, so
single-member aliases replace their semantic parent target immediately while
nested whole-target aliases remain distinct. Retained wildcard-facade uses also
carry their source wildcard identity: replay refreshes routes added by that
wildcard without adopting modules from a later wildcard that reuses its alias,
while still falling back to their retained semantic routes when imports are
renamed or replaced. Retained generations carry semantic identities rather than
reconstructing them from import headers; a failed replacement entry still
leaves the prior generation intact, and uses and imports at other paths remain. `:reset`
clears retained declarations, imports, uses, and session runtime state. Saved transcripts preserve
and replay their original entry boundaries, including declaration-wide forward references; ordinary
source files loaded with `:load` still replay each top-level item independently. Retained explicit
`std/core` imports suppress the normal per-entry prelude; `--no-stdlib`
disables that prelude for the whole session.

## Hosts and Settings

`exec` discovers a program's parameters and selects its entry function. The
REPL obtains imported parameter values from configuration and requires defaults
for entry-local parameters. Both hosts seed engine settings and use the shared
runtime for agents, shell commands, and standard-library services. Agent enum
members cross the host boundary as their nominal record values rather than as
enum wrappers.

Engine settings — `default-agent`, `log`, `log-file`, `strict-json`,
`max-iters`, `timeout` — are root `builtin var` bindings of `std/config`;
scoped `std/config` bindings and other standard-library host-backed bindings
are module-, scope-, and name-keyed values rather than engine settings. Because
a write is an ordinary statement, settings take effect in program order and a
completed REPL write persists across entries. The session also snapshots the
process environment once at startup and supplies it as `std/env::environ`, so
later AgL environment mutations stay inside the session. `std/process::exit`
ends the REPL host: its `SystemExit` is re-raised only after the entry trace
receives its final `run_end`, matching batch execution.

Fresh default-stdlib sessions in one host process reuse an unchanged checked
bootstrap image from a small per-process LRU cache; changed source content,
missing required extern companions, changed canonical module paths, changed
root discovery, setting overrides, roots, and host capabilities each select a
fresh image. Retained user `infixl`/`infixr` fixity resolves relative
priorities against the same bare-visible assembly table used for the submitted
entry, without retaining imported operators as session declarations.

## Code Entry Points

- `src/agm/agl/repl/` — session state, entry pipeline, console, rendering, and
  agent confirmation.
- `src/agm/agl/pipeline.py` — shared preparation and execution pipeline.
- `src/agm/commands/exec.py` and `src/agm/commands/repl.py` — CLI hosts.
- `src/agm/cli_support/` — execution parameters and host engine-setting seeds.
- Tests: `tests/test_agl_repl_*.py` and `tests/test_exec_command.py`.
