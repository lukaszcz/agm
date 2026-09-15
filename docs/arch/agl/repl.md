# AgL REPL

`agm repl` evaluates one entry at a time on the same pipeline and runtime as `agm exec`. This document covers the host behavior that makes it incremental.

## Incremental Sessions

`ReplSession` keeps a persistent environment: each entry is compiled and evaluated exactly once against accumulated symbols, types, declarations, and runtime values, so agent calls never replay. Successful declarations, bindings, imports, and `use` declarations persist; retained imports preserve method routes, and a later declaration at the same path supersedes the earlier one with a fresh declaration identity, so values and methods retained from before keep the exact shape they were checked against. A runtime-failed entry promotes only declarations, `use` contributions, and explicit scope regions reached by its initializer frontier, restores superseded metadata for everything unpromoted, and leaves the prior generation intact. `:reset` clears everything, closes agent sessions, and replaces the extern state.

Each entry runs in a fresh interpreter. Engine-setting writes persist across entries because they are ordinary statements; parameter seeds supplied through the session's optional module resolver are decoded after lowering and persist with the initialized module state, including later writes to seeded `var`s. The process environment is snapshotted once at startup for `std/env::environ`. Saved transcripts replay their original entry boundaries.

## Retained Library Image

Every entry compiles a whole program — the entry plus every library module the session has loaded — but the library half is compiled once. The session retains each library module's resolved, checked, and match-compiled artifacts and reuses one only while the graph still holds the very AST object it was derived from; a reparse or a redeclaration misses and recompiles that module alone. Fresh default-stdlib sessions in one process additionally start from a cached checked bootstrap image — a typed engine seed (e.g. a host-supplied `default-agent`) does not touch the compiled module graph, so it does not disqualify this cache.

Lowering allocates into a persistent `LinkImage` (`lower/repl.py`), separate from the checked-artifact cache above. An entry that fails partway through initializing newly imported library modules keeps that allocation delta rather than rolling it back, even for the modules it drops (so a later reload of one of them reuses the same declaration identities); only the dependency-complete subset is marked linked and cached. A dropped module is therefore never linked into any later entry's program, and [lowering](execution/lowering.md#the-ir) excludes its leftover symbol and function descriptors from what it emits.

## Front-End Seam

The loop body — meta-command dispatch, entry evaluation, result rendering, the continuation predicate — exists once in the UI-free `repl/loop.py`, parameterized by a reader and a writer. Two front ends wire it: `repl/console.py`, the only module that imports prompt_toolkit, drives the real lexer for highlighting and completion; `repl/plain_console.py` is a styling-free line front end for pipes and editor buffers, chosen automatically on a non-tty or `TERM=dumb` and forced by `--plain`.

## Result Rendering

`EntryResult` carries the `ValueDescriptors` view of the entry's own compiled program alongside its value, so `repl/render.py` renders a live value without any global registry; `ReplSession.descriptors()` builds the analogous cumulative view over every retained declaration for `:bindings`. `render_entry_result` decides whether an entry echoes at all from its checked static type, once: a `unit`-typed expression or binding echoes nothing, for both the live value echo and the `--dry-run` type echo, unless the `echo-unit` setting is on.

## Setting Persistence

Every `[repl]` setting (`theme`, `echo`, `echo-unit`) is saved through one path: a meta-command handler reports a change via `MetaOutcome.setting_change`, which `commands/repl.py` persists via `config/general.py`'s `save_repl_setting`. `--quiet` overrides `echo` for that session only, never persisted.

## Code Entry Points

- `src/agm/agl/repl/session.py` — the incremental session core; `entry_pipeline.py` — the multi-module entry pipeline over `PipelineDriver`.
- `src/agm/agl/repl/loop.py`, `console.py`, `plain_console.py` — the shared loop and the two front ends; `meta.py`, `render.py`, `themes.py` — meta commands, echo rendering, themes.
- `src/agm/agl/lower/repl.py` — incremental linking.
- `src/agm/commands/repl.py` — the CLI host and front-end selection.
- Tests: `tests/test_agl_repl_*.py`, `test_repl_command.py`.
