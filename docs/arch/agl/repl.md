# AgL REPL

`agm repl` evaluates one entry at a time on the same pipeline and runtime as `agm exec`. This document covers the host behavior that makes it incremental.

## Incremental Sessions

`ReplSession` keeps a persistent environment: each entry is compiled and evaluated exactly once against accumulated symbols, types, declarations, and runtime values, so agent calls never replay. `eval_entry(check_only=True)` checks an entry without evaluation or promotion. Successful evaluated declarations, bindings, imports, and `use` declarations persist, and scope receives every retained type path's `TypeOwner`, so owner-qualified constructors resolve alike across entries and constructor candidates are derived from them as in file mode (entry grouping never changes the result). A retained alias's own declaration fixes its target by declaration identity, not path, forever — a later redeclaration at the same path is a distinct declaration, never mistaken for it (`TypeOwnerIndex.owner`). While that identity is still current and the alias's spelling still selects it, the alias's inline members and hidden set are re-projected from the target's current owner, so a later import that lifts or adds a `hiding` reaches or hides through it exactly as file mode does; otherwise they freeze at their declaration-time set. Any other path beneath the alias reads its target as written in the current session, as in file mode. Declaring beneath a session alias, or declaring an alias where an earlier entry declared beneath it, is rejected in the entry completing the pair (`reject_declaring_beneath_retained` in `scope/resolver.py`). Retained imports preserve method routes, and a later declaration at the same written path supersedes the earlier one with a fresh declaration identity, so values and methods retained from before keep the exact shape they were checked against. Fixities persist with the session (`ModuleResolution.fixities`, threaded back as `entry_repl_session_fixities`), so an operator declared in one entry keeps its fixity in later ones. The session's own builtin declarations are likewise tracked by current identity, keyed by scoped name from the promoted declarations themselves: a later promoted declaration at the same name supersedes an earlier one instead of clashing with it, and only a `builtin` bare declaration (never a plain alias or def, nor a `builtin var`) counts. A runtime-failed entry promotes only declarations, `use` contributions, and explicit scope regions reached by its initializer frontier, restores superseded metadata for everything unpromoted, and leaves the prior generation intact. Companion `runtime.state` lives in one session-owned bag shared by every entry's interpreter and closed by `ReplSession.close` ([execution/ffi.md](execution/ffi.md)). `:reset` clears everything, closes agent sessions and that bag, and replaces the extern registry.

An entry resolves in source order exactly as a file does, its imports hoisted within the entry, so verdicts never depend on how source is grouped into entries. An import replaces a retained import of the same module at the same written scope path, and a `use` replaces retained uses by form, as scope decides (`ModuleResolution.replaced_uses`). A retained `use` re-reads its target on every read, seeing only the uses before it, and contributes nothing once the target is gone.

Each entry runs in a fresh interpreter. Engine-setting writes persist across entries because they are ordinary statements; parameter seeds supplied through the session's optional module resolver are decoded after lowering and persist with the initialized module state, including later writes to seeded `var`s. The process environment is snapshotted once at startup for `std/env::environ`. Saved transcripts replay their original entry boundaries.

## Retained Library Image

Every entry compiles a whole program — the entry plus every library module the session has loaded — but the library half is compiled once. The session retains each library module's resolved, checked, and match-compiled artifacts and reuses one only while the graph still holds the very AST object it was derived from; a reparse or a redeclaration misses and recompiles that module alone. Fresh default-stdlib sessions in one process additionally start from a cached checked bootstrap image — a typed engine seed (e.g. a host-supplied `default-agent`) does not touch the compiled module graph, so it does not disqualify this cache.

Lowering allocates into a persistent `LinkImage` (`lower/repl.py`), separate from the checked-artifact cache above. An entry that fails partway through initializing newly imported library modules keeps that allocation delta rather than rolling it back, even for the modules it drops (so a later reload of one of them reuses the same declaration identities); only the dependency-complete subset is marked linked and cached. A dropped module is therefore never linked into any later entry's program, and [lowering](execution/lowering.md#the-ir) excludes its leftover symbol and function descriptors from what it emits.

## Front-End Seam

The loop body — meta-command dispatch, entry evaluation, result rendering, the continuation predicate — exists once in the UI-free `repl/loop.py`, parameterized by a reader and a writer. Two front ends wire it: `repl/console.py`, the only module that imports prompt_toolkit, drives the real lexer for highlighting and completion; `repl/plain_console.py` is a styling-free line front end for pipes and other non-terminal consumers, chosen automatically on a non-tty or `TERM=dumb` and forced by `--plain`.

The Emacs integration runs the plain REPL in a `comint-mode` pty and applies the
editor mode's AgL font-lock rules to prompt input. This avoids treating comint
as a terminal emulator while preserving native syntax highlighting. It filters
the terminal echo and intermediate prompts produced while it injects source for
send/reload commands, retaining concise result and definition output followed
by a prompt for manually typed input.

## Introspection and Result Rendering

`EntryResult` carries the `ValueDescriptors` view of the entry's own compiled program alongside its value, so `repl/render.py` renders a live value without any global registry; `ReplSession.descriptors()` builds the analogous cumulative view over every retained declaration for `:bindings`. `render_entry_result` decides whether an entry echoes at all from its checked static type, once: a `unit`-typed expression or binding echoes nothing, unless the `echo-unit` setting is on. An entry that is no value but a type spelling (`S::Q`, `array[F]`) echoes the type scope selected for it.

`:info NAME` reads its spelling through scope's one lookup exactly as an entry would (`_resolve_info_reference`), without evaluating source; a rejection is scope's own error, turned into data only at `_static_verdict` and displayed through `meta._static_query`. It retains the resolver's exact constructor identity and selects its signature through the checker's own constructor selection (`typecheck/constructors.py`), so qualified, applied-owner, and `use`-renamed spellings read as the checker types them. An alias is found by the declaration identity its spelling selects (`TypeEnvironment.type_name_declaration`) and renders as its source declaration however it is spelled. It reports live bindings, functions (at their written path, including qualified library functions and operators), constructors, and retained nominal, generic, or alias types as concise AgL fragments; the rich console styles only those fragments through its live lexer and theme, while plain output remains text.

## Setting Persistence

Every `[repl]` setting (`theme`, `echo`, `echo-unit`) is saved through one path: a meta-command handler reports a change via `MetaOutcome.setting_change`, which `commands/repl.py` persists via `config/general.py`'s `save_repl_setting`. `--quiet` overrides `echo` for that session only, never persisted.

## Code Entry Points

- `src/agm/agl/repl/session.py` — the incremental session core; `entry_pipeline.py` — the multi-module entry pipeline over `PipelineDriver`.
- `src/agm/agl/repl/loop.py`, `console.py`, `plain_console.py` — the shared loop and the two front ends; `meta.py`, `render.py`, `themes.py` — meta commands, echo rendering, themes.
- `src/agm/agl/lower/repl.py` — incremental linking.
- `src/agm/commands/repl.py` — the CLI host and front-end selection.
- Tests: `tests/test_agl_repl_*.py`, `test_repl_command.py`.
