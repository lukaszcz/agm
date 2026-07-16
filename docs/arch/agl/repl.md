# AgL REPL and Program Hosting

Two commands host AgL programs: `agm exec` runs a whole program, and `agm repl` evaluates entries one at a time. Both run the same pipeline ([index.md](index.md)); they differ in how the program is supplied and how parameters, agents, and configuration are wired around it. This document covers that host-facing surface.

## Incremental REPL Session

The REPL is a UI-free incremental driver that runs the full parse → resolve → check → match-compile → lower → deep IR validation → eval pipeline one entry at a time against two persistent images: the static session environment and the linked IR/runtime image. It reuses the firewalled passes' seam parameters so each entry's node ids stay globally unique, new references fall through to prior session bindings, and new declarations shadow. Each entry contributes a delta of initializers executed against one persistent base frame, so earlier initializers and host calls are never replayed, and the session echoes the trailing expression's result unless that result is the non-printable unit value used by statement-like effects.

Every entry is also an inference boundary. Its checker regions and provisional metadata close before promotion; the persistent seed contains only concrete checked types and rigid generic declaration templates. A parse, scope, or type error promotes nothing, so a later entry starts with fresh generic occurrences and no constraints or flexible inference variables from the failed entry.

Runtime failure is deliberately non-transactional: initializers completed before a failure remain installed (including mutations and new bindings), and the static environment advances only for the symbols that actually reached the runtime frame, keeping later name resolution aligned with the partially advanced image. When a session entry redefines a nominal type, prior bindings whose static type mentions that entry-local nominal are invalidated so old runtime values cannot be type-checked against the new type shape.

Match compilation runs before the check-only early return and before `:type` reports a type, so non-exhaustive or redundant cases cannot bypass static validation by avoiding linking or execution.

Type-focused REPL surfaces (`:type`, check-only echoes, and bare type entries) use a REPL-specific type formatter rather than raw semantic `repr`: most types stay compact, while records and enums expand to declaration-like field and constructor listings; bare unapplied generic record/enum names expand to their generic definitions. Ordinary binding/value echoes keep compact type names to remain readable.

The REPL uses the graph pipeline ([modules.md](modules.md)) for entries by default so it can apply the same automatic `std.core` open import as `agm exec`. The graph resolver merges those imported stdlib names with the session's accumulated bindings, constructor candidates, and type names, preserving incremental REPL behavior while keeping stdlib types and constructors available from a fresh prompt. Library modules are cached and incrementally linked into the persistent image; open-imported names are made to persist across entries by accumulating import declarations and replaying them into later entries. User `infixl`/`infixr` fixity likewise persists: resolved priorities are accumulated across entries and fed back to the parser as ambient fixity so an operator declared in one entry is usable in the next.

## Program Parameters

`param` declarations make a program parameterizable. The pipeline can discover the typed parameter inventory before execution, which `agm exec` uses to expose each parameter as a first-class CLI option (boolean params become flag pairs; names that collide with built-in flags are rejected) and to load values from the `[<program>]` config section. Resolution precedence is external value > default expression > error for a required param, and external values are converted and type-checked before any evaluation, so a bad value fails before any agent or shell call runs. In the REPL the same precedence applies, with config values converted in a pre-evaluation check.

## Agent Declaration and Reconciliation

Agents must be declared in source; the host backs declared names but never owns the name set. The pipeline prepares a program once (lex, parse, scope) and reuses that prepared object for both parameter discovery and execution, so source is never parsed twice. Before execution it reconciles the declared agents against the host's registrations: a registration with no matching declaration, or a declared agent with neither a dedicated registration nor a default backing, is an error; a declared-but-uncalled agent is a warning.

`agm exec` wires the backings by reading the declared inventory and registering each name with a runner-backed factory. The runner command is chosen by precedence across config, a source runner hint, CLI flags, and a built-in default floor, so every declared agent resolves and also backs `ask`. The agent-runner mechanics are shared with the rest of AGM (see [agents.md](../agents.md)).

## Engine Settings

The engine settings a program can tune — `runner`, `log`, `log-file`, `strict-json`, `max-iters`, `timeout` — are declared as `builtin var` bindings in the standard-library module `stdlib/std/config.agl`, one per setting (`src/agm/config/engine_keys.py` is the shared name/type catalog, mapped to AgL types by the semantics layer). A `builtin var` is a body-less, runtime-backed mutable binding: the scope and type passes treat it like any binding, and the lowerer routes reads to `IrBuiltinLoad` and writes to `IrBuiltinStore` (`lower/lowerer.py`), keyed by the setting name rather than an ordinary frame cell.

The interpreter splits the six settings into two backings (`eval/ir_interpreter.py` `_store_builtin_setting` / `_load_builtin_setting`):

- **Runtime-live** (`strict-json`, `max-iters`, `timeout`) update live interpreter fields (`_strict_json`, `_loop_limit`, `_shell_exec_timeout`), so the effect takes hold from the write onward for subsequent `ask`, loop, and `exec` evaluation.
- **Host-consumed** (`runner`, `log`, `log-file`) update an in-interpreter register and, when a `HostSettingsReconfigurer` is present, reconfigure the live host service on write (`_reconfigure_host_service`, `runtime/host_settings.py`): a `runner` write rebuilds the default agent for later unnamed `ask` calls; `log`/`log-file` writes repoint the trace destination.

Because a write is an ordinary positional statement, settings take effect in program order and `agm exec` runs the program in a **single phase** — there is no separate startup pass. The CLI and config-file layers seed the initial register/field values (see [config.md](../config.md)); a source write then overrides them from its point onward. In the REPL a write persists across entries the same way any mutation does; `:reset` restores the seeded defaults.

## Console and Confirmation

The REPL console adds interactivity around the session. Live agent calls are gated by a confirmation wrapper holding a shared confirm/auto mode (also toggled by a meta-command); in confirm mode it prompts before each call, and a Ctrl-C during a live call is converted into a cancellation. Syntax highlighting runs the real lexer and classifies each `NAME` semantically rather than by capitalization, using declaration-site context (including the contextual `at`/`prio` keywords of an `infixl`/`infixr` declaration) and the live session's known type and constructor names. Multiline submission is parser-driven, with lexical EOF inside triple-quoted strings treated as continuation input until the closing delimiter is entered; while the cursor remains inside an open string, editor services suppress identifier completion and keep prefix highlighting intact. Color themes are detectable from the terminal and persisted to config. The console supplies all UI; the session itself stays UI-free.

## Code Entry Points

- `src/agm/agl/repl/` — the incremental session, type-focused display helpers, the console, agent confirmation, and themes.
- `src/agm/agl/pipeline.py` — program preparation, parameter discovery, agent reconciliation, and host-environment assembly shared by `exec` and the REPL.
- `src/agm/commands/exec.py` and `src/agm/commands/repl.py` — the hosting commands.
- `src/agm/cli_support/exec_params.py` — parameter discovery and option wiring for `agm exec`.
- Tests: `tests/test_agl_repl_session.py`, `tests/test_agl_repl_console.py`, `tests/test_agl_repl_agents.py`, `tests/test_agl_repl_themes.py`, `tests/test_agl_repl_builtin_settings.py`, `tests/test_agl_builtin_var.py`, `tests/test_agl_builtin_var_host.py`, `tests/test_exec_command.py`.
