# AgL REPL and Program Hosting

Two commands host AgL programs: `agm exec` runs a whole program, and `agm repl` evaluates entries one at a time. Both run the same pipeline ([index.md](agl/index.md)); they differ in how the program is supplied and how parameters, agents, and configuration are wired around it. This document covers that host-facing surface.

## Incremental REPL Session

The REPL is a UI-free incremental driver that runs the full pipeline one entry at a time against two persistent images: the static session environment and the linked IR/runtime image. Node ids stay globally unique across entries, new references fall through to prior session bindings, and new declarations shadow. Each entry contributes a delta of initializers executed against one persistent base frame, so earlier initializers and host calls are never replayed.

Every entry is an inference boundary: its checker regions close before promotion, and a failed entry promotes nothing, so later entries start clean. Runtime failure is deliberately non-transactional — initializers completed before a failure remain installed, and the static environment advances only for the symbols that actually reached the runtime frame. Redefining a nominal type invalidates prior bindings whose static type mentions it. Match compilation runs before the check-only early return, so non-exhaustive cases cannot bypass static validation.

The REPL uses the program pipeline ([modules.md](agl/modules.md)) by default so it gets the same automatic `std/core` open import as `agm exec`, in each entry and its loaded library modules; `agm repl --no-stdlib` disables that injection throughout every loaded REPL program, including after `:reset`. Library modules are cached and incrementally linked. Imports persist by resolved module identity: wildcards expand before retention, and a later entry replaces every prior declaration for each module it names while declarations for one module in the new entry retain normal batch union semantics. This keeps selected-set, suffix, anchored, alias, open, `using`, and `hiding` behavior aligned with batch resolution. User `infixl`/`infixr` fixity also persists by accumulation and replay. Type-focused surfaces (`:type`, check-only echoes) use a REPL-specific type formatter that expands records and enums to declaration-like listings while keeping ordinary echoes compact.

## Program Parameters

`param` declarations make a program parameterizable. The pipeline discovers the typed parameter inventory before execution, which `agm exec` uses to expose each parameter as a first-class CLI option and to load values from the `[<program>]` config section. Resolution precedence is external value > default expression > error for a required param, and external values are converted and type-checked before any evaluation, so a bad value fails before any agent or shell call runs. The REPL applies the same precedence.

## Agent Declaration and Reconciliation

Agents must be declared in source; the host backs declared names but never owns the name set. The pipeline prepares a program once (lex, parse, scope) and reuses that prepared object for both parameter discovery and execution. Before execution it reconciles declared agents against the host's registrations: a registration with no matching declaration, or a declared agent with no backing, is an error; a declared-but-uncalled agent is a warning. `agm exec` registers each declared name with a runner-backed factory, choosing the runner command by precedence across config, a source runner hint, CLI flags, and the built-in default floor (see [agents.md](agents.md)).

## Engine Settings

The engine settings a program can tune — `runner`, `log`, `log-file`, `strict-json`, `max-iters`, `timeout` — are `builtin var` bindings declared in the standard-library module `std/config`; `src/agm/config/engine_keys.py` is the shared name/kind catalog. A `builtin var` is a body-less, runtime-backed mutable binding: the static passes treat it like any binding, while the lowerer routes reads and writes to dedicated IR operations keyed by setting name. The interpreter backs runtime-live settings (`strict-json`, `max-iters`, `timeout`) with live interpreter state, and host-consumed settings (`runner`, `log`, `log-file`) with registers that reconfigure the live host service on write.

Because a write is an ordinary positional statement, settings take effect in program order and `agm exec` runs the program in a single phase — there is no separate startup pass. The CLI and config-file layers seed the initial values ([config.md](config.md)); a source write overrides them from its point onward. In the REPL a completed write persists across entries, matching the non-transactional runtime; `:reset` restores the seeded defaults.

## Console and Confirmation

The REPL console adds interactivity around the UI-free session: a confirmation wrapper gating live agent calls (confirm/auto modes, Ctrl-C converted into cancellation), syntax highlighting that runs the real lexer and classifies names semantically from declaration context rather than capitalization, parser-driven multiline submission that treats an unterminated string as continuation input, and terminal-detected color themes persisted to config.

## Code Entry Points

- `src/agm/agl/repl/` — the incremental session, type-focused display helpers, the console, agent confirmation, and themes.
- `src/agm/agl/pipeline.py` — program preparation, parameter discovery, agent reconciliation, and host-environment assembly shared by `exec` and the REPL.
- `src/agm/commands/exec.py` and `src/agm/commands/repl.py` — the hosting commands; `src/agm/cli_support/exec_params.py` — parameter discovery and option wiring for `agm exec`.
- Tests: `tests/test_agl_repl_*.py`, `tests/test_agl_builtin_var.py`, `tests/test_agl_builtin_var_host.py`, `tests/test_exec_command.py`.
