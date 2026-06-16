# AgL implementation architecture

## Six-component pipeline

```
source (.agl)
  → [1] custom lexer  (INDENT/DEDENT, multiline strings, string interpolation)
  → [2] Lark LALR parser  (grammar in grammar/agl.lark)
  → [3] AST  (pure dataclasses, NO Lark types)   ◄── stable contract / firewall
  → [4] scope / name resolution  (full static pass)
  → [5] type checking  (full static pass; selects output contract specs)
  → host preparation  (materializes output contracts; no program execution)
  → [6] evaluator  (tree-walking interpreter)
        ↘ host runtime: agents, codecs, renderers, trace store
```

## Firewall rule

Components **1→2** are the only Lark-aware code. Component **3** (the AST in
`agm.agl.syntax`) is the *firewall*: everything from component 3 onward depends
**only** on the AST dataclasses, **never** on Lark. This is what makes the
lexer+parser replaceable (e.g. by a tree-sitter front end) without touching
scope, typecheck, or eval.

## Side-table annotation convention

Later passes (scope, typecheck) attach information to AST nodes via **side tables
keyed by the per-node `node_id`** (a monotonic integer assigned by the AST
builder). Do NOT mutate frozen AST nodes, and do NOT use `id()` hashing. The
side tables live in `ResolvedProgram` (scope pass output) and `CheckedProgram`
(typecheck pass output).

## Decimal arithmetic context

AgL semantics must not depend on the host's ambient `decimal` context. The
evaluator (`agm.agl.eval.interpreter`) runs every program under a pinned
`decimal.Context` (`_AGL_DECIMAL_CONTEXT`: 28-digit precision, `ROUND_HALF_EVEN`)
via `decimal.localcontext` in `Interpreter.execute`. A host that lowered
`getcontext().prec` would otherwise change results such as `1 / 3`.

## Incremental REPL session

`agm.agl.repl.session.ReplSession` is a UI-free incremental driver that runs the
same `parse → resolve → check → host-prep → eval` pipeline **one entry at a
time** against a *persistent* environment (session scope, type env, value scope,
declared params, source log). It reuses the firewalled passes' seam parameters:
`parse_program_seeded` (globally-unique node ids across entries),
`resolve(..., parent_scope=...)` (refs fall through to session bindings; new
decls shadow), and `check(..., seed_env=...)` (seed with prior decls/binding
types). Each entry executes **only its own statements** in a child value scope,
so agent calls fire exactly once and a later entry reads stored `Value`s rather
than re-invoking. Promotion into the session is **atomic** — a runtime raise
(`AglRaise`) OR an agent-call cancellation (`AgentCancelled` / `KeyboardInterrupt`
from the confirming wrapper) discards ALL of the entry's in-session effects via a
shared `_rollback` helper: new `let`/`var` bindings (held in the child scope) AND
any `set` mutation of a prior session binding (rolled back from a value snapshot
taken before eval, since `set` only updates an existing binding's value and never
changes the value scope's key set). Only genuinely external effects already issued
during evaluation (agent calls, `exec` shell commands) are irreversible.

**param / program:** `param` declarations are **executable**: `Interpreter._exec_param`
resolves each one in declaration order at evaluation time (no deferred "unset"
state). Resolution precedence: external value (CLI option / `[params.<program>]`
config) > default expression > pre-execution error for required params. The
`program NAME` declaration names the program for config keying — if absent, the
`.agl` file stem is used instead. `WorkflowRuntime.discover_params(prepared)`
runs typecheck on an already-`prepare`d program and returns a `ParamDiscovery`
(program name + typed `ParamDeclInfo` tuples), giving callers the full param
inventory before execution. External values are converted via `convert_input`
before execution; a conversion failure is a pre-execution error (no eval, no
agent calls).

**agm exec param wiring** (`agm.commands.exec`, helpers in
`agm.commands.param_options`): after `prepare(source)`, `discover_params` is
called once; each declared `param` becomes a first-class CLI option via
`parse_param_tokens` / `resolve_param_values` (bool params use `--name/--no-name`
flag pairs; `check_param_collisions` rejects any name that clashes with a
built-in exec flag). Config values are loaded by `load_params_config` (keyed
by `[params.<program>]`). The resolved, type-checked program flows into
`run_prepared` — the source is never parsed or typechecked again. `--input
KEY=VALUE` was removed; all param supply is through per-param options and config.

**REPL param / program (M6):** In the incremental REPL session, `param`
declarations resolve eagerly at evaluation time — same precedence as above. A
`program NAME` decl is session-global: re-entering the same name is a no-op, a
different name rejects. Config values are converted via `convert_input` in a
pre-eval check; a conversion failure rejects the entry cleanly. Atomic rollback
covers `program` + `param` promotions: a failing entry restores the prior program
name and config table. A `params_config_loader: Callable[[str], dict[str, object]]`
is injected at construction (the `agm repl` command supplies a closure over the
config context; tests supply fakes). `EchoInterpreter` inherits the base
`_exec_param` implementation which uses the pre-converted `param_values` dict
passed via constructor.

The session shares the host-environment assembly, input conversion, and
exception→`RunError` mapping with `WorkflowRuntime` via public helpers in
`agm.agl.runtime.runtime` (`assemble_host_environment`/`HostEnvironment`,
`convert_input`, `exception_value_to_run_error`); registration is delegated to an
internal `WorkflowRuntime` so reserved-name/duplicate validation is not
duplicated. `EchoInterpreter` (a thin `Interpreter` subclass) captures a trailing
bare-expression's value for echoing without re-evaluating it.

Agent calls are gated by `agm.agl.repl.agents.ConfirmingAgent`, a wrapper
`AgentFn` holding a shared mutable `AgentMode` (`confirm`/`auto`, also mutated by
the `:agent` meta-command) and an injected confirmation callback. In confirm mode
it asks before each live call; `no` raises `AgentCancelled`, `always` flips the
mode to auto. A `KeyboardInterrupt` during a live call (the runner subprocess runs
in its own process group, so on Ctrl-C the parent group-kills it in
`core/process.py` and re-raises) is converted to `AgentCancelled`. The wrapper is UI-free; the console supplies the real `[Y/n/a]`
prompt via `make_console_confirm`. Per-entry tracing: when a `trace_path` is set,
each evaluated entry opens its own `TraceStore` (a fresh `run_id`) appending JSONL
to the one file, bracketed by `run_start`/`run_end`; `check_only` writes no trace.
`exec` shell-call confirmation is out of scope (the shell path lives inside the
interpreter, not the agent registry). The `agm repl` command builds ONE
`AgentMode` and passes that same instance to both the wrapper and the console.

## Agent declarations and source↔host reconciliation

Named agents must be **declared in source** (`agent NAME [= "runner"]`). The
scope pass owns binding: it collects declarations into
`ResolvedProgram.declared_agents` (name → `AgentDecl`) and rejects any call to
an undeclared name. The **host only backs declared names** — it never owns the
name set. `WorkflowRuntime.prepare(source)` runs the lex + parse + scope phase
ONCE, returning a `PreparedProgram` (captured AST/resolution plus diagnostics
and warnings); `run_prepared` resumes from type checking on that object, and
`run(source)` is just `run_prepared(prepare(source))`. A host that needs the
declared inventory before execution (e.g. `agm exec`, to wire registrations)
calls `prepare` once and hands the same `PreparedProgram` to `run_prepared`, so
the source is never parsed or scoped twice. `declared_agents(source)` is a thin
non-raising accessor over `prepare` (returns `()` on any parse/scope error,
which `run_prepared` resurfaces) yielding `AgentDeclInfo` tuples.

`WorkflowRuntime.run_prepared` enforces the contract before execution (helper
`_reconcile_agents`), reporting all violations as error diagnostics
(`ok=False`, nothing executes): a registered name the source never declares,
and a declared name with neither a dedicated registration nor a default agent.
A declared agent backed only by the default agent is fine; a declared-but-
uncalled agent is a non-fatal scope warning.

`agm exec` (`agm.commands.exec`) wires the backings: it calls `prepare(source)`
once, then `discover_params(prepared)` to obtain the typed param inventory (used
to build per-param CLI options before execution — see *agm exec param wiring*
above), reads `PreparedProgram.declared_agents`, registers each declared name
with a runner-backed factory, then executes via `run_prepared(prepared)` (no
second parse or typecheck). The factory command is chosen by precedence (highest
to lowest): config `[exec.agents.<name>]`, the source runner hint, `--runner`,
`[exec] runner`, `[loop] runner`, built-in `claude -p`. The default runner is
always the floor, so every declared agent resolves and also backs `ask`.
Runner strings (config or source hint) share the `%%` / `%{PROMPT_FILE}`
prompt-file placeholder handling.

## Package layout and test locations

| Package | Component | Tests |
|---------|-----------|-------|
| `agm.agl.lexer` | 1 — custom lexer | `tests/test_agl_lexer.py` |
| `agm.agl.grammar` | 2 — Lark grammar | `tests/test_agl_parser.py` |
| `agm.agl.syntax` | 3 — AST dataclasses | `tests/test_agl_ast.py` |
| `agm.agl.scope` | 4 — name resolution | `tests/test_agl_scope.py` |
| `agm.agl.typecheck` | 5 — type checking | `tests/test_agl_typecheck.py` |
| `agm.agl.eval` | 6 — evaluator | `tests/test_agl_eval.py` |
| `agm.agl.runtime` | host API | `tests/test_agl_runtime.py` |
| `agm.agl.repl` | incremental REPL session (UI-free) | `tests/test_agl_repl_session.py` |
| `agm.commands.exec` | CLI command | `tests/test_exec_command.py` |

The end-to-end acceptance suite lives in `tests/test_agl_e2e.py` and
`tests/agl/`. It is **green and part of the standing gate** — `just test` /
`just check` include it with no `--ignore` flag.  All new AgL work must keep
this suite green.
