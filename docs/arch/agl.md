# AgL implementation architecture

## Six-component pipeline

```
source (.agl)
  → [1] custom lexer  (INDENT/DEDENT, multiline strings, string interpolation;
                        one case-neutral NAME token for all identifiers)
  → [2] Lark LALR parser  (grammar in grammar/agl.lark)
  → [3] AST  (pure dataclasses, NO Lark types)   ◄── stable contract / firewall
  → [4] scope / name resolution  (full static pass)
  → [5] type checking  (full static pass; selects output contract specs)
  → host preparation  (materializes output contracts; no program execution)
  → [6] evaluator  (tree-walking interpreter)
        ↘ host runtime: agents, codecs, trace store
```

## Firewall rule

Components **1→2** are the only Lark-aware code. Component **3** (the AST in
`agm.agl.syntax`) is the *firewall*: everything from component 3 onward depends
**only** on the AST dataclasses, **never** on Lark. This is what makes the
lexer+parser replaceable (e.g. by a tree-sitter front end) without touching
scope, typecheck, or eval.

The lexer emits a single `NAME` token for every identifier: capitalization is
**lexically and semantically meaningless** (the old `VAR_NAME`/`TYPE_NAME` split
is gone). No pass may branch on identifier case; types, constructors, and
variables are distinguished by their declaration/binding namespace, never by
spelling.

## AST — expression-oriented design

AgL v2 is **expression-oriented**: there is no separate statement category.
Every construct (bindings, `:=`, `print`, `if` without `else`, loops) is an
expression with a well-defined type. A block yields the value of its last item.

The unified expression nodes in `agm.agl.syntax.nodes` that replaced the former
`Stmt`/`Expr` split:

- `Block` — a sequence of items whose value is the last item.
- `If` / `Case` / `Do` / `Try` — unified nodes replacing the former
  statement/expression variants. `If` without `else` yields `unit`; `If` with
  `else` yields the common branch type.
- `Call(callee, args, named_args)` — the single call node for all invocations
  (user `def`s, built-ins `print`/`exec`/`ask`/`parse_json`, function values).
  Both the parenthesized form `f(a, b, name: v)` and the single-arg sugar `f x`
  desugar to `Call`.
- `Cast(expr, type_expr, is_test)` — the `as` / `as?` cast node. `is_test=True`
  for `as?` (yields `bool`, never raises); `is_test=False` for `as` (converts
  or raises `CastError`). The typechecker validates the source–target pair
  against the cast specification side table (`cast_specs`) and records whether
  the cast is total or fallible.
- `IndexAccess` — postfix list/dictionary indexing. The lexer emits a distinct
  adjacent-bracket token so `xs[0]` indexes while `f [0]` remains call sugar
  with a list literal argument.
- `AssignStmt` targets are either a name or an indexed mutable-root target; later
  passes require indexed assignment to resolve to a `var` list/dictionary root.
- `FuncDef` / `Lambda` / `Param` — named function declarations (top-level only)
  and anonymous function expressions.
- `UnitLit` — the `()` unit-value literal; also the empty argument list of a
  zero-argument call (unified syntactically).
- `Raise` — diverges; has the bottom type, assignable to any expected type.

Type AST nodes in `agm.agl.syntax.types` include `UnitT`, `AgentT`, and
`FuncT(params, result)` for the new v2 types. `FuncT` is purely positional;
named/default argument information lives only in `FuncDef`/`Param`, not in the
value type. Generics (rank-1 / prenex parametric polymorphism) add `AppliedT(name,
args)` — a type application `Name[args]` for user generic types and parameterized
aliases — and a `type_params: tuple[str, ...]` field on `FuncDef`/`RecordDef`/
`EnumDef`/`TypeAlias`. `Call.type_args` carries explicit `callee::[T](args)` type
arguments; it is static-only and never evaluated (erased at runtime). The former
expression-level `Constructor` node is gone: constructors are now ordinary
`VarRef`/`Call`/`FieldAccess` expressions resolved via scope side tables.

## Side-table annotation convention

Later passes (scope, typecheck) attach information to AST nodes via **side tables
keyed by the per-node `node_id`** (a monotonic integer assigned by the AST
builder). Do NOT mutate frozen AST nodes, and do NOT use `id()` hashing. The
side tables live in `ResolvedProgram` (scope pass output) and `CheckedProgram`
(typecheck pass output).

A key v2 side table: `ResolvedProgram.builtin_calls` — a `dict[int, BuiltinKind]`
mapping `Call.node_id` to `PRINT`, `EXEC`, or `ASK`. The scope pass populates
this when the callee of a `Call` node is one of the three built-in names; it does
**not** attempt to resolve the callee as an ordinary variable reference in that
case. Typecheck and eval consult this table to dispatch to the correct built-in
typing rule and evaluation path.

## Scope pass

`agm.agl.scope` runs two pre-passes before resolving expressions:

1. **Agent pre-pass** — collects `agent` declarations into
   `ResolvedProgram.declared_agents` (name → `AgentDecl`) and defines each
   as an immutable value binding of type `agent` in the root scope.
2. **`def` pre-pass** — collects all top-level `FuncDef` names into the root
   scope as value bindings (enabling **mutual recursion** — every `def` is in
   scope for every other `def` and for itself). The bodies are resolved but not
   yet evaluated.

`let`-continuation scoping replaces the former statement-sequence scoping:
a `let`/`var` binder scopes over the remaining items of the enclosing `Block`.
A block ending in a `let` with no continuation is a static error.

Built-in call classification: when the `Call.callee` is a `VarRef` whose name
is `print`, `exec`, `ask`, or `parse_json`, the resolver records the `BuiltinKind` in
`builtin_calls` and skips the ordinary variable lookup for that name.

**Constructors as value bindings**: record and enum-variant constructors are
resolved in the ordinary value namespace, not a separate one. A pre-pass collects
candidates from every `RecordDef`/`EnumDef` (plus seeded built-in/prelude
constructors) into `ResolvedProgram.constructor_candidates` (name → ordered
`ConstructorRef` tuple). A single candidate resolves to a `ConstructorRef` in the
`constructor_refs` side table (keyed by the `VarRef`/`Call` node); two or more
candidates from distinct owners form an **overload set**, and an unqualified
reference to an ambiguous name is a scope error — type-qualification
(`Owner.variant`) disambiguates and is recorded in `qualified_constructor_refs`
(`FieldAccess` node → `(owner, member)`). `ConstructorRef` carries the owner's
`type_params` so later passes can instantiate generic constructors.

## Type system

`agm.agl.typecheck` adds these semantic types to the v2 system:

- **`UnitType`** — the type of side-effecting expressions that produce no
  meaningful value (`print`, `:=`, `if` without `else`, `do … until`). Its
  single value is `()`.
- **`FunctionType(params, result)`** — purely positional; named/default argument
  information is erased from the value type. Assignability is exact structural
  match.
- **`AgentType`** — opaque; no fields, no equality, no rendering, not
  JSON-shaped.
- **`BottomType`** — the type of `raise`; assignable to any expected type.
- **`TypeVarType(name)`** — a rigid type variable bound by an enclosing generic
  declaration. It is treated as **opaque** by the capability gates
  (`is_json_shaped`, `comparable_types`, `is_assignable`): a bare type variable is
  not JSON-shaped, not comparable, and assignable only to an identical type
  variable. This enforces strict parametricity (D2) — a generic body may not
  inspect or operate on values of a type-variable type.

`RecordType`/`EnumType` carry a `type_args` tuple and have **nominal identity by
name + `type_args`** (fields/variants are excluded from equality and hashing). The
module also exposes the substitution machinery `free_type_vars` / `substitute` /
`contains_type_var`.

**Generic declarations and instantiation** (`agm.agl.typecheck.env`): generic
records/enums are stored as `GenericTypeDef` templates (kind + `type_params` +
a template `RecordType`/`EnumType` whose fields/variants contain `TypeVarType`s).
`TypeEnvironment.instantiate_nominal(name, args)` performs **eager,
non-recursive** substitution to produce a concrete type with `type_args` set.
Parameterized aliases keep their `type_params` and are substituted on resolution.
`FunctionSignature.type_params` records a generic `def`'s parameters; a
`ConstructorSignature` (owner/variant, ordered field names + field templates,
result template, `type_params`) describes each constructor for instantiation.
`resolve_type_expr(..., type_vars=…)` is type-var-aware: it resolves `AppliedT`
(checking arity and substituting), turns in-scope `NameT`s into `TypeVarType`s,
and **rejects a bare generic nominal/alias name** used without arguments.

**Checking generics** (`agm.agl.typecheck.checker`): a generic `def` is checked
with its type parameters in scope as rigid variables (including inside nested
lambda/`let` annotations). A small **one-sided matching/inference solver**
(`_match` / `_match_unsolved`) binds template type variables to concrete argument
types — `_match` reports inconsistent bindings, `_match_unsolved` fills remaining
holes from the expected type — driving both the explicit `::[…]` path and pure
inference. Type-argument matching is **invariant** (D6): a generic nominal matches
only same-name, same-arity, position-wise. Generic record/enum construction,
field access, and `case`/`is` patterns (including qualified patterns) instantiate
the relevant signature. A generic `def` or constructor used as a *value* is
instantiated from the expected type (D5/D7). Agent/`exec`/`ask`-request targets
may not contain a type variable (D3).

**Erasure rationale**: type arguments exist **only during type checking**. They
are never represented at runtime — generic `def`s erase to ordinary closures and
`Call.type_args` is never evaluated (see *Evaluator*).

Built-in typing rules (in `agm.agl.typecheck.checker`) consult `builtin_calls`:

- **`PRINT`** — any-to-`unit` rule: accepts one argument of any renderable type;
  yields `unit`. Rejecting a function or agent value is also done here (D9).
- **`ASK`** and **`EXEC`** — reuse the existing target-type propagation and
  `OutputContractSpec` machinery. `ask` takes its result type from the expected
  type in context (defaulting to `text`). `exec` adds the `ExecResult`
  special-case (D10): when the target type is `ExecResult` (the default when no
  expected type exists), the checker sets `OutputContractSpec.structured_exec =
  True`; otherwise the parsed form is selected and stdout is parsed into the
  target type.
- **`PARSE_JSON`** — single-argument `text → json` rule; always strict parsing;
  raises `JsonParseError` on malformed input.

**Cast type checking.** `Cast` nodes are checked against a `cast_specs` side
table in `agm.agl.typecheck.casts` that encodes the permitted source–target
pairs and their total/fallible classification. Invalid pairs (e.g. `record as
json`, `bool as int`, casts to/from `unit`/`agent`/function types) are static
errors. `as?` nodes are always typed as `bool` regardless of source/target.

The prelude types `ExecResult` (a record with `stdout`, `stderr`, `exit_code`,
`timed_out`) and `ParsePolicy` (enum `Abort | Retry(n: int)`) are registered as
built-in types available without user declarations. Runtime failures such as
`RecursionError`, `IndexError`, and `KeyError` are built-in catchable
exceptions.

Function and agent types are **not JSON-shaped**: the codec-selection and
`is_json_shaped` logic rejects them; interpolating or `print`-ing a function
or agent value is a static error.

## Decimal arithmetic context

AgL semantics must not depend on the host's ambient `decimal` context. The
evaluator (`agm.agl.eval.interpreter`) runs every program under a pinned
`decimal.Context` (`_AGL_DECIMAL_CONTEXT`: 28-digit precision, `ROUND_HALF_EVEN`)
via `decimal.localcontext` in `Interpreter.execute`. A host that lowered
`getcontext().prec` would otherwise change results such as `1 / 3`.

## Evaluator

`agm.agl.eval` introduces three new value kinds in `agm.agl.eval.values`:

- **`Closure`** — a captured definition environment, parameter list (with
  resolved default expressions), and body expression. Top-level `def`s are
  installed as `Closure` values during the evaluator's root pre-pass (enabling
  mutual recursion without a separate linking step).
- **`UnitValue`** — the single value of type `unit`; a module-level singleton
  `UNIT_VALUE` is reused everywhere.
- **`AgentValue`** — an opaque handle carrying the declared agent name; resolved
  against the host agent registry at call time.
- **`ConstructorValue`** — a first-class constructor used as a value, carrying
  only owner/variant identity (no type args — erased). Calling it builds the
  record/enum from **positional** arguments in declaration order, sourcing field
  names/types from the type environment (`GenericTypeDef` template or concrete
  type) rather than the call-site result type, which may be an erased type
  variable when the constructor escapes through a higher-order function.

**Type erasure**: generics carry no runtime representation. Generic `def`s are
ordinary `Closure`s and `Call.type_args` is never evaluated.

All calls go through the unified call dispatch in `Interpreter._eval_call`:

1. Check `builtin_calls` for `PRINT`/`ASK`/`EXEC` and dispatch to the
   appropriate built-in handler.
2. Otherwise evaluate the callee to a `Closure`, bind positional and
   named/defaulted arguments (defaults evaluated in the closure's captured
   scope), open a call scope, and evaluate the body.
3. Before entering a call frame, enforce the **call-depth limit** (default 256,
   configurable via `max_call_depth`). Exceeding it raises the new
   `RecursionError` exception value — distinct from `MaxIterationsExceeded`
   (loop-specific) and catchable with `try`/`catch`.

`exec`'s two evaluation paths are selected by `OutputContractSpec.structured_exec`:
- **Structured form** — returns an `ExecResult` record built from the raw
  subprocess output; a nonzero exit does NOT raise.
- **Parsed form** — parses stdout into the target type via the codec pipeline,
  raises `ExecError` on nonzero exit or parse failure; mirrors the pre-v2
  behavior.

**Cast and `parse_json` evaluation** share a strict-parse/validate conversion
helper in `agm.agl.runtime.convert` (`convert_value`). This helper handles
strict JSON parsing (no lenient recovery), schema validation, and the full
source→target conversion matrix. Both `as` casts on `text`/`json` sources and
the `parse_json` built-in call into this helper — they always use strict
parsing. Agent-output and `exec`-output parsing continues to use the
existing configurable strict/lenient codec pipeline and is not affected.

Agent-value dispatch: `_eval_ask_call` extracts the `AgentValue` from the
`agent:` named argument (or uses the default agent when absent) and issues the
call via the host runtime, exactly as the former `AgentCall` node did.

## Incremental REPL session

`agm.agl.repl.session.ReplSession` is a UI-free incremental driver that runs the
same `parse → resolve → check → host-prep → eval` pipeline **one entry at a
time** against a *persistent* environment (session scope, type env, value scope,
declared params, source log). It reuses the firewalled passes' seam parameters:
`parse_program_seeded` (globally-unique node ids across entries),
`resolve(..., parent_scope=...)` (refs fall through to session bindings; new
decls shadow), and `check(..., seed_env=...)` (seed with prior decls/binding
types). Each entry executes **only its own expressions** in a child value scope,
so agent calls fire exactly once and a later entry reads stored `Value`s rather
than re-invoking. Promotion into the session is **atomic** — a runtime raise
(`AglRaise`) OR an agent-call cancellation (`AgentCancelled` / `KeyboardInterrupt`
from the confirming wrapper) discards ALL of the entry's in-session effects via a
shared `_rollback` helper: new `let`/`var` bindings (held in the child scope) AND
any `:=` mutation of a prior session binding (rolled back from a value snapshot
taken before eval, since `:=` only updates an existing binding's value and never
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
inventory before execution. External values are converted via `convert_param_value`
before execution; a conversion failure is a pre-execution error (no eval, no
agent calls).

**agm exec param wiring** (`agm.commands.exec`, helpers in
`agm.cli_support.exec_params`): after `prepare(source)`, `discover_params` is
called once; each declared `param` becomes a first-class CLI option via
`parse_param_tokens` / `resolve_param_values` (bool params use `--name/--no-name`
flag pairs; `check_param_collisions` rejects any name that clashes with a
built-in exec flag). Config values are loaded by `load_params_config` (keyed
by `[params.<program>]`). The resolved, type-checked program flows into
`run_prepared` — the source is never parsed or typechecked again. Param supply
is through per-param options and config.

**REPL param / program (M6):** In the incremental REPL session, `param`
declarations resolve eagerly at evaluation time — same precedence as above. A
`program NAME` decl is session-global: re-entering the same name is a no-op, a
different name rejects. Config values are converted via `convert_param_value` in a
pre-eval check; a conversion failure rejects the entry cleanly. Atomic rollback
covers `program` + `param` promotions: a failing entry restores the prior program
name and config table. A `params_config_loader: Callable[[str], dict[str, object]]`
is injected at construction (the `agm repl` command supplies a closure over the
config context; tests supply fakes). `EchoInterpreter` inherits the base
`_exec_param` implementation which uses the pre-converted `param_values` dict
passed via constructor.

The session shares the host-environment assembly, param conversion, and
exception→`RunError` mapping with `WorkflowRuntime` via public helpers in
`agm.agl.runtime.runtime` (`assemble_host_environment`/`HostEnvironment`,
`convert_param_value`, `exception_value_to_run_error`); registration is delegated to an
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

The console's `AglPromptLexer` (`agm.agl.repl.console`) highlights by running the
real lexer; since identifier case is meaningless, it classifies a `NAME`
semantically rather than by capitalization. A lexical pass over the buffer
(`_decl_site_styles`) styles declaration sites positionally — the name after
`record`/`enum`/`type` is a type, an enum variant after `|` is a constructor —
and collects those names so their references in the same buffer colour too; this
makes the in-progress declaration colour and disambiguates a type from a
like-named constructor. Every other `NAME` is classified by name against
`syntax.BUILTIN_TYPE_NAMES`, the buffer-local names, and the live
`ReplSession.type_names()` / `constructor_names()`, with a one-token look-ahead
so a constructor call (`Box(…)` / `Box::[…]`) colours as a constructor while a
bare/annotation use colours as a type. Without a session it still colours
keywords, literals, operators, builtins, and the buffer's own declarations.

Colour themes live in `agm.agl.repl.themes`: `DARK_THEME` (VS Code Dark+) and
`LIGHT_THEME` (VS Code Light+). `detect_terminal_theme` reads `$COLORFGBG`
(set by most terminal emulators; trailing segment `15` → light, anything else
or absent → dark). `get_style(theme)` resolves `"auto"` via detection and
returns the appropriate `prompt_toolkit` `Style`. The theme is wired through
`build_prompt_session(theme=…)` and stored on `MetaContext.theme`; `run_console`
observes mutations from the `:theme` meta-command and updates `PromptSession.style`
live, then calls the `on_theme_save` callback (supplied by `commands/repl.py`
via `config.general.save_repl_theme`) to persist the choice to `~/.agm/config.toml`
under `[repl] theme`. Loading uses `load_repl_config` (same merge chain as other
config sections).

## Config pragma pipeline

`config KEY = VALUE` header pragmas are grammatically `ConfigPragma` AST nodes.
The scope pass enforces header-only placement (error if a pragma follows any
non-pragma root item, or appears in a nested block), validates each key and value
kind, and collects the validated set into `ResolvedProgram.config_pragmas`.
Typecheck and eval treat `ConfigPragma` as a no-op. `PreparedProgram.config_pragmas`
exposes the collected map (empty on parse/scope failure) for the host to read.

`agm exec` reads `prepared.config_pragmas` after `WorkflowRuntime.prepare(source)`
and applies each pragma with **CLI > pragma > config-file** precedence: CLI flags
win, then pragma values, then `[exec]` config. Trace logging is off by default;
`config log = true` (or `--log` / `[exec] log = true`) opts in.

`agm repl` (`ReplSession.eval_entry`) rejects `ConfigPragma` entries after parse
with a clear diagnostic. Config pragmas are an exec/program feature; REPL session
options come from CLI flags and config files, not source pragmas.

## Agent declarations and source↔host reconciliation

Named agents must be **declared in source** (`agent NAME [= "runner"]`). The
scope pass owns binding: it collects declarations into
`ResolvedProgram.declared_agents` (name → `AgentDecl`) as part of the agent
pre-pass, and simultaneously defines each declared name as an immutable value
binding of type `agent` in the root scope — agents are now first-class values,
not a separate namespace. The **host only backs declared names** — it never owns
the name set. `WorkflowRuntime.prepare(source)` runs the lex + parse + scope
phase ONCE, returning a `PreparedProgram` (captured AST/resolution plus
diagnostics and warnings); `run_prepared` resumes from type checking on that
object, and `run(source)` is just `run_prepared(prepare(source))`. A host that
needs the declared inventory before execution (e.g. `agm exec`, to wire
registrations) calls `prepare` once and hands the same `PreparedProgram` to
`run_prepared`, so the source is never parsed or scoped twice.
`declared_agents(source)` is a thin non-raising accessor over `prepare` (returns
`()` on any parse/scope error, which `run_prepared` resurfaces) yielding
`AgentDeclInfo` tuples.

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
