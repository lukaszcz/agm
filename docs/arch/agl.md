# AgL implementation architecture

## Production pipeline

```
source (.agl)
  → [1] custom lexer  (INDENT/DEDENT, multiline strings, string interpolation;
                        one case-neutral NAME token for all identifiers)
  → [2] Lark LALR parser  (grammar in grammar/agl.lark)
  → [3] AST  (pure dataclasses, NO Lark types)   ◄── stable contract / firewall
  → [4] scope / name resolution  (full static pass)
  → [5] type checking  (full static pass; selects concrete operations/contracts)
  → [6] lowering + linking  (closed, typeless ExecutableProgram)
  → [7] IR evaluation  (IrInterpreter)
        ↘ host runtime: agents, shell execution, codecs, trace store
```

`WorkflowRuntime` and the REPL always execute this pipeline. Linked IR is the only
execution format; checked frontend objects are never evaluator inputs.

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
value type. Generics (rank-1 / prenex parametric polymorphism) add
`AppliedT(name, args, module_qualifier)` — an optionally qualified type
application for user generic types and parameterized aliases — and a
`type_params: tuple[str, ...]` field on `FuncDef`/`RecordDef`/`EnumDef`/
`TypeAlias`. `Call.type_args` carries explicit postfix `callee::[T](args)` type
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

**Bare names in patterns**: a bare name in a `case` pattern is a constructor
pattern when it denotes an in-scope constructor (lexical or open-imported, and
not shadowed by a nearer ordinary binding), otherwise a variable binder. The
scope pass makes this resolution-directed (never spelling-directed) call in
`_bind_pattern_vars` and records the constructor-pattern node ids in
`ResolvedProgram.bare_variant_patterns`; it binds no variable for those. The
checker validates each as a *nullary* variant of the scrutinee enum (a
field-bearing variant requires an explicit call form), and the interpreter
matches it by variant name without binding. The graph eval merge unions every
module's `bare_variant_patterns`.

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
pairs and their total/fallible classification. Invalid pairs (e.g. `bool as int`,
casts to/from `unit`/`agent`/function types) are static errors. Nominal types
(record, enum, exception) `as json` are permitted as a `TOTAL_JSON` explicit cast
(structural encoding); `exception as text` is permitted as a `TOTAL_RENDER` cast.
Implicit assignability to `json` is unchanged — nominal values are still not
JSON-shaped. `as?` nodes are always typed as `bool` regardless of source/target.

The prelude types `ExecResult` (a record with `stdout`, `stderr`, `exit_code`,
`timed_out`) and `ParsePolicy` (enum `Abort | Retry(n: int)`) are registered as
built-in types available without user declarations. Runtime failures such as
`RecursionError`, `IndexError`, and `KeyError` are built-in catchable
exceptions.

Function and agent types are **not JSON-shaped**: the codec-selection and
`is_json_shaped` logic rejects them; interpolating or `print`-ing a function
or agent value is a static error.

## Decimal arithmetic context

AgL semantics must not depend on the host's ambient `decimal` context. Both
evaluators run every program under a pinned `decimal.Context` (28-digit
precision, `ROUND_HALF_EVEN`) via `decimal.localcontext`. A host that lowered
`getcontext().prec` would otherwise change results such as `1 / 3`.

## Execution IR and evaluator

`agm.agl.lower` consumes checked AST graphs and emits one linked
`ExecutableProgram`. `lowerer.py` performs expected-type-directed expression
lowering; `graph.py` allocates shared identities and links modules in dependency
order. Top-level function closures are initialized before ordinary module
initializers, matching declaration hoisting and enabling forward references.

`agm.agl.ir` is the runtime-neutral data model:

- `ids.py` defines program-local symbol, function, contract, source, and nominal
  identities plus source locations.
- `nodes.py` defines the closed expression families: constants and construction,
  binding/load/assignment, arithmetic and comparison, control flow and matching,
  closures/calls, conversion, and host operations.
- `program.py` holds modules, symbols, functions, parameters, sources, nominals,
  contracts, and dry-run inventory.
- `contracts.py` holds typeless `ContractRequest`, `ParamDecoder`, decode-schema,
  and conversion descriptors compiled while checker types are available.
- `validate.py` is the single structural gate. It runs only when explicitly
  requested at a lowering boundary; production evaluation does not invoke it implicitly.

`agm.agl.eval.ir_interpreter.IrInterpreter` executes only those closed descriptors.
Its frame stack uses values for immutable bindings and shared `Cell`s for mutable
bindings. The base frame is module scope: module-owned lets, vars, params, functions,
and agents are resolved there rather than captured. Function frames contain
parameters and captures of function-owned lexical bindings; assignment falls back
to the base frame for module vars. Generic type arguments are erased, while nominal
identity remains module-qualified.

Host calls are keyed by `ContractId`. `ContractRequest` carries codec selection,
format instructions, canonical JSON schema, and a typeless decode walk; it is also
converted to the agent-facing contract carrier so model-backed agents receive
format/schema metadata. `IrParam.external_decoder` performs host parameter decoding
without consulting checker binding types. Dry-run output is copied from the linked
program's inventory rather than reconstructed from checker call-site tables.

The package boundaries are enforced by `tests/test_agl_dependencies.py`:

- `agm.agl.ir` imports only its own data modules and `agm.agl.modules.ids`.
- `agm.agl.lower` may import syntax, scope, typecheck, IR, and the neutral
  compile-time schema helper, but never evaluator or runtime execution modules.
- `agm.agl.eval` may import IR, runtime services, runtime values, and module IDs,
  but never syntax, scope, typecheck, or REPL modules.

`agm.agl.type_schema` owns pure checker-type-to-JSON-schema compilation and format
instruction generation. This keeps lowering independent of runtime execution.

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

Each checked module retains normalized source text. Lowering copies it into the
linked source table, and runtime diagnostics select the correct text through the
`SourceId` carried by each IR location. Module execution performs no checker-table merge.

## Value rendering

All value display (template/`${…}` interpolation, `print`, `as text`, and REPL
echo) goes through a single recursive renderer in `agm.agl.runtime.render`
(`render_value` / `render_value_repl`). The renderer produces **AgL-native
syntax** for every value kind:

- Scalars (`int`, `decimal`, `bool`) — plain text.
- `text` — verbatim at top level (interpolation), quoted AgL string literal
  when nested or in REPL echo; dollar signs are escaped as `\$`.
- `json` — pretty-printed (2-space indent) at top level, compact single-line
  when nested inside another structured value.
- `list` / `dict` — AgL container syntax (`[…]` / `{"k": v, …}`); dict keys
  always quoted.
- Record / enum / exception — AgL constructor syntax
  (`TypeName{f: v, …}` / `TypeName.Variant(…)` / `TypeName.Variant`); the
  renderer walks `value.fields` verbatim.

Nominal fields are kept in **declaration order** by the value itself: the
interpreter normalizes them once, at construction (`_eval_constructor`), against
the declared type — so the renderer needs no type information and there is a
single canonical field order for every consumer (native render, `as json`,
equality). The renderer depends only on the `Value` union — no semantic types,
no Lark types, no lexer rules.

`as json` on a nominal type (`TOTAL_JSON` cast) uses `value_to_json_obj`
(structural encoding), which walks the same declaration-ordered `value.fields`,
so native and JSON output agree on field order.

## Incremental REPL session

`agm.agl.repl.session.ReplSession` is a UI-free incremental driver that runs the
same `parse → resolve → check → lower → IR eval` pipeline **one entry at a
time** against two persistent images: the static session environment and the
linked IR/runtime image. It reuses the firewalled passes' seam parameters:
`parse_program_seeded` (globally-unique node ids across entries),
`resolve(..., parent_scope=...)` (refs fall through to session bindings; new
decls shadow), and `check(..., seed_env=...)` (seed with prior decls/binding
types). `LinkImage` retains monotonically allocated symbol, function, source, and
contract IDs plus every linked descriptor needed by retained closures. Each entry
adds a delta of initializers and executes it against one persistent IR base frame,
so earlier initializers and host calls are never replayed. Lowering records the
trailing-expression initializer explicitly; the session echoes that initializer's
single execution result.

Runtime failure is deliberately **non-transactional**. Initializers completed
before an `AglRaise` or host cancellation remain installed, including mutations
of prior vars and new bindings. The static environment advances only for symbols
present in the runtime frame, keeping later name resolution aligned with the
partially advanced runtime image. Unreached initializers do not run.
Failed `EntryResult`s list the names installed before failure, and the console
prints that list after the error.

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
pre-eval check; a conversion failure rejects the entry cleanly. A
`params_config_loader: Callable[[str], dict[str, object]]`
is injected at construction (the `agm repl` command supplies a closure over the
config context; tests supply fakes). IR param descriptors map those pre-converted
values to their persistent `SymbolId`s before execution.

The session shares the host-environment assembly, param conversion, and
exception→`RunError` mapping with `WorkflowRuntime` via public helpers in
`agm.agl.runtime.runtime` (`assemble_host_environment`/`HostEnvironment`,
`convert_param_value`, `exception_value_to_run_error`); registration is delegated to an
internal `WorkflowRuntime` so reserved-name/duplicate validation is not
duplicated. REPL entries execute through `IrInterpreter` against a persistent linked image.

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

**REPL module import support (M6):** `ReplSession` supports `import` declarations
at the top of any entry.  When an entry has at least one `ImportDecl` OR there
are cached library modules from a prior entry, `eval_entry` dispatches to
`_eval_entry_graph_mode` which runs the full multi-module graph pipeline:
`build_repl_graph` (builds a `ModuleGraph` from the already-parsed entry program,
loading only new library modules not in the cache), `resolve_graph`, `check_graph`,
then incrementally links the checked graph into the persistent `LinkImage` and
executes its new module/entry initializers with `IrInterpreter`. Library source
loads are cached in `_loaded_lib_modules`; linked descriptors and initialized
module values remain in the IR image/base frame.

**Open-import persistence:** When an entry's `import foo` uses open-import
semantics (no `qualified` keyword), `foo`'s exported names enter the entry's
unqualified scope.  To make these names persist across entries, `ReplSession`
accumulates the `ImportDecl` nodes from each successfully-promoted graph-mode entry
in `_accumulated_imports`.  On the next graph-mode entry, `_inject_accumulated_imports`
prepends the accumulated import declarations to the new entry's program before
building the graph, so prior open-imported names remain in scope.  Re-importing
the same module in a later entry replaces the earlier accumulated import
declaration (deduplication by `(module_path, wildcard)`).

**Root set:** Module search roots are assembled lazily on first import use by
`_ensure_roots` using `assemble_roots`.  The `agm repl` command resolves
`lib_root` from the `[modules] lib_root` config key (or defaults to `~/.agm/lib`)
and passes it to `ReplSession(cwd=..., lib_root=..., configured_roots=...)`.

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

## Source-aware spans

Every `SourceSpan` carries a `SourceId` (a frozen dataclass with a `label: str`)
in its `source` field (default: `UNKNOWN_SOURCE = SourceId("<agl>")`). The field
is `compare=False` so spans from different files with identical positions compare
equal — consistent with how `node_id` is excluded from AST-node equality.

`parse_program` and `parse_program_seeded` accept an optional `source: SourceId`
parameter. When supplied, every span the `AstBuilder` constructs — and the span
of any `AglSyntaxError` raised during parsing — is stamped with that `SourceId`.
The module loader (M2 Task C) passes `SourceId(label=str(canonical_path))` here so
multi-file diagnostics identify the origin file.

`Diagnostic` has an optional `source_label: str | None` field. `diagnostic_from_span`
populates it from `span.source.label` for any non-default `SourceId`; for
`UNKNOWN_SOURCE` spans it leaves `source_label` as `None`, preserving backward
compatibility — existing callers that pass `source_name=` to `format_diagnostic`
continue to see their supplied label.

## Graph-aware scope resolution (`agm.agl.scope.graph`)

`resolve_graph(graph)` runs the scope pass over an entire `ModuleGraph` and
produces a `ResolvedModuleGraph`.  The pipeline has five steps:

1. **Export sets** — `_compute_exports` collects non-private top-level
   `FuncDef`/`RecordDef`/`EnumDef`/`TypeAlias` names for each module.
2. **ImportTarget mapping** — each `ImportDecl` node is mapped to either a
   `SingleTarget` (concrete module) or `WildcardTarget` (set of matched modules).
3. **ImportEnv per module** — `build_import_env` uses the targets and export sets
   to build an `ImportEnv` with `unqualified` (bare name → candidate `QName`
   set) and `qualified` (handle → name → `QName`) maps.
4. **Whole-graph pre-pass** — all public `FuncDef` and type declarations across
   all modules are collected into `decl_info` (node-id/span/binder-kind per
   `(ModuleId, name)`) and `private_info` (marks private names) BEFORE any body
   is resolved.  This enables cross-module mutual recursion.
5. **Per-module resolution** — `_Resolver` is instantiated with `module_id`,
   `import_env`, `decl_info`, `private_info`, and `is_entry`.  Graph-mode
   enriches `_resolve_varref`: qualified refs (`handle::name` or `::name`
   self-refs) are dispatched to dedicated helpers; unqualified refs not found
   in the lexical scope fall back to `import_env.unqualified` lookup with
   clash-on-use enforcement.  A clash (>1 candidate `QName`) is an error at the
   reference site.

`BindingRef` carries a `module_id` field (defaults to `ENTRY_ID`) so
downstream passes can identify which module owns any binding.

**Enforcement** in graph mode: non-entry modules reject config pragmas,
`let`/`var` binders, `agent`/`param`/`program` declarations, assignment
statements, and bare expressions.  Import declarations in non-entry modules must
appear before all other items (header-only).

Single-module programs continue to use `resolve()` (unchanged); they bypass the
graph machinery entirely.

## Graph-aware evaluation (`agm.agl.eval`)

`lower_graph` links a `CheckedModuleGraph` into one `ExecutableProgram`. Modules,
exports, functions, nominals, contracts, and sources are resolved to stable IDs
before `IrInterpreter` starts. Library initializers run in dependency order and
the entry module runs last. Cross-module loads and calls use linked IDs directly;
the evaluator never reads or merges resolver/typechecker side tables.

## Graph-aware type checking (`agm.agl.typecheck.graph`)

`check_graph(resolved_graph, capabilities) → CheckedModuleGraph` extends the type
checker to operate over a full `ResolvedModuleGraph`.

**Module-qualified type identity.** `RecordType` and `EnumType` now carry a
`module_id: ModuleId` field (defaulting to `ENTRY_ID`).  Two types with the same
name but different `module_id`s are distinct types — `foo::Color ≠ bar::Color` even
when structurally identical.

**Graph type pre-pass.** Before any function body is checked, `_build_graph_type_table`
runs a genuinely whole-graph two-phase pass:

1. **Phase 1 (shells)** — ALL type shells for ALL modules are registered first:
   empty `RecordType`/`EnumType` shells stamped with the owning `module_id` go
   into `graph_type_table`; type aliases are tracked in per-module envs.  All
   shells are registered before any body is resolved, so forward references across
   modules work even when the import graph has cycles.
2. **Phase 2 (bodies in topological order)** — the structural type-definition
   dependency graph is computed across all modules (a record/enum/alias depends on
   every type named in its field/variant/alias-target expressions, cross-module
   included), then Kahn's algorithm produces a topological order (ties broken by
   `(ModuleId.segments, name)` for determinism).  Each body is resolved in that
   order so the referenced type is always fully built before it is captured
   by-value as a field/variant/element type.  A genuine structural type cycle (a
   type that contains itself infinitely) is an `AglTypeError` consistent with the
   single-module `_TypeBuilder` behaviour.  Import-graph cycles (D8) are allowed
   and do not imply structural type cycles.  The result is stored in the shared
   `graph_type_table: dict[(ModuleId, name), Type]`.

**Cross-module mismatch diagnostics.** `RecordType.__repr__`/`EnumType.__repr__`
qualify the type name with its owning module when the module is NOT `ENTRY_ID`
(e.g. `foo::Color`), so mismatch messages distinguish `foo::Color` from
`bar::Color` rather than rendering both as `Color`.

**Graph function-signature pre-pass.** Before any function body is checked,
`_build_graph_func_sig_table` resolves the parameter and return type annotations
for every top-level `FuncDef` in every module (using `graph_type_table` and each
module's `ImportEnv`), producing a table keyed by the globally-unique
`FuncDef.node_id`.  No function body is checked in this phase.

**Per-module checking.** `_check_module` creates a `TypeEnvironment` with the
graph type table and `ImportEnv`, seeds it with the module's own fully-resolved
types by bare name, seeds ALL function binding types from the whole-graph
function-signature pre-pass (enabling cross-file mutual recursion — D8/§8.2 —
regardless of per-module checking order), and then runs the existing
`_TypeBuilder + _Checker` pipeline.  Non-entry modules are checked before the
entry module for determinism; function-signature availability no longer depends
on this order.

**Module-aware type resolution.** `TypeEnvironment` gains graph-mode parameters
(`graph_type_table`, `import_env`, `module_id`).  Resolution in graph mode:
- A qualified `MODQUAL::Name` type ref looks up the handle in `import_env.qualified`
  and then fetches the type from `graph_type_table`.
- An empty-segment self-ref `::Name` looks up `(module_id, name)` in
  `graph_type_table`, falling back to the local env for built-ins.
- An unqualified `Name` in graph mode searches `import_env.unqualified`; if exactly
  one module exports that type name, it is resolved; multiple exports is an ambiguity
  error.
- Applied generic types use the same qualified/open-import lookup, backed by the
  graph generic-definition table. That table is also supplied to the whole-graph
  function-signature prepass so imported applications resolve in parameter and
  return annotations.

**Outputs.** `CheckedModuleGraph` holds a `dict[ModuleId, CheckedModule]`, the
shared `graph_type_table`, the `entry_id`, and aggregated warnings.  Each
`CheckedModule` mirrors the single-module `CheckedProgram` shape.

Single-module programs continue to use `check()` (unchanged); `check_graph` on a
single-entry-only graph is equivalent.

## Module-graph loading (`agm.agl.modules`)

The `modules/` package implements the file-based module system load-and-graph
layer (M2). It sits between the parser and the scope/typecheck passes; it
produces no resolved or typed output — that is M3+.

**Module identity.** A module is identified by a `ModuleId` (tuple of segments)
in `modules/ids.py`. The `ENTRY_ID` sentinel (contains a NUL byte) keys the
entry program. `ModuleId.relpath()` maps a module id to its relative file path
(`foo/bar/baz.agl`).

**Root set.** `modules/roots.py` provides `RootSet` — an unordered,
canonicalized, deduplicated set of search roots. Roots are assembled from the
invocation directory, global library root (`~/.agm/lib`), configured roots
(origin-relative), and `-I` CLI flags. The set is unordered by design;
`sorted_roots()` provides a stable order for diagnostics.

**Resolver.** `modules/resolver.py` provides:
- `resolve_module(module_id, roots)` — searches all roots for the file;
  canonicalizes and deduplicates by canonical path; exactly one → ok; zero →
  `ModuleNotFound` listing all searched roots; ≥2 distinct canonical files →
  `AmbiguousModule`. No first-root-wins shadowing.
- `expand_wildcard(prefix, roots)` — globs `<root>/<prefix>.agl` and
  `<root>/<prefix>/**/*.agl` across all roots; maps each file to its `ModuleId`;
  enforces global uniqueness; empty result → `ModulePrefixNotFound`. Returns a
  `dict[ModuleId, Path]` ordered by `ModuleId`.

**Loader.** `modules/loader.py` provides `load_graph(entry_source, *, entry_path, roots)`:
1. Parse the entry source with `parse_program_seeded(start_id=0)`.
2. BFS over transitive `ImportDecl`s; wildcard imports expand via
   `expand_wildcard`. Each file is parsed with a monotonically growing
   `start_id` seed so **node ids are globally unique (disjoint) across all
   modules**.
3. Terminate when a module id is already loaded — makes **cycles finite and
   safe** (D8).
4. Reject any import that resolves to the entry file's canonical path —
   `ImportEntryError` (D9). No rejection for inline (`-c`) entries (no file
   path).
5. Compute SCCs via Tarjan's algorithm for diagnostics.
6. Return `ModuleGraph{modules, entry_id, sccs}` where `modules` is
   `{ModuleId: LoadedModule}` (entry keyed by `ENTRY_ID`).

Each `LoadedModule` carries: `module_id`, `program` (the `Program` AST), `path`
(canonical file path, `None` for inline entries), `source` (the `SourceId`
stamped on every span), and `imports` (top-level `ImportDecl` nodes).

**Errors.** `modules/errors.py` defines `ModuleNotFound`, `AmbiguousModule`,
`ModulePrefixNotFound`, and `ImportEntryError` — all subclasses of `AglError`,
each carrying a `SourceSpan` from the triggering import declaration so
diagnostics are file-attributed.

**Determinism.** All traversal is deterministic regardless of root-set or
filesystem-discovery order: `sorted_roots()` orders roots, BFS queues are
sorted by `ModuleId` before enqueuing, and `expand_wildcard` results are ordered
by `ModuleId`. The SCC algorithm visits nodes in sorted order.

## Host runtime graph integration (`agm.agl.runtime`)

`WorkflowRuntime` exposes a graph-mode API that mirrors the single-file API:

- `prepare_program(source, *, entry_path, roots) → PreparedGraph` — non-raising
  front-end: calls `load_graph` then `resolve_graph`, captures any exception
  (including `AglError` subclasses) as a `Diagnostic`.  Returns a `PreparedGraph`
  dataclass with `resolved_graph: ResolvedModuleGraph | None` (None on error),
  `diagnostics`, and `warnings`.
- `discover_params_graph(prepared) → ParamDiscovery` — type-checks the graph and
  returns typed param declarations from the entry module.  Returns a stub
  `CheckedProgram` in the `checked` field for compatibility with existing param-
  wiring code in `agm exec`.
- `run_prepared_graph(prepared, *, param_values, check_only, log_file) → RunResult`
  — resumes the pipeline from type checking, validates params, materializes
  contracts from linked descriptors, and either executes `IrInterpreter` or returns
  the linked call-site inventory (`check_only=True`). Agent
  reconciliation uses `resolved_graph.entry_agents` (only the entry module owns
  agents).

`PreparedGraph.config_pragmas` and `PreparedGraph.declared_agents` read from the
entry module's `resolved_graph` when available, falling back to empty/() on
failure — the same interface as `PreparedProgram`, so the exec command's pragma
and agent-wiring code is unchanged.

`agm exec` routes ALL programs through the graph pipeline (single-file programs
produce a graph with only the `ENTRY_ID` module).  It assembles module roots via
`assemble_roots(invocation_root, lib_root, configured, cli, cwd)` where
`invocation_root` is the entry file's parent (file exec) or `cwd` (inline `-c`
exec), and `lib_root` defaults to `Path("~/.agm/lib")` when no config overrides
it.  Additional roots are supplied via the repeatable `-I/--module-path DIR`
CLI flag (resolved relative to the invocation cwd) and the `[modules] roots`
config key.

The module-roots configuration (`agm.config.module_roots`) reads `[modules]`
sections from the layered config files and returns a `ModuleRootsConfig` with
`lib_root` (last-write-wins across layers) and `extra` (union across layers).

Multi-file e2e tests live in `tests/test_agl_multifile.py`; static fixture
modules used by those tests are in `tests/agl/multi_file/`.


## Package layout and test locations

| Package | Component | Tests |
|---------|-----------|-------|
| `agm.agl.lexer` | 1 — custom lexer | `tests/test_agl_lexer.py` |
| `agm.agl.grammar` | 2 — Lark grammar | `tests/test_agl_parser.py` |
| `agm.agl.syntax` | 3 — AST dataclasses | `tests/test_agl_ast.py` |
| `agm.agl.scope` | 4 — name resolution | `tests/test_agl_scope.py` |
| `agm.agl.typecheck` | 5 — type checking | `tests/test_agl_typecheck.py` |
| `agm.agl.eval` | 6 — evaluator | `tests/test_agl_eval.py`, `tests/test_agl_eval_graph.py` |
| `agm.agl.runtime` | host API | `tests/test_agl_runtime.py` |
| `agm.agl.repl` | incremental REPL session (UI-free) | `tests/test_agl_repl_session.py` |
| `agm.commands.exec` | CLI command | `tests/test_exec_command.py` |
| `agm.config.module_roots` | module roots config | `tests/test_config_module_roots.py` |
| `agm.agl.syntax.spans` | `SourceId` / source-aware spans | `tests/test_agl_source_identity.py` |
| `agm.agl.modules` | module-graph loading | `tests/test_agl_modules_ids.py`, `tests/test_agl_modules_roots.py`, `tests/test_agl_modules_resolver.py`, `tests/test_agl_modules_loader.py` |
| `agm.agl.scope.imports` | import environment builder | `tests/test_agl_scope_imports.py` |
| `agm.agl.scope.graph` | graph-aware scope resolver | `tests/test_agl_scope_graph.py` |
| `agm.agl.typecheck.graph` | graph-aware type checker | `tests/test_agl_typecheck_graph.py` |

The end-to-end acceptance suite lives in `tests/test_agl_e2e.py` and
`tests/agl/programs/`. It is **green and part of the standing gate** — `just test` /
`just check` include it with no `--ignore` flag.  All new AgL work must keep
this suite green.  Multi-file e2e tests live separately in `tests/test_agl_multifile.py`
(with fixture modules in `tests/agl/multi_file/`) since they require a module
root rather than the single-file `.scenarios.json` harness.
