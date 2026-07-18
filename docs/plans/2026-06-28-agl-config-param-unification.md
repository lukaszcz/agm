# AgL `config` / `param` Unification — Implementation Plan

**Date:** 2026-06-28
**Branch:** `config-param`
**Status:** Planning (decisions settled with owner; ready to implement)

## 1. Summary

Today AgL has two separate program-input mechanisms:

- **`config` pragmas** — a fixed set of engine-setting keys (`log`, `strict_json`,
  `max_iters`, `runner`, `log_file`, `timeout`), header-only, static-literal values,
  **no in-language binding** (runtime no-op), precedence **CLI > source > config-file**.
- **`param` declarations** — user-defined names, root-only, arbitrary default
  expressions, **readable immutable bindings**, precedence **CLI > config-file > default**.

These are nearly the same machine. This plan **unifies them into one shared
declared-binding mechanism**, keeping **both keywords** and keeping **each keyword's
current precedence**. After the change `config` and `param` differ in exactly three
ways and share everything else:

| | `config` | `param` |
| --- | --- | --- |
| Name-space | fixed engine keys only | user-defined names |
| Source-value precedence | source value **overrides** config-file | config-file **overrides** source default |
| Missing value (no `= e`) | falls back to **engine default** (never required) | **required** (pre-execution error) |

Everything else becomes uniform: both are root-only, declaration-order,
runtime-evaluated **readable bindings** (config keys bind **only when declared**);
both project to the CLI **verbatim** (`--<name>`, kebab by convention); both resolve
from a **program-as-subcommand** config-file section (`[<program>].<name>`); both
support `Option[T]` with a special CLI/config projection; and both are usable in the
**REPL**.

> **Interaction with the do-loop redesign.** A parallel change renames the `max_iters`
> pragma to `max_call_depth` (see [project_agl_do_loop_redesign]). This plan treats the
> loop-bound key generically: wherever it says `max-call-depth` it means "whatever the
> current loop-bound config key is", and the kebab rename here (`max_iters` →
> `max-call-depth`) must be coordinated with that work to avoid a double rename.

## 2. Settled decisions (alternatives, trade-offs, owner selection)

Each decision was put to the owner one-by-one. The **Chosen** option is the owner's
selection; alternatives are recorded for rationale.

### D1 — Keywords & precedence
- **Keep both keywords, keep each one's current precedence, unify everything else
  (Chosen).** `config` keeps `CLI > source > config-file`; `param` keeps
  `CLI > config-file > source-default`. *Pro:* preserves both real use cases — an
  in-source engine setting that beats ambient config files (`config`), and an
  in-program default that ops can override per-environment (`param`). *Con:* a single
  surface still carries two precedence rules (mitigated: the keyword names the rule).
- Collapse to one keyword `config` with one precedence. *Con:* loses one of the two
  precedence behaviors; rejected.

### D2 — Scope of `config`
- **`config` stays the fixed engine-key set; `param` stays user-defined; they share
  all machinery (Chosen).** *Pro:* smaller surface, no ambiguity over what a
  user-declared config "means" to the engine. *Con:* the two keywords differ in
  name-space as well as precedence (not precedence alone).
- Fully user-extensible / symmetric `config`. *Con:* needs a meaning for a
  user-declared config with no engine effect; built-in keys enter the user namespace
  unconditionally.

### D3 — Readability (opt-in bindings)
- **Declared config keys become readable immutable root bindings, but only when
  declared (Chosen).** `config log` (or `config log = true`) brings `log` into scope;
  an undeclared key never occupies the value namespace. Both forms `config X` and
  `config X = e` are allowed, mirroring `param x` / `param x = e`. *Pro:* no namespace
  pre-pollution; uniform with `param`. *Asymmetry recorded:* `param x` with no value is
  **required**; `config X` with no value falls back to the **engine default** (never an
  error), because every engine key has a built-in floor.
- Pre-declare all engine keys as ambient bindings. *Con:* pollutes the value namespace
  unconditionally; rejected.
- Keep config engine-only (not readable). *Con:* leaves a readability asymmetry;
  rejected.

### D4 — Typing of the optional engine keys
- **`timeout: Option[text] = none`, `log-file: Option[text] = none`; the rest are
  `log: bool`, `strict-json: bool`, `runner: text`, `max-call-depth: int` (Chosen).**
  `Option[T]` gets a **special CLI/config projection** (D6 of the owner's notes): a
  value supplies `some(value)`, a `--no-<name>` flag supplies `none`, and absence falls
  through to the next precedence layer (ending at the `none` default). *Pro:* idiomatic
  optionality; preserves `timeout`'s `"30"/"30s"` flexibility (text). *Con:* reading the
  value in-program means unwrapping the `Option`.
- Scalar + sentinel (`""`/`0` = off). *Con:* magic encodings the type system should
  avoid; rejected.
- `timeout` as `Option[int]` (seconds). *Con:* drops the `"30s"` duration form;
  rejected.

### D5 — CLI spelling
- **Verbatim 1:1 projection (source name == flag == binding); kebab-case identifiers
  by convention (Chosen).** AgL identifiers already permit `-` as a continuation char
  (`scanner.py`), so `config strict-json` / `param review-prompt` are legal and project
  to `--strict-json` / `--review-prompt` with no mapping layer. Built-in keys are
  renamed to kebab (`strict_json` → `strict-json`, `log_file` → `log-file`), so the
  existing kebab CLI flags are **preserved, not broken**. *Pro:* one spelling rule;
  dissolves the old underscore↔kebab collision-reservation machinery. *Con:* the prior
  "verbatim underscore" param convention is replaced by "verbatim kebab".
- Keep curated kebab for config + underscore for params. *Con:* two spelling
  conventions; rejected.

### D6 — Config values: runtime, like `param`
- **config values are param-like runtime expressions, root-only, declaration-order;
  binding a config key updates the engine's effective setting from that point forward
  (Chosen).** A config key carries its engine default until its declaration executes;
  the binding then changes the live setting (effect-at-binding-point). *Pro:* one
  evaluation model; supports computed/derived values. *Con:* engine-setting effects
  become order-dependent (a setting declared after an agent call won't affect that
  earlier call); see the §15 watch-item on settings consumed before/independently of
  the eval point.
- Static literals, header-only (early-bound). *Rejected by owner* in favor of full
  param symmetry.

### D7 — Config-file schema: program-as-subcommand
- **Each program is a section `[<program>]`; both config keys and params are named keys
  directly under it (`[review].log`, `[review].scope`); global engine defaults stay in
  `[exec]` (Chosen).** What a key *is* (config vs param) is decided by the program's
  declarations. `<program>` resolves from the `program` decl, else the file stem.
  *Pro:* treats the program as a subcommand uniformly; gives per-program engine
  overrides for free. *Con:* a program named `exec`/`loop`/… collides with a real agm
  command section — handled by reserving those names (§15).
- Today's inverted `[params.<program>]` + global-only `[exec]`. *Replaced.*

### D8 — REPL
- **Allow `config` in the REPL (Chosen).** Entering `config log = true` binds `log` and
  applies the setting from that point in the session; per-program `[<program>].X` values
  apply once a `program` decl is entered, mirroring how params already resolve in the
  REPL. *Pro:* removes the REPL carve-out. Today config pragmas are rejected there.
- Keep config REPL-rejected. *Con:* leaves an asymmetry; rejected.

## 3. Resulting unified model

### 3.1 Declaration forms

```
config NAME            # engine key, value from CLI/config-file/engine-default; readable binding
config NAME = expr      # engine key, source value (high precedence); readable binding
param  NAME            # required user input; readable binding
param  NAME = expr      # optional user input with default; readable binding
param  NAME: T         # required, typed
param  NAME: T = expr   # optional, typed, default checked against T
```

- `config` takes **no type annotation** — each engine key has a fixed type from the
  key registry (§3.3). The value expression (when present) is checked against that
  fixed type.
- `param` keeps optional annotation + inference (today's rules: `param x` → `text`;
  `param x: T`; `param x = e` infers from `e`; `param x: T = e` checks `e` against `T`).
- Both are **root-only** and create **immutable** bindings. `config` is no longer
  header-only.

### 3.2 Value precedence (the only place the two keywords diverge)

For a declared **config** key `X`:

```
CLI --X  >  source (config X = e)  >  [<program>].X  >  [exec].X (global)  >  engine default
```

For a declared **param** `Y`:

```
CLI --Y  >  [<program>].Y  >  source default (param Y = e)  >  required-error (if none)
```

The structural difference: a `config` source value sits **above** the config-file
layers (so `config X = e` makes the config-file irrelevant for `X`, only CLI overrides);
a `param` source default sits **below** them. `config` additionally has a global
`[exec]` base and an engine-default floor; `param` has neither.

### 3.3 Engine key registry (fixed config keys)

| Key (kebab) | AgL type | Engine default | Notes |
| --- | --- | --- | --- |
| `log` | `bool` | `false` | trace logging on/off |
| `strict-json` | `bool` | `false` (lenient) | JSON parse strictness |
| `max-call-depth` | `int` | host floor (today 5) | coordinate with do-loop redesign |
| `runner` | `text` | host floor runner | default agent runner |
| `log-file` | `Option[text]` | `none` | trace file path |
| `timeout` | `Option[text]` | `none` | shell exec timeout (`"30"`/`"30s"`) |

### 3.4 Config-file schema (program-as-subcommand)

```toml
[exec]                 # global engine-config defaults, all programs
runner = "claude -p"
timeout = "30s"

[review]               # program 'review' as a subcommand
log = true             # per-program override of config key 'log'
timeout = "60s"        # per-program override of config key 'timeout'
scope = "AAA"          # value for declared param 'scope'
max-issues = 5         # value for declared param 'max-issues'
```

- `[exec]` field names are renamed to **kebab** to match the key identifiers
  (`strict_json` → `strict-json`, `log_file` → `log-file`, `default_loop_limit` →
  `max-call-depth`). Migration note in §13.
- `Option[T]` in config files: a key with a value → `some(value)`; absence → falls
  through (TOML has no literal null, so explicit `none` is expressed via absence or the
  CLI `--no-<name>` path).

## 4. Current state (baseline touch-points)

From reconnaissance (current line numbers; may shift during implementation):

| Concern | `config` pragma | `param` declaration |
| --- | --- | --- |
| Keyword | `KW_CONFIG` `tokens.py:97` | `KW_PARAM` `tokens.py:69`; `KW_PROGRAM` `:70` |
| Kebab idents | `scanner.py:89–92`, `_IDENT_STOP` `:100–111` | (same) |
| Grammar | `config_pragma` `agl.lark:216`; `pragma_value` `:218–222` | `param_decl` `:205`; `program_decl` `:208` |
| Parser | `config_pragma()` + value builders `transform.py:246–282` | `param_decl()` `:197–208`; `program_decl()` `:210–218` |
| AST | `ConfigPragma` + `PragmaValue` `nodes.py:810–828` | `ParamDecl` `:763–778`; `ProgramDecl` `:782–787`; `Declaration` union `:839–842` |
| Scope | `_PRAGMA_KEY_KINDS` `resolver.py:135–143`; `_validate_pragma_value` `:167–234`; header-only `:739` | `_resolve_param` `:1038–1056`; `BinderKind.param_binding` `symbols.py:112` |
| Resolved | `ModuleResolution.config_pragmas` `symbols.py:243–244` | `program_name` `symbols.py:246` |
| Typecheck | skipped `checker.py:382` | `_check_param` `:436–457` |
| Pipeline | `PreparedProgram.config_pragmas` `pipeline.py:114–123` | `discover_params` `:483–533`; `ParamDeclInfo` `runtime/types.py:88–96` |
| Runtime | applied to driver settings `exec.py:260–264` | `_prepare_ir_params`/`decode_param_value`/`convert_param_value` `runtime/params.py:23–145` |
| CLI exec | built-in flags (`--runner`,`--strict-json`,`--log-file`,`--no-log`,`--max-iters`,`--timeout`) | discovery + `parse_param_tokens`/`check_param_collisions`/`resolve_param_values` `cli_support/exec_params.py`; `exec.py:273–324` |
| REPL | rejected: `_check_no_config_pragmas` `repl/session.py:471–492` | `_pre_eval_param_check` `:523–592`; loader `commands/repl.py:88–105` |
| Config | `ExecConfig` + `exec_config_from_merged` `general.py:536–614` | `load_params_config`/`params_config_from_merged` `:652–693` |
| Docs | `program-structure.md:73–116` | `program-structure.md:117–150`; `bindings-and-scope.md`; `host-environment.md` |
| Tests | `test_agl_config_pragma.py` | `test_exec_params.py`, `test_agl_runtime.py`, `test_exec_command.py`, `tests/agl/programs/basics/inputs_and_types.agl` |

## 5. Goals / non-goals

**Goals**
- One shared declared-binding mechanism behind both keywords; config keys become
  opt-in readable bindings; runtime declaration-order evaluation for config;
  program-as-subcommand config-file schema; verbatim-kebab CLI projection unified
  across config and param; `Option[T]` CLI/config projection; config usable in REPL;
  updated docs and tests; `just check` green.

**Non-goals**
- No new user-extensible config keys (D2).
- No change to the two precedence chains (D1).
- No backward-compat shim for the old `[params.<program>]` table or snake-case `[exec]`
  field names (hard migration, pre-1.0).
- No interactive prompting for missing params (unchanged from prior plan).

## 6. DSL design

### 6.1 Lexer (`lexer/tokens.py`)
- No new keywords (`KW_CONFIG`, `KW_PARAM`, `KW_PROGRAM` already exist). Confirm kebab
  identifier support is unchanged.

### 6.2 Grammar (`grammar/agl.lark`)
- Replace `config_pragma: "config" name EQ pragma_value` with
  `config_decl: "config" name (EQ expr)?` — mirroring `param_decl` minus the type
  annotation. Remove the `pragma_value` / `pragma_true|false|int|decimal|str` rules.
- Keep `config_decl` in `?declaration` (root-only). Drop the header-only position.
- Re-run the LALR(1) conflict guard (`test_agl_parser.py`). `param_decl` already proves
  `"<kw>" name (EQ expr)?` is conflict-free, so `config_decl` (a strict subset) should
  remain clean.

### 6.3 AST (`syntax/nodes.py`)
- Replace `ConfigPragma{key, value: PragmaValue}` with `ConfigDecl{name: str,
  value: Expr | None, span, node_id}`. Remove the `PragmaValue` union.
- Keep `ParamDecl`, `ProgramDecl`. Update the `Declaration` union and any exhaustive
  `isinstance` dispatch.

### 6.4 Parser (`parser/transform.py`)
- Replace `config_pragma()` + value builders with a `config_decl()` builder that reuses
  `param_decl()`'s optional-`expr` handling (no annotation).

### 6.5 Scope (`scope/resolver.py`, `scope/symbols.py`)
- Add `_resolve_config(node: ConfigDecl)`:
  - **Key validation:** `node.name` must be in the fixed key registry (§3.3); unknown
    key → scope error (replaces `_ALLOWED_PRAGMA_KEYS`). Duplicate config key → error.
  - **Root-only**, declaration-order, **immutable** binding with a new
    `BinderKind.config_binding` (for accurate `:=`-on-immutable diagnostics). Resolve
    the value expr (when present) in the scope visible at the declaration point, like a
    param default.
- Remove `_PRAGMA_KEY_KINDS` / `_validate_pragma_value` / header-only enforcement; value
  *type* checking moves to typecheck (§6.6) since values are now expressions.
- `ModuleResolution`: replace `config_pragmas: dict[str, PragmaValue]` with an ordered
  list of resolved config decls (name + node_id), parallel to params. Keep
  `program_name`.
- **Reserved program names:** a `program NAME` (or file stem used as the program key)
  that collides with an agm config section name (`exec`, `loop`, `review`, `revise`,
  `refine`, `repl`, `sync`, `tmux`, `workspace`, `worktree`, `dep`, `agents`, …) is a
  scope/host error (see §15). Centralize the reserved set.

### 6.6 Typecheck (`typecheck/checker.py`)
- Add `_check_config(stmt: ConfigDecl)`: look up the fixed type from the key registry;
  if a value expr is present, check it conforms (with `int → decimal` widening as
  elsewhere); record the binding type in the type env (so the binding is readable and
  the CLI/config decoder knows the type). `config` is no longer skipped.

## 7. Runtime & eval semantics

This is the substantive behavioral change: `config` becomes executable, and engine
settings are applied at their binding point.

### 7.1 Discovery (`pipeline.py`, `runtime/types.py`)
- Extend discovery to return both **params** and **config decls** with resolved types
  and `has_default`/`has_value`. Reuse the single typecheck pass already hoisted for
  param discovery. Add a `ConfigDeclInfo` (name, type, has_value, line, col) parallel to
  `ParamDeclInfo`, or a shared `DeclInfo` with a kind tag.
- Keep the **fixed engine-key catalog** available to discovery independent of
  declaration, because the CLI engine flags exist whether or not the program declares
  the key (§8).

### 7.2 Value resolution & binding (`runtime/params.py`, `eval/interpreter.py`)
- Implement the two precedence chains of §3.2. Shared decoder path
  (`convert_param_value` / `decode_param_value`) already accepts TOML-native and string
  values; reuse it for both config and param external values.
- **Binding is at the decl point, in declaration order** (already true for params).
  `ConfigDecl` is no longer a no-op: when the interpreter reaches it, resolve the value
  per precedence and bind it; if the key is declared as a readable binding, it is in
  scope from here on.
- **Engine effect-at-binding:** binding a config key updates a **mutable settings cell**
  that engine consumers read. Engine consumers must read the *current* value at point of
  use during evaluation:
  - `log` / `log-file` → trace sink consults current values per traced event;
  - `runner` → resolved per agent dispatch;
  - `timeout` → resolved per `exec` call;
  - `max-call-depth` → resolved per call/loop entry;
  - `strict-json` → resolved when **parsing** an agent result (verify it is not baked
    into the pre-execution contract; see §15).
- **Required-ness check (pre-execution):** for every **param** with no default, an
  external value (CLI or `[<program>]`) must exist, else a clean exit-1 error before any
  effect (reuse today's path). Config keys are never required (engine-default floor).
- **`Option[T]` projection:** absence → fall through; a supplied scalar → `some(value)`;
  `--no-<name>` / explicit none → `none`. Implement the coercion in the
  external-value→AgL decoder so `--timeout 30s` yields `some("30s")` rather than
  requiring hand-written enum JSON.
- **Dry-run:** unchanged contract — value expressions (config or param) are not
  evaluated for effects under `--dry-run` (same path as a `let` initializer); the
  pre-execution required-param check still runs.

## 8. CLI design (`agm exec`)

- **Engine flags are always available**, regardless of whether the program declares the
  key: the fixed config keys project to `--log`/`--no-log`, `--strict-json`,
  `--max-call-depth`, `--runner`, `--log-file`/`--no-log-file`,
  `--timeout`/`--no-timeout`. (Note `--max-iters` → `--max-call-depth` via the do-loop
  redesign; `--no-log-file` / `--no-timeout` are new, from the `Option[T]` projection.)
- **Param flags** project verbatim from declarations (`--<name>`), only when declared.
- Reuse the existing two-phase parse (`_RUN_CONTEXT_SETTINGS` + manual leftover
  validation). The discovered option map = fixed engine keys ∪ declared params.
  Unknown/misspelled `--flag` → hard usage error.
- **Collision rule (simplified):** a `param` name may not equal a fixed config key name
  or a reserved built-in flag (`--help`/`-h`, `--dry-run`, `-c`/`--command`,
  `--module-path`/`-I`). The old underscore↔kebab normalization check is removed (one
  spelling now).
- **Help & completion:** `agm exec FILE --help` lists the engine flags plus the
  discovered `--param` options (name, type, required/default); completion offers both.
  Degrade to engine flags only on a parse/typecheck failure (surfacing the real
  diagnostic), as today.
- Both engine values and param values flow into the same external-values dict passed to
  `run_prepared`; precedence is applied per §3.2.

## 9. Config-file design (`config/general.py`, `config/config.toml`)

- **Schema:** read a program's settings from `[<program>]` (program decl name or file
  stem). For each declared **config** key `X`: `[<program>].X` over `[exec].X` (global)
  over engine default. For each declared **param** `Y`: `[<program>].Y` only.
  `<program>` for inline `-c` programs with no `program` decl has no section (CLI /
  defaults only), as today.
- Replace `load_params_config`/`params_config_from_merged` with a unified
  `load_program_config(program_name)` that returns the `[<program>]` table (TOML-native
  values), reusing `load_merged_config` / `_merge_config` / `_select_command_table`.
- `exec_config_from_merged` keeps reading the global `[exec]` base; **rename its field
  reads to kebab** (`strict-json`, `log-file`, `max-call-depth`) and let per-program
  `[<program>]` override the global base for engine keys.
- **Undeclared key** in `[<program>]` → warning, non-fatal (today's O4), naming the key
  and section.
- `Option[T]` value conversion per §7.2; feed TOML-native values to the shared decoder.
- **Template (`config/config.toml`):** document the `[exec]` global defaults and a
  worked `[<program>]` example with both a config override and a param.

## 10. REPL design (`commands/repl.py`, `agl/repl/session.py`)

- **Remove** `_check_no_config_pragmas` and its call — `config` is allowed.
- Extend `_pre_eval_param_check` (or a unified pre-eval resolver) to handle config
  decls: resolve per §3.2 with no CLI layer in a session — for `config X`:
  `[<program>].X` > `[exec].X` > engine default; for `config X = e`: evaluate `e`
  (> `[<program>]`/global only when no source value); for params unchanged.
- Apply the engine setting to the session's mutable settings cell at the decl point
  (effect-at-binding within the session). `program` remains session-global and resolved
  once (its `[<program>]` table applies to decls entered afterwards); `reset` clears it.
- The unified config loader replaces the params-only loader closure.

## 11. Affected files (checklist)

- Lexer: `agl/lexer/tokens.py` (verify only)
- Grammar: `agl/grammar/agl.lark`
- Parser: `agl/parser/transform.py`
- AST: `agl/syntax/nodes.py`
- Scope: `agl/scope/resolver.py`, `agl/scope/symbols.py`
- Typecheck: `agl/typecheck/checker.py`
- Pipeline/discovery: `agl/pipeline.py`, `agl/runtime/types.py`
- Runtime/eval: `agl/runtime/params.py`, `agl/eval/interpreter.py`
- CLI: `cli.py`, `cli_support/exec_params.py`, `commands/exec.py`, `parser.py`,
  `completion.py`
- REPL: `commands/repl.py`, `agl/repl/session.py`
- Config: `config/general.py`, `config/config.toml`
- Docs: see §13
- Tests: see §12

## 12. Test plan (TDD — failing tests first)

- **Lexer/parser:** `config NAME` and `config NAME = expr` parse; old `pragma_value`
  literal-only forms still parse as expressions; `param` forms unchanged; LALR guard
  0/0.
- **Scope:** config key must be a known engine key (unknown → error); duplicate config
  key errors; config root-only + immutable (`:=` on a config errors with the
  config-binding phrasing); config no longer header-only (a config after another item is
  fine); reserved program-name collision errors.
- **Typecheck:** each engine key's fixed type enforced; `config X = e` with a
  type-mismatched `e` errors; `int → decimal` widening where relevant; param rules
  unchanged.
- **Runtime/eval:** both precedence chains (config: CLI > source > `[<program>]` >
  `[exec]` > default; param: CLI > `[<program>]` > default > required-error); a declared
  config key is readable as a binding; an *undeclared* engine key is **not** a binding
  yet still configurable via CLI/config (engine effect only); effect-at-binding ordering
  (a setting declared mid-program applies from that point); `Option[T]` coercion
  (`--timeout 30s` → `some`, `--no-timeout` → `none`, absent → default `none`); dry-run
  does not fire effectful values but still runs the required-param check.
- **Exec CLI (integration, multi-scenario):** engine flags always present
  (declared or not); kebab param flags; structured param via JSON string; `--no-<opt>`
  for Option keys; unknown/misspelled flag errors; param-name vs engine-key collision
  errors; help lists engine + param options; completion offers both; syntax-error file
  degrades to engine flags + surfaced parse error.
- **Config files:** `[<program>].X` overrides `[exec].X` for config; `[<program>].Y`
  supplies params; declared-name (program decl) vs file-stem keying; undeclared key in
  `[<program>]` warns; inline `-c` has no section; kebab field names in `[exec]`.
- **REPL:** `config` accepted; config/param resolution from `[<program>]` once a
  `program` is set; effect-at-binding within a session; missing required param errors.
- **E2E programs (`tests/agl/programs/`):** add programs combining `config` (declared +
  read) and `param` with defaults, Option keys, and per-program config — multiple
  input/mock-response scenarios.

Maintain 100% coverage of `src/` and 100% command coverage in e2e. Do not assert exact
help/error text.

## 13. Documentation updates

- `docs/agl/reference/program-structure.md` — replace the "Config pragmas" section with
  the unified `config` declaration; update the grammar pseudo-code (`config_decl`).
- `docs/agl/reference/bindings-and-scope.md` — document `config` as a declared readable
  binding alongside `param`; note the config-binding immutability phrasing and the
  required-ness asymmetry.
- `docs/agl/reference/host-environment.md` — config keys, their fixed types, the two
  precedence chains, `Option[T]` external projection, program-as-subcommand resolution.
- `docs/agl/reference/lexical-structure.md` / `grammar.md` — config grammar + keyword
  list; kebab identifier convention note.
- `docs/arch/agl/*.md` — note `config` decls are now executable (declaration-order,
  effect-at-binding) and share the param discovery/CLI/config path; update where config
  pragmas were described as no-ops.
- `docs/arch/config.md` — the `[<program>]` schema and kebab `[exec]` fields.
- `docs/commands/agl.md` + `agm exec`/`agm repl` help — engine flags, `--no-<opt>`
  forms, `[<program>]` config, REPL config support; remove the pragma/REPL-rejection and
  `[params.<program>]` descriptions.
- `config/config.toml` — `[exec]` defaults + `[<program>]` example; **migration note**
  for the removed `[params.<program>]` table and snake→kebab `[exec]` fields.
- `README.md` — brief mention if it references inputs/pragmas.

## 14. Suggested sequencing (milestones)

1. **DSL core:** grammar `config_decl` + AST `ConfigDecl` + parser; remove
   `pragma_value`; parser & LALR guard tests. (No value-sourcing change yet.)
2. **Static passes:** scope (`_resolve_config`, `BinderKind.config_binding`, key
   registry, reserved program names) + typecheck (`_check_config`, fixed types); tests.
3. **Runtime semantics:** unified discovery (config + param); executable config binding;
   effect-at-binding settings cell + dynamic engine consumers; precedence chains;
   `Option[T]` coercion; runtime tests.
4. **Exec CLI:** engine flags always-on + param projection through one path; simplified
   collisions; `--no-<opt>`; help + completion; integration tests.
5. **Config:** `[<program>]` loader + per-program engine override + kebab `[exec]`
   fields + template; tests.
6. **REPL:** allow config; unified pre-eval resolution + session settings cell; tests.
7. **Docs sweep** (§13) and final `just check`.

Commit per milestone once gates pass.

## 15. Risks / watch-items

- **Settings consumed before/independently of the eval point.** Effect-at-binding (D6)
  assumes engine consumers read the *current* setting during evaluation. Audit each key:
  `strict-json` must be read at agent-result **parse** time, not baked into the
  pre-execution contract materialization (`_materialize_ir_contracts`); `log`/`log-file`
  must be read by the trace sink per event; `runner`/`timeout`/`max-call-depth` per
  dispatch/exec/call. If any is genuinely needed before execution, decide between
  reading it dynamically vs. documenting that it takes effect from its decl point
  (conventionally declared at the top).
- **Reserved program names.** `[<program>]` shares the top-level config namespace with
  agm command sections; a program named `exec`/`loop`/… would collide. Reserve those
  names (scope/host error) and centralize the set.
- **Config-file migration.** `[params.<program>]` → `[<program>]` and snake → kebab
  `[exec]` fields are breaking; ship the template + docs migration note (no shim,
  pre-1.0).
- **do-loop redesign coupling.** Coordinate the `max_iters` → `max-call-depth` rename so
  it happens once (§1).
- **LALR conflict** from `config_decl`'s optional `(EQ expr)?` — mitigated by the
  `param_decl` precedent; verify via the conflict guard.
- **Eager discovery** for `--help`/completion must still degrade gracefully on
  unreadable/invalid files (engine flags only; no crash).
