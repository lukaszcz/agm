# AgL `param` Redesign — Implementation Plan

**Date:** 2026-06-16
**Branch:** `input-redesign`
**Status:** Planning (decisions settled with owner; ready to implement)

## 1. Summary

Redesign AgL's program-input mechanism on three axes:

1. **Rename the keyword** `input` → `param` (hard rename, no alias).
2. **Streamline CLI passing:** each declared param becomes its own first-class CLI
   option on `agm exec`. `param review_prompt: text` ⇒ `--review_prompt VALUE`,
   assigned to `review_prompt` in the program. The old `--input KEY=VALUE`
   mechanism is removed.
3. **Config files can assign params:** per-program tables in `config.toml` supply
   param values; a CLI option overrides the config value.

In addition, params gain **default-value expressions** (`param x: T = e`), which
turns a `param` declaration into an executable binding (a `let` that is
overridable from outside).

## 2. Current state (baseline)

End-to-end the `input` keyword is wired as follows (for orientation when editing):

| Concern | Location |
| --- | --- |
| Keyword (reserved) | `src/agm/agl/lexer/tokens.py:67,99`; remap `:187` |
| Grammar | `src/agm/agl/grammar/agl.lark:149` (`input_decl: "input" VAR_NAME type_ann?`), listed in `closed_stmt` `:84` |
| Parser transform | `src/agm/agl/parser/transform.py:200-211` |
| AST node | `src/agm/agl/syntax/nodes.py:631-644` (`InputDecl`), `Stmt` union `:680` |
| Scope | `src/agm/agl/scope/resolver.py:366-381` (`_resolve_input`, root-only, immutable); `BinderKind.input_binding` in `scope/symbols.py:30-52` |
| Typecheck | `src/agm/agl/typecheck/checker.py:414-459` (`_check_input`; unannotated ⇒ `TextType`) |
| Runtime — declared/validate | `src/agm/agl/runtime/runtime.py:634-684` (Step 4: missing/undeclared) |
| Runtime — bind | `runtime.py:713-742` (Step 6: bind into root scope) |
| Runtime — convert | `runtime.py:976-1095` (`convert_input`) |
| Runtime — prepared | `runtime.py:128-172` (`PreparedProgram`, `declared_agents` property) |
| Eval | `src/agm/agl/eval/interpreter.py:249-250` (no-op static decl) |
| CLI (exec) | `src/agm/cli.py:811-883`; impl `src/agm/commands/exec.py`; `--input` parse via `core/cli_helpers.py::parse_inputs` |
| CLI (repl) | `src/agm/cli.py:886+`; impl `src/agm/commands/repl.py:43-100` (`preset_input`) |
| Config | `src/agm/config/general.py` (`load_exec_config:517`); template `config/config.toml` |

Today values arrive only via `--input KEY=VALUE` (a `dict[str,str]`), are validated
(Step 4) and bound into the root scope (Step 6) **before** execution; the decl is a
runtime no-op. There are no defaults and no config path.

## 3. Goals / non-goals

**Goals**
- `param` keyword; per-param CLI options on `agm exec`; config-file assignment;
  default-value expressions; clean errors; updated docs and tests.

**Non-goals (explicit)**
- No backward-compat alias for `input` (hard removal).
- No `--input KEY=VALUE` on `agm exec`.
- No CLI per-param seeding for `agm repl` (params there resolve from defaults/config).
- No interactive prompting for missing params (possible later follow-up).

## 4. Settled decisions (alternatives, trade-offs, recommendation)

Each decision was put to the owner; the **Chosen** row is the recommendation and
the owner's selection. Alternatives are recorded for rationale.

### D1 — CLI parsing strategy
- **A. Eager pre-parse → first-class options (Chosen).** Parse the program up
  front to discover params, so each `--param` behaves as real: appears in `--help`,
  tab-completes, rejects unknown options. *Pro:* best UX, typo-safe. *Con:* parse
  must run before/with option resolution.
- B. Loose pass-through (`ignore_unknown_options`, parse leftovers). *Pro:*
  simplest. *Con:* no `--help`/completion, silent typos.
- C. Hybrid (loose now, help later). *Con:* deferred UX, two code shapes.

> **"First-class" means behaviour, not Click registration.** §7.1 still uses
> `_RUN_CONTEXT_SETTINGS` (extra-args pass-through) + a manual leftover parse rather
> than dynamically injected `click.Option`s. This is **not** option B: B's defining
> con is that unknown options pass silently and there is no help/completion. Option A
> as implemented re-supplies all three properties explicitly — the leftover validator
> (§7.1 step 2) hard-rejects unknown/misspelled `--param`, and §7.2 wires the custom
> help renderer and completer from the same discovery. We choose the manual route
> over native dynamic options because dynamic `click.Option` injection fights Typer
> and the repo's custom help renderer (see the §7.1 alternative note); the manual
> route is consistent with `loop` and still delivers A's UX contract.

> Cost note: on the **execution** path this adds **zero** extra front-end work — `agm
> exec` already calls `WorkflowRuntime.prepare(source)` once (`exec.py:121`) and
> reuses the result via `run_prepared`. We reuse that single `PreparedProgram`.
> `prepare` runs lex+parse+scope; building the typed CLI options additionally needs
> the **typecheck** pass (to obtain resolved param types — see §6.7), which `run`
> performs anyway. Discovery just hoists that typecheck ahead of option resolution
> and caches the result so `run_prepared` does not typecheck twice. Extra cost is
> incurred only for `--help`/completion, where no parse/typecheck happens today. No
> threads/async — there is a hard data dependency (options depend on the typed parse)
> and the real latency cost is interpreter startup/imports, not the front end.

### D2 — Config layout & precedence
- **Per-program table keyed by a source-declared name, falling back to the file
  stem; CLI overrides config (Chosen).** A `[params.<name>]` table where `<name>`
  is the program's declared name, or the `.agl` file stem when undeclared. *Pro:*
  programs don't clobber each other; stable when a name is declared. *Con:* adds a
  small `program` declaration to the DSL.
- Global flat `[params]`. *Con:* names collide across programs sharing a config.
- Config-wins precedence. *Con:* surprising; CLI can't override.

### D3 — Keyword migration
- **Hard rename `input` → `param` (Chosen).** Matches the repo's recent keyword
  renames (`prompt` → `ask`). *Pro:* one clean surface. *Con:* breaks existing
  `.agl` files using `input` (acceptable; pre-1.0).
- Alias + deprecation warning / keep both. *Con:* legacy surface to carry & remove.

### D4 — Default values
- **In scope; defaults are ordinary expressions (Chosen).** `param x: T = e`
  evaluates `e` lazily, in declaration order, only when no CLI/config value is
  supplied; `e` may reference earlier params and use the full expression language.
  *Pro:* orthogonal with `let`, expressive (computed/derived defaults). *Con:*
  `param` becomes an executable declaration (not a static no-op).
- Static literals only / constant expressions. *Pro:* smaller change. *Con:* less
  expressive; rejected by owner.

### D5 — Name collisions with built-in `exec` options
- **Reserve built-in option names; error (Chosen).** A param whose `--option`
  would shadow a built-in (`--command/-c`, `--runner`, `--log-file`, `--no-log`,
  `--strict-json`, `--max-iters`, `--help/-h`, `--dry-run`) is a clear build-time
  error telling the author to rename. *Pro:* both namespaces stay predictable.
  *Con:* a few names are off-limits to params (documented).
- Params shadow built-ins / built-ins win silently / namespaced `--param NAME=VAL`.
  *Con:* respectively: lost built-ins; silent CLI gap; less streamlined than the
  bare `--review_prompt` target.

### D6 — Bool params on the CLI
- **Flag form `--flag/--no-flag` (Chosen).** `param verbose: bool` ⇒ `--verbose` /
  `--no-verbose`. *Pro:* idiomatic, pairs with defaults. *Con:* distinct from the
  value form used by other types (documented).

### D7 — `--input` on `agm exec`
- **Remove (Chosen).** Per-param options + config fully replace it.

### D8 — REPL params
- **No CLI param seeding (Chosen).** Remove `--input` from `agm repl` and the
  `preset_input` seeding (`repl.py:98-100`). In a session, `param` decls resolve
  via their default expression and/or a `[params.<name>]` config table (only when a
  `program` decl is entered); an unsupplied param with no default raises the
  standard "missing required param" error at its declaration point.

## 5. Finer decisions (settled with owner)

Recorded with the chosen option and the alternatives considered.

- **O1 — `program` declaration syntax: reserved keyword (Chosen).** `program NAME`
  where `NAME` is a bare identifier (`VAR_NAME`), at the program root, at most once;
  `program` becomes a **reserved keyword** (no longer usable as an identifier — same
  class as `let`/`agent`/`record`). *Alternatives:* a string-literal form
  `program "name"`; or a *contextual* keyword usable as an identifier elsewhere.
  Contextual was rejected as the costliest in AgL: a contextual statement-leader lexes
  as `VAR_NAME`, risking an LALR(1) shift/reduce conflict against the grammar's hard
  0-conflict guarantee, and would require lexer-level lookahead. Reserved keeps the
  grammar clean and is consistent with every other declaration leader.
- **O2 — CLI option spelling: verbatim underscores (Chosen).** The param name maps
  to `--<name>` exactly (`param review_prompt` ⇒ `--review_prompt`); no kebab alias.
  *Rationale:* params are a 1:1 projection of program identifiers onto the CLI, so
  source name == flag name == bound variable, with no mapping to learn/round-trip.
  *Alternatives:* kebab-only (`--review-prompt`, the broader GNU/Typer convention,
  matching AGM's curated built-ins) or accept-both (underscore + kebab alias).
  Accepted trade-off: a command line mixes kebab built-ins with underscore params,
  which also visually distinguishes curated flags from program-derived ones.
- **O3 — Structured-type params on the CLI: JSON string (Chosen).** Values for
  `json`/`list`/`dict`/record/enum params are JSON strings (e.g.
  `--tags '["a","b"]'`), fed through the existing `convert_input`
  (`runtime.py:976-1095`), uniform with config. `text` is verbatim; `bool` uses the
  flag form (D6); `int`/`decimal` parse via the same path. *Alternatives:* JSON-string
  or `@file`; or disallow structured params on the CLI — both rejected for now (can
  add `@file` later if quoting proves painful).
- **O4 — Undeclared config key: warning, non-fatal (Chosen).** A key in a program's
  `[params.<name>]` table that no `param` declares emits a warning and continues
  (tolerant of shared/evolving config), while still flagging likely typos.
  *Alternatives:* hard error (symmetric with the CLI's unknown-option rejection, but
  brittle for shared config) or silent ignore. CLI unknown options remain a hard
  error per D1/D5.
- **O5 — Type rules: text default + inference (Chosen).** Mirrors `let` inference
  plus today's text default:
  - `param x` → `text`, required.
  - `param x: T` → `T`, required.
  - `param x = e` → type inferred from `e`, optional (default `e`).
  - `param x: T = e` → `T`, optional; `e` must conform to `T`.

  *Alternatives:* require an annotation when there is no default, or always require an
  annotation — both rejected as less convenient and divergent from `let`.

## 6. DSL design

### 6.1 Grammar (`agl.lark`)
Replace `input_decl` and add `program_decl`:

```lark
param_decl:   "param" VAR_NAME type_ann? (EQ expr)?     // replaces input_decl
program_decl: "program" VAR_NAME
```

- `param_decl` mirrors `let_decl` (`agl.lark:156`) but with an **optional** `= expr`.
  Keep it in `closed_stmt` (replacing `input_decl` at `:84`); it is **not** added to
  `bar_closed_stmt` (params are root-only, never in bar/inline branch positions).
- Add `program_decl` to `closed_stmt`.
- Verify LALR(1) conflict-freeness (mandatory guard, `tests/test_agl_parser.py`,
  grammar header `:9`). `let_decl` already proves `"<kw>" VAR_NAME type_ann? EQ expr`
  is conflict-free; the optional trailing `(EQ expr)?` and the single-token
  `program_decl` should remain clean — confirm via the conflict-guard test.

### 6.2 Lexer (`lexer/tokens.py`)
- Rename `KW_INPUT = "input"` → `KW_PARAM = "param"` (`:67`), update the `KEYWORDS`
  set (`:99`) and the `GRAMMAR_TOKEN_REMAP` (`:187`, `input`→`INPUT` becomes
  `param`→`PARAM`).
- Add `KW_PROGRAM = "program"` to keywords + remap (`PROGRAM`).

### 6.3 AST (`syntax/nodes.py`)
- Rename `InputDecl` → `ParamDecl`; add a `default: Expr | None` field (None ⇒
  required). Keep `name`, `annotation`, `span`, `node_id`.
- Add `ProgramDecl(name: str, span, node_id)`.
- Update the `Stmt` union (`:680`) and any exhaustive `isinstance` dispatch.

### 6.4 Parser transform (`parser/transform.py`)
- Rename `input_decl` → `param_decl`; parse the optional `type_ann` and optional
  default `expr` (reuse the `let_decl` transform's expr handling).
- Add `program_decl` builder.

### 6.5 Scope (`scope/resolver.py`, `scope/symbols.py`)
- Rename `_resolve_input` → `_resolve_param`; keep root-only + immutable.
  `BinderKind.input_binding` → `param_binding` (rename across error messages).
- **Default expressions:** resolve `default` (when present) in the **scope visible at
  the declaration point**, so a default may reference **earlier** params/decls but
  not later ones (declaration-order semantics). Define the binder, then resolve the
  next decl — i.e. bind name *after* resolving its own default (a default cannot
  reference itself).
- **`program` decl:** root-only, at most once; second occurrence is a scope error.
  Record the program name on the resolved program for the runtime/config layer.

### 6.6 Typecheck (`typecheck/checker.py`)
- Rename `_check_input` → `_check_param`. Type rules per O5: with annotation, resolve
  it and check the default conforms; without annotation but with default, infer from
  the default (like `let`); with neither, default to `TextType`. Record the binding
  type in `type_env` (as today, `:459`) for `convert_input`.

### 6.7 Eval / runtime semantics
This is the substantive behavioral change: `param` is no longer a static no-op.

- **`PreparedProgram` (`runtime.py:128`):** add a `declared_params` property
  (mirroring `declared_agents:154`) returning ordered `ParamDeclInfo`
  (`name`, declared/inferred **resolved type**, `has_default: bool`, `line`, `col`).
  Add a `program_name: str | None` property (the `program` decl name, else `None`).
  - **Discovery needs typecheck, not just `prepare`.** `program_name`, `name`,
    declaration order, and `has_default` come from scope alone, so they are available
    after `prepare` (lex+parse+scope). The **resolved type** cannot: under O5 an
    unannotated `param x = e` infers its type from the default, and both the bool flag
    form (D6) and the structured-JSON form (O3) need the resolved type to choose the
    CLI shape — none of which is known without typechecking. So `declared_params`
    exposes resolved types only **after typecheck**. Add a typed-discovery entry point
    — `PreparedProgram.typecheck()` (caching the resulting `type_env`) or a
    `WorkflowRuntime.discover_params(prepared)` helper — that runs typecheck once.
    It is **non-raising** like `prepare`: on a typecheck failure it surfaces the
    diagnostic and degrades to no typed options (mirrors the parse-failure degrade in
    §7.2). `run_prepared` reuses the cached typecheck result so the source is
    typechecked exactly once.
  These let the CLI build options and required-ness checks **without** re-parsing.
- **Value resolution precedence:** CLI option > config value > default expression.
  External values (CLI strings; config TOML-native scalars/tables, see §8) are passed
  to `convert_input` for coercion to the declared type.
- **Required-ness check stays pre-execution:** before evaluation, every param with
  **no default** must have an external (CLI or config) value, else a clean
  pre-execution error (exit 1), reusing the Step-4 diagnostic path
  (`runtime.py:634-684`). We know statically which params have defaults, so this need
  not run the program.
- **Binding (replaces Step 6, `runtime.py:713-742`) + eval
  (`interpreter.py:249-250`):** when an external value exists, bind the converted
  value at the decl point; otherwise evaluate the default expression in declaration
  order and bind its result. Because defaults can reference earlier params, binding
  must occur as the interpreter reaches each `ParamDecl` (in order), not in a single
  pre-pass. `ProgramDecl` is a runtime no-op.
- Keep `convert_input` (`runtime.py:976-1095`) as-is for external values.
- **Dry-run interaction (decision).** A default expression may contain effects
  (agent/exec calls, O5/D4), so it is subject to the same `--dry-run` side-effect-free
  contract as any other expression: under dry-run, default expressions are **not**
  evaluated for their effects — they go through the existing dry-run evaluation path
  exactly like a `let` initializer (no special-casing). The **pre-execution required
  check** (no-default param missing from CLI+config ⇒ exit 1) still runs under
  dry-run, since it is static and effect-free. Supplied (CLI/config) values still
  convert and bind normally. Add a dry-run test (§11) asserting an effectful default
  does not fire.

## 7. CLI design (`agm exec`)

### 7.1 Two-phase parse
Recommended implementation reuses the codebase's existing manual-parse + custom-help
machinery (as `loop` already does) rather than injecting dynamic Click options:

1. Switch `exec_cmd` (`cli.py:811`) to `_RUN_CONTEXT_SETTINGS` (`cli.py:80`:
   `allow_extra_args`, `ignore_unknown_options`). Built-in options (`-c`, `--runner`,
   `--log-file`, `--no-log`, `--strict-json`, `--max-iters`) and the `FILE` positional
   parse normally; leftover `--param`/`--no-param`/`--param value` tokens land in
   `ctx.args`.
2. In `exec.py`, after the existing single `WorkflowRuntime.prepare(source)`
   (`exec.py:121`), run typed discovery (§6.7) and:
   - **Bail out first on a front-end failure.** If `prepared.resolved is None` (parse
     or scope error) or typed discovery failed, **skip** leftover-token validation
     entirely and surface the normal exit-1 diagnostic. Otherwise the param map is
     empty and a valid `--some_param` would be reported as an "unknown option",
     masking the real parse error (cf. §7.2's "syntax error ⇒ no options").
   - Build the param → option map (verbatim `--<name>`, O2; bool ⇒ `--name/--no-name`,
     D6).
   - **Validate** every leftover token against that map: unknown/misspelled →
     hard usage error (D1); a name colliding with a reserved built-in → build-time
     error (D5).
   - Convert each supplied value per type (O3) and assemble the CLI param dict.
3. Merge values: CLI > config (§8) > default (handled in runtime). Pass the merged
   external dict as `inputs=` to `run_prepared` (`exec.py:166`).

> *Alternative considered:* a custom Click `Command` subclass that appends dynamic
> `click.Option`s after locating `FILE`. Equivalent UX, but fights Typer and the
> repo's custom help renderer; the manual route is more consistent with `loop`.

### 7.2 Help & completion
- **Help:** `agm exec FILE --help` must list the discovered `--param` options (name,
  type, required/default); with no FILE, only base options are shown. Reuse the same
  single typed-discovery (§6.7).
  - **Wiring caveat:** the generic help path is insufficient on its own. The shared
    `--help` callback (`cli.py:_print_context_help`) and `print_help_for_command_path`
    receive only the **command path** — not `FILE`, `-c`, or `ctx.args` — so they
    cannot discover params. So exec needs help handling that has access to the argv:
    either give `exec_cmd` its own eager `--help` (a callback that reads the already
    parsed `FILE`/`-c`/`ctx.args`, runs discovery, and renders base + param options
    before the generic callback exits), or extend `print_help_for_command_path` to
    accept the optional discovered-param list and have `exec_cmd` feed it. Plain base
    options still render via the existing path when no FILE/`-c` is present.
- **Completion (`completion.py`):** add an exec completer that locates `FILE` in the
  current argv, prepares it, and offers the `--param` option names (alongside the
  existing `complete_agl_file`).
- A syntax error in the file yields `declared_params == ()` (no options); leftover
  `--param` validation is **skipped** (§7.1 step 2) so the normal exit-1 diagnostic
  path resurfaces the parse error rather than a spurious "unknown option".

### 7.3 Remove `--input` (D7)
- Delete the `--input` option from `exec_cmd` (`cli.py:824-828`) and the
  `parse_inputs` call (`exec.py:64-69`). Repurpose/remove
  `core/cli_helpers.py::parse_inputs` and `ExecArgs.inputs` (`commands/args.py`) in
  favor of the resolved param dict.

## 8. Config design

- **Loader (`config/general.py`):** add `load_params_config(program_name, *, home,
  proj_dir, cwd)` returning a resolved `dict[str, object]` (TOML-native values, see
  conversion rules below — not pre-stringified) from the merged config's
  `[params.<program_name>]` table, following the existing layering/merge
  (`load_merged_config`, `_merge_config`) used by `load_exec_config:517`.
  - **Keying (D2):** `program_name` = `prepared.program_name` if a `program` decl is
    present, else the `.agl` file stem; for `-c` inline programs with no `program`
    decl there is no config table (inline has no stem) — params then come from CLI or
    defaults only.
  - **Value conversion (do not blindly `str()` TOML scalars).** `str(True)` is
    `"True"`, which `convert_input`'s bool path (`json.loads`) rejects (JSON bools are
    lowercase). Instead **preserve the TOML-native value** and feed it to
    `convert_input`, which already accepts native Python objects (bool/int/Decimal and
    JSON-shaped dict/list) alongside strings (`runtime.py:976-1095`). Exact rules:
    - `bool`/`int`/`decimal`: pass the native TOML scalar through unchanged.
    - structured (`json`/`list`/`dict`/record/enum): accept either a native TOML
      table/array (preferred, validated as-is) **or** a JSON string (O3), uniform with
      the CLI — the user need not hand-write JSON inside TOML.
    - `text`: TOML string verbatim. Reject only genuinely incompatible shapes with the
      standard `convert_input` error, naming the param and `[params.<name>]` key.
  - Undeclared keys → warning (O4).
- **Template (`config/config.toml`):** document a `[params.<program>]` section with a
  worked example.
- **Precedence:** CLI overrides config (D2); config overrides nothing below except the
  default expression (i.e. presence of a config value suppresses default evaluation).

## 9. REPL changes (`commands/repl.py`, `cli.py:886+`)

- Remove `--input` and the `preset_input` loop (`repl.py:43-48,98-100`); drop
  `ReplArgs.inputs`.
- Param resolution in a session, made explicit (the REPL evaluates decls
  incrementally, so ordering matters):
  - A `param` decl resolves **at the point it is entered**: if a `[params.<name>]`
    config value exists it is used, else the default expression is evaluated, else (no
    default) the standard missing-param error is raised at that decl.
  - **`program` is session-global and resolved once.** The active program name (and
    thus which `[params.<name>]` table applies) is set when a `program` decl is
    entered. A param declared **before** any `program` decl resolves with **no config
    table** (defaults/error only) — config does not retroactively apply to params
    already bound. Re-entering a different `program` name is an error in a session
    that already has one (mirrors the source-level at-most-once rule, D2/§6.5);
    `reset` clears it along with the rest of session state.
  - This replaces the old pending-input model entirely: there is no deferred/pending
    param buffer — each decl resolves eagerly in order.
- Confirm `ReplSession.preset_input` and related plumbing are removed or left unused
  per the above.

## 10. Affected files (checklist)

- Lexer: `agl/lexer/tokens.py`
- Grammar: `agl/grammar/agl.lark`
- Transform: `agl/parser/transform.py`
- AST: `agl/syntax/nodes.py`
- Scope: `agl/scope/resolver.py`, `agl/scope/symbols.py`
- Typecheck: `agl/typecheck/checker.py`
- Runtime: `agl/runtime/runtime.py`
- Eval: `agl/eval/interpreter.py`
- CLI: `cli.py`, `parser.py`, `completion.py`
- Commands: `commands/exec.py`, `commands/repl.py`, `commands/args.py`,
  `core/cli_helpers.py`
- Config: `config/general.py`, `config/config.toml`
- REPL session: `agl/repl/` (preset_input removal)
- Docs: see §12.

## 11. Test plan (TDD — write failing tests first)

- **Lexer/parser:** `param` tokenizes; `input` is no longer a keyword (now a plain
  identifier); `program NAME` parses; `param x`, `param x: T`, `param x = e`,
  `param x: T = e` parse; LALR conflict-guard still 0/0.
- **Scope:** param root-only + immutable (set on a param errors); default may
  reference an earlier param but not a later one (forward-ref error); duplicate
  `program` decl errors.
- **Typecheck:** O5 type rules incl. inference from default and conformance failure;
  default expr type mismatch against annotation errors.
- **Runtime/eval:** precedence CLI > config > default; default expression evaluated
  in declaration order and only when unsupplied; required (no-default) missing from
  both CLI and config errors pre-execution (exit 1); default referencing an earlier
  param computes correctly; default with an effect (e.g. agent/exec call) fires only
  when unsupplied; under `--dry-run` an effectful default does **not** fire while the
  required-param pre-execution check still applies.
- **Exec CLI (integration):** `--<name>` assigns; bool `--flag/--no-flag`; structured
  type via JSON string; unknown/misspelled `--param` errors; reserved-name collision
  errors; `--input` removed (errors as unknown option); multi-scenario coverage
  (several param sets × CLI/config/default combinations, per repo testing guidance).
- **Help/completion:** `agm exec FILE --help` lists params; completion offers param
  option names; no-FILE help shows base options only; syntax-error file degrades to
  base options + surfaced parse error.
- **Config:** `[params.<declared-name>]` and `[params.<file-stem>]` resolve; CLI
  overrides config; undeclared key warns; inline `-c` has no config table.
- **REPL:** `--input` removed; default/config resolution; missing required errors.

Maintain 100% coverage of `src/`. Do not assert exact help text.

## 12. Documentation updates

- `docs/agl/reference/`: `program-structure.md`, `bindings-and-scope.md`,
  `host-environment.md`, `lexical-structure.md` (keyword list), `grammar.md`,
  `types.md`, `index.md` — replace `input` with `param`, document defaults and the
  `program` declaration (language-only, no implementation references).
- `docs/arch/agl.md`: note `param` decls are now executable (default-expr evaluation
  in declaration order) and the CLI param-option/eager-prepare flow.
- `docs/commands.md` + `agm exec`/`agm repl` help texts: per-param options, config
  `[params.<program>]`, `--input` removal, bool flag form.
- `README.md`: brief mention if it references inputs.
- `config/config.toml`: `[params.<program>]` example.

## 13. Suggested implementation sequencing (milestones)

1. **DSL core:** lexer/grammar/AST/transform rename + `param` defaults + `program`
   decl; parser & conflict-guard tests. (No behavior change to value sourcing yet.)
2. **Static passes:** scope + typecheck (O5, default-expr ordering); tests.
3. **Runtime semantics:** `declared_params`/`program_name` on `PreparedProgram`;
   executable binding + default-expr eval + precedence + pre-exec required check;
   runtime tests.
4. **Exec CLI:** two-phase parse, per-param options, bool flags, collisions, remove
   `--input`; help + completion; integration tests.
5. **Config:** `[params.<program>]` loader + precedence + template; tests.
6. **REPL:** remove seeding; config/default resolution; tests.
7. **Docs sweep** (§12) and final `just check`.

Commit per milestone once gates pass.

## 14. Risks / watch-items

- LALR conflict from the optional `(EQ expr)?` tail — mitigated by the `let_decl`
  precedent; verify via the conflict guard.
- Eager prepare for `--help`/completion must degrade gracefully on unreadable/invalid
  files (no crash; base options only).
- Ensure the program is parsed **exactly once** — thread the single `PreparedProgram`
  from discovery into `run_prepared`; never re-`prepare` in the impl.
- `program`/`param` becoming reserved words may break identifiers in existing `.agl`
  files using those names (acceptable; note in docs).
