# AgL v2 — Implementation task breakdown (orchestration tracking)

Tracks execution of `docs/plans/2026-06-16-agl-user-defined-functions.md`.
Goal: expression-oriented AgL v2 with uniform calls, first-class/recursive
functions, agents-as-values, `unit`. Each plan milestone must end green
(`just check`). Reference grammar: `prototypes/agl_v2/`.

## Key orchestrator design decisions (own; not in plan, derived during planning)

- **D-A (block model).** A block is an **expression-sequence** of items
  (declarations, binders, expressions) separated by newline/`;`. Binders scope
  forward over following items (as today's statement sequence already does). A
  block's value/type is its **last item**. A block ending in a binder is a static
  error (plan D2). This minimizes disruption: the current sequence-based scoping
  is preserved; former statements simply become value-producing. NOT nested
  let-in desugaring.
- **D-B (Program.body is a block).** Root is a block; top-level still admits
  declarations + `def`s (mutually recursive pre-pass).
- **D-C (unify stmt/expr control nodes).** `IfStmt`+`IfExpr` → one `If`;
  `CaseStmt`+`CaseExpr` → one `Case`; branch bodies are **blocks** (Expr).
  `DoUntil` → `Do` (yields unit). `TryCatch` → `Try`. `Raise` stays (bottom).
  `SetStmt` stays (yields unit). `Pass`/`PrintStmt`/`AgentCall`/`CallOptions`
  removed.
- **D-D (Call node).** `Call(callee: Expr, args: tuple[Expr,...], named_args:
  tuple[NamedArg,...])`. Single-arg juxtaposition sugar produces the same node.
  `UnitLit` is `()`. `print`/`exec`/`ask` are ordinary calls classified by callee
  name (built-in table) at scope time.
- **D-E (front-end lands together).** lexer tokens + grammar + AST + transformer
  + parser/lexer/AST tests are one coherent unit (Stage 1). Tree is RED
  downstream until Stage 4/5 restore green. Commit only at green checkpoints
  (respect repo "commit per green chunk" policy); within the rewrite, tasks may
  leave the tree red and are integrated before committing.

## Stages (execution order)

Front-end is monolithic; downstream passes adapt feature-by-feature behind the
frozen AST contract.

- **S1 — Front-end + AST contract** (plan M1, plus the syntax surface of M2–M4).
  - S1a: v2 AST nodes/types/visitor + AST tests (the firewall contract). FIRST.
  - S1b: lexer tokens (`def`,`fn`,`->`,`unit`) + lexer tests. Parallel w/ S1a.
  - S1c: grammar port (`grammar/agl.lark` from prototype → %declare + custom
    lexer; |-continuation; templates re-attached) + transformer + parser tests
    + conflict guard (0/0). After S1a+S1b.
- **S2 — Scope pass** (plan M2 scope + M3 call/agent scoping + M4 def scoping).
  Expression-sequence resolution; built-in call classification; `def` + agent
  value bindings; mutual recursion pre-pass.
- **S3 — Typecheck** (M2 unit + M3 built-in typing + M4 function typing).
  `UnitType`/`FunctionType`/`AgentType`; assignability; built-in rules for
  print/exec/ask reusing OutputContractSpec; def/lambda/call checking; branch
  unification w/ unit.
- **S4 — Eval** (M2 value-producing + M3 unified dispatch + M4 closures).
  Block yields value; `Closure`/`UnitValue`/`AgentValue`; unified `Call`
  dispatch; def pre-pass; route print/exec/ask through Call.
- **S5 — Runtime/REPL/prelude** (M3 exec ExecResult + M5 recursion + prelude).
  `ExecResult` record + `exec` two-form typing (D10); `ParsePolicy` enum +
  option-arg types (D11); `RecursionError` + call-depth limit (D8); agent
  reconciliation preserved; REPL handles new nodes.
- **S6 — Corpus + docs sweep** (M6). Migrate `tests/agl/` corpus + rejections;
  multi-scenario e2e (recursion/defaults/function values/`ask(agent:)`); full
  reference-doc rewrite; `docs/arch/agl.md`; README; commands.md. Green.

## Acceptance (from plan)

All milestone acceptance criteria; 100% src coverage; `just check` green; docs
updated and implementation-free; corpus migrated; new rejection programs per new
static error.

## Staging refinement (decided during execution)

- The built-in typing rules for `exec`/`ask` need the prelude TYPES (`ExecResult`
  record D10; `ParsePolicy` enum D11) to exist, so prelude **type definitions**
  (ExecResult, ParsePolicy) + `RecursionError` exception are registered in the
  type system in **S3** (typecheck/types.py + env.py), where the type
  environment is built. Their runtime **semantics** stay later: exec two-form
  (structured vs parsed, raise-on-nonzero) and ParsePolicy→retry in S5; closures
  + UnitValue/AgentValue + depth-limit RecursionError enforcement in S4.
- Option args `format`/`strict_json`/`on_parse_error` are statically extracted
  from literal/constructor arguments for OutputContractSpec (faithful port of
  v1 CallOptions, which were grammar-literals); a non-static such arg is a clear
  static error for now (documented, extensible). `agent:` is a dynamic
  agent-valued expression (resolved at eval; does not affect the static codec).
- S3 split: **S3a** = type-system core (UnitType/FunctionType/AgentType +
  assignability/opacity) + prelude registration + function-signature env; review;
  then **S3b** = checker (block typing, def/lambda/call, built-in rules,
  branch unification w/ unit).

## Status log

- 2026-06-16: planning complete; codebase mapped. S1 (front-end: AST contract,
  lexer tokens, grammar, transformer) done + committed (588c548, dcec293). S2
  (scope) done + committed (08cfe48). Both reviewed, fixes applied, 100% cov.
- 2026-06-17: starting S3 (typecheck), split S3a/S3b.
- 2026-06-17: S3 (typecheck) done + committed (40a0d2d). S4 (eval) done +
  committed (6c64017). S5 (runtime+REPL) done + committed (0899a58). All
  reviewed (Opus), findings fixed (incl. a BLOCKER in S3b call dispatch, a
  BLOCKER in S4 structured-exec, 2 BLOCKERs in S5b REPL render).
- 2026-06-17: S6 (corpus + e2e + docs) done — 38 programs + 80 rejections + new
  functions/ category migrated to v2; remaining unit tests migrated; full
  reference + arch + README + commands + grammar docs rewritten. Corpus review
  caught 13 green-faking masked type/ rejections (fixed). **`just check` GREEN:
  4782 tests, 100% coverage, ruff + mypy clean.** Plan complete.
