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

## Status log

- 2026-06-16: planning complete; codebase mapped (6 pipeline stages + runtime +
  REPL). Starting S1.
