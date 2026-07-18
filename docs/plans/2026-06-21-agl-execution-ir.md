# AgL Typeless Execution IR — Implementation Plan

Status: planned · Date: 2026-06-21 · **Every** design decision below is owner-approved.

This is the standalone, authoritative design and implementation plan. It records the architectural
decisions, runtime invariants, migration sequence, and acceptance criteria required to implement
the execution IR; no external design note is required to interpret it. The settled decisions in
§3 and later owner-approved decisions recorded in the relevant sections are authoritative over
earlier design material.

## 1. Goal

Introduce a **lowering/linking phase** after type checking that emits a **linked, typeless
execution IR** (`ExecutableProgram`), and rewrite the evaluator to execute **only** that IR. The
IR carries no checker `Type`, `TypeEnvironment`, `FunctionSignature`, `CastSpec`, type expression,
or `node_types`/`binding_types` table. Every implicit coercion, every binding/call/constructor/
operator/pattern/assignment target, and every cross-module reference is **resolved before
evaluation**.

This makes `CheckedModule` no longer an execution format, fixes the cross-module source-slicing
and missing-binding-type defects structurally, and gives a principled, general, extensible
execution layer. `agm.agl.eval` must not import from `agm.agl.typecheck`, `agm.agl.scope`, or
`agm.agl.syntax.nodes`.

The pipeline becomes:

```
source → lex → parse → resolve → check → lower/link (NEW) → host-prep → evaluate
```

## 2. Non-goals (owner-approved)

- No bytecode, SSA, register allocation, optimization, or AOT.
- No change to AgL syntax. Observable semantics remain unchanged except for the owner-approved REPL
  failure rule in §7: effects completed before a failed entry are no longer rolled back. The
  frame-model change in D5 remains a mechanism-only change.
- No erasure of runtime value tags (`IntValue`, `RecordValue`, … stay).
- No removal of schemas/codecs from agent output, casts, or external params — these are runtime
  protocols, not checker types.
- No IR serialization in this implementation.
- No box-only-captured-vars optimization (see D5); all `var`s are boxed uniformly for now.

## 3. Settled design decisions (authoritative)

These six decisions were settled one-by-one with the owner.

### D1 — Migration safety net: **hybrid temporary differential oracle**
- Keep the existing AST interpreter green and usable as a **differential oracle** *only during the
  migration phases*, gated behind a test marker. A harness runs old + new evaluators on the same
  programs and diffs **values, errors, traces, and external-call sequences**.
- Plus **golden structural-IR tests** for each lowering family.
- The oracle and the old AST execution paths are **deleted at the end of migration** (Milestone 9).
  We do **not** carry two production evaluators indefinitely.

### D2 — Runtime nominal identity: **structured `NominalId` now**
- Records, enums, exceptions, match plans, and catch clauses reference a frozen
  `NominalId(module_id, declared_name)` value object allocated by the linker — **not** a bare
  string, **not** a mangled string.
- A separate `display_name` carries the user-facing source name for rendering and errors.
- Nominal equality is an ID compare. This is collision-free across modules from day one (closes
  the imported-same-name hazard the modules×generics seam exposed).

### D3 — Operation-node granularity: **parameterized nodes + pre-resolved enums**
- A small, stable set of operation nodes, each carrying a **closed enum resolved at lowering**:
  `IrArith(op, kind, …)`, `IrCompare(op, kind, …)`, `IrContains(kind, …)`, and
  `IrCoerce(operation, …)` where `Coercion` is a closed union/enum.
- Operation *selection* happens entirely during lowering (no runtime type sniffing); the
  evaluator switches on the pre-resolved enum.
- **Short-circuiting `IrAnd`/`IrOr` stay distinct nodes** (no singleton enums for them).
- Rationale: small node set, adding a representation = one enum member + one handler arm, not a
  new class; still zero runtime *type* dispatch.

### D4 — Dispatch mechanism: **structural `match` + `assert_never`**
- The lowerer (AST→IR) and evaluator (IR→Value) dispatch over the closed IR/AST union with a
  structural `match` whose final arm is `assert_never(node)`.
- mypy exhaustiveness turns "added a node, forgot a case" into a **compile-time** error at
  `just check` — stronger and cheaper than a runtime dispatch test.
- The `validate_ir` traversal may use a closed-set dispatcher in the spirit of the existing
  `syntax/visitor.py::walk` (TypeError on an unhandled class).

### D5 — Frame & closure-capture model: **per-invocation frames; `let` by value; `var` by cell**
- **Frames are per function invocation only.** A loop does **not** allocate a frame per iteration
  (the current interpreter's per-iteration `Scope` is the wrong *mechanism*).
- **`let` (immutable) → stored & captured BY VALUE.** No cell allocated. Sound because the
  binding always refers to the same value; interior mutability of a referenced container is owned
  by that container's `var` cell, not the `let`.
- **`var` (mutable) → stored as a CELL, captured BY REFERENCE** so closures observe later `:=`. A
  `var` binder allocates a **fresh cell each time it executes** (a loop-body `var` captured by an
  escaping closure captures that iteration's cell).
- **Closures capture explicitly**: lowering computes each lambda's free-var set and records, per
  free var, by-value (immutable) vs by-cell-reference (mutable). This is CPython's
  freevars-vs-cellvars split.
- **No observable semantic change.** Canonical program (becomes a regression e2e test):

  ```agl
  var fns: list[unit => int] = [(fn () => 0), (fn () => 0)]
  var x = 0
  do
    let v = x
    fns[v] := fn () => v
    x := x + 1
  until x > 1
  fns[0]()        // == 0  (iteration-1's immutable v; can never change)
  ```

  And mutable capture still observes mutation:
  `var c = 0; let g = fn () => c; c := 5; g()  // == 5`.
- The differential oracle (D1) should **agree** on these — this is a mechanism change, not a
  behavior change.
- All `var`s boxed uniformly; "box only captured vars" deferred.

### D6 — `validate_ir` enforcement: **explicit test/development validation only**
- The structural IR validator runs only when explicitly enabled at the lowering boundary. Tests
  enable it by default; normal production runs do not pay for it. This does not rely on Python's
  `__debug__` flag.
- The evaluator still keeps its own **defensive value-tag assertions** (`AssertionError` /
  `InvalidIrError`) inline, because a well-formed lowered program cannot violate them — those are
  cheap and stay on.

### Additional owner-approved refinements

These refinements are equally authoritative and supersede earlier alternatives:

- Maintain an explicit, exhaustive AST→IR coverage table; parameterized nodes are used only where
  operations genuinely share evaluation structure.
- Put `NominalId` plus `display_name` on every nominal runtime value and keep a program-level nominal
  descriptor table, including prelude and built-in exception IDs.
- Run the differential evaluators independently against equivalent deterministic fake hosts and
  compare normalized observations; never invoke real agent/shell integrations from the oracle.
- Function-owned defaults are lowered once, evaluated at call time in the definition environment,
  included in capture analysis, and bound through shared direct/indirect call machinery.
- IR runtime descriptors are immutable pure data interpreted by runtime code; shared runtime values
  live below evaluator/runtime implementation layers.
- A program-level symbol descriptor is the sole source of symbol ownership/mutability. Loads and
  assignments carry only `SymbolId`; indexed assignment paths remain inline and ordered.
- REPL execution is deliberately non-transactional: completed effects survive a later failure.
- Architecture docs are updated when production behavior switches (M7/M8) and finalized in M9,
  rather than documenting inactive foundations or waiting until all migration work is complete.

## 4. Package & module layout

```
src/agm/agl/ir/
  __init__.py
  ids.py            # SymbolId, FunctionId, ContractId, SourceId, NominalId, Location
  operations.py     # ArithOp/CmpOp/NumericKind/CompareKind/ContainsKind enums; Coercion union; recipes
  nodes.py          # immutable IR dataclasses + the closed IrExpr union
  program.py        # program/module tables, symbol/nominal descriptors, sources, call sites, params
  contracts.py      # ContractRequest, ParamDecoder, ConversionRecipe (runtime descriptors, no Type)
  validate.py       # structural invariant checker explicitly enabled in tests/development (D6)

src/agm/agl/values.py  # runtime Value union shared by eval/runtime; no frontend/checker dependency

src/agm/agl/lower/
  __init__.py
  lowerer.py        # single-module checked → IR lowering (expression-directed match dispatch)
  graph.py          # whole-program lowering + linking (ID allocation, cross-module wiring)
  coercions.py      # compile_coercion(source, target) -> Coercion | None  (last place Type is read)
  conversions.py    # CastSpec + source/target → ConversionRecipe
  captures.py       # free-variable / capture-mode analysis for lambdas (D5)
```

Dependency rules (enforced by an import-boundary test in Milestone 9):
- `agm.agl.values` may depend only on runtime-neutral primitives and IR identity value objects.
- `agm.agl.ir` may depend only on module IDs, source locations, and runtime-neutral primitives; it
  does not depend on `eval`, `runtime`, or `values`. Contract, decoder, schema, coercion, and
  conversion descriptors are immutable data; they contain no callables and are
  interpreted/materialized by `runtime`.
- `agm.agl.lower` may depend on `syntax`, `scope`, `typecheck`, `ir`.
- `agm.agl.eval` may depend on `ir` and `runtime`, **never** on `syntax.nodes`, `scope`, or
  `typecheck`.
- `agm.agl.runtime` and `agm.agl.eval` share `agm.agl.values`; runtime must not import evaluator
  implementation modules.

## 5. IR specification

### 5.1 IDs, locations, symbols (`ids.py`)
- `SymbolId`, `FunctionId`, `ContractId`, `SourceId`: frozen value objects, unique within an
  `ExecutableProgram`, allocated by the linker. Need not preserve parser `node_id`; a deterministic
  mapping is kept for tests/traces.
- `NominalId(module_id: ModuleId, declared_name: str)` (D2), frozen.
- `Location(source_id, start_offset, end_offset, start_line, start_col)` — identifies its source
  directly; error context comes from `program.sources[location.source_id]` (fixes cross-module
  slicing). Every IR node carries one.
- Each `SymbolId` has exactly one program-level `SymbolDescriptor`: immutable/mutable, optional
  public name, and an owner (`ModuleId` or `FunctionId`). Loads and assignments contain only the
  symbol ID; the descriptor is the single source of truth for addressing.

### 5.2 Program model (`program.py`)
`ExecutableProgram(entry_module, modules, symbols, nominals, functions, sources, call_sites,
params)`;
`ExecutableModule(module_id, initializers, exports, agents)`;
`SourceFile(display_name, normalized_text)`. `symbols` maps every `SymbolId` to its
`SymbolDescriptor`. `nominals` maps every user, prelude, and built-in nominal ID to a descriptor
containing its display name, kind, and required runtime metadata. Only the entry module has
top-level `initializers` under current language rules; library function bodies are ordinary linked
`IrFunction`s.

### 5.3 Node catalog (`nodes.py`)
Closed `IrExpr` union. Families:
- **Constants:** `IrConstInt/Decimal/Bool/Text/Unit`, `IrConstJsonNull`, container literals
  `IrMakeList`, `IrMakeDict`.
- **Bindings/storage:** `IrLoad(symbol)`, `IrBind(symbol, value)`,
  `IrAssign(symbol, path: tuple[IrIndexStep, …], value)`. The lowerer turns every `BindingRef`
  into a `SymbolId`; the program-level symbol descriptor determines ownership, mutability, and
  public rendering metadata. Qualified/imported access therefore needs no evaluator name lookup.
  Each `IrIndexStep(index, location)` contains a lowered index expression. Assignment loads the root
  once, evaluates and validates indexes exactly once from left to right, then evaluates/coerces the
  RHS exactly once, rebuilds immutable containers from the leaf outward, and updates the root cell.
- **Access/rendering:** `IrField(value, field)`, `IrIndex(value, index)`, and
  `IrRenderTemplate(segments)` with explicit text/value segments.
- **Coercions/conversions:** `IrCoerce(value, operation)` (D3 `Coercion` union: `IntToDecimal`,
  `ToJson`, `MapList`, `MapDictValues`, `MapRecordFields`, `MapEnumFields`; identity omitted,
  container coercions carry only child ops that do work); `IrConvert(value, recipe, failure_mode)`.
- **Constructors:** `IrMakeRecord(nominal, display_name, fields)`,
  `IrMakeEnum(nominal, display_name, variant, fields)`,
  `IrMakeException(nominal, display_name, fields)` — `nominal: NominalId` (D2). Exception stays
  distinct (allocates a trace ID). Field validity/order/defaults settled at lowering; each field
  expression carries its required coercion. First-class constructor references lower to
  `IrMakeConstructor(nominal, variant, fields)` and evaluate to `ConstructorValue`.
- **Functions/calls:** `IrFunction(function_id, module_id, params, body)`,
  `IrFunctionParam(symbol, default)`, `IrMakeClosure(function_id, captures, location)` where
  `captures` is the explicit D5 by-value/by-cell list; `IrDirectCall(function_id, arguments)` with
  arguments in parameter order, each a caller expression or `UseDefault(param_index)`;
  `IrIndirectCall(callee, arguments)` (positional only, no named/default metadata). Defaults are
  lowered once with the function, evaluated at call time in its definition environment, and share
  one parameter-binding implementation for direct and indirect calls. Free-variable analysis covers
  both defaults and the body. After named arguments are normalized, supplied argument expressions
  retain the current parameter-order evaluation semantics. Return coercion is explicit in the
  function body IR.
- **Built-ins/host:** `IrPrint`, `IrParseJson`, `IrAsk(agent, prompt, contract_id, parse_policy)`,
  `IrAskRequest(agent, request, contract_id, parse_policy)`, `IrAgentHandle(agent_name)`, and
  `IrExec(command, contract_id|structured-result marker, parse_policy)`. Agent declarations/handles
  lower to linked `AgentValue` construction/load operations. Host operations refer to `ContractId`
  only.
- **Operators (D3):** `IrArith(op, kind, lhs, rhs)`, `IrCompare(op, kind, lhs, rhs)`,
  `IrContains(kind, item, container)`, `IrAnd`, `IrOr` (distinct, short-circuit). Operands include
  explicit widening where the checker selected it. `IrUnary(op, kind, value)` covers logical not
  and numeric negation; `IrVariantIs(nominal, variant, value, negated)` covers `is` tests.
- **Control flow:** `IrBlock`, `IrSequence`, `IrIf`, `IrCase`, `IrLoop`, `IrTry`, `IrRaise`.
  Patterns lower to runtime **match plans** containing only literal comparisons, enum `NominalId`s,
  variant names, field paths, and target `SymbolId`s. Catch clauses carry exception `NominalId`s or
  a catch-all marker. No syntax pattern or type annotation reaches evaluation.

The lowerer and its tests maintain the following exhaustive AST→IR coverage contract. Helper AST
records (branches, named arguments, template segments, patterns, and assignment targets) are lowered
as children/descriptors of the owning IR node and never reach evaluation.

| Source AST family | Required execution representation |
|---|---|
| `UnitLit`, `IntLit`, `DecimalLit`, `BoolLit`, `NullLit`, `StringLit` | corresponding `IrConst*` |
| `ListLit`, `DictLit` | `IrMakeList`, `IrMakeDict` |
| `VarRef` | `IrLoad` or resolved constructor/agent operation |
| `FieldAccess`, `IndexAccess` | `IrField`, `IrIndex`, except resolved qualified constructors |
| `Template` | `IrRenderTemplate` with lowered interpolation expressions |
| `BinaryOp` | `IrArith`, `IrCompare`, `IrContains`, `IrAnd`, or `IrOr` |
| `UnaryNot`, `UnaryNeg` | `IrUnary` with a resolved operation/kind |
| `Cast`, `IsTest` | `IrConvert`/constant sequence, or `IrVariantIs` |
| `Call` | dedicated host/constructor node, `IrDirectCall`, or `IrIndirectCall` |
| `Lambda` | linked `IrFunction` plus `IrMakeClosure` |
| `Block`, `If`, `Case`, `Do`, `Try`, `Raise` | corresponding structured control-flow IR |
| `LetDecl`, `VarDecl`, `AssignStmt`, `ParamDecl` | `IrBind`, `IrAssign`, or `IrParam` metadata/init |
| `FuncDef`, `AgentDecl` | linked function/agent tables plus module initialization |
| `RecordDef`, `EnumDef`, `TypeAlias`, `ProgramDecl`, `ConfigPragma`, `ImportDecl` | erased after their required nominal, link, host, or program metadata is emitted |

Adding an executable AST variant requires updating this table, the closed `IrExpr` union, lowering,
validation, golden tests, and evaluator dispatch in the same change.

### 5.3.1 Runtime nominal values
`RecordValue`, `EnumValue`, `ExceptionValue`, and `ConstructorValue` carry a `NominalId` plus a
`display_name`. Equality, hashing, construction, matching, catching, and `is` use `NominalId`;
rendering, traces, serialization labels, and diagnostics use `display_name`. Prelude nominals and
built-in exceptions receive reserved linker-owned IDs and appear in the same nominal descriptor
table as user declarations. No runtime nominal operation uses a bare display string as identity.

### 5.4 Operations (`operations.py`)
Closed enums: `ArithOp`, `CmpOp`, `NumericKind` (int/decimal), `CompareKind` (int/decimal/text),
`ContainsKind` (list/dict/text), and the `Coercion` union. All selected at lowering.

### 5.5 Runtime descriptors (`contracts.py`)
- `ContractRequest(codec_name, strictness, schema, structured_exec_flag)` — keyed by `ContractId`;
  derived during lowering while static types are available; **no checker `Type`**. Host prep
  materializes it into an opaque `OutputContract` before execution.
- `ConversionRecipe` — compiled from `CastSpec` + source/target; executable steps + runtime
  schema/decoder + user-facing source/target labels for `CastError`; `ConversionFailureMode`
  (`RAISE_CAST_ERROR` | `RETURN_BOOL`). Steps, schemas, and decoders are closed tagged-data unions,
  never stored callables. Total `as?` lowers to `IrConstBool(True)` after preserving any effectful
  source (`IrSequence((source, IrConstBool(True)))`).
- `IrParam(symbol, public_name, required, default, external_decoder, location)`;
  `ParamDecoder` compiled while types are available, exposes public descriptor + display label.
- Dry-run inventory is emitted as program metadata by lowering (from `call_sites`), **not**
  reconstructed by walking IR/AST.

## 6. Lowering & linking algorithm

**Expected-type-directed**, consuming the checker's recorded results (never rechecking):
`lower_expr(node)` inspects resolved binding/call/constructor data and checked node type/op specs,
recursively lowers children, selects one concrete runtime operation, inserts coercions at child/
result boundaries, attaches a module-qualified `Location`, returns an `IrExpr`. A missing side-table
entry is a **lowering/compiler bug** that fails before any program effect.

Coercion selection is centralized in `compile_coercion(source, target) -> Coercion | None`
(`coercions.py`) — the **last** function allowed to read static assignability; output is runtime IR,
identity returns `None`. First implementation prefers a **uniform `IrCoerce`** representation over
pushing coercions into children.

**Whole-program linking** (`graph.py`), only after the full graph type-checks:
1. Allocate `SymbolId`, `FunctionId`, `SourceId`, and `NominalId` for all modules.
2. Lower each module using its own checked side tables.
3. Resolve cross-module references directly to allocated IDs.
4. Collect functions + contract requests into program-wide tables.
5. Store every module's normalized source under its `SourceId`.
6. When explicitly requested, run `validate_ir`.

The public lowering boundary is `lower_program(..., validate: bool = False)`. Production callers use
the default. Lowering tests use a fixture that passes `validate=True` by default; focused validator
tests may call `validate_ir` directly. Validation activation never depends on Python's `__debug__`,
optimization flags, or ambient environment variables.

No side-table merge: once a module is lowered its type environment is discarded. Cyclic imports and
mutual recursion are represented by IDs allocated **before** body lowering. This replaces
`execute_graph`'s table merging and `_merge_graph_into_checked_program`.

## 7. Evaluator design

New `Interpreter` (rewrite of `eval/`), constructed with `program`,
`registry`, `contracts`, `loop_limit`, `strict_json`, `shell_exec_timeout`, `trace`,
`max_call_depth`, `param_values: Mapping[SymbolId, Value]`. **Absent:** `CheckedModule`,
`TypeEnvironment`, source AST, source string, node-type/binding tables, signatures, cast specs.

- **Frames/cells (D5):** a runtime frame is `dict[SymbolId, Slot]` per function invocation. A
  `let` slot holds a `Value` directly; a `var` slot holds a `Cell` (boxed `Value`). The symbol
  descriptor tells `IrBind` whether to allocate a fresh value slot or cell; `IrAssign`/`:=` mutates
  an existing cell. `IrMakeClosure` snapshots
  captures: by-value for immutable free vars, by-cell-reference for mutable ones. Module storage is
  a per-`ModuleId` frame keyed by `SymbolId`. The runtime `Binding`/`Scope` in `eval/scope.py` is
  replaced by this `SymbolId`-keyed model; final entry bindings are rendered through the entry
  module's `public_name` metadata.
- **Dispatch (D4):** one structural `match` over `IrExpr` with `assert_never`.
- **Errors/diagnostics:** runtime language errors carry IR `Location`; excerpts use its `SourceId`
  via `program.sources`, never an ambient entry source. Defensive tag checks raise
  `AssertionError`/`InvalidIrError` (D6). User-triggerable failures remain AgL exception values.

**REPL:** each entry is parsed/resolved/checked/**lowered** against the current link image before
execution; the session keeps a linked runtime image (persistent symbol/function/source IDs + value
bindings) separate from the static check environment. After lowering, parameter validation, and
contract materialization succeed, the entry executes directly against the persistent runtime image.
There is **no runtime rollback**: assignments, bindings, declarations, output, and host calls that
complete before a runtime failure remain observable. Operations not reached before the failure do
not occur. The interpreter reports which symbols/declarations were installed so static name
visibility and the runtime image advance consistently; source/function/contract metadata needed by
retained values also remains linked. The echo interpreter observes the lowered trailing expression
result via explicit entry metadata.

## 8. Differential oracle (D1)

A test-only harness `tests/agl/oracle/` that, for a corpus program: runs the **old** AST
interpreter and the **new** IR evaluator, and asserts equality of (a) final value/bindings snapshot,
(b) raised error kind + message + source excerpt, (c) trace event sequence, (d) external
agent/exec call sequence. Gated behind a pytest marker; exercised across the existing
`tests/agl/programs/` corpus during Milestones 2–8; deleted with the old evaluator in Milestone 9.
Both evaluators run independently against equivalent deterministic fake agents and a fake shell
executor; the oracle must never invoke real agent or shell integrations. Observation normalization
removes/remaps run IDs, event/trace IDs, durations, temporary paths, and other explicitly identified
nondeterministic fields before comparison. Stateful fakes are freshly instantiated from the same
scenario for each evaluator, and the harness captures print/stdout/stderr as part of the observation.
The D5 closure/loop programs are expected to **agree** (mechanism-only change).

## 9. Milestones (TDD; commit per milestone; `just check` green at each)

Each milestone follows TDD (failing tests first), keeps 100% `src/` coverage, and is committed only
once gates pass. Implementation may be decomposed into ≤5 parallel bounded implementer subagents
per milestone with per-task review; the orchestrator owns design and verifies diffs.

- **M1 — IR foundations.** Introduce `agm.agl.values` and migrate runtime-facing, non-closure value
  tags to it; keep the old evaluator's AST-backed closure/constructor compatibility types isolated
  in the legacy evaluator until M9. Add `ir/ids.py`, `ir/operations.py`, `ir/program.py`, and
  skeleton `nodes.py` (constants, blocks, bindings, loads, assignments, `IrCoerce`); add
  `validate.py` (cheap + deep tiers, explicitly enabled by tests). Runtime/IR descriptors are
  immutable data with no callables. Golden-structure tests. No new evaluator yet.
- **M2 — Lower coercion-sensitive core + minimal evaluator.** Lower `let`/`var`/assign/params with
  **explicit `IrCoerce`** at every boundary; `compile_coercion`. New evaluator executes M1+M2 nodes
  with the D5 frame/cell model. Stand up the **differential oracle**; prove every implicit binding/
  assignment/param coercion (incl. nested containers) appears in IR and round-trips through eval.
- **M3 — Expressions, constructors, operators, casts, patterns.** Full `IrExpr` coverage; `IrArith`
  /`IrCompare`/`IrContains`/`IrAnd`/`IrOr` (D3); constructors with `NominalId` (D2);
  migrate all nominal runtime values and consumers to ID-based identity; `IrConvert`/
  `ConversionRecipe` + total-`as?` rule; match plans + catch clauses. Exhaustive
  `match`/`assert_never` dispatch and AST→IR coverage table enforcement (D4) — adding an AST/IR node
  fails `just check` without a case.
- **M4 — Functions, closures, calls.** `IrFunction`/`IrMakeClosure` with explicit captures
  (`captures.py`, D5), including free variables in defaults; function-owned defaults evaluated at
  call time through shared direct/indirect parameter binding; `IrDirectCall` argument normalization;
  `IrIndirectCall`; add the AST-free linked closure/constructor value forms to `agm.agl.values`;
  mutual recursion via pre-allocated IDs. Pin the D5 canonical loop-closure program
  (`fns[0]()==0`) and the mutable capture program as regression e2e tests.
- **M5 — Module linking.** Whole-program lowering/linking (`graph.py`); replace `execute_graph` table
  merge. Regression tests: imported functions with local bindings, cross-module mutual recursion,
  same-named functions/types (NominalId collisions), library-failure source excerpts.
- **M6 — Host preparation as metadata.** Compile `ContractRequest`s + `ParamDecoder`s during
  lowering as pure-data descriptors interpreted by runtime; dry-run inventory from `call_sites`;
  built-in/host nodes refer only to `ContractId`. Remove runtime reads of `contract_specs`,
  `call_sites`, param binding types.
- **M7 — Runtime switch.** `PipelineDriver` lowers after checking and executes only
  `ExecutableProgram`. Oracle runs over the full corpus comparing values/errors/traces/external
  calls. Update `docs/arch/agl.md` for the production lowering/linking pipeline and transitional
  oracle boundary.
- **M8 — REPL switch.** Persistent incremental linked IR image with direct, non-transactional
  execution; retain completed mutations/bindings/declarations after runtime failure; echo via
  lowered trailing-expression metadata. Test successful incremental linking and partial effects
  across failed entries. Update `docs/arch/agl.md` for incremental REPL linking.
- **M9 — Enforce & delete.** Remove AST execution paths, `_merge_graph_into_checked_program`,
  `_binding_type_for`, evaluator `Type` imports and side-table lookups, and the oracle. Add the
  **import-boundary test** (`eval` ↛ `syntax`/`scope`/`typecheck`) and the **reflective
  reachability test** (no AST/resolver/checker/`Type` object reachable from `ExecutableProgram`).
  Finalize `docs/arch/agl.md` by removing migration-only/oracle descriptions, and add/adjust
  `docs/agl/reference/` only if any closure-capture wording needs clarifying (no semantic change).

## 10. Testing strategy

Per the AgL area guidance and TDD: golden structural lowering tests per IR node; oracle semantic
parity during migration; implicit coercion at every boundary incl. nested containers; effectful
expressions evaluated exactly once through coercion and `as?`; named/positional/defaulted/recursive/
indirect/cross-module calls; mutable indexed assignment at multiple depths; records/enums/exceptions
/pattern bindings/catches/nominal-collision; ask/exec/parse-json contracts + dry-run inventory;
required/default/external params + conversion failures; library exceptions + loop-limit errors with
correct source excerpts; the D5 closure/loop regression programs; REPL persistence after partial
failure; the import-boundary and reflective-reachability tests; `just check` at 100% `src/`
coverage and 100% command coverage.

## 11. Required invariants and acceptance

`validate_ir(program)` must enforce at least:

- every referenced symbol, function, contract, source, module, and nominal ID exists;
- every symbol has one descriptor, its owner exists, immutable symbols are never assignment roots,
  and module exports reference symbols owned by that module;
- every direct call has one argument/default marker per parameter, and indirect calls contain no
  named/default metadata;
- every coercion, conversion, contract, schema, and param-decoder descriptor belongs to its closed
  pure-data union and contains no callable or checker object;
- every constructor field name is unique and every nominal runtime constructor references a
  descriptor of the correct kind;
- every source range lies within its identified normalized source;
- no AST, resolver, checker, `Type`, `TypeEnvironment`, or legacy evaluator object is reachable from
  `ExecutableProgram`.

The migration is complete only when:

1. `Interpreter` accepts `ExecutableProgram`, not checked frontend objects.
2. Evaluator code imports no frontend syntax, scope, or type-checking implementation and reads no
   static side tables.
3. Every implicit conversion is represented explicitly in IR.
4. Module execution performs no resolver/typechecker table merge.
5. Runtime diagnostics and traces use the correct source for every module.
6. Params, output contracts, casts, constructors, patterns, and catches execute from runtime
   identities/descriptors rather than checker types or display-name identity.
7. Module, graph, dry-run, and non-transactional REPL behavior matches the approved semantics.
8. Legacy AST execution and migration-oracle code are deleted, architecture docs describe the final
   pipeline, and `just check` passes at required coverage.

## 12. Risks & mitigations

- **Two evaluators during M2–M8.** Mitigated by the bounded, marker-gated oracle (deleted in M9),
  not a permanent dual path.
- **Capture analysis correctness (D5).** Mitigated by the pinned canonical programs + oracle
  agreement; uniform var-boxing avoids a premature optimization.
- **Coercion completeness.** `compile_coercion` is the single chokepoint; nested-container tests +
  the "every boundary" tests guard it.
- **NominalId wiring (D2).** Linker allocates before body lowering; reflective-reachability test
  ensures no string-only nominal leaks where an ID is required.
