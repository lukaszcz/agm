# AgL Typeless Execution IR — Implementation Plan

Status: planned · Date: 2026-06-21 · **Every** design decision below is owner-approved.

Source design: `notes/design_ir.md` (the proposed design). This plan is the implementation
breakdown plus the record of the six architectural decisions settled with the owner. Where this
plan and the note differ, **this plan wins** (the note's class names are illustrative; the note
predates the decisions in §3).

## 1. Goal

Introduce a **lowering/linking phase** after type checking that emits a **linked, typeless
execution IR** (`ExecutableProgram`), and rewrite the evaluator to execute **only** that IR. The
IR carries no checker `Type`, `TypeEnvironment`, `FunctionSignature`, `CastSpec`, type expression,
or `node_types`/`binding_types` table. Every implicit coercion, every binding/call/constructor/
operator/pattern/assignment target, and every cross-module reference is **resolved before
evaluation**.

This makes `CheckedProgram` no longer an execution format, fixes the cross-module source-slicing
and missing-binding-type defects structurally, and gives a principled, general, extensible
execution layer. `agm.agl.eval` must not import from `agm.agl.typecheck`, `agm.agl.scope`, or
`agm.agl.syntax.nodes`.

The pipeline becomes:

```
source → lex → parse → resolve → check → lower/link (NEW) → host-prep → evaluate
```

## 2. Non-goals (owner-approved)

- No bytecode, SSA, register allocation, optimization, or AOT.
- No change to AgL syntax or **observable** semantics (the frame-model change in D5 is a
  *mechanism* change only — observable behavior is preserved; see D5).
- No erasure of runtime value tags (`IntValue`, `RecordValue`, … stay).
- No removal of schemas/codecs from agent output, casts, or external params — these are runtime
  protocols, not checker types.
- No IR serialization in this implementation.
- No box-only-captured-vars optimization (see D5); all `var`s are boxed uniformly for now.

## 3. Settled design decisions (authoritative)

These six decisions were settled one-by-one with the owner. They override any conflicting detail
in `notes/design_ir.md`.

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

### D6 — `validate_ir` enforcement: **test/debug-only**
- The structural IR validator runs in **tests and debug builds only**; normal production runs do
  not pay for it.
- The evaluator still keeps its own **defensive value-tag assertions** (`AssertionError` /
  `InvalidIrError`) inline, because a well-formed lowered program cannot violate them — those are
  cheap and stay on.

## 4. Package & module layout

```
src/agm/agl/ir/
  __init__.py
  ids.py            # SymbolId, FunctionId, ContractId, SourceId, NominalId, Location, StorageClass
  operations.py     # ArithOp/CmpOp/NumericKind/CompareKind/ContainsKind enums; Coercion union; recipes
  nodes.py          # immutable IR dataclasses + the closed IrExpr union
  program.py        # ExecutableProgram, ExecutableModule, SourceFile, RuntimeCallSite, IrParam
  contracts.py      # ContractRequest, ParamDecoder, ConversionRecipe (runtime descriptors, no Type)
  validate.py       # structural invariant checker (test/debug-only per D6)

src/agm/agl/lower/
  __init__.py
  lowerer.py        # single-module checked → IR lowering (expression-directed match dispatch)
  graph.py          # whole-graph lowering + linking (ID allocation, cross-module wiring)
  coercions.py      # compile_coercion(source, target) -> Coercion | None  (last place Type is read)
  conversions.py    # CastSpec + source/target → ConversionRecipe
  captures.py       # free-variable / capture-mode analysis for lambdas (D5)
```

Dependency rules (enforced by an import-boundary test in Milestone 9):
- `agm.agl.ir` may depend only on module IDs, source locations, and runtime-neutral primitives.
- `agm.agl.lower` may depend on `syntax`, `scope`, `typecheck`, `ir`.
- `agm.agl.eval` may depend on `ir` and `runtime`, **never** on `syntax.nodes`, `scope`, or
  `typecheck`.

## 5. IR specification

### 5.1 IDs, locations, storage (`ids.py`)
- `SymbolId`, `FunctionId`, `ContractId`, `SourceId`: frozen value objects, unique within an
  `ExecutableProgram`, allocated by the linker. Need not preserve parser `node_id`; a deterministic
  mapping is kept for tests/traces.
- `NominalId(module_id: ModuleId, declared_name: str)` (D2), frozen.
- `Location(source_id, start_offset, end_offset, start_line, start_col)` — identifies its source
  directly; error context comes from `program.sources[location.source_id]` (fixes cross-module
  slicing). Every IR node carries one.
- `StorageClass(LOCAL, MODULE)`.

### 5.2 Program model (`program.py`)
`ExecutableProgram(entry_module, modules, functions, sources, call_sites, params)`;
`ExecutableModule(module_id, initializers, exports, agents)`;
`SourceFile(display_name, normalized_text)`. Only the entry module has top-level `initializers`
under current language rules; library function bodies are ordinary linked `IrFunction`s.

### 5.3 Node catalog (`nodes.py`)
Closed `IrExpr` union. Families:
- **Constants:** `IrConstInt/Decimal/Bool/Text/Unit`, `IrConstJsonNull`, container literals
  `IrMakeList`, `IrMakeDict`.
- **Bindings/storage:** `IrLoad(symbol, storage)`, `IrBind(symbol, value, mutable, public_name)`,
  `IrAssign(symbol, path: tuple[IrIndexStep, …], value)`. The lowerer turns every `BindingRef`
  into a `SymbolId`; qualified/imported access → module-storage loads. `IrBind.mutable` drives the
  D5 cell-vs-value storage choice.
- **Coercions/conversions:** `IrCoerce(value, operation)` (D3 `Coercion` union: `IntToDecimal`,
  `ToJson`, `MapList`, `MapDictValues`, `MapRecordFields`, `MapEnumFields`; identity omitted,
  container coercions carry only child ops that do work); `IrConvert(value, recipe, failure_mode)`.
- **Constructors:** `IrMakeRecord(nominal, display_name, fields)`,
  `IrMakeEnum(nominal, display_name, variant, fields)`,
  `IrMakeException(nominal, display_name, fields)` — `nominal: NominalId` (D2). Exception stays
  distinct (allocates a trace ID). Field validity/order/defaults settled at lowering; each field
  expression carries its required coercion.
- **Functions/calls:** `IrFunction(function_id, module_id, params, body)`,
  `IrFunctionParam(symbol, default)`, `IrMakeClosure(function_id, captures, location)` where
  `captures` is the explicit D5 by-value/by-cell list; `IrDirectCall(function_id, arguments)` with
  arguments in parameter order, each a caller expression or `UseDefault(param_index)`;
  `IrIndirectCall(callee, arguments)` (positional only, no named/default metadata).
- **Built-ins/host:** `IrPrint`, `IrParseJson`, `IrAsk(agent, prompt, contract_id, parse_policy)`,
  `IrExec(command, contract_id|structured-result marker, parse_policy)`. They refer to
  `ContractId` only.
- **Operators (D3):** `IrArith(op, kind, lhs, rhs)`, `IrCompare(op, kind, lhs, rhs)`,
  `IrContains(kind, item, container)`, `IrAnd`, `IrOr` (distinct, short-circuit). Operands include
  explicit widening where the checker selected it.
- **Control flow:** `IrBlock`, `IrSequence`, `IrIf`, `IrCase`, `IrLoop`, `IrTry`, `IrRaise`.
  Patterns lower to runtime **match plans** containing only literal comparisons, enum `NominalId`s,
  variant names, field paths, and target `SymbolId`s. Catch clauses carry exception `NominalId`s or
  a catch-all marker. No syntax pattern or type annotation reaches evaluation.

### 5.4 Operations (`operations.py`)
Closed enums: `ArithOp`, `CmpOp`, `NumericKind` (int/decimal), `CompareKind` (int/decimal/text),
`ContainsKind` (list/dict/text), and the `Coercion` union. All selected at lowering.

### 5.5 Runtime descriptors (`contracts.py`)
- `ContractRequest(codec_name, strictness, schema, structured_exec_flag)` — keyed by `ContractId`;
  derived during lowering while static types are available; **no checker `Type`**. Host prep
  materializes it into an opaque `OutputContract` before execution.
- `ConversionRecipe` — compiled from `CastSpec` + source/target; executable steps + runtime
  schema/decoder + user-facing source/target labels for `CastError`; `ConversionFailureMode`
  (`RAISE_CAST_ERROR` | `RETURN_BOOL`). Total `as?` lowers to `IrConstBool(True)` after preserving
  any effectful source (`IrSequence((source, IrConstBool(True)))`).
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

**Whole-graph linking** (`graph.py`), only after the full graph type-checks:
1. Allocate `SymbolId`, `FunctionId`, `SourceId`, and `NominalId` for all modules.
2. Lower each module using its own checked side tables.
3. Resolve cross-module references directly to allocated IDs.
4. Collect functions + contract requests into program-wide tables.
5. Store every module's normalized source under its `SourceId`.
6. (test/debug) `validate_ir`.

No side-table merge: once a module is lowered its type environment is discarded. Cyclic imports and
mutual recursion are represented by IDs allocated **before** body lowering. This replaces
`execute_graph`'s table merging and `_merge_graph_into_checked_program`.

## 7. Evaluator design

New `Interpreter` (rewrite of `eval/`), constructed approximately as in the note: `program`,
`registry`, `contracts`, `loop_limit`, `strict_json`, `shell_exec_timeout`, `trace`,
`max_call_depth`, `param_values: Mapping[SymbolId, Value]`. **Absent:** `CheckedProgram`,
`TypeEnvironment`, source AST, source string, node-type/binding tables, signatures, cast specs.

- **Frames/cells (D5):** a runtime frame is `dict[SymbolId, Slot]` per function invocation. A
  `let` slot holds a `Value` directly; a `var` slot holds a `Cell` (boxed `Value`). `IrBind`
  allocates a fresh slot/cell; `IrAssign`/`:=` mutates an existing cell. `IrMakeClosure` snapshots
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
bindings) separate from the static check environment. Promotion is atomic (lower → validate params
+ materialize new contracts → execute in a child transaction → promote both static + runtime
additions on success → discard on failure). The echo interpreter observes the lowered trailing
expression result via explicit entry metadata.

## 8. Differential oracle (D1)

A test-only harness `tests/agl/oracle/` that, for a corpus program: runs the **old** AST
interpreter and the **new** IR evaluator, and asserts equality of (a) final value/bindings snapshot,
(b) raised error kind + message + source excerpt, (c) trace event sequence, (d) external
agent/exec call sequence. Gated behind a pytest marker; exercised across the existing
`tests/agl/programs/` corpus during Milestones 2–8; deleted with the old evaluator in Milestone 9.
The D5 closure/loop programs are expected to **agree** (mechanism-only change).

## 9. Milestones (TDD; commit per milestone; `just check` green at each)

Each milestone follows TDD (failing tests first), keeps 100% `src/` coverage, and is committed only
once gates pass. Implementation may be decomposed into ≤5 parallel bounded implementer subagents
per milestone with per-task review; the orchestrator owns design and verifies diffs.

- **M1 — IR foundations.** `ir/ids.py`, `ir/operations.py`, `ir/program.py`, skeleton `nodes.py`
  (constants, blocks, bindings, loads, assignments, `IrCoerce`), `validate.py` (cheap + deep tiers,
  test/debug). Golden-structure tests. No evaluator yet.
- **M2 — Lower coercion-sensitive core + minimal evaluator.** Lower `let`/`var`/assign/params with
  **explicit `IrCoerce`** at every boundary; `compile_coercion`. New evaluator executes M1+M2 nodes
  with the D5 frame/cell model. Stand up the **differential oracle**; prove every implicit binding/
  assignment/param coercion (incl. nested containers) appears in IR and round-trips through eval.
- **M3 — Expressions, constructors, operators, casts, patterns.** Full `IrExpr` coverage; `IrArith`
  /`IrCompare`/`IrContains`/`IrAnd`/`IrOr` (D3); constructors with `NominalId` (D2);
  `IrConvert`/`ConversionRecipe` + total-`as?` rule; match plans + catch clauses. Exhaustive
  `match`/`assert_never` dispatch (D4) — adding an AST/IR node fails `just check` without a case.
- **M4 — Functions, closures, calls.** `IrFunction`/`IrMakeClosure` with explicit captures
  (`captures.py`, D5); `IrDirectCall` arg/default normalization; `IrIndirectCall`; mutual recursion
  via pre-allocated IDs. Pin the D5 canonical loop-closure program (`fns[0]()==0`) and the mutable
  capture program as regression e2e tests.
- **M5 — Module linking.** Whole-graph lowering/linking (`graph.py`); replace `execute_graph` table
  merge. Regression tests: imported functions with local bindings, cross-module mutual recursion,
  same-named functions/types (NominalId collisions), library-failure source excerpts.
- **M6 — Host preparation as metadata.** Compile `ContractRequest`s + `ParamDecoder`s during
  lowering; dry-run inventory from `call_sites`; built-in/host nodes refer only to `ContractId`.
  Remove runtime reads of `contract_specs`, `call_sites`, param binding types.
- **M7 — Runtime switch.** `WorkflowRuntime` lowers after checking and executes only
  `ExecutableProgram`. Oracle runs over the full corpus comparing values/errors/traces/external
  calls.
- **M8 — REPL switch.** Persistent incremental linked IR image; atomic promotion/rollback; echo via
  lowered trailing-expression metadata. REPL promotion/rollback tests.
- **M9 — Enforce & delete.** Remove AST execution paths, `_merge_graph_into_checked_program`,
  `_binding_type_for`, evaluator `Type` imports and side-table lookups, and the oracle. Add the
  **import-boundary test** (`eval` ↛ `syntax`/`scope`/`typecheck`) and the **reflective
  reachability test** (no AST/resolver/checker/`Type` object reachable from `ExecutableProgram`).
  Update `docs/arch/agl.md` (new lowering phase) and add/adjust `docs/agl/reference/` only if any
  closure-capture wording needs clarifying (no semantic change).

## 10. Testing strategy

Per the AgL area guidance and TDD: golden structural lowering tests per IR node; oracle semantic
parity during migration; implicit coercion at every boundary incl. nested containers; effectful
expressions evaluated exactly once through coercion and `as?`; named/positional/defaulted/recursive/
indirect/cross-module calls; mutable indexed assignment at multiple depths; records/enums/exceptions
/pattern bindings/catches/nominal-collision; ask/exec/parse-json contracts + dry-run inventory;
required/default/external params + conversion failures; library exceptions + loop-limit errors with
correct source excerpts; the D5 closure/loop regression programs; REPL promotion/rollback; the
import-boundary and reflective-reachability tests; `just check` at 100% `src/` coverage and 100%
command coverage.

## 11. Risks & mitigations

- **Two evaluators during M2–M8.** Mitigated by the bounded, marker-gated oracle (deleted in M9),
  not a permanent dual path.
- **Capture analysis correctness (D5).** Mitigated by the pinned canonical programs + oracle
  agreement; uniform var-boxing avoids a premature optimization.
- **Coercion completeness.** `compile_coercion` is the single chokepoint; nested-container tests +
  the "every boundary" tests guard it.
- **NominalId wiring (D2).** Linker allocates before body lowering; reflective-reachability test
  ensures no string-only nominal leaks where an ID is required.
```
