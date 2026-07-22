# AgL Recursive Return-Type Inference — Implementation Plan

Status: planned · Date: 2026-07-22 · **Every owner decision in this plan is resolved.**

## 1. Goal

Improve ordinary `def` return-type inference so recursive calls can participate in inference.
In particular, this program must infer `fib: int -> int` and pass the ordinary type checker:

```agl
def fib(n: int) =
  if n < 2 => n else => fib(n - 1) + fib(n - 2)
```

The implementation is deliberately **best-effort and sound-on-acceptance**, not a complete
principal-type solver. A preliminary pass discovers candidate return annotations using provisional
types; the existing checker then validates those candidates as authoritative annotations. Candidate
inference may reject a function that would typecheck with an explicit annotation, but it must never
cause an ill-typed program to be accepted.

The solution applies uniformly to forward references, direct recursion, mutual recursion, generics,
and modules. It must not recognize Fibonacci, base-case branches, or any other source shape as a
special case.

## 2. Current behavior and cause

The checker currently has two relevant inference mechanisms with incompatible lifetimes:

- The program function-signature pre-pass resolves only declarations with explicit return types.
  Unannotated functions are omitted from the program signature table.
- An unannotated `def` is inferred later, when declaration bodies are checked in source order. Its
  signature is registered only after its complete body has produced a concrete result.
- Every expression owns a short-lived `InferenceEngine`. Before the expression publishes node types,
  call metadata, contracts, or bindings, the region zonks its flexible variables and asserts that none
  remain. Checked modules are validated and sealed again before lowering.

Consequently, a recursive or not-yet-checked unannotated function has no registered result type at its
use site, and the checker raises “Cannot infer return type ... before it is checked.” The failure is
caused by signature availability, not by the `if` expression or arithmetic rules themselves.

The implementation must preserve the valuable output boundary: `InferenceVarType` remains a
checker-internal provisional form and never reaches `CheckedModule`, the type environment after
sealing, lowering, or IR.

## 3. Resolved owner decisions

| # | Decision |
|---|----------|
| D1 | Process imports by **module SCC**. With no import cycle this is ordinary dependency-ordered, per-module inference; all modules in an import cycle are inferred together. |
| D2 | Recursive generic calls that change their type-parameter instantiation are **polymorphically recursive** and require explicit return annotations. Existing annotated polymorphic recursion remains supported. |
| D3 | A recursive group with no concrete result evidence is underconstrained and requires an annotation. Do not infer `bottom` or default to `unit`. |
| D4 | Use a **scoped two-pass** design. Candidate inference traverses only unannotated bodies; ordinary validation then rechecks them and produces the sole authoritative checked artifacts. |
| D5 | Return inference is **definition-local**. Callers and use-site expected types never constrain a declaration's inferred signature. |
| D6 | Candidate discovery may use all internal typing evidence, including branches, returns, operators, constructors, annotations inside the body, and calls. It does not perform termination analysis. |
| D7 | Candidate inference is intentionally best-effort and may choose an unsuitable candidate. The ordinary checker is the soundness authority, and a failed candidate asks for an explicit annotation. |
| D8 | Multiple concrete pieces of evidence combine with the checker's **existing common-type rules**: equality, `int`/`decimal` widening, bottom elimination, and structural provisional unification. Incomparable evidence requires an annotation; there is no default type priority. |
| D9 | Candidate-validation diagnostics use checker-only inferred-return provenance. Errors that consume inferred-return-derived types get inference framing and retain the original error as related context; unrelated errors remain ordinary type errors. |
| D10 | An explicit type annotation is a provenance barrier. A successfully checked annotated binding/function/lambda result receives no inferred-return mark. A successful `as` cast is also a barrier. The marked child expression remains available when diagnosing failure at the boundary. |

The earlier idea of a full assignability/overload constraint layer is explicitly superseded. The
existing `InferenceEngine` remains an exact first-order unifier; this feature adds only the small
provisional propagation behavior needed to discover useful candidates.

## 4. Non-goals and invariants

- Do not implement complete Hindley–Milner inference, subtyping constraints, overload search, type
  classes, termination analysis, or polymorphic-recursion inference.
- Do not use call-site annotations or downstream importing modules as evidence for a function's
  return type.
- Do not add an `InferredReturnType` to `semantics.types.Type`. Diagnostic provenance is not semantic
  type identity and must not affect equality, assignability, schemas, contracts, or lowering.
- Do not retain provisional candidate-pass node types, call sites, warnings, contracts, bindings, or
  other checked side tables. The final pass is the sole publisher.
- Do not create a second AST type checker or duplicate per-expression typing rules.
- Do not change lambda recursion rules, runtime recursion, lowering, IR, or evaluator behavior.
- Do not make candidate selection depend on source traversal order. Common-type evidence must be
  normalized before provisional variables are committed.
- Preserve existing behavior for explicitly annotated functions and non-recursive unannotated
  functions, except that unannotated forward references become inferable when their definitions
  provide sufficient evidence.

## 5. Target typecheck pipeline

The program-level typecheck becomes:

```text
whole-program type declarations
  -> explicit/builtin/extern function signatures
  -> for each import SCC, dependencies first:
       build unannotated-function dependency graph
       process function SCCs, dependencies first
         - allocate provisional result variables
         - run disposable candidate inference
         - close variables into concrete candidate signatures
       publish concrete signatures for later module SCCs
  -> ordinary per-module body checking with all signatures concrete
  -> checked-output closure validation and sealing
```

The loader already computes deterministic import SCCs in reverse topological order. Preserve that
information through scope resolution by adding an `import_sccs` (or equivalently named) immutable
field to `ResolvedProgram`, populated directly from `ModuleGraph.sccs`; do not recompute a second
import graph in typecheck. Update the scope/program artifact documentation and its construction tests.

`check_module` uses the same machinery with a synthetic singleton module SCC. This keeps the
single-module white-box API behavior equivalent to the production `check_program` path.

### 5.1 Function dependencies within a module SCC

Within the current import SCC, build a graph of top-level unannotated ordinary `FuncDef`s. A function
depends on another eligible function when a resolved function binding is referenced anywhere in its
body, including through a first-class reference or explicit type application; inspect resolver
`BindingRef`s by globally unique declaration node id rather than names. Annotated, builtin, and extern
functions already have complete signatures and are graph leaves for candidate purposes.

Use `agm.util.graph.sccs` for deterministic function components. Process sink/dependency components
first so an acyclic unannotated generic function is closed into a concrete type scheme before a later
function instantiates it. Only a genuinely recursive function SCC needs several provisional results
alive together.

Defaults do not contribute to a declaration's return type and are skipped by candidate inference;
they remain checked once in authoritative validation. Top-level callers are also excluded by D5.

### 5.2 Signature records and shared header resolution

Replace the growing tuple used by the program signature pre-pass with a named internal record carrying:

- declaration node id and name;
- resolved `FunctionSignature` and erased `FunctionType`;
- whether it is builtin/extern;
- whether its return was explicitly declared or candidate-inferred;
- the candidate's evidence/provenance needed for diagnostics.

Refactor parameter and annotated-result resolution into one shared helper used by the standalone
checker, program pre-pass, and candidate coordinator. This removes the parameter/signature-building
duplication currently present in `checker.py` and `program.py`.

## 6. Best-effort candidate inference

### 6.1 Provisional function signatures

For each unannotated function SCC:

1. Create one shared `InferenceEngine`.
2. Allocate one flexible result variable per function and build a provisional signature from its
   already-resolved parameter types, declared rigid type parameters, and that result variable.
3. Seed every temporary module environment in the batch with all explicit signatures, candidates
   already closed in dependency components, and the current component's provisional signatures.
4. Check each unannotated body through the existing `_Checker` expression logic in a disposable
   candidate mode.
5. Combine explicit `return` operands and the body tail with the existing common-type rules, then
   constrain the declaration's provisional result to that candidate.
6. Zonk the component. Every result must be concrete apart from legitimate rigid source type
   variables. If any flexible variable remains, report the group as underconstrained and request an
   annotation.
7. Convert results to ordinary concrete signatures, publish them to the program signature table, and
   discard the candidate checker and all of its side tables.

Bottom-only evidence (including a body that always raises) does not solve a result. A component made
only of calls to itself or its peers remains unresolved under D3.

### 6.2 Candidate-mode lifecycle

Refactor expression-region setup so candidate checking can supply the SCC's shared engine and select a
disposable mode. The normal mode keeps its current behavior unchanged. Candidate mode must:

- record temporary node/binding information only as needed to continue checking the body;
- allow SCC-owned variables to remain unresolved when an individual expression ends;
- skip authoritative contract/schema construction, extern inventory publication, warnings, and
  checked-output closure assertions;
- suppress an operation error only when a still-flexible provisional type participates; fully
  concrete errors remain ordinary type errors;
- never install temporary artifacts in a final `CheckedModule` or a persistent REPL environment.

Prefer a small explicit mode/session object over scattered boolean flags. It should centralize the
shared engine, provisional declaration ids, disposable artifacts, and candidate evidence.

### 6.3 General provisional propagation rules

Extend the existing checker helpers, not individual syntax examples. Candidate mode follows these
rules:

- **Known expected shape:** when assignability is checked between a flexible provisional type and a
  concrete expected type, use exact structural unification as candidate evidence. Ordinary concrete
  assignability/coercion is still validated only by the final checker.
- **Branches, returns, and literal elements:** compute the common type of concrete evidence first,
  using today's equality/`int`-to-`decimal`/bottom rules, then unify provisional participants with the
  result. If all participants are provisional, unify their variables without inventing a concrete
  type. This makes evidence order-independent.
- **Shape-preserving operations:** propagate or unify provisional operand types where the operation's
  result has the same type (for example unary negation and same-family arithmetic). A concrete peer
  may provide an exact candidate.
- **Fixed-result operations:** expose their fixed result (`bool` for comparisons/logical tests,
  `decimal` for division) while leaving unresolved operand validity for final checking. Concrete
  invalid operands still fail immediately.
- **Calls and constructors:** their declared parameter/field shapes may constrain provisional
  arguments exactly. Their provisional result flows normally into the containing expression.
- **Unknown nominal/container shape:** do not guess record owners, enum owners, field types, or
  container shapes. Return a fresh provisional result when checking must continue, and leave it for
  other evidence to solve. If no evidence does, require an annotation.

These rules intentionally trade completeness for simplicity. Add them through centralized helpers for
candidate assignability, common-type selection, and provisional operation results so adding a future
expression form does not require recursion-specific logic.

Only add neutral `InferenceEngine` APIs that the shared lifecycle genuinely needs (for example a
stable unresolved/progress query). Do not teach it AgL assignability, coercion, field lookup, or
operator overload policy.

### 6.4 Generic recursive functions

An inferred generic result may contain the declaration's rigid `TypeVarType`s, such as:

```agl
enum NonEmpty[T]
  | Last(value: T)
  | More(value: T, rest: NonEmpty[T])

def first[T](items: NonEmpty[T]) =
  case items of
    | Last(value) => value
    | More(value, rest) => first(rest)
```

Such a uniform recursive call is eligible for inference.

Record the concrete generic instantiation selected for every dependency edge inside a recursive
function SCC. A self-edge is uniform only when the callee's type arguments are the caller's rigid
parameters positionally and unchanged. For a mutual generic SCC, establish the corresponding rigid
parameter vector for each member and require every recursive edge to preserve it; an arity change,
container growth, permutation, fixed replacement, or otherwise changed instantiation is polymorphic
recursion. If any edge is non-uniform or cannot be proven uniform, reject candidate inference for the
affected recursive group and require explicit returns. Explicitly annotated functions retain today's
polymorphic-recursion support.

Close each acyclic generic function component before dependents instantiate it. Never expose an
unsolved result variable as a reusable polymorphic scheme.

## 7. Authoritative validation and diagnostics

After candidate discovery finishes for all import SCCs, run the existing full per-module checker with
the completed signature table. Candidate signatures behave like implicit annotations for type
checking, coercion selection, contract creation, side-table publication, and lowering. Normal
expression regions still close independently and must contain no flexible variables.

### 7.1 Checker-only inferred-return provenance

Maintain temporary provenance maps in the final checker, separate from semantic types:

```text
expression node id -> set of candidate-inferred function declaration ids
binding node id    -> set of candidate-inferred function declaration ids
```

The set form handles mutual recursion and values whose type depends on more than one inferred
signature. Seed provenance when a call or first-class function result obtains its static result from a
candidate signature. Propagate it only when the enclosing expression's **static type** depends on the
marked child type; use conservative union where a candidate-inferred generic instantiation or
common-type selection uses several child types. A `let`/`var` without an annotation copies its
initializer's provenance to the binding, and a later `VarRef` restores it.

Do not blindly treat runtime value flow as type provenance. A fixed-result operation or an ordinary
explicitly annotated function result is clean after its inputs have been checked, although an error
while checking a marked input can still use that input's provenance.

The maps are diagnostic scratch state only. They are absent from `TypeEnvironment`'s published types,
`CheckedModule`, schemas, lowering, and IR.

### 7.2 Annotation barriers

No node whose resulting type is fixed by its own explicit annotation may receive inferred-return
provenance:

- an annotated `let` or `var` checks its marked initializer, then installs a clean binding;
- an explicitly annotated `def` or `fn` publishes clean call results;
- a successful `as Target` cast produces a clean result;
- the inner expression remains marked while the annotation/cast boundary is being checked, so a
  failure at that boundary can still identify the inferred function responsible.

Explicit generic type arguments select an instantiation but are not themselves a result-type
annotation; apply the ordinary “does this result type depend on marked evidence?” rule.

### 7.3 Error framing

At checker operations that fail, inspect provenance of the expressions/types directly participating
in the failed check. When nonempty:

- raise a stable top-level diagnostic stating that the named function's return type could not be
  inferred/validated and that an explicit annotation is required;
- retain the original error text and source span as related context;
- for several candidate functions, name the deterministic declaration-order group and attach relevant
  declaration/evidence spans without depending on dictionary order.

When provenance is empty, preserve the normal type error unchanged. Do not classify errors by parsing
their message strings.

An unresolved or conflicting candidate fails in the preliminary phase with the same inference-focused
guidance. An unsuitable concrete candidate fails in authoritative validation with the provenance-aware
wrapper above.

## 8. File-level implementation map

### `src/agm/agl/scope/program.py`

- Carry loader-computed import SCCs on `ResolvedProgram`.
- Preserve immutability and deterministic dependency order.

### `src/agm/agl/typecheck/program.py`

- Refactor the signature pre-pass into explicit-header collection followed by module-SCC candidate
  inference.
- Orchestrate function dependency SCCs and publish only closed candidate signatures.
- Replace internal signature tuples with a named metadata record.
- Seed final module environments from the completed program signature table.

### `src/agm/agl/typecheck/checker.py`

- Share function-header construction instead of duplicating it in the program pass.
- Add the explicit disposable candidate-checking session/lifecycle.
- Generalize provisional-aware common-type, assignability, and operation helpers as described in §6.3.
- Make final `def` checking distinguish explicit from candidate signatures only for diagnostics; both
  are otherwise validated identically.
- Add checker-only expression/binding provenance and annotation barriers.

### `src/agm/agl/typecheck/inference.py`

- Keep exact unification semantics intact.
- Add only lifecycle/query support proven necessary by SCC candidate inference, with direct unit tests.
- Extend `ConstraintRole`/origin data only where it improves deterministic candidate diagnostics.

### Recommended focused module

Put dependency collection, candidate signature metadata, SCC orchestration helpers, and candidate
closure diagnostics in a focused module such as `typecheck/function_inference.py`. It may coordinate
the checker and exact engine, but it must not duplicate expression traversal or typing rules.

### Environment and checked artifacts

- Add only internal metadata/accessors needed to distinguish candidate signatures during final
  checking.
- Keep public `FunctionSignature`, `FunctionType`, and every checked-output type concrete and
  provenance-free.
- Preserve all `assert_checked_*_closed` checks unchanged or strengthen their tests; never weaken them
  to accommodate this feature.

## 9. TDD implementation sequence

Every milestone starts with failing behavioral tests and ends with its focused tests passing. Tests
must assert observable types/diagnostics rather than private traversal order.

### M1 — Characterize the desired single-module behavior

Add failing tests in `tests/test_agl_typecheck.py` for:

- the exact `fib` example inferring `int`;
- direct recursion where the recursive call is the whole alternative branch;
- non-tail recursion where recursive calls are operator operands;
- same-module unannotated mutual recursion (`is_even`/`is_odd`) inferring `bool`;
- an unannotated forward reference and first-class/partial reference to a later function;
- early `return` evidence and a nested `case`/`try` representative;
- `int`/`decimal` evidence selecting the existing widened common type;
- evidence order permutations producing the same signature.

Keep/add negative tests for:

- `def loop() = loop()` and evidence-free mutual recursion requiring annotations;
- bottom-only bodies requiring annotations;
- incomparable concrete return evidence requiring annotations;
- caller annotations not constraining an underconstrained definition.

### M2 — Candidate lifecycle and exact-engine integration

Add focused `tests/test_agl_inference.py` coverage for any new neutral engine API, provisional structural
unification, deterministic origins, unresolved detection, and rigid-variable solutions. Then implement
the singleton candidate pass used by `check_module`, including disposable regions and candidate-aware
common-type helpers. Make all M1 tests pass without changing checked-output closure assertions.

### M3 — Function dependency SCCs and generics

Add tests for dependency ordering, acyclic generic scheme closure, ordinary uniform generic recursion,
and mutually recursive generic functions whose parameters are preserved. Add rejection tests for
self and mutual polymorphic recursion with changed/growing instantiations, alongside a control proving
that the same program remains legal with explicit return annotations.

Cover generic functions used as values and explicit type application so a provisional result is never
generalized before its dependency component closes.

### M4 — Program/module SCC integration

In `tests/test_agl_typecheck_program.py`, add failing tests for:

- an imported unannotated function inferred in a dependency module and consumed by its importer;
- qualified and open-import cross-module mutual recursion with omitted returns inside an import cycle;
- deterministic results independent of module discovery order;
- same-spelled functions in different modules remaining keyed by declaration id;
- private functions participating within their own module without leaking visibility;
- a cross-module underconstrained group producing a useful multi-source diagnostic.

Add scope/program tests proving `ResolvedProgram` preserves loader SCCs. Update the existing
cross-module recursion fixtures to retain annotated controls and add unannotated variants rather than
losing coverage of explicit signatures.

### M5 — Validation diagnostics and provenance barriers

Add tests showing:

- an error directly caused by a candidate recursive result receives inference framing and retains the
  original error/span as related context;
- provenance flows through an unannotated `let` and a later `VarRef`;
- an unrelated concrete error in the same inferred function remains an ordinary error;
- annotated `let`/`var`, annotated `def`/`fn` results, and successful casts clear provenance;
- failure at an annotation/cast boundary can still identify the marked child;
- generic-result and common-type propagation carries provenance only when the result type depends on
  marked evidence;
- no provenance appears in checked artifacts and no `InferenceVarType` escapes.

### M6 — REPL and end-to-end coverage

- Add a REPL test defining and calling an unannotated recursive function in one entry and confirming
  that only its concrete signature is promoted. A failed candidate entry must promote nothing.
- Exercise inferred direct and mutual recursion through the existing function program/scenario suite;
  keep runtime recursion-depth tests mocked/configured exactly as today and never invoke a real agent.
- Update obsolete tests that specifically expected an annotation merely because an unannotated
  function was later/recursive; retain rejection expectations for genuinely underconstrained cases.

## 10. Documentation updates required with implementation

Architecture changes must land with code:

- `docs/arch/agl/frontend/types.md` — describe explicit-header collection, module/function SCC
  candidate inference, authoritative recheck, and the closed-output boundary.
- `docs/arch/agl/modules.md` — describe import-SCC-ordered inferred signature publication and
  `ResolvedProgram` carrying loader SCCs.
- `docs/arch/agl/frontend/scope.md` — mention the preserved import-SCC metadata if the resolved-program
  artifact description warrants it.
- `docs/arch/agl/repl.md` — state that candidate inference is entry-transactional and only concrete,
  validated signatures are promoted.

User documentation:

- `docs/agl/reference/functions.md` — replace the current statement that recursive unannotated calls
  necessarily require annotations; document best-effort recursive inference, mutual/forward support,
  underconstrained groups, definition-local behavior, and annotations as the escape hatch.
- `docs/agl/reference/generics.md` — distinguish inferred uniform function recursion from annotated-only
  polymorphic function recursion (without changing recursive-type rules).
- Update any command/help examples only if they currently claim recursive returns cannot be inferred.

Keep user-facing docs semantic and concise; do not expose checker class names or provisional side-table
mechanics there.

## 11. Performance and safety checks

Candidate inference traverses only unannotated function bodies. It skips annotated bodies, defaults,
top-level expressions, contract/schema finalization, lowering, and every runtime operation. The final
pass remains unchanged and authoritative. This adds linear work proportional to unannotated bodies,
not a second whole compilation.

Do not add provisional-artifact caching in this change. Reusing candidate side tables would create a
second publication path and weaken the closed-output invariant. If profiling later shows a meaningful
cost, optimize it as a separate measured design.

Maintain deterministic ordering by module id, declaration location/node id, and solver origin order;
never by set/dict iteration or object identity.

## 12. Verification

During implementation, run focused tests after each milestone with `uv run pytest ...`. Before
completion run:

```sh
just check
```

The final gate must retain 100% `src/` coverage, strict mypy, ruff formatting/linting, all rejection
fixtures, REPL tests, multi-file tests, and end-to-end command coverage. No `type: ignore`, `noqa`, or
formatter suppression is permitted.
