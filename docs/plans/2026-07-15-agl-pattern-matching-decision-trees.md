# AgL Pattern Matching Decision Trees — Implementation Plan

Status: implemented · Date: 2026-07-15 · **Every** design decision below is owner-approved.

This is the authoritative implementation plan for replacing recursive runtime match plans with
compile-time pattern-matrix compilation. It covers the compiler algorithm, static diagnostics,
the one-level execution IR, pipeline integration, tests, migration, and documentation.

## 1. Goal

Compile every source `case` expression after type checking into a decision tree before ordinary
AST→IR lowering. Complex nested frontend patterns remain a source-language facility; the execution
IR contains only one-level tests whose branch bodies contain further one-level tests.

The pipeline becomes:

```text
source → parse → resolve → typecheck → match compile → lower/link → IR eval
```

The match compiler must provide all of these results from the same compiled decision structure:

- preserve source-order, first-match semantics;
- evaluate the scrutinee once and test any occurrence at most once on an execution path;
- reject every non-exhaustive `case`, with one deterministic missing-pattern witness;
- reject every redundant source arm;
- select good test orders with Maranget's `qba` heuristic;
- hash-cons identical subtrees within one source `case` so the immutable result is a DAG rather
  than an exponentially duplicated object tree;
- lower to a small typeless `IrCase` with one-level variant/literal keys, immediate payload
  bindings, and an optional default expression; and
- delete recursive `IrMatchPlan` interpretation from the evaluator.

`MatchError` remains a normal builtin exception. The compiler never inserts it implicitly: a user
who intentionally wants a partial match writes an exhaustive final arm which explicitly raises
`MatchError`.

## 2. Research basis

The design follows three primary sources:

- Luc Maranget, [Compiling Pattern Matching to Good Decision
  Trees](https://moscova.inria.fr/~maranget/papers/ml05e-maranget.pdf), ML 2008: pattern matrices,
  specialization/default, decision-tree compilation, column heuristics, and maximal sharing.
- Fabrice Le Fessant and Luc Maranget, [Optimizing Pattern
  Matching](https://pauillac.inria.fr/~maranget/papers/opat/), ICFP 2001: source-priority
  preservation, clause matrices, incompatibility, exhaustiveness-informed defaults, and the
  runtime/code-size trade-off between backtracking automata and decision trees.
- Luc Maranget, [Warnings for Pattern
  Matching](https://www.cambridge.org/core/journals/journal-of-functional-programming/article/warnings-for-pattern-matching/3165B75113781E2431E3856972940347), JFP 2007:
  formal usefulness/exhaustiveness definitions and construction of missing-pattern witnesses.

The 2001 paper's backtracking automaton, mixture rule, contexts, traps, and labeled exits are not
the target representation. AgL instead uses the 2008 decision-tree compiler: specialization
removes a tested occurrence from descendants, so the evaluator never needs a backtracking state or
a sophisticated matching plan. The 2007 usefulness algorithm informs correctness and witness
construction, but diagnostics are derived from `Fail` nodes and reachable action leaves in the
compiled tree, not from a second independent usefulness pass.

## 3. Current state and defects

Today the frontend already supports wildcard, binder, literal, bare nullary-variant, and nested
enum-constructor patterns. Scope resolution classifies bare constructor names, and type checking
resolves constructor arguments into declaration-order field bindings.

The remaining pipeline has three structural problems:

1. `typecheck/checker.py::_warn_non_exhaustive` only inspects the outer constructor name. It can
   incorrectly call `Variant(refutable_payload_pattern)` exhaustive, ignores boolean and nested
   coverage, and does not detect redundant arms.
2. `lower/lowerer.py::_compile_plan` translates each source pattern one-for-one into recursive
   `IrMatchPlan` data. `IrCase` therefore retains the source matching problem instead of a compiled
   decision.
3. `eval/ir_interpreter.py::_try_match` linearly retries arms and recursively interprets plans.
   A subject subterm may be inspected repeatedly, and exhaustiveness/redundancy information is no
   longer available.

The new stage replaces all three. Type checking continues to validate and annotate patterns, but
does not decide coverage. Match compilation consumes its side tables and owns coverage,
redundancy, witnesses, and test ordering. Lowering consumes only successfully compiled matches.

## 4. Settled design decisions

### D1 — Dedicated phase after type checking

Add an explicit match-compilation stage between checking and lowering. It consumes the checked AST
and type metadata, returns a distinct match-compiled program artifact, and never imports the
execution IR. This keeps checking semantic, makes compilation independently testable, and ensures
the REPL check-only path runs the analysis even though it does not link or execute IR.

### D2 — Exhaustiveness and redundancy are static errors

Both a non-exhaustive `case` and a redundant arm reject the program. No implicit failure leaf may
reach executable IR. `MatchError` remains constructible and catchable for explicit source raises.

### D3 — One deterministic exhaustiveness witness

Report one missing structural pattern per non-exhaustive case. Closed domains use concrete nested
constructors/literals. Open domains use a precise remainder description such as “an integer other
than 1 or 2”; do not enumerate an unbounded uncovered space.

### D4 — One generic one-level `IrCase`

`IrCase(subject, arms, default)` dispatches on a closed union of typeless enum-variant and literal
keys. Keys are data, never expressions or patterns. Arm bodies may be `IrCase` expressions, which
form the decision tree. Deep IR validation rejects mixed/incompatible key families.

### D5 — Constructor arms bind demanded immediate fields

An enum-variant arm carries `(field_name, synthetic_symbol)` bindings for only the immediate fields
needed by its subtree. The evaluator extracts those fields after the variant matches; deeper tests
use ordinary `IrLoad`. No occurrence paths or recursive patterns enter IR.

### D6 — `qba` column selection

Choose columns lexicographically by: longest leading constructor prefix (`q`), lowest distinct
branch count (`b`), lowest introduced constructor arity (`a`), then original occurrence order.
Keep the selector behind a small compiler-internal interface, but implement only `qba` now.

### D7 — Immutable DAG sharing without CFG machinery

Memoize compilation states and hash-cons equivalent compiler nodes. Lower each decision node once
and reuse the same immutable `IrExpr` object for repeated references. Sharing is legal only when
free occurrences, action/binder identities, and source provenance agree. Validators and structural
test helpers must use an identity-based visited set.

### D8 — Collect all match-compilation errors

After type checking succeeds, compile every source case in every module. Collect all redundant-arm
errors and one exhaustiveness error per incomplete case, sort them by source location, and produce
no match-compiled artifact when the list is non-empty.

### D9 — Compile source `case` only

Do not route `is`/`is not` or `try` handlers through this compiler. `IrVariantIs` remains the direct
boolean variant predicate; exception handlers retain their nominal dispatch and inheritance rules.

### D10 — Distinct match-compiled artifact

The new artifact wraps the corresponding `CheckedProgram` or `CheckedModuleGraph` and carries a
total `Case.node_id → Decision` mapping. Lowering accepts the match-compiled artifact, not parallel
unchecked maps and not a partially populated `CheckedProgram`.

### D11 — Always bind the root scrutinee

Every source case lowers to an `IrSequence` which first binds the scrutinee to a fresh immutable
symbol, then evaluates the lowered decision DAG using `IrLoad` of that symbol. This preserves
exactly-once evaluation even for a tree that reduces immediately to a leaf.

### D12 — No new pattern syntax

Compile today's pattern union only. The compiler's constructor/signature interface and source
provenance model must permit future or-pattern expansion without changing the matrix algorithm or
execution IR, but this work adds no or-pattern, guard, or view-pattern syntax.

### D13 — Closed enum and boolean signatures

Every enum signature is the complete declaration-order variant set from `TypeTable`; boolean is
the closed `{false, true}` signature. `int`, `decimal`, `text`, `json`, and unresolved type-variable
domains are open and require an irrefutable arm. Apply the same rule recursively to payload fields.

### D14 — Deterministic structural quality tests

Use generated finite matrices and curated paper examples to assert correctness, node/path test
counts, no repeated occurrence test per path, and DAG sharing. Do not add wall-clock thresholds or
a benchmark framework.

### D15 — Complete compilation, without cutoffs

Use memoization and sharing but impose no arbitrary pattern-count, depth, time, or generated-node
limit and no fallback matcher. A future resource policy, if evidence requires one, is separate
language-tooling work.

### D16 — Sharing is local to one source case

Each case owns its memo/hash-cons tables. Do not share decision nodes across modules, cases, or REPL
entries; their actions, binders, source provenance, and lifetimes differ.

## 5. Scope and non-goals

In scope:

- all existing pattern forms at arbitrary nesting depth;
- partial constructor field patterns, normalized with omitted fields as wildcards;
- generic enums after type substitution, imported/prelude enums, and module-qualified nominals;
- literal equality exactly matching current `value_eq`, including int/decimal equality;
- single-file, module-graph, dry-run/check-only, startup-config, and incremental REPL pipelines;
- static diagnostics, IR validation, evaluator simplification, reference docs, and architecture docs.

Out of scope:

- new source patterns, guards, or changes to branch binding scope;
- `is`/`is not`, exception handler dispatch, records as patterns, or collection patterns;
- backtracking automata, matching bytecode, CFG/basic blocks, jumps, or synthesized helper
  functions;
- a separate usefulness checker, pragmatic complexity fallback, or wall-clock benchmark gate;
- IR serialization or cross-case/global DAG interning.

## 6. New phase and package boundary

Create `src/agm/agl/matchcompile/` as a sibling of `typecheck/` and `lower/`:

```text
matchcompile/
  __init__.py       public compile/result entry points
  model.py          patterns, constructors, occurrences, rows, decisions, artifacts
  normalize.py      checked Pattern → canonical rows and signatures
  matrix.py         specialization, defaulting, column scoring
  compiler.py       memoized decision-DAG construction and reachability
  diagnostics.py    failure-path witness reconstruction and source diagnostics
```

The package may import syntax pattern/span nodes, semantic types, `TypeTable`, resolved/checker side
tables, diagnostics, and module IDs. It must not import `agm.agl.ir`, `agm.agl.lower`, the evaluator,
or runtime services. Add it to `tests/test_agl_dependencies.py` with this dependency contract;
allow `lower` to consume its public model.

Conceptual public results:

```python
MatchCompiledProgram(checked, cases)
MatchCompiledModuleGraph(checked_graph, cases_by_module)
MatchCompilationResult(compiled | None, issues)
```

Use concrete frozen dataclasses and closed unions; exact public names may follow local naming style.
Construction succeeds only when issues are empty, and every `Case` reachable in the wrapped
AST must have exactly one decision root. Add an invariant checker used by tests to catch missing,
extra, or cross-program node IDs.

Represent failures inside this package as a closed `MatchIssue` union rather than adding
match-specific fields to the repository-wide `Diagnostic` type:

- `NonExhaustiveIssue(case_node_id, span, witness)`; and
- `RedundantArmIssue(case_node_id, action_id, span)`.

The pipeline adapter converts sorted issues to ordinary error-severity `Diagnostic` records only at
the package boundary. Compiler tests inspect structured issues; pipeline/e2e tests inspect ordinary
diagnostic severity and locations without depending on exact prose.

## 7. Compiler-private model

### 7.1 Constructors and signatures

Normalize refutable heads into a closed constructor union:

- enum constructor: semantic nominal identity, variant, and declaration-order field names/types;
- boolean constructor: `false` or `true`, arity zero;
- literal constructor: canonical typed scalar value, arity zero.

The subject type supplies a `ClosedSignature(constructors)` or `OpenSignature`. Enum signatures
come solely from `TypeTable`, never from constructors observed in the matrix. Literal constructors
must canonicalize according to the subject's equality domain so `1` and `1.0` are not compiled as
distinct branches when runtime `value_eq` considers them equal.

Keep constructor/signature lookup behind a total function over the current semantic `Type` union.
Unsupported future pattern/type combinations must fail as compiler invariant violations rather than
silently becoming open domains.

### 7.2 Occurrences

An occurrence identifies a value available during matching and carries:

- a stable compiler-local ID and deterministic creation order;
- its checked semantic type;
- root or parent-constructor/field provenance; and
- source provenance sufficient for diagnostics.

Occurrences are compiler data only. Lowering maps the root to the mandatory scrutinee symbol and
maps demanded child occurrences to arm-local symbols. No path object appears in IR.

### 7.3 Rows and binders

A matrix row contains a vector of canonical pattern cells, its source action/arm ID, and binder
provenance. Normalize:

- `_` to an unannotated wildcard;
- a real `VarPattern` binder to a wildcard annotated with its pattern node ID/name;
- a resolver-classified bare variant to its nullary enum constructor;
- literals to canonical literal constructors;
- constructor patterns to a constructor with one child pattern for **every** declared field,
  ordered by `TypeTable`; use `ArgumentBindings.constructor_patterns` for supplied fields and insert
  wildcards for omitted ones.

During specialization, move binder annotations into the row's occurrence→binder environment before
discarding a wildcard column. A leaf therefore knows exactly which already-evaluated occurrences to
bind before its body, without embedding source patterns in the decision node. Preserve original row
order through every transformation.

### 7.4 Decisions

Use a minimal closed compiler union:

- `DecisionSwitch(occurrence, keyed_children, default)`;
- `DecisionLeaf(action_id, binder_assignments)`; and
- `DecisionFail`.

`DecisionFail` is legal during compilation and diagnosis but cannot occur in a successful artifact.
A leaf references a source action instead of containing a lowered body, so the compiler remains
independent of IR and the lowerer can lower/cache each action consistently.

## 8. Matrix compilation algorithm

Implement one recursive, memoized `compile(matrix, occurrences)` operation.

1. **Empty matrix:** return `DecisionFail`.
2. **Irrefutable first row:** when every cell in the first row is a wildcard/binder, return a leaf
   for that row's action and finalized binder assignments. Textual priority makes every lower row
   irrelevant on this path.
3. **Choose a column:** score every refutable column with `qba`. Prefer larger leading constructor
   prefix, then fewer distinct head constructors, then lower total introduced arity, then earlier
   occurrence ID.
4. **Specialize:** for every distinct constructor head in deterministic order, specialize the
   matrix. Constructor rows contribute their child patterns; wildcard/binder rows contribute
   arity-many wildcards; incompatible rows disappear. Replace the selected occurrence with fresh
   declaration-order child occurrences and recursively compile.
5. **Default:** if the observed heads do not complete the occurrence's closed signature, or the
   signature is open, build the default matrix from wildcard/binder rows and recursively compile it.
   Omit the default only when observed heads cover a closed signature.
6. **Switch:** return/hash-cons a switch over the selected occurrence.

The transformation must maintain these checked invariants after every specialization/default step:

- matrix width equals occurrence-vector width;
- all rows have equal width and retain source order;
- every cell is compatible with its occurrence type;
- constructor child count and order equal its signature;
- each occurrence is tested at most once along a root-to-leaf path; and
- binder assignments dominate their leaf and refer to available occurrences.

### 8.1 Memoization and hash-consing

Use an immutable canonical compilation-state key containing the normalized matrix, occurrence
interface, and action/binder identities. Memoize recursive results, then intern decision nodes by
their complete semantic key. Do not use source AST object identity as pattern semantics, but retain
source IDs in keys where they affect bindings, leaves, or diagnostics.

The result is acyclic because specialization removes the selected refutable occurrence and only
introduces its structurally smaller child patterns. Assert acyclicity in compiler tests. Do not
mutate an interned node to attach path-specific witness state; reconstruct paths in a separate DAG
walk.

## 9. Diagnostics derived from the decision DAG

After constructing a case DAG:

1. Traverse it with path constraints and an identity-aware visited/state set.
2. Collect every `action_id` reachable at a leaf. Any source arm whose action is absent is
   redundant; diagnose the arm's pattern span. A partially overlapping arm remains useful if any
   leaf reaches it.
3. Collect failure paths. Their presence makes the case non-exhaustive. Choose the first path under
   deterministic constructor ordering and reconstruct one witness at the source case span.
4. Continue compiling other cases/modules and return all diagnostics sorted by source file and
   span.

Witness reconstruction works outward from root/field constraints:

- choose the concrete missing constructor for a finite remainder;
- recursively fill constrained children and use `_` for unconstrained children;
- for boolean, emit the missing boolean literal;
- for an open literal signature, render a symbolic complement (“a text value other than …”) rather
  than inventing an unstable arbitrary value;
- use source-qualified nominal names only where needed for unambiguous display.

Keep witness data structured until diagnostic formatting. This lets a future or-pattern feature or
IDE protocol consume it without parsing message text. Compiler tests assert issue kind, primary
span, action ID, and structured witness—not exact prose.

Remove `_warn_non_exhaustive` from the type checker. The new errors flow through ordinary static
diagnostics on every pipeline surface; none are stored as checker warnings.

## 10. One-level execution IR

Delete `IrWildcardPlan`, `IrBindPlan`, `IrLiteralPlan`, `IrVariantPlan`,
`IrConstructorPlan`, and `IrMatchPlan`. Replace `IrCaseArm.plan` with one-level data:

```text
IrCase
  subject: IrExpr
  arms: tuple[IrCaseArm, ...]
  default: IrExpr | None

IrCaseArm
  key: IrCaseKey
  field_bindings: tuple[(field_name, SymbolId), ...]
  body: IrExpr

IrCaseKey = IrEnumCaseKey(nominal, variant)
          | IrLiteralCaseKey(kind, scalar_value)
```

Use the existing IR enum/value conventions to represent literal kind and scalar payload without
embedding `IrConst*` expressions. A case's explicit arms must all belong to one compatible
discriminant family. Keys are unique under runtime semantic equality, not merely Python dataclass
equality. Literal equality is the existing runtime semantic equality; enum equality checks both
`NominalId` and variant.

`default` is necessary even though source cases are exhaustive: an intermediate switch may route
all constructors/literals not named by earlier rows to a default subtree. A complete finite switch
may omit it. A well-formed compiled program cannot reach “no key and no default”; the evaluator
raises `InvalidIrError` for such malformed IR, never an implicit `MatchError`.

Deep validation must check:

- key union exhaustiveness, uniqueness, and family consistency;
- enum nominal/variant existence;
- field names belong to the selected variant and do not repeat;
- field-binding symbols exist and are immutable/private compiler temporaries;
- subject, arm bodies, and default recursively validate; and
- shared expression nodes are validated once by object identity while cycles remain invalid.

Update IR exports, closed unions, dependency tests, helper constructors, golden fixtures, and the
AST→IR coverage table. Document `IrCase` as a one-level switch, not a source-pattern interpreter.

## 11. Lowering the compiled DAG

Replace `_lower_case`/`_compile_plan` with lowering from the compiled decision root:

1. Allocate a private immutable root symbol.
2. Emit `IrBind(root, lower(source_subject))` exactly once.
3. Maintain `OccurrenceId → SymbolId`, initially root only.
4. Lower a decision switch to `IrCase(IrLoad(occurrence_symbol), ...)`.
5. For an enum arm, allocate/bind only child occurrence symbols in the child's computed free
   occurrence set. Put those bindings in `IrCaseArm.field_bindings`.
6. Lower a leaf to an `IrSequence` of source binder `IrBind`s from occurrence loads followed by the
   already type-directed lowering of the selected branch body. If there are no binders, use the
   body directly.
7. Memoize `Decision object identity → IrExpr` so shared decision nodes remain shared IR objects.

Compute a decision node's free-occurrence interface bottom-up and store/cache it in compiler data.
Hash-consing already requires identical interfaces; lowering must assert that every requested
occurrence has a dominating root or arm binding. Source pattern symbols still use the existing
node-ID-based allocation, preserving lexical references in branch bodies.

The enclosing result is always:

```text
IrSequence(IrBind(root_scrutinee, lower(subject)), lower(decision_root))
```

This remains true for a single wildcard/binder arm whose decision root is a leaf, so effects in the
scrutinee are never skipped.

## 12. Evaluator simplification

Replace recursive `_try_match` with one `IrCase` evaluator arm:

1. evaluate the one-level subject once;
2. select an enum or literal key;
3. on an enum arm, copy the named immediate payload fields into its declared symbols;
4. evaluate that arm body, otherwise the default;
5. raise `InvalidIrError` if neither exists.

Remove the implicit `make_match_error` call and delete `eval/matching.py` if `_describe_value` and
its helper are then unused. Do not remove the prelude `MatchError` type or its normal constructor,
raising, catching, rendering, schema, and documentation support.

## 13. Pipeline integration

Add non-raising helpers parallel to `_run_typecheck`/`_run_typecheck_graph` which convert match
compilation failures into sorted static diagnostics. Thread the successful artifact through every
path that currently passes a checked object directly to lowering:

- single-program `run_prepared` and dry-run;
- graph `run_prepared_graph` and dry-run;
- `discover_params` / `discover_params_graph`, including cached artifacts reused by the subsequent
  run so matching is not compiled twice;
- startup-config graph evaluation;
- incremental REPL graph mode before its check-only early return;
- REPL `type_of`/`:type` after type checking, so a case expression cannot bypass the same static
  exhaustiveness/redundancy rules merely because it is being inspected rather than executed; and
- direct test/development lowering helpers.

Update `ParamDiscovery`, `StartupConfigResult`, and optional prechecked pipeline parameters to carry
the match-compiled artifact where later lowering reuses it. Do not retain a checked-only bypass that
can reach `lower_program` or `lower_graph`. The match-compiled graph wraps the exact checked graph,
so imported-module case errors surface even when the entry module does not execute those cases.

Lowering entry points become conceptually:

```text
lower_program(MatchCompiledProgram, ...)
lower_graph(MatchCompiledModuleGraph, ...)
lower_repl_graph(MatchCompiledModuleGraph, ...)
```

## 14. TDD implementation sequence

Follow the milestones in order. For every behavioral defect exposed during implementation, add the
failing regression test before the fix.

### M1 — Reference matcher and matrix model

Write failing unit tests for normalization and a test-only simple source-pattern reference matcher.
Cover all current pattern forms, omitted constructor fields, imported/generic enum signatures,
numeric literal equality, binder provenance, and closed/open signatures. Then add the compiler model
and normalization code.

Gate: normalized rows preserve source priority and have widths/child arities consistent with their
occurrences.

### M2 — Specialization/default and `qba`

Write table-driven failing tests from the papers plus AgL nested enum/literal matrices. Test exact
specialized/default matrices and deterministic `qba` choices, including tie-breaking. Implement the
pure matrix operations and selector.

Gate: specialization/default partitions the reference value space without loss or overlap relevant
to first-match actions.

### M3 — Decision compilation, sharing, and diagnostics

First add failing tests for leaves, switches, source priority, nested exhaustiveness, boolean
exhaustiveness, open-domain incompleteness, duplicate/subsumed arms, and partially useful arms.
Add generated small finite enum/boolean matrices and compare every value's selected action with the
reference matcher. Then implement recursive compilation, memoization, hash-consing, reachability,
and witness reconstruction.

Gate:

- reference and decision action agree for every generated value;
- `Fail` exists iff some generated value is unmatched;
- reachable actions equal useful source arms;
- no path tests an occurrence twice;
- curated matrices meet expected node/path counts and demonstrate shared object identity.

### M4 — Match-compiled artifacts and pipeline error plumbing

Write failing single, graph, imported-module, dry-run, parameter-discovery, startup-config, and REPL
check-only tests. Implement the new artifact/result types and stage calls. Replace the old
enum-only warning tests with static-error tests, including multiple errors sorted across cases and
modules.

Gate: no invalid program reaches lowering; no valid path compiles a case twice when a cached
artifact is supplied.

### M5 — One-level IR

Write failing IR construction/validation tests for enum/literal keys, defaults, immediate field
bindings, invalid mixed/duplicate keys, bad variants/fields/symbols, DAG traversal, and cycle
rejection. Replace the recursive plan nodes and update exports/unions/validators.

Gate: the IR package contains no recursive pattern-plan type and imports no frontend package.

### M6 — DAG-to-IR lowering

Write failing structural golden tests before changing `_lower_case`: mandatory root binding, nested
one-level cases, demanded-field-only bindings, leaf binder sequences, defaults, source body
selection, and preserved DAG identity. Implement lowering and delete `_compile_plan`.

Gate: traversing lowered case IR finds only one-level keys; source `Pattern` objects are absent from
the executable graph.

### M7 — Evaluator migration

Write evaluator tests directly over hand-built valid/invalid one-level cases, then implement the new
case dispatch and remove `_try_match`. Convert existing case runtime tests to the new static rule:
non-exhaustive sources fail before evaluation; explicit exhaustive `_ => raise MatchError(...)`
still raises and can be caught.

Gate: evaluator imports and dispatch contain no `IrMatchPlan`; all existing exhaustive case
workflows retain their values and effects.

### M8 — Structural quality and full workflow coverage

Expand generated matrices to multiple columns and nested constructors within practical deterministic
sizes. Add e2e cases for first-row priority despite `qba` reordering, subject effects exactly once,
binders in default rows, recursive/generic enums, imported nominals with same variant spellings, and
all redundant/non-exhaustive shapes.

Gate: 100% source/branch coverage remains, 100% command coverage remains, and structural quality
assertions are stable under parallel tests.

### M9 — Documentation and cleanup

Delete obsolete warning code, plan nodes, evaluator helpers, tests, imports, and comments. Update:

- `docs/arch/agl/index.md`, `frontend.md`, `execution.md`, `modules.md`, and `repl.md` for the new
  phase/artifact and one-level IR;
- `docs/agl/reference/pattern-matching.md`, `control-flow.md`, `expressions.md`, and `exceptions.md`
  to state that cases must be exhaustive/non-redundant and that `MatchError` is explicit only; and
- relevant module docstrings and the execution IR AST→IR coverage table.

Gate: repository search finds no claim that non-exhaustiveness is advisory or implicitly raises
`MatchError`, and no reference to recursive match plans remains.

### M10 — Final verification

Run focused tests throughout with `uv run pytest ...`, then finish with:

```text
just check
```

Do not suppress Ruff, mypy, coverage, or test failures.

## 15. Acceptance criteria

The implementation is complete only when:

- every valid source case compiles through the new explicit stage;
- every non-exhaustive case is rejected with one deterministic structured witness;
- every fully redundant source arm is rejected, while partially useful arms are accepted;
- all match-compilation diagnostics across a type-correct program are returned in source order;
- enum and boolean signatures are complete and open literal domains require a catch-all;
- source priority is preserved for every tested/generated matrix regardless of selected column;
- a scrutinee and each tested occurrence are evaluated/tested at most once per execution path;
- `qba` selection and per-case hash-consing have deterministic structural tests;
- IR cases contain only one-level keys, demanded immediate field bindings, bodies, and defaults;
- no recursive match-plan class or evaluator exists;
- implicit case failure cannot raise `MatchError`; explicit source raising still works normally;
- single-file, module graph, imports, generics, startup config, dry-run, and REPL/check-only paths
  share the same compiler contract;
- architecture and language reference docs describe the implemented system; and
- `just check` passes with 100% coverage and no analysis suppression.
