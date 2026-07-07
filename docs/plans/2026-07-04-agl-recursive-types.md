# Plan: AgL recursive types (records, enums, exceptions)

## Overview

AgL supports recursive type declarations for records, enums, and exceptions.
This plan records the work that lifted the earlier ban, replacing acyclic body
ordering with nominal handles, table-backed shape lookup, and whole-table
analyses for inhabitation, equality, and finite-schema boundaries:

```agl
enum Tree
  | Leaf
  | Node(value: int, left: Tree, right: Tree)

record Category
  name: text
  subcategories: list[Category]

enum Expr[T]
  | Lit(value: T)
  | Add(lhs: Expr[T], rhs: Expr[T])
```

The enabling change is representational. Today a record/enum embeds its field
and variant types **by value**: every referenced type is fully built before it
is captured, so a `Type` is a self-contained finite tree — a representation
that physically cannot express a cycle. The migration makes nominal types
**handles**: `RecordType`/`EnumType`/`ExceptionType` carry only their identity
`(module_id, name, type_args)` (equality is already nominal today), while
definitions — parameter lists and field/variant type *templates* — live in a
shared **type table**. A reference to a nominal type inside another type's body
is just another handle. Types remain finite immutable trees, so every existing
and future type walker terminates by construction; recursion becomes a
property of the table's reference graph, not of the Python object graph.

Whole-type semantic questions (is this type inhabited? does it support
equality? does it have a finite schema?) are answered by **declaration-level
analyses** over the finite declaration graph, computed once per program /
module graph / REPL entry. Only the agent/JSON boundary (schema and decode
derivation) ever expands concrete instantiations, and it is guarded by a
finiteness pre-check.

The runtime needs almost nothing: values are finite trees, the execution IR is
typeless, and `ExceptionValue`/`RecordValue`/`EnumValue` already key identity
on `NominalId` (module + name). The work is concentrated in `semantics/types`,
`typecheck/`, `type_schema.py`, and the decode contracts.

## Resolved Owner Decisions

These were resolved with the owner and frame the implementation.

| # | Decision |
|---|----------|
| D1 | **Inhabitation check.** A recursive type declaration is legal iff it has at least one finite value. Uninhabitable declarations (`record R { next: R }`, `enum E { A(e: E) }`) are rejected at the declaration with an "uninhabitable type" error. Recursion guarded by an enum base-case variant or by a `list`/`dict` field is legal. |
| D2 | **By-reference representation + shared type table.** Nominal heads are lightweight handles; field/variant type templates live in a table keyed by `(module_id, name)`. No cyclic Python object graphs; no by-value embedding. |
| D3 | **Full polymorphic recursion.** Generic recursive types are supported with no uniformity restriction: a body may reference its own type at different arguments (`Perfect[Pair[T, T]]` inside `Perfect[T]`). |
| D4 | **Cross-module cycles are allowed.** A type cycle may span any modules, consistent with import cycles (already legal) and same-module mutual recursion. |
| D5 | **JSON Schema uses `$defs`/`$ref` only for recursive types.** Instantiations that participate in a cycle of the concrete instantiation graph get `$defs` entries and are referenced via `$ref`; everything non-recursive stays fully inlined exactly as today. The decode-schema mirror gets a matching reference node + defs table. |
| D6 | **No-finite-schema types error at the use site.** A type whose reachable instantiation closure is infinite (growing polymorphic recursion) is legal and fully usable in-language, but using it where a finite schema is required — agent output target, `as`/`as?` cast target — is a typecheck error at that use site. |
| D7 | **Exceptions are unified with records.** `ExceptionType` becomes nominal (`module_id`, `name` identity; fields excluded from equality), exception recursion is permitted under the same rules, and record type-checking / scoping / lowering logic is reused for exceptions as much as possible. Exceptions keep their hierarchy semantics (abstract `Exception` root, `extends` base) and remain non-generic. |
| D8 | **Deep-value `RecursionError` is out of scope.** Pathologically deep values surface as Python's `RecursionError`, exactly as deeply nested lists already can today. No depth guards or iterative rewrites in this feature. |

Forced (not a decision): **recursive type aliases stay banned.** An alias is
transparent and has no nominal identity to anchor a cycle; recursion must pass
through a named record/enum/exception. The existing alias-cycle error remains.

### D1 — Inhabitation

Computed as a least fixpoint over the type table after collection, at the
declaration level:

- scalars, `json`, `unit`, `agent`, function types: inhabited;
- `list[T]` / `dict[T]`: always inhabited (the empty collection);
- type variables: treated as inhabited (arguments are themselves types that
  passed this check, and every occurrence is in a positive position, so the
  declaration-level abstraction is exact);
- a record/exception is inhabited iff all its field templates are;
- an enum is inhabited iff at least one variant has all fields inhabited;
- a nominal handle is inhabited iff its declaration is.

Start with every declaration marked uninhabited and iterate to fixpoint; any
declaration still uninhabited is rejected with an error at its declaration
span (e.g. *"Record type 'R' is uninhabitable: every value of 'R' would be
infinite. Recursion must be guarded by an enum base-case variant or a
list/dict field."*).

### D2 — Handles + type table

`semantics/types.py`:

- `RecordType` / `EnumType` drop their embedded `fields` / `variants`
  mappings entirely. A nominal type is `(name, type_args, module_id)` — the
  identity that equality and hashing already use. The "shell vs built form"
  distinction disappears: every handle is always valid.
- `ExceptionType` gains `module_id` and the same handle shape (D7).

A new `TypeTable` (new module in `semantics/`, the foundation layer) maps
`(module_id, name)` to a `TypeDef`: kind (record/enum/exception), type
parameter names, field/variant type *templates* (finite trees over scalars,
containers, `TypeVarType`, and nominal handles), and exception-specific
metadata (`abstract`, base). The table provides memoized lazy instantiation —
`record_fields(handle)`, `enum_variants(handle)`, `exception_fields(handle)` —
substituting `type_args` into the templates. Substitution walks finite trees
and needs no cycle guards.

Threading: the `TypeEnvironment` (`typecheck/env.py`) owns the table in
single-module mode; graph mode shares one global table across all per-module
environments (subsuming today's `graph_type_table`, whose values become
handles). Consumers that currently read `.fields`/`.variants` directly —
checker member access, pattern typing, constructor signatures, comparability,
`type_schema.py`, decode-schema building in lowering — switch to table
lookups. This is a wide but mechanical refactor; the type table travels with
the objects that already carry type context (env in the checker, an explicit
parameter for `type_schema.py` and lowering).

Consequences worth naming:

- `substitute` / `free_type_vars` / `contains_type_var`
  (`semantics/types.py`) operate on finite trees again — no guards needed;
  `substitute` no longer expands nominal bodies at all (it rewrites
  `type_args` on handles).
- The `_building` in-progress set, the shell-replacement dance in
  `builder.py`, and the topological body-resolution + `_CycleInTypeDeps`
  machinery in `graph.py` are deleted, not adapted. Body resolution no longer
  needs referenced types built first — resolving a body just produces handles;
  only *headers* (name, kind, arity) must be pre-collected, which phase 1
  already does.
- Whole-type capability queries that today recurse through embedded fields
  (`_has_no_value_equality` behind `comparable_types`) become
  declaration-level precomputed flags on the table (fixpoint: does this
  declaration's body, excluding parameter positions, contain a
  function/agent type or reach a declaration that does?) combined with a
  finite walk of the concrete handle's `type_args`. Exact, and never expands
  instantiations.

### D3 + D6 — Polymorphic recursion and the finiteness analysis

The checker imposes no uniformity restriction on self/mutual references. The
consequence: a concrete type's **instantiation closure** (all concrete
`(decl, args)` instantiations reachable by expanding fields) can be infinite —
e.g. `Perfect[int]` reaching `Perfect[Pair[int, int]]`,
`Perfect[Pair[Pair[…]]]`, … Such types have finite values and work everywhere
in-language; they only lack a finite schema.

Finiteness is decided at the declaration level, once per table build:

1. Build the declaration reference graph (which declarations reference which,
   with the argument templates of each reference) and its SCCs.
2. Within each SCC, build the parameter-dependency graph: an edge `q → p`
   whenever a reference's argument template for parameter `p` contains
   parameter `q`; the edge is **growing** when `q` occurs as a proper subterm
   (under at least one constructor, e.g. `T` inside `Pair[T, T]`).
3. An SCC has an infinite closure iff its parameter-dependency graph contains
   a cycle with at least one growing edge. Permutation cycles
   (`Swap[B, A]` inside `Swap[A, B]`) and argument-constant references
   (`R[int]` inside `R[T]`) stay finite and legal at every boundary.

Each declaration gets a precomputed `finite_closure` flag. At a schema-needing
use site (agent output target, cast target), the check is: every declaration
reachable from the root type has `finite_closure`. Otherwise the use site is
rejected with e.g. *"type 'Perfect[int]' cannot be used as an agent output
type: its recursive instantiations never close, so it has no finite JSON
schema."* Everything else about such types (constructors, matching, equality,
rendering, passing to functions) works normally.

### D4 — Cross-module cycles

With handles, the module boundary is irrelevant to representation: a handle
carries its `module_id` and the table is global to the graph build. The graph
pass changes from "toposort type bodies, reject cycles" to "collect all
headers across the graph, then resolve all bodies in any order, then run the
declaration-level analyses (inhabitation, finiteness, capability flags) over
the whole table". The cross-module structural-cycle error and its tests are
removed. Import-graph handling (`scope/`, module loading) is untouched —
import cycles were already legal.

### D5 — Schema and decode derivation (`type_schema.py`, `ir/contracts.py`)

`derive_schema(typ)` becomes table-aware and works in two steps:

1. **Instantiation graph.** Expand the concrete instantiation graph reachable
   from the root (memoized on handles; guaranteed finite by the D6 pre-check
   at the use site). Compute its SCCs. Instantiations in non-trivial SCCs or
   with self-loops are *recursive for this root*.
2. **Emission.** Recursive instantiations get entries under `$defs`, keyed by
   a canonical, schema-safe name derived from the qualified type display form
   (e.g. `Tree`, `mod.Tree_int` — deterministic, collision-free within one
   schema); references to them emit `{"$ref": "#/$defs/<key>"}`. Everything
   else is inlined exactly as today, so existing schemas, prompt
   format-instructions, and their tests are byte-for-byte unchanged for
   non-recursive types.

The `jsonschema` Draft 2020-12 validator used by `runtime/convert.py`
supports `$ref`/`$defs` natively; validation needs no changes.

The typeless decode mirror gets the same treatment: `ir/contracts.py` adds a
`RefDecode(key)` node to the `DecodeSchema` union plus a `defs` mapping
carried on the output contract / cast recipe; `build_decode_schema` mirrors
the emission decisions above; `decode_value` (`runtime/convert.py`) resolves
`RefDecode` through the defs mapping while walking the (finite) value.
`ParamDecoder`-based CLI/config parameter decoding reuses the same mechanism,
so recursive-typed `param`s work too.

### D7 — Exception unification

- `ExceptionType` becomes a nominal handle like records: gains `module_id`,
  drops structural field equality, moves its fields into the table as a
  `TypeDef`. The runtime is already there: `ExceptionValue` keys identity on
  `NominalId(module, name)` today, so no evaluator change.
- `_TypeBuilder`'s exception path is folded into the record path wherever they
  duplicate each other (field resolution, duplicate-field checks, constructor
  field-kind registration); what stays exception-specific is the hierarchy
  (`extends` base resolution, implicit `message`/`trace_id` fields, `abstract`
  root handling) and catchability.
- Exception recursion is permitted under D1/D3/D5/D6 like records.
  (Exceptions are product types, so direct self-reference by a required field
  is uninhabitable; in practice recursive exceptions need list/dict guards.)
- Audit for structural-equality dependence: any checker/lowering site that
  compares `ExceptionType`s or relies on `fields` participating in equality
  (catch-clause typing, exception assignability, prelude exception tables)
  switches to nominal comparison + table lookup. Built-in exceptions keep
  their prelude/std module ids.
- Exceptions remain **non-generic** — no type-parameter syntax is added.

### Unaffected / near-unaffected

- **Runtime values and evaluator**: `RecordValue`/`EnumValue`/`ExceptionValue`
  are finite trees keyed by `NominalId`; rendering, pattern matching, value
  equality, and the typeless IR are untouched (beyond the decode `defs`
  threading).
- **Assignability**: already nominal for records/enums; unchanged.
- **Scope pass**: constructors are already ordinary value-namespace bindings;
  unchanged apart from any shared record/exception collection plumbing.
- **REPL**: entries seed the accumulated table; type redefinition remains
  blocked ("use `:reset`"), so there is no versioning problem for recursive
  types. Recursive declarations within a single entry and references to
  earlier entries' types both work through the table.

## Documentation

Per `src/agm/agl/CLAUDE.md`, every language change lands with reference
updates, written user-side with no implementation references:

- `docs/agl/reference/types.md` — recursive and mutually recursive records
  and enums; the inhabitation rule and the uninhabitable-declaration error;
  aliases cannot be recursive; nominal type identity restated as needed.
- `docs/agl/reference/generics.md` — recursive generic types; polymorphic
  recursion; the finite-schema boundary rule (which types can cross the
  agent/cast boundary and the use-site error when they cannot).
- `docs/agl/reference/exceptions.md` — exception identity/unification as far
  as it is user-visible; recursive exception declarations.
- `docs/agl/reference/agent-calls.md` (and cast docs in `expressions.md` if
  applicable) — the no-finite-schema use-site rule at agent output and cast
  targets.
- `docs/arch/agl/frontend.md` — handle + type-table model, declaration-level
  analyses; `docs/arch/agl/execution.md` — decode defs;
  `docs/arch/agl/modules.md` — graph type pass without body toposort.

## Testing

TDD throughout: each milestone starts by writing its failing tests. Per
`src/agm/agl/CLAUDE.md`, e2e test programs under `tests/agl/programs/` are the
first artifacts written for the language-visible milestones (M3/M4), before
implementation.

- **E2e programs** (`tests/agl/programs/`): build/traverse/match a recursive
  enum tree; recursive record with list/dict guards; mutual recursion (incl.
  record↔enum); cross-module recursive types; generic `Tree[T]` used at
  several instantiations; a polymorphically recursive (growing) type used
  purely in-language; agent call returning a recursive type (mocked agent,
  `$defs` schema round-trip, lenient + strict); `as`/`as?` casts to recursive
  types incl. failure paths; recursive exception thrown/caught; error
  programs: uninhabitable record/enum, no-finite-schema type at an agent
  output and at a cast target, recursive alias. Multiple scenarios per
  program (varied inputs and mock responses).
- **Typecheck unit tests**: inhabitation fixpoint (accept/reject matrix incl.
  mutual and generic cases), finiteness analysis (uniform, permutation,
  argument-constant, growing; direct and mutual), cross-module cycles,
  declaration-level capability flags (equality gating for recursive types
  containing functions), exception nominal equality and shared-path behavior.
- **Schema/decode unit tests**: `$defs`/`$ref` emission (golden schemas),
  non-recursive schemas byte-identical to today, canonical `$defs` key
  scheme incl. distinct generic instantiations, `RefDecode` round-trips,
  validator acceptance/rejection of recursive JSON.
- **Migration guardrail**: the entire existing suite stays green after M1/M2
  with zero language-visible changes; existing recursion-ban tests
  (`test_agl_typecheck.py`, `test_agl_typecheck_graph.py`,
  `test_agl_stdlib.py`) are replaced in M3 by the new error semantics.
- **REPL tests**: recursive type declared and used across entries;
  redefinition still blocked.
- 100% coverage of `src/` maintained; `just check` gates every milestone.

## Milestones

Each milestone is committed separately with `just check` green.

1. **M1 — Handle + table representation migration.** No language change.
   Introduce `TypeTable`/`TypeDef` in `semantics/`; strip embedded
   fields/variants from `RecordType`/`EnumType`; convert all consumers to
   table lookups; delete the shell dance, `_building` set, and graph-pass
   body toposort, replacing the recursion ban with a temporary explicit
   "any cycle in the declaration graph" rejection so existing ban tests and
   behavior are preserved bit-for-bit. Entire suite green.
2. **M2 — Exception unification.** `ExceptionType` → nominal handle with
   `module_id`; exception defs into the table; fold exception building into
   the record path; audit structural-equality call sites. Still no language
   change (temporary cycle rejection covers exceptions). Suite green.
3. **M3 — Recursion in-language.** Write the in-language e2e programs and
   typecheck tests first (failing). Replace the temporary cycle rejection
   with the D1 inhabitation fixpoint and the D3/D6 finiteness analysis;
   allow same-module, mutual, generic, and cross-module cycles; new
   diagnostics; declaration-level capability flags. Recursive types fully
   work for construction, matching, equality, rendering. Reference docs for
   types/generics/exceptions updated in the same commit.
4. **M4 — Agent/JSON boundary.** Write the boundary e2e programs and
   schema/decode tests first (failing). `$defs`/`$ref` emission over the
   concrete instantiation graph; `RefDecode` + defs in `ir/contracts.py`,
   lowering, and `decode_value`; no-finite-schema use-site errors at agent
   output and cast targets; recursive `param` decoding. Reference docs for
   agent-calls/casts updated in the same commit.
5. **M5 — Docs and architecture sweep.** `docs/arch/agl/*.md` brought in line
   with the final shape; final audit that no stale "recursive types are not
   supported" wording remains in code comments, help texts, or reference
   docs.

## Implementation notes / derived decisions (flagged for owner review)

- **Temporary M1 cycle rejection.** M1 deliberately re-implements the current
  ban as a trivial declaration-graph cycle check so the representation
  migration is behavior-neutral and separately verifiable. It is deleted two
  milestones later.
- **`$defs` key scheme.** Keys derive from the canonical qualified display
  form of the instantiation (module-qualified only when needed for
  uniqueness), sanitized to JSON-Schema-safe identifiers. Deterministic and
  stable across runs; exact scheme settled during M4 with golden tests.
- **Finiteness analysis exactness.** The growing-edge-cycle criterion is
  believed exact for AgL's reference structure (references are syntactic
  substitutions). Should an edge case surface where it is conservative, the
  failure mode is a spurious use-site error (never unsoundness); such a case
  would be documented and revisited.
- **Capability flags replace deep walks.** `comparable_types` and any similar
  whole-type predicate consult per-declaration fixpoint flags. Any future
  whole-type predicate must follow the same pattern rather than expanding
  instantiations.
- **Exceptions stay non-generic** and keep hierarchy semantics; unification
  is about identity and shared machinery, not about making exceptions
  records.

## Out of scope (future work)

- Depth guards or iterative rewrites for value walkers (D8) — pathological
  depth remains a Python `RecursionError`, as today.
- Approximate schemas / lazy decode for infinite-closure types — such types
  simply cannot cross the schema boundary (D6).
- Generic exceptions.
- REPL type redefinition/versioning — redefinition remains blocked behind
  `:reset`.
