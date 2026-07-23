# AgL Named Scopes — Implementation Plan

Status: planned · Date: 2026-07-23 · **Every owner decision in this plan is resolved.**

## 1. Goal

Add user-defined named scopes to AgL: nestable, extendable namespaces declared inside modules,
kept separate from module identity. Both of these forms declare a function in scope `Point`, may
coexist in one file, and are fully equivalent:

```agl
scope Point

def distance(p1: Point, p2: Point) -> decimal =
  let dx = p1.x - p2.x
  let dy = p1.y - p2.y
  sqrt (dx * dx + dy * dy)

end Point

def Point::distance'(p1: Point, p2: Point) -> decimal =
  let dx = p1.x - p2.x
  let dy = p1.y - p2.y
  sqrt (dx * dx + dy * dy)
```

Scopes unify with the existing type-qualification of constructors: a type declaration implicitly
establishes its same-named scope, `Option::None` becomes ordinary scope-member access, and the
constructor-qualification machinery is absorbed into one general qualifier-chain resolution
mechanism. The unification is both semantic **and** implementational — after this change there is
one qualifier resolver, not a scope resolver next to a constructor resolver.

All solutions must be general, principled, and extensible; no special cases keyed on particular
source shapes.

## 2. Resolved owner decisions

| # | Decision |
|---|----------|
| D1 | **Full unification.** Every type declaration implicitly establishes a same-named scope; enum variants are its members; `scope T` blocks and `def T::f` shorthand extend any scope, including type-named ones. One qualifier namespace per module, one resolution path. The implementation is unified as well: constructor qualification is reimplemented on top of general scope-member resolution. |
| D2 | **Flat regions with mandatory matching closer.** `scope NAME … end NAME`; the closer name must match the innermost open scope. `scope` and `end` are contextual soft keywords, promoted positionally like the module-header keywords. |
| D3 | **Full nesting.** Textually nested blocks, multi-segment headers (`scope A::B … end A::B`), and chained shorthand (`def A::B::x`) are three spellings of one scope-path mechanism, arbitrary depth. |
| D4 | **Static declarations only.** A scope region may contain functions, externs, types, and agents (agents wherever the enclosing module kind allows them). No statements, `let`/`var`, `infix`, `import`/`export`, `program`, or `param` inside scopes. Path shorthand applies to all name-headed declaration forms. |
| D5 | **Membership-based visibility.** The scope interior is a lexical layer for both the block form and the shorthand form: sibling members are visible bare, nearest-enclosing layer wins on collision, and `::name` keeps its module-anchor meaning as the escape hatch to the module root. |
| D6 | **Exact scope paths from outside.** References use the full scope path relative to the referencing position. Suffix resolution remains a module-route feature only; scope segments never suffix-match. No bare outside access by default. |
| D7 | **Scope opening mirrors import syntax.** A new `open <scope-ref> [using item, … | hiding name, …]` item injects members bare into its enclosing region. Plain `open X` bares all members, `using` bares only the listed ones (with canonical `as` renames), `hiding` bares all but the listed ones; at most one clause. Unlike imports there is no plain non-`open` form (qualified routes exist automatically), so `open X using …` is valid. The scope ref accepts a module route (`open geo::Point`). |
| D8 | **Path atoms at module boundaries.** A scoped declaration's public name is its full scope path. Import/export clauses accept member paths and scope names; a scope name selects its whole subtree; renames re-root (`using Point::distance as d` yields top-level `d`; `using Point as P` yields `P::…`). "Bare" keeps meaning module-bare: the scope path is part of the name. |
| D9 | **Member-level privacy only.** `private` on individual members keeps its module-boundary meaning (`Point::helper` invisible to importers, visible module-wide). No `private scope` bulk form; no scope-internal visibility dimension. |

Explicitly rejected alternatives: separate scope/type entities behind shared syntax; indented
scope bodies; suffix resolution across scope segments; flat opaque scoped-name strings in
selection sets; scope-internal privacy.

## 3. Language specification

### 3.1 Regions and headers

- `scope <path>` opens a region at the module root or inside another scope region; `end <path>`
  closes it. The closer path must match the innermost open header exactly (`scope A::B` closes
  with `end A::B`). Mismatched, missing, or extra closers are parse-time errors that name the
  expected closer.
- A multi-segment header `scope A::B` is equivalent to `scope A` containing `scope B`, closed by
  one `end A::B`.
- Scope regions may repeat: multiple blocks and shorthand declarations with the same path all
  extend one scope. A scope needs no prior declaration — its first mention creates it.
- `scope` and `end` are contextual soft keywords promoted at item start (`scope` when followed by
  a scope path, `end` only while a scope region is open). Outside these positions both remain
  ordinary identifiers. Promotion of `end` requires the lexer to track open-region depth, in the
  same stateful style as the existing module-header window.

### 3.2 Namespace model

- A **scope path** is a tuple of NAME segments. Declarations are identified by
  `(ModuleId, scope_path, name)`; the empty path is the module root. All existing root
  declarations are the zero-length-path case — root handling must not become a special branch.
- **Types are scopes.** A type declaration at path `p` with name `T` establishes scope `p + (T,)`.
  Enum variants are members of it; record/exception construction stays the bare type name
  (`Point(1, 2)`), unchanged. A type declared inside a scope owns a nested scope (`A::T::V`).
- **One namespace per path.** Each `(ModuleId, scope_path)` has a single declaration namespace,
  exactly as the module root does today. A `scope X` mention merges with a same-path type `X`
  (they are one entity) and collides with any other same-path declaration named `X`
  (function, agent, extern) as a declaration-time duplicate error. Same-path duplicate members
  are declaration-time errors; the existing constructor-collision nuance (a bare spelling yields
  when the constructor stays reachable another way) is preserved per path.
- **Cross-module scopes never merge.** Scope identity is per-module. Two modules each declaring
  scope `Point` coexist; `Point::distance` at a use site resolves if exactly one selected module
  contributes that path, and collisions are ambiguity-at-use errors repaired with a module route —
  the existing bare-name rules applied to path atoms.

### 3.3 Qualifier chains and resolution

- The reference grammar generalizes from `qual_prefix type_qual? name` to a uniform chain: an
  optional anchor (`/` module anchor or `::` current-module anchor), followed by qualifier
  segments, followed by the member name. Lexically each tight `NAME(/NAME)*::` run is already one
  `MODQUAL` token, so chains arrive as a `MODQUAL` sequence; only the grammar and AST change.
- **At most the leading segment is a module route.** A slash-multi segment (`utils/geo::…`) is
  always a module route. A single-name leading segment may be a module suffix/alias or a local
  scope/type; if both interpretations resolve, that is a hard clash error with the existing repair
  guidance (`/`-anchor for the module reading, `::`-anchor for the local reading) — the
  generalization of today's type-name-vs-module-route clash.
- **Type application on segments** (`Option[int]::None`) is permitted exactly when the segment
  names a type; applying type arguments to a plain scope segment is a static error. The applied
  form is resolved by the same chain resolver, not a separate rule.
- **Inside a scope**, resolution walks the lexical layers: innermost scope members, enclosing
  scope members, module root, then imported bare names — nearest layer wins. The shorthand form's
  body resolves in its scope's layer regardless of textual placement. `::name` anchors at the
  module root; `::A::B::x` is an absolute in-module path.
- **Outside a scope**, the full path relative to the referencing position is required. Module
  routes keep their existing suffix/alias/anchored behavior and are filtered by which module
  contributes the requested path atom.
- Bare enum-variant references keep working through the existing constructor-candidate mechanism
  (single candidate resolves; overloads demand qualification). That mechanism now keys candidates
  by scoped owner paths but is otherwise unchanged.

### 3.4 The `open` item

- Grammar: `open_decl ::= "open" scope_ref [using_clause | hiding_clause]` with clause shapes
  identical to imports; `scope_ref ::= [module-route "::"] scope_path`.
- Semantics: contributes the selected members **bare** to the enclosing region (module root or
  scope region), for the whole region, with collisions detected at use — the same
  bare-contribution model the import environment already implements. `using N as M` renames
  canonically; renames re-root to plain names.
- Placement mirrors imports: `open` items must precede the non-header items of their region.
  Opening a type-named scope injects its variants and extension members alike.

### 3.5 Modules, selection, and privacy

- Public names, `decl_info`, export sets, and selection sets become path-keyed. `using`/`hiding`
  clauses accept paths; a scope name in a clause selects (or hides) its subtree, including the
  same-named type when one exists. Re-exports forward path atoms with unchanged origin-identity
  rules.
- `open import` bares selected path atoms at the module level: `Point::distance` becomes writable
  without a module route. Full bareness composes with the scope-level `open` item.
- A scope path is externally reachable iff it has at least one public member in its subtree or is
  a public type; selecting a path with no public content is the existing
  selection-of-non-public-name error.
- `private` members are excluded from export sets, selection, and cross-module `open`, and remain
  visible module-wide.

### 3.6 REPL

Scope regions are entry-local syntax: a region must close within its entry. Membership
accumulates across entries exactly like other REPL declarations — a later entry declaring
`Point::distance` replaces the retained member of that name and extends the scope otherwise.
`open` items persist like retained imports and are cleared by `:reset`.

### 3.7 Derived rules (owner-visible consequences, not new decisions)

These follow from the decisions above plus existing invariants; they are listed so nothing lands
silently:

- **Scoped externs**: the co-located Python symbol for `extern def A::f` is the member name `f`;
  two scoped externs in one module that map to the same Python symbol are a declaration-time
  error.
- **`builtin` declarations remain root-only** (std-internal forms; nothing needs them scoped).
- **`infix` declarations remain root-only**: an infix operator cannot be spelled qualified at an
  infix use site, so a scoped one would be unreachable; revisit if scope opening makes it useful.
- **Typed calls** (`f::[T]`) compose at the end of a chain (`Point::distance::[T]` if ever
  applicable); the existing "`NAME :: [` is not a qualifier" lexer rule is preserved.

## 4. Non-goals and invariants

- No `open` interactions beyond §3.4 — no re-exportable opens, no transitive opening.
- No `private scope` bulk sugar and no scope-internal visibility (rejected under D9).
- No suffix matching of scope segments (rejected under D6).
- No statements or value bindings in scopes (rejected under D4).
- Do not keep a parallel constructor-qualification resolver. `TypeQualifier` special-casing and
  the dedicated qualified-constructor side tables are absorbed by chain resolution; the
  typechecker's remaining constructor duty is validating variant-ness and signatures, not
  re-resolving names.
- Nominal type identity stays structured — extend the existing `(module, name)` keys with the
  scope path; never encode paths by string concatenation in identity keys.
- Function/symbol erasure in the IR is unchanged: linked integer handles, no name-based calls.
- The case-neutral namespace principle holds: scope segments are NAMEs; capitalization never
  affects resolution.
- Existing programs that use no scopes must resolve, typecheck, lower, and run identically.

## 5. File-level implementation map

### Lexer — `src/agm/agl/lexer/`

- `tokens.py`: add contextual `SCOPE`/`END` token kinds alongside the module-header tokens.
- `lexer.py`: promote `scope` at item start when followed by a scope path; track open-region
  depth to promote `end`; reuse the `_promote_soft_keywords` state machinery. `_merge_modqual`
  already emits chain-friendly `MODQUAL` runs — verify multi-`::` chains and keep the
  spaced-qualifier advisories working for every segment.

### Grammar and AST — `src/agm/agl/grammar/agl.lark`, `src/agm/agl/syntax/`, transform

- Add `scope_region` (header, items, closer) and `open_decl` rules; reuse the import clause
  rules for `open`.
- Generalize `qual_var_ref`/`applied_qual_ref`/`pattern_atom`/qualified type references and
  `is`-tests from "prefix + optional type qualifier" to a segment chain. Introduce a single
  `QualifierChain` AST form (anchor kind + segments, each segment a name with optional type
  arguments) replacing the `Qualifier` + `TypeQualifier` pair everywhere a reference is
  qualified; keep node spans precise per segment for diagnostics.
- Extend name-headed declaration heads (`def`, `extern def`, `record`, `enum`, `exception`,
  `type`, `agent`) to accept a scope-path prefix; the transform normalizes both spellings into
  one "declaration at scope path" representation, so later passes never see two forms.
- Import/export clause items accept paths.

### Scope pass — `src/agm/agl/scope/`

- `symbols.py`: key declarations by `(ModuleId, scope_path, name)`; `BindingRef` carries the
  owning path; `ScopeNode` gains scope-region layers between module root and function scopes.
- `resolver.py`: the collection pre-passes walk scope regions and shorthand paths uniformly;
  duplicate detection per path (§3.2); chain resolution per §3.3 replaces
  `_resolve_type_qualified_constructor`/`_resolve_single_qualifier_constructor` with one
  chain resolver whose outputs subsume `qualified_constructor_refs`; constructor candidates keyed
  by scoped owner; `open` items resolved into region bare-contribution layers; library-module
  item restrictions applied inside scope regions unchanged.
- `imports.py`: selection sets, rename maps, bare contributions, and route-member filtering
  become path-keyed; subtree expansion for scope-name atoms; rename re-rooting. This file remains
  the pure policy seam — path atoms are a type change plus subtree expansion, not new policy
  entry points.
- `program.py`: path-keyed export maps, re-export fixed point, `decl_info`/`private_info`, and
  the reachability rule of §3.5.

### Typecheck — `src/agm/agl/typecheck/`

- `constructors.py`: consume unified chain-resolution results; drop its name-resolution half;
  keep variant/signature validation. Delete side tables that duplicated resolver output.
- Semantic nominal types (`RecordType`/`EnumType`/`ExceptionType`) and `TypeTable` keys gain the
  scope path; `display()` renders `A::T` (with module prefix as today).
- Function-signature tables and environment seeding key by path atoms; scope-interior lexical
  layers feed the checker's binding environment through the resolver output as today.

### Lower / IR — `src/agm/agl/lower/`, `src/agm/agl/ir/`

- `NominalId` gains the scope path next to `module_id` and `declared_name`; schema `$defs`
  naming and `RefDecode` boundaries derive from the structured id. Functions stay erased to
  handles; linking is unaffected beyond the id shape.

### REPL — per §3.6; retained-declaration replacement keyed by path atoms; `open` retention
mirrors import retention.

### Diagnostics

New or generalized errors, each with a focused message and span: unclosed region / closer
mismatch (naming the expected `end`), duplicate member per path, unknown scope path, ambiguous
leading segment with anchor repair guidance, type arguments on a non-type segment, disallowed
item inside a scope region, `open` selection of unknown/non-public members, bare collisions from
`open` at use sites, empty-subtree selection.

## 6. TDD implementation sequence

Every milestone starts with failing behavioral tests, tests observable syntax/semantics rather
than pass internals, and ends green. New e2e programs go under `tests/agl/programs/` with
multiple scenarios; agents are always mocked.

### M1 — Syntax

Failing parse/AST tests for: flat regions incl. nesting and multi-segment headers; closer
matching errors; shorthand paths on every name-headed form; qualifier chains in expression,
pattern, type, and `is`-test positions; applied segments; `open` with all clause forms; path
atoms in import/export clauses; `scope`/`end` still usable as identifiers outside promotion
positions; rejection of disallowed items inside regions.

### M2 — Single-module resolution

Scope trees, membership visibility for both spellings, nearest-enclosing shadowing, `::` root
anchoring, exact-path outside references, duplicates per path, equivalence of the block and
shorthand forms (the goal example verbatim), extension across multiple blocks.

### M3 — Constructor unification

Migrate `Option::None`, applied `Option[int]::None`, pattern qualification, and overloaded bare
variants onto chain resolution with the full existing constructor test suite as the regression
harness; extend enums via `scope Option` with functions; nested type scopes (`A::T::V`).

### M4 — Typecheck and identity

Scoped nominal types distinct from same-named root types; scoped generics; display forms;
signature tables keyed by path; scoped externs incl. the Python-symbol collision error; scoped
agents in the entry module.

### M5 — Modules

Path-atom selection (`using Point::distance`, `using Point`, `hiding`, renames re-rooting,
subtree rules), `open import` with scoped members, re-exports, cross-module coexistence and
ambiguity repair, privacy exclusion, reachability of empty/private subtrees, wildcard imports
distributing path clauses.

### M6 — `open` item

All three clause forms, renames, cross-module refs, region-scoped effect and placement rule,
collisions at use, composition with `open import`, opening type-named scopes (variants + members).

### M7 — Execution and REPL

End-to-end programs exercising scoped functions/types/variants through lowering, IR eval, and
schema boundaries; REPL entry-local regions, member replacement, `open` retention and `:reset`;
multi-file fixtures under `tests/agl/multi_file/`.

## 7. Documentation updates required with implementation

- `docs/agl/reference/`: a new scopes page (regions, shorthand, nesting, visibility, `open`);
  update `modules.md` (path atoms, selection, renames), `lexical-structure.md` (chains,
  contextual `scope`/`end`), and the enum/constructor sections (variant access as scope-member
  access; bare-variant convenience unchanged).
- `docs/arch/agl/`: `frontend/scope.md` (scope paths, chain resolution, open contributions),
  `modules.md` (path-keyed selection), `frontend/types.md` (path-extended nominal identity),
  `execution/` (NominalId shape) — succinct, architectural, no implementation minutiae.
- `docs/specs/2026-07-18-agl-namespaces.md`: amend the reference-route grammar to chains and note
  the scope construct, keeping the spec authoritative.

## 8. Verification

Focused tests per milestone via `uv run pytest …`; before completion:

```sh
just check
```

The final gate retains 100% `src/` coverage, strict mypy, ruff lint/format, all rejection
fixtures, REPL and multi-file suites, and e2e command coverage. No `type: ignore`, `noqa`, or
formatter suppressions.
