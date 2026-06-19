# Plan: AgL generics (prenex parametric polymorphism)

## Overview

AgL needs parametric polymorphism. Type parameters are written in square
brackets after the declared name:

```agl
def id[T](x: T) -> T = x

record Box[T]
  value: T

enum Option[T]
  | None
  | Some(value: T)

type Pair[A, B] = dict[text, json]   # parameterized alias (abbreviation)
```

The system supports **prenex (rank-1) polymorphism only**: quantifiers sit at
the outermost binder of a named declaration (`def`, `record`, `enum`, `type`).
There is no rank-2/higher-rank polymorphism, and lambdas/inner functions are
monomorphic (they may *use* type variables already in scope, but may not
introduce their own).

The implementation is **not** based on monomorphisation. AgL already evaluates
dynamically-typed `Value`s in a tree-walking interpreter, so the natural,
principled runtime model is **type erasure**: type arguments exist only during
type checking and vanish before evaluation. A single function body serves every
instantiation. The type checker is the principled core; the evaluator is almost
entirely unaffected.

This is a type-system feature first. The existing checker is already
**bidirectional** (`_check_expr(node, expected)` threads an expected type
everywhere) and the grammar already carries an explicit type-application form
`callee::[T](args)` (`Call.type_arg`, currently used by `ask-request`). Both are
reused.

## Resolved Owner Decisions

These were resolved with the owner and frame the implementation.

| # | Decision |
|---|----------|
| D1 | **Type arguments are inferred** via local inference (from argument types *and* the contextual expected type), with explicit `::[T]` as an override/escape hatch. A type variable that cannot be solved is a static error. |
| D2 | **Strict parametricity.** A value whose static type is a bare type variable `T` is fully opaque: it may be bound, passed, returned, and placed in containers/records/enums, but supports **no** equality, ordering, arithmetic, rendering/interpolation, field access, indexing, or `is` tests. |
| D3 | **Forbid type-variable `ask`/`exec`/`ask-request` target types.** Type-directed runtime operations (codec/schema generation) require a concrete type; under erasure no concrete type exists for a `T`. The checker rejects a target type that contains a free type variable. |
| D4 | **`def`, `record`, `enum`, and `type` alias may be generic.** Parameterized aliases are supported as type-level abbreviations. Lambdas/inner functions stay monomorphic. |
| D5 | **A generic `def` referenced as a value is instantiated at the reference site** to a monomorphic `FunctionType` (type variables solved from the expected type, or supplied explicitly). Unsolvable → error. Function values remain monomorphic. |
| D6 | **Invariant type arguments.** `C[a…]` is assignable to `C[b…]` only when the constructor matches and each argument is *equal*. No coercion (`int→decimal`, `json`-absorption) propagates into a type-argument position. Sound in the presence of AgL's mutable lists/dicts. |

### D1 — Inference with explicit override

Given a generic signature, a call site solves the type parameters by:

1. **Matching** each argument's type against the corresponding parameter
   *template* (the declared type with type variables as holes).
2. **Folding in the contextual expected type** by matching it against the result
   template — this resolves variables that appear only in the result
   (`def empty[T]() -> list[T]`).
3. If any parameter remains unsolved, raising
   *"cannot infer type argument `T`; supply it explicitly via `f::[…]`"*.

Explicit `f::[int](…)` skips solving and substitutes directly (arity-checked).
Coercions are **not** used while solving (solving is by equality, per D6); the
usual `int→decimal` / `json` coercions apply only in the final assignability
check *after* substitution.

### D2 — Strict parametricity

Inside a generic body, a `T`-typed value is opaque. The checker's existing
capability gates (`comparable_types`, JSON-shaped/renderable checks, arithmetic
typing, field-access, indexing, `is`-test typing) all reject a `TypeVarType`
operand with a clear diagnostic. Operations become available only after `T` is
instantiated to a concrete type at the call site. This preserves parametricity
("free theorems") and is the smallest possible rule; B (pragmatic
rendering/equality) and C (bounded polymorphism / type classes) are deferred
(see *Out of scope*).

### D3 — No generic agent/exec targets

The output contract (`OutputContractSpec`, codec + JSON schema) is materialized
statically per call site from a concrete target type. A type-variable target has
no concrete type at runtime under erasure, so:

```agl
def fetch[T](a: agent, p: text) -> T = ask::[T](p, agent: a)   # static error
```

is rejected with *"agent/exec target type cannot contain a type variable (`T`)"*.
Generic functions may still pass agents around, build prompts, and return
non-agent-derived `T` values. Runtime type reification (the general alternative)
is deferred (see *Out of scope*).

### D4 — What may be generic

`def`, `record`, `enum`, and `type` alias may declare `[T, U, …]`. Parameterized
aliases are expanded by substitution during type resolution. A lambda may
*reference* the type variables of its enclosing generic `def` but may not declare
`[…]` of its own.

### D5 — Generic `def` as a value

Referencing a generic `def` without calling it yields a monomorphic
`FunctionType`, with type variables solved from the expected type
(`let f: (int) -> int = id`) — or, where grammatically available, from explicit
type arguments. With no usable expected type it is an error. `FunctionType`
stays monomorphic (no quantifiers, no let-polymorphism).

### D6 — Invariant type arguments

Assignability of constructed types matches the current `ListType`/`DictType`
behavior: same constructor and **equal** arguments. This is sound for AgL's
mutable containers (`var` lists, indexed assignment) and introduces no
regression.

## Syntax & grammar strategy

Type parameters and type application both use square brackets. Type-parameter
*names* are `TYPE_NAME` tokens (uppercase-leading), so `[T]`, `[A, B]` fit the
existing lexer with no new tokens. Reuse the existing `type_lsqb` helper
(`LSQB | INDEX_LSQB`) so both spaced (`def id [T]`) and adjacent (`def id[T]`)
forms are accepted.

### Declaration type-parameter lists

```ebnf
type_params: type_lsqb type_param_list RSQB
type_param_list: TYPE_NAME (COMMA TYPE_NAME)*

func_def:   "def" VAR_NAME type_params? LPAR param_list? RPAR THIN_ARROW type_expr EQ func_body
record_def: "record" TYPE_NAME type_params? _INDENT field_def (_NEWLINE field_def)* _NEWLINE? _DEDENT
enum_def:   "enum" TYPE_NAME type_params? variant_def+
type_alias: "type" TYPE_NAME type_params? EQ type_expr
```

After `def VAR_NAME` / `record TYPE_NAME` / `enum TYPE_NAME` / `type TYPE_NAME`,
a `[` can only introduce a type-parameter list (no other production applies), so
this is conflict-free.

### Type application in type position

```ebnf
type_expr: ...
         | TYPE_NAME type_lsqb type_list RSQB   -> applied_type
```

`list[T]` / `dict[text, V]` keep their existing productions (`generic_type_1`,
`dict_type`) — `list`/`dict` arrive as `VAR_NAME`, user types as `TYPE_NAME`, so
the new `applied_type` alternative (TYPE_NAME head) does not collide.

A bare `TYPE_NAME` in type position still parses via `named_type`; whether it is
a *type variable* or a nominal type is decided during type resolution
(scope-aware), not in the parser.

### Conflict-freeness

The grammar must remain LALR(1) with **0 shift/reduce and 0 reduce/reduce**
conflicts. The mandatory conflict-guard regression in
`tests/test_agl_parser.py` must stay green; adding these productions is the
first verification gate of M1.

### Not added (kept minimal; see *Implementation notes*)

- No constructor type-application syntax (`Box[int](…)`) — generic constructors
  rely on inference + expected type.
- No bare value-position type application (`id::[int]` without a call) in v1 —
  generic-`def`-as-value relies on the expected type (D5). Optional future
  ergonomic addition.

## AST changes (`src/agm/agl/syntax`)

`types.py` — new syntactic type-expression node:

```python
@dataclass(frozen=True, slots=True)
class AppliedT:
    """A type application ``Name[args]`` (user generic type or alias)."""
    name: str
    args: tuple[TypeExpr, ...]
    span: SourceSpan = field(compare=False)
    node_id: int = field(compare=False)
```

Add `AppliedT` to the `TypeExpr` union. (Type variables are *not* a distinct
syntactic node — a variable use is a bare `NameT`; the checker classifies it.)

`nodes.py` — type-parameter lists on the four declaration nodes:

- `FuncDef` gains `type_params: tuple[str, ...] = ()`.
- `RecordDef` gains `type_params: tuple[str, ...] = ()`.
- `EnumDef` gains `type_params: tuple[str, ...] = ()`.
- `TypeAlias` gains `type_params: tuple[str, ...] = ()`.

Wire through: `syntax/__init__.py` exports, `syntax/visitor.py` traversal,
parser `transform.py` handlers (`applied_type`, `type_param_list`, and the four
declaration builders), and `tests/test_agl_ast.py`.

## Semantic type model (`src/agm/agl/typecheck/types.py`)

### New semantic type: `TypeVarType`

```python
@dataclass(frozen=True, slots=True)
class TypeVarType:
    """A rigid type variable bound by an enclosing generic declaration.

    Opaque under strict parametricity (D2). Equality is by name within a
    declaration's checking scope; variables from different declarations are
    never compared directly (each declaration is checked independently and a
    callee's scheme is instantiated before matching).
    """
    name: str

    @property
    def kind(self) -> str: return "typevar"
    def __repr__(self) -> str: return self.name
```

Add to the `Type` union.

### Generic nominal instances

`RecordType` and `EnumType` gain `type_args: tuple[Type, ...] = ()`.

- **Identity/equality** of a nominal type becomes **name + `type_args`**, with
  `fields`/`variants` **excluded from equality and hashing** (`compare=False`).
  This is sound because the type namespace maps each name to exactly one
  definition, and it is *required* to support recursive generic types
  (`enum List[T] { Nil | Cons(head: T, tail: List[T]) }`) without infinite
  expansion during construction or comparison.
- `fields`/`variants` of a generic *instance* are computed **on demand** by
  substituting `type_args` into the stored template (see env, below). A nested
  self-reference resolves to another `RecordType(name, type_args)` whose fields
  are *not* eagerly expanded, terminating the recursion.

Non-generic records/enums are the `type_args == ()` case and behave exactly as
today (name-based identity is at least as correct as the prior field-structural
identity, since names are unique).

`list`/`dict` remain `ListType(elem)` / `DictType(value)` — they are already
structural built-in type constructors and participate in matching/assignability
unchanged.

### Capability gates updated for `TypeVarType` (D2)

- `is_json_shaped(TypeVarType)` → `False` (so render/interpolation/`print`
  reject it, and it can never be an `ask`/`exec` target — reinforces D3).
- `comparable_types` → `False` for any `TypeVarType` operand.
- `is_assignable`: a `TypeVarType` is assignable only to an *equal*
  `TypeVarType` (rigid-variable identity); nothing widens to or from it.
- Arithmetic/field-access/index/`is`-test typing in the checker reject
  `TypeVarType` operands with targeted diagnostics.

### Helpers

- `free_type_vars(t: Type) -> frozenset[str]` — recursively collect free type
  variable names (used by D3's target-type guard and by the inference solver).
- `substitute(t: Type, subst: Mapping[str, Type]) -> Type` — capture-free
  substitution of solved type arguments (prenex ⇒ no nested binders ⇒ no capture
  concerns).
- `contains_type_var(t) -> bool` — `bool(free_type_vars(t))`.

## Type environment (`src/agm/agl/typecheck/env.py`)

### Generic type definitions (templates)

Store generic record/enum definitions as **templates**: the type parameter names
plus the field/variant types expressed with `TypeVarType` placeholders. Add:

```python
@dataclass(frozen=True, slots=True)
class GenericTypeDef:
    kind: str                 # "record" | "enum"
    type_params: tuple[str, ...]
    template: Type            # RecordType/EnumType whose fields use TypeVarType
```

- `register_generic_type(name, GenericTypeDef)` / `get_generic_type(name)`.
- `instantiate_nominal(name, args) -> RecordType | EnumType`: arity-check
  `len(args) == len(type_params)`, build a substitution, and produce a
  `RecordType`/`EnumType` carrying `type_args=args` with fields/variants resolved
  by substitution **on demand** (lazily, to support recursion).

### Parameterized aliases

`register_alias` already stores a raw `TypeExpr`. Extend alias storage to also
record `type_params`. When `resolve_named_type`/`resolve_type_expr` resolves an
`AppliedT` whose head is an alias, substitute the arguments into the alias body
(arity-checked, existing cycle detection preserved).

### `FunctionSignature`

Gains `type_params: tuple[str, ...]`. The value-level `FunctionType` (param/result
tuple) is unchanged and stays monomorphic; `type_params` lives only on the
signature (used by inference and D5 instantiation).

### `resolve_type_expr` becomes type-var aware

Add a `type_vars: frozenset[str]` parameter (default empty):

- `NameT(name)` with `name in type_vars` → `TypeVarType(name)`.
- `NameT(name)` otherwise → existing nominal/alias resolution.
- `AppliedT(name, args)` → resolve the generic type or parameterized alias by
  name, arity-check, substitute (recursively passing `type_vars`).

The same `type_vars` set is threaded while resolving a generic declaration's
parameter types, return type, field types, and variant payloads.

## Scope pass (`src/agm/agl/scope`)

Minimal: the scope pass resolves *value* names; type expressions are resolved in
the checker. Changes:

- Validate type-parameter lists at each generic declaration: **no duplicate**
  parameter names; (optional) warn on an unused type parameter.
- Record the per-declaration type-parameter scope keyed by the declaration's
  `node_id` in a `ResolvedProgram` side table, OR let the checker read
  `FuncDef.type_params` etc. directly (preferred — the checker already visits
  declarations). Type-variable *resolution* stays in the checker.

No change to value resolution: a bare `NameT("T")` inside a generic body is a
type expression, untouched by value-name resolution.

## Type checker (`src/agm/agl/typecheck/checker.py`)

The core of the feature.

### Checking a generic `def`

1. `type_vars = set(node.type_params)`. Resolve parameter and return types with
   this set so they may contain `TypeVarType`.
2. Register a `FunctionSignature` carrying `type_params` (in the `def` pre-pass,
   so generic mutual recursion works as today).
3. Check the body with the rigid `TypeVarType`s in scope. Strict parametricity
   (D2) is enforced by the capability gates above — e.g. comparing two `T`s,
   `print`-ing a `T`, or `t.field` on a `T` are static errors.

### Generic records / enums

- Register a `GenericTypeDef` template (in the type pre-pass).
- **Constructor inference**: `Box(value: e)` / `Some(value: e)` solve the type
  parameters by matching field-argument types against the field templates, then
  folding in the expected type (so `None` and other payload-free / unconstrained
  cases are resolved from context). Unsolvable → the standard "cannot infer"
  error. Result is the instantiated `RecordType`/`EnumType` with `type_args`.
- **Field access** `b.value` on `b : Box[int]`: instantiate `Box`'s field
  template with `b`'s `type_args`.
- **Patterns** (`case`, `is`): a `Some(value: v)` pattern against `Option[int]`
  binds `v : int` via the same instantiation.

### Call-site inference (declared generic `def`)

In `_check_declared_name_call`, when the signature has `type_params`:

1. **Explicit path** — `node.type_arg` present (`f::[A, B](…)`): arity-check,
   build `subst` directly, substitute into the signature, then run the existing
   per-argument assignability checks (coercions allowed here).
2. **Inference path** — two phases:
   - *Collect & solve.* For each positional/named argument, infer the argument
     type (with an expected type derived from the *already-known* parts of the
     parameter template; bare unsolved variables contribute no expectation), and
     `match(param_template, arg_type, subst)`. Then, if `expected is not None`,
     `match(result_template, expected, subst)`.
   - *Verify.* Any `type_param` absent from `subst` → "cannot infer type
     argument" error. Otherwise substitute into the signature and run the normal
     assignability checks (now with coercions).
3. Return the substituted result type.

`match(template, concrete, subst)` (one-sided; only the template has holes):

- `TypeVarType(p)`: if bound, require `subst[p] == concrete` (else *inconsistent
  type argument* error); else bind `subst[p] = concrete`.
- `ListType`/`DictType`/`FunctionType`/generic `RecordType`/`EnumType`: shape
  must agree; recurse componentwise (and over `type_args`).
- Anything else: require equality (invariant solving, D6).
- Shape mismatch is *not* itself fatal — matching is best-effort for solving;
  the post-substitution assignability check produces the user-facing error. The
  two genuinely solver-owned errors are *inconsistent binding* and *uninferable
  variable*.

### Value calls & generic-`def`-as-value (D5)

- `_check_value_call` is unchanged for monomorphic callees.
- When a `VarRef` resolving to a generic `def` is used in **non-call** position,
  produce its `FunctionType` by solving `type_params` against the **expected**
  type (which must be a `FunctionType`); no usable expectation → "cannot infer
  type arguments for generic function used as a value" error.

### D3 guard

In `_check_ask_call`, `_check_exec_call`, and the `ask-request` path: after the
target type is determined, reject it if `contains_type_var(target_type)`.

### Diagnostics

New, specific messages for: arity mismatch on `[…]`/`::[…]`; unknown type
parameter; duplicate type parameter; inconsistent inferred type argument;
uninferable type argument (suggest `::[…]`); operation not permitted on an
abstract type variable (D2); agent/exec target may not contain a type variable
(D3). Honor the checker's first-error-abort convention.

## Evaluator & runtime (erasure — minimal change)

Type arguments are **erased**. The evaluator already operates on dynamically
typed `Value`s:

- Generic `def`s are ordinary `Closure`s; no type arguments are bound, stored, or
  passed at runtime.
- Record/enum runtime values carry only their nominal name and field values
  (no `type_args`); construction, field access, pattern matching, equality, and
  rendering are unchanged.
- `Call.type_arg` remains static-only (already documented as never evaluated).
- No runtime type representation is introduced (D3 keeps the only type-directed
  runtime operation — codec/schema generation — on concrete types).

Expected outcome: **no semantic changes** to `eval`; at most incidental wiring.
Confirm with the existing eval/e2e suites plus new generic programs.

## Documentation

- **`docs/agl/reference/`** — new "Generics" section (language-user facing, no
  implementation references): declaration syntax for `def`/`record`/`enum`/
  `type`; type application `Name[args]`; inference and the `::[…]` override;
  strict parametricity (what you can and cannot do with a `T`); the no-generic-
  agent-target rule; invariance. Update any affected pages (types, functions,
  records/enums, agents).
- **`docs/arch/agl.md`** — extend the type-system section: `TypeVarType`,
  generic-definition templates + on-demand instantiation, name+`type_args`
  nominal identity, the inference/matching solver, and the erasure rationale.
- **`docs/commands.md`** / command help — update only if user-visible CLI
  behavior changes (none expected; verify `--dry-run` inventory still renders
  generic types sensibly).

## Testing

Follow TDD; write failing tests first. Maintain 100% `src/` coverage and the
green e2e gate.

- **Parser/AST** (`test_agl_parser.py`, `test_agl_ast.py`): type-parameter lists
  on all four declarations; `Name[args]` applied types; spaced vs adjacent
  brackets; the LALR conflict-guard.
- **Type resolution / env** (`test_agl_typecheck.py`): `TypeVarType` resolution
  in scope; `AppliedT` arity errors; parameterized-alias substitution; recursive
  generic enum (e.g. `List[T]`); nominal identity by name+`type_args`;
  invariance (`list[int]` not assignable to `list[decimal]`/`list[json]` at the
  argument position; `Box[int]` vs `Box[text]`).
- **Inference** (`test_agl_typecheck.py`): `id`, `map`-like signatures;
  argument-driven solving; result-only solving from expected type; explicit
  `::[…]` override; inconsistent-binding error; uninferable-variable error;
  generic constructors (`Box`, `Option.Some`, payload-free `None` from context);
  field access and pattern binding on generic instances.
- **Parametricity (D2)**: rejecting `=`/`<`/`+`/`print`/interpolation/field/
  index/`is` on a `T`.
- **D3**: rejecting `ask::[T]` / `exec` / `ask-request` with a type-variable (or
  type-var-containing) target; allowing the concrete-target cases.
- **D5**: `let f: (int) -> int = id` succeeds; bare `let f = id` (no expectation)
  errors.
- **Eval & e2e** (`tests/agl/programs/`, `test_agl_eval.py`, `test_agl_e2e.py`):
  generic identity/`map`/`Option`/`Result`/`Box`/generic stack programs
  exercising generics together with `case`, records, enums, lists/dicts, and
  higher-order functions; confirm erasure (a single body serves multiple
  instantiations at runtime).

## Milestones

Commit per milestone once `just check` passes (lint + tests + strict mypy).

- **M1 — Syntax & AST.** Grammar (type-parameter lists, `applied_type`), parser
  transform, AST node/field additions, visitor + exports, conflict-guard and AST
  tests. Gate: parser produces correct AST; 0 grammar conflicts.
- **M2 — Semantic types & environment.** `TypeVarType`; `type_args` on
  `RecordType`/`EnumType` with name+args identity; `GenericTypeDef` templates +
  on-demand instantiation; parameterized aliases; `FunctionSignature.type_params`;
  type-var-aware `resolve_type_expr`; `free_type_vars`/`substitute`/
  `contains_type_var`; capability-gate updates. Unit tests.
- **M3 — Type checker.** Generic `def` checking with parametricity gating; the
  matching/inference solver; declared-call inference (explicit + inferred);
  generic constructor/field/pattern inference; generic-`def`-as-value
  instantiation; diagnostics. Checker tests.
- **M4 — Erasure guards & evaluation.** D3 target-type rejection; confirm
  evaluator unchanged; new e2e generic programs; eval/e2e tests.
- **M5 — Documentation & polish.** Reference docs, `docs/arch/agl.md`, command
  docs/help review; final full e2e sweep.

## Implementation notes / derived decisions (flagged for owner review)

These follow from D1–D6 rather than being independent architectural choices, but
are surfaced explicitly so they can be vetoed:

1. **Nominal identity moves to name + `type_args`** (fields excluded from
   equality/hashing). Required for recursive generic types; sound because names
   are unique in the namespace.
2. **Generic constructors infer from field arguments + expected type** (no
   `Box[int](…)` constructor syntax). Keeps the expression grammar unchanged.
3. **Bare value-position type application** (`id::[int]` without a call) is not
   added in v1; generic-`def`-as-value relies on the expected type (D5). Optional
   future ergonomic addition.
4. **Inference is incomplete by design** for some higher-order / payload-free
   cases (e.g. passing one generic function to another). The explicit `::[…]`
   override and type annotations are the documented escape hatch.
5. **Coercions do not participate in solving** (`int→decimal`, `json`-absorption
   apply only at the post-substitution assignability check), per the invariant
   rule D6.

## Out of scope (future work)

- **Runtime type reification / dictionary passing** to enable generic
  `ask`/`exec` target types (the alternative to D3's restriction).
- **Bounded polymorphism / type classes** (`T: Comparable`, `T: Renderable`) to
  relax strict parametricity (D2).
- **Pragmatic rendering/equality on `T`** (the D2 "pragmatic" option).
- **Higher-rank polymorphism** and **generic lambdas / polymorphic function
  values** (let-polymorphism), excluded by the prenex restriction (D4/D5).
- **Declaration-site variance annotations** (D6 alternative C).
