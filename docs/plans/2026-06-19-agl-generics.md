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
`callee::[T](args)` (currently used by `ask-request`). The syntax is generalized
to carry a tuple of type arguments so declarations with multiple parameters can
be instantiated explicitly.

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
| D7 | **Constructors are ordinary value bindings with no capitalization rule.** Record constructors and enum variants are registered in the lexical value namespace and may be lowercase or uppercase. Constructor-only duplicate names form an overload set: an unqualified reference with multiple candidates is statically ambiguous before inference. A unique generic constructor infers its type arguments from payload + context, or accepts them explicitly via `some::[int](…)`. |

### D1 — Inference with explicit override

Given a generic signature, a call site solves the type parameters by:

1. **Matching** each argument's type against the corresponding parameter
   *template* (the declared type with type variables as holes).
2. **Folding in the contextual expected type** by matching it against the result
   template, but binding **only variables that are still unsolved** — this
   resolves variables that appear only in the result
   (`def empty[T]() -> list[T]`).
3. If any parameter remains unsolved, raising
   *"cannot infer type argument `T`; supply it explicitly via `f::[…]`"*.

Explicit `f::[int](…)` skips inference entirely and substitutes the supplied
arguments directly (arity-checked).
Coercions are **not** used while solving (solving is by equality, per D6); the
usual `int→decimal` / `json` coercions apply only in the final assignability
check *after* substitution. Consequently, context never changes a type argument
already inferred from an argument: `let x: decimal = id(1)` infers `T = int`,
then succeeds because the instantiated `int` result is assignable to `decimal`.
A genuine result mismatch is reported by the final expected-type check, not as
an inconsistent-inference error.

### D2 — Strict parametricity

Inside a generic body, a `T`-typed value is opaque. The checker's existing
capability gates (`comparable_types`, JSON-shaped/renderable checks, arithmetic
typing, field-access, indexing, `is`-test typing) all reject a `TypeVarType`
operand with a clear diagnostic. Operations become available only after `T` is
instantiated where the resulting concrete value is used; the generic body itself
is checked once with rigid variables, so it cannot contain those operations.
This preserves parametricity ("free theorems") and is the smallest possible
rule; B (pragmatic rendering/equality) and C (bounded polymorphism / type
classes) are deferred (see *Out of scope*).

### D3 — No generic agent/exec targets

The output contract (`OutputContractSpec`, codec + JSON schema) is materialized
statically per call site from a concrete target type. A type-variable target has
no concrete type at runtime under erasure, so:

```agl
def fetch[T](a: agent, p: text) -> T = ask::[T](p, agent: a)   # static error
```

is rejected with *"agent/exec target type cannot contain a type variable (`T`)"*.
For `ask`, `exec`, and `ask-request`, `::[Target]` directly supplies the target
type and disables contextual/default target inference. These built-ins require
exactly one explicit type argument. The call's resulting value is still checked
against its surrounding expected type in the normal way.
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

### D7 — Constructor value bindings, lookup, and explicit arguments

Record constructors and enum variants are normal bindings in the value
namespace. Capitalization has no semantic meaning for identifiers: `some`,
`Some`, `box`, and `Box` are all syntactically valid constructor, function,
variable, and type names. Type and value namespaces remain semantically
separate; a record therefore introduces its name once in each namespace (the
record type and its value constructor) without collision.

Record constructors (including zero-field records) and payload variants are
callable values. Nullary enum variants are ordinary values. Generic constructors
support the same local type-argument inference and expected-type instantiation
as generic functions:

```agl
enum Option[T]
  | none
  | some(value: T)

let inferred: Option[int] = some(value: 1)
let explicit = some::[int](value: 1)
let empty: Option[int] = none
```

Lookup precedes inference. The scope pass resolves an unqualified constructor
reference through the value namespace:

- no binding → the ordinary undefined-name error;
- one candidate → infer its type arguments, or use the explicit `::[…]`
  arguments without inference;
- a constructor overload set with two or more candidates → static ambiguity
  error naming all candidate enums.

Payload shape, payload types, explicit type arguments, and contextual expected
type do **not** filter an ambiguous candidate set. Thus if both `Option` and
`Other` declare `some`, `some(value: 1)` and `some::[int](value: 1)` are both
ambiguous. Qualification (`Option.some(value: 1)`) selects the owner before
inference and is the required disambiguation.

At one lexical scope, an ordinary value binding (function, parameter, `let`,
`var`, agent, or constructor) conflicts with any other ordinary value binding of
the same name. The sole exception is constructor-versus-constructor: constructors
from distinct enums may share an unqualified name and form the overload set
above. Normal nested-scope shadowing still applies; a nearer ordinary binding
hides an outer constructor or constructor overload set.

The explicit syntax applies uniformly to generic record constructors
(`box::[int](value: 1)`). Qualified explicit syntax such as
`Option.some::[int](…)` is not added in v1; qualification plus payload/context
inference is sufficient, and an expected annotation handles payload-free
variants.

Because capitalization no longer distinguishes pattern binders from nullary
constructors, a bare name in a pattern always introduces a binder. Match a
nullary constructor with call syntax (`none()`) or a qualified constructor
pattern (`Option.none`). Payload constructors already use call syntax
(`some(value: x)`). This rule is syntactic, deterministic, and independent of
which bindings happen to be in scope.

This is a source-level compatibility change for existing bare nullary patterns:
`case Pass` must become `case Pass()` or `case Review.Pass`. Existing uppercase
constructor expressions remain valid because names are case-neutral; they now
resolve through scope instead of parser classification.

## Syntax & grammar strategy

Type parameters and type application both use square brackets. Reuse the
existing `type_lsqb` helper (`LSQB | INDEX_LSQB`) so both spaced (`def id [T]`)
and adjacent (`def id[T]`) forms are accepted.

### Case-neutral names

**The `VAR_NAME`/`TYPE_NAME` distinction is abolished at the source.** The
scanner emits a **single `NAME` token** for every identifier; the case of the
first letter has **no** lexical, grammatical, or semantic meaning. There is no
longer a separate "type name" token class — `record box`, `let X = 1`, and
`enum option | none | some(value: T)` all lex and parse identically regardless
of capitalization. Keywords remain reserved as today (matched by exact spelling
before the `NAME` fallback).

The grammar uses the `NAME` terminal directly in every identifier position:
type declarations and expressions, function/parameter declarations, `let`,
`var`, `param`, agent, loop/catch/pattern binder, assignment-target,
named-argument, enum variant, field, and ordinary-reference productions. The
grammar provides syntax; the scope/type passes decide which semantic namespace a
name occupies from its position and declaration kind — never from its
capitalization.

Concretely this means the lexer (`scanner.py`/`tokens.py`/`lexer.py`) drops the
`word[0].isupper()` branch and the `TYPE_NAME`/`VAR_NAME` token constants in
favor of one `NAME` constant, the grammar `%declare`s `NAME` instead of the two
old terminals, and every consumer of the token stream (the parser transform, the
REPL syntax highlighter) treats identifiers uniformly. No code anywhere may
branch on identifier capitalization.

### Declaration type-parameter lists

```ebnf
type_params: type_lsqb type_param_list RSQB
type_param_list: name (COMMA name)*

func_def:   "def" name type_params? LPAR param_list? RPAR THIN_ARROW type_expr EQ func_body
record_def: "record" name type_params? _INDENT field_def (_NEWLINE field_def)* _NEWLINE? _DEDENT
enum_def:   "enum" name type_params? variant_def+
variant_def: PIPE name variant_payload?
type_alias: "type" name type_params? EQ type_expr
```

After `def name` / `record name` / `enum name` / `type name`,
a `[` can only introduce a type-parameter list (no other production applies), so
this is conflict-free.

### Type application in type position

```ebnf
type_expr: ...
         | name type_lsqb type_list RSQB   -> applied_type
         | name                           -> named_or_builtin_type

type_arg_list: type_expr (COMMA type_expr)*
typed_call_atom: name DCOLON LSQB type_arg_list RSQB LPAR arg_list? RPAR

var_ref: name

pattern: name LPAR pattern_fields? RPAR       -> pat_constructor
       | name DOT name (LPAR pattern_fields? RPAR)? -> pat_qualified_constructor
       | name                                  -> pat_binder

is_target: name | name DOT name
```

The type transformer classifies built-in spellings (`int`, `list`, `dict`, etc.)
by value and otherwise produces `NameT`/`AppliedT`; capitalization is irrelevant.
Whether a bare name is a type variable, nominal type, or alias is decided during
scope-aware type resolution, not in the parser.

### Conflict-freeness

The grammar must remain LALR(1) with **0 shift/reduce and 0 reduce/reduce**
conflicts. The mandatory conflict-guard regression in
`tests/test_agl_parser.py` must stay green; adding these productions is the
first verification gate of M1.

### Not added (kept minimal; see *Implementation notes*)

- No bracket-only constructor type-application syntax (`box[int](…)`). Generic
  constructors use inference + expected type or explicit
  `box::[int](…)` / `some::[int](…)` syntax.
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

`nodes.py` — generalize explicit call-site type arguments:

- Replace `Call.type_arg: TypeExpr | None` with
  `Call.type_args: tuple[TypeExpr, ...] = ()`.
- Update the parser transformer, syntax visitor, evaluator's static-only
  handling, checker, and existing `ask-request` tests. A typed call always has a
  non-empty tuple; ordinary calls use `()`.

`nodes.py` — normalize constructor expressions into value expressions:

- Remove the expression-level `Constructor` special form. An unqualified
  constructor reference is a `VarRef`; applying a payload constructor is an
  ordinary `Call`, including `Call.type_args` for explicit instantiation.
- Retain `ConstructorPattern`, but update it to carry a syntactic
  `qualified: bool`/owner name rather than relying on token capitalization.
- Retain or rename the existing qualified-access node for `Option.some`; the
  scope pass resolves it to the selected constructor binding. It is a value
  expression, not eager construction.
- Update syntax exports, traversal, parser transforms, and AST tests. No parser
  transform may classify a value name as a constructor based on capitalization.

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
  definition and avoids making representation details part of nominal identity.
- `fields`/`variants` of a generic instance are computed eagerly by substituting
  `type_args` into the stored template when the instance is created. Recursive
  records/enums remain prohibited, so eager substitution terminates.

Non-generic records/enums are the `type_args == ()` case and behave exactly as
today (name-based identity is at least as correct as the prior field-structural
identity, since names are unique).

`list`/`dict` remain `ListType(elem)` / `DictType(value)` — they are already
structural built-in type constructors and participate in matching/assignability
unchanged.

### Capability gates updated for `TypeVarType` (D2)

- `is_json_shaped(TypeVarType)` → `False`.
- `comparable_types` → `False` for any `TypeVarType` operand.
- `is_assignable`: a `TypeVarType` is assignable only to an *equal*
  `TypeVarType` (rigid-variable identity); nothing widens to or from it.
- Arithmetic/field-access/index/`is`-test typing in the checker reject
  `TypeVarType` operands with targeted diagnostics.
- Rendering gates, including `print` and interpolation, explicitly reject
  `TypeVarType`; `print` does not currently use `is_json_shaped`, so it needs a
  direct check.

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
  `RecordType`/`EnumType` carrying `type_args=args` with fields/variants eagerly
  substituted from the template.
- Preserve the existing direct/indirect recursion detection for both generic
  and non-generic records/enums. A recursive reference such as `List[T]` inside
  `List[T]` remains a static error.

### Parameterized aliases

`register_alias` already stores a raw `TypeExpr`. Extend alias storage to also
record `type_params`. When `resolve_named_type`/`resolve_type_expr` resolves an
`AppliedT` whose head is an alias, substitute the arguments into the alias body
(arity-checked, existing cycle detection preserved).

### `FunctionSignature`

Gains `type_params: tuple[str, ...]`. The value-level `FunctionType` (param/result
tuple) is unchanged and stays monomorphic; `type_params` lives only on the
signature (used by inference and D5 instantiation).

### Constructor signatures

Add a `ConstructorSignature` keyed by constructor binding declaration id. It
carries the owner record/enum name, optional variant name, declared field names
and templates, result template, and `type_params`. This is the constructor
equivalent of `FunctionSignature`: direct calls retain named-field checking,
while a constructor used as a value is instantiated to a monomorphic
`FunctionType` with fields in declaration order. A nullary enum variant has no
callable signature and its binding directly has the instantiated enum value
type. A zero-field record constructor has `FunctionType(params=(), result=R)`
and is invoked as `r()`.

### `resolve_type_expr` becomes type-var aware

Add a `type_vars: frozenset[str]` parameter (default empty):

- `NameT(name)` with `name in type_vars` → `TypeVarType(name)`.
- `NameT(name)` otherwise → existing nominal/alias resolution. If `name` denotes
  a generic nominal type or alias with nonzero arity, reject the bare use and
  report the required number of type arguments.
- `AppliedT(name, args)` → resolve the generic type or parameterized alias by
  name, arity-check, substitute (recursively passing `type_vars`).

The same `type_vars` set is threaded while resolving a generic declaration's
parameter types, return type, field types, and variant payloads.

## Scope pass (`src/agm/agl/scope`)

The scope pass owns constructor lookup because constructors are value bindings;
type-expression resolution remains in the checker. Changes:

- Validate type-parameter lists at each generic declaration: **no duplicate**
  parameter names; (optional) warn on an unused type parameter.
- Add `BinderKind.constructor_binding`. During the program-root declaration
  pre-pass, register one value binding for each record constructor and enum
  variant. Store owner declaration/variant metadata in the `BindingRef` or a
  constructor-signature side table.
- Use the normal value-scope lookup for unqualified constructor references.
  A same-scope constructor-versus-constructor duplicate is represented as an
  ordered candidate set; resolving any unqualified use of that set raises the D7
  ambiguity error before type checking. Any same-scope collision involving a
  non-constructor value remains the ordinary duplicate-binding error.
- Normal nested lexical shadowing applies to constructor bindings and candidate
  sets.
- Resolve `TypeName.variant` by looking up the enum in the type namespace and
  selecting that enum's constructor binding directly; qualification bypasses
  the unqualified candidate set.
- For patterns, parse bare names as binders. Resolve only explicit constructor
  pattern forms (`some(...)` and `Option.none`) as value-namespace constructor
  references. This avoids scope-dependent parsing.
- Let the checker read declaration `type_params` directly; type-variable
  resolution stays in the checker.

A bare `NameT("T")` inside a generic body is a type expression and remains
untouched by value-name resolution.

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

- Register a `GenericTypeDef` template and a `ConstructorSignature` for each
  constructor binding in the type pre-pass.
- Preserve the current recursion checks while building templates; recursive
  generic and non-generic nominal types are both rejected.
- **Constructor lookup**: consume the constructor `BindingRef` produced by the
  scope pass. Do not scan the type environment by spelling and do not classify
  names by capitalization. Qualified access already resolves to the selected
  binding; ambiguous unqualified references never reach inference.
- **Constructor inference**: `box(value: e)` / `some(value: e)` solve the type
  parameters by matching field-argument types against the selected binding's
  field templates, then folding in the expected type only for still-unsolved
  parameters (so `None` and other payload-free / unconstrained cases are
  resolved from context). Unsolvable → the standard "cannot infer" error.
  Result is the instantiated `RecordType`/`EnumType` with `type_args`.
- **Explicit constructor arguments**: when a constructor-binding `Call` has
  non-empty `type_args`, arity-check and substitute those arguments directly; do
  not run inference. Then perform ordinary field and expected-result
  assignability checks. Explicit arguments never disambiguate an unqualified
  constructor overload set because ambiguity is rejected by the scope pass.
- **Constructor values**: a payload constructor referenced without a direct call
  is instantiated from an expected `FunctionType`, exactly like D5 generic defs;
  without enough context, generic constructors are an inference error. Fields
  map to positional function parameters in declaration order after the
  constructor escapes as a value. A nullary variant reference is checked as its
  owner enum value, using expected type to instantiate generic parameters.
- **Field access** `b.value` on `b : Box[int]`: instantiate `Box`'s field
  template with `b`'s `type_args`.
- **Patterns** (`case`, `is`): a `some(value: v)` pattern against `Option[int]`
  binds `v : int` via the same instantiation.

### Call-site inference (declared generic `def`)

In `_check_declared_name_call`, when the signature has `type_params`:

1. **Explicit path** — `node.type_args` non-empty (`f::[A, B](…)`): arity-check,
   build `subst` directly, substitute into the signature, then run the existing
   per-argument and result assignability checks (coercions allowed here). Do not
   run inference when explicit arguments were supplied.
2. **Inference path** — two phases:
   - *Collect & solve.* For each positional/named argument, infer the argument
     type (with an expected type derived from the *already-known* parts of the
     parameter template; bare unsolved variables contribute no expectation), and
     `match(param_template, arg_type, subst)`. Then, if `expected is not None`,
     `match_unsolved(result_template, expected, subst)`, which may bind only
     currently unbound type parameters and never challenges an existing binding.
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

After obtaining the instantiated result, check it against `expected` with normal
assignability. This is where contextual result mismatches and result coercions
are handled.

### Value calls and polymorphic bindings (D5/D7)

- `_check_value_call` is unchanged for monomorphic callees.
- When a `VarRef` resolving to a generic `def` is used in **non-call** position,
  produce its `FunctionType` by solving `type_params` against the **expected**
  type (which must be a `FunctionType`); no usable expectation → "cannot infer
  type arguments for generic function used as a value" error.
- Apply the same expected-type instantiation to generic payload-constructor
  bindings. Generic nullary constructor bindings solve against the expected
  nominal enum type rather than a `FunctionType`.

### D3 guard

In `_check_ask_call`, `_check_exec_call`, and the `ask-request` path:

- If `node.type_args` is non-empty, require exactly one argument and resolve it
  as the target type. Do not infer/default the target in this path.
- Otherwise retain each built-in's existing contextual/default target behavior.
- After the target is determined, reject it if `contains_type_var(target_type)`.
- Check the built-in's result against the surrounding expected type using normal
  assignability, where applicable.

### Diagnostics

New, specific messages for: arity mismatch on `[…]`/`::[…]`; unknown type
parameter; duplicate type parameter; inconsistent inferred type argument;
uninferable type argument (suggest `::[…]`); ambiguous unqualified enum variant
with all candidate enum names; operation not permitted on an abstract type
variable (D2); agent/exec target may not contain a type variable (D3). Honor the
checker's first-error-abort convention.

## Evaluator & runtime (type erasure + constructor bindings)

Type arguments are **erased**. The evaluator already operates on dynamically
typed `Value`s, but making constructors ordinary values requires explicit
runtime binding support:

- Generic `def`s are ordinary `Closure`s; no type arguments are bound, stored, or
  passed at runtime.
- Add a runtime `ConstructorValue` (or equivalent callable binding) for record
  and payload-variant constructors. It stores only owner/variant identity and
  declared field order; it carries no static type arguments. Calling it builds
  the existing record/enum runtime value.
- Bind nullary enum variants directly as their existing enum runtime values.
- Populate constructor bindings in the evaluator's program environment using
  the same declaration identities recorded by scope resolution. Qualified
  access selects the same constructor value by owner rather than constructing
  eagerly.
- Calling a constructor value after it escapes through a variable uses
  positional fields in declaration order. Direct constructor-binding calls may
  continue accepting named fields through their static `ConstructorSignature`.
- Record/enum runtime values still carry only nominal owner/variant identity and
  field values (no `type_args`); field access, pattern matching, equality, and
  rendering remain unchanged.
- `Call.type_args` remains static-only and is never evaluated.
- No runtime type representation is introduced (D3 keeps the only type-directed
  runtime operation — codec/schema generation — on concrete types).

Confirm constructor binding, shadowing, qualified lookup, first-class
constructor calls, and erasure with the eval/e2e suites.

## Documentation

- **`docs/agl/reference/`** — new "Generics" section (language-user facing, no
  implementation references): declaration syntax for `def`/`record`/`enum`/
  `type`; type application `Name[args]`; inference and the `::[…]` override;
  inferred and explicit generic constructors; unqualified variant ambiguity and
  qualification; case-neutral value names; constructor value bindings and
  first-class use; explicit nullary-constructor pattern syntax; strict
  parametricity (what you can and cannot do with a `T`); the no-generic-agent-
  target rule; invariance. Update any affected pages (lexical structure,
  bindings/scope, expressions, pattern matching, types, functions,
  records/enums, agents).
- **`docs/arch/agl.md`** — extend the type-system section: `TypeVarType`,
  generic-definition templates + eager non-recursive instantiation, name+`type_args`
  nominal identity, constructor schemes/value bindings, the inference/matching
  solver, value-namespace overload sets, and the erasure rationale.
- **`docs/commands.md`** / command help — update only if user-visible CLI
  behavior changes (none expected; verify `--dry-run` inventory still renders
  generic types sensibly).

## Testing

Follow TDD; write failing tests first. Maintain 100% `src/` coverage and the
green e2e gate.

- **Parser/AST** (`test_agl_parser.py`, `test_agl_ast.py`): type-parameter lists
  on all four declarations; `Name[args]` applied types; spaced vs adjacent
  brackets; case-neutral type and value declarations/references; lowercase and
  uppercase constructor names; constructors represented as `VarRef`/`Call`;
  ordinary/one/multiple call-site type arguments; nullary constructor patterns
  requiring `name()` or qualification; migration of existing bare nullary
  pattern fixtures; the LALR conflict-guard.
- **Scope** (`test_agl_scope.py`): record/variant constructor bindings; ordinary
  same-scope collision errors; constructor-only overload sets; ambiguity before
  payload/context/type-argument analysis; qualified resolution; nested ordinary
  bindings shadowing constructor sets; no capitalization-dependent resolution.
- **Type resolution / env** (`test_agl_typecheck.py`): `TypeVarType` resolution
  in scope; `AppliedT` arity errors; parameterized-alias substitution; rejection
  of recursive generic records/enums; nominal identity by
  name+`type_args`; rejection of bare generic nominal/alias names;
  invariance (`list[int]` not assignable to `list[decimal]`/`list[json]` at the
  argument position; `Box[int]` vs `Box[text]`).
- **Inference** (`test_agl_typecheck.py`): `id`, `map`-like signatures;
  argument-driven solving; result-only solving from expected type; explicit
  `::[…]` override; inconsistent-binding error; uninferable-variable error;
  generic constructors (`box`, `Option.some`, payload-free `none` from context);
  explicit `box::[…]`/`some::[…]`; unqualified unique-variant lookup; ambiguity
  whenever two enums declare the same unqualified variant regardless of payload,
  context, or explicit arguments; qualified disambiguation; constructor-as-value
  instantiation; field access and pattern binding on generic instances.
- **Parametricity (D2)**: rejecting `=`/`<`/`+`/`print`/interpolation/field/
  index/`is` on a `T`.
- **D3**: rejecting `ask::[T]` / `exec` / `ask-request` with a type-variable (or
  type-var-containing) target; explicit target arguments bypassing inference;
  explicit target arity errors; allowing the concrete-target cases.
- **D5**: `let f: (int) -> int = id` succeeds; bare `let f = id` (no expectation)
  errors.
- **Eval & e2e** (`tests/agl/programs/`, `test_agl_eval.py`, `test_agl_e2e.py`):
  generic identity/`map`/`Option`/`Result`/`Box`/generic stack programs
  exercising generics together with `case`, records, enums, lists/dicts, and
  higher-order functions; lowercase constructors; constructor shadowing and
  qualified access; first-class payload constructors; nullary constructor
  values/patterns; confirm erasure (a single body/constructor value serves
  multiple instantiations at runtime).

## Milestones

Commit per milestone once `just check` passes (lint + tests + strict mypy).

- **M1 — Syntax, names & AST.** Single `NAME` token (lexer drops the
  `VAR_NAME`/`TYPE_NAME` split and the capitalization branch); case-neutral
  grammar built on `NAME`; constructor declarations/references as normal values;
  deterministic pattern syntax; type-parameter lists; `applied_type`;
  multi-argument typed calls; parser transform; removal of expression-level
  `Constructor`; `Call.type_args`; visitor + exports; REPL highlighter collapses
  to one identifier style; scope binding/collision/overload resolution;
  lexer, conflict-guard, AST, and scope tests. Gate: full `just check` green;
  0 grammar conflicts.
- **M2 — Semantic types & environment.** `TypeVarType`; `type_args` on
  `RecordType`/`EnumType` with name+args identity; `GenericTypeDef` templates +
  eager non-recursive instantiation; parameterized aliases; rejection of bare
  generic names; `FunctionSignature.type_params`; `ConstructorSignature`;
  type-var-aware `resolve_type_expr`; `free_type_vars`/`substitute`/
  `contains_type_var`; capability-gate updates. Unit tests.
- **M3 — Type checker.** Generic `def` checking with parametricity gating; the
  matching/inference solver; declared-call inference (explicit + inferred);
  generic constructor/field/pattern inference; generic-def/constructor-as-value
  instantiation; diagnostics. Checker tests.
- **M4 — Constructor values, erasure guards & evaluation.** Runtime constructor
  bindings/calls and nullary values; D3 target-type rejection; new e2e generic
  programs; eval/e2e tests.
- **M5 — Documentation & polish.** Reference docs, `docs/arch/agl.md`, command
  docs/help review; final full e2e sweep.

## Implementation notes / derived decisions (flagged for owner review)

These follow from D1–D7 rather than being independent architectural choices, but
are surfaced explicitly so they can be vetoed:

1. **Nominal identity moves to name + `type_args`** (fields excluded from
   equality/hashing). Sound because names are unique in the namespace;
   recursive nominal types remain unsupported.
2. **Generic constructors infer from field arguments + expected type**, or take
   explicit arguments through `box::[int](…)` / `some::[int](…)`. Bracket-only
   `box[int](…)` is not constructor syntax.
3. **Bare value-position type application** (`id::[int]` without a call) is not
   added in v1; generic-`def`-as-value relies on the expected type (D5). Optional
   future ergonomic addition.
4. **Inference is incomplete by design** for some higher-order / payload-free
   cases (e.g. passing one generic function to another). The explicit `::[…]`
   override and type annotations are the documented escape hatch.
5. **Coercions do not participate in solving** (`int→decimal`, `json`-absorption
   apply only at the post-substitution assignability check), per the invariant
   rule D6.
6. **Recursive records/enums remain unsupported**, including recursive generic
   definitions. Supporting them later requires a separate cycle-safe semantic
   representation and codec/schema design.
7. **Unqualified enum-variant ambiguity is name-based and precedes inference.**
   Payload, context, and explicit type arguments never choose between enums that
   declare the same variant name; qualification is required.
8. **Constructor names are case-neutral value bindings.** Capitalization never
   classifies a value as a constructor. Bare pattern names are binders;
   nullary-constructor patterns use `name()` or qualification.

## Out of scope (future work)

- **Runtime type reification / dictionary passing** to enable generic
  `ask`/`exec` target types (the alternative to D3's restriction).
- **Bounded polymorphism / type classes** (`T: Comparable`, `T: Renderable`) to
  relax strict parametricity (D2).
- **Pragmatic rendering/equality on `T`** (the D2 "pragmatic" option).
- **Higher-rank polymorphism** and **generic lambdas / polymorphic function
  values** (let-polymorphism), excluded by the prenex restriction (D4/D5).
- **Declaration-site variance annotations** (D6 alternative C).
- **Recursive nominal types**, including recursive generic records/enums.
