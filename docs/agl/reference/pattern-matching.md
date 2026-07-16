# Pattern Matching

[← Index](index.md)

Patterns appear in `case` expressions ([Control flow](control-flow.md),
[Expressions](expressions.md)). The `|` before the first branch is optional;
each additional branch is introduced by `|`.

## Pattern forms

```ebnf
pattern        ::= "_"                                    (* wildcard *)
                 | literal                                (* literal pattern *)
                 | name                                   (* binder or bare constructor *)
                 | name "(" pattern_fields? ")"          (* unqualified constructor *)
                 | qual_prefix type_qual? name
                     ("(" pattern_fields? ")")?           (* qualified constructor *)

qual_prefix    ::= module_path "::"         (* module or type prefix *)
                 | "::"                     (* current-module prefix *)
type_qual      ::= name "::"                (* owning type after a module prefix *)
module_path    ::= NAME ("." NAME)*
name           ::= NAME | OP_NAME
field_name     ::= NAME | "agent" | "to" | "downto" | "by"

pattern_fields ::= pattern_field ("," pattern_field)* ","?
pattern_field  ::= pattern                                (* positional sub-pattern *)
                 | field_name "=" pattern                 (* named sub-pattern *)
```

### Wildcard `_`

Matches anything, binds nothing:

```agl
case result of
  | Complete(output) => artifact := output
  | _ => ()
```

### Variable binder

A bare name in a pattern is a **variable binder** — it matches anything and
binds the scrutinee to that name as an immutable, branch-local value — *unless*
the name denotes an in-scope constructor, in which case it is a **constructor
pattern** for that variant (see [Constructor patterns](#constructor-patterns)
below). What a bare name denotes is fixed by whether it resolves to a
constructor, never by its spelling: capitalization carries no meaning
([Lexical structure](lexical-structure.md)). So `other`, `result`, and
`leftover` are binders precisely when no constructor of that name is in scope:

```agl
case result of
  | Blocked(reason) => raise Abort(message = reason)
  | other => print other            # 'other' names no constructor → binder
```

A nearer ordinary binding — a `let`, `var`, parameter, or enclosing pattern
variable — **shadows** a constructor, exactly as in expression position
([Bindings and scope](bindings-and-scope.md)). Under such a shadow a bare name
the constructor would otherwise claim is again a plain binder.

### Literal patterns

An `int`, `decimal`, `bool`, `null`, or string literal matches by value
equality (the same `==` semantics, including `int`/`decimal` numeric
equivalence — the pattern `1` matches the value `1.0`):

```agl
case attempt of
  | 1 => print "first try"
  | _ => print "retry"
```

Restrictions:

- A pattern string literal cannot contain interpolation.
- There are no negative-number patterns (patterns have no unary minus).
- The literal's type must be comparable with the scrutinee's static type
  (same type after `int → decimal` widening); an `int` pattern against a
  `text` scrutinee — or any scalar literal against a `json` scrutinee other
  than `null` — is a static error, not a silently dead branch.

### Constructor patterns

A constructor pattern matches one enum variant and optionally destructures its
payload. A pattern is a constructor pattern when it is one of:

- a **bare name that denotes an in-scope constructor** — matches that variant
  (nullary variants only; see below),
- a **call form** `name(…)`, where the parentheses may be empty, or
- a **qualified** `Enum::variant` or `module::Enum::variant` form.

```agl
enum Review
  | Pass
  | Fail(reason: text)

def summarize(review: Review) -> text =
  case review of
    | Review::Pass => "passed"              # qualified nullary constructor
    | Fail(reason = "stuck") => "blocked"  # nested scalar literal
    | Fail(reason = other) => other          # named binder
```

The first branch could equivalently use bare `Pass` or explicit `Pass()`.

A **bare** constructor name matches **nullary** variants only. A bare name for a
variant that has fields is a static error directing you to an explicit form, so
the discarded payload is acknowledged: write `Fail()` or destructure the
fields. Empty parentheses ignore every payload field, including named-only fields. The call and qualified forms apply to every variant;
the bare form is a convenience for the common nullary case.

When two enums share an unqualified variant spelling, a bare reference in
expression position is a static ambiguity error ([Bindings and scope](bindings-and-scope.md)),
but in a pattern the scrutinee's static type selects the intended enum, so a
bare pattern needs no qualification.

#### Module-qualified constructor patterns

When a type comes from an imported module, the constructor may be prefixed
with a module qualifier. Both the module/type boundary and the type/variant
boundary use `::`:

```agl
import mylib

case value of
  | mylib::Color::Red  => print "red"
  | mylib::Color::Blue => print "blue"
```

The prefix may name an owning type (`Color::Red`), a module and owning type
(`mylib::Color::Red`), or the current module (`::Color::Red`). A module may
also qualify an exposed constructor directly (`mylib::Red`). This is useful
when two open-imported modules export enum types with the same variant name,
since qualification always disambiguates. Dotted names occur only inside a
module path such as `company.colors::Color::Red`; constructor qualification
itself uses `::`, never `.`.

**Payload sub-patterns** follow the same positional-greedy binding as calls:

- **Positional sub-patterns** fill positional-capable (pos-only/standard) field
  slots left to right, in declaration order.
- **Named sub-patterns** (`field = pattern`) bind a specific field by name; they
  may follow positional sub-patterns.
- **Bare-name shorthand** (`x` in a position where positional slots are exhausted)
  means `x = x` — it binds field `x` to pattern variable `x`. Only valid when the
  bare name lands on a named-only field.

```agl
enum Result
  | Ok(value: int)           # single field, standard zone
  | Err(reason: text, fatal: bool)   # two fields, named-only by default

# Each arm below is an alternative spelling for a separate case:
case r of
  | Ok(v) => ...  # positional: binds field 'value' to v
  | Err(reason, fatal) => ...  # shorthand for Err(reason = reason, fatal = fatal)

# The equivalent fully named spellings are Ok(value = v) and
# Err(reason = r, fatal = f), respectively.
```

Named sub-patterns nest arbitrarily — the sub-pattern may be a wildcard, literal,
binder, or another constructor pattern. Here the final binder is required because
`int` is an open domain:

```agl
enum Response
  | Complete
  | Failed(code: int)

def describe(response: Response) -> text =
  case response of
    | Complete => "complete"
    | Failed(code = 503) => "unavailable"  # nested scalar literal
    | Failed(code = code) => "error ${code}"  # named binder covers other ints
```

Static rules:

1. A constructor pattern requires an **enum-typed scrutinee**; matching one
   against any other type is a static error.
2. The variant must belong to the scrutinee's enum; a qualifier must resolve
   (alias-transparently) to that enum.
3. Each field may appear at most once in a pattern.
4. Fields not mentioned in the pattern are simply ignored (patterns need not
   be complete).
5. A name may be bound only once per pattern.
6. A positional sub-pattern must precede all named sub-patterns. A positional
   expression that lands on a named-only field (with no positional slots available)
   is a static error unless it is a bare name (which is reinterpreted as `name = name`).

### Patterns on generic enums

Constructor patterns work on instances of **generic** enums
([Generics](generics.md)) exactly as on monomorphic ones, in both unqualified
and qualified form. When the scrutinee is a concrete instance such as
`Option[int]`, a destructured payload is bound at the **instantiated** type —
matching `some(value)` against an `Option[int]` binds `value: int`:

```agl
enum Option[T]
  | none
  | some(value: T)

def describe_option(o: Option[int]) -> text =
  case o of
    | Option::none => "missing"
    | Option::some(value) => "found ${value}"   # value: int, so it can be interpolated
```

The qualifier (`Option::`) names the owning enum; it is otherwise optional and
serves to disambiguate when two enums share a variant name (see
[Expressions](expressions.md)).

## Matching semantics

1. The scrutinee is evaluated exactly once.
2. Branch patterns are tried **in order**; the first matching branch runs.
3. Pattern variables are bound as immutable values in a fresh branch scope
   ([Bindings and scope](bindings-and-scope.md)).
4. Every possible scrutinee value must match a branch, and every branch must
   be selectable for at least one value. Violations are static errors, so a
   valid `case` always selects a branch.

## Exhaustiveness and redundancy

Every `case` must be **exhaustive**: its arms, taken together, must cover every
value of the scrutinee type. Coverage includes the complete nested pattern,
not only the outer constructor. For example, matching `Some(true)` and `None`
does not cover `Some(false)`.

Enum variants and the two boolean values form closed domains and can be
covered by listing every remaining constructor or literal. This rule applies
recursively to enum payloads: a payload of enum or `bool` type may itself be
covered with a complete set of nested patterns.

The domains of `int`, `decimal`, `text`, `json`, lists, dictionaries, records,
exceptions, `unit`, agents, functions, and an unresolved type parameter are open.

`bottom` has an empty domain: a diverging expression (`raise`, `return`, `break`,
or `continue`) produces no value to match. Consequently, every arm written for
a `case` over `bottom` is redundant.
A finite set of literals cannot exhaust such a domain; the remaining values
require an irrefutable wildcard or variable-binder pattern. The same requirement
applies when an open domain occurs inside a constructor payload.

Every arm must also be **non-redundant**. Because matching uses source order,
an arm is redundant when all values it could match are already covered by
earlier arms. Duplicate arms and arms after a wildcard are common examples.
Partial overlap is allowed when the later arm still matches at least one new
value.

`MatchError` is an ordinary constructible exception, not an automatic `case`
fallback. To reject part of a domain deliberately, cover it with a final arm
that explicitly raises the exception ([Exceptions](exceptions.md)):

```agl
enum Response
  | Complete
  | Rejected

let response = Complete
case response of
  | Complete => response
  | _ =>
    raise MatchError(
      message = "response rejected by this workflow",
      scrutinee_type = "Response",
      scrutinee = response as json,
    )
```

## `is` versus `case`

Use `is` / `is not` ([Expressions](expressions.md)) to *test* a variant
without destructuring — typically in `if` and `until` conditions. Use `case`
when you need the payload:

```agl
until review is Pass

case review of
  | Fail(issues) => artifact := ask("Fix ${issues}", agent = impl)
  | Pass => ()
```

`is` / `is not` also apply to generic enum instances; qualify the variant the
same way as in a pattern:

```agl
let probe: Option[int] = some(value = 99)
if probe is Option::some => print "probe is some"
if probe is not Option::none => print "probe is not none"
```
