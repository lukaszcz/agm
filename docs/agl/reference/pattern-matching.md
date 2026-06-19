# Pattern Matching

[← Index](index.md)

Patterns appear in `case` statements ([Control flow](control-flow.md)) and
`case` expressions ([Expressions](expressions.md)). Every branch is
introduced by `|`, including the first.

## Pattern forms

```ebnf
pattern        ::= "_"                                    (* wildcard *)
                 | literal                                (* literal pattern *)
                 | NAME                                   (* variable binder *)
                 | constructor_pattern

constructor_pattern ::= NAME ("." NAME)? "(" pattern_fields? ")"
                 | NAME "." NAME                          (* qualified nullary *)

pattern_fields ::= pattern_field ("," pattern_field)* ","?
pattern_field  ::= NAME                                   (* shorthand *)
                 | NAME ":" pattern                       (* field: subpattern *)
```

### Wildcard `_`

Matches anything, binds nothing:

```agl
case result of
  | Complete(output) => artifact := output
  | _ => ()
```

### Variable binder

A **bare name** in a pattern is always a variable binder: it matches anything
and binds the scrutinee to that name as an immutable, branch-local value.
Capitalization carries no meaning ([Lexical structure](lexical-structure.md)),
so this is true regardless of how the name is written — `other`, `Other`, and
`OTHER` are all binders:

```agl
case result of
  | Blocked(reason) => raise Abort(message: reason)
  | other => print other
```

> **A bare name never matches a constructor.** Because there is only one class
> of identifier, a bare name cannot be a nullary variant by virtue of its
> spelling. To match a constructor you must use a form the grammar recognizes
> as a constructor pattern — call syntax `name()` or a qualified
> `Enum.variant` (see below). A bare `Pass` in a pattern binds a variable
> called `Pass`; it does **not** match the variant `Pass`.

### Literal patterns

An `int`, `decimal`, `bool`, `null`, or string literal matches by value
equality (the same `=` semantics, including `int`/`decimal` numeric
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

A constructor pattern matches one enum variant and optionally destructures
its payload. A constructor pattern is distinguished from a bare-name binder by
its **syntax**, not by spelling: it is either a call form `name(…)` (the
parentheses may be empty) or a qualified `Enum.variant`:

```agl
Pass()                     # nullary variant (empty call form)
Review.Pass                # nullary variant, qualified (aliases resolve transparently)
Fail(issues)               # shorthand: binds field 'issues' to name 'issues'
Fail(issues: xs)           # binds field 'issues' to name 'xs'
Fail(issues: ["stuck"])    # nested literal pattern on a field
```

> **Migration note.** A *bare* nullary pattern such as `case Pass` no longer
> matches the variant `Pass` — it now binds a variable. Rewrite each bare
> nullary pattern as a call form (`Pass()`) or a qualified reference
> (`Review.Pass`).

Payload patterns are **field-based, not positional**. `Fail(x)` is valid
only if the variant actually has a field named `x`; to bind field `issues`
to another name, write `Fail(issues: x)`. The general form
`field: pattern` nests arbitrarily — the sub-pattern may itself be a
wildcard, literal, binder, or (where the field is enum-typed) another
constructor pattern.

Static rules:

1. A constructor pattern requires an **enum-typed scrutinee**; matching one
   against any other type is a static error.
2. The variant must belong to the scrutinee's enum; a qualifier must resolve
   (alias-transparently) to that enum.
3. Each field may appear at most once in a pattern.
4. Fields not mentioned in the pattern are simply ignored (patterns need not
   be complete).
5. A name may be bound only once per pattern.

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

def render(o: Option[int]) -> text =
  case o of
    | Option.none => "missing"
    | Option.some(value) => "found ${value}"   # value: int, so it can be interpolated
```

The qualifier (`Option.`) names the owning enum; it is otherwise optional and
serves to disambiguate when two enums share a variant name (see
[Expressions](expressions.md)).

## Matching semantics

1. The scrutinee is evaluated exactly once.
2. Branch patterns are tried **in order**; the first matching branch runs.
3. Pattern variables are bound as immutable values in a fresh branch scope
   ([Bindings and scope](bindings-and-scope.md)).
4. If no branch matches, **`MatchError`** is raised, carrying the
   scrutinee's type name and its JSON representation
   ([Exceptions](exceptions.md)).

## Exhaustiveness

Exhaustiveness is **advisory, not enforced**. When a `case` scrutinizes an
enum and its constructor patterns do not cover every variant — and no
wildcard or variable binder is present — the checker emits a *warning*:

```text
Non-exhaustive case on enum 'Review': missing variant(s) Fail.
An unmatched value raises MatchError at runtime.
```

The program still runs; an unmatched value raises `MatchError`. Scrutinees
of non-enum type are not analyzed for exhaustiveness.

## `is` versus `case`

Use `is` / `is not` ([Expressions](expressions.md)) to *test* a variant
without destructuring — typically in `if` and `until` conditions. Use `case`
when you need the payload:

```agl
until review is Pass

case review of
  | Fail(issues) => artifact := ask("Fix ${issues}", agent: impl)
  | Pass() => ()
```

`is` / `is not` also apply to generic enum instances; qualify the variant the
same way as in a pattern:

```agl
let probe: Option[int] = some(value: 99)
if probe is Option.some => print "probe is some"
if probe is not Option.none => print "probe is not none"
```
