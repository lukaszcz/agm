# Pattern Matching

[← Index](index.md)

Patterns appear in `case` statements ([Control flow](control-flow.md)) and
`case` expressions ([Expressions](expressions.md)). Every branch is
introduced by `|`, including the first.

## Pattern forms

```ebnf
pattern        ::= "_"                                    (* wildcard *)
                 | literal                                (* literal pattern *)
                 | VAR_NAME                               (* variable binder *)
                 | constructor_pattern

constructor_pattern ::= TYPE_NAME ("." TYPE_NAME)? ("(" pattern_fields? ")")?

pattern_fields ::= pattern_field ("," pattern_field)* ","?
pattern_field  ::= VAR_NAME                               (* shorthand *)
                 | VAR_NAME ":" pattern                   (* field: subpattern *)
```

### Wildcard `_`

Matches anything, binds nothing:

```agl
case result of
  | Complete(output) => set artifact = output
  | _ => pass
```

### Variable binder

A lowercase name matches anything and binds the scrutinee to that name as an
immutable, branch-local value:

```agl
case result of
  | Blocked(reason) => raise Abort(message: reason)
  | other => print other
```

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
its payload:

```agl
Pass                       # nullary variant
Review.Pass                # qualified (aliases resolve transparently)
Fail(issues)               # shorthand: binds field 'issues' to name 'issues'
Blocked(reason: why)       # binds field 'reason' to name 'why'
Blocked(reason: "stuck")   # nested literal pattern on a field
```

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
  | Fail(issues) => set artifact = impl "Fix ${issues}"
  | Pass => pass
```
