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
                 | qual_constructor_pattern               (* module-qualified constructor *)

constructor_pattern      ::= NAME ("." NAME)? ("(" pattern_fields? ")")?
                           | NAME "." NAME                (* qualified nullary *)
qual_constructor_pattern ::= qual_prefix NAME ("." NAME)? ("(" pattern_fields? ")")?

qual_prefix    ::= NAME ("." NAME)* "::"   (* module-qualified prefix *)
               | "::"                      (* current-module prefix *)

pattern_fields ::= pattern_field ("," pattern_field)* ","?
pattern_field  ::= pattern                                (* positional sub-pattern *)
                 | NAME "=" pattern                       (* named sub-pattern: field = subpattern *)
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
- a **qualified** `Enum.variant`.

```agl
Pass                       # nullary variant — bare name that names a constructor
Pass()                     # nullary variant, explicit call form
Review.Pass                # nullary variant, qualified (aliases resolve transparently)
Fail(issues)               # shorthand: binds field 'issues' to name 'issues'
Fail(issues = xs)          # binds field 'issues' to name 'xs'
Fail(issues = ["stuck"])   # nested literal pattern on a field
```

A **bare** constructor name matches **nullary** variants only. A bare name for a
variant that has fields is a static error directing you to an explicit form, so
the discarded payload is acknowledged: write `Fail(...)`, `Fail(_)`, or
destructure the fields. The call and qualified forms apply to every variant;
the bare form is a convenience for the common nullary case.

#### Module-qualified constructor patterns

When a type comes from an imported module, the constructor may be prefixed
with a module qualifier:

```agl
import mylib

case value of
  | mylib::Color.Red  => print "red"
  | mylib::Color.Blue => print "blue"
```

The `qual_prefix` form (a module qualifier `module::`) or the self-reference
form (`::`) may appear before the constructor name. This is useful when two
open-imported modules export enum types with the same variant name, since
qualification always disambiguates.

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

case r of
  | Ok(v)                 => ...  # positional: binds field 'value' to v
  | Ok(value = v)         => ...  # named: same effect
  | Err(reason, fatal)    => ...  # shorthand for Err(reason = reason, fatal = fatal)
  | Err(reason = r, fatal = f) => ...  # fully named
```

Named sub-patterns nest arbitrarily — the sub-pattern may be a wildcard, literal,
binder, or another constructor pattern:

```agl
Fail(issues = ["stuck"])   # nested literal pattern
Fail(issues = xs)          # binds field 'issues' to pattern variable xs
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
  | Fail(issues) => artifact := ask("Fix ${issues}", agent = impl)
  | Pass => ()
```

`is` / `is not` also apply to generic enum instances; qualify the variant the
same way as in a pattern:

```agl
let probe: Option[int] = some(value = 99)
if probe is Option.some => print "probe is some"
if probe is not Option.none => print "probe is not none"
```
