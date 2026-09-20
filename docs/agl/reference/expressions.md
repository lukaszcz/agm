# Expressions

[← Index](index.md)

This chapter covers literals, constructors, member access, record updates,
operators, calls, and `case`/`if` expressions, together with the static typing
rules and runtime semantics of each. Operator precedence is tabulated in
[Lexical structure](lexical-structure.md).

In AgL **everything is an expression**: there is no separate statement
category. Bindings, `:=`, `print`, `if` without `else`, and loops are all
expressions with well-defined types. An executable block, such as a function or
branch body, is a sequence of items whose value is the value of its last item.
A final `let` or `var` contributes `unit` as the block value, or bottom when its
initializer exits. A file-backed module root is static and does not produce a
block value.

## Discarded values

A bare expression before the final block item is evaluated in a discarded-value
position and must have type `unit` or `bottom`. `bottom` is the type of an
expression that does not return normally, such as `raise`, `return`, `break`,
or `continue`; it is assignable wherever a value is expected. Binders and
declarations are valid intermediate items. A `unit`-valued call is written
bare; to run a value-producing expression for effect without retaining its
result, bind it to `_`:

```agl
program def main() -> unit =
  let _ = ["report"]
  print "report fetched"
```

`_` creates no readable name and may be used repeatedly. The same rule applies
to a loop body, whose value is discarded on every iteration.

## Literals

<!-- agl-check: fragment -->
```agl
42            # int
1.5           # decimal
true false    # bool
null          # json
()            # unit — printable unit value
"text %{x}"   # template (type text; see Strings and interpolation)
```

The unit literal `()` is the unit value. `()` is also the empty argument list
of a zero-argument call — the two are syntactically unified.

### Array literals

```agl
let issues = ["missing tests", "unclear API"]
```

Elements must share a type, up to `int → decimal` widening. Under an expected
type, each element is checked against the expected element type. An empty array
may obtain its element type from an expected container type or another
constraint in the same enclosing expression:

```agl
def second[T](left: T, right: T) -> T = right

program def main() -> unit =
  let items = second([], [1])
```

If no such constraint determines the element type before the enclosing
expression ends, add an annotation:

<!-- agl-check: fragment -->
```agl
let items: array[Issue] = []
```

Under an expected type of `json`, each element is checked against `json`
directly, so elements need **not** share a type — `[1, "two", true, null]` is
a valid `json` value — and the literal itself is typed as `json`, not
`array[json]`: a fresh literal is not an existing value being absorbed, so no
`as json` cast is needed at that boundary. The same rule applies to
dictionary literals below.

### Dictionary literals

```agl
let metadata: dict[text, json] = {
  "source": "reviewer",
  "attempt": 2,
}
```

Keys are literal strings; an unquoted identifier key is shorthand for the same
string. Interpolated keys are rejected. Duplicate keys are a static error. An
empty dictionary may obtain its value type from an expected dictionary type or
another constraint in the same enclosing expression; otherwise it needs an
annotation:

```agl
let metadata: dict[text, json] = {}
```

## Constructors

```ebnf
constructor ::= constructor_ref value_type_args? constructor_args?
constructor_ref ::= name
                  | qualifier_chain name
                  | applied_type_qualified_constructor
applied_type_qualified_constructor ::= qualifier_chain NAME "[" type_expr ("," type_expr)* "]" "::" NAME
                                           | NAME "[" type_expr ("," type_expr)* "]" "::" NAME
                                           (* `[` is byte-adjacent to the preceding NAME *)
qualifier_chain ::= "::" qualifier_segment* | qualifier_segment+
qualifier_segment ::= ["/"] NAME ("/" NAME)* "::"
                    | NAME "[" type_expr ("," type_expr)* "]" "::"
constructor_args ::= "(" (ctor_arg ("," ctor_arg)* ","?)? ")"
value_type_args ::= "::" "[" type_expr ("," type_expr)* "]"
ctor_arg    ::= element_expr      (* positional; expr without a bare record update *)
              | field_name "=" element_expr
```

Constructor arguments follow the same **positional-greedy** binding as function
calls — positional arguments fill positional-capable (pos-only/standard) field
slots left to right; named arguments (`field = value`) follow. The optional
value-position `::[…]` pins the type arguments of a generic constructor (see
[Generic constructors](#generic-constructors)). In the explicit
applied-type-qualified form, the `[` must be byte-adjacent to the type name:
`Option[int]::Some` is valid, while `Option [int]::Some` is not. This
restriction does not apply to ordinary applied types, so both `Option[int]` and
`Option [int]` are valid type expressions. This form requires `NAME` for both
the applied type and constructor; it does not accept `OP_NAME` there.

**Per-type field zones.** Record fields, inline enum-member fields, and an
exception's own fields default to the **standard** zone (positional or named),
regardless of the number of fields. The `@arg-pos`, `@arg-std`, and
`@arg-named` [attributes](attributes.md#zone-attributes) constrain fields to a
different zone. An exception's inherited `message` field
is named-only.

**Bare-name shorthand.** A bare name `x` in a positional slot that lands on a
**named-only** field (where positional binding is impossible) is reinterpreted as
`x = x`. This shorthand applies in any call context — functions and constructors
alike — whenever a named-only parameter is in play.

### Record construction

<!-- agl-check: fragment -->
```agl
Issue(title = "Bug", severity = 2, description = "...")
```

Every declared field must be supplied; unknown and duplicate fields are
static errors.

### Enum member construction

An enum establishes a same-named scope and each inline member is a record in
that scope. Qualification uses the same chain syntax as every other scope
member; every enum member also contributes its terminal name as an injected
bare constructor candidate.

Qualified or unqualified:

<!-- agl-check: fragment -->
```agl
Review::Pass
Review::Fail(issues = ["missing tests"])

let review: Review = Pass           # checked in an enum-typed slot
```

A member constructor produces its own record type. Assign it to an enum slot
to widen it: `let pass = Pass` has type `Review::Pass`, while the annotated
binding above has type `Review`. This is a directed check, not common-type
inference: `let items = [Pass, Fail(issues = [])]` has no inferred enum type,
so write `let items: array[Review] = [Pass, Fail(issues = [])]`.

In ordinary value position, an unqualified member name must resolve to exactly
one visible constructor candidate in lexical scope. An expected enum type
checks the selected constructor but cannot choose between same-named
candidates. A nullary member is constructed by writing its name alone (no
parentheses). Field-bearing members use positional-greedy binding. Every
unmarked member-record field is standard (positional or named), regardless of
the number of fields.

```agl
enum Outcome
  | Ok(value: int)
  | Err(reason: text, fatal: bool)

let ok = Outcome::Ok(42)
let ok2 = Outcome::Ok(value = 42)
let err = Outcome::Err("bad", false)
let named-err = Outcome::Err(reason = "bad", fatal = false)
```

### Unqualified member ambiguity

If two or more visible constructor candidates have the same unqualified name,
a bare reference in ordinary value position is a **static scope ambiguity
error**, even in a context with an expected enum type. Scope reports the
ambiguity before type checking can use that type. Disambiguate by qualifying
with the member's declaring record or, for an inline member, its owning enum:

```agl
enum Holder[T]
  | Empty
  | Tagged(by: T)

enum Other
  | Tagged(name: text)

let h: Holder[int] = Holder::Tagged(by = 7)   # qualified; unqualified 'Tagged' is an error
```

A nearer binding (a `let`/`var`/parameter of the same name) **shadows** a
constructor or an overloaded set of constructors entirely; within that scope
the name refers to the binding, not the constructor.

### Generic constructors

The constructors of a generic record or enum ([Generics](generics.md)) are
generic too. Their type arguments are normally inferred — from constructor
arguments, the expected type, or other evidence in the surrounding expression:

```agl
record Box[T]
  value: T

let bi: Box[int] = Box(value = 5)        # T = int, inferred from the argument
let bt: Box[text] = Box(value = "hi")    # same definition, T = text
```

Pin the instantiation explicitly with `::[…]` when inference cannot (or
should not) determine it:

<!-- agl-check: fragment -->
```agl
let be = Box::[int](value = 99)
```

This also applies when a field-bearing constructor value or partial
constructor is an argument to another call: a sibling argument may determine
its type arguments.

```agl
record Box[T]
  value: T

def build[T](factory: (T) -> Box[T], value: T) -> Box[T] = factory(value)

program def main() -> unit =
  let b = build(Box(value = ?), 5)
```

A fieldless member of a generic enum may need contextual evidence (or an
owner-applied qualification) to determine the enum instantiation:

```agl
enum Option[T]
  | None
  | Some(value: T)

let e: Option[int] = None          # T = int, fixed by the annotation
let s = Some::[int](value = 1)      # T pinned explicitly
let q = Option[int]::Some(value = 2) # qualification disambiguates the owner
```

### Field-bearing constructors as values

A constructor with fields is an **ordinary function value**: it can be stored,
passed to a function, and called like any other function value. When a
constructor is reached **through a variable** rather than written directly, it
is a positional callable — its arguments are supplied positionally in
**declaration order**, since a function value has no named parameters
([Functions](functions.md)):

<!-- agl-check: fragment -->
```agl
let mk: int -> Box[int] = Box     # the constructor as a value
let made = mk(1)                    # called positionally
print made.value
```

This lets constructors be passed to higher-order functions:

<!-- agl-check: fragment -->
```agl
def apply[A, B](x: A, f: A -> B) -> B = f(x)

let built = apply(42, mk)           # mk applied inside a generic HOF
print built.value
```

A **generic** constructor used as a value needs expression-local evidence to
fix its instantiation, exactly like a generic `def` used as a value. An
annotation supplies that evidence, and a surrounding higher-order call may
supply it through another argument or its result. A bare `let f = Some` is a
static error because the binding has no such evidence.

### Fieldless constructor references

A fieldless constructor reference constructs its value immediately in value
position. This applies uniformly to a standalone record and an enum member,
whether bare or qualified:

```agl
record Marker()
enum Status
  | Ready
  | Failed(reason: text)

let marker = Marker
let ready = Ready
let qualified-ready = Status::Ready
```

Calls remain direct constructor calls, so `Marker()` and `Status::Ready()`
construct the same values as the bare references above. A fieldless constructor reference is not a `() -> T` function
value. Supply an explicit function when one is required:

```agl
record Marker()
def invoke(factory: () -> Marker) -> Marker = factory()

program def main() -> unit =
  let result = invoke(fn() => Marker)
```

A generic fieldless constructor may obtain its type arguments from context or
from `::[…]`:

```agl
record Token[T]()
let from-context: Token[int] = Token
let explicit = Token::[int]
```

### Exception construction

Built-in exception types are constructed like records:

```agl
program def main() -> unit =
  raise Abort(message = "Cannot continue.")
```

## Member access

`expr.member` projects either a field or a method. Fields belong to records,
exceptions, and `ExecResult`; methods belong to records, enums, exceptions, and
built-in receiver types. A field projection yields its field value. An ordinary
or `extern` method projection yields a bound function value whose receiver is
the value on the left of the dot. A `builtin def` method projection does the
same, lowering the eventual value call to its host operation.

```agl
record Meter
  var value: int

def Meter::add(self, amount: int) -> int = self.value + amount

program def main() -> unit =
  let meter = Meter(value = 4)
  let value = meter.value
  let add = meter.add
  let plus = meter.add(?)
  meter.value := 7
  print(value)
  print(add(3))
  print(plus(5))
```

Thus `meter.add(3)` calls the method with `meter` as its receiver, while
`meter.add` can be stored, passed to another function, or partially applied.
A leading-dot invocation such as `.add(3)` instead produces a unary function;
its receiver type comes from the surrounding function context. This supports
forms such as `meters |> .map(.add(3))`. See
[Methods](functions.md#methods) for its contextual typing rules.
Method selection uses the receiver's static type and the declaration routes
visible in this module. A same-named field and visible method are ambiguous,
including as an assignment target. Arrays and dictionaries have no fields; use
indexing to read their elements or values. `std/prelude` re-exports their
receiver scopes, so their methods are available wherever the prelude is
enabled. With `--no-stdlib`, import a route to the method before calling it.

A member record value exposes its own fields and methods, plus — as part of
the same selection level — the methods of every current enum that declares or
references it; selecting one of those widens the receiver to the enum (see
[Methods](functions.md#methods)). An enum-typed value exposes only methods
declared by that enum: it has no fields, even when every member record
defines the same field. Use a pattern or a member-record cast before
accessing a member-only field or method from an enum value.

A field assignment `receiver.field := value` requires `field` to be declared
with `var` on a record or enum-member record. It updates that field in place,
so aliases of `receiver` observe the new value. A `let` receiver is valid: it
prevents rebinding the name, not updating a `var` field. An enum-typed receiver
has no fields; narrow it with a `case` pattern or cast it to a member record
before assignment. Exceptions and fields without `var` cannot be assigned.
See [Bindings and scope](bindings-and-scope.md#--destructive-assignment) for
assignment targets, evaluation order, and cycle behavior.

## Record update

`target with field = value, ...` builds a **shallow copy** of a **record** or
**exception** value with the listed fields replaced; an enum member record is
a record for this rule. All other fields keep their values — an unlisted array
or dict field is shared with the target, not copied, so mutating it through the
update's result is observed through the
target too. The target itself is unchanged, even when it has `var` fields:

<!-- agl-check: fragment -->
```agl
let base = Issue(title = "Bug", severity = 2, description = "...")
let urgent = base with severity = 5, title = "Bug!"
```

The result has the target's static type. Every listed field must exist on
that type, each value must be assignable to the declared field type (with the
usual expected-type propagation and `int` → `decimal` coercion), and listing
the same field twice in one update is a static error. Thus `with` rebuilds a
new outer value, while `:=` updates a `var` field of the existing value in
place. `with` does not apply
to enums (match and reconstruct instead), dictionaries, arrays, or `json`
values. The target is evaluated once, then the update values left to right.

See [Types](types.md#copying-values) for the general-purpose `copy` and
`shallow-copy` built-ins, which work over any type rather than only records
and exceptions.

Update values are ordinary expressions in the enclosing scope — there is no
implicit field scope. To derive a new value from an old field, read it off the
target:

<!-- agl-check: fragment -->
```agl
let louder = issue with severity = issue.severity + 1
```

**Binding and chaining.** `with` binds looser than every operator, on both
sides: the target is the whole operator expression before `with`, and the
update list extends to the end of the enclosing expression. Comparing an
updated value therefore requires parentheses — `(r with x = 1) == r2`.
Chaining is left-associative: `r with x = 1 with y = 2` updates the result of
the first update. An update as an update *value* must be parenthesized:
`r with inner = (r.inner with x = 1)`.

**Restricted positions.** In comma-separated element positions — call
arguments, array and dict literal elements — and in inline `=>` branch bodies,
a bare `with` would be ambiguous against the enclosing array or branch
separator, so the whole update expression must be parenthesized there:
`f((r with x = 1), y = 2)`. Suite (indented) bodies have no such restriction.

Updating an exception through a binding typed as a base exception preserves
the value's concrete exception type — see
[Exceptions](exceptions.md).

## Indexing

`expr[index]` reads from an array, dictionary, or text:

<!-- agl-check: fragment -->
```agl
let third = xs[2]
let last = xs[-1]
let value = metadata["source"]
let initial = "Ada"[0]
```

Indexing is a postfix operator and may be chained with calls and field access:

<!-- agl-check: fragment -->
```agl
let cell = matrix[0][1]
let name = rows[0].name
let item = make-items()[0]
```

Whitespace matters. `xs[0]` is indexing because the `[` is adjacent to `xs`.
`f [0]` remains the single-argument call sugar `f([0])`.

Array and text indexes must be `int`. Negative indexes count from the end, as
in Python: `xs[-1]` selects the last element. A text index returns one Unicode
code point. An out-of-range array or text index raises catchable `IndexError`
with `index`, `length`, and `message` fields. Text is immutable, so it cannot
be an indexed-assignment target.

Dictionary indexes must be `text`. Missing keys raise catchable `KeyError`
with `key` and `message` fields.

## Calls

All calls use the same uniform parenthesized syntax. This applies equally to
user `def`s, built-in functions (`ask`, `exec`, `print`, `render`, `copy`,
`shallow-copy`, `parse`, `try-parse`, `resource`, `resource-dir`), and
function values stored in bindings:

```ebnf
call_expr ::= postfix "(" arg_list? ")"
arg_list        ::= arg ("," arg)* ","?
arg             ::= element_expr                 (* positional *)
                  | placeholder_arg              (* positional hole *)
                  | field_name "=" element_expr  (* named *)
                  | field_name "=" placeholder_arg (* named hole *)
placeholder_arg ::= "?" | "?<digits>"
```

`element_expr` is `expr` without a bare record update — see
[Record update](#record-update).

The `postfix` callee may already carry explicit type arguments, so
`id::[int](5)` is a typed call.

**Single-argument sugar.** When there is exactly one positional argument and
no named arguments, the parentheses may be dropped:

<!-- agl-check: fragment -->
```agl
print review          # equivalent to print(review)
ask "Hello?"          # equivalent to ask("Hello?")
print res.stdout      # field-access path is valid sugar argument
print classify(x)     # equivalent to print(classify(x))
f Option::Some(value = 1)  # equivalent to f(Option::Some(value = 1))
ask::[int] "How many?"     # equivalent to ask::[int]("How many?")
```

A callee's explicit type arguments carry over to the sugar, so `f::[int] x`
is the typed call `f::[int](x)`.

Application binds **tighter than all operators**:

<!-- agl-check: fragment -->
```agl
print x + 1           # parsed as (print x) + 1
```

With user-defined symbolic infix operators, `OP_NAME` after an expression is an
operator position. To pass an operator-name value as a call argument, use
parentheses: `print(>>)`.

For details on named arguments, defaults, function types, and partial
application with placeholder arguments, see [Functions](functions.md). For
`ask`'s named parameters, see [Agent calls](agent-calls.md).

## `print`

`print` is a built-in function that accepts one argument of any type, writes
its rendered value (followed by a newline) to the host's standard output, and
returns `()`. It renders with `pretty = false` and `quote-strings = false`:

<!-- agl-check: fragment -->
```agl
print "Review round failed; retrying."
print review                           # renders review in AgL form
print(classify(-4))                    # compound argument needs parens
print res.stdout                       # field-access chain as sugar arg
```

`print` raises `CyclicValueError` when its argument contains a reference
cycle. `print` is a generic function value when an expected type or explicit
type argument fixes its input, for example `let f: text -> unit = print` or
`let f = print::[text]`. An explicit type argument
(`print::[decimal](5)`) is accepted, requires the argument to be
assignable to it, and prints the argument coerced to that type — so
`print::[json]("hi")` prints the quoted json form `"hi"`, not the bare text
`hi`.

## `render`

`render` is a built-in function that converts any value to `text` using the
same renderer as interpolation, `print`, casts to `text`, and REPL echo.

<!-- agl-check: fragment -->
```agl
render(value: T, pretty: bool = true, quote-strings: bool = true) -> text
```

`pretty` selects single-line versus multi-line indented rendering for
structured values and JSON. `quote-strings` controls only a top-level `text`
argument; when it is `false`, rendering text is identity.

```agl
program def main() -> unit =
  let _ = render("hi")
  let _ = render("hi", quote-strings = false)
  let _ = render([1, 2])
  let _ = render([1, 2], pretty = false)
```

`render` raises `CyclicValueError` when its argument contains a reference
cycle. `render` is a generic function value when an expected type or explicit
type argument fixes its input, for example `let f: json -> text = render`. An
explicit type argument (`render::[decimal](5)`) is accepted, requires the argument to be
assignable to it, and renders the argument coerced to that type — so
`render::[json]("hi", quote-strings = false)` renders the quoted json form
`"hi"`: `quote-strings` controls only a top-level `text` argument, and the
coerced argument has type `json`.

## JSON parsing

The language has no parsing form for `json` values: text is parsed by ordinary
standard-library functions, which are values like any other and can be passed
where their function type is expected.

## `resource` and `resource-dir`

`resource(path: path) -> path` returns the absolute path of an existing packaged
or adjacent resource. Its only argument must be a text literal using a relative,
forward-slash path with no `..` segment; it cannot be computed or passed through a
binding. `resource-dir() -> path` returns the same absolute resource anchor.

Resources in a loose module are anchored at that module's directory. Resources in a
[package-owned](packages.md) module are anchored at the package root, so they remain stable when the
module is imported from another project. A module with no backing file has no resource
anchor, so either call in such a module is a static error. Both calls are constant
expressions and may initialize root bindings and `builtin var` defaults. They resolve
during linking; a missing target is a static error. Package checks and package creation verify each
literal resource target. `resource` produces a path only; use `std/fs` to
perform filesystem effects.

<!-- agl-check: fragment -->
```agl
let prompt-path = resource("prompts/review.md")
let package-root = resource-dir()
```

## `copy` and `shallow-copy`

`copy` and `shallow-copy` are the built-ins that ask for an independent value
when binding is by reference — see [Copying values](types.md#copying-values)
for the full deep-vs-shallow model and how each treats a cyclic value.

## `parse` and `try-parse`

`parse` and `try-parse` read a text as a value of a data type, like a
`text`-to-`T` cast but raising `ValueParseError` or returning a `Result` —
see [Parsing values](types.md#parsing-values).

## Operators

### Equality: `==` and `!=`

`==` is **equality** (a single `=` is never a comparison — it is a
binder/named-argument separator). Both operands must have the same type after
`int → decimal` widening. Equality is full value equality
([Types](types.md)).

Operands whose type is, or transitively contains, a function or `unit` value
are a static error — this applies to bare values as well as to
containers (`array`, `dict`), records, enums, or exceptions that hold such
a type at any depth.

`==` is non-associative; `x == y == z` is a parse error.

### Ordering: `<` `<=` `>` `>=`

Both operands must be numeric or both `text`. Text ordering is lexicographic
by code point.

### Membership: `in`

<!-- agl-check: fragment -->
```agl
issue in issues          # element membership:  issues: array[T]
"source" in metadata     # key membership:      metadata: dict[text, V]
"missing" in body        # substring:           both text
```

### Arithmetic: `+` `-` `*` `/` and unary `-`

1. Both operands must be numeric. `+ - *` on two `int` values yield `int`; if
   either is `decimal`, the result is `decimal`.
2. `/` **always yields `decimal`**, even for two `int` operands.
3. Division by zero raises `ArithmeticError` at runtime.
4. Unary `-` negates an `int` or `decimal`.

Text is concatenated with the prelude operator `++`, not `+`; see
[Prelude combinators](lexical-structure.md#prelude-combinators).

### Boolean: `and`, `or`, `not`

Operands must be `bool`. `and` and `or` short-circuit.

### Casts: `as` and `as?`

<!-- agl-check: fragment -->
```agl
EXPR as T     # cast: convert EXPR to type T
EXPR as? T    # optional cast: Option[T], never raises
```

`as` converts the value to the named type; `as?` performs the same conversion
without raising, yielding `Some(value)` carrying the converted value on
success and `None` on failure — so a successful test also supplies the value.
`as text` and `as json` raise `CyclicValueError` when conversion walks a
reference cycle; their `as?` forms yield `None` instead. Casting from an enum
to one of its member records is an identity downcast;
casting a member record to a containing enum is an identity upcast. Enums can
be cast when they share constructors; the runtime constructor must belong to
the target enum. An exception value follows the same identity-cast shape over
its `extends` chain: casting to itself or an ancestor, including the root
`Exception`, is a no-op; casting to a descendant checks the runtime type.
Unrelated exception types, including siblings, are a static error.
The full conversion matrix and semantics are in
[Types](types.md#casts-and-convertibility).

**Precedence.** Cast operators sit between unary `-` (tighter) and `* /`
(looser). They are **left-associative**:

| Expression | Parsed as |
| ---------- | --------- |
| `-1 as text` | `(-1) as text` — unary minus binds tighter |
| `2 * 3 as text` | `2 * (3 as text)` — `*` is looser than cast |
| `1 + 2 as text` | `1 + (2 as text)` — `+` is looser than cast |
| `f x as int` | `(f x) as int` — application binds tighter |
| `x as json as text` | `(x as json) as text` — left-associative |
| `a as? int == b` | `(a as? int) == b` — `==` is looser than cast |

**`as?` is a single token**: the `?` is part of the keyword and must be
adjacent to `as` (no whitespace). `as` and `as?` are always reserved
keywords — they cannot be used as variable names.

Examples:

<!-- agl-check: fragment -->
```agl
let n: int = raw-value as int          # raises CastError if not an int
let parsed: Option[int] = raw-value as? int

case parsed of
  | Some(value) => print value
  | None => print "not an int"

let s: text = some-int as text         # total — always succeeds
let j: json = my-record.count as json  # total — int is JSON-shaped

# left-associativity chains
let t: text = some-int as json as text   # (some-int as json) as text

# the optional cast carries its value, so no second cast is needed
case count-json as? int of
  | Some(value) => print value
  | None => print "not an int"
```

A `text` cast from a fallible source reads the value and formats it as text;
it is always total regardless of the source type (any data value has a text
rendering). For structured source types (`array`, records, enums, exceptions …)
`as text` produces the same AgL-form text that `print` would render.
For scalar types (`bool`, `int`, `decimal`) it produces the plain scalar text.

**Recursive cast targets.** A [recursive record or enum](types.md#recursive-types)
works as a `text`/`json` cast target exactly like any other: the value is
validated and decoded through the same JSON Schema (`$defs`/`$ref` for the
hoisted parts) used at the agent-call boundary, and `as`/`as?` behave
normally, including inside a container target such as `array[Tree]`. The same
finite-schema restriction applies as for an agent output type: a
[polymorphically recursive](generics.md#recursive-generic-types) generic type
whose reachable instantiations never close cannot be used as a cast target
either — a static error at the `as`/`as?` expression, not a runtime failure.

### Enum-member and exception tests: `is`, `is not`

<!-- agl-check: fragment -->
```agl
review is Pass
status is Status::Blocked     # qualified; aliases resolve transparently
error is HttpError            # exception: HttpError or a descendant of it
```

The left operand must have enum or exception type.

For an **enum** left operand, the right-hand name must be one of that enum's
members. A member may be written by its bare injected name, its record
declaration name, or a qualified enum-member spelling. The test compares
nominal member identity. When one bare spelling exposes members from several
enums, the left operand's enum type selects the member; several distinct
matching members remain ambiguous.

For an **exception** left operand, the right-hand name must name the left
operand's static type, an ancestor of it in its `extends` chain, or a
descendant; an unrelated type is a static error. `x is T` holds exactly when
`x as? T` is `Some` — see [Casts and convertibility](types.md#casts-and-convertibility).

Either right-hand name may be qualified, including through a named scope, the
same way a [constructor pattern](pattern-matching.md#module-qualified-constructor-patterns)
is.

`is`/`is not` never narrows the static type of the left operand; cast to the
target type first to access its fields or methods.

### Operators as function values

A built-in symbolic binary operator written alone in parentheses is a function
value: `(+)`, `(-)`, `(*)`, `(/)`, `(==)`, `(!=)`, `(<)`, `(<=)`, `(>)`, and
`(>=)`. It behaves like the lambda `fn(a: L, b: R) => a op b`, whose operand
types `L` and `R` come from context: an expected function type, the parameter
of a higher-order function, or the arguments of a direct or partial call.

```agl
program def main() -> unit =
  print([1, 2, 3].fold(0, (+)))         # 6
  print(["a", "b"].map((==)(?, "a")))   # [true, false]
  print([3, 1, 2].sort((-)))            # [1, 2, 3]
  print([1, 5, 9].select((<)(?, 4)))    # [1]
  let ratio: (int, int) -> decimal = (/)
  print(ratio(1, 4))                    # 0.25
```

The operator's typing rules then apply to those operand types, including
`int → decimal` widening. As for a lambda, a concrete expected result type is
adopted when the operator's result is assignable to it, so
`let f: (int, int) -> decimal = (*)` is valid. Operand types that the context
does not determine are a static error, as in `let f = (+)`; annotate the
binding instead. Like any function value, an operator value receives both
operands already evaluated.

`(-)` is subtraction, not negation. The word operators have no operator-value
form (`(and)` is a parenthesized name); use a lambda instead. A user-defined
operator is an ordinary function, so `(|>)` names its declaration directly.

## `case` expressions

A `case` expression selects among pattern branches whose bodies use the
canonical `branch_body` form:

```ebnf
case_expr       ::= "case" or_expr "of" case_body
case_body       ::= case_branch_seq
                  | NEWLINE INDENT case_branch_seq NEWLINE? DEDENT
case_branch_seq ::= "|"? case_branch ("|" case_branch)*
case_branch     ::= pattern "=>" branch_body
```

<!-- agl-check: fragment -->
```agl
let next: text = case action of
  Stop => "Stop."
  | Continue(prompt) => prompt
  | Escalate(reason) => "Investigate blocker:\n%{reason}"
```

The branch list either follows `of` on the same line or forms an indented
block on the lines below it; the first branch's `|` is optional in both forms.
`branch_body` is the canonical branch-body production: a suite or one
`closed_item` (`or_expr`, inline assignment, `raise`, or `return`). See
[Grammar](grammar.md#if).

All branch result types must agree after `int → decimal` widening. An outer
expected type propagates into every branch. The patterns must be exhaustive
and non-redundant, as described in [Pattern matching](pattern-matching.md).

A `case` expression is one of the loosest expression forms: in positions where
a following `|` could be ambiguous (branch bodies, `if`/`until` conditions) it
must be parenthesized. The same expression form is valid at block level; AgL
has no separate `case` statement.

## `if` expressions

An `if` **expression** selects among branches by boolean condition:

```ebnf
if_expr        ::= "if" "|"? if_cond_branch ("|" if_cond_branch)* if_else_branch?
if_cond_branch ::= or_expr "=>" branch_body
if_else_branch ::= "|"? "else" "=>" branch_body
```

<!-- agl-check: fragment -->
```agl
let label: text = if | score > 90 => "A" | score > 75 => "B" | else => "C"
```

The `else` branch is optional. It uses the canonical `branch_body` production
([Grammar](grammar.md#if)). With `else`, all branch result types must agree after
`int → decimal` widening. Without `else`, the `if` remains an expression
but has type `unit`, as described below.

## Expressions with type `unit`

The following expressions have type `unit` — they exist for their side
effect, not their value:

| Form | Type | Notes |
|------|------|-------|
| `print(e)` | `unit` | writes to stdout and returns `()` |
| `x := e` | `unit` | mutates `x` and returns `()` |
| `if c => body` (no `else`) | `unit` | branch body must be `unit`; returns `()` |
| loop expressions | `unit` | loops run for effect and return `()` |
| `return` | bottom expression returning `()` | valid only in a `unit` function |
| `()` | `unit` | the unit literal |

An `if` without `else` always has type `unit`, and each branch body must also
have type `unit`. A loop likewise has type `unit` and returns `()`; a `case`
has the common type of its branch bodies.

`unit` values may appear anywhere in a block, including as the final
expression. A function declared `-> unit` has its body checked against
`unit`.

## `let` and `var` as expressions

`let` and `var` are **binders**: `let` binds one immutable name, while `var`
binds one mutable name. Both scope their binding over the
*continuation* — the remaining items in the block. The type and value of a
`let`/`var` binding is the type and value of the continuation, not of the bound
expression. Use `case` when a value must be matched or destructured:

```agl
program def main() -> unit =
  let x = 3
  let y = x + 1
  let _ = y
```

A block may end in a `let` or `var`. With no following item, the binder has no
in-block continuation and the block evaluates to `unit` (or bottom when its
initializer exits); the binding still scopes over an enclosing continuation,
such as a loop's `until` condition.

```agl
def prepare() -> unit =
  let status = "ready"
```

## Divergent expressions

`raise expr` and `return expr` diverge — they never yield a value at their
expression site. Their type is the **bottom type** (assignable to any expected
type), so they may appear in expression positions such as a branch suite or the
initializer of a `let`/`var`:

<!-- agl-check: fragment -->
```agl
let x: int = raise Abort(message = "Cannot continue.")
let y: int = if finished => (return 0) else => 1

case status of
  | Ok => ()
  | Error(reason) =>
      raise Abort(message = reason)
```

`raise` is described in [Exceptions](exceptions.md); `return` is described in
[Functions](functions.md).

## `break` and `continue` as expressions

`break` and `continue` are loop-control expressions. Both have the **bottom
type** (assignable to any expected type), so they may appear in any expression
position inside a loop:

<!-- agl-check: fragment -->
```agl
let x: int = if finished => break else => count
```

`break` exits the innermost enclosing loop immediately; the loop produces
`unit`. `continue` starts the next iteration. From a body or `until` condition,
it skips the remaining work in the current iteration (including the `until`
condition when it occurs in the body); from a `while` guard, it starts over
without entering the body. The next iteration performs its normal `for` advance
and loop checks.

Both are valid anywhere in a loop's **interior** — its `while` guard, body, or
`until` condition — provided the target loop is in the same function or lambda.
Using either outside a loop, including in a `fn` or `lambda` that would cross
into an enclosing loop, is a static error.

## `try` as an expression

`try … catch …` yields a value: the body and every handler must agree on a
common type (with `int → decimal` widening). This type becomes the `try`
expression's type:

<!-- agl-check: fragment -->
```agl
let result: decimal =
  try
    10 / 0
  catch ArithmeticError =>
    -1
```

## Expected-type propagation

An expected type propagates top-down where it helps:

| Context | Propagated expectation |
| ------- | ---------------------- |
| `let x: T = e` / `var x: T = e` | `T` into `e` |
| `x := e` | declared type of `x` into `e` |
| Constructor argument | declared field type |
| `array[T]` / `dict[text, V]` expectation | element/value type into each element |
| `case` / `if` expression with outer expectation | into every branch |
| `ask` / typed `exec` | becomes the call's target type |
| Function call | each parameter type into the corresponding argument |
| Lambda with omitted parameter annotations | matching function parameter types into the omissions |
| Function body | `-> RetType` propagated in |

After scope has resolved an unqualified enum-member constructor (`let r: Review = Pass`),
propagation checks it against the expected enum type, types empty containers,
and gives agent calls their output contracts. It does not select among
same-named constructor candidates. A target that depends on sibling constraints
is resolved with the enclosing expression before its codec and schema are
chosen. Where no expectation exists, inference is bottom-up, and an untyped `ask`
defaults to `text`; the same default applies to a generic type parameter that
nothing else in the enclosing expression pins (as in `print <| exec "…"`),
once sibling constraints have settled.
