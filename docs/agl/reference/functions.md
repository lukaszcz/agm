# Functions

[← Index](index.md)

AgL supports **user-defined functions**: named `def` declarations at the
module root or in named scope regions, and anonymous `fn` expressions. Functions are first-class
values; they may be stored in bindings, passed as arguments, and returned
from other functions. The type of a function value is written
`(A, B) -> C`.

## `def` — named function declarations

```ebnf
func_def         ::= attributes? "def" func_decl_head type_params? "(" param_list? ")" ("->" type_expr)? ("=" func_body | suite)
builtin_func_def ::= attributes? "builtin" NEWLINE? "def" func_decl_head type_params? "(" param_list? ")" "->" type_expr
extern_func_def  ::= attributes? "extern" NEWLINE? "def" func_decl_head type_params? "(" param_list? ")" "->" type_expr
func_decl_head   ::= decl_head | builtin_receiver "::" name
decl_head     ::= [scope_path "::"] name
builtin_receiver ::= "array" "[" name "]" | "dict" "[" "text" "," name "]"
                   | "text" | "json" | "int" | "decimal" | "bool"
func_body     ::= expr | suite
type_params   ::= "[" name ("," name)* "]"
param_list    ::= param ("," param)* ","?
param         ::= attributes? field_name ":" type_expr ("=" or_expr)?
                | "self" [":" type_expr]       (* first parameter of a method *)
```

A `def` is a static declaration at the module root or in a
[named scope](scopes.md). A declaration head may include its scope path, as in
`def Text::render(value: text) -> text = value`. An inline body requires `=`.
For an indented suite body, the `=` before the newline is optional. The body is
a single expression — which may be a block (a sequence of items ending in an
expression):

```agl
def classify(n: int) -> text =
  if
    | n > 0 => "pos"
    | n < 0  => "neg"
    | else   => "zero"

def summarize(doc: text, sentences: int = 3) -> text =
  let prompt = "Summarize in at most %{sentences} sentences:\n%{doc}"
  ask(prompt)

def double(n: int) = n * 2

# Compact one-line block body.
def incremented() -> int = (let x = 0; x + 1)
```

An inline `def` body is a single item. It ends at the next block separator —
a newline or `;` — and a `;` after the body starts the next block item rather
than extending the body. Write a multi-item body as a parenthesized block or
as a suite — see [Inline bodies](grammar.md#inline-bodies).

### Return type

The `-> RetType` annotation is optional on ordinary `def` declarations. When
present, the body is checked against it; a mismatch is a static error. When
omitted, AgL makes a best-effort inference of the return type from the body,
as it does for lambdas. A monomorphic function can infer through direct
recursion, same-module forward references, and mutual recursion when the group
provides concrete evidence:

```agl
def is-even(n: int) = if n == 0 => true else => is-odd(n - 1)
def is-odd(n: int) = if n == 0 => false else => is-even(n - 1)
```

This also applies when a later function is used as a function value or through
partial application. Inference uses only the definitions in that dependency
group, never callers or their expected types. A recursive generic function is
inferred when every recursive call preserves its type parameters in the same
order; a call at changed, permuted, fixed, or nested type arguments needs an
explicit result annotation. If a group has no concrete result evidence, for
example because it always raises or only calls within the group, AgL asks for
return type annotations.

A function may also exit early with `return`; see [Early return](#early-return).

### Early return

```ebnf
return_expr ::= "return" or_expr?
```

`return expr` exits the nearest enclosing `def` or `fn` body immediately and
makes `expr` the function call's result. A bare `return` is equivalent to
`return ()` and is valid only when the function result type is `unit`.

```agl
def first-positive(xs: array[int]) -> int =
  for x in xs do
    if x > 0 =>
      return x
  done
  -1

def log-and-stop() -> unit =
  print "stopping"
  return
```

The usual tail-value rule still applies: a body that does not execute a
`return` yields its last expression. With an explicit return type annotation,
each `return` operand is checked against that result type. Without an
annotation, the inferred result type is the common type of all `return`
operands and the body's tail value, using the same branch-unification rules as
`if` and `case` (`int` may widen to `decimal`, and divergent branches are
ignored). If these values have no common type, add a return type annotation.

A `return` inside a lambda returns from that lambda, not from an enclosing
`def`. A `return` is valid only inside a function body; it is a static error at
the program top level or in parameter defaults.

Like `raise`, `return` is admitted directly in an inline branch or `catch`
body (`if ready => return x else => 0`) and in a loop body. It remains an
`expr`-level form, so in an `or_expr` position — an `until` condition, a
binder's right-hand side, an operand — it must be parenthesized. A `return`
followed by a newline is a bare `return`; the operand never continues onto the
next line.

### Built-in functions

`builtin def` declares a function implemented by the host, so it has no body.
Its return type annotation is required. The declared name and signature must
match a recognized built-in exactly. This form is used by the standard library; ordinary
programs normally call those declarations through the default standard-library
import instead of redeclaring them.

A call to a built-in name resolves exactly like a call to any other name: the
callee must reach a `builtin def` declaration — bare through the standard
library's default import or a program's own declaration, or qualified through
its declaring path — and a name with no such declaration in scope is an
ordinary undefined-name error, not a host dispatch.

A built-in's complete scoped name — whether it names a `builtin def` or a
`builtin record`/`enum`/`exception` — is its host identity. It may be declared
only once at that path across the program. This lets a root `ask` coexist with
`Agent::ask`, while still preventing a second declaration at either exact path.
A program loading the default standard library, as it does unless started with
`--no-stdlib` (see [Modules](modules.md#prelude)), therefore cannot redeclare
a standard-library builtin at the same scoped name.

`builtin` is a declaration modifier: it may precede `def` on the same line or
on the line directly above it (the newline after the modifier is
insignificant).

Runtime built-ins are first-class values. A value occurrence is contextually
specialized and behaves like an eta-expanded lambda over the declaration's
required parameters; omitted optional parameters retain their host defaults:

```agl
program def main() -> unit =
  let emit: text -> unit = print
  let query: text -> int = ask
  let run: text -> ExecResult = exec
  let open-session: Agent -> Session = Session::open
  let current-session: () -> Session = Session::default
  ()
```

Generic occurrences need an expected function type or explicit type arguments,
except that an unconstrained `ask` value defaults its result to `text`, like a
direct `ask` call. `ask-request` similarly defaults its embedded output
contract to `text`. Configured variants can be expressed with a lambda around a
direct call. Built-in methods produce receiver-capturing values, and qualified
method or `Session` static references produce ordinary positional function
values. `resource` is call-only because its path must be a source literal resolved
at link time; the nullary `resource-dir` is a function value. `parse` and
`try-parse` are call-only because their target type comes from the call's
explicit type argument or expected type ([Parsing values](types.md#parsing-values)).

### Externally implemented functions

`extern def` declares a function implemented by a companion Python file
instead of an AgL body or the host — like `builtin def`, it has no body and a
mandatory return type, but its implementation lives in ordinary program
source (a co-located `.py` file) rather than the host. See
[Python FFI](ffi.md) for the declaration syntax, the type mapping across the
boundary, and the error model. A type parameter that no parameter mentions
makes it **type-directed**: each call site's resolved type reaches the
companion ([Target type parameters](ffi.md#target-type-parameters)).

### Parameters

Parameters are listed with explicit types. Each parameter belongs to one of
three **zones** that determine how arguments at the call site are matched:

| Zone | Binding | Written as |
|------|---------|------------|
| **Positional-only** | Positional argument only; cannot be passed by name | `@arg-pos`; a method receiver `self` |
| **Standard** | Positional or named | `@arg-std` |
| **Named-only** | Named argument only (or bare-name shorthand) | `@arg-named` |

An [attribute](attributes.md#zone-attributes) in front of a parameter puts that
parameter in the zone it names. The same attribute in front of the declaration
sets the zone of every parameter that does not name one itself.

For `def`/`extern def`/`builtin def`/lambda, the **default zone is standard**: a
parameter list with no zone attribute has all parameters in the standard zone
(positional or named). A method receiver `self` is the exception: it is always
positional-only. A `program def`'s parameter list defaults to the **named-only**
zone instead: a plain `name: text` parameter is addressed only by `--name`
([Host environment](host-environment.md#program-arguments)); an `@arg-pos`
parameter opens a positional slot.

<!-- agl-check: fragment -->
```agl
def f(@arg-pos x: int, y: int) -> int = x + y     # x pos-only, y standard
def g(@arg-pos x: int, y: int, @arg-named z: int) -> int = ...  # all three zones
def simple(x: int, y: int) -> int = x + y         # both standard (default)

@arg-pos
def h(x: int, y: int) -> int = x + y              # both positional-only
```

Parameters are listed in zone order: positional-only, then standard, then
named-only. A parameter that follows one from a later zone is a static error.

**Defaults.** A parameter default is an `or_expr` (`param: type = or_expr`).
Open forms such as `if`, `case`, `try`, loops, `raise`, and `fn` must therefore
be parenthesized in a default. Only positional-fillable (pos-only or standard)
parameters are subject to the ordering constraint: no *required*
pos-only/standard parameter may follow a *defaulted* pos-only/standard one.
Named-only defaults may appear in any order:

```agl
def greet(name: text, greeting: text = "Hello") -> text =
  "%{greeting}, %{name}!"

def with-named-default(x: int, @arg-named tag: text = "ok") -> text =
  "%{tag}: %{x}"   # tag is named-only; its default is unconstrained
```

A [record field default](types.md#record-types) follows the same `= <constant
expr>` shape and the same zone-ordering constraint, and is likewise omittable
at a constructor call; see [Record types](types.md#record-types) for its
constant-expression rule and [Value syntax](host-environment.md#value-syntax)
for how an omitted defaulted field is filled outside AgL source (JSON, value
syntax, agent output).

## Methods

A `def` whose first parameter is `self` and whose enclosing scope resolves to
a receiver type — a record, enum, enum member, exception, or builtin receiver
— is a **method**: a scoped function selected by its receiver's **static type**.
`p.f(x)` supplies `p` as the first argument of the selected function, while
`Type::f(p, x)` calls one function by its ordinary scope path. A method is not
dynamically dispatched.

The receiver part of `def Type::f(self)` (or `scope Type` containing that
`def`) resolves as a type name in the module containing the declaration. A
locally declared record, enum, enum member, or exception wins. Otherwise, exactly one
type made bare-visible in that region by an import tail or `use` must provide
the name. Renamed contributions may provide that spelling. A qualified import
alone provides no bare receiver name, and a type alias cannot name a method
receiver. The declaration extends the resolved type's plain scope in the
module that contains the `def`; it does not add a declaration to the type's
home module.

<!-- agl-check: fragment -->
```agl
# geometry.agl
record Point
  x: int
  y: int

def Point::shift(self, amount: int) -> Point = Point(x = self.x + amount, y = self.y)

# metrics.agl
import geometry::*

def Point::norm(self) -> int = self.x * self.x + self.y * self.y

# main.agl
import geometry::*
import metrics

program def main() -> unit =
  let p = Point(x = 2, y = 3)
  print(p.shift(1).norm())
  print(metrics::Point::norm(p))
```

`metrics::Point` in this example is a plain scope of `metrics`. A bare
`Point::norm(p)` follows ordinary scope-path resolution; dot selection is the
operation that gathers methods for a receiver type across modules.

A method is selectable in a module when that module declares it or can reach
its declaration by a qualified import route. Any import form — a plain import,
alias, bare tail, wildcard, or facade re-export — can provide that route;
`hiding` removes it. `use` adds bare names only and does not affect method
visibility. The implicit
`import std/prelude::*` is an ordinary route, so it also makes the methods its
receiver-scope re-exports expose available.

Method candidates for a name form **one level**, fixed by the receiver's
static type `T`:

- record `T` — `T`'s own methods plus those of every current enum that
  declares or references `T` as a member;
- exception `T` — `T`'s own methods plus every ancestor's in its `extends`
  chain;
- enum `T` — only `T`'s own methods; a method declared on one of `T`'s own
  members is not part of this level;
- builtin receiver — only that builtin type's methods.

Route visibility then decides within the level: exactly one visible method
wins; two or more are a static ambiguity, listing every declaration; none
visible but candidates exist reports that the method exists but is not
visible here; no candidates is no member. A home-module method and an orphan
carry equal weight — neither position in the level nor declaration order
breaks a tie.

Fields and methods are distinct member kinds. A field of the receiver type
(its own or an inherited one) and any visible method in its level are
ambiguous, for a read and for `receiver.name := value` alike; hide the
method's route to reach the field, or call the method by a qualified path. An
enum method named like a member field stays legal to declare: it clashes with
the field only on a member-typed receiver, since an enum-typed receiver has
no fields and simply selects the method. A method declaration itself is
rejected when its resolved owner or an ancestor already has a field of that
name (see below for the other rejection: a same-module, same-level method
pair).

When the selected method belongs to an owning enum `E` rather than to the
member itself, the receiver **widens** to `E` — an identity upcast, so
mutation through a `var` field stays visible through the member binding.
`E`'s type parameters that the member's fields capture come from the
member's own type arguments; any parameter the member doesn't capture must be
solved by the call's arguments, its expected type, or its result, or the call
is rejected with a suggestion to annotate or cast the receiver.

Along the member→owning-enum axis and the exception descendant→ancestor axis,
changing the receiver's static type by an annotation or an identity upcast
may turn a call into an error or back, but never changes *which* function the
call selects. Widening one enum to an overlapping enum is outside this: they
remain distinct nominal types, each with its own methods, even when one's
members are a subset of the other's.

An enum in another module that references a record adds that enum's methods
to the record's level wherever they are reachable by route in the using
module — the same exposure orphan methods already accept. Repair a resulting
ambiguity or field clash by hiding the contributing route, calling the
intended function by a qualified path, or widening the receiver explicitly
with `as`.

A module may not declare two same-named methods whose owners share a level —
a record and an enum that declares or references it, or an exception and one
of its ancestors; the later declaration is rejected. The same pair is legal
across modules and becomes an ambiguity only where both routes are visible at
a call site; repair it with a qualified call, a rename, `hiding` one route, or
an annotation or `as` that narrows or widens the receiver's static type.

A builtin receiver may be named in a method declaration in any module. This
syntax is available to ordinary, `builtin`, and `extern` definitions:
`array[E]::name` and `dict[text, V]::name` bind their receiver element or value
parameter; `text`, `json`, `int`, `decimal`, and `bool` are bare receivers. A
builtin receiver must use its bare generic form, so `array[int]::name` and
`dict[text, array[int]]::name` are invalid. As with a nominal generic receiver,
`_` may occupy an unused receiver slot; it binds a private rigid parameter and
cannot be named by the method body.

The prelude re-exports builtin receiver scopes, making standard-library methods
reachable wherever the prelude is enabled. With `--no-stdlib`, import a route
to the module or facade that exports the method. Free functions remain subject
to ordinary imports. Ordinary and `extern` builtin-receiver methods use the
same direct-call, bound-method, and generic-specialization rules as nominal
methods.

A `builtin def` receiver method is a host route. Its name and signature must be
one of `print`, `render`, `copy`, or `shallow-copy`; each takes only `self`.
`copy` and `shallow-copy` return the receiver's exact type, `print` returns
`unit`, and `render` returns `text`. Calling it reuses the corresponding bare
builtin operation with `self` as its value. Projecting it produces a bound
nullary function value, while a qualified reference includes `self` as its
first positional parameter.

```agl
record Person
  name: text
  address: text

def Person::with-address(self, address: text) -> Person =
  Person(name = self.name, address = address)

program def main() -> unit =
  let person = Person(name = "Ada", address = "Main Street")
  let by-member = person.with-address("East Road")
  let by-path = Person::with-address(person, "East Road")
  print(by-member.address)
  print(by-path.address)
```

A method may update a `var` field through `self`; `self` need not be a mutable
binding because the assignment updates the record, not the receiver name:

```agl
record Counter
  var value: int

def Counter::add(self, amount: int) -> unit =
  self.value := self.value + amount

program def main() -> unit =
  let counter = Counter(value = 1)
  let alias = counter
  counter.add(2)
  print(alias.value)
```

`self` must be the first parameter. It has no default and cannot be
supplied by name. Its annotation is optional; when written, it must be exactly
the resolved receiver type with its receiver type parameters. Thus an orphan
on `geometry::Point` may write either `self: Point` or `self: geometry::Point`;
for a generic `Box` receiver, `self: Box[E]` is valid while `self: Box[int]`
is not.

On a generic receiver type, the first type parameters of a method bind the
receiver type parameters in positional order. Their names are the method's
choice. `_` may fill any unused receiver slot and may repeat; see
[Generics](generics.md#generic-methods) for the complete rule.

```agl
record Box[T]
  value: T

def Box::get[E](self: Box[E]) -> E = self.value
def Box::size[_](self) -> int = 1

program def main() -> unit =
  let box = Box(value = 7)
  print(box.get())
  print(box.size())
```

An inline enum member likewise establishes a record type scope, including the
captured generic parameters of its fields:

```agl
enum Tree[T]
  | Leaf
  | Node(value: T)

def Tree::Node::extract[E](self) -> E = self.value

def Tree::or-default[T](self, fallback: T) -> T =
  case self of
    | Leaf => fallback
    | Node(value) => value

program def main() -> unit =
  print(Node(value = 1).extract())    # 1: Tree::Node's own method
  print(Leaf.or-default(4))           # 4: Tree's method, receiver widened to Tree[int]
```

`Tree::Node::extract` is a method of `Tree::Node[E]` alone. `Tree::or-default`
belongs to `Tree`, `Tree::Node`'s owning enum, so it is also in
`Tree::Node[E]`'s level: selecting it widens the receiver to `Tree[E]`. `Leaf`
captures no type parameter, so calling `or-default` on it leaves `T`
phantom; here the argument's type solves it.

A method member used without a call is a **bound method**: a function value
that has captured its receiver and has parameters only for the remaining
method parameters. It can be stored, passed to another function, or partially
applied like any other function value.

A **leading-dot method invocation** omits the receiver and produces a unary
function that accepts it:

```ebnf
leading_dot_expr ::= "." field_name value_type_args? "(" arg_list? ")"
```

`.map(f)` is equivalent to `fn(self: ContextType) => self.map(f)`, except that
`ContextType` is inferred rather than written. The surrounding expression must
provide a concrete unary function type; otherwise the receiver type cannot be
inferred and the expression is rejected. Method selection still uses that
receiver's static type, with the same visibility and ambiguity rules as an
ordinary member call. Explicit method type arguments and named arguments are
supported.

```agl
program def main() -> unit =
  let numbers = [1, 2, 3]
  let incremented = numbers |> .map(fn(value: int) => value + 1)
  let stringify: (array[int]) -> array[text] = .map::[text](fn(value: int) => "%{value}")
  print(incremented)
  print(stringify(numbers))
```

```agl
record Meter
  value: int

def Meter::add(self, amount: int) -> int = self.value + amount
def apply(value: int, f: int -> int) -> int = f(value)

program def main() -> unit =
  let meter = Meter(value = 4)
  let add = meter.add
  let plus = meter.add(?)
  print(apply(3, add))
  print(plus(5))
```

`self` is special only as a method receiver. Elsewhere it is an ordinary
identifier and, as an ordinary parameter, requires an annotation. `def` and
`extern def` may declare methods. A `builtin def` may also declare a host method
when its signature is a recognized host contract: the standard library declares
`Agent::ask` and `Agent::ask-request`. These methods use the same selection,
bound-function, and specialization rules; see [Agent calls](agent-calls.md).

### Scope and forward references

A root `def` may call itself or any other root `def`, including one declared
later. `def`s in the same named scope have the same visibility, so root and
same-scope `def`s may use mutual recursion:

```agl
def is-even(n: int) -> bool =
  if n == 0 => true else => is-odd(n - 1)

def is-odd(n: int) -> bool =
  if n == 0 => false else => is-even(n - 1)
```

`def` is valid at the program root and in named scope regions, but not inside
a block (`do` body, `if` branch, etc.). A static error is raised if a `def` is
nested in a block.

A root or same-scope `def` may likewise read or write a `let`/`var` binding
declared later, in a module with a static root ([Names and
visibility](scopes.md#names-and-visibility)).

## `fn` — anonymous functions (lambdas)

```ebnf
lambda_expr  ::= "fn" lambda_params ("->" type_expr)? "=>" expr
lambda_params ::= field_name | "(" lambda_param_list? ")"
lambda_param_list ::= lambda_param ("," lambda_param)* ","?
lambda_param ::= attributes? field_name [":" type_expr] ("=" or_expr)?
```

`fn` produces a function value. A unary lambda may omit its parameter
parentheses. The return type annotation is **optional**: when omitted it is
inferred from the body, unless a concrete expected function type checks the
body against its result type.

A parameter annotation may also be omitted when the lambda has a matching
function context. Each omitted type comes from the corresponding parameter of
that function type; annotations may be omitted independently. A function-typed
binding and a higher-order function or method parameter provide such context:

```agl
program def main() -> unit =
  let double = fn(x: int) => x * 2
  let increment: int -> int = fn x => x + 1
  let add: (int, int) -> int = fn (x, y) => x + y
  let add-left: (int, int) -> int = fn (x: int, y) => x + y
  let add-right: (int, int) -> int = fn (x, y: int) => x + y
  let greet = fn(name: text) -> text => "Hello, %{name}!"
```

An omitted parameter type without a matching context, or with a context of a
different arity, is a static error. When generic argument inference supplies
the context, concrete sibling arguments are considered before the contextual
lambda.

A lambda is an ordinary expression and may appear anywhere an expression is
accepted — in a binding, as a call argument, or in an array:

```agl
program def main() -> unit =
  let ops: array[int -> int] = [fn(x: int) => x + 1, fn(x: int) => x * 2]
```

When used in juxtaposition position (as the right operand of an operator or
the lone argument to a single-arg call), a lambda must be parenthesized:

```agl
program def main() -> unit =
  let result = (fn(x: int) => x + 1)(5)
```

### Lambdas are not self-recursive

A lambda's name (the binding introduced by `let`) is not in scope inside
the lambda body. Local recursion is expressed via a top-level `def`. The
restriction is intentional: lambda return-type inference is local and safe
precisely because the body never depends on the lambda's own type.

## Generic functions

A `def` may declare **type parameters** in square brackets after its name,
making it polymorphic over those types. This is prenex (rank-1) parametric
polymorphism: the type parameters are universally quantified over the whole
declaration. See [Generics](generics.md) for the full treatment; this section
covers the function-specific surface.

```agl
def id[T](x: T) -> T = x

def fst[A, B](a: A, b: B) -> A = a
```

An ordinary function in a named scope or in a module with a path identity may
use the same name as a built-in function because it is reached through its own
qualified namespace. The bare built-in spelling remains reserved, even after a
`use` declaration makes that scope's or module's members bare:

<!-- agl-check: fragment -->
```agl
use Codec::*

scope Codec
  def render[T](value: T) -> array[T] = [value]
end Codec

let wrapped = Codec::render(1)  # the scoped generic function
let shown = render(1)           # the built-in render
```

An ordinary user `def` at the root of a module with no path identity of its own
(see [Modules](modules.md)) cannot use a built-in function name: the bare
namespace is the only one it declares into. A module that has a path identity
keeps its qualified namespace whether it is the entry program or one of its
imports.

A type parameter is an ordinary name; it may be used anywhere a type may
appear within the declaration — parameter types, the return type, and any
annotation **nested inside the body**:

```agl
def singleton[T](x: T) -> array[T] =
  let single: array[T] = [x]
  single

def via-lambda[A](x: A) -> A =
  let g: A -> A = fn(y: A) -> A => y
  g(x)
```

Here `array[T]` is an annotation on an inner `let`, and the lambda's parameter
and return types refer to the enclosing `A`. A type variable is in scope
throughout the body of the `def` that introduces it.

### Inference and explicit type arguments

At a call site the type arguments are normally **inferred** — from the
argument types and from the expected type of the call. Argument evidence fixes
an instantiation before an expected result type is considered, so ordinary
assignability (including `int` to `decimal`) applies only afterwards:

<!-- agl-check: fragment -->
```agl
print(id(5))          # T = int, inferred from the argument
print(id("hi"))       # T = text
print(fst("x", 9))    # A = text, B = int
```

When inference is insufficient or you want to pin the instantiation, supply
the type arguments explicitly with the typed-call form `::[…]`:

<!-- agl-check: fragment -->
```agl
print(id::[int](5))
```

The type-argument list must match the declared type parameters in number and
order.

### A generic `def` as a first-class value

A generic `def` can be used as a value wherever surrounding constraints fix
its instantiation. An expected function type does so for an annotated binding:

```agl
def id[T](x: T) -> T = x

program def main() -> unit =
  let f: text -> text = id
  print(f("via value"))
```

A higher-order declared call can supply those constraints through its other
arguments, so a generic function occurrence is fresh at each use:

```agl
def apply[T](f: T -> T, value: T) -> T = f(value)
def id[T](value: T) -> T = value

program def main() -> unit =
  let n = apply(id, 5)
```

You can also pin the instantiation explicitly without calling the function:

<!-- agl-check: fragment -->
```agl
let g = id::[int]
print(g(5))
```

A bare `let f = id` with no constraints is a **static error**: there is
nothing to infer the type arguments from. Bindings finalize their initializer,
so a later `f(5)` cannot retroactively specialize `f`. (Calling `id` directly,
where the arguments drive inference, needs no annotation.)

### Strict parametricity

Inside a generic `def`, a value whose static type is a bare type variable
`T` is **opaque**. The body knows nothing about `T` beyond the fact that
values of it exist, so such a value can only be passed to other functions,
returned, or stored. It may **not** be:

- compared with `=`, `!=`, or the ordering operators,
- used in arithmetic,
- printed or interpolated in a template,
- field- or index-accessed,
- tested with `is` / `is not`.

<!-- agl-check: error -->
```agl
def bad[T](x: T, y: T) -> bool = x == y   # static error: '==' on type variable T
```

Each of these is a static error. This *parametricity* guarantee means a
generic function treats its type-variable values uniformly regardless of the
concrete type they are instantiated at. (The restriction applies only to the
bare type variable itself — a value of a concrete or composite type such as
`array[T]` supports every operation that type normally allows.)

## Calling functions

All calls use the same uniform parenthesized syntax:

```ebnf
call_expr ::= postfix "(" arg_list? ")"
arg_list        ::= arg ("," arg)* ","?
arg             ::= element_expr                 (* positional *)
                  | placeholder_arg              (* positional hole *)
                  | field_name "=" element_expr  (* named *)
                  | field_name "=" placeholder_arg (* named hole *)
placeholder_arg ::= "?" | "?<digits>"
```

`element_expr` is `expr` without a bare record update, which must be
parenthesized in a comma-separated element position — see
[Record update](expressions.md#record-update).

The `postfix` callee may already carry explicit type arguments, so
`id::[int](5)` is a typed call; the `::[…]` application itself may also be used
without a call as described in [Generic functions](#generic-functions).

**Single-argument sugar.** When there is exactly one positional argument
and no named arguments, the parentheses may be dropped and the argument
written directly after the callee:

<!-- agl-check: fragment -->
```agl
print review          # equivalent to print(review)
ask "Hello?"          # equivalent to ask("Hello?")
print res.stdout      # field-access path is a valid sugar argument
print classify(x)     # equivalent to print(classify(x))
f Option::Some(value = 1)  # equivalent to f(Option::Some(value = 1))
```

Application binds **tighter than all operators**:

<!-- agl-check: fragment -->
```agl
print x + 1           # parsed as (print x) + 1
```

### Calling functions

Arguments are matched **positional-greedy**: positional arguments fill
positional-capable (pos-only and standard) slots left to right, in
declaration order. Named arguments use `name = value` and may follow the
positional arguments in any order. Positional arguments must precede named
arguments at the call site.

```agl
def add(x: int, y: int) -> int = x + y
def f(@arg-pos x: int, y: int) -> int = x + y
def g(x: int, @arg-named z: int) -> int = x + z

program def main() -> unit =
  let r = add(3, 4)
  let s = add(3, y = 4)
  let a = f(1, 2)
  let b = f(1, y = 2)
  let c = g(5, z = 3)
```

**Named-only shorthand.** When a bare variable name `x` appears in a
positional argument slot but all positional-capable parameters are already
filled, it is reinterpreted as the named argument `x = x` — but only if
`x` is a bare name (not an expression). A non-bare expression in that
position is an error:

```agl
def h(a: int, @arg-named key: text) -> text = "%{a}: %{key}"

program def main() -> unit =
  let key = "hello"
  print(h(1, key))
  print(h(1, key = key))
```

**Defaults.** Defaulted parameters may be omitted. Named-only defaults may be
supplied in any order:

```agl
def format-msg(message: text, prefix: text = "[INFO]") -> text =
  "%{prefix} %{message}"

program def main() -> unit =
  let _ = format-msg("Done.")
  let _ = format-msg("Done.", prefix = "!")
```

Unknown names, duplicates, and supplying a positional-only parameter by name
are static errors.

**Named arguments at declared-name sites only.** Named arguments are
available when calling a **declared name** (`def` or built-in). A function
*value* (bound in a `let` or passed as an argument) has a purely positional
type and is called with positional arguments only.

### Calling function values

A function value is called like any other call. The callee is an expression
of function type:

<!-- agl-check: fragment -->
```agl
let g: int -> text = classify
let label = g(7)               # positional call of a function value
```

A generic call that produces a function value may use the arguments of that
value call to determine its type arguments. The constraints of one enclosing
expression are considered together, so this needs no intermediate annotation:

```agl
def maker[T]() -> T -> T = fn(value: T) => value

program def main() -> unit =
  let number = maker()(7)
```

Each occurrence is inferred independently. As with any generic expression,
an unconstrained function-producing call still requires explicit type arguments
or an expected function type at its binding boundary.

## Partial application

A parenthesized call that contains one or more placeholder arguments evaluates
to a new function value instead of immediately calling the callee. A placeholder
must be the whole value of a positional argument (`?`, `?1`, `?2`, …) or the
whole value of a named argument (`name = ?`). Placeholders work with declared
functions, constructors, and function values.

```agl
def add(a: int, b: int) -> int = a + b
def digits(a: int, b: int, c: int) -> int = a * 100 + b * 10 + c

program def main() -> unit =
  let inc: (int) -> int = add(?, 1)
  print(inc(4))
  let plus: (int, int) -> int = add
  let plus-two: (int) -> int = plus(?, 2)
  print(plus-two(5))
  let fill-edges: (int, int) -> int = digits(?, 9, ?)
  print(fill-edges(1, 2))
```

The resulting function type has one parameter for each placeholder. Each
parameter has the type of the callee parameter or constructor field that the
placeholder binds to, and the result type is the result type of the underlying
call.

With bare `?` placeholders, the resulting function's parameter order is the
placeholders' order of appearance in the written argument list, including named
arguments. With numbered placeholders, the number gives the resulting
function's parameter position explicitly:

<!-- agl-check: fragment -->
```agl
let reordered: (int, int) -> int = digits(?2, 9, ?1)
print(reordered(1, 2))          # 291
```

Within one call, placeholders must be either all bare or all numbered. Numbered
placeholders must form exactly one use of each index from `?1` through `?n`:
there may be no `?0`, gaps, repeats, or mixing such as `f(?, ?1)`.

Named-argument holes bind to the named parameter, including named-only
parameters:

```agl
def shaped(x: int, @arg-named y: int, @arg-named z: int = 0) -> int = x * 100 + y * 10 + z

program def main() -> unit =
  let fill-y: (int) -> int = shaped(3, y = ?, z = 9)
  let fill-x: (int) -> int = shaped(x = ?, y = 4)
  print(fill-y(5))
  print(fill-x(2))
```

For constructors, the same argument binding rules apply:

```agl
record Box[T]
  value: T

program def main() -> unit =
  let make-box: (int) -> Box[int] = Box(value = ?)
  print(make-box(8).value)
```

Non-placeholder arguments, and the callee expression for a function-value call,
are evaluated once from left to right when the partial-application expression
itself is evaluated. The created closure captures those values. Defaults for
omitted parameters are not captured; they are evaluated each time the closure
is invoked.

```agl
var ticks = 0
var saved = 4

def next-tick() -> int =
  ticks := ticks + 1
  ticks

def add-saved(a: int, b: int) -> int = a + b

def stamped(x: int, suffix: int = next-tick()) -> int = x * 10 + suffix

program def main() -> unit =
  let use-saved = add-saved(?, saved)
  saved := 100
  print(use-saved(6))             # 10; captured saved = 4

  let stamp: (int) -> int = stamped(?)
  print(stamp(2))                 # 21
  print(stamp(2))                 # 22; default ran again
```

An exception raised while evaluating a captured callee or non-placeholder
argument is raised when the closure is created. An exception from the
underlying call is raised when the closure is invoked:

```agl
def add(a: int, b: int) -> int = a + b
def fail-created() -> int = raise Abort(message = "created")
def fail-called(x: int) -> int = raise Abort(message = "called %{x}")

program def main() -> unit =
  try
    let f = add(?, fail-created())
    print(f(1))
  catch Abort as e =>
    print(e.message)              # created

  try
    let g = fail-called(?)
    print(g(9))
  catch Abort as e =>
    print(e.message)
```

Generic callees infer type arguments jointly from non-placeholder arguments,
placeholder parameter types, and an expected function type when one is
available. This applies equally when the callee is a function-valued expression;
fixed arguments are considered before an expected shape completes any remaining
type arguments. If a type argument is still not known, give it explicitly with
the `::[…]` form.

```agl
def id[T](x: T) -> T = x
def singleton[T](x: T) -> array[T] = [x]
def map-one[A, B](f: (A) -> B, xs: array[A]) -> array[B] = [f(xs[0])]

program def main() -> unit =
  let keep-ints: (array[int]) -> array[int] = map-one(id, ?)
  print(keep-ints([5])[0])
  let make-single: (int) -> array[int] = singleton(?)
  print(make-single(7)[0])
  let make-text = singleton::[text](?)
  print(make-text("hi")[0])
```

Error conditions are reported statically:

- A placeholder is a partial-application marker only in a parenthesized call;
  forms such as a standalone `?`, `f(? + 1)`, and the single-argument sugar
  `f ?` do not parse.
- Placeholder partial application is not supported by free special built-in
  calls such as `print(?)` or `ask(?)`; referencing the built-in directly
  produces its defaulted function value instead. Receiver methods first
  produce a bound value, so forms such as `reviewer.ask::[text](?)` work.
- Numbered placeholders must be a permutation from `?1` through `?n`; examples
  such as `f(?0)`, `f(?2)`, `f(?1, ?1)`, and `f(?, ?1)` are rejected.
- Existing argument-binding errors still apply: arity mismatches such as too
  many holes, unknown or duplicate named arguments such as `f(missing = ?)`,
  missing required arguments, positional arguments for named-only parameters,
  and named arguments when calling a function value such as `g(x = ?)`.
- A generic partial application whose type arguments remain unknown, such as
  `let make = singleton(?)`, needs explicit type arguments or an expected
  function type.

## Function types

The type of a function value is `A -> B` for one parameter,
`(A, B, …) -> C` for multiple parameters, and `() -> C` for no parameters.
The arrow is right-associative, so `A -> B -> C` means `A -> (B -> C)`:

```ebnf
func_type ::= type_atom "->" type_expr
            | "(" type_list? ")" "->" type_expr
type_list ::= type_expr ("," type_expr)* ","?
```

<!-- agl-check: fragment -->
```agl
let f: int -> text = classify
let g: (int, int) -> int = add
let h: () -> bool = fn() => true
```

Function types are assignable by **exact structural match**: the number of
parameters, their types (in order), and the result type must all agree. No
variance or subtyping applies.

Named and defaulted arguments are erased from the function *value* type.
A `def` with optional parameters still has a fully positional function type;
only the declared name retains the named/default information at call sites.

## Opacity

Function values have **opaque rendering, no JSON encoding, and no equality**.
Rendering a function value, interpolating it in a template, or printing it
produces a diagnostic surface form such as `<function: (int, int) -> int>`.
Storing a function value in a `json` slot, passing it to `ask`, or using it
where a JSON-shaped type is expected are static errors. These restrictions
exist because function values are capability handles, not data.

The REPL echoes bare function values with the same opaque rendering. That
display is available through AgL rendering, but it is not JSON data.

## Recursion and the call-depth limit

Top-level `def`s may call themselves and each other without restriction
at the language level. The host enforces a **call-depth limit**, with a portable
default of 256, and may select a different limit before execution. The limit is
not an AgL engine-setting binding and cannot be changed by the program. Exceeding
it raises `RecursionError` ([Exceptions](exceptions.md)):

```agl
def fact(n: int) -> int =
  if n <= 1 => 1 else => n * fact(n - 1)

program def main() -> unit =
  let r = fact(10)
  let s = fact(10000)
```

`RecursionError` is catchable with `try`/`catch`. The limit counts
activation frames across all `def` calls including mutual recursion.

## Syntactic arguments

The types the language's own constructs name are ordinary values: a
`ParsePolicy` or an `ExecResult` can be bound, passed to a function, and
returned from one.

```agl
def make-policy(retries: int) -> ParsePolicy =
  if retries == 0 => ParsePolicy::Abort else => Retry(n = retries)
```

The `on-parse-error` argument of `ask`/`exec` is the one exception: it requires
a **syntactic** static constructor written at the call site (`Abort`, or
`Retry(n = <int literal>)`), so a `ParsePolicy` held in a binding or returned
from a function like `make-policy` cannot be passed to it.

## Complete example

```agl
enum Review
  | Pass
  | Fail(issues: array[text])

let reviewer = AgentCommand("reviewer")

def summarize-issues(issues: array[text]) -> text =
  "Issues found:\n%{issues}"

def review-artifact(artifact: text) -> Review =
  let r: Review = reviewer.ask(
    "Review this artifact:\n%{artifact}",
    on-parse-error = Retry(n = 2)
  )
  r

program def main(spec: text) -> unit =
  let artifact: text = ask "Implement %{spec}"
  let result = review-artifact(artifact)

  case result of
    | Pass => print "Accepted."
    | Fail(issues) => print(summarize-issues(issues))
```
