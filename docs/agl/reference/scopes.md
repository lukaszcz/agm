# Named Scopes

[← Index](index.md)

A named scope is a namespace inside a module. It groups static declarations
under a `::` path without creating a module or a runtime value. Scopes may be
nested and extended by later declarations.

## Regions and declaration paths

A `scope` region starts at the module root or inside another scope region and
ends with the same path:

```agl
scope Geometry
  record Point
    x: int
    y: int

  def origin() -> Point = ::Geometry::Point(0, 0)

  scope Format
    def label(point: ::Geometry::Point) -> text = "%{point.x},%{point.y}"
  end Format
end Geometry

def Geometry::translate(point: Geometry::Point) -> Geometry::Point =
  Geometry::Point(point.x + 1, point.y)

program def main() -> unit =
  let origin = Geometry::origin()
  print(Geometry::Format::label(Geometry::translate(origin)))
```

The closer is mandatory and must repeat the complete header path: `scope
A::B` closes with `end A::B`. A multi-segment region is equivalent to nested
single-segment regions.

A region's items may either sit in an indented block under the header, as
above, or share the header's own layout level. Both spell the same region; the
closer stands at the header's column either way. The indented form is the one
this project writes.

Repeating a region, or mixing a region with declaration paths, extends the same
scope:

```agl
scope Text
  def normalize(value: text) -> text = value
end Text

def Text::display(value: text) -> text = "[%{normalize(value)}]"

program def main() -> unit =
  print(Text::display("ready"))
```

A region contains nested regions, header `use` and `import` declarations,
`export` declarations, static declarations (`def`, `program def`, `extern def`,
`record`, `enum`, `exception`, `type`, every `builtin` form), and `let`/`var`
bindings. Bare expressions, `:=` assignments, and infix declarations are not allowed there.

## Binder paths

A region admits `let` and `var` bindings alongside its static declarations. A
binding's type is inferred from its initializer, exactly as at the module
root, and an explicit annotation is checked against it on either spelling:

```agl
scope Config
  let retries = 3
  var attempts: int = 0
end Config
```

`let` and `var` also accept a scope-path prefix on a single-name binder at the
module root, declaring a binding at that path directly — the same
declaration-path shorthand available for `def` and the type forms:

```agl
scope Config
  let retries = 3
  var attempts = 0
end Config
```

A `let` or `var` declaration head may be a plain qualifier chain: one or more
`::`-separated name segments, not anchored at the module root. It declares a
single scoped binding and cannot be a module route or contain type arguments:

| Spelling | Meaning |
|---|---|
| `let A::x = e` | scoped binding `A::x` |
| `let x = e` | root binding |
| `var A::x = e` | mutable scoped binding `A::x` |

A binding's initializer runs at its region's position in the module body: in
item order, together with the rest of the module's initializers, wherever the
region falls in the source text. A scope split across separate blocks resumes
exactly where the earlier block left off; a region never defers, reorders, or
repeats initialization.

## Import and export

A region also admits `import` and `export` declarations. Both are header
items, like `use`: they must precede the region's other items. A scoped
import tail's bare contribution narrows to its own region; its qualifier
route stays available module-wide, like any other import. A scoped export
re-roots every atom it forwards under the region's own path. See
[Modules](modules.md#imports-and-use-inside-a-scope-region) for the
complete semantics.

A simple (single-name, not `_`) scoped `let`/`var` is exported under its
declaration path exactly like a root one; see
[Modules](modules.md#re-exports-and-visibility). Any other scoped binding is
static state private to its module.

## Builtin declarations

A region admits every `builtin` form — `builtin record`, `builtin enum`,
`builtin exception`, `builtin def`, and `builtin var` — as a member of the
region, following the same visibility rules as every other member. A
`builtin` declaration's complete scoped name is different from an ordinary
member's, though: it is one host identity shared across the whole program at
that exact path, so it must be declared only once there. A `builtin var` host
identity additionally includes its defining module, scope path, and name — see [Built-in
functions](functions.md#built-in-functions). This allows a receiver method
such as `Agent::ask` to coexist with root `ask`. The example below therefore
presumes a program started with `--no-stdlib` ([Modules](modules.md#prelude)),
since `ExecResult` and `print` are otherwise already declared at those paths
by the automatically injected `std/prelude` prelude:

```agl
scope Host
  builtin record ExecResult
    stdout: text
    exit-code: int
    stderr: text
    timed-out: bool

  builtin def print[T](value: T) -> unit
end Host

program def main() -> unit =
  let result = Host::ExecResult(stdout = "x", exit-code = 0, stderr = "", timed-out = false)
  Host::print(result.stdout)
```

A scoped `builtin record`/`enum`/`exception` carries its declared scope path
as part of its nominal identity, exactly like an ordinary scoped type; a
scoped `builtin def` dispatches to the same host implementation as a root
one, reached bare inside its region or after `use`, and by its exact path
outside. A `builtin def` with first parameter `self` in a type scope is a
builtin method: `Agent::ask`, `Agent::ask-request`, and the receiver-based
`Session` operations use ordinary method selection, with the receiver supplying
the target agent or session. `builtin var` remains restricted to standard-library
modules regardless of scoping; `std/config` owns engine settings — see
[Program structure](program-structure.md#declarations).

## Names and visibility

A declaration belongs to its complete scope path. Members of the same scope are
visible by their bare names within that scope; enclosing scopes are considered
outward, then the module root and imported bare names. `::name` starts at the
module root, so it bypasses a nearer scoped member.

A static declaration (`def` or a type) is visible throughout its
scope regardless of textual order, matching the module root, where a `def` may
call another declared later in the same file. A `let` or `var` binding of a
module with a static root ([Library modules and
cycles](modules.md#library-modules-and-cycles)) — every file-backed module,
entry or library — is visible the same way: a `def` may read a binding
declared later in the same region or at the module root, and a `var` binding
may be written there too, since its initializer is already required to be a
constant.
This holds across separate blocks of the same scope too: a member declared in
an earlier `scope A` block is visible to a later `scope A` block and vice
versa.

Without a static root — the REPL, and an inline `-c` program with no
`program def` — a `let` or `var` binding keeps the textual rule instead: it
is visible only to references that follow it, in its own region or elsewhere
in the module, exactly as its root statements execute. A member declared in
an earlier `scope A` block there cannot see a binding a later `scope A` block
introduces, while the reverse order works. The same textual rule governs a
binding reached through `use`: a reference sees the binding once the
reference itself follows the binding's own declaration, regardless of where
the `use` appears — a `use` written before the scope that declares the
binding still exposes it to a later reference, just not to an earlier one.

A `let` or `var` name may not repeat a named scope's own name at the same
scope path, in either declaration order, at the module root or nested inside
another scope region — the same rule a `def` or type name follows against a
same-path scope.

A scoped `var` is assigned through its path (`A::count := 1`) or, inside its
region or after a `use`, through its bare name — the same forms that read it,
including through an import when the scope belongs to another module. A
scoped `let` is not assignable: `:=` on it is the same immutable-binder error
a root-level `let` raises. Assigning to a path that names a `def` or a type is
likewise rejected as immutable, and a path with no such member is a focused
error.

Outside a scope, qualify a member with its exact path. Scope paths never use
suffix matching: `Outer::Inner::work` does not make `Inner::work` available at
the module root. The leading `::` form makes an in-module path absolute, as in
`::Outer::Inner::work`. Module routes and scope paths share qualifier-chain
syntax; see [Lexical structure](lexical-structure.md#qualifier-chains).

Types establish same-named scopes. An inline enum member declares a record in
the enum's scope, so `Review::Pass` is an ordinary scoped record type and
constructor. Referenced members remain at their own declaration paths rather
than appearing in the referencing enum's scope. A `def` whose first parameter
is `self` is a method when its enclosing scope resolves to a record, enum, enum
member, or exception. It is called through a receiver value with `.`; a `def`
in the same scope with an ordinary first parameter remains a scoped function and
is called by its qualified path.

A declaration-path method and a method written in a `scope Type` region declare
members of the same resolved type scope. The two spellings can be mixed when
extending a type:

```agl
record Point
  x: int
  y: int

def Point::shift(self, amount: int) -> Point =
  Point(x = self.x + amount, y = self.y)

scope Point
  def total(self) -> int = self.x + self.y
end Point

program def main() -> unit =
  let point = Point(x = 2, y = 3)
  let shifted = point.shift(4)
  print(shifted.total())
```

A plain scope hosts methods for its resolved record, enum, enum member, or
exception. The receiver may be declared locally or supplied by exactly one bare
import reaching the method's region; a qualified-only import supplies no
receiver name. Built-in receiver heads are the exception: `array[E]::name` and
`dict[text, V]::name` declare methods for those generic receiver types, while
`text`, `json`, `int`, `decimal`, and `bool` are bare receiver heads. A type
alias may be used as a target type, but its scope cannot declare methods.
`Point::norm(p)` written bare follows ordinary scope-path rules, while
`p.norm()` aggregates visible method declarations across modules.

An enum member's terminal name is an injected bare constructor candidate. In
ordinary value position, scope resolution requires it to be the only visible
constructor candidate with that name: several candidates are a static scope
ambiguity, even when an expected enum type contains one of them. The expected
type checks the constructor after scope has selected it; it does not select a
same-named member. Enum-member patterns and `is` tests are different: their
scrutinee's static enum type selects the member. Module-root record and
exception construction keeps its bare type spelling (`Point(...)`); a scoped
type is constructed through its full path or after a `use` selects its enclosing
scope. A scope path is a route, not a type qualifier, so a scoped generic
constructor takes explicit type arguments after its name just as an
unqualified one does (`A::Pair::[int]`). An inline member may instead be
selected from an applied enum owner (`Option[int]::Some`); type arguments
applied directly to a generic member follow that member (`Option::Some::[int]`).

## Using a scope

`use` contributes selected members of a local scope, or of a scope in an
imported module, as bare names in its enclosing module or scope region. It is
a header declaration, so it appears before the region's other items.

```agl
use Math::*
use Text::{show as format}

scope Math
  def add(left: int, right: int) -> int = left + right

  scope Metrics
    def scale(value: int) -> int = value * 2
  end Metrics
end Math

scope Text
  def show(value: int) -> text = "value %{value}"
end Text

program def main() -> unit =
  let result = add(1, 2) + Metrics::scale(3)
  print(format(result))
```

`::*` selects every member. Brace tails select relative paths, and `hiding`
removes members from a glob. `as` adds a renamed bare route while leaving the
original path reachable. A use in a scope region contributes only to that
region and its nested regions.

A use reaches a scope in another module through an existing import route:

<!-- agl-check: fragment -->
```agl
import geo/shapes
use geo/shapes::Point::* hiding internal-distance
```

A use neither exports its contributions nor makes another module's uses
transitive. If several contributions provide the same bare name, the ambiguity
is reported when that name is used. Import selection and cross-module reach
are described in [Modules](modules.md).

## REPL

A scoped `let`/`var` persists across REPL entries by its full path,
exactly like a scoped `def` or type: a later entry may extend an existing
scope with a new member, and a same-path binding declared later replaces the
earlier one rather than colliding with it. A duplicate at the same path
within one entry is still an error. `:reset` clears every scoped binding
along with the rest of the session.

A named scope region itself is retained state, not a replaceable member: once
an entry declares `scope A`, a later entry's plain `let A = …`/`var A = …`
cannot reuse the name `A`, and once an entry declares a root `let A`/`var A`,
a later entry cannot reopen it as `scope A` — the same restriction governs a
retained `def` or type name against a later `scope` declaration, and vice
versa.
