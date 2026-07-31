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
record Point(x: int, y: int)

def origin() -> Point = ::Geometry::Point(0, 0)

scope Format
def label(point: ::Geometry::Point) -> text = "%{point.x},%{point.y}"
end Format
end Geometry

def Geometry::translate(point: Geometry::Point) -> Geometry::Point =
  Geometry::Point(point.x + 1, point.y)

let origin = Geometry::origin()
print(Geometry::Format::label(Geometry::translate(origin)))
```

The closer is mandatory and must repeat the complete header path: `scope
A::B` closes with `end A::B`. A multi-segment region is equivalent to nested
single-segment regions. Repeating a region, or mixing a region with declaration
paths, extends the same scope:

```agl
scope Text
def normalize(value: text) -> text = value
end Text

def Text::display(value: text) -> text = "[%{normalize(value)}]"

print(Text::display("ready"))
```

A region contains nested regions, header `open` and `import` declarations,
`export` declarations, static declarations (`def`, `extern def`, `record`,
`enum`, `exception`, `type`, every `builtin` form, and, in an entry module,
`agent`), `param` declarations, and `let`/`var` bindings. Bare expressions,
`:=` assignments, program declarations, and infix declarations are not
allowed there.

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
let Config::retries = 3
var Config::attempts = 0
```

A `let` pattern written as a plain qualifier chain — one or more `::`-separated
name segments, not anchored at the module root — declares a scoped binding
rather than matching a pattern: it is exactly the chain a declaration path
could spell. Writing an argument list, even an empty one, an `as` binder, a
module route, a type-argument-applied segment, or a `::` anchor keeps the
pattern's ordinary match meaning:

| Spelling | Meaning |
|---|---|
| `let A::x = e` | scoped binding `A::x` |
| `let A::x() = e` | nullary constructor pattern, qualified |
| `let A::x(a, b) = e` | constructor pattern with fields |
| `let A::x as y = e` | constructor pattern bound to `y` |
| `let x = e` | root binding |
| `let ::x = e` | constructor pattern, anchored at the module root |

The path prefix names a single binder, so it has no destructuring spelling.
Written inside a region instead, a destructuring `let` binds every name its
pattern selects as a member of that scope:

```agl
record Bounds(low: int, high: int)

scope Config
let Bounds(low, high) = Bounds(low = 0, high = 10)
end Config

print(Config::low)
print(Config::high)
```

A binding's initializer runs at its region's position in the module body: in
item order, together with the rest of the module's initializers, wherever the
region falls in the source text. A scope split across separate blocks resumes
exactly where the earlier block left off; a region never defers, reorders, or
repeats initialization.

## Parameters

A region also admits `param` declarations, with no declaration-path
shorthand — only the region form:

```agl
scope Deploy
param region: text = "eu"
param replicas: int
end Deploy

print("%{Deploy::region} x %{Deploy::replicas}")
```

A scoped parameter follows the same member and duplicate rules as every other
member: visible bare inside its region, by its exact path from outside, and
through `open`. Its **external key** — the name the CLI flag and the config
table entry use to supply a value — is its full path spelling
(`Deploy::region`), which is what makes grouping related parameters under one
scope useful. See [Host environment](host-environment.md#params) for how the
host resolves an external param value.

## Import and export

A region also admits `import` and `export` declarations. Both are header
items, like `open`: they must precede the region's other items. A scoped
import's bare contribution —
`open import` or `import … using` — narrows to its own region; its qualifier
route stays available module-wide, like any other import. A scoped export
re-roots every atom it forwards under the region's own path. See
[Modules](modules.md#import-and-export-inside-a-scope-region) for the
complete semantics.

Scoped bindings are never exported: library modules reject top-level `let`
and `var` entirely, so a region's `let`/`var` members have no cross-module
story.

## Builtin declarations

A region admits every `builtin` form — `builtin record`, `builtin enum`,
`builtin exception`, `builtin def`, and `builtin var` — following the same
member, duplicate, and visibility rules as every other member:

```agl
scope Host
builtin record ExecResult
  stdout: text
  exit_code: int
  stderr: text
  timed_out: bool

builtin def print[T](value: T) -> unit
end Host

let result = Host::ExecResult(stdout = "x", exit_code = 0, stderr = "", timed_out = false)
Host::print(result.stdout)
```

A scoped `builtin record`/`enum`/`exception` is a nominal distinct from a
same-named one at another path, exactly like an ordinary scoped type; a
scoped `builtin def` dispatches to the same host implementation as a root
one, reached bare inside its region or after `open`, and by its exact path
outside. `builtin var` keeps its separate restriction to the canonical
`std/config` module regardless of scoping — see
[Program structure](program-structure.md#declarations).

## Names and visibility

A declaration belongs to its complete scope path. Members of the same scope are
visible by their bare names within that scope; enclosing scopes are considered
outward, then the module root and imported bare names. `::name` starts at the
module root, so it bypasses a nearer scoped member.

A static declaration (`def`, a type, an `agent`) is visible throughout its
scope regardless of textual order, matching the module root, where a `def` may
call another declared later in the same file. A `let`, `var`, or `param`
binding is different: it is visible only to references that follow it
textually, in its own region or elsewhere in the module — exactly like a
root-level `let` or `param`. This holds across separate blocks of the same
scope: a member declared in an earlier `scope A` block cannot see a binding a
later `scope A` block introduces, while the reverse order works. The same
textual rule governs a binding reached through `open` — plain, `using`, or
`hiding` alike: a reference sees the binding once the reference itself
follows the binding's own declaration, regardless of where the `open`
appears — an `open` written before the scope that declares the binding still
exposes it to a later reference, just not to an earlier one.

A scoped `var` is assigned through its path (`A::count := 1`) or, inside its
region or after an `open`, through its bare name. A scoped `let` is not
assignable: `:=` on it is the same immutable-binder error a root-level `let`
raises. Assigning to a path that names a `def`, a type, or an `agent` is
likewise rejected as immutable, and a path with no such member is a focused
error.

Outside a scope, qualify a member with its exact path. Scope paths never use
suffix matching: `Outer::Inner::work` does not make `Inner::work` available at
the module root. The leading `::` form makes an in-module path absolute, as in
`::Outer::Inner::work`. Module routes and scope paths share qualifier-chain
syntax; see [Lexical structure](lexical-structure.md#qualifier-chains).

Types establish same-named scopes. Enum variants are members of the enum's
scope, so `Review::Pass` is an ordinary scoped member. A `def` whose first
parameter is `self` is a method when its enclosing scope is a record, enum, or
exception. It is called through a receiver value with `.`; a `def` in the same
scope with an ordinary first parameter remains a scoped function and is called
by its qualified path.

A declaration-path method and a method written in a `scope Type` region declare
members of the same type scope. The two spellings can be mixed when extending a
type:

```agl
record Point
  x: int
  y: int

def Point::shift(self, amount: int) -> Point =
  Point(x = self.x + amount, y = self.y)

scope Point
def total(self) -> int = self.x + self.y
end Point

let point = Point(x = 2, y = 3)
let shifted = point.shift(4)
print(shifted.total())
```

Methods may be declared only in the module that declares their receiver type.
To add behavior to a type from another module, declare a plain function that
takes the value as an ordinary parameter and call that function directly. A
type alias may be used as its target type, but its scope cannot declare methods;
methods are declared only on records, enums, and exceptions.

The familiar bare-variant spelling remains available when it is unambiguous or
selected by the expected enum type. Module-root record and exception
construction keeps its bare type spelling (`Point(...)`); a scoped type is
constructed through its full path or after opening its enclosing scope. A scope
path is a route, not a type qualifier, so a scoped generic constructor takes
explicit type arguments on the constructor just as an unqualified one does
(`A::Pair::[int]`); only a variant qualified by its owning enum puts them on the
type (`Option[int]::some`).

## Opening a scope

`open` contributes selected members of a local scope, or of a scope in an
imported module, as bare names in its enclosing module or scope region. It is
a header declaration, so it appears before the region's other items.

```agl
open Math
open Text using show as format

scope Math
def add(left: int, right: int) -> int = left + right
scope Metrics
def scale(value: int) -> int = value * 2
end Metrics
end Math

scope Text
def show(value: int) -> text = "value %{value}"
end Text

let result = add(1, 2) + Metrics::scale(3)
print(format(result))
```

A plain `open` selects every member. `using` selects listed relative paths, and
`hiding` selects every member except those paths. `using name as replacement`
renames the selected path: a direct member becomes `replacement`, while a
selected nested scope retains its path below the replacement. An open in a
scope region contributes only to that region and its nested regions.

An open reaches a scope in another module through its import route:

<!-- agl-check: fragment -->
```agl
import geo/shapes
open geo/shapes::Point hiding internal-distance
```

An open neither exports its contributions nor makes another module's opens
transitive. If several contributions provide the same bare name, the ambiguity
is reported when that name is used. Import selection and cross-module reach
are described in [Modules](modules.md).

## REPL

A scoped `let`/`var`/`param` persists across REPL entries by its full path,
exactly like a scoped `def` or type: a later entry may extend an existing
scope with a new member, and a same-path binding declared later replaces the
earlier one rather than colliding with it. A duplicate at the same path
within one entry is still an error. `:reset` clears every scoped binding
along with the rest of the session.
