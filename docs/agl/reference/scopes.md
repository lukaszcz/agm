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

A region contains only nested regions, `open` declarations, and static
declarations: `def`, `extern def`, `record`, `enum`, `exception`, `type`, and,
in an entry module, `agent`. Bindings, expressions, imports, exports,
parameters, program declarations, infix declarations, and `builtin`
declarations are not allowed there.

## Names and visibility

A declaration belongs to its complete scope path. Members of the same scope are
visible by their bare names within that scope; enclosing scopes are considered
outward, then the module root and imported bare names. `::name` starts at the
module root, so it bypasses a nearer scoped member.

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
