# Modules

[← Index](index.md)

AgL programs are composed from file-based modules. Imports expose public
members through qualified routes and, when requested, bare names.

## Slash-path identity

A module identity is its slash path: the relative path to its `.agl` file,
without the suffix. For example, `utils/strings.agl` has identity
`utils/strings`. The entry program has no path identity.

A slash path is written byte-adjacent wherever it appears — in a header, a
qualifier, or a wildcard tail. `a/b` is a path; `a / b`, spaced on both sides,
is division. A `/` touching an operand on exactly one side (`a/ b`) is neither
and is rejected; see
[Qualifier chains](lexical-structure.md#qualifier-chains).

A module must resolve to exactly one file across the configured library roots.
No matching file is an error; more than one matching file is also an error.
There is no root-priority shadowing. Wildcard imports select matching modules
from the same global module set.

## One-set imports

```ebnf
import_decl ::= ["open"] "import" module_path ["/*"]
                ["as" ref_name]
                [using_clause | hiding_clause]

module_path ::= NAME ("/" NAME)*
using_clause ::= "using" path_atom ["as" ref_name] ("," path_atom ["as" ref_name])*
hiding_clause ::= "hiding" path_atom ("," path_atom)*
path_atom ::= NAME ("::" NAME)*
```

Each import contributes a selected set **S** of its target module's public
members:

- no clause selects every public member;
- `using` selects exactly its listed members;
- `hiding` selects every public member except its listed members.

Both bare access and qualified access are bounded by **S**. Repeated imports
of the same module union their selected sets and bare-name contributions.
This makes a small bare surface plus a full qualified API explicit:

<!-- agl-check: fragment -->
```agl
import utils/strings
import utils/strings using trim
```

`using` and `hiding` name public declaration paths. Selecting a scope path
selects its complete public subtree, including a same-named type and its enum
variants. Selecting a path with no public content is an error. A bare path
contributed by several imports is an error when used, not when imported; scopes
with the same spelling from different modules never merge.

## `open`, `using`, and `hiding`

A plain import contributes **S** only to qualified routes. `using` injects its
selected names into the bare namespace. `open import` injects all of **S**
bare; combining `open` and `using` is redundant and invalid.

<!-- agl-check: fragment -->
```agl
import utils/strings
open import app/vocabulary hiding internal-word
import text/format using render as format
```

A `using N as M` rename is canonical: it replaces the selected path prefix
for both bare and qualified access through that import. Thus `using Point as P`
exposes `P::…`, while `using Point::distance as d` exposes `d`. The original
path is inaccessible through that import. `hiding` removes a path and its
subtree from both channels, so it can also remove a qualification ambiguity.

## Import and export inside a scope region

`import` and `export` are also legal [named scope](scopes.md) region items.
Only the *bare* contribution narrows to the region: `open import` or
`import … using` inside `scope A` makes the selected names bare inside `A`
only, not at the module root or in a sibling region. The qualifier route a
scoped import establishes is unaffected and stays available module-wide, so a
plain `import m` inside a region behaves exactly as it does at the root — the
meaningful scoped forms are `open import m` and `import m using …`:

<!-- agl-check: fragment -->
```agl
scope Vec
open import geom/planar
def norm(p: Point) -> float = mag(p)
end Vec
```

`import` and `export` are both header items inside a region, exactly like
`open`: they precede the region's other items. A scoped `export` re-roots
every atom it forwards under the region's own scope path, exactly as a
`using … as` rename re-roots a selected atom:

<!-- agl-check: fragment -->
```agl
scope Geo
export geom/planar using Point
end Geo
```

publishes the atom `Geo::Point`, forwarding to `geom/planar`'s `Point`. An
importer reaches it as `facade::Geo::Point`. Wildcard and `hiding` forms
re-root every forwarded atom the same way, and a rename composes with the
re-rooting.

## Opening scopes

An `open` declaration makes a [named scope](scopes.md)'s selected members
available bare in its enclosing module or scope region:

<!-- agl-check: fragment -->
```agl
open Point
open Text using render as format
open geo/shapes::Point hiding internal-distance
```

A plain `open` selects every member. `using` selects only the listed paths;
its `as` renames re-root the selected path, so a direct member becomes the new
plain name. `hiding` selects every member except its listed paths. Members of
nested scopes retain their relative paths. An `open` in a scope region affects
only that region and its nested regions.

An opened scope may be local or reached through an imported module route.
Selecting an unknown member is an error. Type-named scopes include enum variants and extension
members. Opens neither export their members nor make another module's opens
transitively available. Bare-name collisions are reported when the name is
used, including collisions with an `open import` contribution.

## Aliases

`as A` gives an import the single-name alias `A` instead of a path route. It
does not make names bare. An aliased import is reached only through its alias;
it does not participate in suffix or anchored path matching.

<!-- agl-check: fragment -->
```agl
import company/tools/config as settings

settings::timeout
```

Distinct modules may share an alias. The alias then acts as a facade: the
requested member resolves when exactly one aliased module contributes it.
Imports of one module merge normally, so importing it both plainly and with an
alias makes both routes available.

## Wildcards

`import prefix/*` expands to one import per module whose slash path is `prefix`
or starts with `prefix/`. The import's `open`, selection clause, and alias
apply independently to every matched module. A `using` or `hiding` path must
be public in every matched module, so scoped path selections distribute to
each matched module.

<!-- agl-check: fragment -->
```agl
import tools/*
open import domain/* hiding debug
import codecs/* as codec
```

An alias on a wildcard is a shared alias facade, not a path rewrite.

## Suffix and anchored references

A module qualifier ends in `::`:

<!-- agl-check: fragment -->
```agl
import company/tools/config
import service/config as settings

company/tools/config::timeout
config::timeout                 # suffix route
settings::timeout               # alias route
/company/tools/config::timeout  # anchored plain-path route
```

A non-aliased imported path may be named by any trailing sequence of its path
segments. A qualifier route may match several imported modules; AgL filters
those candidates by the requested member's contributed set **S**. One
remaining candidate resolves; several are ambiguous; none is an error.
There is no preference by route length, alias, or import order.

A leading `/` anchors a qualifier to the complete plain module path. Anchored
qualifiers never match aliases and are always module routes. Aliases are
single-segment routes only.

Qualified type references follow the same rules and preserve module-and-scope
nominal identity:

<!-- agl-check: fragment -->
```agl
import shapes/points as points

let p: points::Point = points::Point(x = 0, y = 0)
```

`::name` refers to a declaration in the current module root and bypasses a
lexical shadow. The same form works for `::Type` and `::Type::Variant`.
Type-qualified constructors use `Type::Variant`; a short spelling can name an
in-scope type or a module route and is resolved at the use site.

## Re-exports and visibility

`def`, `record`, `enum`, `exception`, and `type` declarations are exported
under their full declaration paths. Grouping helpers in a
[named scope](scopes.md) keeps them off a module's bare surface: an importer
reaches such a member only through its full scope path. A module that must
publish a narrower surface does so with a facade — the implementation lives in
one module, and another re-exports the selection it means to publish.

`export` re-exports members without injecting them into the exporting module's
local scope. A method travels with its receiver type: any module with a value
of that type can call the method without importing the module that declared it.
Import selections and facades cannot hide a method; they control access to
qualified declarations, not member calls.

<!-- agl-check: fragment -->
```agl
export math/basic using add, multiply as mul
export math/advanced hiding internal-helper
export math/*
```

Re-exports preserve the original defining-module identity. Conflicting exposed
names with different origins are static errors; duplicate paths to the same
origin are allowed.

## Prelude

Every loaded entry and library module, except `std/core` itself, implicitly
behaves as if it began with `open import std/core`. The `--no-stdlib` option
disables that automatic opening throughout the loaded program; an explicit
`import std/core` or `open import std/core` always follows the ordinary import
rules.

## Library modules and cycles

Imported modules are declaration-only: they may contain imports, exports,
functions, type declarations, and infix declarations, but not executable
top-level expressions, bindings, agents, parameters, or program declarations.
Imports and exports appear before other declarations in a library module; a
named scope region is one declaration for this rule, so an import or export
inside a region does not need to precede the module's other root-level
declarations — only a region's own header rule governs its own items.

Import cycles are valid. Functions and nominal types may refer to public
declarations across an import cycle.

## REPL

REPL imports persist after a successful entry, retained as written: a retained
wildcard expands again on every later entry, so it picks up modules added since.
A later entry replaces the earlier import declaration for every module it names,
so its selection, open mode, or alias takes effect for that module. Multiple
declarations for one module in the same entry merge normally.
A failed entry changes no imports, and `:reset` clears imports with the session
bindings. Each REPL entry and its loaded library modules receive the
`std/core` prelude unless the session was launched with `--no-stdlib`.

## Diagnostics

Imports report a missing or ambiguous module path, a selected name the module
does not declare, redundant `open ... using`, or an import placed after a non-import
item. A qualified use reports an unknown qualifier, a member outside its
contributed set, or every candidate of an ambiguous route. A bare use reports an ambiguous bare name only at its use
site. These diagnostics identify a direct repair: add a longer suffix or an
anchored path, use an alias, adjust `hiding`, or select the required name.
