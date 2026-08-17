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
path_atom ::= (NAME "::")* name
name      ::= NAME | OP_NAME
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
re-rooting. Ordinary re-export cycles that preserve names are allowed, but a
cycle that repeatedly expands a scoped path is rejected as a scope error.

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

## Standard library modules

- `std/core` declares the automatically opened core types, exceptions, and built-ins. It re-exports `std/option`, `std/pair`, `std/either`, and `std/result`, so their types and helpers remain available through the prelude and through `import std/core using …`. It also defines the `|>`, `<|`, `>>`, and `<<` combinators.
- `std/option` declares `Option[T]`, its methods, and `UnwrapError`.
- `std/pair` declares `Pair[A, B]` and its mapping and swapping methods.
- `std/either` declares the neutral `Either[A, B]` sum and its mapping, query, optional-projection, and swapping methods.
- `std/result` declares `Result[T, E]`, its outcome methods, and `attempt`, which turns a raising nullary function into a `Result`.
- `std/config` exposes the host engine settings as `builtin var` bindings.
- `std/array` owns array methods and array utility functions; see
  [`std/array`](#stdarray).
- `std/dict` owns dictionary methods and conversion from key/value pairs; see
  [`std/dict`](#stddict).
- `std/text` owns the ambient `text` methods and exposes
  `interp(template, vars) -> text` for name-only runtime interpolation; import
  it to call `interp`, and see [Types](types.md#stdtext) and
  [Strings and interpolation](strings-and-interpolation.md#runtime-interpolation).
- `std/json` owns ambient `json` inspection methods and explicit strict and
  lenient parsers; see [`std/json`](#stdjson).
- `std/toml` converts TOML documents to and from `json`; see
  [`std/toml`](#stdtoml).
- `std/fs` exposes explicit text filesystem operations: `read`, `write`,
  `append`, `exists`, and `list`; see [`std/fs`](expressions.md#stdfs).

## `std/array`

`std/array` owns methods on `array[E]`. The `std/builtin-methods` registry makes
these methods available on every array without an import. Import `std/array` to
call its free functions, for example `array::range(1, 5)`; a plain import keeps
them qualified, while `open import std/array` also makes them bare.

Operations with a `?` suffix return `Option`; `first`, `last`, and `pop`
raise `IndexError` when no element is available. `index-of` instead returns
`-1` when absent, while `index-of?` returns `Option::None`. A `!` suffix marks
an in-place counterpart of a pure operation. `append`, `insert`, `pop`,
`remove-at`, `clear`, and `extend` are inherently mutating. `remove-at(i)`
accepts the same negative positions as normal array indexing. `insert` accepts
insertion boundaries from `-size()`
(the start) through `size()` (the end) and raises `IndexError` outside that
range. `slice` follows the half-open `[start, end)` convention; `take` and
`drop` clamp a negative count to zero. `range(a, b)` includes both endpoints
and descends when `a > b`.

| Method | Result |
| --- | --- |
| `size()` / `is-empty()` | Element count / whether it is zero. |
| `first()` / `first?()` / `last()` / `last?()` | First or last element, raising or as `Option`. |
| `append(x)` / `insert(i, x)` / `clear()` / `extend(other)` | Mutate the receiver and return `unit`. |
| `pop()` / `pop?()` / `remove-at(i)` | Remove and return an element, raising or optional where provided. |
| `contains(x)` / `index-of(x)` / `index-of?(x)` | Membership and the first index (`-1` or `Option::None` when absent); all use AgL equality. |
| `count(p)` / `any(p)` / `all(p)` | Count matching elements, or test whether any/all match. |
| `map(f)` / `map!(f)` | Transform every element into a new array (possibly with a new element type) / mutate the receiver while preserving its element type. |
| `filter(p)` / `filter!(p)` | Keep matching elements in a new array / mutate the receiver. |
| `each(f)` | Call `f` for each element in order and return `unit`. |
| `fold(init, f)` / `fold-right(init, f)` | Reduce left-to-right / right-to-left; `f` receives `(accumulator, element)`. |
| `find?(p)` / `find-index?(p)` | First matching element or index as `Option`. |
| `reverse()` / `reverse!()` | Reversed copy / reverse the receiver. |
| `sort(cmp)` / `sort!(cmp)` | Sorted copy / sort the receiver with `cmp(left, right) -> int`. |
| `slice(start, end)` / `take(n)` / `drop(n)` | A half-open slice, first `n`, or all but first `n` elements. |
| `concat(other)` | A new array containing both arrays. |
| `zip(other)` | `array[Pair[E, F]]`, truncated to the shorter input. |
| `enumerate()` | `array[Pair[int, E]]` pairing each element with its zero-based index. |

The callback-taking methods invoke their AgL closures in encounter order. A
callback may capture local bindings and may raise normally.

| Free function | Result |
| --- | --- |
| `join(xs, sep)` | Concatenates `array[text]` with `sep`. |
| `flatten(xs)` | Concatenates nested arrays in order. |
| `unzip(xs)` | `Pair(first = array[A], second = array[B])`. |
| `repeat(x, n)` | `n` copies of `x` (empty when `n` is negative). |
| `range(a, b)` | Inclusive integer sequence from `a` to `b`. |

## `std/dict`

`std/dict` owns methods on `dict[text, V]`. The `std/builtin-methods` registry
makes these methods available on every dictionary without an import. Import
`std/dict` to call `dict::from-entries`; a plain import keeps it qualified,
while `open import std/dict` also makes it bare.

Operations with a `?` suffix return `Option`; `get` and `remove` raise
`KeyError` for a missing key. A `!` suffix marks the in-place counterpart of a
pure operation. `clear` and `remove` are inherently mutating. Dictionary order
is preserved by `keys`, `values`, `entries`, and callback traversal.

| Method | Result |
| --- | --- |
| `size()` / `is-empty()` | Entry count / whether it is zero. |
| `get(k)` / `get?(k)` | Value for `k`, raising or as `Option`. |
| `remove(k)` / `remove?(k)` | Remove and return the value for `k`, raising or as `Option`. |
| `contains(k)` | Whether `k` is present. |
| `clear()` | Remove every entry and return `unit`. |
| `keys()` / `values()` / `entries()` | Ordered `array[text]`, `array[V]`, or `array[Pair[text, V]]`. |
| `merge(other)` / `merge!(other)` | New overlaid dictionary / overlay this dictionary with `other`; the right-hand value wins. |
| `map-values(f)` | New dictionary whose values are transformed by `f`. |
| `filter(p)` / `filter!(p)` | New filtered dictionary / filter this dictionary; `p` receives `(key, value)`. |
| `each(f)` | Call `f(key, value)` for each entry in order and return `unit`. |

| Free function | Result |
| --- | --- |
| `from-entries(xs)` | `dict[text, V]` built from `array[Pair[text, V]]`; later duplicate keys win. |

## `std/json`

`std/json` provides parsing and inspection for untyped `json` values. The
`std/builtin-methods` registry makes its methods ambient; import `std/json` to
call its free functions. A plain import keeps those functions qualified, so
use `json::parse(raw)`; `open import std/json` also makes them bare.

```agl
import std/json

program def main() -> unit =
  let strict = json::parse('{"count": 2}')
  let recovered = json::parse-lenient("```json\n[1, 2]\n```")
  print(strict)
  print(recovered)
```

`parse(text)` accepts exactly one JSON value, apart from surrounding
whitespace, and raises `JsonParseError` otherwise. `parse?(text)` returns
`Option::None` instead of raising. `parse-lenient(text)` and
`parse-lenient?(text)` use the same recovery rules as structured agent and
shell output, allowing fenced or prose-wrapped JSON; their strict counterparts
never recover or repair input. Rendering remains `render(value)` or `value as
text`, rather than a `std/json` operation.

| Method | Result |
| --- | --- |
| `kind()` | One of `null`, `bool`, `int`, `decimal`, `text`, `array`, or `object`. |
| `size()` | Object property or array element count; `0` for scalars. |
| `keys()` | Object keys in source order, or an empty array for other values. |
| `has(key)` | Whether an object has `key`; `false` for other values. |
| `get(key)` / `get?(key)` | Object value, raising `KeyError` or returning `Option::None` when the key is absent or the receiver is not an object. |

JSON values remain index-only for data access; use `value["key"]` or
`value[index]` to index the JSON tree directly.

## `std/toml`

`std/toml` converts TOML documents to and from untyped `json` values. Import it
to use its functions:

```agl
import std/toml

program def main() -> unit =
  let settings = toml::parse('''
[server]
port = 8080
''')
  print(toml::render(settings))
```

`parse(text)` accepts one TOML document and returns its root table as `json`.
Nested tables and arrays retain their JSON object and array shapes. TOML floats
become `decimal`, while TOML date, time, and datetime values become ISO-8601
text. Malformed input raises `TomlParseError`; `parse?(text)` returns
`Option::None` instead.

`render(value)` serializes a JSON object as a TOML document. A TOML document
must have a table root, and TOML has no null value, so rendering a non-object
root or any value containing `null` raises `TomlRenderError`. TOML integers
must fit the signed 64-bit range; an out-of-range integer also raises that
exception. Decimal values render as TOML floats, including ordinary signed
`nan` and infinity values. TOML cannot preserve a signaling or payload NaN,
so either also raises `TomlRenderError`. The resulting text can be passed to
`parse` to recover the same JSON representation.

## Library modules and cycles

Every file-backed module has a static root: imports, declarations, parameters,
and `let`/`var` bindings are allowed there, while bare expressions and assignments
are not. Root binding initializers must be constant expressions: literals,
literal containers, constructor applications, and unary operators over those.
Put executable
workflow code in a `program def` body. Parameters are also legal in named scope
regions in every module. A program receives values for the params in its module
and transitive imports before it starts.
Imports and exports appear before other declarations at a module's root, in
every module, entry or library; a named scope region is one declaration for
this rule, so an import or export inside a region does not need to precede
the module's other root-level declarations — only a region's own header rule
governs its own items.

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
