# Modules

[← Index](index.md)

AgL programs are composed from file-based modules. An `import` makes another
module's public declarations available through qualified routes. An import tail
or a `use` declaration adds selected declarations to a bare namespace.

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

## Imports

```ebnf
import_decl ::= "import" module_path ["/*"]
                ("as" NAME | "::" tail)? [hiding_clause]

tail             ::= "*" | import_item | "{" import_item ("," import_item)* ","? "}"
import_item      ::= path_atom ["as" ref_name]
hiding_clause    ::= "hiding" path_atom ("," path_atom)*
path_atom        ::= (NAME "::")* name
name             ::= NAME | OP_NAME
```

An import alias must be an identifier because it becomes a qualifier segment.
An import always contributes the target module's full public qualified surface,
except for paths named by `hiding`. A positive `::` tail controls only its bare
contribution:

- `import m` contributes qualified routes only.
- `import m::*` makes every public member bare.
- `import m::{f, Config::timeout}` makes those members bare.

A selected scope path includes its complete public subtree. An item rename adds
a corresponding bare path while leaving the selected source path available.
`hiding` removes a path and its subtree from both qualified and bare access.
It may accompany a plain import or a `::*` tail, but not a positive selection.
Selected and hidden paths must be public declarations of every module matched
by a wildcard.

<!-- agl-check: fragment -->
```agl
import utils/strings
import text/format::{render as format}
import app/vocabulary::* hiding internal-word
```

An import alias supplies a single-segment qualified route instead of the
module's slash-path routes. It does not make names bare and cannot be combined
with an import tail. A tail rename is additive: it adds a bare route without
removing the selected source path.

<!-- agl-check: fragment -->
```agl
import company/tools/config as settings

settings::timeout
```

Repeated imports of a module combine their qualified routes and bare
contributions. A bare-name collision is reported at the use site. Qualified
routes may also be ambiguous; a longer suffix, an anchored path, or an alias
selects one route.

## `use` declarations

A `use` declaration selects public members of an already-nameable named scope
or module root. Its target can be a local scope, a scope or module root reached
through an import route, or a scope anchored at the current module root.

```ebnf
use_decl      ::= "use" use_target ("::" tail | "as" ref_name)
                  [hiding_clause]
use_target    ::= ["/"] module_path ["::" scope_path] | "::" scope_path
scope_path    ::= NAME ("::" NAME)*
```

When an unanchored target with no `::` scope suffix names an imported module,
it targets that module's root. The module must already be imported:

<!-- agl-check: fragment -->
```agl
import library
use library::*
use library as Alias
```

`use Scope::*` contributes every member of `Scope` bare. A braced or
single-atom tail selects members, and item renames add renamed bare paths.
When `use Scope::member as Alias` ends at an ordinary member, it is the
single-item rename; when the complete path names a scope, it is a whole-target
alias. `use Scope as Alias` and `use library as Alias` contribute every selected
member beneath `Alias`; a whole-target alias must be an identifier because it
becomes a qualifier segment. `hiding` is valid only with a `::*` tail. A `use` declaration contributes names
only to its enclosing module or named scope region; it does not make a module
available. Import the module first when its target is not local.

```agl
use Math::*

scope Math
def add(left: int, right: int) -> int = left + right
end Math

program def main() -> unit =
  let _ = print(add(1, 2))
```

An explicit `/` anchors a module route, and a leading `::` anchors a local
scope path. Without an anchor, a target can be resolved through a local scope
or an already-bare imported scope; ambiguity is a static error. A named scope exposed by one
`use` is already bare and can therefore be the target of a later `use`. Selection, renaming, and
hiding determine which nested scope paths the later declaration can target.

<!-- agl-check: fragment -->
```agl
import geo/shapes
use geo/shapes::Point::{distance as point-distance}
use /geo/shapes::Point::* hiding internal-distance

import library
use library::Outer::{Inner as Selected}
use Selected::*
```

## Imports and `use` inside a scope region

`import`, `use`, and `export` are header items inside a named scope region.
They precede the region's other items. A scoped import's qualified route remains
available throughout its module, while a tail's bare names belong only to that
region and its nested regions. A scoped `use` likewise contributes only to its
enclosing region and nested regions.

<!-- agl-check: fragment -->
```agl
scope Vec
import geom/planar::{Point}
use Point::*
def norm(p: Point) -> float = mag(p)
end Vec
```

A scoped export re-roots every forwarded atom under the region's own scope
path. An importer reaches it through that path.

## Wildcards

`import prefix/*` expands to one import per module whose slash path is `prefix`
or starts with `prefix/`. Its alias, tail, and hiding clause apply independently
to every matched module. An alias on a wildcard is a shared alias facade, not a
path rewrite.

<!-- agl-check: fragment -->
```agl
import tools/*::*
import domain/*::{render}
import codecs/* as codec
```

## Suffix and anchored references

A module qualifier ends in `::`:

<!-- agl-check: fragment -->
```agl
import company/tools/config
import service/config as settings

company/tools/config::timeout
config::timeout
settings::timeout
/company/tools/config::timeout
```

A non-aliased imported path may be named by any trailing sequence of its path
segments. A qualifier route may match several imported modules; the requested
member resolves when their contributions identify one declaration origin. Duplicate
routes to that origin are allowed. A leading `/` anchors a qualifier to the complete
plain module path. Anchored qualifiers never
match aliases. Aliases are single-segment routes only.

`::name` refers to a declaration in the current module root and bypasses a
lexical shadow. The same form works for `::Type` and `::Type::Variant`.
Qualified type references follow the same routing rules and preserve
module-and-scope nominal identity.

## Re-exports and visibility

`def`, `record`, `enum`, `exception`, and `type` declarations, plus annotated
simple `let` bindings, are exported under their full declaration paths. Grouping helpers in a
[named scope](scopes.md) keeps them off a module's bare surface: an importer
reaches such a member through its scope path or makes it bare with an import
tail or `use` declaration.

`export` forwards public declarations without injecting them into the exporting
module's local scope. A brace tail selects the declarations to forward; a plain
export forwards the complete public surface, and `hiding` removes paths from
that surface. Renames in a brace tail change the forwarded path. Once the
declaring module is loaded, a method travels with its receiver type: any module
with a value of that type can call the method without importing the module that
declared it. Selections, renames, and `hiding` cannot hide a method; they
control access to qualified declarations, not member calls.

<!-- agl-check: fragment -->
```agl
export math/basic::{add, multiply as mul}
export math/advanced hiding internal-helper
export math/*
```

Re-exports preserve the original defining-module identity. Conflicting exposed
names with different origins are static errors; duplicate paths to the same
origin are allowed. A public name cannot be both an ordinary declaration and a
named scope, including when either side is re-exported. Type declarations keep
their legal same-named, type-owned namespaces.

## Prelude

Every loaded entry and library module, except `std/core` itself, receives an
implicit `import std/core::*`. Any explicit import declaration whose expansion
includes `std/core`, including one inside a named scope region, supplies the
core contribution instead. Thus `import std/core` and `import std/*` leave core
names qualified-only, while `import std/core::* hiding ask` makes every core
member except `ask` bare.

```agl
import std/core::* hiding ask

program def main() -> unit =
  let _ = print("ready")
```

When the prelude is enabled, the optional `std/builtin-methods` registry is
also loaded when the standard library provides it, making its receiver methods
ambient. The `--no-stdlib` option disables both automatic additions; a custom
standard library may also omit the registry. Either way, importing a method's
owning module loads its methods. An explicit core import remains available with
`--no-stdlib`.

## Standard library modules

- `std/core` declares the core types, exceptions, and built-ins. It re-exports `std/option`, `std/pair`, `std/either`, and `std/result`, so their types and helpers reach a module through the prelude and through `import std/core::*`. It also defines the `|>`, `<|`, `>>`, and `<<` combinators.
- `std/option` declares `Option[T]`, its methods, and `UnwrapError`.
- `std/pair` declares `Pair[A, B]` and its mapping and swapping methods.
- `std/either` declares the neutral `Either[A, B]` sum and its mapping, query, optional-projection, and swapping methods.
- `std/result` declares `Result[T, E]`, its outcome methods, and `attempt`, which turns a raising nullary function into a `Result`.
- `std/config` exposes the host engine settings as `builtin var` bindings.
- `std/env` provides the ambient `Environ` snapshot and environment helpers; see
  [`std/env`](#stdenv).
- `std/process` provides process metadata and controlled program termination; see
  [`std/process`](#stdprocess).
- `std/math` owns scalar numeric methods, aggregates, and mathematical constants; see
  [`std/math`](#stdmath).
- `std/time` provides UTC clock, parsing, formatting, and sleep operations; see
  [`std/time`](#stdtime).
- `std/random` provides a seedable pseudo-random sequence, array selection and shuffling,
  and UUID generation; see [`std/random`](#stdrandom).
- `std/regex` provides Python-compatible regular-expression searching, rewriting, and
  splitting; see [`std/regex`](#stdregex).
- `std/array` owns array methods and array utility functions; see
  [`std/array`](#stdarray).
- `std/dict` owns dictionary methods and conversion from key/value pairs; see
  [`std/dict`](#stddict).
- `std/text` owns `text` methods and exposes `interp(template, vars) -> text`
  for name-only runtime interpolation; see [Types](types.md#stdtext) and
  [Strings and interpolation](strings-and-interpolation.md#runtime-interpolation).
- `std/json` owns `json` inspection methods and explicit strict and lenient
  parsers; see [`std/json`](#stdjson).
- `std/toml` converts TOML documents to and from `json`; see
  [`std/toml`](#stdtoml).
- `std/path` provides lexical host-platform path manipulation; see
  [Standard library](standard-library.md#stdpath).
- `std/fs` exposes text filesystem and directory operations, including typed
  `FsError` failures; see [Standard library](standard-library.md#stdfs).

## `std/env`

`std/env` models the ambient environment as `Environ(vars: dict[text, text])`.
Its `builtin var environ` is seeded by `agm exec` and `agm repl` from one full
snapshot of their startup process environment. It is independent from
`os.environ`: AgL changes never alter the host process, and the snapshot does
not change after startup. When the standard library is suppressed, no ambient
environment binding is installed.

`Environ` methods are `get(name)`, `get?(name)`, `set(name, value)`,
`unset(name)`, `contains(name)`, and `extended(overrides)`. `get` and `unset`
raise `KeyError` when the name is absent; `get?` returns `Option[text]`.
`extended` returns a new environment with an overlaid copy of the variables.

After `import std/env::*`, `getenv`, `getenv?`, `setenv`, and `unsetenv` are
shortcuts over `environ`. They mutate only this AgL-side environment:

<!-- agl-check: fragment -->
```agl
import std/env::*

let _ = setenv("MODE", "test")
let mode = getenv("MODE")
let child_env = environ.extended({"DEBUG": "1"})
let _ = unsetenv("MODE")
```

## `std/process`

`std/process` provides operations for the process executing an AgL program.
Import it explicitly:

```agl
import std/process

program def main() -> unit =
  let directory = process::cwd()
  let process_id = process::pid()
  let host = process::hostname()
  ()
```

`cwd() -> text` returns the current working directory, `pid() -> int` returns
that process's identifier, and `hostname() -> text` returns the host name.

`exit(code: int = 0) -> unit` terminates the program and its host process with
`code`. It does not return to subsequent AgL expressions. `code` must be in the
portable process-status range `0..255`, so the calling environment observes the
same number on every supported host. Omitting `code` uses zero; a zero code
indicates success and a nonzero code indicates failure. An out-of-range code is
a normal runtime error and does not terminate the host.

## `std/math`

`std/math` owns methods on `int` and `decimal`. When the standard-library
prelude injects `std/builtin-methods`, these methods are available on scalar
values without an import. Otherwise, import `std/math` before calling them;
import it also to call its free functions or read its constants. A plain import
keeps free functions qualified, for example `math::sum([1, 2, 3])` and
`math::pi`.

| Receiver | Method | Result |
| --- | --- | --- |
| `int` | `abs()`, `min(other)`, `max(other)`, `clamp(lower, upper)` | Integer absolute value or selected bound. |
| `int` | `compare(other)`, `sign()` | `-1`, `0`, or `1`. `compare` is suitable for an `array.sort` comparator through `fn(left, right) => left.compare(right)`. |
| `int` | `pow(exponent)` | Integer power for a non-negative exponent; a negative exponent raises `RangeError`. |
| `int` | `to-decimal()` | The same numeric value as `decimal`. |
| `decimal` | `abs()`, `min(other)`, `max(other)`, `clamp(lower, upper)` | Decimal absolute value or selected bound. |
| `decimal` | `compare(other)`, `sign()` | `-1`, `0`, or `1`. |
| `decimal` | `floor()`, `ceil()` | Greatest integer at or below / least integer at or above the value. |
| `decimal` | `round(digits = 0)` | Decimal rounded to `digits` fractional places using the language decimal context. |
| `decimal` | `sqrt()`, `pow(exponent)` | Square root or integer-exponent power under the language decimal context. |

`sum(values: array[int]) -> int` and
`sum-decimal(values: array[decimal]) -> decimal` add their values using their
respective numeric semantics. `pi` and `e` are `decimal` constants.

## `std/time`

`std/time` supplies clock and calendar conversion functions. Import it explicitly:

```agl
import std/time

program def main() -> unit =
  let epoch = time::parse-iso("2024-01-02T03:04:05+00:00")
  print(time::format(epoch, "%Y-%m-%d"))
```

`now() -> decimal` returns Unix epoch seconds and `now-iso() -> text` returns the current
UTC time as ISO-8601 text. `monotonic() -> decimal` returns a non-decreasing
clock suitable for measuring elapsed time, not calendar timestamps. `sleep(seconds) -> unit`
pauses for the requested number of seconds.

`parse-iso(text) -> decimal` parses an ISO-8601 timestamp with or without an offset; `format-iso(epoch) ->
text` emits the corresponding UTC timestamp. `parse(text, fmt) -> decimal` and `format(epoch,
fmt) -> text` use Python-compatible `strptime` and `strftime` directives. A parsed timestamp
without an offset is interpreted as UTC, as are all formatted epoch values. Invalid input or an
out-of-range temporal conversion in either parser raises `TimeParseError(raw: text)`.

## `std/random`

`std/random` owns an independent pseudo-random sequence for each running interpreter. Import it
explicitly and seed it when a reproducible sequence is needed:

```agl
import std/random

program def main() -> unit =
  random::seed(42)
  print(random::between(1, 6))
```

`seed(n)` resets that sequence. `below(n) -> int` returns an integer in `0..n-1`,
`between(lo, hi) -> int` includes both bounds, and `uniform() -> decimal` returns a decimal in
`[0, 1)`. `choice(xs) -> T` selects an element and raises `IndexError` for an empty array;
`choice?(xs) -> Option[T]` returns `Option::None` instead. `shuffle!(xs)` shuffles its array
receiver in place through the normal live array view. `uuid() -> text` returns a fresh UUIDv4
identifier and is intentionally independent of the seedable sequence.

## `std/regex`

`std/regex` provides regular-expression operations using Python's
[`re`](https://docs.python.org/3/library/re.html) pattern and replacement syntax. Import it
explicitly:

```agl
import std/regex

program def main() -> unit =
  case regex::find?("(?P<word>[A-Za-z]+)-([0-9]+)", "item-42") of
    | Option::Some(value = _ as found) => print(found.named-groups["word"])
    | Option::None => ()
```

`test(pattern, s) -> bool` reports whether the pattern occurs anywhere in `s`.
`find?(pattern, s) -> Option[Match]` returns the first occurrence, while
`find-all(pattern, s) -> array[Match]` returns non-overlapping occurrences from left to right.
Each `Match` has `matched`, zero-based half-open `start` and `end`, and `groups` in
numbered-group order. A group that did not participate is `Option::None`; a participating one
is `Option::Some(text)`. `named-groups` contains the participating named groups by name.

`replace(pattern, s, replacement) -> text` replaces every match and supports Python replacement
backreferences such as `\\1` and `\\g<name>`. `split(pattern, s) -> array[text]` uses Python
`re.split` behavior: boundary empty strings and captured separators are retained; an unmatched
captured separator is represented as the empty text because split results are text.
`escape(s) -> text` returns a pattern that matches `s` literally.

Every operation that accepts a pattern raises `RegexError(pattern: text)` when Python cannot
compile it. Compiled patterns are reused by an internal bounded companion cache; the cache has
no AgL-visible state.

## `std/array`

`std/array` owns methods on `array[E]`. When the standard-library prelude
injects `std/builtin-methods`, these methods are available on every array
without an import. Otherwise, import `std/array` before calling them; import it
also to call its free functions, for example `array::range(1, 5)`. A plain
import keeps free functions qualified, while `import std/array::*` also makes
them bare.

See [Standard-library conventions](standard-library.md#conventions) for the
`?`/`!` naming rules. `first`, `last`, and `pop` raise `IndexError` when no
element is available. `index-of` instead returns `-1` when absent, while
`index-of?` returns `Option::None`. `append`, `insert`, `pop`, `remove-at`,
`clear`, and `extend` are inherently mutating. `remove-at(i)` accepts the same
negative positions as normal array indexing. `insert` accepts insertion
boundaries from `-size()` (the start) through `size()` (the end) and raises
`IndexError` outside that
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

`std/dict` owns methods on `dict[text, V]`. When the standard-library prelude
injects `std/builtin-methods`, these methods are available on every dictionary
without an import. Otherwise, import `std/dict` before calling them; import it
also to call `dict::from-entries`. A plain import keeps it qualified, while
`import std/dict::*` also makes it bare.

See [Standard-library conventions](standard-library.md#conventions) for the
`?`/`!` naming rules. `get` and `remove` raise `KeyError` for a missing key.
`clear` and `remove` are inherently mutating. Dictionary order is preserved by
`keys`, `values`, `entries`, and callback traversal.

| Method | Result |
| --- | --- |
| `size()` / `is-empty()` | Entry count / whether it is zero. |
| `get(k)` / `get?(k)` | Value for `k`, raising or as `Option`. |
| `remove(k)` / `remove?(k)` | Remove and return the value for `k`, raising or as `Option`. |
| `set(k, v)` | Insert or replace `k` with `v` and return `unit`. |
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

`std/json` provides parsing and inspection for untyped `json` values. When the
standard-library prelude injects `std/builtin-methods`, its methods are ambient.
Otherwise, import `std/json` before calling them; import it also to call its
free functions. A plain import keeps those functions qualified, so use
`json::parse(raw)`; `import std/json::*` also makes them bare.

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
and `let`/`var` bindings are allowed there, while bare expressions and
assignments are not. Root binding initializers must be constant expressions:
literals, literal containers, constructor applications, and unary operators
over those. Put executable workflow code in a `program def` body. Parameters
are also legal in named scope regions in every module. A program receives
values for the params in its module and transitive import/export dependencies
before it starts.

Imports, uses, and exports appear before other declarations at a module root and in
every named scope region. A region is one declaration for its enclosing root's
ordering; its own headers are ordered within the region.

Import cycles are valid. Functions and nominal types may refer to public
declarations across an import cycle.

## REPL

REPL imports and `use` declarations persist after a successful entry, retained
as written: a retained wildcard expands again on every later entry, so it picks
up modules added since. A later successful entry replaces the earlier import
declaration for every module it names at the same scope path and the earlier
`use` declaration for the same resolved target at that path. A failed entry
changes neither imports nor uses, and `:reset` clears both with the session
bindings. An
explicit `import std/core` retained from a successful entry
suppresses the synthetic prelude in later entries; `--no-stdlib` disables it
for the whole session.

## Diagnostics

Imports report a missing or ambiguous module path, a selected or hidden name
the module does not declare, or a header placed after a non-header item. A
`use` target must name a local or already imported scope. A qualified use
reports an unknown qualifier, a hidden or absent member, or an ambiguous route.
A bare use reports an ambiguous name at its use site. These diagnostics identify
a direct repair: import the required module, use a longer suffix or an anchored
path, add an alias, or adjust a tail or hiding clause.
