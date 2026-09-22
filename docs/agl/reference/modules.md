# Modules

[← Index](index.md)

AgL programs are composed from file-based modules. An `import` makes another
module's public declarations available through qualified routes. An import tail
or a `use` declaration adds selected declarations to a bare namespace.

## Slash-path identity

A module identity is its slash path: the relative path to its `.agl` file,
without the suffix. For example, `utils/strings.agl` has identity
`utils/strings`. Each segment is an ordinary AgL name, so a kebab-case path such
as `review-tools/main` is a module identity like any other. An entry program that
no [package](packages.md) owns has no path identity; a file inside a package keeps
its package-qualified identity even when it is the entry.

A slash path is written byte-adjacent wherever it appears — in a header, a
qualifier, or a wildcard tail. `a/b` is a path; `a / b`, spaced on both sides,
is division. A `/` touching an operand on exactly one side (`a/ b`) is neither
and is rejected; see
[Qualifier chains](lexical-structure.md#qualifier-chains).

A module must resolve to exactly one file across the configured library roots.
No matching file is an error; more than one matching file is also an error.
There is no root-priority shadowing. Wildcard imports select matching modules
from the same global module set.

A module inside a [package](packages.md) has the package name as the first
segment of its path and may import only its own package, the package's
declared dependencies, and the standard library.

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

Selecting an enum scope includes its inline member-record subtree. A referenced
record member keeps its own declaration and export path; naming it in an enum
does not add that record to the enum's public scope.

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
  print(add(1, 2))
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
  def norm(p: Point) -> decimal = magnitude(p)
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

`def`, `record`, `enum`, `exception`, and `type` declarations, plus simple
`let`/`var` bindings, are exported under their full declaration paths. An
unannotated binding's exported type is the one inferred from its initializer,
exactly as an importer would see for an annotated binding's declared type —
except an unannotated `var` whose inferred type is too narrow to reassign, which
is rejected instead (see [Typing of
bindings](bindings-and-scope.md#typing-of-bindings)). Grouping helpers in a
[named scope](scopes.md) keeps them off a module's bare surface: an importer
reaches such a member through its scope path or makes it bare with an import
tail or `use` declaration.

An exported `var` is writable across a module boundary the same way it is
read: a qualified target, or a bare target reached through an import tail or
`use`. An exported `let` remains immutable at every reference site, including
one reached through a re-export.

`export` forwards public declarations without injecting them into the exporting
module's local scope. A brace tail selects the declarations to forward; a plain
export forwards the complete public surface, and `hiding` removes paths from
that surface. Renames in a brace tail change the forwarded path. Methods are
ordinary declaration paths for these operations: a facade may write
`export metrics::{Point::norm}`, and an importer can call that re-export by its
qualified route, such as `facade::Point::norm(point)`. Separately, the import
makes the declaration route visible for dot selection, so `point.norm()` is
available. Selection, renaming, and `hiding` therefore control both direct-call
routes and dot-method visibility.

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

Every loaded entry and library module, except `std/prelude` itself, receives an
implicit `import std/prelude::*`. Any explicit import declaration whose expansion
includes `std/prelude`, including one inside a named scope region, supplies the prelude
contribution instead. Thus `import std/prelude` and `import std/*` leave prelude
names qualified-only, while `import std/prelude::* hiding ask` makes every prelude
member except `ask` bare.

```agl
import std/prelude::* hiding ask

program def main() -> unit =
  print("ready")
```

The prelude re-exports the receiver scopes of `std/array`, `std/dict`,
`std/text`, `std/json`, and `std/math`, making their exported methods available
wherever the prelude is enabled. That visibility follows the ordinary import
route: `import std/prelude::* hiding text::trim` hides `trim` while retaining
other prelude methods. The `--no-stdlib` option disables the implicit prelude;
an explicit route to a module or facade exporting a method makes it visible.
An explicit prelude import remains available with `--no-stdlib`.

## Standard library modules

The standard library is an ordinary module tree mounted under `std/`: its
modules are imported, aliased, re-exported, and hidden from exactly like any
other module. Two have a language-level role:

- `std/prelude` is the prelude described above. It declares nothing itself: it
  re-exports the modules declaring the types, exceptions, and built-ins the
  language itself refers to, together with the generic sum and product types,
  the receiver scopes that make builtin methods available, `std/path`'s
  `path` type, `std/url`'s `url` type, and `std/env`'s `getenv`, which
  environment holes read.
- `std/config` exposes the host engine settings as `builtin var` bindings; see
  [Host environment](host-environment.md).

Every other `std/*` module carries no special status; the prelude re-exports
the first six rows below in full, the receiver scopes from the following two
rows, `std/path`'s `path` type alone, `std/url`'s `url` type alone, and
`std/env`'s `getenv` alone. The rest are imported explicitly:

| Module | Provides |
| ------ | -------- |
| `std/errors` | the built-in exception hierarchy |
| `std/fun` | the function application and composition combinators |
| `std/io`, `std/value` | printing; rendering, copying, and parsing values |
| `std/exec`, `std/agent`, `std/session` | shell execution, agent calls, and agent sessions |
| `std/package` | package resource lookup |
| `std/option`, `std/optional`, `std/pair`, `std/either`, `std/result` | `Option[T]`, `Optional[T]`, `Pair[A, B]`, `Either[A, B]`, and `Result[T, E]` |
| `std/array`, `std/dict`, `std/text`, `std/json` | builtin receiver scopes and free functions; the prelude re-exports the scopes |
| `std/math` | builtin receiver scopes for `int`, `decimal`, and `bool`, aggregates, and constants |
| `std/toml` | conversion between TOML documents and `json` |
| `std/regex` | Python-compatible searching, rewriting, and splitting |
| `std/time` | UTC clock, parsing, formatting, and sleeping |
| `std/random` | a seedable pseudo-random sequence and UUIDs |
| `std/path` | the `path` type and lexical path manipulation |
| `std/url` | the `url` type, parsing, rendering, joining, and percent-/query-encoding |
| `std/http` | HTTP requests, headers, responses, and the http-timeout setting |
| `std/fs` | UTF-8 filesystem and directory operations |
| `std/env` | the ambient environment snapshot and its helpers |
| `std/process` | process metadata and termination |

A few conventions run through all of them. Every name the library exposes —
functions, fields, and named arguments alike — is spelled in kebab-case, with
types and constructors in `CamelCase`. A value naming a filesystem location
is typed `path` — the builtin `text` alias ([Type aliases](types.md#type-aliases))
`std/path` declares and the prelude forwards — so a signature says which of its strings are locations
without making them a separate type. A value naming a URL is typed `url` the
same way — a plain `text` alias `std/url` declares and the prelude forwards,
but a plain alias rather than a `builtin type`: no program parameter completes
it as a location the way a `path` parameter does. Each module declares the exception
types its own operations raise, so an error type lives beside the operations
that produce it; the exceptions the language itself raises live in
`std/errors`. An operation that raises may have a `?` twin returning `Option`,
which drops the failure, and a `try-` twin returning `Result[T, E]` with `E` the
specific exception, which keeps it. Both suffixes name twins only: an operation
whose natural result is an `Option` and which has no raising form carries no
`?`. A `!` suffix marks an in-place operation — usually the counterpart of a
copy-producing one, sometimes an inherently mutating operation with no
copy-producing form. An operation with neither suffix may still mutate when that
is its purpose, as `push` and `set` do. An `-or` suffix supplies a fallback
eagerly, as a value; its `-else` twin supplies the same fallback lazily, as a
function evaluated only on the failing branch — `Option` and `Result` carry both
pairs, `or`/`or-else` and `unwrap-or`/`unwrap-or-else`.

Each module's own source is its reference.

## Library modules and cycles

Every file-backed module has a static root: imports, declarations, and
`let`/`var` bindings are allowed there, while bare expressions and
assignments are not. Root binding initializers must be [constant
expressions](bindings-and-scope.md#constant-expressions). Put executable
workflow code in a `program def` body. Because every initializer is a
constant, declaration order does not matter for a static root's own `let`/`var`
bindings, root or scoped — see [Names and
visibility](scopes.md#names-and-visibility).

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
explicit `import std/prelude` retained from a successful entry
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
