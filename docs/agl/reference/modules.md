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
                ("as" ref_name | "::" tail)? [hiding_clause]

tail             ::= "*" | import_item | "{" import_item ("," import_item)* ","? "}"
import_item      ::= path_atom ["as" ref_name]
hiding_clause    ::= "hiding" path_atom ("," path_atom)*
path_atom        ::= (NAME "::")* name
name             ::= NAME | OP_NAME
```

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
member beneath `Alias`. `hiding` is valid only with a `::*` tail. A `use` declaration contributes names
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
or an already-bare imported scope; ambiguity is a static error.

<!-- agl-check: fragment -->
```agl
import geo/shapes
use geo/shapes::Point::{distance as point-distance}
use /geo/shapes::Point::* hiding internal-distance
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
member resolves when exactly one matching route contributes it. A leading `/`
anchors a qualifier to the complete plain module path. Anchored qualifiers never
match aliases. Aliases are single-segment routes only.

`::name` refers to a declaration in the current module root and bypasses a
lexical shadow. The same form works for `::Type` and `::Type::Variant`.
Qualified type references follow the same routing rules and preserve
module-and-scope nominal identity.

## Re-exports and visibility

`def`, `record`, `enum`, `exception`, and `type` declarations are exported
under their full declaration paths. Grouping helpers in a
[named scope](scopes.md) keeps them off a module's bare surface: an importer
reaches such a member through its scope path or makes it bare with an import
tail or `use` declaration.

`export` forwards public declarations without injecting them into the exporting
module's local scope. A brace tail selects the declarations to forward; a plain
export forwards the complete public surface, and `hiding` removes paths from
that surface. Renames in a brace tail change the forwarded path. A method
travels with its receiver type: any module with a value of that type can call
the method without importing the module that declared it.

<!-- agl-check: fragment -->
```agl
export math/basic::{add, multiply as mul}
export math/advanced hiding internal-helper
export math/*
```

Re-exports preserve the original defining-module identity. Conflicting exposed
names with different origins are static errors; duplicate paths to the same
origin are allowed.

## Prelude

Every loaded entry and library module, except `std/core` itself, receives an
implicit `import std/core::*`. Any explicit `import std/core` declaration in
the module, including one inside a named scope region, supplies the core
contribution instead. Thus `import std/core` makes the core public surface
qualified-only, while `import std/core::* hiding ask` makes every core member
except `ask` bare.

```agl
import std/core::* hiding ask

program def main() -> unit =
  let _ = print("ready")
```

The `--no-stdlib` option disables the implicit declaration. An explicit core
import remains available with that option.

## Standard library modules

- `std/core` declares the core types, exceptions, and built-ins.
- `std/config` exposes the host engine settings as `builtin var` bindings.
- `std/text` exposes `interp(template, vars) -> text` for name-only runtime
  interpolation; see [Strings and interpolation](strings-and-interpolation.md#runtime-interpolation).
- `std/fs` exposes explicit text filesystem operations: `read`, `write`,
  `append`, `exists`, and `list`; see [`std/fs`](expressions.md#stdfs).

## Library modules and cycles

Every file-backed module has a static root: imports, declarations, parameters,
and `let`/`var` bindings are allowed there, while bare expressions and
assignments are not. Root binding initializers must be constant expressions:
literals, literal containers, constructor applications, and unary operators
over those. Put executable workflow code in a `program def` body. Parameters
are also legal in named scope regions in every module. A program receives
values for the params in its module and transitive import/export dependencies
before it starts.

Imports and exports appear before other declarations at a module root and in
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
