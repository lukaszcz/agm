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
[Module qualifiers](lexical-structure.md#module-qualifiers).

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
using_clause ::= "using" import_item ("," import_item)*
hiding_clause ::= "hiding" ref_name ("," ref_name)*
import_item ::= ref_name ["as" ref_name]
```

Each import contributes a selected set **S** of its target module's public
members:

- no clause selects every public member;
- `using` selects exactly its listed members;
- `hiding` selects every public member except its listed members.

Both bare access and qualified access are bounded by **S**. Repeated imports
of the same module union their selected sets and bare-name contributions.
This makes a small bare surface plus a full qualified API explicit:

```agl
import utils/strings
import utils/strings using trim
```

`using` and `hiding` name top-level declarations. Enum variants travel with
their enum: selecting `Color` makes `Color::Red` available, while selecting
`Red` alone is invalid. A bare name contributed by several imports is an error
when used, not when imported.

## `open`, `using`, and `hiding`

A plain import contributes **S** only to qualified routes. `using` injects its
selected names into the bare namespace. `open import` injects all of **S**
bare; combining `open` and `using` is redundant and invalid.

```agl
import utils/strings
open import app/vocabulary hiding internal-word
import text/format using render as format
```

A `using N as M` rename is canonical: `M` is the member name for both bare and
qualified access through that import, and `N` is inaccessible through it.
`hiding` removes a member from both channels, so it can also remove a
qualification ambiguity.

## Aliases

`as A` gives an import the single-name alias `A` instead of a path route. It
does not make names bare. An aliased import is reached only through its alias;
it does not participate in suffix or anchored path matching.

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
or starts with `prefix/`. The import's
`open`, selection clause, and alias apply independently to every matched
module. A `using` or `hiding` name must be public in every matched module.

```agl
import tools/*
open import domain/* hiding debug
import codecs/* as codec
```

An alias on a wildcard is a shared alias facade, not a path rewrite.

## Suffix and anchored references

A module qualifier ends in `::`:

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

Qualified type references follow the same rules and preserve nominal identity:

```agl
import shapes/points as points

let p: points::Point = points::Point(x = 0, y = 0)
```

`::name` refers to a declaration in the current module root and bypasses a
lexical shadow. The same form works for `::Type` and `::Type::Variant`.
Type-qualified constructors use `Type::Variant`; a short spelling can name an
in-scope type or a module route and is resolved at the use site.

## Re-exports and visibility

Public top-level `def`, `record`, `enum`, `exception`, and `type` declarations
are exported by default. Prefix a declaration with `private` to keep it within
its defining module.

`export` re-exports public members without injecting them into the exporting
module's local scope:

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
Imports and exports appear before other declarations in a library module.

Import cycles are valid. Public declarations are collected before bodies are
resolved, so functions and nominal types may refer across cycles.

## REPL

REPL imports persist after a successful entry. Wildcards expand before they
are retained. A later entry replaces the earlier import declaration for every
module it names, so its selection, open mode, or alias takes effect for that
module. Multiple declarations for one module in the same entry merge normally.
A failed entry changes no imports, and `:reset` clears imports with the session
bindings. Each REPL entry and its loaded library modules receive the
`std/core` prelude unless the session was launched with `--no-stdlib`.

## Diagnostics

Imports report a missing or ambiguous module path, a selected name that is not
public, redundant `open ... using`, or an import placed after a non-import
item. A qualified use reports an unknown qualifier, a member outside its
contributed set (including a private member), or every candidate of an
ambiguous route. A bare use reports an ambiguous bare name only at its use
site. These diagnostics identify a direct repair: add a longer suffix or an
anchored path, use an alias, adjust `hiding`, or select the required name.
