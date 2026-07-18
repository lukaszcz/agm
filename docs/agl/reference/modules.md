# Modules

[← Index](index.md)

AgL programs are composed from file-based modules. Imports make selected public
module members available through qualified routes and, only when requested,
bare names.

## Module identity and roots

A module identity is its slash path: the relative path to its `.agl` file,
without the suffix. For example, `utils/strings.agl` has identity
`utils/strings`. The entry program has no path identity.

A module must resolve to exactly one file across the configured library roots.
No matching file is an error; more than one matching file is also an error.
There is no root-priority shadowing. Wildcard imports select matching modules
from the same global module set.

The standard library includes `std/core`. Batch programs receive its normal
prelude; explicit imports of `std/core` use the rules below.

## Imports

```ebnf
import_decl ::= "open"? "import" module_path ("/" "*")?
                ("as" ref_name)?
                (using_clause | hiding_clause)?

module_path ::= NAME ("/" NAME)*
using_clause ::= "using" import_item ("," import_item)*
hiding_clause ::= "hiding" ref_name ("," ref_name)*
import_item ::= ref_name ("as" ref_name)?
```

Examples:

```agl
import utils/strings
open import utils/strings
import utils/strings using trim as clean, split
import utils/* as util
```

Each import contributes a selected set **S** of its target module's public
members:

- no clause selects every public member;
- `using` selects exactly its listed members;
- `hiding` selects every public member except its listed members.

`using` and `hiding` name top-level declarations. Enum variants travel with
their enum; selecting `Color` makes `Color::Red` available, while selecting
`Red` alone is invalid.

A plain import contributes **S** only to qualified routes. It does not inject
bare names. `open import` injects all of **S** into the bare namespace.
`using` also injects its selected members as bare names, so `open import ...
using ...` is redundant and invalid. A bare name contributed by several
imports is an error when used, not when imported.

A `using N as M` rename is canonical: `M` is the member name in both bare and
qualified access, and `N` is not available through that import.

`as A` supplies an alias route instead of the module's path route. It does not
make names bare by itself. Repeated imports of one module union their selected
sets and their bare-name contributions. Distinct modules may share an alias;
the requested member resolves the resulting alias facade when it identifies a
single module.

A wildcard import distributes the same selection independently to every
matched module. An alias on a wildcard is one shared alias facade rather than
a path rewrite.

## Qualified access

A qualifier is followed by `::`:

```agl
import company/tools/config
import service/config as settings

company/tools/config::timeout
config::timeout                 # suffix route
settings::timeout               # alias route
/company/tools/config::timeout  # anchored plain-path route
```

Plain import routes support suffix matching. A suffix can match more than one
imported module; the requested member filters those routes first. If one
remaining module contributes the member, it is selected; if several do, the
reference is ambiguous. If a route matches but none contributes the member,
the member is inaccessible through that qualifier.

A leading `/` anchors a qualifier to the complete plain module path. Anchored
qualifiers never match aliases and are always interpreted as module routes;
they cannot name a local type. Aliases are single-segment routes only and do
not participate in suffix or anchored matching.

Qualified access is bounded by **S**. For example:

```agl
import calc using add

calc::add(1, 2)   # valid
calc::mul(2, 3)   # static error
```

`::name` refers to a declaration in the current module root and bypasses a
lexical shadow. The same form works for `::Type` and `::Type::Variant`.

Qualified type references use the same routes and preserve nominal identity:

```agl
import shapes/points as points

let p: points::Point = points::Point(x = 0, y = 0)
```

Two same-named types from different modules are distinct. Type-qualified enum
constructors use `module::Type::Variant`; a short `Type::Variant` may name a
local or open type. A module route competes with that short spelling only when
it contributes the referenced variant member.

## Exports and visibility

Public top-level `def`, `record`, `enum`, `exception`, and `type` declarations
are exported by default. Prefix a declaration with `private` to keep it within
its defining module.

`export` re-exports public members without injecting them into the exporting
module's local scope:

```agl
export math/basic using add, multiply as mul
export math/advanced hiding internal_helper
export math/*
```

Re-exports preserve the original defining module identity. Conflicting exposed
names with different origins are static errors; duplicate paths to the same
origin are allowed.

## Library modules and cycles

Imported modules are declaration-only: they may contain imports, exports,
functions, type declarations, and infix declarations, but not executable
top-level expressions, bindings, agents, parameters, or program declarations.
Imports and exports appear before other declarations in a library module.

Import cycles are valid. Public declarations are collected before bodies are
resolved, so functions and nominal types may refer across cycles.

## Interactive sessions

REPL imports persist after a successful entry. A later import of the same
module or wildcard prefix replaces the earlier declaration, so a new selection,
open mode, or alias takes effect for following entries. A failed entry does not
change the accumulated imports. `:reset` clears them along with session
bindings. REPL entries otherwise retain entry-module behavior: declarations,
bindings, expressions, and imports may be interleaved.
