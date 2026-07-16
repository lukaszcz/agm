# Modules

[← Index](index.md)

AgL supports a **file-based module system** that lets programs be split across
multiple `.agl` files. Every file is a module; modules are identified by their
path relative to the library search roots.

## Module identity

A module's identity is its **dot-path** — the relative file path with `/`
replaced by `.` and the `.agl` suffix stripped. For example, a file at
`utils/strings.agl` under a library root has the dot-path `utils.strings`.

The **entry module** is the program being run — it has no dot-path of its own
and is always the root of the import graph.

## Standard library

The core standard library module is `std.core`. It defines common names such as
`Option[T]`, `ExecResult`, `ParsePolicy`, `AgentRequest`, built-in exception
types, and the host-implemented built-in functions.

For batch execution, `std.core` is opened unqualified in the entry module by
default. This is equivalent to an implicit leading:

```agl
import std.core
```

Use `agm exec --no-stdlib` to disable that implicit import. Explicit
`import std.core` declarations still work and follow the same duplicate and
ambiguity rules as any other import. If another open import also exports
`Option`, for example, an unqualified use of `Option` is ambiguous until one of
the references is qualified.

## Importing modules

```ebnf
import_decl ::= "import" module_path [".*"]
                    ["qualified"]
                    ["as" NAME]
                    [import_clause]

import_clause ::= "using" import_item ("," import_item)*
                | "hiding" NAME ("," NAME)*

import_item ::= NAME ["as" NAME]
module_path ::= NAME ("." NAME)*
```

- `qualified` suppresses unqualified injection (all names require a qualifier).
- `as NAME` replaces the qualifier handle: `import foo.bar as B` makes `B::x`
  valid instead of `foo.bar::x`.  It does NOT suppress unqualified injection on
  its own — combine with `qualified` to suppress unqualified access.
- `using N1, N2` restricts the imported set to the listed names.
- `using N as M` imports `N` under the canonical name `M`; only `M` is
  accessible (neither unqualified `N` nor qualified `::N` work; use `M` or
  `qualifier::M`).
- `hiding N1, N2` imports all public names except the listed ones.
- The combinations `qualified using`, `qualified hiding`, and `qualified as A
  using …` are all valid.

Import and export declarations must appear **before any other declaration or
expression** in the module's body. A non-entry module that places an `import` or
`export` after a `def` (or any other item) is a static error.

### Open import

```agl
import utils.strings
```

Brings all public names from `utils.strings` into scope unqualified. Bare names
accessible via multiple open imports are a **clash**: using the clashing name
is a static error, but importing the modules is not. Names that are never
referenced are fine even if they clash.

### Selective import: `using` and `hiding`

```agl
import utils.strings using trim, split
import utils.strings hiding internal_helper
```

`using N1, N2` restricts the open import to only the listed names. `hiding N1,
N2` opens all public names except the listed ones. Both forms still register
the module handle for qualified access.

The names in `using`/`hiding` identify **top-level declarations** — `def`,
`record`, `enum`, and `type` aliases. **Enum variants travel with their enum**
and are not separately listable in `using`/`hiding`. To import an enum type
named `Color` (and thereby gain access to `Color::Red` etc.), write `using
Color`. Writing `using Red` — naming only the variant constructor — is a static
error ("Red is not exported by module …").

#### Name renames in `using`

```agl
import utils.strings using trim as clean, split
```

`using N as M` makes `M` the canonical exposed name everywhere — both
unqualified (`clean`) and qualified (`utils.strings::clean`).  The original
name `trim` becomes inaccessible after the rename; only `M` is valid.

### Aliased import: `as`

```agl
import utils.strings as str
```

Replaces the module's qualifier handle with the alias `str`, so qualified access
uses `str::name` instead of `utils.strings::name`.  Bare names are still
brought into scope unqualified unless `qualified` is also specified.

### Qualified import

```agl
import utils.strings qualified
import utils.strings qualified as str
import utils.strings qualified as str using trim
```

`qualified` prevents any unqualified injection: names from the module are only
accessible via the qualifier handle.  `qualified as A` is the most common form
when you want a short alias with no unqualified pollution.

### Wildcard import

```agl
import utils.*
import utils.* as U
```

Imports all modules whose dot-path starts with `utils.` — a glob over the
library roots. The same modifiers (`qualified`, `as`, `using`, `hiding`) may be
combined with `.*`.

**Wildcard alias re-rooting** (`as A`): the matched prefix is replaced by the
alias in every qualifier handle.  For example, given modules `foo.bar` and
`foo.bar.baz`:

```agl
import foo.bar.* as A
```

produces qualifier handles `A` (for `foo.bar`) and `A.baz` (for `foo.bar.baz`),
so `foo.bar::x` becomes `A::x` and `foo.bar.baz::y` becomes `A.baz::y`. The
handles `foo.bar` and `foo.bar.baz` are not registered for that import.

## Re-exporting

A module may **re-export** names from another module, making them visible to
consumers of the module as if they were defined locally. `export` is
declaration-only: it loads the target module and contributes to the current
module's public export set, but it does not inject names into the current
module's local scope.

```ebnf
export_decl ::= "export" module_path [".*"] [export_clause]

export_clause ::= "using" export_item ("," export_item)*
                | "hiding" NAME ("," NAME)*

export_item ::= NAME ["as" NAME]
```

Re-exports are transparent: a name re-exported through a chain of modules
always carries its original defining module as its identity. A consumer that
imports the facade sees re-exported names both unqualified and through the
facade qualifier.

Plain `export` re-exports all public names from the target module:

```agl
export math.ops
```

Use `using` to re-export only selected names:

```agl
export math.ops using add, mul
```

`using` may rename the exposed name:

```agl
export math.ops using add as plus
```

Use `hiding` to re-export all public names except the listed names:

```agl
export math.ops hiding _impl
```

Wildcard export re-exports public names from every module in a subtree:

```agl
export math.*
```

Private names are never re-exported. **Name conflict**: if two `export`
declarations would expose the same name with
different origins, it is a static error.  Diamond re-exports (the same name
re-exported via two paths from the same original definition) are allowed and
collapse silently.

### Merging imports

Multiple import declarations for the same module are allowed and merge:

```agl
import utils.strings using trim
import utils.strings qualified as str
```

The unqualified set is the union of all open contributions; the qualified alias
is registered from the `as` form.

## Qualified access: `::`

A name may be accessed with an explicit module qualifier:

```agl
import math
let r = math::sqrt(2.0)
```

The qualifier is the **handle** under which the module was imported — either
the full dot-path (the default, e.g. `math` for `import math` or `foo.bar` for
`import foo.bar`) or its `as` alias.

**Qualified access is bounded by the imported set S.** If you restricted the
import with `using` or `hiding`, the qualifier can only reach the names in S.
A `using x, y` import means `module::z` is a static error for any name `z`
not listed in `using`:

```agl
import calc using add          # S = {add}
let r = calc::add(1, 2)        # OK — add is in S
let s = calc::mul(2, 3)        # STATIC ERROR — mul is not in S
```

Qualified access to a **private** name is a static error even if the name is
otherwise known.

### Self-reference: `::name`

The `::` prefix with no left-hand qualifier refers to the **current module's
own top-level declaration** — it resolves directly in the module root and
bypasses any lexical shadow (e.g. a function parameter or local `let` that
happens to share the name):

```agl
def helper() -> int = 1
def public() -> int = ::helper()   # unambiguously calls this module's helper

def g(helper: int) -> int = ::helper()  # calls the top-level def, not the param
```

This is useful when a top-level name might otherwise be shadowed by an open
import or by a local binding inside a function body.

## Public and private names

A declaration is **public** by default; prefixing it with `private` makes it
visible only within the defining module:

```agl
private def internal_helper(x: int) -> int = x * 2
def public_api(x: int) -> int = internal_helper(x)
```

`private` is a declaration modifier that behaves like a decorator: it may
precede the declaration on the same line or on the line directly above it (the
newline after the modifier is insignificant).

Private names are never included in import environments. Attempting qualified
access to a private name is a static error with a clear message.

## Library modules (non-entry)

A module that is imported (not the entry) is a **library module**. Library
modules may only contain declarations:

- `def` (including `extern def`; see [Python FFI](ffi.md)) — function
  definitions
- `record`, `enum`, `type` — type declarations
- `import` — import declarations (header only, before all other items)

Statements, binders (`let`/`var`), expressions, agent declarations, `param`
declarations, and `program` declarations are all **static errors** in a library
module.

## Cross-module mutual recursion

Cyclic imports between modules are explicitly allowed: two modules may import
each other. Because modules are declaration-only (they contain no executable
top-level code), importing a module never executes it, so there is no
initialization-order issue.

Functions in different modules may call each other recursively. All public
function declarations across the full import graph are collected before any
function body is resolved, so forward references between modules work without
ordering constraints.

## Example

```
# File: math/arith.agl
def square(x: int) -> int = x * x
private def internal() -> int = 0

# File: main.agl (entry)
import math.arith
let s = square(5)
print s
```

```
# Using qualified access and aliasing:
import math.arith as arith
let s = arith::square(3)

# Selective import:
import math.arith using square
let s = square(4)
```

## Module search roots

A module's dot-path must resolve to **exactly one** module definition. If no
module with that dot-path exists, a module-not-found error is raised. If two or
more distinct definitions exist for the same dot-path, an ambiguity error is
raised — there is no priority ordering and no silent shadowing.

## Imports in an interactive session

In an interactive session, `import` declarations are supported. An `import`
declaration may be entered at any time — it is not restricted to a module header.
Imported modules are loaded on first use and cached for the rest of the session.

```agl
# First entry
import utils.strings
let result = trim("  hello  ")   # works immediately

# Second entry — prior import is still in scope
let result2 = trim("  world  ")  # still works
```

**Open imports persist across entries.** When an entry imports a module with an
open import (`import foo`), the names it brings into scope remain available in
subsequent entries. Entering a new `import foo` in a later entry replaces the
earlier one (for example to change `using`/`hiding`/`as` options).

**Qualified imports across entries** also persist: `import foo qualified as F` in
entry 1 makes `F::name` available in entry 2 and later.

**`::name` self-reference** in the REPL refers to any name declared in the
accumulated REPL session — including names declared in prior entries. This lets
you explicitly bypass an open-imported name that shadows an earlier session
binding:

```agl
# Suppose an imported module exports 'helper'
import util   # exposes 'helper' unqualified

def helper() -> int = 42   # declares a session binding named 'helper'

::helper()   # calls the session's own 'helper', not the imported one
```

**Declaration-only restriction does not apply in the REPL.** Unlike imported
library modules, the REPL session behaves like the entry module: bare
expressions, `let`/`var`, `print`, `exec`, and `ask` calls are all valid at
any point, interleaved freely with `import` declarations and function/type
definitions.
