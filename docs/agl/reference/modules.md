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

## Importing modules

```ebnf
import_decl ::= ["qualified"] "import" module_path [".*"]
                    ["as" NAME]
                    [("using" | "hiding") name_list]

module_path ::= NAME ("." NAME)*
name_list   ::= name_item ("," name_item)*
name_item   ::= NAME ["as" NAME]
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

Import declarations must appear **before any other declaration or expression**
in the module's body. A non-entry module that places an `import` after a `def`
(or any other item) is a static error.

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
so `foo.bar::x` becomes `A::x` and `foo.bar.baz::y` becomes `A.baz::y`.  The
original handles `foo.bar` and `foo.bar.baz` are no longer registered.

### Merging imports

Multiple import declarations for the same module are allowed and merge:

```agl
import utils.strings using trim
import utils.strings qualified as str
```

The unqualified set is the union of all open contributions; the qualified alias
is registered from the `as` form.

## Qualified access: `::`

A name may be accessed with an explicit module qualifier, regardless of what
was imported:

```agl
import math
let r = math::sqrt(2.0)
```

The qualifier is the **handle** under which the module was imported — either
the full dot-path (the default, e.g. `math` for `import math` or `foo.bar` for
`import foo.bar`) or its `as` alias. Qualified access to a **private** name is
a static error even if the name is otherwise known.

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

Private names are never included in import environments. Attempting qualified
access to a private name is a static error with a clear message.

## Library modules (non-entry)

A module that is imported (not the entry) is a **library module**. Library
modules may only contain declarations:

- `def` — function definitions
- `record`, `enum`, `type` — type declarations
- `import` — import declarations (header only, before all other items)

Statements, binders (`let`/`var`), expressions, agent declarations, `param`
declarations, `program` declarations, and config pragmas are all **static
errors** in a library module.

## Cross-module mutual recursion

Functions in different modules may call each other recursively. The scope pass
collects all public function declarations across the full import graph before
resolving any function body, so forward references between modules work without
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

When an AgL program is executed, the runtime resolves module imports by searching
an unordered set of **search roots** — directories that are scanned for `.agl`
files matching the module's dot-path. A module whose dot-path resolves to exactly
one file across all roots is found; zero matching files is a module-not-found
error; two or more distinct files for the same id is an ambiguity error (no
silent shadowing, no priority ordering).
