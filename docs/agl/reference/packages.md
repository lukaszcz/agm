# Packages

[← Index](index.md)

A package is a portable, versioned collection of AgL modules. To the language
it is a module tree with a name: every module the package provides has a
[slash path](modules.md#slash-path-identity) whose first segment is the package
name, and a package module sees only the modules it is entitled to. The
manifest format, dependency resolution, the package store, and the `agm pkg`
commands are host mechanisms described in the
[package command reference](../../commands/pkg.md); this chapter covers what a
package means to AgL source.

## Layout and identity

A package is a directory holding a `package.toml` manifest and a module tree
in its `src/` subdirectory. Everything else in the directory — prompts,
templates, data — is a resource, laid out freely outside the module tree.

```
review-tools/
  package.toml          # manifest: name, version, dependencies, commands
  src/                  # module tree: modules import as review-tools/...
    review.agl
    review.py           # extern companion, beside its module
    lint.agl
  prompts/              # resources, any layout outside the module tree
    review.md
```

The package name must be a single AgL identifier segment and not a reserved
keyword. It is the mandatory first segment of every module the package
provides: `review-tools/src/review.agl` has identity `review-tools/review`,
both inside the package and from any importer. The `src/` tree is mounted
under the package name; the package root itself is not a search root, so a
module file placed outside `src/` does not belong to the package and is not
mounted.

A package under development — a source checkout discovered from an execution
root or from a source file's containing tree — behaves exactly like an
installed one. Its modules keep their package-qualified identity, so a file
inside the package is compiled the same way whether it is executed directly,
imported, or invoked through a registered command.

## Import visibility

Package modules import with the ordinary [import forms](modules.md#imports).
What they may import is restricted:

- modules of the same package, through their package-qualified paths;
- modules of packages the manifest declares as dependencies, direct
  dependencies only — a dependency's own dependencies are not visible unless
  declared too;
- the standard library, which never needs a declaration.

An import outside this set is a static error naming the missing dependency,
even when a module with that path exists on some search root. Loose modules —
`-c` source, REPL entries, and files no package owns — remain unrestricted:
they may import any mounted package module alongside their loose neighbours.
Ownership follows the file's location, not the path spelling and not how the
file was reached: a file inside a package imports under its package's
entitlements whether an import reached it or it is the entry program, and a
loose module never borrows package visibility by sharing a declared
dependency's leading path segment.

<!-- agl-check: fragment -->
```agl
import review-tools/lint          # same package
import judge/verdict              # declared in [dependencies]
import std/fs                     # always visible
```

Ambiguity rules are unchanged: a package module path must resolve to exactly
one file across the mounted roots, and a loose module whose path collides with
a package module is an ordinary ambiguity error.

## Programs and commands

A package module may declare [`program def`](program-structure.md) entry
points like any other module. A program is addressed by its package-qualified
module route and declaration path, `review-tools/review::main`, which is how
the host selects it for execution and how a manifest `[commands]` entry names
it. A command-backed program may declare value parameters and takes no type
parameters; its result is unit like that of any `program def`, whether written
or left to inference. It remains an ordinary callable function inside the
package.

```agl
program def main() -> unit =
  let _ = print("review loop")
```

A registered command runs its program exactly as `agm exec` would: the selected
`program def`'s own value parameters become flags, and configuration is read by
qualified key.

## Program parameters and configuration keys

A package program may declare its own value parameters like any other
[`program def`](program-structure.md). The CLI projects those parameters onto
the program's own flags and positional slots. Because a package-owned entry
keeps its package-qualified module route, its configuration table does too: a
`review::main` program in `review-tools/review` reads its value parameters from
`[review-tools.review.review.main]` (or an unambiguous suffix), while an exact
quoted route such as `["review-tools/review".review.main]` disambiguates. The
suffix rules are those of the
[host environment](host-environment.md#config-file-schema).

A program the manifest registers as a command gains that command path as an
equivalent address: with `"dev review"` registered for
`review-tools/review::main`, `[dev.review]` configures the same program as
`[review-tools.review.review.main]` does, whether it is run as `agm dev
review`, by reference, or by file path.

```toml
[dev.review]
strict = true
```

## Resources and companions

[`resource` and `resource-dir`](expressions.md#resource-and-resource-dir) in a
package-owned module are anchored at the package root, not at the module's
directory, so a resource path is stable wherever the package is mounted and
whoever imports it. The literal path is relative, forward-slash, and cannot
escape the package; it resolves during linking, and a missing target is a
static error. Package validation and archive creation check every literal
resource target, so a resource excluded from an archive fails creation rather
than producing a broken package.

<!-- agl-check: fragment -->
```agl
import std/fs

let prompt = resource("prompts/review.md")

program def main() -> unit =
  let _ = print(fs::read(prompt))
```

An [`extern def`](ffi.md) companion is the `.py` sibling of its module inside
the module tree and travels with the package; there is no separate declaration
for it.

## Diagnostics

An import from a package module that reaches outside its package, its declared
dependencies, and the standard library is a visibility error naming the
undeclared package. A package with no `src/` module tree, a command that names
no valid `program def`, and a literal resource with no target are reported by
package validation before anything runs. All of these are static errors.
