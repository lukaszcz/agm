# Standard library style

These rules apply to AgL sources under `src/`. The library favors the shortest
clear spelling: syntax should show only information needed to resolve a name or
bind an argument.

## Names

- Module and file names use lower kebab-case.
- Functions, methods, fields, parameters, and local bindings use lower
  kebab-case. A method receiver is named `self`.
- Types, records, enums, exceptions, and their constructors use UpperCamelCase.
- Type parameters use short UpperCamelCase names, normally one letter (`T`,
  `E`, `A`).
- `?` marks an `Option`-returning fallible twin; `!` marks in-place mutation.
  Use `try-` for a `Result`-returning twin, `-or` for an eager fallback, and
  `-else` for a lazy fallback.

## Qualification

- Use a bare constructor or function when its visible name resolves to one
  declaration, including constructors imported from another module. For
  example, use `None` rather than `Option::None` when no other visible `None`
  conflicts.
- When qualification is required, use the shortest unambiguous route that is
  valid for every supported stdlib arrangement. Prefer `Option::None` to a
  longer module path. Retain a package-qualified route such as
  `std/config::timeout` when an alternate stdlib layout does not expose the
  shorter suffix.
- Qualify a name only when another visible declaration has the same spelling,
  when a qualification is required by declaration syntax, or when an explicit
  type application is needed. Keep the owner on method declarations
  (`Option::map`).
- Apply the same rule in patterns and expressions. Do not qualify a local
  constructor merely to document its owner; use qualification to resolve an
  actual ambiguity.

## Arguments

- Pass constructor and function arguments positionally by default, in
  declaration order.
- Use a named argument only for a named-only parameter, to skip earlier
  optional parameters, or to resolve a real argument ambiguity. In particular,
  avoid redundant forms such as `Some(value = value)`; write `Some(value)`.
- Keep named arguments when the declaration explicitly requires them, such as
  an exception field marked `@arg-named`.

## Companion Python

Python companions follow Python naming and formatting conventions. Their
generated AgL nominal classes accept fields as keyword arguments, so a
companion call such as `agl.option_some(value)`/`agl.option_none()` is an FFI
boundary requirement, not an AgL style violation.
