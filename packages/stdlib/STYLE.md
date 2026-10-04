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

## Declarations

- Separate extern and builtin function declarations with a blank line before the next
  declaration's attributes.

- Document functions with `@doc` on a separate line before the declaration.
  Start with one concise sentence describing the function. Put exception
  conditions and other notes in separate paragraphs; use a block string for
  multiline documentation.

- Declare a record's fields in the indented block form, one field per line,
  rather than in a parenthesized list:

  ```agl
  record Choice[C]
    choice: C
    confidence: decimal
    probabilities: dict[text, decimal]
  ```

- Declare an exception's new fields the same way, one field per line:

  ```agl
  exception ApiError extends Exception
    status: int
    body: json
  ```

- Omit the parentheses when a record or exception has no (new) fields:

  ```agl
  record Token
  exception AuthError extends ApiError
  ```

- Write an enum's member fields in one form throughout. Use the indented block
  form, one field per line under the member, for every member once any member
  has more than three fields or an attributed field. Otherwise keep the
  parenthesized lists:

  ```agl
  enum FixResult
    | Complete(output: text)
    | Blocked(reason: text, recoverable: bool)

  enum VerifyResult
    | Verified
    | Rejected
        reason: text
    | Completed
        @doc("absolute file path of the next unblocked task")
        next-task-file: path
  ```

## Companion Python

Python companions follow Python naming and formatting conventions, not the AgL style from this file.
