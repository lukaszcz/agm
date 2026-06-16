# Strings and Interpolation

[← Index](index.md)

Every string literal in AgL is a **template**: a sequence of literal text
fragments and `${…}` interpolation holes. A template evaluates to `text`.
The lexical forms — single- and triple-quoted strings, escapes, and the
triple-quoted dedent rule — are specified in
[Lexical structure](lexical-structure.md). This chapter specifies what
interpolation *means*.

## Interpolation

```ebnf
interpolation ::= "${" expr "}"
```

The expression may be anything — a variable, field access, arithmetic, an
agent call, a parenthesized `case` expression. Its value is converted to
text using a single **uniform rendering rule**, the same regardless of
whether the template appears in a `prompt` argument, a `print` operand,
an `exec` command, or any other position.

## Uniform rendering rules

| Value type | Rendered as |
| ---------- | ----------- |
| `text` | verbatim text |
| `int`, `decimal`, `bool` | plain scalar text |
| `json` | pretty JSON (2-space indent) |
| `list`, `dict` | pretty JSON (2-space indent) |
| records | pretty JSON of the fields (2-space indent) |
| enums | pretty JSON with the `"$case"` tag (2-space indent) |
| exceptions | pretty JSON of the diagnostic fields (2-space indent) |

Scalar text conventions:

- `bool` renders as `true` / `false`.
- `decimal` renders in plain fixed-point notation — never scientific
  notation — with trailing zeros dropped (`1.50` → `1.5`, `1E+2` → `100`).
  Decimals in JSON output are emitted as exact unquoted numbers, never via
  binary floats.
- Enum JSON uses the `"$case"` variant tag; exception JSON is the flat
  object of the exception's fields.

No boundary tags or other wrappers are added around interpolated values.

## Templates in `exec` commands

`exec` shell templates use the same uniform rendering. Interpolated values
are inserted **verbatim** into the command string — there is **no automatic
shell quoting**. The workflow author is responsible for writing shell-safe
commands. See [Shell execution](shell-execution.md) for details.

## Errors

- Newline inside `${…}` — lexical error.
- Unterminated string, unterminated interpolation, unknown escape — lexical
  errors.
