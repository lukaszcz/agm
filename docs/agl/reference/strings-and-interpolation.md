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

The expression may be anything with a **rendering** — a variable, field
access, arithmetic, a call, a parenthesized `case` or `if` expression. Its
value is converted to text using the **uniform rendering rule**, the same
regardless of whether the template appears in an `ask` prompt, a `print`
argument, an `exec` command, or any other position.

## Uniform rendering rules

| Value type | Rendered as |
| ---------- | ----------- |
| `text` | verbatim (no quotes) |
| `int`, `decimal`, `bool` | plain scalar text |
| `json` | pretty JSON (2-space indent, multi-line) |
| `list[E]` | `[e1, e2, …]` — AgL list syntax |
| `dict[text, V]` | `{"k1": v1, "k2": v2}` — AgL dict syntax; keys always quoted |
| record | `TypeName(f1: v1, f2: v2)` — AgL constructor form; fields in declaration order |
| enum | `TypeName.Variant(f1: v1, …)` — qualified; nullary variant as `TypeName.Variant` (no parens) |
| exception | `TypeName(f1: v1, …)` — record-style with all fields including `trace_id`, in declaration order |

AgL structured values (`list`, `dict`, record, enum, exception) always render on
a **single line** — no injected newlines. A `json` value that appears *nested*
inside such a structured value renders compact (single-line) to preserve that
property; a `json` value that is interpolated directly renders pretty (multi-line).

Scalar text conventions:

- `bool` renders as `true` / `false`.
- `decimal` renders in plain fixed-point notation — never scientific
  notation — with trailing zeros dropped (`1.50` → `1.5`, `1E+2` → `100`).
- Nested `text` values (a `text` field inside a record, list element, etc.)
  are emitted as a quoted AgL string literal with full JSON escaping plus
  `\$` for dollar signs, so they cannot be mis-read as interpolation syntax.

No boundary tags or other wrappers are added around interpolated values.

To obtain JSON output use an explicit `as json` cast inside the interpolation:

```agl
let r: R = R(x: 1)
print "${r}"           # → R(x: 1)          (AgL form — the default)

# A json value interpolated directly is top-level, so it renders pretty
# (multi-line, 2-space indent):
print "${r as json}"   # → {
                       #      "x": 1
                       #    }
```

## Types that cannot be interpolated

**Function values** and **agent values** have **no rendering**. Interpolating
either in a template is a **static error** with a targeted diagnostic:

```agl
let f = fn(x: int) => x
print "function is ${f}"   # static error: function value has no rendering
```

Similarly, storing a function or agent in a `json` slot or `print`ing it
directly is a static error.

## Templates in `exec` commands

`exec` shell templates use the same uniform rendering. Interpolated values
are inserted **verbatim** into the command string — there is **no automatic
shell quoting**. The workflow author is responsible for writing shell-safe
commands. See [Shell execution](shell-execution.md) for details.

## Errors

- Newline inside `${…}` — lexical error.
- Unterminated string, unterminated interpolation, unknown escape — lexical
  errors.
- Interpolating a function or agent value — static error.
