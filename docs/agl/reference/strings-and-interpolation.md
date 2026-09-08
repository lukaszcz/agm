# Strings and Interpolation

[← Index](index.md)

Every string literal in AgL is a **template**: a sequence of literal text
fragments, `%{…}` expression holes, and `${…}` environment holes. A template
evaluates to `text`.
The lexical forms — single- and triple-quoted strings, escapes, and the
triple-quoted dedent rule — are specified in
[Lexical structure](lexical-structure.md). This chapter specifies what
interpolation *means*. AgL text literals have their own full lexical escape
table, including `\n`, `\t`, `\"`, `\%`, and `\${`; the `%{name}` interpolation hole
— whether written directly in a string literal or evaluated by
`std/text::interp` — instead uses only `\%{` to write a literal hole marker.

## Interpolation

```ebnf
interpolation ::= "%{" expr "}"
                | "${" identifier "}"
```

The expression may be anything with a **rendering** — a variable, field
access, arithmetic, a call, a parenthesized `case` or `if` expression. Its
value is converted to text using the **uniform rendering rule**, the same
regardless of whether the template appears in an `ask` prompt, a `print`
argument, an `exec` command, or any other position. A percent sign not
followed by `{` is literal; `\%` produces a literal percent sign.

An environment hole `${NAME}` is shorthand for
`%{std/env::getenv("NAME")}`. It uses the environment snapshot supplied to
the running program, and a missing variable raises the same exception as that
call. `NAME` is an AgL identifier, and the `std/env` module must be available
to the source just as it must be for the explicit form.

Use `\${` to write a literal `${` in a string template.

## Runtime interpolation

A template can also be interpolated at runtime, against a dictionary of names
supplied explicitly, through a standard-library function:

```agl
import std/text

program def main() -> unit =
  let vars = {"name": "Ada"}
  print std/text::interp("Hello, \%{name}!", vars)
```

Runtime holes are **name-only**: `%{name}` names a single AgL identifier and
looks it up in the dictionary. In contrast, a string-literal `%{expr}` hole
is a compile-time template hole containing an arbitrary expression. The two
forms follow identical rules for splicing text into a hole at runtime: the
same `%{...}` delimiters, the same `\%{` escape for a literal hole marker,
and the same error on an unterminated hole.
Use `\%{` in a runtime template for a literal `%{`; because a normal AgL
string is itself a compile-time template, write `\\\%{` in source to pass that
escape to `interp`.

A missing dictionary key, an invalid runtime name, or an unterminated runtime
hole raises a catchable `ExternError` from `std/text::interp`.

## Text indexing and methods

Text indexes address Unicode code points: `"é😀"[0]` is `"é"` and
`"é😀"[-1]` is `"😀"`. An out-of-range index raises `IndexError`; text is
immutable, so indexed assignment is not allowed. Text methods are supplied by
the standard library, ambient when the prelude injects `std/builtin-methods`
and otherwise reached by importing their owning module.

## Uniform rendering rules

| Value type | Rendered as |
| ---------- | ----------- |
| `text` | verbatim (no quotes) |
| `int`, `decimal`, `bool` | plain scalar text |
| `json` | compact JSON by default; use `render(value, pretty = true)` for indented display |
| `array[E]` | `[e1, e2, …]` — AgL array syntax |
| `dict[text, V]` | `{"k1": value1, "k2": value2}` — AgL dict syntax; keys always quoted |
| record | `TypeName(f1 = value1, f2 = value2)` — AgL constructor form; fields in declaration order. Fieldless: bare `TypeName` (no parens) — a nullary constructor is an auto-value, so its bare spelling round-trips as written. |
| enum | inline member: `TypeName::Member(f1 = value1, …)`; fieldless inline member: `TypeName::Member` (no parens). A referenced member retains its record's own display form. |
| exception | `TypeName(f1 = value1, …)` — record-style with all fields in declaration order |

AgL structured values (`array`, `dict`, record, enum, exception) always render on
a **single line** — no injected newlines. A `json` value renders as **compact**
(single-line) JSON whether it is nested inside another structured value or
interpolated directly; use `render(value, pretty = true)` for indented,
multi-line output.

Scalar text conventions:

- `bool` renders as `true` / `false`.
- `decimal` renders in plain fixed-point notation — never scientific
  notation — with trailing zeros dropped (`1.50` → `1.5`, `1E+2` → `100`).
- Nested `text` values (a `text` field inside a record, array element, etc.)
  are emitted as a quoted AgL string literal with full JSON escaping plus
  `\%` for percent signs, so they cannot be mis-read as interpolation syntax.

No boundary tags or other wrappers are added around interpolated values.

### Container literals in interpolation

A non-empty array or dict literal written directly in an interpolation is
checked as `array[json]` or `dict[text, json]`, respectively. A container
literal nested inside that direct literal is checked as `json`. This preserves
AgL container rendering for the direct literal, including its escaping rules;
use an explicit `as json` cast when JSON rendering is required.

To obtain JSON output use an explicit `as json` cast inside the interpolation:

<!-- agl-check: fragment -->
```agl
let r: R = R(x = 1)
print "%{r}"           # → R(x = 1)         (AgL render form — the default)

# A json value renders as compact JSON, whether nested or interpolated directly:
print "%{r as json}"   # → {"x": 1}
```

For indented, multi-line JSON, call `render` explicitly:

<!-- agl-check: fragment -->
```agl
print render(r as json, pretty = true)   # → {
                                         #      "x": 1
                                         #    }
```

## Opaque values in interpolation

Function values render as opaque handles in templates:

```agl
program def main() -> unit =
  let f = fn(x: int) => x
  print "function is %{f}"
```

They still cannot be stored in a `json` slot or used where a JSON-shaped value
is required. `Agent` values are ordinary enum data and render like other enum
values.

## Templates in `exec` commands

`exec` shell templates use the same uniform rendering. Interpolated values
are inserted **verbatim** into the command string — there is **no automatic
shell quoting**. The workflow author is responsible for writing shell-safe
commands. See [Shell execution](shell-execution.md) for details.

## Errors

- Newline inside `%{…}` — lexical error.
- Unterminated string, unterminated interpolation, unknown escape — lexical
  errors.
