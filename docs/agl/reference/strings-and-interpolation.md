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
interpolation ::= "${" expr ("as" renderer_name)? "}"
renderer_name ::= VAR_NAME            (* e.g. raw, json, bullets, default *)
```

The expression may be anything — a variable, field access, arithmetic, an
agent call, a parenthesized `case` expression. Its value is converted to
text by a **renderer**. Without an `as` clause the *default* renderer for
the surrounding context applies; with one, the named renderer is used.

Renderer names are validated statically: an unknown renderer is a static
error listing the known renderers, and a host-registered renderer that
declares the type kinds it supports rejects unsupported operand types at
check time ([Host environment](host-environment.md)).

Interpolation is **type-directed and safe by default**: rendering depends on
the value's type, and in prompts the default rendering marks value
boundaries so model-produced or user-provided text is less likely to be
confused with the surrounding instructions. This is a clarity measure, not a
security sandbox.

## Rendering contexts

A template is rendered in one of three contexts, each with its own default:

| Context | Where | Default rendering |
| ------- | ----- | ----------------- |
| **Prompt** | the template argument of an agent call | boundary-marked (below) |
| **Shell** | the template argument of `exec` | shell-quoted ([Shell execution](shell-execution.md)) |
| **Console** | everywhere else — `print` operands, plain template expressions, templates bound with `let`/`var` | plain (below) |

Note the consequence: a template *built ahead of time* (`let p = "Fix
${artifact}"`) is rendered with console rules at the point the `let` runs;
only a template written directly as an agent call's argument gets prompt
rendering. To get boundary-marked values into an agent prompt, interpolate
in the call's own template.

## Prompt rendering (default renderer, prompt context)

| Value type | Rendering |
| ---------- | --------- |
| `text` | boundary-marked verbatim text |
| `int`, `decimal`, `bool` | bare scalar text |
| `json` | boundary-marked pretty JSON |
| `list`, `dict` | boundary-marked pretty JSON |
| records | boundary-marked pretty JSON of the fields |
| enums | boundary-marked pretty JSON with the `"$case"` tag |
| exceptions | boundary-marked pretty JSON of the diagnostic fields |

The boundary marker is a stable tag wrapping the rendered value:

```text
<dsl-value name="artifact" type="text">
…value…
</dsl-value>
```

- `name` is the interpolated variable's name when the expression is a plain
  variable reference; otherwise `value`.
- `type` is the value's type label: a built-in kind (`text`, `int`,
  `decimal`, `bool`, `json`, `list`, `dict`) or the declared type name for
  records, enums, and exceptions (e.g. `Review`, `AgentParseError`).

Scalars (`int`, `decimal`, `bool`) are rendered bare, without markers.

## Console rendering (default renderer, console context)

- `text` — verbatim, no markers.
- `int`, `decimal`, `bool` — scalar text.
- everything else (`json`, lists, dicts, records, enums, exceptions) —
  pretty-printed JSON (2-space indent), no markers.

Scalar text conventions, in every context:

- `bool` renders as `true` / `false`.
- `decimal` renders in plain fixed-point notation — never scientific
  notation — with trailing zeros dropped (`1.50` → `1.5`, `1E+2` → `100`).
  Decimals in JSON output are emitted as exact unquoted numbers, never via
  binary floats.
- Enum JSON uses the `"$case"` variant tag; exception JSON is the flat
  object of the exception's fields.

## Built-in renderers

Four renderers are always available; hosts may register more
([Host environment](host-environment.md)).

### `default`

The context default described above. Writing `as default` explicitly is
allowed and means the same as omitting the clause.

### `raw`

Plain conversion with **no boundary markers**: `text` verbatim; numbers and
booleans as scalar text; `json` as compact single-line JSON; lists, dicts,
records, enums, and exceptions as pretty JSON. Use `raw` deliberately — in
prompts it removes the boundary that separates data from instructions, and
in `exec` templates it bypasses shell quoting entirely
([Shell execution](shell-execution.md)).

```agl
let report = critic "Raw model output follows:\n${e.raw as raw}"
```

### `json`

Always pretty-printed JSON (2-space indent), regardless of type, without
markers:

```agl
print "${review as json}"
```

### `bullets`

A `list` renders as one `- item` line per element, each element rendered as
scalar text (structured elements as pretty JSON). A non-list value falls
back to pretty JSON.

```agl
impl "Fix these issues:\n${issues as bullets}"
```

## Templates in `exec` commands

In shell context the default rendering is *plain text then shell-quoted*:
scalars render as scalar text, structured values as **compact** single-line
JSON, and the result is quoted as a single shell word. `as raw` inserts the
plain text verbatim (unquoted); any other explicit renderer is applied first
and its output is then shell-quoted. Full rules in
[Shell execution](shell-execution.md).

## Errors

- Unknown renderer name — static error.
- Renderer/type mismatch for a kind-restricted host renderer — static error.
- Newline inside `${…}` — lexical error.
- Unterminated string, unterminated interpolation, unknown escape — lexical
  errors.
