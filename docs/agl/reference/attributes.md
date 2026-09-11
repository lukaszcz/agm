# Attributes

[← Index](index.md)

An attribute is a `@name` or `@name(args)` prefix on a defining declaration.
It annotates the declaration for the compiler or the host; nothing in the
program can read it.

```ebnf
attributes ::= attribute+
attribute  ::= "@" NAME ["(" arg_list? ")"] NEWLINE?
```

## Placement

An attribute may prefix a `record`, `enum`, enum member, `exception`, `type`
alias, any `def` form (`program`, `builtin`, and `extern` included), a
`builtin var`, a field, a parameter, and a `let`/`var` binding. `import`,
`use`, `export`, `infix`, and scope regions take none.

Several attributes may be written in a row, on the line above the declaration
or in front of it on the same line. An enum member's attributes follow its
`|`. Token-level rules: [Lexical structure](lexical-structure.md#attributes).

```agl
@doc("Formats one entry.")
def g(a: int, @arg-named key: text) -> text = "%{a}: %{key}"

record Entry
  @arg-named
  label: text
  @arg-named count: int

enum Shape
  | @doc("a rectangle") @arg-pos Rect(width: int, height: int)
  | Empty
```

Every built-in attribute takes literal constants only, positionally: either
nothing or one text literal. An unknown name, a misplaced, repeated, or
conflicting attribute, and an argument list the attribute does not admit are
static errors.

## Catalog

| Attribute | Argument | Allowed on | Effect |
| --------- | -------- | ---------- | ------ |
| `@doc("…")` | prose | every defining declaration | Describes the declaration; hosts show it in help. |
| `@arg-pos` | none | parameter, field, or their declaring `def`, `record`, `exception`, or enum member | Positional-only zone. |
| `@arg-std` | none | same | Standard zone (positional or named). |
| `@arg-named` | none | same | Named-only zone. |
| `@extern-name("…")` | Python identifier | `extern def` | Names the companion function. |
| `@opt-name("…")` | flag word | `program def` parameter | External spelling: flag, `--no-` negation, config key, completion. |
| `@opt-short("c")` | one ASCII letter | `program def` parameter | One-letter flag `-c`. |
| `@opt-env("VAR")` | variable name | `program def` parameter | Environment fallback when no CLI token supplies the value. |
| `@opt-metavar("…")` | placeholder | `program def` parameter | Value placeholder in usage and help. |
| `@opt-hidden` | none | `program def` parameter | Omits the flag from help and completion; the parameter still binds. |

## Zone attributes

`@arg-pos`, `@arg-std`, and `@arg-named` are mutually exclusive. In front of
a parameter or field they zone that entry; in front of the declaration they
set the default zone of every entry that names none. Entries are listed in
zone order: positional-only, standard, named-only.

Default zones: `def`, `builtin def`, `extern def`, and `fn` parameters are
standard; `program def` parameters are named-only; a method receiver `self`
is always positional-only. Record, exception, and enum-member fields are
standard. Zone semantics: [Functions](functions.md#parameters),
[Types](types.md#record-types), [Pattern matching](pattern-matching.md).

## `@doc`

One text literal of prose. It never changes a declaration's meaning. A host
shows a `program def`'s `@doc` as the program's description and each value
parameter's `@doc` as that parameter's help ([Host environment](host-environment.md#help)).

## `@extern-name`

Names the Python companion function of an `extern def` when it differs from
the declared AgL name ([Python FFI](ffi.md#declarations-and-companions)).

## Program parameter attributes

`@opt-name`, `@opt-short`, `@opt-env`, `@opt-metavar`, and `@opt-hidden`
shape how a `program def`'s value parameter is addressed on the host's command
line ([Program definitions](program-structure.md#program-definitions)). They
are legal nowhere else.

- `@opt-name` takes a flag word: ASCII letters and digits with single interior
  hyphens, so both `--word` and `--no-word` can be formed. It replaces the
  declared name everywhere the parameter is addressed externally.
- `@opt-short` takes exactly one ASCII letter.
- `@opt-env` reads its variable when no CLI token supplies the parameter. An
  unset variable and an empty one both supply nothing; resolution falls
  through to the config table and then the declared default.
- `@opt-hidden` keeps a positional-capable parameter's usage slot.

A positional-only parameter is never addressed by name, so `@opt-name`,
`@opt-short`, `@opt-env`, and `@opt-hidden` on one are static errors.
`@opt-metavar` and `@doc` still apply to it.

```agl
@doc("Publish one artifact.")
program def main(
  @doc("Artifact to publish.") @opt-metavar("PATH") @arg-pos artifact: text,
  @doc("Where to publish it.") @opt-short("t") @opt-env("PUBLISH_TARGET") target: text = "staging",
  @opt-hidden trace-id: text = "",
) -> unit =
  print "%{artifact} -> %{target}"
```

Value resolution and help rendering: [Host environment](host-environment.md#program-arguments).
