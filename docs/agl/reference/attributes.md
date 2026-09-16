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
| `@param` | none | module-root or scope-region `let`/`var` binding | Exposes the binding as a host parameter. |
| `@opt-name("…")` | flag word | `program def` parameter or `@param` binding | External spelling: flag, `--no-` negation, config key, completion. |
| `@opt-short("c")` | one ASCII letter | `program def` parameter or `@param` binding | One-letter flag `-c`. |
| `@opt-env("VAR")` | variable name | `program def` parameter or `@param` binding | Environment fallback when no CLI token supplies the value. |
| `@opt-metavar("…")` | placeholder | `program def` parameter or `@param` binding | Value placeholder in usage and help. |
| `@opt-hidden` | none | `program def` parameter or `@param` binding | Omits the flag from help and completion; the parameter still binds. |
| `@name("…")` | AgL identifier | field, enum member, or record declaration | Alternative external name; default JSON name. |
| `@json-name("…")` | non-empty text, not `"$case"` | same | Overrides the JSON name only. |
| `@command("…")` | command path | `program def` | Registers the program as that package command. |
| `@description("…")` | prose | `program def` | The registered command's one-line description. |
| `@help("…")` | prose | `program def` | Further prose the registered command's help shows. |

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
shows a `program def`'s `@doc` as the program's description and each host-facing
parameter's `@doc` as that parameter's help ([Host environment](host-environment.md#help)).

## `@extern-name`

Names the Python companion function of an `extern def` when it differs from
the declared AgL name ([Python FFI](ffi.md#declarations-and-companions)).

## `@name` and `@json-name`

Placement: a field of a record, exception, or enum member; an inline enum
member; or a record declaration (a record's own `@name`/`@json-name` doubles
as its `$case` tag wherever it is an enum member). Not legal on an enum or
exception declaration, a parameter, a binding, a function, or a `type` alias.

`@name("…")` takes an AgL identifier that is not a hard keyword and not a
raw-tail name (`exec$`, `ask$`, …); a soft keyword is fine. It is the
declaration's alternative external name, and the default JSON name when no
`@json-name` is given.

`@json-name("…")` takes any non-empty text except `"$case"` (reserved for
the enum-member discriminator). It overrides the JSON name only, leaving the
declared name and any `@name` untouched everywhere else.

Effective JSON name: `@json-name` if present, else `@name`, else the
declared name. This is the object key a record or exception field encodes
and decodes under, and the `$case` tag a record uses wherever it inhabits an
enum, wherever a value crosses JSON: agent structured output, `as`/`as?`
casts, and program parameters. Rendering (`print`, string interpolation, …)
always uses the declared name.

Two sibling fields or enum members whose effective JSON names collide, or
whose declared/`@name` spellings collide, are static errors, checked across
a record's own fields, an exception's inherited field chain, and an enum's
members (inline and referenced alike).

A record's or enum member's `@name` alias is also a valid constructor spelling
in [host value syntax](host-environment.md#value-syntax): `Square` or `sq`
both match the `Square` member below, and a field's `@name` alias binds it in
a constructor call the same way it would bind a declared name.

```agl
enum Shape
  | @name("sq") Square(side: int)
  | Rect(@json-name("w") width: int, @json-name("h") height: int)
  | Circle

record Job
  @name("file") @json-name("file_path") file-path: path
  shape: Shape
```

## Command attributes

`@command` registers a `program def` as a command of the package that owns its
module ([Packages](packages.md#commands)). Its argument is a command path:
space-separated words naming the command a reader invokes, so
`@command("devel review")` is invoked as `devel review`. A path whose first
word is one of the host's own commands, or any word of which looks like an
option, is a static error.

`@description` and `@help` are the registered command's prose. They describe
the registration, not the program, so each is a static error without
`@command` beside it; a program's own prose is `@doc`, which describes it
wherever it runs.

Each of the three appears at most once on a declaration, and all three are
legal on a `program def` alone.

```agl
@command("devel review")
@description("Review changes")
@help("This program reviews changes")
@doc("Change review")
program def main(@arg-pos subject: text) -> unit =
  print "reviewing %{subject}"
```

A program outside a package registers nothing: nothing reads its `@command`.

## Module parameters

`@param` exposes an ordinary static `let` or `var` binding as a module
parameter. The binding must be at module root or directly in a scope region,
must bind exactly one name rather than `_` or a destructuring pattern, and must
have an initializer. Its type annotation is optional. It is not a `builtin var`
and cannot appear in a function, block, lambda, or other nested binding.
Its inferred or annotated type must be closed and decodable at the host
parameter boundary.

Its initializer supplies the value when the host leaves the parameter unset;
when the host supplies a value, that value is bound instead and the initializer
is not evaluated. A `var` remains mutable after that initial host value is
bound. When exported (see [Modules](modules.md#re-exports-and-visibility)), an
importer may also write it.

The presentation attributes `@opt-name`, `@opt-short`, `@opt-env`,
`@opt-metavar`, `@opt-hidden`, and `@doc` apply to the binding. Without
`@opt-name`, its external name is its declared name.

Host command-line and configuration channels use the module parameter's
presentation. The external name is the config leaf as well as the flag name;
qualified module and scope spellings are dotted, while its declaration path
uses `::`. Its [program route and module route](host-environment.md#module-parameters)
use that same external name; the latter identifies its declaring module. See
that section for precedence, help, and REPL behavior.

```agl
@param @doc("Log level") @opt-short("l") var level: int = 1

scope debug
  @param @opt-name("trace") let tracing = false
end debug
```

## Host parameter attributes

`@opt-name`, `@opt-short`, `@opt-env`, `@opt-metavar`, and `@opt-hidden`
shape how a `program def` value parameter or `@param` binding is addressed by
the host. Program parameters are described under
[Program definitions](program-structure.md#program-definitions).

- `@opt-name` takes a flag word: ASCII letters and digits with single interior
  hyphens, so both `--word` and `--no-word` can be formed. It replaces the
  declared name everywhere the parameter is addressed externally.
- `@opt-short` takes exactly one ASCII letter.
- `@opt-env` reads its variable when no CLI token supplies the parameter. An
  unset variable and an empty one both supply nothing; resolution falls
  through to the config table and then the declared default.
- `@opt-hidden` keeps a positional-capable parameter's usage slot.

A positional-only program parameter is never addressed by name, so `@opt-name`,
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

Value resolution and help rendering: [Host environment](host-environment.md#program-arguments)
and [Module parameters](host-environment.md#module-parameters).
