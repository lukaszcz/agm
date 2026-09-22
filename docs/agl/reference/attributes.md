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

Every built-in attribute but `@config` takes one constant text expression
positionally, or nothing at all; see [Constant arguments](#constant-arguments)
below. `@config` instead takes one or more `key = value` entries; see
[`@config`](#config). An unknown name, a misplaced, repeated, or conflicting
attribute, and an argument list the attribute does not admit are static
errors.

## Constant arguments

An attribute's text argument is a literal, a template whose holes name
constants, or a reference to a constant — anything the compiler can reduce to
text before the program runs:

```agl
let aspects = "correctness, efficiency"
let rounds = 3

@doc("Review %{aspects} over %{rounds} rounds.")
@command(default-command)
program def review(scope: text) -> unit = print "%{scope}: %{aspects}"

let default-command = "devel review"
```

A name in such an argument, in a hole or on its own, is a constant of the
**declaring module**: one reached from the attribute's own scope region
outward to the module root, or under an explicit scope path. A name reached
only through another module is a static error, as is one that no constant of
this module declares. Order does not matter — a constant denotes its value, so
an argument may name one declared further down the file — while a constant
defined in terms of itself is a static error.

A hole renders a `text`, `int`, `decimal`, or `bool` constant exactly as
interpolation renders it at runtime. An array, dict, constructor, `unit`, or
`null` constant has no compile-time text and cannot fill a hole, though it
remains an ordinary constant elsewhere. The whole argument must be text: a
number folds inside a hole, not as the argument itself.

Folding happens first, so an attribute that narrows its text to a particular
spelling — `@name`, `@json-name`, `@command`, `@extern-name`, and the `@opt-*`
attributes — checks the folded result.

A binding a host can override — a [module parameter](#module-parameters) or an
[engine setting](host-environment.md#engine-settings) — may be named too, and
folds to the value its declaration writes. Attribute text is fixed when the
program is compiled, so it states the declared default, while a reference to
that same binding in an expression reads whatever value the host bound.

## Catalog

| Attribute | Argument | Allowed on | Effect |
| --------- | -------- | ---------- | ------ |
| `@doc("…")` | prose | every defining declaration | Describes the declaration; hosts show it in help, and enum-member prose annotates derived JSON Schema. |
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
| `@config(key = value, ...)` | one or more keyed entries | `program def` | States the values module parameters and engine settings take when this program is selected. |

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

One [constant text expression](#constant-arguments) of prose. It never changes
a declaration's meaning. A host shows a `program def`'s `@doc` as the
program's description — wherever the program runs, including as the prose of
the command it registers — and each host-facing parameter's `@doc` as that
parameter's help ([Host environment](host-environment.md#help)). Where a host
has room for one line only, such as a listing of commands, it shows the
opening paragraph.
An inline enum member's `@doc`, or a referenced member record's own `@doc`, is
the `description` of that member's `oneOf` alternative in a
[derived JSON Schema](agent-calls.md#derived-json-schema).

## `@extern-name`

Names the Python companion function of an `extern def` when it differs from
the declared AgL name ([Python FFI](ffi.md#declarations-and-companions)).

## `@name` and `@json-name`

Placement: a field of a record, exception, or enum member; an inline enum
member; or a record declaration (a record's own `@name`/`@json-name` doubles
as its `$case` tag wherever it is an enum member). Not legal on an enum or
exception declaration, a parameter, a binding, a function, or a `type` alias.

`@name("…")` takes an AgL identifier that is not a hard keyword; a soft
keyword is fine. It is the declaration's alternative external name, and the
default JSON name when no `@json-name` is given.

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

## `@command`

`@command` registers a `program def` as a command of the package that owns its
module ([Packages](packages.md#commands)). Its argument is a command path:
space-separated words naming the command a reader invokes, so
`@command("devel review")` is invoked as `devel review`. A path whose first
word is one of the host's own commands, or any word of which looks like an
option, is a static error. It appears at most once, and on a `program def`
alone.

A command is a way to refer to a program, so the registration carries no prose
of its own: the program's `@doc` describes it wherever it runs, the listing of
its command included.

```agl
@command("devel review")
@doc("Review the working tree and report findings by severity.")
program def main(@arg-pos subject: text) -> unit =
  print "reviewing %{subject}"
```

A program outside a package registers nothing: nothing reads its `@command`.

## Module parameters

`@param` exposes an ordinary static `let` or `var` binding as a module
parameter. The binding must be at module root or directly in a scope region,
must bind a readable name rather than `_`, and must
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

## `@config`

`@config` states, in source, the values a `program def` selects for module
parameters and engine settings when it is the program a host runs — the
source-level counterpart of that program's own configuration table. It takes
one or more `key = value` entries, comma-separated and optionally spanning
several lines with a trailing comma, and is legal only on a `program def`, at
most once.

Each `key` is an ordinary AgL reference — bare, suffix-qualified, fully
qualified, anchored with a leading `::`, or a scope path — resolved by
ordinary scope lookup in the scope that declares the `program def` (not its
own parameter scope). The same visibility and import rules apply as for any
reference: a target reached only through another module still needs that
module imported to be named.

Each key must resolve to a [module parameter](#module-parameters) — a
`@param` binding anywhere in the program's transitive import closure,
including its own module — or one of the root `std/config` [engine
settings](host-environment.md#engine-settings). Any other target (a plain
`let`/`var`, a non-engine `builtin var`, a function, a type) is a static
error, as is a key that resolves to no declaration — which includes a name
that only a signature parameter binds, since a `program def`'s own parameter
scope is not where a `@config` key resolves. Two entries addressing the same
target, by any two spellings, are a static error.

Each `value` is a [constant expression](bindings-and-scope.md#constant-expressions),
checked against the target's declared type — the module parameter's own
type, or the engine setting's type.

```agl
import std/config
import std/log

@param let retries: int = 1

scope Debug
  @param let verbose: bool = false
end Debug

@config(
  retries = 5,
  Debug::verbose = true,
  config::timeout = Some("30m"),
  log::level = log::Level::Debug,
)
program def main() -> unit = print retries
```

`@config` applies only when its `program def` is the **selected** program: a
program reached through an ordinary call, or a sibling `program def` in the
same module that is not selected, keeps its own parameters and settings
unaffected by it. `agm check` and the REPL validate a `@config` attribute
statically and apply nothing. Where `@config` ranks in the full resolution
order for a module parameter and an engine setting, and how it interacts with
a program's own configuration table: [Host environment](host-environment.md#host-configurable-settings).

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
