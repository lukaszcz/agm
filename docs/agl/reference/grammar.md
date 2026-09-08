# Grammar

[← Index](index.md)

The collected surface grammar. Names, numbers, templates, and layout rules
are specified in [Lexical structure](lexical-structure.md).

Notation: `::=` defines a production; `|` separates alternatives; `?`, `*`,
`+` mark optional and repeated elements; quoted strings are literal source
text. `NEWLINE`, `INDENT`, and `DEDENT` below are **source-layout notation**:
they describe a line break and indentation change, not text written in source.

## Programs and blocks

```ebnf
name       ::= NAME | OP_NAME
field_name ::= NAME

program      ::= module_block EOF

module_block ::= [ module_item ((NEWLINE | ";") module_item)* (NEWLINE | ";")? ]
module_item   ::= scope_region | item
block         ::= item ((NEWLINE | ";") item)* (NEWLINE | ";")?

item       ::= import_decl                  (* header position only; scope_item also permits it *)
             | use_decl                     (* module-root or scope-region header only *)
             | builtin_var_def              (* root or standard-library scope region *)
             | record_def                   (* root only *)
             | enum_def                     (* root only *)
             | type_alias                   (* root only *)
             | exception_def                (* root only *)
             | export_decl                  (* header position only; scope_item also permits it *)
             | program_func_def             (* root or scope region *)
             | infix_decl                   (* root only *)
             | func_def                     (* root only *)
             | builtin_func_def             (* root only *)
             | extern_func_def              (* module root only; file-backed modules only *)
             | let_decl | var_decl | assign_stmt
             | expr

builtin_modifier ::= "builtin" NEWLINE?
```

A block's value is its last item. A final `let_decl` or `var_decl` has type
`unit` (or bottom when its initializer exits); its binding remains visible to
any enclosing construct that evaluates a continuation after the block, such as
a loop's `until` condition.

### Scope regions

```ebnf
scope_region ::= "scope" scope_path (NEWLINE | ";")
                 [scope_item ((NEWLINE | ";") scope_item)* (NEWLINE | ";")?]
                 "end" scope_path
               | "scope" scope_path INDENT
                 scope_item ((NEWLINE | ";") scope_item)* (NEWLINE | ";")?
                 DEDENT NEWLINE "end" scope_path
scope_path   ::= NAME ("::" NAME)*
scope_item   ::= scope_region | use_decl
               | import_decl                  (* header position only *)
               | export_decl
               | record_def | enum_def | exception_def | type_alias
               | func_def | program_func_def | extern_func_def
               | builtin_var_def | builtin_func_def
               | let_decl | var_decl
```

A scope region has a mandatory matching closer: `scope A::B` closes with
`end A::B`. Its items either share the header's layout level or sit in an
indented block beneath it; the closer stands at the header's own level either
way. Regions may appear only as module-root items or as items of another
scope region. They may nest, and a multi-segment header is equivalent to
nested single-segment regions. Scope
regions contain nested regions, header `use` and `import` declarations,
`export` declarations, static declarations (including every `builtin` form),
`program def` declarations, and `let`/`var` bindings; bare expressions, `:=`
assignments, and infix declarations are not permitted. `scope` is contextual at item start before a scope path, and `end`
is contextual only for a complete closer at an open region's layout level;
both remain ordinary names in expression positions.

`"builtin"` is a **declaration modifier** that behaves like a decorator: it may
sit on the same line as the declaration it adorns (`builtin enum …`) or on the
line directly above it (`builtin` then `enum …`). The newline after a modifier
is insignificant. `builtin` prefixes a `record`, `enum`, or `exception`.
`builtin def` is a body-less declaration form, not a modifier applied to an
ordinary `def`. `builtin` is not accepted for type aliases or extern
functions. The `extern` of an `extern def` and the `program` of a
`program def` place the same way: each may sit on its declaration's line or on
the line directly above it.

## Import and export declarations

```ebnf
import_decl ::= "import" module_path ["/*"]
                ("as" NAME | "::" tail)? [hiding_clause]
use_decl    ::= "use" use_target ("::" tail | "as" ref_name) [hiding_clause]
export_decl ::= "export" module_path ["/*"] ["::" braces] [hiding_clause]

tail          ::= "*" | braces | path_atom ["as" ref_name]
braces        ::= "{" brace_item ("," brace_item)* ","? "}"
brace_item    ::= path_atom ["as" ref_name]
use_target    ::= "/" qualifier_path | "::" scope_path | qualifier_path
module_path   ::= NAME ("/" NAME)*    (* byte-adjacent, as is a trailing "/*" *)
qualifier_path ::= NAME ("/" NAME)* ("::" NAME)*
ref_name      ::= name
hiding_clause ::= "hiding" path_atom ("," path_atom)*
path_atom     ::= (NAME "::")* name
```

`"import"`, `"use"`, and `"export"` are contextual at item start when they
begin their declaration form. `"hiding"` is contextual within those headers.
They remain valid identifiers elsewhere. An import alias is an identifier
because it becomes a qualifier segment. An import alias and a tail are
exclusive. Braces cannot be empty or nested, cannot contain `*`, and cannot
be combined with `hiding`. `hiding` is valid on a plain import, an import glob,
a module wildcard import, or a use glob. An export accepts brace tails but not
`::*`. A `use` alias for a complete scope or module target must be a `NAME`,
because it becomes a qualifier segment; selected member renames may use any
`name`.

Examples:

<!-- agl-check: fragment -->
```agl
import foo/bar
import foo/bar as A
import foo/bar::{x, y}
import foo/bar::x as X
import foo/bar::* hiding internal
import foo/*::*
import foo/bar/* as A
export foo/bar::{x as X, y}
export foo/bar/* hiding internal
use Point::{distance as d}
use ::Scope::*
```

### Suites (indented blocks)

```ebnf
suite ::= NEWLINE INDENT block DEDENT
```

A suite starts on the line after its introducer. Its block is indented more
than the introducing construct and ends when the indentation returns to that
construct's level.

### Inline bodies

Every body may be written as a suite. Written inline, a `;` sequence is
admissible in the three body positions whose end is marked by a body-specific
token: parenthesized blocks, loops, and `try` expressions.

```ebnf
marked_body   ::= (marked_item ";")* marked_item
marked_item   ::= expr | inline_assign | let_decl | var_decl
inline_assign ::= assign_target ":=" or_expr
```

Those three *marked* bodies share `marked_body` and allow any marked item in
final position, including a `let` or `var` binder. A final binder makes the
body `unit`-valued unless its initializer exits, in which case it is
bottom-valued.

| body | delimiter | inline form |
| ---- | --------- | ----------- |
| `( … )` | `)` | `marked_body` |
| `do … until`/`done` | `until` / `done` | `marked_body` |
| `try … catch` | `catch` | `marked_body`, last item not a `try` or lambda |
| `def f() = …` | enclosing newline or `;` | exactly one expression; delimiter starts the next block item |
| `… => …` | *none* | exactly one `closed_item` |

An inline `def` body is exactly one expression, not a marked-body sequence.
Its enclosing block separator — a newline or `;` — ends the body and starts
the next block item. A `=>` body ends at nothing at all — a following `|`,
`else`, or `catch` could belong either to the body or to the enclosing branch
list.

What differences remain among the marked bodies are derived, not stipulated: a
body's last item may not be a form that could consume the body's own
terminator. Nothing consumes `)`. A loop's `until`/`done` is mandatory and
occurs exactly once per loop, so even a nested loop in final position is
unambiguous — `do … until p until q` binds the inner `until` innermost-first.
But `catch` is a repeatable clause, so two forms are barred from a `try`
body's final position. A nested `try` there consumes every following `catch`,
leaving none for the body's own `try`, which requires one — so that spelling
could never parse in any case. A lambda is barred because its body is
introduced by `=>` and is itself unmarked, extending rightwards, so a
following `catch` is contested whenever that body ends in a `try`. Both stay
legal anywhere earlier in the sequence, and in final position once
parenthesized — `try (fn(x: int) -> int => x) catch _ => 0` — which restores
the marker.

A `=>` body holds a single `closed_item` — no `;`, no binder, and none of the
right-extending forms whose own branch lists would swallow the enclosing form's
continuation.

This is not a special restriction on `;`. Within a block, `;` and a newline
are the same separator (see [Programs and blocks](#programs-and-blocks)), and
a newline cannot appear inside an inline body either. To write a multi-item
body after `=>`, parenthesize it or use the suite form:

<!-- agl-check: fragment -->
```agl
| 1 => (let doubled = k * 2; print "doubled:%{doubled}")
| 2 =>
    let doubled = k * 2
    print "doubled:%{doubled}"
```

Parentheses directly after a callee are that call's argument list, so a
parenthesized block passed as an argument carries its own parentheses:
`print((let x = 4; x * 2))`.

## Attributes

```ebnf
attributes ::= attribute+
attribute  ::= "@" NAME ["(" arg_list? ")"] NEWLINE?
```

An attribute prefixes a **defining declaration**: a record, an enum, an enum
member, an exception, a type alias, a `def` in any of its `program`, `builtin`,
and `extern` forms, a `builtin var`, a record/enum-member/exception field, a
function or lambda parameter, and a `let` or `var` binding. `import`, `use`,
`export`, and `infix` declarations and scope regions define no name of their
own and take no attribute.
[Lexical structure](lexical-structure.md#attributes) gives the placements an
attribute may take.

An attribute's arguments are an ordinary `arg_list`, so both positional and
named arguments are admitted syntactically. Every built-in attribute takes only
literal constants, positionally — `@doc("Prints a greeting")` — so a computed
or named argument to one is a static error.

Each attribute has its own meaning and its own set of declarations it may
prefix; an unknown attribute name, a misplaced, repeated, or contradicted
attribute, and an argument list its meaning does not admit are static errors.
`@arg-pos`, `@arg-std`, and `@arg-named` place parameters and fields in
[zones](functions.md#parameters).

`@doc(text)` carries one text literal of human-readable prose about the
declaration it prefixes. It is the one attribute every defining declaration
admits, and it never changes a declaration's meaning: the prose describes the
declaration, and nothing in the program can read it. A host surfaces it where
it shows a declaration to a person — the prose on a `program def` describes
that program, and the prose on one of its value parameters describes that
parameter ([Host environment](host-environment.md#program-arguments)).

`@extern-name` names an `extern def`'s companion function
([Python FFI](ffi.md#declarations-and-companions)). `@opt-name`, `@opt-short`,
`@opt-env`, `@opt-metavar`, and `@opt-hidden` shape how a `program def`'s value
parameter is addressed and presented externally
([Host environment](host-environment.md#program-arguments)).

## Type declarations

```ebnf
decl_head        ::= [scope_path "::"] name
record_def       ::= attributes? builtin_modifier? "record" decl_head type_params?
                    "="? record_body
record_body      ::= NEWLINE INDENT field_def (NEWLINE field_def)* NEWLINE? DEDENT
                   | "(" field_list? ")"
                   | field_list
field_def        ::= attributes? "var"? field_name ":" type_expr

enum_def         ::= attributes? builtin_modifier? "enum" decl_head type_params?
                    "="? enum_body
enum_body        ::= enum_member_seq
                   | NEWLINE INDENT enum_member_seq NEWLINE? DEDENT
enum_member_seq  ::= first_enum_member ("|" enum_member)*
first_enum_member ::= "|"? enum_member
enum_member      ::= attributes? name member_payload? | qualifier_chain name member_type_args?
member_type_args ::= "[" type_expr ("," type_expr)* "]"
member_payload   ::= "(" field_list? ")"
field_list       ::= field_inline ("," field_inline)* ","?
field_inline     ::= attributes? "var"? field_name ":" type_expr

exception_def    ::= attributes? builtin_modifier? "exception" decl_head
                    exception_base? exception_body
exception_base   ::= "extends" name
exception_body   ::= NEWLINE INDENT field_def (NEWLINE field_def)* NEWLINE? DEDENT
                   | "(" field_list? ")"
                   | field_list

type_alias       ::= attributes? "type" decl_head type_params? "=" type_expr

type_params      ::= "[" type_param ("," type_param)* "]"
type_param       ::= name | "_"

program_func_def ::= attributes? "program" NEWLINE? "def" decl_head type_params? "(" param_list? ")" ("->" type_expr)? ("=" func_body | suite)

```

A zone attribute in front of a field puts that field in the zone it names; one
in front of the declaration zones every field that carries none of its own. In
the indented block form, the attribute may sit on the field's line or on the
line above it. Fields are listed in zone order — positional-only, then
standard, then named-only.

A `type_params` list declares the declaration's type parameters; each named
entry is an ordinary name in scope as a type throughout the declaration's body.
`_` is an unused positional slot and introduces no type name. See
[Generics](generics.md).

An enum member written as a bare `name` declares a record in the enum's scope;
its optional field list is that record's field list, including optional `var`
field markers. `var` is valid for records and enum-member records,
but not exception fields. A qualified member is a
reference to an existing record, so it has no field list. Qualification is the
declare/reference discriminator: `Entry(x: int)` declares `Enum::Entry`, while
`::Entry` references the current module's `Entry`. See [Enums](types.md#enum-types).

## Type expressions

```ebnf
type_ann  ::= ":" type_expr

type_expr ::= "unit"
            | "text" | "json" | "bool" | "int" | "decimal"
            | name
            | name "[" type_expr ("," type_expr)* "]"   (* applied type *)
            | qualifier_chain name "[" type_expr ("," type_expr)* "]"
            | qualifier_chain name
            | "array" "[" type_expr "]"
            | "dict" "[" "text" "," type_expr "]"
            | func_type

func_type ::= type_atom "->" type_expr
            | "(" type_list? ")" "->" type_expr
type_atom ::= "unit" | "text" | "json" | "bool" | "int" | "decimal"
            | name
            | name "[" type_expr ("," type_expr)* "]"
            | qualifier_chain name "[" type_expr ("," type_expr)* "]"
            | qualifier_chain name
            | "array" "[" type_expr "]"
            | "dict" "[" "text" "," type_expr "]"
type_list ::= type_expr ("," type_expr)* ","?

qualifier_chain   ::= "::" qualifier_segment*
                    | qualifier_segment+
qualifier_segment ::= ["/"] NAME ("/" NAME)* "::"
                    | NAME "[" type_expr ("," type_expr)* "]" "::"
```

`name "[" … "]"` is an applied type: a generic declaration instantiated at
concrete type arguments (`Box[int]`, `Outcome[int, text]`). The built-in
`array[T]` and `dict[text, V]` are the same form.

## Function declarations

```ebnf
func_def         ::= attributes? "def" func_decl_head type_params? "(" param_list? ")" ("->" type_expr)? ("=" func_body | suite)
builtin_func_def ::= attributes? "builtin" NEWLINE? "def" func_decl_head type_params? "(" param_list? ")" "->" type_expr
extern_func_def  ::= attributes? "extern" NEWLINE? "def" func_decl_head type_params? "(" param_list? ")" "->" type_expr
func_decl_head   ::= decl_head | builtin_receiver "::" name
builtin_receiver ::= "array" "[" name "]" | "dict" "[" "text" "," name "]"
                   | "text" | "json" | "int" | "decimal" | "bool"
func_body        ::= expr | suite
param_list      ::= param ("," param)* ","?
param           ::= attributes? field_name [":" type_expr] ("=" or_expr)?
```

A parameter annotation may be omitted only for `self` as the first parameter of
a method; every other parameter requires an annotation.

An inline `def` body after `=` is exactly one expression. It ends at the next
block separator — a newline or `;` — which starts the next block item, so an
inline body admits neither a binder nor a `;` sequence. Parenthesize a
multi-item block or use a suite when the body needs binders or a sequence; for
suite bodies, the `=` before the newline is optional. The return type
annotation is optional for ordinary `def` declarations and required for
`builtin def` and `extern def` —
neither has a body. A zone attribute in front of a parameter, or in front of
the declaration, places parameters in zones; see [Functions](functions.md) for
full zone semantics. No required
positional-fillable (pos-only/standard) parameter may follow a defaulted one
in the same zone. An optional `type_params` list after the function name makes
the `def` generic (e.g. `def id[T](x: T) -> T`); see [Generics](generics.md).

All three function declaration forms accept the same `func_decl_head` surface.
A builtin receiver is valid only in its owning standard-library module and must
use the bare generic form (`array[E]` or `dict[text, V]`); see
[Methods](functions.md#methods). `extern_func_def` is never followed by a body;
it declares a function implemented by a companion Python file (see
[Python FFI](ffi.md)) rather than an AgL expression.

## Infix declarations

```ebnf
infix_decl      ::= ("infixl" | "infixr") infix_op infix_priority?
infix_priority  ::= "at" INT
                  | "at" "prio" infix_op ("+" | "-") INT
infix_op        ::= "or" | "and" | "in"
                  | "==" | "!=" | "<" | "<=" | ">" | ">="
                  | "+" | "-" | "*" | "/"
                  | OP_NAME
```

`infixl` and `infixr` declare a symbolic operator's associativity and optional
integer priority. Larger priorities bind tighter; omitted priority defaults to
the `+`/`-` level. `prio <op> +/- <int>` is resolved from a builtin, a local
operator declaration, an operator made bare-visible by an import wildcard or
tail, or a member made bare by a `use` declaration; a plain qualified import
does not make its fixity available. A chain cannot mix `infixl` and `infixr`
operators at the same priority: parenthesize one side or assign distinct
priorities.

## Bindings and mutation

```ebnf
let_decl       ::= attributes? "let" pattern type_ann? "=" expr
var_decl       ::= attributes? "var" decl_head type_ann? "=" expr
builtin_var_def ::= attributes? "builtin" NEWLINE? "var" name type_ann ["=" expr]  (* standard library only *)
assign_stmt ::= assign_target ":=" expr
assign_target ::= qualifier_chain? name
                | postfix "[" expr "]"
                | postfix "." field_name
```

A `builtin var` is a body-less, host-backed mutable binding with a mandatory
type and an optional constant initializer. Its host identity is its defining
module, scope path, and name, so same-named declarations in distinct scope
regions remain independent. The initializer must have the
declared type and use only literals, literal containers, constructors, and
unary operators over those. It
becomes the binding default only when the host supplies no initial value. The
`builtin` modifier may sit on the same line or the line directly above (like
`builtin def`). It may be declared only at the root, or in a named scope region,
of a module whose path identity lies under `std`, the entry program included
when its path lies there; no other module may declare one. `std/config`
reserves builtin vars for engine settings.

`var`'s `decl_head` accepts the same optional scope-path prefix as the type
declarations above (`var A::count = 0`). `let` needs no separate grammar for
its own shorthand: a `pattern` that is exactly a bare qualifier chain
spellable as a declaration head (no argument list, no `as` binder, and
otherwise a plain, non-`::`-anchored name chain) declares a scoped binding
instead of matching a pattern — see [Bindings and scope](bindings-and-scope.md)
for the full disambiguation.

Assignment has type `unit` and returns `void`. `assign_target`'s qualifier
accepts any number of segments: a local scope path (`A::B::count`) reaches a
scoped `var` exactly as a qualified read does, while a bare (non-indexed)
cross-module target — written with a qualifier, or bare when an import tail or
`use` puts the name in scope — is valid only when it resolves to a `builtin var`;
type-qualified constructor forms are not assignment targets. An indexed
assignment target's object expression is evaluated like any other read, so
`assign_target` accepts any array- or dict-typed expression there; a field
assignment likewise accepts any record-typed postfix receiver, provided its
field is marked `var`. See [Bindings and scope](bindings-and-scope.md#--destructive-assignment)
for which roots are legal and the evaluation order. Each opening `[` must be
adjacent to the target name or preceding index: `xs[0]` is indexed assignment,
while `xs [0]` is not.

## Loops

```ebnf
loop        ::= for_clause? while_clause? "do" loop_bound?
                (suite loop_end? | inline_body loop_end)
for_clause  ::= "for" name "in" or_expr range_tail? NEWLINE?
range_tail  ::= ("to" | "downto") or_expr ("step" or_expr)?
while_clause::= "while" or_expr NEWLINE?
loop_bound  ::= "[" or_expr "]"           (* int; n <= 0 runs zero iterations *)
loop_end    ::= "until" or_expr | "done"   (* omitted terminator allowed in suite form *)
inline_body ::= marked_item (";" marked_item)*
marked_item ::= expr | inline_assign | let_decl | var_decl
```

A loop body is the one inline position that admits the *open* forms
(`case`, `if`, `try`, and a nested loop). A loop is closed by its terminator
(`until` / `done`), which no open form can extend into, so their branch lists
end unambiguously (see [Inline bodies](#inline-bodies)). A nested
`do … until p until q` binds the inner `until` innermost-first.

At most one `for` and one `while` clause, in that order. `done` and an
omitted (suite-form) terminator are equivalent to `until false`. `break` and
`continue` are nullary expressions of the bottom type, valid anywhere in a
loop's interior — its `while` guard, body, or `until` condition — within the
same function/lambda. `return` may also appear inside a loop; it exits the
nearest enclosing function. See
[Control flow](control-flow.md) for the full clause, scope, and bound
semantics.

## `if`

```ebnf
if_expr        ::= "if" "|"? if_cond_branch ("|" if_cond_branch)* if_else_branch?
if_cond_branch ::= or_expr "=>" branch_body
if_else_branch ::= "|"? "else" "=>" branch_body

branch_body    ::= suite | closed_item
closed_item    ::= or_expr
                 | inline_assign                  (* assign_target ":=" or_expr *)
                 | raise_expr
                 | return_expr
```

`branch_body` is shared by `if` branches, `case` branches, and `catch`
clauses. An inline body after `=>` is exactly **one** item: no `;` sequence
and no binder. See [Inline bodies](#inline-bodies) for why, and for the two
ways to write a multi-item body inline. An inline `:=` has the form
`assign_target := or_expr`; the suite form keeps the unrestricted
`assign_target := expr` form.

Without an `else` branch the `if` expression has type `unit` and returns
`void`. With all branches returning a common type `T`, the `if` expression has
type `T`.

## `case`

```ebnf
case_expr       ::= "case" or_expr "of" case_body
case_body       ::= case_branch_seq
                  | NEWLINE INDENT case_branch_seq NEWLINE? DEDENT
case_branch_seq ::= "|"? case_branch ("|" case_branch)*
case_branch     ::= pattern "=>" branch_body
```

The branch list either follows `of` on the same line or forms an indented
block on the lines below it. The first branch's `|` is optional in both forms;
every later branch starts with `|`.

## `try` / `catch`

```ebnf
try_expr          ::= "try" try_body catch_clause+
try_body          ::= suite | (marked_item ";")* try_tail
try_tail          ::= or_expr | inline_assign | try_letvar_decl | raise_expr
                    | return_expr | if_expr | case_expr | loop
try_letvar_decl   ::= attributes? "let" pattern type_ann? "=" try_value
                    | attributes? "var" decl_head type_ann? "=" try_value
try_value         ::= or_expr | raise_expr | return_expr | if_expr | case_expr | loop
catch_clause      ::= "catch" catch_pattern "=>" branch_body
catch_pattern     ::= name ("as" name)?
                    | "_" ("as" name)?
```

`catch` marks where a `try` body ends, so an inline try body is a full `;`
sequence — binders and `assign_target := or_expr` assignments included. A final `let`
or `var` binder is allowed
and makes the body `unit`-valued unless its initializer exits. A final item and
a final binder RHS must remain closed: an open form there would consume the
`catch`.

## Patterns

```ebnf
pattern        ::= pattern_atom ("as" name)*
pattern_atom   ::= "_"
                 | INT | DECIMAL | "true" | "false" | "null" | STRING
                 | name
                 | name "(" pattern_fields? ")"
                 | qualifier_chain name
                 | qualifier_chain name "(" pattern_fields? ")"
pattern_fields ::= pattern_field ("," pattern_field)* ","?
pattern_field  ::= pattern              (* positional sub-pattern *)
                 | field_name "=" pattern
                                           (* named sub-pattern: field = subpattern *)
```

`pattern as name` binds `name` to the complete value matched by `pattern`.
It has the lowest pattern precedence, may be chained, and cannot use `_` as
its binder name. The binder is always a variable binder.

A qualified member pattern (`Option::Some(value)`,
`module::Option::Some(value)`, or `/module::Option::Some(value)`) names a
member record with `::`. A leading `/` is an anchored qualifier; without it,
the qualifier is resolved as a suffix. The complete qualifier through `::` is
byte-adjacent. A qualified pattern's argument list is optional
(`Option::None` and `Option::None()` are both nullary matches) except at the
root of a `let` pattern, where writing it or not distinguishes a match from a
scoped binding — see [Bindings and scope](bindings-and-scope.md). Unqualified
constructor ownership is selected by the scrutinee's static nominal type, even
when multiple enums contribute the same member name or a record constructor
spelling collides with an injected member name; a qualifier is optional and
must agree with that type when present ([Generics](generics.md),
[Pattern matching](pattern-matching.md)). Type arguments are carried by the
scrutinee type rather than written in a pattern.

In a constructor pattern, **positional sub-patterns** (`pattern` without a
`name "="` prefix) fill positional-capable (pos-only/standard) constructor fields
left to right. Named sub-patterns follow. A bare name positional sub-pattern
that lands on a named-only field is reinterpreted as the shorthand `name = name`.

A `STRING` pattern may not contain interpolation.

## Raw-tail calls

```ebnf
raw_tail_form   ::= raw_call | dotted_raw_call | raw_juxt
raw_call        ::= raw_callee type_args? raw_tail
dotted_raw_call ::= postfix "." raw_callee type_args? raw_tail
raw_juxt        ::= postfix raw_call
                  | postfix juxt_atom juxt_suffix* "." raw_callee type_args? raw_tail
raw_callee      ::= "exec$" | "ask$"
type_args       ::= "::" "[" type_expr ("," type_expr)* "]"
raw_tail        ::= inline_raw_tail | block_raw_tail
```

A raw tail may start a call directly or follow a runtime member projection.
`receiver.ask$ payload` is equivalent to `receiver.ask(payload)`, including
normal field-then-method resolution. The optional `type_args` group is
recognized only when its `::` is immediately adjacent to the raw name:
`exec$::[T]` and `receiver.ask$::[T]`. Whitespace before the `::` makes it
payload text instead, so `ask$ ::[T]` and `receiver.ask$ ::[T]` have no type
arguments.

An inline raw tail is all text from its first non-whitespace character through
the end of the line, except that trailing spaces and tabs are removed. A block
raw tail follows the name (and optional type arguments) with a newline and a
more-indented block; its dedented lines become one newline-joined payload,
dropping the blank lines that trail its last content line while keeping any
before and between content lines. A raw call requires a nonempty inline tail or
a block with at least one nonblank line. Raw text is tokenized as fragments and
`%{expr}` interpolations, not as ordinary AgL expressions.

A `raw_tail_form` stands in for `expr` only where the grammar guarantees that
nothing else follows on its line: as an `item`; as a `let_decl`, `var_decl`, or
`assign_stmt` right-hand side; as an inline `func_body`; as the operand of a
`return` in those same positions; or, through `raw_juxt`, as the
single-argument juxtaposition argument of a call whose callee precedes it on
the line (for example, `print receiver.ask$ prompt`). It is not valid inside
brackets, branch/catch inline bodies, or another inline expression. Use the
ordinary call form there.

```agl
program def main() -> unit =
  let path = "."
  let output: text = exec$ printf '%s' %{path}
```

## Expressions

```ebnf
expr      ::= case_expr | if_expr | loop | try_expr | raise_expr
            | return_expr | record_update | lambda_expr | or_expr

record_update ::= update_target "with" field_update ("," field_update)*
update_target ::= record_update | or_expr     (* chaining is left-associative *)
field_update  ::= field_name "=" or_expr

or_expr       ::= infix_expr
infix_expr    ::= infix_operand (infix_op infix_operand)*
infix_operand ::= "not"* is_expr

is_expr   ::= cast "is" "not"? qualified_constructor
            | cast

cast           ::= cast "as" type_expr      (* type cast; may raise CastError *)
               | cast "as?" type_expr      (* convertibility test; yields bool *)
               | unary
               (* left-associative; "as?" is a single token — no whitespace *)

unary          ::= "-" unary | juxt

juxt           ::= postfix juxt_arg     (* single-arg sugar; non-chaining *)
               | postfix

juxt_arg       ::= juxt_atom juxt_suffix*
juxt_suffix    ::= "." field_name
               | "[" expr "]"                  (* adjacent bracket only *)
               | "(" arg_list? ")"
               | "::" "[" type_expr ("," type_expr)* "]" "(" arg_list? ")"

postfix        ::= postfix "." field_name          (* runtime field access *)
               | postfix "(" arg_list? ")"         (* call with parentheses *)
               | postfix "[" expr "]"              (* adjacent bracket only *)
               | postfix "::" "[" type_expr ("," type_expr)* "]"   (* explicit type application *)
               | atom

applied_type_qualified_constructor ::= qualifier_chain NAME "[" type_expr ("," type_expr)* "]" "::" NAME
                                           | NAME "[" type_expr ("," type_expr)* "]" "::" NAME
                                           (* `[` is byte-adjacent to the preceding NAME *)

atom           ::= INT | DECIMAL | "true" | "false" | "null"
               | "(" ")"                           (* unit literal *)
               | array_literal
               | dict_literal
               | name                              (* variable / constructor reference *)
               | qualifier_chain name               (* qualified ref / constructor *)
               | applied_type_qualified_constructor
               | template
               | "(" expr ")"                      (* parenthesized expr *)
               | "(" paren_block ")"               (* parenthesized block *)
               | break_expr
               | continue_expr

paren_block    ::= (marked_item ";")+ marked_item
                 | inline_assign
                 | let_decl | var_decl

juxt_atom      ::= INT | DECIMAL | "true" | "false" | "null"
                 | array_literal | dict_literal | template
                 | NAME                              (* bare references exclude OP_NAME *)
                 | qualifier_chain name               (* qualified ref / constructor *)
                 | applied_type_qualified_constructor

qualified_constructor ::= qualifier_chain name | name

raise_expr     ::= "raise" or_expr
return_expr    ::= "return" or_expr?
break_expr     ::= "break"
continue_expr  ::= "continue"
```

A bare name atom is resolved by scope and position: it may name a variable, a
record constructor, an injected enum-member constructor, or a generic
`def`/constructor used as a first-class value. The typed postfix form carries
explicit type arguments to a generic `def` or bare constructor
(`id::[int](5)`, `Some::[int](value = 1)`, `apply::[int, int](…)`), or
instantiates a generic function value (`id::[int]`). A member selected through
its owning generic enum puts the type arguments on the type side
(`Option[int]::Some(value = 1)`).
A qualifier that is a scope or module route rather than an owning type leaves
the constructor's own spelling intact, so it carries type arguments exactly as
the unqualified form does (`A::Pair::[int]`, `boxes::A::Box::[int]`). In that explicit
type-qualified constructor form, the `[` is byte-adjacent to the applied type
name: `Option[int]::Some` is valid, while `Option [int]::Some` is not. This
restriction does not apply to ordinary applied types, so both `Option[int]` and
`Option [int]` are valid type expressions. Both the applied type name and
constructor name are `NAME` (not `OP_NAME`). A `postfix "." field_name` is
always runtime field access; constructor qualification uses `::`. See
[Generics](generics.md).

## Lambda expressions

```ebnf
lambda_expr ::= "fn" "(" param_list? ")" ("->" type_expr)? "=>" expr
```

The return type annotation is optional; when omitted, it is inferred from
the body. Parameter types are always required.

## Calls

```ebnf
arg_list        ::= arg ("," arg)* ","?
arg             ::= element_expr                 (* positional *)
                  | placeholder_arg              (* positional hole *)
                  | field_name "=" element_expr  (* named *)
                  | field_name "=" placeholder_arg (* named hole *)
placeholder_arg ::= "?" | "?<digits>"
```

`element_expr` is `expr` without a bare record update: in comma-separated
element positions (call arguments, array and dict literal elements) a record
update — including one that forms the body of an inline lambda — must be
parenthesized, so that its update list cannot be confused with the enclosing
comma-separated list. See
[Record update](expressions.md#record-update).

A placeholder is legal only as a whole parenthesized call argument: either a
positional argument (`f(?, x)`) or the value of a named argument (`f(x = ?)`).
It is not an expression, so forms such as `f(? + 1)`, a standalone `?`, or a
placeholder in single-argument sugar do not parse. The numbered form has no
whitespace between `?` and its digits; `? 1` is a bare placeholder followed by
an integer argument, not `?1`.

Named arguments are available at declared-name call sites (`def`s and
built-ins). A function value is called with positional arguments only.
A bare name in positional position that lands on a **named-only** parameter
(after all positional-capable slots are filled) is reinterpreted as the named
argument `name = name` — this shorthand works in any call context (functions,
constructors) and is triggered solely by the parameter's zone.

## Literals

```ebnf
array_literal ::= "[" (element_expr ("," element_expr)* ","?)? "]"

dict_literal ::= "{" (dict_entry ("," dict_entry)* ","?)? "}"
dict_entry   ::= STRING ":" element_expr    (* no interpolation in keys *)
               | field_name ":" element_expr (* shorthand for the string key *)
```

`element_expr` excludes bare record updates — see [Calls](#calls).

## Templates

```ebnf
template      ::= '"' (text_fragment | interpolation)* '"'
                | "'" (text_fragment | interpolation)* "'"
                | '"""' (text_fragment | interpolation)* '"""'
                | "'''" (text_fragment | interpolation)* "'''"

interpolation ::= "%{" expr "}"
```

A `text_fragment` is literal template text between the surrounding quote
delimiters and any interpolation; its escapes and, for triple-quoted templates,
dedent are described in [Lexical structure](lexical-structure.md). Newlines are
not permitted inside `%{…}`.

## Deterministic-parse notes

- `==` is the equality operator; a single `=` is reserved for bindings, named
  arguments, and declarations, and is not an expression operator.
- Chained comparisons are rejected with a non-associativity message after the
  infix chain is grouped.
- The `[N]` after `do` is a single lexical unit, so it never conflicts with
  an array literal.
- A `|`, `else`, `catch`, or `until` at the start of a line attaches to the
  innermost construct that can accept it; the layout rules guarantee each
  such token belongs to exactly one construct
  ([Lexical structure](lexical-structure.md)).
- The optional single-argument call syntax applies a function to one following
  argument: `f x` means `f(x)`. It does not chain, so `f x y` is invalid.
- `()` is both the unit literal and the empty argument list of a zero-arg
  call — the two are syntactically unified.
- Inline branch and `catch` bodies hold a single *closed* item — `or_expr`,
  `:=`, `raise`, or `return`. They admit neither a `;` sequence nor a binder,
  nor the *right-extending* forms (`case`, `if`, `try`, a loop) or a lambda, whose body is
  itself an `expr`: those would extend rightwards into the enclosing branch
  list's `|` / `else` / `catch`. Write them as a suite or parenthesize them
  (see [Inline bodies](#inline-bodies)).
- `until` conditions reference `or_expr` directly, so a right-extending form there must
  be parenthesized.
- Bodies whose end is marked by a token — a parenthesized block, a loop body,
  a `try` body — take a full `;` sequence; loop bodies additionally admit the
  right-extending forms, because the loop terminator closes the body.
- A `return` followed by a newline is a bare `return`; its operand does not
  continue onto the next line.
