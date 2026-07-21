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
field_name ::= NAME | "agent" | "to" | "downto" | "by"

program    ::= block EOF

block      ::= item ((NEWLINE | ";") item)* (NEWLINE | ";")?

item       ::= import_decl                  (* header position only *)
             | builtin_var_def              (* root only; standard library only *)
             | modifier? record_def         (* root only *)
             | modifier? enum_def           (* root only *)
             | modifier? type_alias         (* root only; "builtin" not allowed *)
             | modifier? exception_def      (* root only *)
             | export_decl                  (* root only *)
             | param_decl                   (* root only *)
             | program_decl                 (* root only *)
             | agent_decl                   (* root only *)
             | infix_decl                   (* root only *)
             | modifier? func_def           (* root only *)
             | ("private" NEWLINE?)? extern_func_def  (* root only; file-backed modules only *)
             | let_decl | var_decl | assign_stmt
             | expr

modifier   ::= ("private" | "builtin") NEWLINE?
```

A block's value is its last item. A final `let_decl` or `var_decl` has type
`unit` (or bottom when its initializer exits); its binding remains visible to
any enclosing construct that evaluates a continuation after the block, such as
a loop's `until` condition.

`"private"` and `"builtin"` are **declaration modifiers** that behave like
decorators: a modifier may sit on the same line as the declaration it adorns
(`builtin enum …`) or on the line directly above it (`builtin` then `enum …`)
— the newline after the modifier is insignificant. `"builtin"` is not allowed
on a `type_alias` or on `extern_func_def`; `"private"` composes with
`extern_func_def` the same way it composes with `func_def`.

## Import and export declarations

```ebnf
import_decl ::= ["open"] "import" module_path ["/*"]
                ["as" ref_name]
                [using_clause | hiding_clause]

export_decl ::= "export" module_path ["/*"]
                [using_clause | hiding_clause]

module_path ::= NAME ("/" NAME)*    (* byte-adjacent, as is a trailing "/*" *)
ref_name    ::= name

using_clause  ::= "using" import_item ("," import_item)*
hiding_clause ::= "hiding" ref_name ("," ref_name)*
import_item   ::= ref_name ("as" ref_name)?
```

`"open"` is a contextual soft keyword only when it directly precedes an
item-start `"import"`. `"import"`, `"export"`, and `"private"` are
contextual at item-start; `"using"` and `"hiding"` are contextual within
import and export declarations. They remain valid identifiers elsewhere.

Examples:

<!-- agl-check: skip -->
```agl
import foo/bar
open import foo/bar as A
import foo/bar using x, y
import foo/bar hiding x, y
import foo/bar using x as X, y
import foo/*
import foo/bar/* as A
export foo/bar using x as X, y
export foo/bar/* hiding internal
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
open forms whose own branch lists would swallow the enclosing form's
continuation.

This is not a special restriction on `;`. Within a block, `;` and a newline
are the same separator (see [Programs and blocks](#programs-and-blocks)), and
a newline cannot appear inside an inline body either. To write a multi-item
body after `=>`, parenthesize it or use the suite form:

<!-- agl-check: skip -->
```agl
| 1 => (let doubled = k * 2; print "doubled:${doubled}")
| 2 =>
    let doubled = k * 2
    print "doubled:${doubled}"
```

Parentheses directly after a callee are that call's argument list, so a
parenthesized block passed as an argument carries its own parentheses:
`print((let x = 4; x * 2))`.

## Type declarations

```ebnf
record_def       ::= "record" name type_params? "="? record_body
record_body      ::= param_marker? NEWLINE INDENT block_entry (NEWLINE block_entry)* NEWLINE? DEDENT
                   | "(" field_list? ")"
                   | field_list
block_entry      ::= field_def | param_marker
field_def        ::= field_name ":" type_expr

enum_def         ::= "enum" name type_params? "="? enum_body
enum_body        ::= enum_variant_seq
                   | NEWLINE INDENT enum_variant_seq NEWLINE? DEDENT
enum_variant_seq ::= first_variant_def ("|" variant_def)*
first_variant_def ::= "|"? name variant_payload?
variant_def      ::= name variant_payload?
variant_payload  ::= "(" field_list? ")"
field_list       ::= field_entry ("," field_entry)* ","?
field_entry      ::= field_inline | param_marker
field_inline     ::= field_name ":" type_expr

exception_def    ::= "exception" name exception_base? exception_body
exception_base   ::= "extends" name
exception_body   ::= param_marker? NEWLINE INDENT block_entry (NEWLINE block_entry)* NEWLINE? DEDENT
                   | "(" field_list? ")"
                   | field_list

type_alias       ::= "type" name type_params? "=" type_expr

type_params      ::= "[" name ("," name)* "]"

param_marker     ::= "/" | "*" | "@" NAME    (* NAME must be pos, std, or named *)

param_decl       ::= "param" name (":" type_expr)? ("=" expr)?
program_decl     ::= "program" name

agent_decl       ::= "agent" name ("=" STRING)?
```

A `param_marker` splits a parameter or field list into **zones**: `/` (≡ `@std`)
ends the positional-only zone and begins standard; `*` (≡ `@named`) ends the
standard zone and begins named-only; `@pos` opens the positional-only zone and
must be the first entry. In the indented block form, a marker may appear as the
optional leading entry on the header line and/or on its own line between field
definitions.

A `type_params` list declares the declaration's type parameters; each is an
ordinary name in scope as a type throughout the declaration's body. See
[Generics](generics.md).

The runner string of an `agent` declaration must be a literal string with no
`${…}` interpolation; an interpolation hole is a static error.

## Type expressions

```ebnf
type_expr ::= "unit"
            | "text" | "json" | "bool" | "int" | "decimal"
            | name
            | name "[" type_expr ("," type_expr)* "]"   (* applied type *)
            | qual_prefix name "[" type_expr ("," type_expr)* "]"
            | qual_prefix name         (* module-qualified type *)
            | "list" "[" type_expr "]"
            | "dict" "[" "text" "," type_expr "]"
            | "agent"
            | func_type

func_type ::= type_atom "->" type_expr
            | "(" type_list? ")" "->" type_expr
type_atom ::= "unit" | "text" | "json" | "bool" | "int" | "decimal"
            | name
            | name "[" type_expr ("," type_expr)* "]"
            | qual_prefix name "[" type_expr ("," type_expr)* "]"
            | qual_prefix name
            | "list" "[" type_expr "]"
            | "dict" "[" "text" "," type_expr "]"
            | "agent"
type_list ::= type_expr ("," type_expr)* ","?
```

`name "[" … "]"` is an applied type: a generic declaration instantiated at
concrete type arguments (`Box[int]`, `Outcome[int, text]`). The built-in
`list[T]` and `dict[text, V]` are the same form.

## Function declarations

```ebnf
func_def        ::= "def" name type_params? "(" param_list? ")" ("->" type_expr)? ("=" func_body | suite)
extern_func_def ::= "extern" "def" name type_params? "(" param_list? ")" "->" type_expr
func_body       ::= expr | suite
param_list      ::= param_entry ("," param_entry)* ","?
param_entry     ::= param | param_marker
param           ::= field_name ":" type_expr ("=" or_expr)?
```

An inline `def` body after `=` is exactly one expression. It ends at the next
block separator — a newline or `;` — which starts the next block item, so an
inline body admits neither a binder nor a `;` sequence. Parenthesize a
multi-item block or use a suite when the body needs binders or a sequence; for
suite bodies, the `=` before the newline is optional. The return type
annotation is optional for ordinary `def` declarations and required for
`builtin def` and `extern def` —
neither has a body. Zone markers (`/`, `*`,
`@pos`, `@std`, `@named`) may appear as `param_entry` items between parameters;
see [Functions](functions.md) for full zone semantics. No required
positional-fillable (pos-only/standard) parameter may follow a defaulted one
in the same zone. An optional `type_params` list after the function name makes
the `def` generic (e.g. `def id[T](x: T) -> T`); see [Generics](generics.md).

`extern_func_def` shares this signature surface with `func_def` but is never
followed by a body; it declares a function implemented by a companion Python
file (see [Python FFI](ffi.md)) rather than an AgL expression.

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
the `+`/`-` level. `prio <op> +/- <int>` is resolved from an existing builtin or
previously declared user operator.

## Bindings and mutation

```ebnf
let_decl ::= "let" name (":" type_expr)? "=" expr
var_decl ::= "var" name (":" type_expr)? "=" expr
builtin_var_def ::= "builtin" "var" name ":" type_expr  (* body-less; std/config only *)
assign_stmt ::= assign_target ":=" expr
assign_target ::= name ("[" expr "]")*
                | qual_prefix name
```

A `builtin var` is a body-less, host-backed mutable binding with a mandatory
type and no initializer; the `builtin` modifier may sit on the same line or the
line directly above (like `builtin def`). It may be declared only at the root of
`std/config`; entry modules and other library modules cannot declare one.

Assignment has type `unit` and returns `void`. A cross-module assignment target
— written with a qualifier, or bare when an open import puts the name in scope —
is valid only when it resolves to a `builtin var`; type-qualified constructor
forms are not assignment targets. In an indexed assignment target,
each opening `[` must be adjacent to the target name or preceding index:
`xs[0]` is indexed assignment, while `xs [0]` is not.

## Loops

```ebnf
loop        ::= for_clause? while_clause? "do" loop_bound?
                (suite loop_end? | inline_body loop_end)
for_clause  ::= "for" name "in" or_expr range_tail? NEWLINE?
range_tail  ::= ("to" | "downto") or_expr ("by" or_expr)?
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
`continue` are nullary expressions of the bottom type, valid only lexically
inside a loop body within the same function/lambda. `return` may also appear
inside a loop; it exits the nearest enclosing function. See
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
case_expr    ::= "case" or_expr "of" "|"? case_branch ("|" case_branch)*
case_branch  ::= pattern "=>" branch_body
```

## `try` / `catch`

```ebnf
try_expr          ::= "try" try_body catch_clause+
try_body          ::= suite | (marked_item ";")* try_tail
try_tail          ::= or_expr | inline_assign | try_letvar_decl | raise_expr
                    | return_expr | if_expr | case_expr | loop_expr
try_letvar_decl   ::= ("let" | "var") name type_ann? "=" try_value
try_value         ::= or_expr | raise_expr | return_expr | if_expr | case_expr | loop_expr
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
                 | qual_prefix type_qual? name ("(" pattern_fields? ")")?
qual_prefix    ::= ["/"] NAME ("/" NAME)* "::" | "::"    (* byte-adjacent throughout *)
type_qual      ::= name "::"
pattern_fields ::= pattern_field ("," pattern_field)* ","?
pattern_field  ::= pattern              (* positional sub-pattern *)
                 | field_name "=" pattern
                                           (* named sub-pattern: field = subpattern *)
```

`pattern as name` binds `name` to the complete value matched by `pattern`.
It has the lowest pattern precedence, may be chained, and cannot use `_` as
its binder name. The binder is always a variable binder.

A qualified variant pattern (`Option::some(value)`,
`module::Option::some(value)`, or `/module::Option::some(value)`) names the
owning enum and variant with `::`. A leading `/` is an anchored qualifier;
without it, the qualifier is resolved as a suffix. The complete qualifier
through `::` is byte-adjacent.
Unqualified constructor ownership is selected by the scrutinee's static enum
type, even when multiple enums share the variant name; a qualifier is optional
and must agree with that type when present ([Generics](generics.md),
[Pattern matching](pattern-matching.md)). Type arguments are carried by the
scrutinee type rather than written in a pattern.

In a constructor pattern, **positional sub-patterns** (`pattern` without a
`name "="` prefix) fill positional-capable (pos-only/standard) constructor fields
left to right. Named sub-patterns follow. A bare name positional sub-pattern
that lands on a named-only field is reinterpreted as the shorthand `name = name`.

A `STRING` pattern may not contain interpolation.

## Expressions

```ebnf
expr      ::= case_expr | if_expr | loop | try_expr | raise_expr
            | return_expr | lambda_expr | or_expr

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

juxt_arg       ::= atom_no_call juxt_suffix*
juxt_suffix    ::= "." field_name
               | "[" expr "]"                  (* adjacent bracket only *)
               | "(" arg_list? ")"
               | "::" "[" type_expr ("," type_expr)* "]" "(" arg_list? ")"

postfix        ::= postfix "." field_name          (* runtime field access *)
               | postfix "(" arg_list? ")"         (* call with parentheses *)
               | postfix "[" expr "]"              (* adjacent bracket only *)
               | postfix "::" "[" type_expr ("," type_expr)* "]"   (* explicit type application *)
               | atom

applied_type_qualified_constructor ::= qual_prefix? NAME "[" type_expr ("," type_expr)* "]" "::" NAME
                                           (* `[` is byte-adjacent to the preceding NAME *)

atom           ::= INT | DECIMAL | "true" | "false" | "null"
               | "(" ")"                           (* unit literal *)
               | list_literal
               | dict_literal
               | name                              (* variable / constructor reference *)
               | qual_prefix type_qual? name       (* qualified ref / constructor *)
               | applied_type_qualified_constructor
               | template
               | "(" expr ")"                      (* parenthesized expr *)
               | "(" paren_block ")"               (* parenthesized block *)

paren_block    ::= (marked_item ";")+ marked_item
                 | inline_assign
                 | let_decl | var_decl

atom_no_call   ::= (* same as atom but excludes "(" — prevents sugar conflict *)
               INT | DECIMAL | "true" | "false" | "null"
               | list_literal | dict_literal | NAME
               | template

qualified_constructor ::= qual_prefix type_qual? name | name

raise_expr  ::= "raise" or_expr
return_expr ::= "return" or_expr?
```

A bare name atom is resolved by scope and position: it may name a variable,
an agent, a record constructor, an enum variant, or a generic `def`/constructor
used as a first-class value. The typed postfix form carries explicit type
arguments to a generic `def` or bare constructor (`id::[int](5)`,
`some::[int](value = 1)`, `apply::[int, int](…)`), or instantiate a generic
function value (`id::[int]`). Qualified generic constructors put type arguments
on the type side (`Option[int]::some(value = 1)`). In that explicit
type-qualified constructor form, the `[` is byte-adjacent to the applied type
name: `Option[int]::some` is valid, while `Option [int]::some` is not. This
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
arg             ::= expr                         (* positional *)
                  | placeholder_arg              (* positional hole *)
                  | field_name "=" expr          (* named *)
                  | field_name "=" placeholder_arg (* named hole *)
placeholder_arg ::= "?" | "?<digits>"
```

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
list_literal ::= "[" (expr ("," expr)* ","?)? "]"

dict_literal ::= "{" (dict_entry ("," dict_entry)* ","?)? "}"
dict_entry   ::= STRING ":" expr        (* no interpolation in keys *)
               | field_name ":" expr    (* shorthand for the string key *)
```

## Templates

```ebnf
template      ::= '"' (text_fragment | interpolation)* '"'
                | '"""' (text_fragment | interpolation)* '"""'

interpolation ::= "${" expr "}"
```

Newlines are not permitted inside `${…}`. Triple-quoted templates are
dedented as described in [Lexical structure](lexical-structure.md).

## Deterministic-parse notes

- `==` is the equality operator; a single `=` is reserved for bindings, named
  arguments, and declarations, and is not an expression operator.
- Chained comparisons are rejected with a non-associativity message after the
  infix chain is grouped.
- The `[N]` after `do` is a single lexical unit, so it never conflicts with
  a list literal.
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
  nor the *open* forms (`case`, `if`, `try`, a loop) or a lambda, whose body is
  itself an `expr`: those would extend rightwards into the enclosing branch
  list's `|` / `else` / `catch`. Write them as a suite or parenthesize them
  (see [Inline bodies](#inline-bodies)).
- `until` conditions reference `or_expr` directly, so an open form there must
  be parenthesized.
- Bodies whose end is marked by a token — a parenthesized block, a loop body,
  a `try` body — take a full `;` sequence; loop bodies additionally admit the
  open forms, because the loop terminator closes the body.
- A `return` followed by a newline is a bare `return`; its operand does not
  continue onto the next line.
