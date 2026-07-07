# Grammar

[← Index](index.md)

The collected surface grammar. Lexical tokens (`NEWLINE`, `INDENT`,
`DEDENT`, identifier terminals `NAME`/`OP_NAME`, numbers, template tokens)
and the layout rules that produce them are specified in
[Lexical structure](lexical-structure.md).

Notation: `::=` defines a production; `|` separates alternatives; `?`, `*`,
`+` mark optional and repeated elements; quoted strings are literal tokens.

## Programs and blocks

```ebnf
name       ::= NAME | OP_NAME

program    ::= block EOF

block      ::= item ((NEWLINE | ";") item)* (NEWLINE | ";")?

item       ::= import_decl                  (* header position only *)
             | config_decl                  (* root only *)
             | modifier? record_def         (* root only *)
             | modifier? enum_def           (* root only *)
             | modifier? type_alias         (* root only; "builtin" not allowed *)
             | param_decl                   (* root only *)
             | program_decl                 (* root only *)
             | agent_decl                   (* root only *)
             | infix_decl                   (* root only *)
             | modifier? func_def           (* root only *)
             | ("private" NEWLINE?)? extern_func_def  (* root only; file-backed modules only *)
             | let_decl | var_decl
             | expr

modifier   ::= ("private" | "builtin") NEWLINE?
```

A block's value is its last item. A `let_decl` or `var_decl` as the final
item (with no continuation) is a static error.

`"private"` and `"builtin"` are **declaration modifiers** that behave like
decorators: a modifier may sit on the same line as the declaration it adorns
(`builtin enum …`) or on the line directly above it (`builtin` then `enum …`)
— the newline after the modifier is insignificant. `"builtin"` is not allowed
on a `type_alias` or on `extern_func_def`; `"private"` composes with
`extern_func_def` the same way it composes with `func_def`.

## Import declarations

```ebnf
import_decl ::= "import" module_path ("." "*")? "qualified"?
                ("as" ref_name)?
                (using_clause | hiding_clause)?

module_path ::= NAME ("." NAME)*      (* all lower-case segments *)
ref_name    ::= name

using_clause  ::= "using" import_item ("," import_item)*
hiding_clause ::= "hiding" ref_name ("," ref_name)*
import_item   ::= ref_name ("as" ref_name)?
```

`"import"`, `"qualified"`, `"using"`, and `"hiding"` are contextual soft
keywords — they remain valid identifiers outside import lines.

`"private"` is a contextual soft keyword at item-start — it remains a valid
identifier elsewhere.

Examples:

```agl
import foo.bar
import foo.bar as A
import foo.bar qualified
import foo.bar qualified as A
import foo.bar using x, y
import foo.bar hiding x, y
import foo.bar using x as X, y
import foo.*
import foo.bar.* as A
```

### Suites (indented blocks)

```ebnf
suite ::= NEWLINE INDENT block DEDENT
```

## Type declarations

```ebnf
record_def       ::= "record" name type_params? "="? record_body
record_body      ::= param_marker? NEWLINE INDENT block_entry (NEWLINE block_entry)* NEWLINE? DEDENT
                   | "(" field_list? ")"
                   | field_list
block_entry      ::= field_def | param_marker
field_def        ::= name ":" type_expr

enum_def         ::= "enum" name type_params? "="? enum_body
enum_body        ::= enum_variant_seq
                   | NEWLINE INDENT enum_variant_seq NEWLINE? DEDENT
enum_variant_seq ::= first_variant_def ("|" variant_def)*
first_variant_def ::= "|"? name variant_payload?
variant_def      ::= name variant_payload?
variant_payload  ::= "(" field_list? ")"
field_list       ::= field_entry ("," field_entry)* ","?
field_entry      ::= field_inline | param_marker
field_inline     ::= name ":" type_expr

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
type_list ::= type_expr ("," type_expr)* ","?
```

`NAME "[" … "]"` is an applied type: a generic declaration instantiated at
concrete type arguments (`Box[int]`, `Outcome[int, text]`). The built-in
`list[T]` and `dict[text, V]` are the same form.

## Function declarations

```ebnf
func_def        ::= "def" name type_params? "(" param_list? ")" ("->" type_expr)? ("=" func_body | suite)
extern_func_def ::= "extern" "def" name type_params? "(" param_list? ")" "->" type_expr
func_body       ::= expr | suite | inline_binder_block
inline_binder_block ::= (let_decl | var_decl | assign_expr)
                        (";" (let_decl | var_decl | assign_expr))* ";" expr
param_list      ::= param_entry ("," param_entry)* ","?
param_entry     ::= param | param_marker
param           ::= name ":" type_expr ("=" expr)?
```

The `def` body is a single expression (which may be an indented `suite` or a
compact one-line block starting with `let`, `var`, or assignment); for suite
bodies, the `=` before the newline is optional. The return type
annotation is optional for ordinary `def` declarations and required for
`builtin def` and `extern def` — neither has a body. Zone markers (`/`, `*`,
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
assign_expr ::= assign_target ":=" expr
assign_target ::= name ("[" expr "]")*
```

Assignment has type `unit` and returns `void`. In an indexed assignment target,
each opening `[` must be adjacent to the target name or preceding index:
`xs[0]` is indexed assignment, while `xs [0]` is not.

## Loops

```ebnf
loop        ::= for_clause? while_clause? "do" loop_bound? body loop_end
for_clause  ::= "for" name "in" or_expr range_tail? _NL?
range_tail  ::= ("to" | "downto") or_expr ("by" or_expr)?
while_clause::= "while" or_expr _NL?
loop_bound  ::= "[" or_expr "]"           (* int; n <= 0 runs zero iterations *)
body        ::= suite | inline_body
loop_end    ::= "until" or_expr | "done"   (* omitted terminator allowed in suite form *)
inline_body ::= item (";" item)*
```

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
if_cond_branch ::= or_expr "=>" (suite | or_expr)
if_else_branch ::= "|"? "else" "=>" (suite | or_expr)
```

Without an `else` branch the `if` expression has type `unit` and returns
`void`. With all branches returning a common type `T`, the `if` expression has
type `T`.

## `case`

```ebnf
case_expr    ::= "case" expr "of" "|"? case_branch ("|" case_branch)*
case_branch  ::= pattern "=>" (suite | or_expr)
```

## `try` / `catch`

```ebnf
try_expr      ::= "try" (suite | inline_body) catch_clause+
catch_clause  ::= "catch" catch_pattern "=>" (suite | or_expr)
catch_pattern ::= name ("as" name)?
                | "_" ("as" name)?
```

## Patterns

```ebnf
pattern        ::= "_"
                 | INT | DECIMAL | "true" | "false" | "null" | STRING
                 | name
                 | constructor_pattern
                 | qual_constructor_pattern

constructor_pattern      ::= name ("(" pattern_fields? ")")?
qualified_type           ::= name ("[" type_expr ("," type_expr)* "]")?
qual_constructor_pattern ::= qual_prefix? qualified_type "::" name
                             ("(" pattern_fields? ")")?
pattern_fields ::= pattern_field ("," pattern_field)* ","?
pattern_field  ::= pattern              (* positional sub-pattern *)
                 | name "=" pattern     (* named sub-pattern: field = subpattern *)
```

A `qual_constructor_pattern` is a **qualified** variant pattern
(`Option::some(value)` or `module::Option::some(value)`), naming the owning enum
and the variant; it is required when an unqualified variant name is ambiguous
across enums ([Generics](generics.md), [Pattern matching](pattern-matching.md)).

In a constructor pattern, **positional sub-patterns** (`pattern` without a
`name "="` prefix) fill positional-capable (pos-only/standard) constructor fields
left to right. Named sub-patterns follow. A bare name positional sub-pattern
that lands on a named-only field is reinterpreted as the shorthand `name = name`.

A `STRING` pattern may not contain interpolation.

## Expressions

```ebnf
expr      ::= case_expr | if_expr | infix_expr

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
juxt_suffix    ::= "." name
               | "[" expr "]"                  (* adjacent bracket only *)
               | "(" arg_list? ")"
               | "::" "[" type_expr ("," type_expr)* "]" "(" arg_list? ")"

postfix        ::= postfix "." name                (* runtime field access *)
               | postfix "(" arg_list? ")"         (* call with parentheses *)
               | postfix "[" expr "]"              (* adjacent bracket only *)
               | postfix "::" "[" type_expr ("," type_expr)* "]" "(" arg_list? ")"
               | atom

qual_prefix    ::= NAME ("." NAME)* "::"   (* dotted module path followed by :: *)
               | "::"                      (* self-reference: current module *)

atom           ::= INT | DECIMAL | "true" | "false" | "null"
               | "(" ")"                           (* unit literal *)
               | list_literal
               | dict_literal
               | name                              (* variable / constructor reference *)
               | qual_prefix name                  (* module-qualified ref *)
               | qual_prefix? qualified_type "::" name
                                                    (* type-qualified constructor ref *)
               | template
               | "(" expr ")"                      (* parenthesized expr *)
               | lambda_expr
               | assign_expr
               | do_until
               | raise_expr
               | return_expr

atom_no_call   ::= (* same as atom but excludes "(" — prevents sugar conflict *)
               INT | DECIMAL | "true" | "false" | "null"
               | list_literal | dict_literal | name
               | template

qualified_constructor ::= qual_prefix? qualified_type "::" name
                        | name

raise_expr  ::= "raise" expr
return_expr ::= "return" or_expr?
```

A bare name atom is resolved by scope and position: it may name a variable,
an agent, a record constructor, an enum variant, or a generic `def`/constructor
used as a first-class value. The typed postfix form carries explicit type
arguments to a generic `def` or bare constructor (`id::[int](5)`,
`some::[int](value = 1)`, `apply::[int, int](…)`), or instantiate a generic
function value (`id::[int]`). Qualified generic constructors put type arguments
on the type side (`Option[int]::some(value = 1)`). A `postfix "." name` is
always runtime field access; constructor qualification uses `::`. See
[Generics](generics.md).

## Lambda expressions

```ebnf
lambda_expr ::= "fn" "(" params? ")" ("->" type_expr)? "=>" expr
```

The return type annotation is optional; when omitted, it is inferred from
the body. Parameter types are always required.

## Calls

```ebnf
arg_list  ::= arg ("," arg)* ","?
arg       ::= expr                  (* positional *)
            | name "=" expr         (* named *)
```

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
               | name ":" expr          (* shorthand for the string key *)
```

## Templates

```ebnf
template      ::= '"' (text_fragment | interpolation)* '"'
                | '"""' (text_fragment | interpolation)* '"""'

interpolation ::= "${" expr "}"
```

Newlines are not permitted inside `${…}`. Triple-quoted templates are
dedented as described in [Lexical structure](lexical-structure.md).

## Config declarations

```ebnf
config_decl ::= "config" name ("=" expr)?
```

The value expression is optional (a bare `config name` resolves from the host
default) and must have the engine-key's declared type; see
[Program structure](program-structure.md).

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
- `juxt` is a **concrete** (non-transparent) rule so that application does
  not cascade shift/reduce conflicts into every operator rule.
- `()` is both the unit literal and the empty argument list of a zero-arg
  call — the two are syntactically unified.
- Branch bodies, loop inline bodies, and `until` conditions reference `or_expr`
  directly; a `case`, `if`, `raise`, or `return` expression in those positions
  must use a suite form or be parenthesized.
- A `return` followed by a newline is a bare `return`; its operand does not
  continue onto the next line.
