# Grammar

[← Index](index.md)

The collected surface grammar. Lexical tokens (`NEWLINE`, `INDENT`,
`DEDENT`, the single identifier terminal `NAME`, numbers, template tokens)
and the layout rules that produce them are specified in
[Lexical structure](lexical-structure.md).

Notation: `::=` defines a production; `|` separates alternatives; `?`, `*`,
`+` mark optional and repeated elements; quoted strings are literal tokens.

## Programs and blocks

```ebnf
program    ::= block EOF

block      ::= item ((NEWLINE | ";") item)* (NEWLINE | ";")?

item       ::= import_decl                  (* header position only *)
             | config_pragma                (* header position only *)
             | "private"? record_def        (* root only *)
             | "private"? enum_def          (* root only *)
             | "private"? type_alias        (* root only *)
             | param_decl                   (* root only *)
             | program_decl                 (* root only *)
             | agent_decl                   (* root only *)
             | "private"? func_def          (* root only *)
             | let_decl | var_decl
             | expr
```

A block's value is its last item. A `let_decl` or `var_decl` as the final
item (with no continuation) is a static error.

## Import declarations

```ebnf
import_decl ::= "import" module_path ("." "*")? "qualified"?
                ("as" ref_name)?
                (using_clause | hiding_clause)?

module_path ::= NAME ("." NAME)*      (* all lower-case segments *)
ref_name    ::= NAME                  (* any case *)

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
record_def      ::= "record" NAME type_params? record_body
record_body     ::= NEWLINE INDENT field_def (NEWLINE field_def)* NEWLINE? DEDENT
                  | "{" field_list? "}"
                  | field_list
field_def       ::= NAME ":" type_expr

enum_def        ::= "enum" NAME type_params? "="? enum_body
enum_body       ::= enum_variant_seq
                  | NEWLINE INDENT enum_variant_seq NEWLINE? DEDENT
enum_variant_seq ::= first_variant_def ("|" variant_def)*
first_variant_def ::= "|"? NAME variant_payload?
variant_def     ::= NAME variant_payload?
variant_payload ::= "(" field_list? ")"
field_list      ::= field_inline ("," field_inline)* ","?
field_inline    ::= NAME ":" type_expr

type_alias      ::= "type" NAME type_params? "=" type_expr

type_params     ::= "[" NAME ("," NAME)* "]"

param_decl      ::= "param" NAME (":" type_expr)? ("=" expr)?
program_decl    ::= "program" NAME

agent_decl      ::= "agent" NAME ("=" STRING)?
```

A `type_params` list declares the declaration's type parameters; each is an
ordinary `NAME` in scope as a type throughout the declaration's body. See
[Generics](generics.md).

The runner string of an `agent` declaration must be a literal string with no
`${…}` interpolation; an interpolation hole is a static error.

## Type expressions

```ebnf
type_expr ::= "unit"
            | "text" | "json" | "bool" | "int" | "decimal"
            | NAME
            | NAME "[" type_expr ("," type_expr)* "]"   (* applied type *)
            | qual_prefix NAME "[" type_expr ("," type_expr)* "]"
            | qual_prefix NAME         (* module-qualified type *)
            | "list" "[" type_expr "]"
            | "dict" "[" "text" "," type_expr "]"
            | func_type

func_type ::= "(" type_list? ")" "->" type_expr
type_list ::= type_expr ("," type_expr)* ","?
```

`NAME "[" … "]"` is an applied type: a generic declaration instantiated at
concrete type arguments (`Box[int]`, `Outcome[int, text]`). The built-in
`list[T]` and `dict[text, V]` are the same form.

## Function declarations

```ebnf
func_def   ::= "def" NAME type_params? "(" params? ")" "->" type_expr "=" expr
params     ::= param ("," param)* ","?
param      ::= NAME ":" type_expr ("=" expr)?
```

The `def` body is a single expression (which may be a `block`). Defaulted
parameters must follow all required parameters. An optional `type_params` list
after the function name makes the `def` generic (e.g. `def id[T](x: T) -> T`);
see [Generics](generics.md).

## Bindings and mutation

```ebnf
let_decl ::= "let" NAME (":" type_expr)? "=" expr
var_decl ::= "var" NAME (":" type_expr)? "=" expr
assign_expr ::= assign_target ":=" expr
assign_target ::= NAME ("[" expr "]")*
```

Assignment yields `unit`. In an indexed assignment target, each opening `[` must be
adjacent to the target name or preceding index: `xs[0]` is indexed assignment,
while `xs [0]` is not.

## Loops

```ebnf
do_until   ::= "do" loop_bound? suite "until" or_expr
             | "do" loop_bound? inline_body "until" or_expr

loop_bound ::= "[" INT "]"            (* INT must be positive *)

inline_body ::= item (";" item)*
```

## `if`

```ebnf
if_expr        ::= "if" "|"? if_cond_branch ("|" if_cond_branch)* if_else_branch?
if_cond_branch ::= or_expr "=>" (suite | or_expr)
if_else_branch ::= "|"? "else" "=>" (suite | or_expr)
```

Without an `else` branch the `if` expression has type `unit`. With all
branches returning a common type `T`, the `if` expression has type `T`.

## `case`

```ebnf
case_expr    ::= "case" expr "of" "|"? case_branch ("|" case_branch)*
case_branch  ::= pattern "=>" (suite | or_expr)
```

## `try` / `catch`

```ebnf
try_expr      ::= "try" (suite | inline_body) catch_clause+
catch_clause  ::= "catch" catch_pattern "=>" (suite | or_expr)
catch_pattern ::= NAME ("as" NAME)?
                | "_" ("as" NAME)?
```

## Patterns

```ebnf
pattern        ::= "_"
                 | INT | DECIMAL | "true" | "false" | "null" | STRING
                 | NAME
                 | constructor_pattern
                 | qual_constructor_pattern

constructor_pattern      ::= NAME ("." NAME)? ("(" pattern_fields? ")")?
qual_constructor_pattern ::= qual_prefix NAME ("." NAME)? ("(" pattern_fields? ")")?
pattern_fields ::= pattern_field ("," pattern_field)* ","?
pattern_field  ::= NAME                 (* shorthand: NAME bound to NAME *)
                 | NAME "=" pattern
```

A `constructor_pattern` with the `NAME "." NAME` form is a **qualified**
variant pattern (`Option.some(value)`), naming the owning enum and the variant;
it is required when an unqualified variant name is ambiguous across enums
([Generics](generics.md), [Pattern matching](pattern-matching.md)).

A `STRING` pattern may not contain interpolation.

## Expressions

```ebnf
expr      ::= case_expr | if_expr | or_expr

or_expr   ::= and_expr ("or" and_expr)*
and_expr  ::= not_expr ("and" not_expr)*
not_expr  ::= "not" not_expr | comparison

comparison ::= additive comparison_tail?
comparison_tail
           ::= "is" "not"? qualified_constructor
            | cmp_op additive
cmp_op     ::= "==" | "!=" | "<" | "<=" | ">" | ">=" | "in"
              (* at most one comparison per expression: non-associative *)

additive       ::= multiplicative (("+" | "-") multiplicative)*
multiplicative ::= cast (("*" | "/") cast)*

cast           ::= cast "as" type_expr      (* type cast; may raise CastError *)
               | cast "as?" type_expr      (* convertibility test; yields bool *)
               | unary
               (* left-associative; "as?" is a single token — no whitespace *)

unary          ::= "-" unary | juxt

juxt           ::= postfix juxt_arg     (* single-arg sugar; non-chaining *)
               | postfix

juxt_arg       ::= atom_no_call juxt_suffix*
juxt_suffix    ::= "." NAME
               | "[" expr "]"                  (* adjacent bracket only *)
               | "(" arg_list? ")"
               | "::" "[" type_expr ("," type_expr)* "]" "(" arg_list? ")"

postfix        ::= postfix "." NAME                (* field access OR variant qualification *)
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
               | NAME                              (* variable / constructor reference *)
               | qual_prefix NAME                  (* module-qualified ref *)
               | template
               | "(" expr ")"                      (* parenthesized expr *)
               | lambda_expr
               | assign_expr
               | do_until
               | raise_expr

atom_no_call   ::= (* same as atom but excludes "(" — prevents sugar conflict *)
               INT | DECIMAL | "true" | "false" | "null"
               | list_literal | dict_literal | NAME
               | template

qualified_constructor ::= NAME ("." NAME)?

raise_expr ::= "raise" expr
```

A bare `NAME` atom is resolved by scope and position: it may name a variable,
an agent, a record constructor, an enum variant, or a generic `def`/constructor
used as a first-class value. The typed postfix form carries explicit type
arguments to a generic `def` or constructor (`id::[int](5)`,
`some::[int](value = 1)`, `Option.some::[int](value = 1)`,
`apply::[int, int](…)`); see [Generics](generics.md).
A `postfix "." NAME` is either a field access or a variant qualification,
disambiguated by the resolved type of the left operand.

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
            | NAME "=" expr         (* named *)
```

Named arguments are available at declared-name call sites (`def`s and
built-ins). A function value is called with positional arguments only.
Constructor calls additionally accept a bare `NAME` as shorthand for
`NAME = NAME`; ordinary `def` calls do not.

## Literals

```ebnf
list_literal ::= "[" (expr ("," expr)* ","?)? "]"

dict_literal ::= "{" (dict_entry ("," dict_entry)* ","?)? "}"
dict_entry   ::= STRING ":" expr        (* no interpolation in keys *)
               | NAME ":" expr          (* shorthand for the string key *)
```

## Templates

```ebnf
template      ::= '"' (text_fragment | interpolation)* '"'
                | '"""' (text_fragment | interpolation)* '"""'

interpolation ::= "${" expr "}"
```

Newlines are not permitted inside `${…}`. Triple-quoted templates are
dedented as described in [Lexical structure](lexical-structure.md).

## Config pragmas

```ebnf
config_pragma ::= "config" NAME "=" config_value
config_value  ::= "true" | "false" | INT | DECIMAL | STRING
```

## Deterministic-parse notes

- `==` is the equality operator; a single `=` is reserved for bindings, named
  arguments, and declarations, and is not an expression operator.
- Chained comparisons are unparseable by construction (one optional
  comparison tail) and rejected with a non-associativity message.
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
- The `bar_safe` stratification of v1 is **removed**. Branch bodies and
  `until` conditions reference `or_expr` directly; a `case` or `if`
  expression in those positions must be parenthesized.
