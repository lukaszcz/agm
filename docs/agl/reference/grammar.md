# Grammar

[← Index](index.md)

The collected surface grammar. Lexical tokens (`NEWLINE`, `INDENT`,
`DEDENT`, identifiers, numbers, template tokens) and the layout rules that
produce them are specified in [Lexical structure](lexical-structure.md).

Notation: `::=` defines a production; `|` separates alternatives; `?`, `*`,
`+` mark optional and repeated elements; quoted strings are literal tokens.

## Programs and blocks

```ebnf
program    ::= block EOF

block      ::= item ((NEWLINE | ";") item)* (NEWLINE | ";")?

item       ::= config_pragma                (* header position only *)
             | record_def | enum_def | type_alias  (* root only *)
             | param_decl                    (* root only *)
             | program_decl                  (* root only *)
             | agent_decl                    (* root only *)
             | func_def                      (* root only *)
             | let_decl | var_decl
             | expr
```

A block's value is its last item. A `let_decl` or `var_decl` as the final
item (with no continuation) is a static error.

### Suites (indented blocks)

```ebnf
suite ::= NEWLINE INDENT block DEDENT
```

## Type declarations

```ebnf
record_def      ::= "record" TYPE_NAME NEWLINE INDENT field_def
                    (NEWLINE field_def)* NEWLINE? DEDENT
field_def       ::= VAR_NAME ":" type_expr

enum_def        ::= "enum" TYPE_NAME variant_def+
variant_def     ::= "|" TYPE_NAME variant_payload?
variant_payload ::= "(" field_list? ")"
field_list      ::= field_inline ("," field_inline)* ","?
field_inline    ::= VAR_NAME ":" type_expr

type_alias      ::= "type" TYPE_NAME "=" type_expr

param_decl      ::= "param" VAR_NAME (":" type_expr)? ("=" expr)?
program_decl    ::= "program" VAR_NAME

agent_decl      ::= "agent" VAR_NAME ("=" STRING)?
```

The runner string of an `agent` declaration must be a literal string with no
`${…}` interpolation; an interpolation hole is a static error.

## Type expressions

```ebnf
type_expr ::= "unit"
            | "text" | "json" | "bool" | "int" | "decimal"
            | TYPE_NAME
            | "list" "[" type_expr "]"
            | "dict" "[" "text" "," type_expr "]"
            | func_type

func_type ::= "(" type_list? ")" "->" type_expr
type_list ::= type_expr ("," type_expr)* ","?
```

## Function declarations

```ebnf
func_def   ::= "def" VAR_NAME "(" params? ")" "->" type_expr "=" expr
params     ::= param ("," param)* ","?
param      ::= VAR_NAME ":" type_expr ("=" expr)?
```

The `def` body is a single expression (which may be a `block`). Defaulted
parameters must follow all required parameters.

## Bindings and mutation

```ebnf
let_decl ::= "let" VAR_NAME (":" type_expr)? "=" expr
var_decl ::= "var" VAR_NAME (":" type_expr)? "=" expr
set_expr ::= "set" VAR_NAME "=" expr
```

`set` yields `unit`.

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
case_expr    ::= "case" expr "of" ("|" case_branch)+
case_branch  ::= pattern "=>" (suite | or_expr)
```

## `try` / `catch`

```ebnf
try_expr      ::= "try" (suite | inline_body) catch_clause+
catch_clause  ::= "catch" catch_pattern "=>" (suite | or_expr)
catch_pattern ::= TYPE_NAME ("as" VAR_NAME)?
                | "_" ("as" VAR_NAME)?
```

## Patterns

```ebnf
pattern        ::= "_"
                 | INT | DECIMAL | "true" | "false" | "null" | STRING
                 | VAR_NAME
                 | constructor_pattern

constructor_pattern ::= TYPE_NAME ("." TYPE_NAME)? ("(" pattern_fields? ")")?
pattern_fields ::= pattern_field ("," pattern_field)* ","?
pattern_field  ::= VAR_NAME
                 | VAR_NAME ":" pattern
```

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
cmp_op     ::= "=" | "!=" | "<" | "<=" | ">" | ">=" | "in"
              (* at most one comparison per expression: non-associative *)

additive       ::= multiplicative (("+" | "-") multiplicative)*
multiplicative ::= unary (("*" | "/") unary)*
unary          ::= "-" unary | juxt

juxt           ::= postfix juxt_arg     (* single-arg sugar; non-chaining *)
               | postfix

juxt_arg       ::= atom_no_call        (* excludes "("-led forms and calls *)

postfix        ::= postfix "." VAR_NAME            (* field access *)
               | postfix "." TYPE_NAME             (* variant qualification *)
               | postfix "(" arg_list? ")"         (* call with parentheses *)
               | atom

typed_call     ::= VAR_NAME "::" "[" type_expr "]" "(" arg_list? ")"  (* typed call *)

atom           ::= INT | DECIMAL | "true" | "false" | "null"
               | "(" ")"                           (* unit literal *)
               | list_literal
               | dict_literal
               | TYPE_NAME                         (* bare constructor *)
               | VAR_NAME                          (* variable reference *)
               | typed_call                        (* e.g. ask-request::[Review](…) *)
               | template
               | "(" expr ")"                      (* parenthesized expr *)
               | lambda_expr
               | set_expr
               | do_until
               | raise_expr

atom_no_call   ::= (* same as atom but excludes "(" — prevents sugar conflict *)
               INT | DECIMAL | "true" | "false" | "null"
               | list_literal | dict_literal | TYPE_NAME | VAR_NAME
               | template | postfix "." VAR_NAME | postfix "." TYPE_NAME

qualified_constructor ::= TYPE_NAME ("." TYPE_NAME)?

raise_expr ::= "raise" expr
```

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
            | VAR_NAME ":" expr     (* named *)
```

Named arguments are available at declared-name call sites (`def`s and
built-ins). A function value is called with positional arguments only.

## Literals

```ebnf
list_literal ::= "[" (expr ("," expr)* ","?)? "]"

dict_literal ::= "{" (dict_entry ("," dict_entry)* ","?)? "}"
dict_entry   ::= STRING ":" expr        (* no interpolation in keys *)
               | VAR_NAME ":" expr      (* shorthand for the string key *)
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
config_pragma ::= "config" VAR_NAME "=" config_value
config_value  ::= "true" | "false" | INT | DECIMAL | STRING
```

## Deterministic-parse notes

- `==` is not in the grammar; it is rejected with *"Use `=` for equality."*
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
