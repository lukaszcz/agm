# Grammar

[← Index](index.md)

The collected surface grammar. Lexical tokens (`NEWLINE`, `INDENT`,
`DEDENT`, identifiers, numbers, template tokens) and the layout rules that
produce them are specified in [Lexical structure](lexical-structure.md);
the inline-form restrictions encoded by `do_body`, `try_body`,
`branch_body`, `catch_body`, and `bar_expr` are explained in
[Program structure](program-structure.md).

Notation: `::=` defines a production; `|` separates alternatives; `?`, `*`,
`+` mark optional and repeated elements; quoted strings are literal tokens.

## Program and statements

```ebnf
program     ::= stmt_list EOF

stmt_list   ::= stmt (stmt_sep stmt)* stmt_sep?
stmt_sep    ::= NEWLINE | ";"

stmt        ::= closed_stmt | open_stmt

closed_stmt ::= record_def
              | enum_def
              | type_alias
              | input_decl
              | agent_decl
              | let_decl
              | var_decl
              | set_stmt
              | pass_stmt
              | print_stmt
              | raise_stmt
              | expr_stmt
              | do_until

open_stmt   ::= if_stmt | case_stmt | try_stmt

pass_stmt   ::= "pass"
print_stmt  ::= "print" expr
raise_stmt  ::= "raise" expr
expr_stmt   ::= bar_expr            (* a statement-level bare case is a case_stmt *)
```

## Blocks and bodies

```ebnf
suite        ::= NEWLINE INDENT stmt_list DEDENT

do_body      ::= suite
               | closed_stmt (";" closed_stmt)* (";" open_stmt)?
               | open_stmt

try_body     ::= suite
               | closed_stmt (";" closed_stmt)*

branch_body  ::= suite | bar_safe_stmt | try_stmt
catch_body   ::= suite | bar_safe_stmt
```

A *bar-safe statement* is any closed-statement form whose trailing
expression slots (`let`/`var`/`set` initializers, `print` and `raise`
operands) contain only bar-safe expressions (`bar_expr`).

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

input_decl      ::= "input" VAR_NAME (":" type_expr)?

agent_decl      ::= "agent" VAR_NAME ("=" STRING)?
```

The runner string of an `agent` declaration must be a literal string with no
`${…}` interpolation; an interpolation hole is a static error.

## Type expressions

```ebnf
type_expr ::= "text" | "json" | "bool" | "int" | "decimal"
            | TYPE_NAME
            | "list" "[" type_expr "]"
            | "dict" "[" "text" "," type_expr "]"
```

## Bindings and mutation

```ebnf
let_decl ::= "let" VAR_NAME (":" type_expr)? "=" expr
var_decl ::= "var" VAR_NAME (":" type_expr)? "=" expr
set_stmt ::= "set" VAR_NAME "=" expr
```

## Loops

```ebnf
do_until   ::= "do" loop_bound? do_body "until" bar_expr
loop_bound ::= "[" INT "]"            (* INT must be positive *)
```

## `if`

```ebnf
if_stmt   ::= "if" if_branch ("|" if_branch)*
if_branch ::= bar_expr "=>" branch_body
            | "else" "=>" branch_body   (* must be last *)
```

## `case`

```ebnf
case_stmt        ::= "case" expr "of" ("|" case_stmt_branch)+
case_stmt_branch ::= pattern "=>" branch_body

case_expr        ::= "case" expr "of" ("|" case_expr_branch)+
case_expr_branch ::= pattern "=>" bar_expr
```

## `try` / `catch`

```ebnf
try_stmt      ::= "try" try_body catch_clause+
catch_clause  ::= "catch" catch_pattern "=>" catch_body
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
expr      ::= case_expr | bar_expr
bar_expr  ::= or_expr               (* case_expr re-enters via "(" expr ")" *)

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
unary          ::= "-" unary | access

access ::= access "." VAR_NAME          (* field access *)
         | access "." TYPE_NAME         (* variant qualification *)
         | access constructor_payload   (* constructor application *)
         | atom

atom   ::= INT | DECIMAL | "true" | "false" | "null"
         | list_literal
         | dict_literal
         | TYPE_NAME                    (* bare constructor *)
         | VAR_NAME                     (* variable reference *)
         | agent_call
         | template
         | "(" expr ")"

qualified_constructor ::= TYPE_NAME ("." TYPE_NAME)?
constructor_payload   ::= "(" constructor_args? ")"
constructor_args      ::= named_arg ("," named_arg)* ","?
named_arg             ::= VAR_NAME ":" expr
```

## Agent calls

```ebnf
agent_call   ::= VAR_NAME call_options? template
                 (* VAR_NAME includes the contextual names prompt and exec *)

call_options ::= "[" call_option ("," call_option)* ","? "]"
call_option  ::= "format" ":" format_name
               | "strict_json" ":" ("true" | "false")
               | "on_parse_error" ":" parse_policy

format_name  ::= "text" | "json" | VAR_NAME
parse_policy ::= "abort" | "retry" "[" INT "]"
```

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

interpolation ::= "${" expr ("as" VAR_NAME)? "}"
```

Newlines are not permitted inside `${…}`. Triple-quoted templates are
dedented as described in [Lexical structure](lexical-structure.md).

## Deterministic-parse notes

- `==` is not in the grammar; it is rejected with *"Use `=` for equality."*
- Chained comparisons are unparseable by construction (one optional
  comparison tail) and rejected with a targeted non-associativity message.
- The `[N]` after `do` is a single lexical unit, so it never conflicts with
  a list literal beginning a `do` body.
- A `|`, `catch`, or `until` at the start of a line attaches to the
  innermost construct that can accept it; the layout rules guarantee each
  such token belongs to exactly one construct
  ([Lexical structure](lexical-structure.md),
  [Program structure](program-structure.md)).
