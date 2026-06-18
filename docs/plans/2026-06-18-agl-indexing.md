# Plan: AgL list and dict indexing

## Overview

AgL needs square-bracket indexing for lists and dictionaries:

```agl
let third = l[2]
let name = d["a"]
```

The hard grammar requirement is that this must stay conflict-free while still
distinguishing:

```agl
l[2]     # indexing
f [2]    # single-argument sugar: f([2])
```

The existing grammar already has the relevant pressure point:

- postfix syntax currently owns parenthesized calls and field access:
  `postfix "(" ... ")"` and `postfix "." field`;
- juxtaposition sugar is one grammar level lower: `postfix juxt_arg`;
- list literals are valid `juxt_arg`s, and this is tested by
  `print [1,2,3]`;
- square brackets are also used by type syntax, typed calls, list literals, and
  the lexer-level `do[N]` loop-bound merge.

The implementation should therefore avoid adding a plain
`postfix "[" expr "]"` rule against the same `LSQB` token that starts list
literals in juxtaposition. The selected shape is a lexer-assisted split:
emit a distinct token for an adjacent indexing bracket, and leave spaced
brackets as ordinary list literals.

## Resolved Owner Decisions

These decisions were resolved with the owner and frame the implementation.

| # | Decision |
|---|----------|
| D1 | Distinguish indexing from list-literal juxtaposition by adjacency: `l[2]` indexes, while `f [2]` remains `f([2])`. |
| D2 | Invalid list indexes and missing dict keys raise catchable AgL exceptions. |
| D3 | Support Python-style negative list indexes. |
| D4 | Dictionary indexes are `text` only, matching `dict[text, V]`. |
| D5 | Add indexed assignment, but only for lists/dicts declared as `var`; function arguments and non-`var` bindings remain immutable. |
| D6 | Indexing is a postfix operator with chaining, like calls and field access. |

### D1 — Adjacency distinguishes indexing from list-literal juxtaposition

Indexing is whitespace-sensitive:

- `expr[` with no whitespace/newline between the expression and `[` is indexing;
- `expr [` with whitespace keeps the existing single-argument call sugar when
  the right side is a list literal.

Implementation: have the lexer remap an adjacent `LSQB` after an
expression-ending token to the new terminal `INDEX_LSQB`. This preserves
both required forms:

```agl
l[2]   # indexing
f [2]  # f([2])
```

### D2 — Runtime behavior for invalid indexes and missing keys

Invalid list indexes and missing dict keys raise catchable AgL exceptions.
Add built-in catchable exception records, `IndexError` and `KeyError`, with
useful fields:

- `IndexError`: `index`, `length`, and `message`;
- `KeyError`: `key` and `message`.

These exceptions surface through normal `try`/`catch`. Indexed expression result
types remain precise: `list[int][i]` has type `int`, not `json` or a nullable
type.

### D3 — Negative list indexes

List indexing supports Python-style negative indexes:

- `xs[-1]` selects the last element;
- `xs[-2]` selects the second-to-last element;
- an index is valid when `-len(xs) <= i < len(xs)`;
- out-of-range negative indexes raise the D2 `IndexError`.

### D4 — Dictionary key type

Current semantic dict type is `dict[text, V]`; dict literals already accept
quoted text keys and unquoted lowercase-name shorthand.

Dictionary indexing only supports `text` keys:

```agl
d["a"]
```

The checker rejects non-`text` dict indexes.

### D5 — Assignment through indexes

Add indexed assignment, restricted to `var` list/dict bindings:

```agl
var xs = [1, 2]
set xs[0] = 10

var d = {"a": 1}
set d["a"] = 2
```

Indexed assignment through `let`, params/function arguments, function-return
temporaries, fields, and non-`var` bindings is rejected. Function arguments are
immutable by default.

### D6 — Indexing precedence and chaining

Indexing is a postfix operator with the same precedence as calls and field
access. It is left-associative and chains with other postfix forms:

```agl
rows[0].name
matrix[0][1]
make()[0]
d["a"]["b"]
```

## Grammar Strategy

Add a new token, `INDEX_LSQB`, and use it only in postfix indexing:

```ebnf
%declare INDEX_LSQB

?postfix: postfix LPAR arg_list? RPAR      -> call
        | postfix DOT field_name           -> field_access
        | postfix DOT TYPE_NAME            -> type_access
        | postfix INDEX_LSQB expr RSQB     -> index_access
        | atom
```

Keep list literals unchanged:

```ebnf
lit_list: LSQB (expr (COMMA expr)* COMMA?)? RSQB
```

This means:

- `l[2]` tokenizes as `VAR_NAME INDEX_LSQB INT RSQB` and parses as indexing;
- `f [2]` tokenizes as `VAR_NAME LSQB INT RSQB` and parses as juxtaposition with
  a `ListLit` argument;
- `[2][0]` works because an adjacent `[` after `RSQB` is remapped to
  `INDEX_LSQB`;
- `f([2])[0]` works because an adjacent `[` after `RPAR` is remapped.

The lexer must only emit `INDEX_LSQB` when `[` is adjacent to a previous real
token that can end an expression. Previous token types:

- literals/names/constructors: `VAR_NAME`, `TYPE_NAME`, `INT`, `DECIMAL`,
  `true`, `false`, `null`, template end;
- closing delimiters: `RPAR`, `RSQB`, `RBRACE`.

The token list should be verified against the grammar during implementation.

The remap should check raw-source adjacency using token offsets:

```text
previous_real.end_pos == current_lsqb.start_pos
```

Whitespace, comments, or newlines between the tokens therefore keep ordinary
`LSQB`.

## AST Changes

Add a frozen expression node in `src/agm/agl/syntax/nodes.py`:

```python
@dataclass(frozen=True, slots=True)
class IndexAccess:
    obj: Expr
    index: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
```

Wire it through:

- `Expr` union;
- `src/agm/agl/syntax/__init__.py` exports;
- `src/agm/agl/syntax/visitor.py` traversal and visitor protocol;
- parser transform method `index_access`;
- AST tests.

Indexed assignment also needs an assignment-target representation for `set`:

- keep `IndexAccess` as the expression node;
- add an AST node for indexed set targets:

```python
@dataclass(frozen=True, slots=True)
class IndexTarget:
    obj: Expr
    index: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
```

- change `SetStmt` from `name: str` to a target union:
  `SetTarget = NameTarget | IndexTarget`;
- parse `set name = expr` exactly as today and add `set postfix[index] = expr`
  with a restricted target grammar.

The target grammar must avoid accepting arbitrary non-assignable expressions,
including `set make()[0] = 1`. For this feature, the target must ultimately
resolve to a `var` binding whose value is a list or dict.

## Type Checking

Add `_check_index_access` to `src/agm/agl/typecheck/checker.py`:

- for `list[T]`, check the index expression against `int`, return `T`;
- for `dict[text, V]`, check the key expression against `text`, return `V`;
- otherwise raise a type error like:
  `"indexing requires a list or dict; got '<type>'."`

Additional type errors:

- list indexed by non-`int`;
- dict indexed by non-`text`;
- indexing `json`, records, enums, functions, agents, and unit.

Store the inferred type in `_node_types` like other expression nodes.

For indexed assignment:

- resolve the root target and require it to be a mutable `var` binding;
- reject indexed assignment through `let`, `param`, function arguments,
  function-return temporaries, fields, and any non-`var` root;
- for `var xs: list[T]`, require `int` index and value assignable to `T`;
- for `var d: dict[text, V]`, require `text` key and value assignable to `V`;
- reject indexed assignment on non-list/dict variables.

## Evaluation

Add `_eval_index_access` to `src/agm/agl/eval/interpreter.py`:

- evaluate object first, then index/key;
- for `ListValue`, require `IntValue`, normalize negative indexes Python-style,
  then return `elements[i]` if in range;
- for `DictValue`, require `TextValue`, then return `entries[key]` if present;
- invalid index/key failures should follow D2.

Runtime type mismatches after static checking can remain defensive
`RuntimeError` / `assert_never` style, matching nearby evaluator code.

For indexed assignment:

- evaluate the target variable's current value;
- evaluate index/key and assigned value;
- lists are currently stored as immutable tuples, so assignment should replace
  the binding with a new `ListValue` containing the changed element;
- dicts are frozen by convention, so assignment should replace the binding with
  a new `DictValue` containing the changed entry;
- list assignment outside the valid normalized range raises the D2 index
  exception;
- dict assignment updates existing keys; assignment to a missing key raises the
  D2 key exception.

## Tests First

Follow TDD. Add failing tests before implementation.

### Parser and Grammar

In `tests/test_agl_parser.py`:

- preserve conflict guard: `TestConflictGuard::test_zero_conflicts`;
- `l[2]` parses as `IndexAccess(VarRef("l"), IntLit(2))`;
- `d["a"]` parses as `IndexAccess(VarRef("d"), StringLit("a"))`;
- `f [2]` still parses as `Call(f, (ListLit([2]),), ())`;
- `print [1,2,3]` still parses as a juxtaposition call;
- `matrix[0][1]`, `rows[0].name`, and `make()[0]` parse as postfix chains;
- `l [2]` is not indexing.

### Lexer

In `tests/test_agl_lexer.py`:

- adjacent bracket after expression-ending token emits `INDEX_LSQB`;
- spaced bracket emits `LSQB`;
- newline/comment-separated bracket emits `LSQB`;
- adjacent bracket after `do` is still handled by the existing `LOOP_BOUND`
  merge and does not regress `do[3]`.

### Type Checker

In `tests/test_agl_typecheck.py` or rejection fixtures:

- `let xs: list[int] = [10, 20]; xs[0]` has type `int`;
- `let d: dict[text, int] = {"a": 1}; d["a"]` has type `int`;
- list with text key is rejected;
- dict with int key is rejected;
- non-container indexing is rejected.
- `set xs[0] = 3` is accepted only when `xs` is a `var list[int]`;
- `set d["a"] = 3` is accepted only when `d` is a `var dict[text, int]`;
- indexed assignment through `let`, `param`, and function arguments is rejected;
- indexed assignment value type mismatches are rejected.

### Evaluator / E2E

In `tests/test_agl_eval.py` and/or scenario fixtures:

- list indexing returns the selected value;
- negative list indexes select from the end Python-style;
- dict indexing returns the selected value;
- indexed list assignment updates a `var` binding;
- indexed dict assignment updates a `var` binding;
- chained indexing works for nested lists/dicts;
- invalid index and missing key raise catchable AgL exceptions.

## Documentation

Update after implementation:

- `docs/agl/reference/expressions.md`: add an "Indexing" section after field
  access or after literals; document adjacency-sensitive syntax and
  Python-style negative list indexes;
- `docs/agl/reference/bindings-and-scope.md`: document indexed assignment and
  the restriction to `var` list/dict bindings;
- `docs/agl/reference/grammar.md`: add `index_access`;
- `docs/agl-grammar.md`: keep generated/reference grammar in sync if it mirrors
  `agl.lark`;
- any examples using list/dict access where helpful.

## Implementation Sequence

1. Add parser/lexer tests that fail for indexing and preserve `f [2]`.
2. Add indexed-assignment parse/type/eval tests, including `var` restrictions.
3. Add `INDEX_LSQB` token constant, `%declare`, remap logic, and conflict
   guard coverage.
4. Add `IndexAccess` and assignment-target AST nodes, transform methods,
   visitor/export wiring.
5. Add type checker support, including mutable-root checks for indexed
   assignment.
6. Add evaluator support for indexing, Python-style negative indexes, and
   copy-on-write indexed assignment.
7. Add runtime failure behavior from D2.
8. Update docs.
9. Run `uv run pytest` for focused AgL tests while iterating.
10. Finish with `just check`.

## Risks and Mitigations

- **LALR conflict from square brackets.** Mitigation: parser sees
  `INDEX_LSQB` for indexing and `LSQB` for list literals; conflict guard remains
  mandatory.
- **Whitespace surprise.** Mitigation: document that `x[y]` indexes, while
  `x [y]` is call sugar with a list argument; add tests for both.
- **Lexer remap accidentally affects type brackets.** Mitigation: remap only in
  expression-token contexts and test typed calls such as `ask-request::[T]("p")`
  still tokenize/parse.
- **`do[N]` regression.** Mitigation: keep existing `DO LSQB INT RSQB` merge
  priority and add a lexer/parser regression.
- **Indexed dict assignment missing-key behavior.** Mitigation: test that
  `set d["new"] = v` raises the same catchable `KeyError` as `d["new"]`.
