# AgL Syntax: Lexer, Parser, and AST

The hand-written lexer handles layout (INDENT/DEDENT), string templates with `%{}` expressions and `${NAME}` environment interpolation, and the tight slash-path and `::` qualifier syntax. A Lark LALR grammar recognizes the token stream, and the AST builder validates and constructs frozen dataclass nodes with stable ids. Comments produce no tokens, but their spans are exposed as a side channel for highlighters.

Single- and triple-quoted templates share hole recognition and token emission. Environment holes scan names with the shared identifier rules and desugar to expression tokens. Triple-quoted dedenting measures indentation across literal/hole segments, assembles retained line slices, and maps only literal boundaries, preserving hole tokens' source positions without per-character position tables. A `$` verbatim literal (to end of line, or an indented block) reuses the same template tokens, hole scanner, and `template` grammar rule as a quoted template.

The lexer's escape, number, identifier, and environment-hole scanning rules live in `agl/value_syntax/lexical.py`, a leaf below the lexer that also backs the value-syntax reader (`agl/value_syntax/reader.py`), so both scan AgL literals identically.

## Keywords

`keywords.py` is the single inventory of reserved words and of the soft keywords, which are ordinary names outside their promotion window: the header words (`import`, `use`, `export`, `hiding`, `scope`, `end`) outside their declaration contexts, and the operator words (`and`, `or`, `not`, `is`, `in`, `to`, `downto`, `step`, `with`) outside operator position, which is what lets a member be named `or` or `not`. The lexer, the grammar's token contract, the REPL highlighter, and the editor modes all derive from it rather than repeating spellings.

## Operator Position

`lexer/operators.py` holds the one inventory of the token types that close an operand, begin one, or mark a name, and the predicates over it. Two passes ask the same question of it and so cannot disagree: layout suppresses a line break that would separate an operator from its operand — a line ending with one, or a more indented line opening with one — and the later promotion pass decides whether an operator word is an operator or a member name. Layout asks with the break itself ignored, which is exactly the neighbourhood promotion later sees. A module header spells a path rather than an expression, so layout leaves its line alone.

## What the AST Preserves

The AST records source structure faithfully so later passes never reconstruct spellings:

- User-operator chains cross the parser unresolved as raw infix-chain nodes; module-graph assembly rewrites them into ordinary applications once import-visible fixities are known ([modules.md](../modules.md)). Scope and later passes see only resolved applications.
- A parenthesized built-in symbolic operator, `(+)`, is its own operator-value node carrying the operator; a user operator in parentheses stays an ordinary name reference.
- Qualified expressions, types, patterns, and `is` tests share one qualifier-chain node with per-segment spans and type arguments.
- Declarations and scope-region items carry canonical scope paths; enum members record whether they were declared inline or reference an existing record; immutable `let` names, `var` field markers, and assignment-target shapes (name, index, field) are all retained.
- A function header may carry an applied builtin receiver (`array[E]::map`) beside its `self` parameter; its declaration path uses the receiver constructor's plain scope (`array`), while the receiver retains its type arguments for scope and typecheck classification.
- A lambda retains each optional parameter annotation and may use a bare unary parameter form; type checking fills omissions from function context.
- A leading-dot method invocation (`.map(f)`) becomes a marked unary lambda over a generated `self` binding. The marker lets type checking require a contextual receiver type while scope, traversal, capture analysis, and lowering reuse ordinary lambda and member-call paths.
- Declaration nodes carry an optional attribute prefix — a name plus ordinary call arguments — held verbatim. The AST makes no claim about which attributes exist or what they mean; recognition happens against the catalog in `agl/attributes.py`, a dependency-free leaf the AST itself never imports. The parser attaches an attribute to its declaration, parameter, or field and stops there; scope is where an attribute acquires meaning ([scope.md](scope.md)).

An inline-source host (`agm exec -c`, the REPL) parses statement-oriented source with `PipelineDriver.parse_entry(inline_command=True)`, which applies a pure syntactic wrapper in `parser/wrap.py` before the static passes run. The wrapper keeps scoped bindings and root `@param` bindings at the module root; executable items become a synthetic entry.

## Code Entry Points

- `src/agm/agl/keywords.py`, `src/agm/agl/lexer/` — keyword inventories and indentation-aware lexing; `lexer/operators.py` is the shared operator-position inventory, `lexer/layout.py` the INDENT/DEDENT and continuation filter.
- `src/agm/agl/grammar/agl.lark`, `src/agm/agl/parser/` — grammar, parsing, AST construction, inline-source wrapping.
- `src/agm/agl/syntax/` — AST nodes, spans, advisories, the constant-expression predicate, and resource-call classification.
- `src/agm/agl/attributes.py` — the built-in attribute catalog: targets, argument shapes, repetition, and conflicts.
- Tests: `tests/test_agl_lexer.py`, `test_agl_parser.py`, `test_agl_ast.py`, `test_agl_wrap.py`.
