# AgL Syntax: Lexer, Parser, and AST

The hand-written lexer handles layout (INDENT/DEDENT), string templates with `%{}` expressions and `${NAME}` environment interpolation, raw tails (`exec$`, `ask$`), and the tight slash-path and `::` qualifier syntax. A Lark LALR grammar recognizes the token stream, and the AST builder validates and constructs frozen dataclass nodes with stable ids. Comments produce no tokens, but their spans are exposed as a side channel for highlighters.

Single- and triple-quoted templates share hole recognition and token emission; raw tails reuse the expression-hole scanner. Environment holes scan names with the shared identifier rules and desugar to expression tokens. Triple-quoted dedenting measures indentation across literal/hole segments, assembles retained line slices, and maps only literal boundaries, preserving hole tokens' source positions without per-character position tables.

## Keywords

`keywords.py` is the single inventory of reserved words and of the soft keywords, which are ordinary names outside their promotion window: the header words (`import`, `use`, `export`, `hiding`, `scope`, `end`) outside their declaration contexts, and the operator words (`and`, `or`, `not`, `is`, `in`, `to`, `downto`, `step`, `with`) outside operator position, which is what lets a member be named `or` or `not`. The lexer, the grammar's token contract, the REPL highlighter, and the editor modes all derive from it rather than repeating spellings.

## What the AST Preserves

The AST records source structure faithfully so later passes never reconstruct spellings:

- User-operator chains cross the parser unresolved as raw infix-chain nodes; module-graph assembly rewrites them into ordinary applications once import-visible fixities are known ([modules.md](../modules.md)). Scope and later passes see only resolved applications.
- Qualified expressions, types, patterns, and `is` tests share one qualifier-chain node with per-segment spans and type arguments.
- Declarations and scope-region items carry canonical scope paths; enum members record whether they were declared inline or reference an existing record; `var` field markers, complete `let` patterns, and assignment-target shapes (name, index, field) are all retained.
- A function header may carry an applied builtin receiver (`array[E]::map`) beside its `self` parameter; scope classifies it once the full declaration path is known.
- Declaration nodes carry an optional attribute prefix — a name plus ordinary call arguments — held verbatim. The AST makes no claim about which attributes exist or what they mean; recognition happens against the catalog in `agl/attributes.py`, a dependency-free leaf the AST itself never imports. The parser attaches an attribute to its declaration, parameter, or field and stops there; scope is where an attribute acquires meaning ([scope.md](scope.md)).

An inline-source host (`agm exec -c`, the REPL) wraps statement-oriented source with a pure syntactic wrapper in `parser/wrap.py` before the static passes run.

## Code Entry Points

- `src/agm/agl/keywords.py`, `src/agm/agl/lexer/` — keyword inventories and indentation-aware lexing.
- `src/agm/agl/grammar/agl.lark`, `src/agm/agl/parser/` — grammar, parsing, AST construction, inline-source wrapping.
- `src/agm/agl/syntax/` — AST nodes, spans, advisories, the constant-expression predicate, and resource-call classification.
- `src/agm/agl/attributes.py` — the built-in attribute catalog: targets, argument shapes, repetition, and conflicts.
- Tests: `tests/test_agl_lexer.py`, `test_agl_parser.py`, `test_agl_ast.py`, `test_agl_wrap.py`.
