# AgL Syntax: Lexer, Parser, and AST

The hand-written lexer handles layout (INDENT/DEDENT), string templates with `%{}` interpolation, raw tails (`exec$`, `ask$`), and the tight slash-path and `::` qualifier syntax. A Lark LALR grammar recognizes the token stream, and the AST builder validates and constructs frozen dataclass nodes with stable ids. Comments produce no tokens, but their spans are exposed as a side channel for highlighters.

## Keywords

`keywords.py` is the single inventory of reserved words and of the contextual header words (`import`, `use`, `export`, `hiding`, `scope`, `end`), which are ordinary names outside their declaration contexts. The lexer, the grammar's token contract, the REPL highlighter, and the editor modes all derive from it rather than repeating spellings.

## What the AST Preserves

The AST records source structure faithfully so later passes never reconstruct spellings:

- User-operator chains cross the parser unresolved as raw infix-chain nodes; module-graph assembly rewrites them into ordinary applications once import-visible fixities are known ([modules.md](../modules.md)). Scope and later passes see only resolved applications.
- Qualified expressions, types, patterns, and `is` tests share one qualifier-chain node with per-segment spans and type arguments.
- Declarations and scope-region items carry canonical scope paths; enum members record whether they were declared inline or reference an existing record; `var` field markers, parameter zones, complete `let` patterns, and assignment-target shapes (name, index, field) are all retained.
- A function header may carry an applied builtin receiver (`array[E]::map`) beside its `self` parameter; scope classifies it once the full declaration path is known.

An inline-source host (`agm exec -c`, the REPL) wraps statement-oriented source with a pure syntactic wrapper in `parser/wrap.py` before the static passes run.

## Code Entry Points

- `src/agm/agl/keywords.py`, `src/agm/agl/lexer/` — keyword inventories and indentation-aware lexing.
- `src/agm/agl/grammar/agl.lark`, `src/agm/agl/parser/` — grammar, parsing, AST construction, inline-source wrapping.
- `src/agm/agl/syntax/` — AST nodes, spans, advisories, the constant-expression predicate, and resource-call classification.
- Tests: `tests/test_agl_lexer.py`, `test_agl_parser.py`, `test_agl_ast.py`, `test_agl_wrap.py`.
