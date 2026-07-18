# AgL Syntax: Lexer, Parser, and AST

## Lexer and Parser

The lexer is hand-written because AgL is indentation-sensitive: it produces INDENT/DEDENT tokens, handles multiline strings and string interpolation, and emits `NAME`/`OP_NAME` tokens for identifiers — capitalization never classifies a name. The parser is a Lark LALR grammar over those tokens, and an AST builder constructs the AST. User `infixl`/`infixr` declarations are parser metadata: the builder uses their priorities to rewrite flat infix chains into ordinary AST nodes before anything crosses the firewall. These two passes are the only Lark-aware code in the system.

The implementation-level token contract lives in `src/agm/agl/lexer/tokens.py` (the declared single source of truth) and the lexer pass docstrings; the surface grammar is documented from the user's perspective in the AgL reference (`docs/agl/reference/`).

## The AST

The AST is plain frozen dataclasses with no parser types — the firewall every later pass depends on. Because AgL is expression-oriented there is no statement/expression split: one unified node family covers blocks, bindings, control flow, and a single call node for every kind of invocation. Surface forms that need dedicated representation — partial-application placeholders, value-position type application, qualified constructor references, casts, divergence expressions — are explicit nodes whose shape the AST builder validates before they cross the firewall.

Each node carries a stable id assigned at build time. Later passes never mutate nodes; they record conclusions in side tables keyed by that id. This is the universal annotation convention — it is why nodes can be frozen and shared, and why `id()`-based identity is never used.

## Code Entry Points

- `src/agm/agl/lexer/` — the indentation-aware lexer; `tokens.py` is the token-contract source of truth.
- `src/agm/agl/grammar/` and `src/agm/agl/parser/` — the Lark grammar and the AST builder.
- `src/agm/agl/syntax/` — the AST dataclasses, type nodes, and source-id-stamped spans.
- Tests: `tests/test_agl_lexer.py`, `tests/test_agl_parser.py`, `tests/test_agl_ast.py`.
