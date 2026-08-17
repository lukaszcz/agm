# AgL Syntax: Lexer, Parser, and AST

The syntax frontend turns AgL source into the AST consumed by every later
pass. The hand-written lexer handles indentation, templates, raw tails, and
tight slash-path and qualifier syntax. The Lark LALR parser recognizes the
resulting token stream, and the AST builder validates and constructs the
language's source forms.

## Syntax Boundary

The lexer and parser are the only Lark-aware layers. Their output is a family
of frozen AST dataclasses, which forms the frontend firewall: scope, typecheck,
and later passes depend on AST nodes rather than parser types. Nodes have
stable ids, and later passes attach conclusions in side tables instead of
mutating the AST.

`import`, `use`, `export`, `hiding`, `scope`, and `end` are contextual header
words. They remain ordinary names outside their declaration contexts. Ambiguous item-start `use`
forms are classified with a parser rooted at the canonical `use_decl` grammar rather than a second
hand-maintained recognizer. The AST
preserves import and use tails, hiding clauses, aliases, and scope paths for
scope resolution; it does not decide their visibility.

## Code Entry Points

- `src/agm/agl/keywords.py` and `src/agm/agl/lexer/` — keyword inventory and
  indentation-aware lexing.
- `src/agm/agl/grammar/` and `src/agm/agl/parser/` — grammar, parsing, AST
  construction, and inline-source wrapping.
- `src/agm/agl/syntax/` — AST nodes, spans, advisories, and syntax-only helpers.
- Tests: `tests/test_agl_lexer.py`, `tests/test_agl_parser.py`, and
  `tests/test_agl_ast.py`.
