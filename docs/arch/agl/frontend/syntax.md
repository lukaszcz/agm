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

User-operator chains cross that boundary unresolved: the builder keeps them as
raw infix-chain nodes, and module-graph assembly rewrites them into ordinary
applications at each chain's lexical scope, using local declarations plus every
bare-visible operator fixity. Standalone parser callers may resolve against an
explicit ambient table instead. Scope and every later pass see only resolved
applications.

`import`, `use`, `export`, `hiding`, `scope`, and `end` are contextual header
words. They remain ordinary names outside their declaration contexts. Item-start
`use` promotion is decided from its header token shape; the grammar then
requires a suffix for every declared use. Qualified use paths use the ordinary
module-qualifier token contract, and the AST preserves their unresolved target,
tails, hiding clauses, aliases, and scope paths for scope resolution to decide
their visibility.

Qualified expressions, types, patterns, and `is` tests share a structured
`QualifierChain`; every segment retains its span and optional type arguments.
Enum members preserve whether the source declared an inline `VariantDef` or
referenced a record through `VariantRef`. A record or enum-member field may be
prefixed with `var`; `Param` preserves that marker independently of its
constructor zone. Assignment targets keep their distinct name, index, or
`FieldTarget` shape; a field target retains an arbitrary postfix receiver and
the selected field name. The marker is rejected on exception fields. Let
bindings retain a complete pattern, while declarations and region items carry
canonical scope paths, so
later passes do not reconstruct source spellings.

A function header can retain an applied builtin receiver (`array[E]::map` or
`dict[text, V]::get`) alongside an ordinary `self` parameter; scope
classification waits for the completed declaration path, because a scope region
prefixes its already-built child declarations.

## Code Entry Points

- `src/agm/agl/keywords.py` and `src/agm/agl/lexer/` — keyword inventory and
  indentation-aware lexing.
- `src/agm/agl/grammar/` and `src/agm/agl/parser/` — grammar, parsing, AST
  construction, and inline-source wrapping.
- `src/agm/agl/syntax/` — AST nodes, function receiver types, spans, advisories,
  and syntax-only helpers.
- Tests: `tests/test_agl_lexer.py`, `tests/test_agl_parser.py`, and
  `tests/test_agl_ast.py`.
