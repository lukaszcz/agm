# AgL Syntax: Lexer, Parser, and AST

The hand-written lexer handles layout (INDENT/DEDENT), string templates with `%{}` expressions and `${NAME}` environment interpolation, and the tight slash-path and `::` qualifier syntax. A Lark LALR grammar recognizes the token stream, and the AST builder validates and constructs frozen dataclass nodes with stable ids. Comments produce no tokens, but their spans are exposed as a side channel for highlighters.

Single- and triple-quoted templates share hole recognition and token emission. Environment holes scan names with the shared identifier rules and emit a single token each, spanning the hole and carrying its name; the AST builder turns that token into the qualified environment read it stands for. Triple-quoted dedenting measures indentation across literal/hole segments, assembles retained line slices, and maps only literal boundaries, preserving hole tokens' source positions without per-character position tables. A `$` literal (to end of line, or an indented block) opens and closes with its own delimiter tokens, but shares the same fragment and hole tokens, hole scanner, and `template` grammar rule as a quoted template.

The lexer's escape, number, identifier, and environment-hole scanning rules live in `agl/value_syntax/lexical.py`, a leaf below the lexer that also backs the value-syntax reader (`agl/value_syntax/reader.py`), so both scan AgL literals identically. A `\uXXXX` escape denotes a scalar value: an adjacent high and low pair combines into one character, as in JSON, and any other surrogate escape is a lexical error, as is a raw surrogate anywhere in the source.
The same leaf spells values back as literals (`quote_text`, `scalar_text`), so runtime rendering and compile-time constant folding cannot diverge.

## Constant Folding

`syntax/constants.py` is the syntactic constant-expression predicate. `syntax/module_constants.py` folds a module's own constants to text from its AST alone: it indexes every static `let`/`var` by declaration path, resolves a reference by the same-module part of whole-path lookup ([scope.md](scope.md#full-path-lookup)) through `syntax/qualifiers.py` (enclosing scope region outward, `::` from the root), and folds templates whose holes name scalars. It depends on no resolver, which is what lets both the scope pass and the parse-only package command scan (`packages/source_commands.py`) fold one attribute argument the same way — and is why a reference reaches the declaring module only.

The same module's `fold_scalar` factors that folding's literal/unary-operator/template mechanics behind a reference-resolution callback, so a second caller with its own resolution can fold a scalar identically; a callback may refuse a reference whose own declared type is not itself scalar, since only a scalar's initializer is guaranteed to render the way its folded text would claim. The same module's `constant_key` is the comparison key for the checker's static dict-literal duplicate-key check (the checker resolves references through `typecheck/constant_bindings.py` `ModuleConstantBindings.initializer_for`: fixed `let`s only): null/bool/numeric equality, text after scalar folding, and constructor or constant-builtin calls by callee identity with field-normalized arguments; a key with no such form is left to a runtime `DuplicateKeyError`.

## Keywords

`keywords.py` is the single inventory of reserved words and of the soft keywords, which are ordinary names outside their promotion window: the header words (`import`, `use`, `export`, `hiding`, `scope`, `end`) outside their declaration contexts, and the operator words (`and`, `or`, `not`, `is`, `in`, `to`, `downto`, `step`, `with`) outside operator position, which is what lets a member be named `or` or `not`. The lexer, the grammar's token contract, the REPL highlighter, and the editor modes all derive from it rather than repeating spellings.

## Operator Position

`lexer/operators.py` holds the one inventory of the token types that close an operand, begin one, or mark a name, and the predicates over it. Two passes ask the same question of it and so cannot disagree: layout suppresses a line break that would separate an operator from its operand — a line ending with one, or a more indented line opening with one — and the later promotion pass decides whether an operator word is an operator or a member name. Layout asks with the break itself ignored, which is exactly the neighbourhood promotion later sees. A module header spells a path rather than an expression, so layout leaves its line alone.

## What the AST Preserves

The AST records source structure faithfully so later passes never reconstruct spellings:

- A chain of builtin operators is grouped at parse; one containing a user operator crosses the parser as a raw infix-chain node, which scope groups by the selected declarations' fixities and rewrites into ordinary applications (`agl/infix.py`, [scope.md](scope.md#operators)). Typecheck and later passes see only grouped applications.
- A parenthesized built-in symbolic operator, `(+)`, is its own operator-value node carrying the operator; a user operator in parentheses stays an ordinary name reference.
- Qualified expressions, types, patterns, and `is` tests share one qualifier-chain node with per-segment spans and type arguments.
- Declarations and scope-region items carry canonical scope paths; enum members record whether they were declared inline or reference an existing record; immutable `let` names, `var` field markers, and assignment-target shapes (name, index, field) are all retained.
- A function header may carry an applied builtin receiver (`array[E]::map`) beside its `self` parameter; its declaration path uses the receiver constructor's plain scope (`array`), while the receiver retains its type arguments for scope and typecheck classification.
- A generic function declaration may carry a `{Hashable K, Eq T}` constraint block naming structural bounds (`agl/constraints.py`) on its type parameters; the constrained name must resolve to an in-scope type parameter, checked by scope ([scope.md](scope.md)).
- A partial-application placeholder is a call argument, never an expression (`CallArg = Expr | Placeholder`). `is_complete_call` narrows a placeholder-free call to `CompleteCall`; the checker and lowerer take the complete or partial path by it.
- A lambda retains each optional parameter annotation and may use a bare unary parameter form; type checking fills omissions from function context.
- A leading-dot postfix chain (`.field`, `.method`, `.copy().value`) becomes a marked unary lambda over a hygienic receiver binding. It shares ordinary postfix construction and closes at the chain boundary; later passes reuse lambda and member-selection paths.
- Declaration nodes carry an optional attribute prefix — a name plus ordinary call arguments — held verbatim. Enum member prefixes may span indented continuation lines; the AST keeps one ordered attribute list. The AST makes no claim about which attributes exist or what they mean; recognition happens against the catalog in `agl/attributes.py`, a dependency-free leaf the AST itself never imports. The parser attaches an attribute to its declaration, parameter, or field and stops there; scope is where an attribute acquires meaning ([scope.md](scope.md)).

An inline-source host (`agm exec -c`, the REPL) parses statement-oriented source with `PipelineDriver.parse_entry(inline_code=True)`, which applies a pure syntactic wrapper in `parser/wrap.py` before the static passes run. The wrapper keeps scoped bindings and root `@param` bindings at the module root; executable items become a synthetic entry.

## Code Entry Points

- `src/agm/agl/keywords.py`, `src/agm/agl/lexer/` — keyword inventories and indentation-aware lexing; `lexer/operators.py` is the shared operator-position inventory, `lexer/layout.py` the INDENT/DEDENT and continuation filter.
- `src/agm/agl/grammar/agl.lark`, `src/agm/agl/parser/` — grammar, parsing, AST construction, inline-source wrapping.
- `src/agm/agl/syntax/` — AST nodes, spans, advisories, the generic walk (`visitor.py`, dispatched through a node-class table that `tests/test_agl_ast.py` checks covers every node class), the constant-expression predicate, the constant folder's search order, module-constant folding, and resource-call classification.
- `src/agm/agl/attributes.py` — the built-in attribute catalog: targets, argument shapes, repetition, and conflicts.
- Tests: `tests/test_agl_lexer.py`, `test_agl_parser.py`, `test_agl_ast.py`, `test_agl_wrap.py`, `test_agl_module_constants.py`.
