# AgL Frontend

The frontend turns source text into a fully resolved, type-checked, match-compiled program. It is five passes — lexer, parser, scope, typecheck, match compilation — over one shared AST. Everything here is static: no agent calls, no shell execution, no evaluation. See [../index.md](../index.md) for how the frontend sits in the overall pipeline.

## Pass Structure

Each pass wraps the previous pass's artifact rather than mutating it: parsing yields the AST, scope yields a resolved program, typecheck yields a checked program, and match compilation yields the artifact that lowering consumes. Conclusions are recorded in immutable side tables keyed by stable node ids, so AST nodes stay frozen and shared. Graph-aware variants of scope, typecheck, and match compilation generalize these passes to whole module graphs ([../modules.md](../modules.md)).

## What To Read Next

- Read [syntax.md](syntax.md) for the lexer, parser, and the AST firewall.
- Read [scope.md](scope.md) for name resolution.
- Read [types.md](types.md) for the semantic type model and the typecheck pass.
- Read [matchcompile.md](matchcompile.md) for the pattern-match compiler.
