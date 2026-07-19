# AgL Frontend

The frontend turns source text into a fully resolved, type-checked, match-compiled program. It is five passes — lexer, parser, scope, typecheck, match compilation — over one shared AST. Everything here is static: no agent calls, no shell execution, no evaluation. See [index.md](agl/index.md) for how the frontend sits in the overall pipeline.

## Pass Structure

Each pass wraps the previous pass's artifact rather than mutating it: parsing yields the AST, scope yields a resolved program, typecheck yields a checked program, and match compilation yields the artifact that lowering consumes. Conclusions are recorded in pass artifacts and side tables keyed by stable node ids, so AST nodes stay frozen and shared. Scope emits immutable pattern-slot `BindingRef`s for field-directed branch names; typecheck selects their final meanings in checker-owned maps, and consumers use checked-artifact accessors to resolve them. No pass rewrites another pass's resolution tables. Program-level variants of scope, typecheck, and match compilation generalize these passes to whole module graphs ([modules.md](agl/modules.md)).

## What To Read Next

- Read [syntax.md](agl/frontend/syntax.md) for the lexer, parser, and the AST firewall.
- Read [scope.md](agl/frontend/scope.md) for name resolution.
- Read [types.md](agl/frontend/types.md) for the semantic type model and the typecheck pass.
- Read [matchcompile.md](agl/frontend/matchcompile.md) for the pattern-match compiler.
