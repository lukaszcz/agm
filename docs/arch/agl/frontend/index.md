# AgL Frontend

The frontend turns source text into a fully resolved, type-checked, match-compiled program in five passes — lexer, parser, scope, typecheck, match compilation — over one shared AST. Everything here is static: no agent calls, no shell execution, no evaluation.

## Pass Structure

Each pass wraps the previous pass's artifact rather than mutating it: parsing builds an AST that may retain raw infix chains; module-graph assembly resolves those chains with import-visible fixities; scope yields a resolved program, typecheck a checked program, and match compilation the artifact that lowering consumes. AST nodes stay frozen and shared; conclusions live in side tables keyed by node id.

Every pass reports through one diagnostic channel (`agl/diagnostics.py`). Scope and typecheck diagnostics carry a phase tag so callers can distinguish them without matching text. Each pass recurses over the tree, so `agl/recursion.py` converts stack exhaustion on over-deep source into an ordinary diagnostic ([hosting.md](../hosting.md)).

## What To Read Next

- [syntax.md](syntax.md) — the lexer, parser, and the AST firewall.
- [scope.md](scope.md) — name resolution.
- [types.md](types.md) — the semantic type model and the typecheck pass.
- [matchcompile.md](matchcompile.md) — the pattern-match compiler.
