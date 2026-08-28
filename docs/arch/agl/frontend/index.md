# AgL Frontend

The frontend turns source text into a fully resolved, type-checked, match-compiled program. It is five passes — lexer, parser, scope, typecheck, match compilation — over one shared AST. Everything here is static: no agent calls, no shell execution, no evaluation. See [index.md](../index.md) for how the frontend sits in the overall pipeline.

## Pass Structure

Each pass wraps the previous pass's artifact rather than mutating it: parsing builds an AST that may retain raw infix chains; module-graph assembly resolves those chains with import-visible fixities; scope yields a resolved program, typecheck yields a checked program, and match compilation yields the artifact that lowering consumes. Conclusions are recorded in pass artifacts and side tables keyed by stable node ids, so AST nodes stay frozen and shared. Scope emits immutable pattern-slot `BindingRef`s for field-directed branch names and candidate sets for ambiguous bare `is` variants; typecheck selects their final meanings from matched or scrutinee nominal types in checker-owned maps, and consumers use checked-artifact accessors to resolve them. No pass rewrites another pass's resolution tables. Program-level variants of scope, typecheck, and match compilation generalize these passes to whole module graphs ([modules.md](../modules.md)).

Every static pass recurses over the syntax tree, so source nested deeply enough exhausts the host stack before the pass can report anything. `agl/recursion.py` owns the boundary that converts that exhaustion into an ordinary pre-execution diagnostic; the pipeline and the REPL apply it around each pass they drive, so over-deep source is reported rather than crashed on. Unlike the evaluator, the frontend does not raise Python's recursion limit — it has no language-level depth budget to protect.

Every pass reports through one diagnostic channel. Diagnostics produced by scope and typecheck carry a `phase` tag naming that pass, so a caller can tell a name-resolution failure from a type failure without matching message text; the other passes leave it unset, and it is never part of rendered output.

## What To Read Next

- Read [syntax.md](syntax.md) for the lexer, parser, and the AST firewall.
- Read [scope.md](scope.md) for name resolution.
- Read [types.md](types.md) for the semantic type model and the typecheck pass.
- Read [matchcompile.md](matchcompile.md) for the pattern-match compiler.
