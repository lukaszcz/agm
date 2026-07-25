# AgL Name Resolution

The scope pass performs full name resolution and records its results in side tables. Its collection pre-passes key declarations by module, named-scope path, and name; the empty path is the module root. They collect agents, functions, types, and constructors before expression bodies are resolved, so root declarations remain visible regardless of order and mutual recursion works.

## Namespace-Directed Resolution

Scope-region syntax and declaration-path shorthand cross the parser firewall in canonical AST forms. Collection materializes a `ScopeNode` layer and static member map for each named path, merging repeated regions, shorthand paths, and same-named type scopes. A type's enum variants are collected as members of that type scope, while record and exception construction retains its bare declaration spelling. One ordered chain resolver first considers exact local paths through the active scope layers, then import routes; type-owning segments emit the same `ConstructorRef` result as bare constructor candidates. The checker consumes that result for values and calls, validating variant shape and signatures without re-resolving names. Patterns and `is` tests record a valid owner through the same chain, while unresolved routes and clashes defer to the checker's historic qualification and enum diagnostics. The type environment applies the same local/module anchor and clash policy. `::` starts at the module root, and scope paths never suffix-match. Plain scope segments reject type arguments in expression, type, pattern, and `is` chains. Module-boundary selection is path-aware: scoped declaration paths are public atoms, selecting a scope expands its public subtree, and selection renames re-root that subtree. Atoms retain their defining module, so same-spelled scoped paths coexist until a route or bare use selects one.

Declared identity is structured, never spelled. A declaration is identified by its module, its scope path, and its unqualified name as three separate components — `ConstructorRef`, `BindingRef`, `NominalId`, and agent keys all carry them apart. A joined `A::Name` string is a display spelling produced for diagnostics and never reconstituted into an identity, so a scope path is never recovered by splitting a name. Passes that need a name in its declaring region re-enter that region instead: the type environment exposes a scope cursor that alias resolution, scoped function bodies, and constructor field lookup enter so unqualified sibling references resolve the way the declaration site sees them.

Resolution is namespace- and scope-directed, never capitalization-directed — a direct consequence of AgL's case-neutral name model:

- Built-in calls (`print`, `exec`, `ask`, and friends) are recognized by resolving the callee to a known built-in declaration, not by keyword.
- Constructors live in the value namespace; an ambiguous unqualified constructor name is a static error, disambiguated with `Type::Ctor` qualification. Arbitrary qualifier depths resolve local scope members where their exact path exists; a local qualifier chain that also contributes an imported module member is a clash with anchor repair guidance. Applied segments are checked against that chain's active lexical owner, never an unrelated same-suffixed scope.
- A declaration may claim a constructor's unqualified spelling exactly when the constructor stays reachable some other way: enum variants (reachable as `Owner::variant`) and constructors owned by another module (reachable by module qualification) yield to it, while a same-module record, exception, or alias constructor — whose declaration *is* the bare name — collides as a duplicate declaration.
- Scope records constructor candidates for bare pattern names independently of ordinary value bindings. Top-level names are constructor-only; nested field-directed names resolve to immutable pattern-slot `BindingRef`s, with slot metadata recording candidates and visible alternatives, grouped per case branch so typechecking selects exactly the branch it just checked. Candidate metadata retains whether a spelling can be a bare nullary enum pattern, allowing scope to reject definite duplicate binders eagerly while leaving genuine field-directed cases to typechecking. Typechecking selects each slot's final binder or constructor in checker-owned maps; consumers use the checked artifact's accessors for those meanings. No later pass rewrites scope's resolution tables. An `as`-pattern name is always a variable binder.

Assignment follows the same split. Scope resolves an unqualified `:=` target and rejects an undeclared name, but leaves assignability to typechecking, which alone knows which binding a pattern slot selected. A qualified target is settled in scope, since only `builtin var` is assignable across a module boundary and no qualified name is a pattern slot.

## Import Environments

`scope/imports.py` is the pure import-policy seam. Its contribution environment
merges every declaration for a module into its selected path atoms, bare injection, aliases,
and plain-path routes. Scoped paths retain their structure through selection and re-exporting; policy expands scope-prefix selections and re-roots renamed subtrees. The selected set bounds both routes and bare injection: plain
imports are qualified-only, while `using` and `open import` inject bare names. Scope `open`
declarations use the same structured bare-contribution layers on their enclosing `ScopeNode`.
Constructor contributions retain their owner path in a parallel region-local candidate layer, so a
nearer opened enum variant wins without changing global bare-constructor candidates. Local or
directly imported scope subtrees are selected there, and collisions remain deferred
to the use site. Value and type lookup both consume those layers, so selected and renamed
type members follow the same region boundaries and provenance; they never alter export maps
or import contributions. One shared suffix/anchored resolver serves value reads and writes,
constructors, and type qualification, retaining ambiguity and route identity until the use
site; its diagnostics distinguish private declarations from names outside a contribution.
One shared translator walks those verdicts and raises an error the caller constructs, so
the scope and typecheck passes share the walk while keeping their own exception types and
wording.
Whitespace-separated qualifier near-misses are reported from the lexer's advisories
rather than reconstructed from AST shapes: when a reference fails to resolve at an offset
an advisory covers, the pass offers the tight spelling — but only when re-resolving that
route actually contributes the intended member, preserving valid division and
juxtaposition expressions.

## Static Guarantees

Agents must be declared in source; the pass retains every declaration and
unused-agent warning by its scope path and member name, and binds it as a
first-class value of agent type, including named-scope members. Scoped externs resolve their member names through the declaring module's companion, and collection rejects duplicate scoped companion symbols. `let _ = value` and `var _ = value` still resolve their right-hand sides but register no binding, so `_` may be repeated. `_` never resolves as a readable identifier, even when another binding form uses that name in an enclosing scope. Register-backed `builtin var` declarations are admitted only in the canonical `std/config` module. The pass enforces lexical control-flow boundaries — `break`/`continue` must stay within a loop in the same function, `return` must appear inside a function body — and the extern (Python FFI) placement rule that externs are only allowed in file-backed modules.

Program resolution extends this pass across modules and preserves the loader's immutable,
reverse-topological import-SCC sequence on `ResolvedProgram`. Typecheck consumes that exact sequence
to publish closed inferred function signatures from dependency SCCs before importers, without
rebuilding the module graph. See [modules.md](agl/modules.md).

## Code Entry Points

- `src/agm/agl/scope/` — `resolve_module`, `resolve_program`, their resolution side tables, and the pure import-policy models in `imports.py`.
- Tests: `tests/test_agl_scope.py`, `tests/test_agl_scope_program.py`,
  `tests/test_agl_scope_imports.py`, `tests/test_agl_scope_contributions.py`,
  `tests/test_agl_namespace_wiring.py`, and `tests/test_agl_pattern_slots.py`.
