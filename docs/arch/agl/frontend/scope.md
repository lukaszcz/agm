# AgL Name Resolution

The scope pass performs full name resolution and records its results in side tables. Its collection pre-passes key declarations by module, named-scope path, and name; the empty path is the module root. They collect agents, functions, types, and constructors before expression bodies are resolved, so root declarations remain visible regardless of order and mutual recursion works.

## Namespace-Directed Resolution

Scope-region syntax and declaration-path shorthand cross the parser firewall in canonical AST forms. Collection materializes a `ScopeNode` layer and static member map for each named path, merging repeated regions, shorthand paths, and same-named type scopes. A type's enum variants are collected as members of that type scope, while record and exception construction retains its bare declaration spelling. One ordered chain resolver first considers exact local paths through the active scope layers, then import routes; type-owning segments emit the same `ConstructorRef` result as bare constructor candidates. The checker consumes that result for values and calls, validating variant shape and signatures without re-resolving names. Patterns and `is` tests record a valid owner through the same chain, while unresolved routes and clashes defer to the checker's historic qualification and enum diagnostics. The type environment applies the same local/module anchor and clash policy. `::` starts at the module root, and scope paths never suffix-match. Plain scope segments reject type arguments in expression, type, pattern, and `is` chains. Module-boundary selection is path-aware: scoped declaration paths are public atoms, selecting a scope expands its public subtree, and selection renames re-root that subtree. Atoms retain their defining module, so same-spelled scoped paths coexist until a route or bare use selects one.

Declared identity is structured, never spelled. A declaration is identified by its module, its scope path, and its unqualified name as three separate components — `ConstructorRef`, `BindingRef`, `NominalId`, and agent keys all carry them apart. A joined `A::Name` string is a display spelling produced for diagnostics and never reconstituted into an identity, so a scope path is never recovered by splitting a name. Passes that need a name in its declaring region re-enter that region instead: the type environment exposes a scope cursor that alias resolution, scoped function bodies, and constructor field lookup enter so unqualified sibling references resolve the way the declaration site sees them.

Resolution is namespace- and scope-directed, never capitalization-directed — a direct consequence of AgL's case-neutral name model:

- Built-in calls (`print`, `exec`, `ask`, and friends) are recognized by resolving the callee to a known built-in declaration, not by keyword.
- Constructors live in the value namespace; an ambiguous unqualified constructor name is a static error, disambiguated with `Type::Ctor` qualification. Arbitrary qualifier depths resolve local scope members where their exact path exists; a local qualifier chain that also contributes an imported module member is a clash with anchor repair guidance. Applied segments are checked against that chain's active lexical owner, never an unrelated same-suffixed scope.
- A declaration may claim a constructor's unqualified spelling exactly when the constructor stays reachable some other way: enum variants (reachable as `Owner::variant`) and constructors owned by another module (reachable by module qualification) yield to it, while a same-module record, exception, or alias constructor — whose declaration *is* the bare name — collides as a duplicate declaration.
- A nominal type alias contributes its target's constructor, so an alias chain ending at an enum contributes none — an enum's variants are its constructors, not the enum type itself. Such an alias still occupies its name as a type binding, so a value-position use gets the "type name, not a value" diagnostic rather than an undefined-name error.
- Scope records constructor candidates for bare pattern names independently of ordinary value bindings. Pattern slots are owned by match sites — a case branch or a `let` declaration — with metadata recording candidates, visible alternatives, and the requested resulting binder kind; `match_site_pattern_slots` groups each owner's slots so typechecking selects exactly the site it just classified. A case root bare name remains constructor-only, while a let root bare name always binds even when a constructor shares its spelling; nested bare names retain the field-directed policy. Candidate metadata retains whether a spelling can be a bare nullary enum pattern, allowing scope to reject definite duplicate binders eagerly while leaving genuine field-directed cases to typechecking. Typechecking selects each slot's final binder or constructor in checker-owned maps; consumers use the checked artifact's accessors for those meanings. No later pass rewrites scope's resolution tables. An `as`-pattern name always binds and `_` never binds.

Assignment follows the same split. Scope resolves an unqualified `:=` target and rejects an undeclared name, but leaves assignability to typechecking, which alone knows which binding a pattern slot selected. A qualified target is settled in scope, since no qualified name is a pattern slot: a local scope path is consulted first, through the same member-namespace resolver a qualified read uses, so a scoped `var` is assignable through its path and any other member of that path (a `let`, a `def`, a type, an agent) reuses the immutable-binder diagnostic; only when the qualifier is not a local path is a cross-module target attempted, and only `builtin var` is assignable across a module boundary. An indexed target (`obj[index] := value`) has no binding of its own: scope resolves `obj` and `index` as ordinary expressions, since indexed assignment mutates whatever container `obj` evaluates to rather than rebinding a name — legal on any array/dict-typed expression, not only a bare name.

Every `let` binder is identified by its own pattern node, whether the pattern is a
single name or a destructuring form. The `let` item's node identifies the match
site, never a binder. A `var` binder has no pattern and is identified by its
declaration node.

A scoped `let`/`var` is a member of its scope path, registered into the same
`ScopeNode` member map and duplicate check as static declarations — but during
the body walk, not the collection pre-pass, which only creates the path's
layer. That is what gives a binding textual precedence like a root-level
`let`, and makes a local `open`'s contribution of a binder, unlike a
declaration's, potentially precede the member it names: header placement puts
an `open` before the scope block it targets. A local `open` -- plain,
`using`, or `hiding` alike -- is therefore never snapshotted: the region
records the target path together with its selection (`LocalOpenSelection`),
and `resolve_bare_contribution`/`resolve_bare_constructor_contribution`
reapply that selection against the current `ScopeNode` tree at every bare
lookup, so a reference reached after the binder's own registration finds it
regardless of selection form, even though the `open` came first. After the
walk, filtered local selections are validated against the completed tree, so
an unknown `using` or `hiding` path is rejected even when no reference forced
a lookup. A cross-module `open`'s contribution is snapshotted eagerly, as before: an
imported module's public members are complete before the walk starts, so
there is nothing to defer.

## Import Environments

`scope/imports.py` is the pure import-policy seam. Its contribution environment
merges every declaration for a module into its selected path atoms, bare injection, aliases,
and plain-path routes. Scoped paths retain their structure through selection and re-exporting; policy expands scope-prefix selections and re-roots renamed subtrees. The selected set bounds both routes and bare injection: plain
imports are qualified-only, while `using` and `open import` inject bare names. A cross-module
`open` declaration uses the same eager structured bare-contribution layers on its enclosing
`ScopeNode`, with its constructor contributions retaining their owner path in a parallel
region-local candidate layer, so a nearer opened enum variant wins without changing global
bare-constructor candidates. A local `open`'s selection (`apply_open_selection`, shared with the
eager path) is applied live instead, against the scope tree; its constructor identity comes from
the resolved binding's own structured `(scope path, name)` lookup, the same one an ordinary
reference already uses, not a separate contribution layer. Local or directly imported scope
subtrees are selected there either way, and collisions remain deferred
to the use site. Value and type lookup both consume those layers, so selected and renamed
type members follow the same region boundaries and provenance; they never alter export maps
or import contributions. One shared suffix/anchored resolver serves value reads and writes,
constructors, and type qualification, retaining ambiguity and route identity until the use
site; bare candidates remain limited to open imports. Its diagnostics distinguish an
unknown route from a name outside a contribution.
One shared translator walks those verdicts and raises an error the caller constructs, so
the scope and typecheck passes share the walk while keeping their own exception types and
wording.
Constructor *owner* selection runs through the same ordered chain resolver: a chain ending at
a local type path, an opened contribution, or an imported route yields one `ConstructorRef`,
including the type-name versus module-route clash. Expression positions
raise that verdict as a scope error; pattern positions defer every failure to an empty candidate
set, because a pattern's owner cannot be settled before its subject type is known, leaving
typecheck to produce the more specific diagnostic.
Whitespace-separated qualifier near-misses are reported from the lexer's advisories
rather than reconstructed from AST shapes: when a reference fails to resolve at an offset
an advisory covers, the pass offers the tight spelling — but only when re-resolving that
route actually contributes the intended member, preserving valid division and
juxtaposition expressions.

`import` and `export` are also legal region items (`import` header-only, like `open`).
A region-scoped import still merges its selected members into the module-wide contribution
for qualifier routing, but `build_import_env` keeps its bare selection out of the shared
`unqualified` table and records it per declaration (`ImportEnv.decl_bare`) instead; the scope
pass snapshots that per-declaration set onto the importing region's own `ScopeNode`, the same
eager `contribute_bare` path a cross-module `open` already uses, so the narrowing rides existing
machinery rather than a new one. A region-scoped export re-roots the atoms it forwards under the
region's own scope path before they enter the module's export map (`scope/program.py`), mirroring
how a `using … as` rename already re-roots a selected atom. The header-only ordering check for
imports and exports applies uniformly to every module root and region, local to each
block-resolution call, so a region is one item for its enclosing block's ordering while its own
items get an independent check.

## Static Guarantees

Agents must be declared in source; the pass retains every declaration and
unused-agent warning by its scope path and member name, and binds it as a
first-class value of agent type, including named-scope members. A `def` whose
first parameter is `self` in a record, enum, or exception scope is classified as
a method and published in `method_declarations`, keyed by structured declaration
identity with its nominal owner path. Classification follows path collection, so
declaration shorthand and scope regions agree; aliases reject `self`, while an
annotated `self` outside a type scope remains an ordinary parameter. The pass
owns every rule about a classified receiver, including that it carries no
default value; the parser keeps only the token-order rules. Scoped
externs resolve their member names through the declaring module's companion,
and collection rejects duplicate scoped companion symbols.
`let _ = value` and `var _ = value` still resolve their right-hand sides but
register no binding, so `_` may be repeated. `_` never resolves as a readable
identifier, even when another binding form uses that name in an enclosing scope.
Register-backed `builtin var` declarations are admitted only in the canonical
`std/config` module, at its root or inside a named scope region; every
`builtin` form otherwise rides the same member/duplicate/visibility path as
an ordinary declaration, with no shape-specific placement logic of its own.
The pass enforces lexical control-flow boundaries —
`break`/`continue` must stay within a loop in the same function, `return` must
appear inside a function body — and the extern (Python FFI) placement rule that
externs are only allowed in file-backed modules.

Program resolution extends this pass across modules and preserves the loader's immutable,
reverse-topological import-SCC sequence on `ResolvedProgram`. Typecheck consumes that exact sequence
to publish closed inferred function signatures from dependency SCCs before importers, without
rebuilding the module graph. See [modules.md](agl/modules.md).

## Code Entry Points

- `src/agm/agl/scope/` — `resolve_module`, `resolve_program`, their resolution side tables, and the pure import-policy models in `imports.py`.
- Tests: `tests/test_agl_scope.py`, `tests/test_agl_scope_program.py`,
  `tests/test_agl_scope_imports.py`, `tests/test_agl_scope_contributions.py`,
  `tests/test_agl_namespace_wiring.py`, and `tests/test_agl_pattern_slots.py`.
