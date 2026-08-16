# AgL Name Resolution

The scope pass performs full name resolution and records its results in side tables. Its collection pre-passes key source declarations by module, named-scope path, and name; the empty path is the module root. They collect functions, types, and constructors before expression bodies are resolved, so root declarations remain visible regardless of order and mutual recursion works. A parser-generated inline host entry is walked so its body resolves normally, but its synthetic function is excluded from declaration registration, lookup, import contributions, and exports. Because wrapping partitions static and executable items, references in the synthetic body also consult original source offsets before accepting local `let`/`var`/`param` bindings; partitioning therefore does not turn a later textual binding into a forward-visible one.

## Namespace-Directed Resolution

Scope-region syntax and declaration-path shorthand cross the parser firewall in canonical AST forms. Collection materializes a `ScopeNode` layer and static member map for each named path, merging repeated regions, shorthand paths, and same-named type scopes. A type's enum variants are collected as members of that type scope, while record and exception construction retains its bare declaration spelling. One ordered chain resolver first considers exact local paths through the active scope layers, then import routes; type-owning segments emit the same `ConstructorRef` result as bare constructor candidates. The checker consumes that result for values and calls, validating variant shape and signatures without re-resolving names. Patterns and `is` tests record a valid owner through the same chain, while unresolved routes and clashes defer to the checker's historic qualification and enum diagnostics. The type environment applies the same local/module anchor and clash policy. `::` starts at the module root, and scope paths never suffix-match. Plain scope segments reject type arguments in expression, type, pattern, and `is` chains. Module-boundary selection is path-aware: scoped declaration paths are public atoms, selecting a scope expands its public subtree, and selection renames re-root that subtree. Atoms retain their defining module, so same-spelled scoped paths coexist until a route or bare use selects one.

Declared identity is structured, never spelled. A declaration is identified by its module, its scope path, and its unqualified name as three separate components — `ConstructorRef`, `BindingRef`, and the nominal type handles (`RecordType`/`EnumType`/`ExceptionType`, `semantics/types.py`) all carry them apart, alongside the `decl_id` that is a handle's real identity for equality. A joined `A::Name` string is a display spelling produced for diagnostics and never reconstituted into an identity, so a scope path is never recovered by splitting a name. Passes that need a name in its declaring region re-enter that region instead: the type environment exposes a scope cursor that alias resolution, scoped function bodies, and constructor field lookup enter so unqualified sibling references resolve the way the declaration site sees them.

Resolution is namespace- and scope-directed, never capitalization-directed — a direct consequence of AgL's case-neutral name model:

- Built-in calls (`print`, `exec`, `ask`, and friends) are recognized by resolving the callee to a `builtin def` declaration, not by name alone. Resolution uses lexical lookup, region contributions, and the import environment, so aliases, re-exports, and REPL-retained bindings preserve their original builtin provenance. A receiver method may share a builtin's name but is reached only through qualification or member selection; it does not intercept the bare builtin call.
- Constructors live in the value namespace; an ambiguous unqualified constructor name is a static error, disambiguated with `Type::Ctor` qualification. Arbitrary qualifier depths resolve local scope members where their exact path exists; a local qualifier chain that also contributes an imported module member is a clash with anchor repair guidance. Applied segments are checked against that chain's active lexical owner, never an unrelated same-suffixed scope.
- A declaration may claim a constructor's unqualified spelling exactly when the constructor stays reachable some other way: enum variants (reachable as `Owner::variant`) and constructors owned by another module (reachable by module qualification) yield to it, while a same-module record, exception, or alias constructor — whose declaration *is* the bare name — collides as a duplicate declaration.
- A nominal type alias contributes its target's constructor, so an alias chain ending at an enum contributes none — an enum's variants are its constructors, not the enum type itself. Such an alias still occupies its name as a type binding, so a value-position use gets the "type name, not a value" diagnostic rather than an undefined-name error.
- Scope records constructor candidates for bare pattern names independently of ordinary value bindings. Pattern slots are owned by match sites — a case branch or a `let` declaration — with metadata recording candidates, visible alternatives, and the requested resulting binder kind; `match_site_pattern_slots` groups each owner's slots so typechecking selects exactly the site it just classified. A case root bare name remains constructor-only, while a let root bare name always binds even when a constructor shares its spelling; nested bare names retain the field-directed policy. Candidate metadata retains whether a spelling can be a bare nullary enum pattern, allowing scope to reject definite duplicate binders eagerly while leaving genuine field-directed cases to typechecking. Typechecking selects each slot's final binder or constructor in checker-owned maps; consumers use the checked artifact's accessors for those meanings. No later pass rewrites scope's resolution tables. An `as`-pattern name always binds and `_` never binds.

Assignment follows the same split. Scope resolves an unqualified `:=` target and rejects an undeclared name, but leaves assignability to typechecking, which alone knows which binding a pattern slot selected. A qualified target is settled in scope, since no qualified name is a pattern slot: a local scope path is consulted first, through the same member-namespace resolver a qualified read uses, so a scoped `var` is assignable through its path and any other member of that path (a `let`, a `def`, or a type) reuses the immutable-binder diagnostic; only when the qualifier is not a local path is a cross-module target attempted, and only `builtin var` is assignable across a module boundary. An indexed target (`obj[index] := value`) has no binding of its own: scope resolves `obj` and `index` as ordinary expressions, since indexed assignment mutates whatever container `obj` evaluates to rather than rebinding a name — legal on any array/dict-typed expression, not only a bare name.

Every `let` binder is identified by its own pattern node, whether the pattern is a
single name or a destructuring form. The `let` item's node identifies the match
site, never a binder. A `var` binder has no pattern and is identified by its
declaration node.

A scoped `let`/`var` is a member of its scope path, registered into the same
`ScopeNode` member map and duplicate check as static declarations during the
body walk. This gives bindings textual precedence like root bindings. Regional
import-tail contributions are snapshotted onto that region's bare-contribution
layer; `use` target resolution remains a separate follow-up concern.

`ScopeNode.members` is written only through `register_member`/`clear_members`.

## Import Environments

`scope/imports.py` is the pure import-policy seam. Each import declaration
contributes its own routes: a plain path and its suffixes, or an alias route,
expose every public atom except that declaration's hidden paths. Repeated
imports union only where they share a route, so hiding remains effective on a
separate alias route. Positive tails never narrow qualified access.

A tailed import also contributes bare atoms. `::*` contributes every non-hidden
public atom; explicit tail atoms contribute selected subtrees, and a rename adds
a spelling without removing the original. Root contributions populate
`ImportEnv.unqualified`; regional contributions remain in `ImportEnv.decl_bare`
and are snapshotted onto the importing `ScopeNode`. Bare collisions remain
use-site errors. The shared suffix/anchored resolver filters each candidate
route by its public-minus-hidden atoms before reporting ambiguity.
One shared translator walks those verdicts and raises an error the caller constructs, so
the scope and typecheck passes share the walk while keeping their own exception types and
wording.
Constructor *owner* selection runs through the same ordered chain resolver: a chain ending at
a local type path, a bare tail contribution, or an imported route yields one `ConstructorRef`,
including the type-name versus module-route clash. Expression positions
raise that verdict as a scope error; pattern positions defer every failure to an empty candidate
set, because a pattern's owner cannot be settled before its subject type is known, leaving
typecheck to produce the more specific diagnostic.
Whitespace-separated qualifier near-misses are reported from the lexer's advisories
rather than reconstructed from AST shapes: when a reference fails to resolve at an offset
an advisory covers, the pass offers the tight spelling — but only when re-resolving that
route actually contributes the intended member, preserving valid division and
juxtaposition expressions.

`import` and `export` are legal region header items. A region-scoped import
keeps its qualified routes module-wide while `build_import_env` records its tail
bare atoms per declaration for the importing region. A region-scoped export
re-roots forwarded atoms under the region path. Re-export propagation is bounded
by simple module paths, and header ordering is checked independently for every
module root and region.

## Static Guarantees

A `def` whose
first parameter is `self` in a record, enum, or exception scope is classified as
a method and published in `method_declarations`, keyed by structured declaration
identity with its nominal owner path. This applies equally to a `builtin def`;
typecheck later accepts only a method signature with a host dispatch contract.
Classification follows path collection, so declaration shorthand and scope regions
agree; aliases reject `self`, while an annotated `self` outside a type scope remains
an ordinary parameter. The pass
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
externs are only allowed in file-backed modules. The module's origin path also
travels on its `ModuleResolution`, so both this pass and typecheck can add the
inline-module explanation to a static-root rejection for file-less source that
declares its own `program def` — the one situation it describes. A module header
the inline entry transform moved into the synthetic body is reported as the
header-ordering violation it is, since the block it now sits in is not one the
source wrote.

Program resolution extends this pass across modules and preserves the loader's immutable,
reverse-topological import-SCC sequence on `ResolvedProgram`. Typecheck consumes that exact sequence
to publish closed inferred function signatures from dependency SCCs before importers, without
rebuilding the module graph. See [modules.md](../modules.md).

## Code Entry Points

- `src/agm/agl/scope/` — `resolve_program`, its per-module resolution side tables, and the pure import-policy models in `imports.py`. The pass runs only as part of whole-program resolution; there is no per-module entry point.
- Tests: `tests/test_agl_scope.py`, `tests/test_agl_scope_program.py`,
  `tests/test_agl_scope_imports.py`, `tests/test_agl_scope_contributions.py`,
  `tests/test_agl_namespace_wiring.py`, and `tests/test_agl_pattern_slots.py`.
