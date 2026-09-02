# AgL Lowering and the Execution IR

## Lowering and Linking

Lowering consumes the match-compiled program and emits one linked, typeless executable program. It translates expressions directed by expected types, allocates program-local identities (symbols, functions, contracts, sources, nominals), and links modules in the loader's reverse-topological import-SCC order — function closures first, then static `let`/`var` initializers in source order — so forward references work. Params from every module become descriptors installed before initialization. Each emitted initializer records the source item it completes, which the REPL uses to promote declarations independently.

Everything type-dependent is read from the checker's side tables — argument bindings, output contracts, selected constructors and methods, cast recipes, codec schema and decode walks, JSON encode plans — never re-inferred; that is what keeps the IR and evaluator typeless. Type arguments are erased. Nominal identity survives as `NominalId`, the checker's own declaration identity, with module, scope path, and name on the linked descriptor; shapes, including mutable fields and enum member layouts, come from the shared `TypeTable`.

Notable lowering shapes:

- **Methods.** A checker-selected ordinary method call lowers to a direct receiver-first call; a host-backed builtin method reuses its builtin route with the receiver as operand; `Session` operations lower to dedicated session nodes; a free `ask` lowers to the lazy default session.
- **Match sites.** Case and destructuring `let` traverse the same decision DAG: singleton decisions project only the demanded fields, alternatives become `IrCase` keyed by member-record identity, and `IrField` is the single nominal projection. A bare-name or `_` `let` is one bind instruction.
- **Mutation.** `IrAssign` stores a `var` cell; `IrIndexSet` mutates a container reached by reference; `IrFieldSet` writes a `var` field of a precisely typed record.
- **Resources.** `resource`/`resource-dir` resolve while linking against the declaring module's filesystem anchor into absolute paths; a missing or escaping target stops linking.
- **Bindings.** Scoped `let`/`var`/`param` publish under their full path spelling in run results and REPL echo; functions are never published as bindings.

## The IR

The IR is a runtime-neutral data model: identities and source locations, a closed family of expression nodes, and a program container holding modules, symbols, functions, entry symbols, sources, nominals, contracts, the host-backed binding identities, and the dry-run inventory. `ask` and `exec` nodes carry typeless contract requests — codec selection, format instructions, JSON schema, decode walk. Each linked `program def` also gets a host-facing parameter signature (`program_signatures`, keyed by its entry symbol): name, zone, required-ness, and a decoder per declared value parameter. `validate_ir` checks structure cheaply; deep validation of the lowerer's own output runs only under the self-validation toggle ([testing.md](../../testing.md)).

## Code Entry Points

- `src/agm/agl/lower/lowerer.py`, `program.py` — expected-type-directed lowering and module linking; `coercions.py`, `conversions.py` — coercion and cast recipes; `repl.py` — incremental linking for the REPL.
- `src/agm/agl/ir/` — nodes, identities, contracts, the program container, builtin nominal and variable tables, `validate.py`.
- Tests: `tests/test_agl_lower.py`, `test_agl_case_lowering_decisions.py`, `test_agl_ir_*.py`.
