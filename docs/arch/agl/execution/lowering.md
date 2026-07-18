# AgL Lowering and the Execution IR

## Lowering and Linking

Lowering consumes a `MatchCompiledProgram` or `MatchCompiledModuleGraph` and emits one linked executable program. It translates expressions directed by expected types, allocates stable program-local identities (symbols, functions, contracts, sources, nominal types), and links modules in dependency order, initializing top-level function closures before ordinary initializers so forward references work. Type arguments are erased; nominal identity survives as module-qualified identity, with shapes resolved through the shared `TypeTable` from checking. Lowering reads the checker's side tables — argument bindings, parameter types, output contracts — rather than re-binding or re-inferring; that is what keeps the IR and evaluator typeless.

Match-compiled decision DAGs translate into nested one-level `IrCase` switches over a once-bound scrutinee, preserving decision sharing. Partial applications lower into ordinary closure descriptors with no dedicated IR node. Custom codec contracts are materialized before this boundary, while checker types are still available; only their typeless payload is embedded in IR.

## The IR

The IR is a runtime-neutral data model: program-local identities and source locations, a closed family of expression nodes, and a program container holding modules, symbols, functions, sources, nominals, contracts, and the dry-run inventory. Host operations carry typeless contract requests — codec selection, format instructions, JSON schema, and a decode walk — compiled from checker types during lowering; a recursive target type's decode walk mirrors its JSON Schema `$defs`/`$ref` shape so it closes exactly where the schema does.

The IR has exactly one loop primitive, `IrLoop`, plus `IrBreak`/`IrContinue` (and function-level `IrReturn`, which unwinds through loops). All richer loop forms — `for` over collections and ranges, `while` guards, `[n]` bounds, `until` — are desugared by the lowerer into ordinary pre-loop bindings and in-body checks around that single primitive; no richer loop node exists. The host's global `max-iters` safety valve caps only loops that are not self-bounded.

`validate_ir` provides node-local and deep whole-program validation tiers. Like match-artifact validation, it re-checks output the lowerer just produced from already-checked source, so it runs only under the AgL self-validation toggle ([testing.md](testing.md)) — never in production.

## Code Entry Points

- `src/agm/agl/lower/` — expected-type-directed lowering, loop and case desugaring, and module linking.
- `src/agm/agl/ir/` — the IR data model: identities, nodes, program container, contracts, and `validate_ir`.
- Tests: `tests/test_agl_lower.py`, `tests/test_agl_case_lowering_decisions.py`, `tests/test_agl_ir_*.py`.
