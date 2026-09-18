# AgL Lowering and the Execution IR

## Lowering and Linking

Lowering consumes the match-compiled program and emits one linked, typeless executable program. It translates expressions directed by expected types, allocates program-local identities (symbols, functions, contracts, sources, nominals), and links modules in the loader's reverse-topological import-SCC order — function closures first, then static `let`/`var` initializers in source order — so forward references work. Each `program def`'s own value parameters become a host-facing signature, while marked static parameters retain their pre-allocated binding symbols and host decoders in the executable. Each emitted initializer records the source item it completes, which the REPL uses to promote declarations independently.

Imported modules use stable allocation namespaces, allowing their IR and linkable tables to be persisted independently (`lower/module.py`). Reuse validates source dependencies, capabilities, builtin identities, host-materialized contracts, and resource anchors; resource existence and containment are checked again. Whole-program metadata and initialization order are assembled for each invocation. REPL linking retains its session-owned allocation image ([../repl.md](../repl.md)).

Everything type-dependent is read from the checker's side tables — argument bindings, output contracts, selected constructors and methods, cast recipes, codec schema and decode walks, JSON encode plans — never re-inferred; that is what keeps the IR and evaluator typeless. Type arguments are erased. Nominal identity survives as `NominalId`, the checker's own declaration identity, with module, scope path, and name on the linked descriptor; shapes, including mutable fields and enum member layouts, come from the shared `TypeTable`.

Notable lowering shapes:

- **Built-in values.** Each typed runtime built-in reference lowers to a synthetic ordinary closure whose body uses the same dedicated host-operation IR as a direct call. Free functions, statics, unbound methods, and receiver-capturing method projections share this path. The closure exposes required parameters and evaluates omitted host defaults when invoked, while its occurrence-specific output contract stays compiled into the linked program. `resource-dir` follows this path; `resource` remains a direct link-time form.
- **Operator values.** `(op)` lowers to the same kind of synthetic closure over its checked operand types; its body is the operator IR a binary expression lowers to, shared through one operand-level helper, coerced to the closure's result type.
- **Methods.** A checker-selected ordinary method call lowers to a direct receiver-first call; a host-backed builtin method reuses its builtin route with the receiver as operand; `Session` operations lower to dedicated session nodes; a free `ask` lowers to the lazy default session. Leading-dot invocations already have lambda/member-call shape, so they lower to an ordinary unary closure around the selected method call.
- **Match sites.** `case` traverses the compiled decision DAG: singleton decisions project only the demanded fields, alternatives become `IrCase` keyed by member-record identity, and `IrField` is the single nominal projection. A named `let` is one bind instruction; `let _` evaluates and discards its initializer.
- **Mutation.** `IrAssign` stores a `var` cell; `IrIndexSet` mutates a container reached by reference; `IrFieldSet` writes a `var` field of a precisely typed record.
- **Externs.** An `extern def` lowers to a body carrying two names: the declared one, which every diagnostic and raised `ExternError` uses, and the Python companion name taken from scope's `extern_names` fact ([../frontend/scope.md](../frontend/scope.md)). Nothing is derived or mangled here.
- **Resources.** `resource`/`resource-dir` resolve while linking against the declaring module's filesystem anchor into absolute paths; a missing or escaping target stops linking.
- **Bindings.** Scoped `let`/`var` publish under their full path spelling in run results and REPL echo; functions are never published as bindings.

## The IR

The IR is a runtime-neutral data model: identities and source locations, a closed family of expression nodes, and a program container holding modules, symbols, functions, entry symbols, sources, nominals, contracts, the host-backed binding identities, and the dry-run inventory. `ask` and `exec` nodes carry typeless contract requests — codec selection, format instructions, JSON schema, decode walk. Each linked `program def` also gets a host-facing parameter signature (`program_signatures`, keyed by its entry symbol): name, zone, required-ness, and a decoder per declared value parameter. Nominal descriptors are closed over every record referenced by a retained enum, including shared members whose original fallback owner was superseded. The emitted symbol and function tables carry only descriptors owned by a module the program actually links — a REPL entry's persistent link state can hold allocations for a module a prior entry started but never completed (see [repl.md](../repl.md#retained-library-image)), and `lower_program` filters those out when assembling each new program. `validate_ir` checks structure cheaply; deep validation of the lowerer's own output runs only under the self-validation toggle ([testing.md](../../testing.md)).

## Code Entry Points

- `src/agm/agl/lower/lowerer.py`, `program.py` — expected-type-directed lowering and module linking; `coercions.py`, `conversions.py` — coercion and cast recipes; `repl.py` — incremental linking for the REPL.
- `src/agm/agl/ir/` — nodes, identities, contracts, the program container, builtin nominal and variable tables, `validate.py`.
- Tests: `tests/test_agl_lower.py`, `test_agl_case_lowering_decisions.py`, `test_agl_ir_*.py`.
