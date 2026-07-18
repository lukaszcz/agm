# AgL Type System and Checking

The semantic type model lives in the `semantics` foundation package and is consumed by the typecheck pass. Alongside scalar, container, record, and enum types it carries what the expression-oriented design needs: a unit type for side-effecting expressions, function and agent types, a bottom type for divergence, and rigid type variables for generics. Solver-internal inference variables are structurally distinct from rigid source variables and never escape checked output.

## Nominal Types Are Handles; the TypeTable Holds Shapes

Record, enum, and exception types are lightweight handles — nominal identity is name plus owning module (plus type arguments), never structure — carrying no field or variant data of their own. Every shape lookup goes through the shared `TypeTable` (`semantics/type_table.py`), the single source of record/enum field-and-variant shapes and exception field/hierarchy shapes, consumed by typecheck, match compilation, compile-time schema derivation, and lowering. Built-in prelude and exception shapes are registered in the same table, so host-produced and source-constructed values share one definition.

Because handles resolve regardless of build order, recursive declarations — same-module, mutual, generic, and cross-module — are legal without any acyclicity requirement on the declaration graph.

## Whole-Table Analyses

Recursive types make some questions global. `semantics/analyses.py` answers them over the whole table, cached: **inhabitation** (rejecting a declaration that can never produce a finite value), **value-equality capability** (whether `=`/`!=` can be decided without unbounded recursion), and **finite-schema closure** (whether a type admits a finite JSON schema). The finite-schema result is consulted only at the sites that need a schema — agent/`exec` output targets, fallible cast targets, `param` types, and extern signatures — with a use-site error naming the offending declaration.

## The Typecheck Pass

The checker selects the concrete behavior the evaluator later relies on and publishes it in checked-program side tables so lowering never re-infers:

- **Built-in typing rules** are dispatched from the resolver's classification — for example `exec` chooses between returning a structured result and parsing stdout into a target type, and `ask` takes its result type from context.
- **Casts** are validated against a table of permitted source/target pairs, each classified as total or fallible.
- **Generics** use rank-1 parametric polymorphism: a generic definition is checked with rigid type parameters, and each occurrence is freshly instantiated with flexible variables inside an expression *region* handled by the shared solver (`typecheck/inference.py`). Region finalization resolves and validates before publishing, and type arguments are erased after checking. Partial applications are typed as function values with the same machinery.
- **Argument binding** goes through one routine (`typecheck/arguments.py`) shared by function calls, constructor calls, and constructor patterns; parameter zone markers are resolved to per-parameter kinds in the AST builder, so no later pass sees them. The checker records every resolved binding, the concrete post-substitution parameter types, and final field-directed bare-pattern classifications for lowering and match compilation.
- **Output contracts** — the schema and format metadata that agent/`exec` calls need — are computed once each region resolves its targets, then compiled into the typeless execution layer ([execution/index.md](agl/execution/index.md)).

An `extern def` shares the body-less builtin-`def` signature path, plus extern-only checks (the name must be a valid Python identifier; no function or agent type may occur in the signature); extern calls typecheck like ordinary calls.

A checked-output validation walk covers every type-bearing side table before anything crosses to lowering and seals the type environment; a leaked flexible variable is a compiler invariant failure, so checked artifacts contain concrete types and no solver state.

## Code Entry Points

- `src/agm/agl/semantics/` — the value model, semantic types, `TypeTable`, whole-table analyses, and exceptions.
- `src/agm/agl/typecheck/` — the checker, built-in typing rules, the inference solver, and argument binding.
- `src/agm/agl/type_schema.py` — compile-time JSON schema and format-instruction derivation.
- Tests: `tests/test_agl_typecheck.py`, `tests/test_agl_types.py`, `tests/test_agl_type_table.py`, `tests/test_agl_inference.py`, `tests/test_agl_arguments.py`, `tests/test_agl_typecheck_program.py`.
