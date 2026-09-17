# AgL Match Compilation

The match compiler turns every checked `case` into an immutable decision DAG that lowering executes. It is the last static pass: it consumes checked pattern metadata only and depends on nothing downstream.

## Compilation Model

Patterns are normalized from the checker's binder and constructor classifications into typed pattern matrices, then decomposed into a decision DAG that preserves source priority, chooses tests with Maranget's column-selection heuristic, and shares decision nodes rather than expanding paths. Enum, record, and boolean domains are closed, with signatures from the type table (each enum member keyed by its member-record declaration); scalar and type-variable domains are open. A single-constructor signature — any record, a single-member enum — decomposes without a runtime test; alternatives and literals become switches.

## Diagnostics Cannot Disagree with Execution

Reachable-arm information and missing-pattern witnesses are derived from the same DAG that will execute, so exhaustiveness and redundancy diagnostics can never disagree with runtime behavior. A refutable `let` is a static diagnostic carrying the same witness form as case exhaustiveness. Any issue yields diagnostics and no artifact, so lowering only ever sees fully compiled programs.

## Whole-Program Artifacts

The stage visits every case in every reachable module. The result wraps the checked artifact plus a total site-to-DAG mapping whose case-arm payloads are sealed. Artifact validation is a self-check under the self-validation toggle ([testing.md](../../testing.md)).

## References

- Luc Maranget, *Compiling Pattern Matching to Good Decision Trees* (ML Workshop 2008) — matrix decomposition, the `qba` heuristic, node sharing.
- Luc Maranget, *Warnings for Pattern Matching* (JFP 17(3), 2007) — the witness formulation; here witnesses are reconstructed from the compiled DAG.

## Code Entry Points

- `src/agm/agl/matchcompile/model.py`, `normalize.py` — constructor identities, checked patterns, closed signatures.
- `src/agm/agl/matchcompile/matrix.py`, `compiler.py`, `diagnostics.py` — decomposition, decision DAGs, issues and witnesses.
- `src/agm/agl/matchcompile/stage.py` — whole-program artifacts and diagnostic adaptation.
- `src/agm/agl/lower/` — the consumer ([execution/lowering.md](../execution/lowering.md)).
- Tests: `tests/test_agl_matchcompile_*.py`.
