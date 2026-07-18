# AgL Match Compilation

The match compiler turns every checked source `case` into an immutable decision DAG — the artifact lowering consumes to emit executable switches. It is the last static pass: it runs after type checking, consumes checked pattern metadata only, and depends on nothing downstream — not lowering, the IR, the evaluator, or runtime services. See [index.md](agl/index.md) for the surrounding pipeline.

## Compilation Model

Source patterns are normalized into typed pattern matrices, retaining every binder for each matched occurrence, and compilation decomposes the matrices into a decision DAG: it preserves source priority while choosing tests with a deterministic heuristic (the `qba` composite from Maranget's decision-tree paper), and shares decision nodes rather than expanding paths, so compiled decisions stay compact even for overlapping patterns. Enum and boolean domains use their complete checked signatures from the type table; scalar and type-variable domains remain open.

## Diagnostics Cannot Disagree with Execution

The same DAG provides reachable-arm information and deterministic structured witnesses, so exhaustiveness and redundancy diagnostics are derived from the exact decision structure that will execute — they can never disagree with it. Diagnostics carry structured issues and witnesses adapted into ordinary static diagnostics, with rendering-only type qualification kept separate from the type checker's enum-owner forms.

## Whole-Program Artifacts

Whole-program entry points visit every nested case after type checking, including cases in all reachable modules — entry code never calling a case does not exempt it. Success yields a `MatchCompiledModule` or `MatchCompiledProgram` wrapping the exact checked artifact plus a total case-to-DAG mapping; any issue yields sorted static diagnostics and no artifact, so lowering can only ever see fully compiled programs. Downstream pipelines reuse a static artifact only when its resolved-program identity and host capabilities match the consuming pipeline; otherwise they recheck before lowering.

Artifact validation — source ownership, mapping totality, and decision semantics — is a self-check gated by the AgL self-validation toggle ([testing.md](testing.md)), so the suite re-verifies every compiled case while production lowering trusts the artifact.

## Package Boundary

The package API is deliberately limited to whole-program artifacts and stage entry points, structured issues/witnesses with their diagnostic adapters, and the small decision contract lowering consumes. Matrix and heuristic machinery, normalization, and validation helpers stay internal to their defining submodules and are reached only by white-box tests.

## References

The implementation follows Luc Maranget's pattern-matching compilation work:

- *Compiling Pattern Matching to Good Decision Trees* (ACM SIGPLAN Workshop on ML, 2008) — matrix specialization/default decomposition, the `qba` column-selection heuristic, and decision-node sharing.
- *Warnings for Pattern Matching* (Journal of Functional Programming 17(3), 2007) — the exhaustiveness/redundancy witness formulation; here the witnesses are reconstructed from the compiled DAG rather than computed by a separate usefulness pass.

## Code Entry Points

- `src/agm/agl/matchcompile/normalize.py` — checked patterns and closed signatures.
- `src/agm/agl/matchcompile/matrix.py` — matrix decomposition and column selection.
- `src/agm/agl/matchcompile/compiler.py` and `diagnostics.py` — decision DAGs, issues, and witnesses.
- `src/agm/agl/matchcompile/stage.py` — whole-program artifacts and diagnostic adaptation.
- `src/agm/agl/lower/` — the consumer side: decision DAGs lowered into one-level `IrCase` switches ([execution/lowering.md](agl/execution/lowering.md)).
- Tests: `tests/test_agl_matchcompile_*.py`.
