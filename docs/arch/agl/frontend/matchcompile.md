# AgL Match Compilation

The match compiler turns every checked source match site — a `case` or immutable `let` — into an immutable decision DAG. Lowering consumes the same decisions in action mode for cases and binding mode for lets. It is the last static pass: it runs after type checking, consumes checked pattern metadata only, and depends on nothing downstream — not lowering, the IR, the evaluator, or runtime services. See [index.md](agl/index.md) for the surrounding pipeline.

## Compilation Model

Source patterns are normalized from the checker's final binder/constructor classifications into typed pattern matrices, retaining every binder for each matched occurrence, and compilation decomposes the matrices into a decision DAG: it preserves source priority while choosing tests with a deterministic heuristic (the `qba` composite from Maranget's decision-tree paper), and shares decision nodes rather than expanding paths, so compiled decisions stay compact even for overlapping patterns. Enum, record, and boolean domains use complete checked signatures from the type table; scalar and type-variable domains remain open. A record is its nominal type's one closed constructor, with declaration-order fields and exact module/name/type-argument identity; enum variants and records share the field-bearing constructor machinery. When a selected closed signature has exactly one nominal constructor — every record and a true single-variant enum — the DAG emits `DecisionDecompose`: it exposes declaration-order child occurrences without a runtime discriminant test, records only child fields demanded by its continuation, and retains the parent/child provenance needed for dominance validation. `DecisionSwitch` remains the discriminant node for alternatives and open literal domains. Lowering materializes the resulting occurrence graph uniformly for case actions and let bindings.

## Diagnostics Cannot Disagree with Execution

The same DAG provides reachable-arm information and deterministic structured witnesses, so exhaustiveness and redundancy diagnostics are derived from the exact decision structure that will execute — they can never disagree with it. Diagnostics carry structured issues and witnesses adapted into ordinary static diagnostics. Enum and record witnesses retain a checked, import-aware source owner spelling so rendered patterns remain checker-accepted.

## Whole-Program Artifacts

Whole-program entry points visit every nested case and immutable let after type checking, including sites in all reachable modules — entry code never calling a site does not exempt it. Success yields a `MatchCompiledModule` or `MatchCompiledProgram` wrapping the exact checked artifact plus a total immutable match-site-to-DAG mapping. Each site preserves its source kind: cases retain their ordered arm actions, while a let retains one binding action plus its root occurrence provenance and checker-published matched type. Consumers filter the one immutable site mapping by source kind; there is no parallel case-only view.

A let is normalized as one row over its initializer occurrence. A reachable DAG failure is a static refutable-let diagnostic carrying the same structured missing-pattern witness as case exhaustiveness; accepted lets therefore have failure-free decisions. No runtime match failure, continuation capture, lowering, or evaluation is introduced at this stage. Any issue yields sorted static diagnostics and no artifact, so lowering can only ever see fully compiled programs. Downstream pipelines reuse a static artifact only when its resolved-program identity and host capabilities match the consuming pipeline; otherwise they recheck before lowering.

Artifact validation — source kind and ownership, mapping totality, provenance, and decision semantic replay — is a self-check gated by the AgL self-validation toggle ([testing.md](testing.md)), so the suite re-verifies every compiled match site while production lowering trusts the artifact.

## Package Boundary

The package API is deliberately limited to whole-program artifacts and stage entry points, structured issues/witnesses with their diagnostic adapters, and the small decision contract lowering consumes. Matrix and heuristic machinery, normalization, and validation helpers stay internal to their defining submodules and are reached only by white-box tests.

## References

The implementation follows Luc Maranget's pattern-matching compilation work:

- *Compiling Pattern Matching to Good Decision Trees* (ACM SIGPLAN Workshop on ML, 2008) — matrix specialization/default decomposition, the `qba` column-selection heuristic, and decision-node sharing.
- *Warnings for Pattern Matching* (Journal of Functional Programming 17(3), 2007) — the exhaustiveness/redundancy witness formulation; here the witnesses are reconstructed from the compiled DAG rather than computed by a separate usefulness pass.

## Code Entry Points

- `src/agm/agl/matchcompile/model.py` — canonical constructor identities, including singleton record constructors.
- `src/agm/agl/matchcompile/normalize.py` — checked patterns and closed signatures.
- `src/agm/agl/matchcompile/matrix.py` — matrix decomposition and column selection.
- `src/agm/agl/matchcompile/compiler.py` and `diagnostics.py` — decision DAGs, issues, and witnesses.
- `src/agm/agl/matchcompile/stage.py` — whole-program artifacts and diagnostic adaptation.
- `src/agm/agl/lower/` — the consumer side: decision DAGs lowered into switches, projections, case actions, or immutable let bindings ([execution/lowering.md](agl/execution/lowering.md)).
- Tests: `tests/test_agl_matchcompile_*.py`.
