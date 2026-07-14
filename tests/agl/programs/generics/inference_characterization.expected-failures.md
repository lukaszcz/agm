# Expected pre-inference failures

Until the inference tasks land, every scenario in
`inference_characterization.scenarios.json` is expected to fail static checking
at the higher-order `app(id, 0)` workflow. The `id(raise ...)` rejection fixture
also remains red: the current checker lets it reach runtime instead of rejecting
its unresolved generic instantiation. The later implementation tasks replace
these red cases with their declared successful/static-rejection outcomes.
