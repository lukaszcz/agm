Perform a thorough review of the test suite. Evaluate the general robustness and thoroughness of the test suite, as well as the following concrete aspects.
  - Do the tests actually make meaningful non-trivial assertions about the tested code?
  - Are error paths tested in addition to success paths?
  - Are all edge cases covered?
  - Are complex user workflows tested?
  - Is real behavior actually tested, not mocked?
  - Do the tests exercise real bussiness logic requirements instead of overfitting to implementation details?

Iterate on fixing all found issues until the test suite satisfies ALL of the
requirements above.
