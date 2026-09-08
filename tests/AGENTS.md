# Testing Policy

Prefer behavior tests that assert user-visible outcomes: command output, exit codes,
filesystem changes, git state, and invocations of external tools through the existing
fake-binary e2e harness.

Mock only at external boundaries such as subprocess helpers, `tmux`, `git`, `claude`,
`srt`, `shutil.which`, environment variables, the filesystem, and clocks. Avoid tests
whose primary assertion is that an internal AGM function was called with a specific
argument list or call order.

Parser-contract tests may mock handlers to verify the CLI surface maps accepted flags
and arguments to command fields. Private helper tests should remain only when the helper
is a pure, stable contract that is clearer to verify directly than through a command.

The e2e backbone in `tests/test_e2e.py` covers command behavior for all commands. Add new command-behavior coverage there before deleting lower-level tests for the same path.

## Instructions

- Keep tests fast. `just test` fails any test that overruns the CPU ceiling in its `check_cpu_budget`; `just test-budget` ranks tests by cost so the ceiling can be recalibrated, and `just test-budget test_cpu_budget=<seconds>` tries out a candidate number. Measure cost in CPU seconds, never wall clock — under `-n auto` a test's wall time tracks the load average rather than the test, while its CPU time varies by well under half.
- Do not let an assertion pass on the strength of its own test's name: pytest builds `tmp_path` from the test name, so a path in an error message can contain the word being asserted. `just test-neutral-tmp` re-runs the suite with neutrally named temp directories and fails any assertion that does.
