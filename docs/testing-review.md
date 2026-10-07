# Test-suite review

## Assessment

The suite has a strong behavioral backbone: real CLI processes, local Git repositories,
filesystem state, shell execution, whole-program AgL acceptance, static rejection, and
recovery workflows. Line/branch and command coverage are useful gates, but do not establish
that assertions distinguish correct behavior from an incorrect implementation.

This review combined suite-wide structural inspection with focused subsystem review,
mutation experiments, failure-first regressions, and runtime measurements. It was not an
exhaustive manual reading of every test or mutation of every production statement.

| Requirement | Assessment |
| --- | --- |
| Meaningful assertions | Confirmed weak oracles were repaired; targeted mutations now fail. |
| Success and error paths | Broad coverage, strengthened for rejection before effects, failed replacement, conflicts, and termination. |
| Edge cases | Additional finite-domain, falsey-value, delimiter, timestamp, encoding, and layout cases; exhaustive coverage is not claimed. |
| Complex workflows | Added stateful offline/sync, conflict/recovery, destructive close, config-copy, replacement/rollback, and session-exit workflows. |
| Real behavior | New workflows execute real AGM/Git/filesystem/shell logic; unavailable or unsafe external services remain faked. |
| Business requirements | Replaced sync/config-update call-order mocks with persisted state and subsequent-command observations; internal invariant tests remain where useful. |
| Every test below one second | **Not satisfied.** Cold compiler workflows still exceed one CPU second with branch coverage. |
| Repository testing policy | Changes use failing-first checks, isolated fixtures, external-boundary fakes, neutral paths, and no analysis suppressions. Complete legacy-test compliance is not claimed. |

## Confirmed findings and repairs

- **Enum decomposition was vacuous:** subjects used enum-owner identities instead of
  member identities, so every subject selected the wildcard. Correct member values now
  exercise explicit expected arms, including nested values and first-match priority
  (`test_agl_matchcompile_matrix_properties.py`).
- **Successful execution masked incorrect results:** cast tests discarded converted
  values, multifile selection omitted output, and a final dictionary `clear` concealed
  preceding no-ops. Assertions now inspect returned values, selected stdout, and each
  intermediate aliased dictionary state (`test_agl_ir_casts.py`, `test_agl_multifile.py`,
  `test_agl_stdlib_dict.py`).
- **A leak check could clean up and pass:** emergency `SIGKILL` concealed missing CLI
  child cleanup. A surviving process now fails before emergency cleanup (`test_e2e.py`).
- **Rollback assertions did not distinguish replacement content:** directory/archive
  replacements now change manifests, executable modules, resources, and file membership.
  Conflict and late external-installer failures must restore recorded file membership and
  contents, activation bytes, and executable old behavior (`test_packages_install.py`,
  `test_packages_python_deps.py`).
- **Later commands masked dry-run effects:** package dry-run workflows now immediately
  inspect archive, store, and activation state before another command can overwrite it
  (`test_e2e.py`).
- **Sync was tested through fabricated orchestration:** real Git workflows now establish
  fetch-only versus pull semantics, stale main/dependency worktree pruning, fetch-before-
  merge, stopping at a conflict, excluding outside worktrees, and resuming after resolution
  (`test_fetch.py`, `test_pull.py`, `test_e2e.py`).
- **Offline/destructive flags lacked a full workflow oracle:** CLI tests now distinguish
  cached from advanced remote refs for `--no-fetch`, and unmerged versus dirty worktrees
  for ordinary close, `-D`, `--force`, and `--keep-branch` (`test_e2e.py`).
- **Config update asserted internal calls:** real checkout/config/history tests now prove
  written dependency selections, preserved unrelated content, selective commits, and
  repeated-update idempotence (`test_config_update.py`).
- **Same-typed declarations had indistinguishable static oracles:** file, inline, and REPL
  executions now print distinct values for qualified declarations with identical types
  (`test_agl_qualified_declaration_output.py`).
- **Rejection expectations could combine unrelated errors:** located expectations now
  require one diagnostic matching both location and message fragments. Host-error and
  exit-code acceptance scenarios also check stdout, calls, and session lifecycle
  (`test_agl_e2e.py`, `tests/agl/programs/sessions/host_termination.*`).
- **A cache-witness test never reached cache hits:** reuse of one source root now proves
  cold, memory-hit, and disk-image observations; source edits still change the outcome
  (`test_agl_artifact_cache.py`).
- **Diagnostic assertions overfit production prose:** reviewed parser, REPL, registered-
  command, and source-identity checks now prefer classes, spans, severity, supplied
  identifiers, and caller-controlled formatting sentinels.
- **Core edge contracts lacked discriminators:** new real shell/filesystem and pure utility
  tests cover explicit environment/cwd, non-UTF-8 capture, symlink identity, replacement
  and resize with preserved timestamps, settle boundaries, private atomic-write streams,
  requested encoding, falsey BFS results, ordering, identifier delimiters, and optional
  interpolation context (`test_core_env.py`, `test_core_fs.py`, `test_util_*.py`).

Two production defects were also demonstrated by failing regressions:

1. `workspace open` fetched before refusing an occupied session. Rejection now precedes
   fetch and workspace mutation (`src/agm/commands/workspace/open.py`).
2. Embedded branch worktrees were classified as the main checkout merely because they
   lived underneath it. Checkout identity now determines the main workspace, so config
   copy from a nested branch directory uses branch overrides
   (`src/agm/project/layout.py`). Nested `REPO_DIR` overrides resolve to their checkout
   roots. Both split and embedded layouts are exercised.

## Mutation evidence

The repeatable semantic sample covers `agm.core.env`, `agm.core.fs`, `agm.util.graph`,
`agm.util.ident`, and `agm.util.interp`. It contains 54 compile-valid, deliberately
behavior-changing, single-site AST mutations.

| Focused test selection | Killed | Survived | Unreached | Invalid |
| --- | ---: | ---: | ---: | ---: |
| Original tests (line trace) | 32 | 15 | 7 | 0 |
| Strengthened tests | 54 | 0 | 0 | 0 |

The original classifications were independently replayed from unchanged test snapshots
with the former line-trace harness. The final kills comprise 52 assertion failures and two
runtime exceptions during test
calls. Import/collection/setup/teardown errors, infrastructure failures, and timeouts are
not counted as kills. AST evaluation markers link reachability to individual test nodeids;
a kill requires the failing test itself to evaluate the mutated site, not merely execute
its source line. Import-time evaluation and other passing tests cannot establish a kill.
Replaying all 54 strengthened-suite mutants with these markers preserved the classifications.
These are **selected-suite results, not a repository-wide mutation score**: other test files
might already detect a focused-suite survivor.

Additional directed probes verified enum discrimination, skipped selected-program
execution, no-op dictionary mutations, incorrect casts, rollback restoration, leaked
children, ignored offline flags, dry-run side effects, conflict continuation, omitted
pruning, close flags, and config-layer selection. Several passed the old oracle and failed
the strengthened oracle. These directed probes are separate from the 54-site manifest;
no combined mutation percentage is claimed.

Reproduce the maintained sample without rewriting production files:

```sh
uv run python tools/mutation_check.py --jobs 4 --report mutation-report
uv run python tools/mutation_check.py --ids M008 M019 --report mutation-report
```

`tools/mutations.json` pins source hashes, AST sites, replacements, semantic intent, and
relevant tests. Fresh unmutated selections must pass first. Changed source requires
reviewing the corresponding manifest entries; stale sites fail rather than silently
mutating something else. JSON reports preserve individual evidence and pytest output.

## Performance

Cost includes setup, call, teardown, reaped children, and CLI-launcher CPU. Wall-clock
measurements under parallel scheduling are not interchangeable with CPU cost.

- Three paired serial sysmon runs reduced summed median CPU for five decimal-boundary
  tests from 0.96s to 0.50s and two huge-integer extern probes from 0.53s to 0.17s. Real
  exact boundaries remain tested; unrelated huge powers use cheaper equivalent magnitudes.
- Qualifier tests now use the production imported-match artifact cache rather than
  repeatedly compiling imported match sites. In an isolated covered workflow, imported
  match compilation/validation calls fell from 741 to 93; observed test CPU changed from
  4.27–5.85s to 3.71–4.10s. Source-edit rejection remains verified. Run variability limits
  conclusions about aggregate speedup.
- Neither optimization proves a suite-wide speedup or the requested one-second ceiling.
  Genuine cold standard-library preparation and repeated REPL checks remain expensive.
  The existing gate is 15 CPU seconds, not one second. Reducing assertions, shifting setup
  outside accounting, or bypassing coverage would not constitute compliance.

## Final validation

The unchanged quality thresholds passed with eight xdist workers per suite:

```sh
PYTEST_XDIST_AUTO_NUM_WORKERS=8 AGM_TEST_REPORT_TOP=100 just check
PYTEST_XDIST_AUTO_NUM_WORKERS=8 just test-neutral-tmp
```

- `just check`: 23,842 tests passed; 100% configured line/branch coverage and the
  whole-suite command-coverage gate passed. Mypy, Ruff, AgL style, and vulture passed.
- Neutral temporary paths: all 23,842 tests passed.
- Maintained mutation replay: five unchanged baselines passed; all 54 mutants killed.
- The covered full-suite run took 684.99s; the neutral no-coverage run took 632.70s.
  They ran concurrently, so these are validation timings, not comparative benchmarks.
- Maximum observed covered test cost: **6.72 CPU seconds**, for the scoped-local-shadowing
  REPL legal-grouping workflow in `test_agl_repl_session.py`. The one-second requirement
  remains unmet despite passing the repository's existing 15-second gate.

## Remaining limits

A 54-site sample cannot establish mutation adequacy across the entire compiler, agent,
configuration, package, and workspace implementation. Broader capability/cache combinations,
FFI callback failures, cleanup combinations, package portability and project-pinned command
surfaces warrant additional behavioral sampling. Some legacy internal mocks and diagnostic
prose assertions remain outside the reviewed replacements. Exhaustive edge-case coverage,
complete policy compliance, and the one-second maximum are **not verified achieved**.
