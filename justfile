set shell := ["bash", "-euo", "pipefail", "-c"]

default_prefix := env_var_or_default("HOME", "") + "/.local"
prefix := default_prefix
prompts_dir := justfile_directory() + "/prompts"

# Display the available recipes when no recipe is specified
[private]
default:
    @just --list

# Create the virtualenv and install the project with dev dependencies
setup:
    uv venv .venv --python 3.14
    uv pip install --python .venv/bin/python -e ".[dev]" --group dev

check_coverage := "100"

# Measure coverage through sys.monitoring rather than the C trace function.
# It is the difference between a suite that costs its own runtime again and one
# where measurement is nearly free. The trade is that it does not observe the
# arc into a `with` body; `[tool.coverage.report] partial_also` in pyproject.toml
# records those lines as one-sided so the gate stays at a real 100%.
coverage_core := "sysmon"

# Ceiling on any one test's CPU cost, in the configuration below. The most
# expensive test today costs about 12 CPU seconds, so this fails a test that
# regresses well past today's worst without tripping on scheduling noise.
# Run `just test-budget` to see the ranking this number is calibrated against.
check_cpu_budget := "15"

# Run the test suite
test:
    cleanup_coverage() { find . -maxdepth 1 -type f -name '.coverage*' -delete; }; trap cleanup_coverage EXIT; \
    COVERAGE_CORE={{coverage_core}} AGM_TEST_MAX_CPU_SECONDS={{check_cpu_budget}} \
    uv run python -m pytest tests/ -q -n auto --dist worksteal --cov=agm --cov=stdlib/std --cov-branch --cov-fail-under={{check_coverage}} --cov-report=term:skip-covered

test_cpu_budget := ""
test_report_top := "25"

# Cost is CPU seconds — this process plus the subprocesses it reaped — not wall
# clock, so a number means the same thing on a busy machine as on an idle one
# and is safe to enforce. Coverage measurement still shifts it, so a budget is
# only meaningful against this recipe's configuration; see tests/_durations.py.
#
# Report the most expensive tests, or fail any that overrun a CPU budget
test-budget:
    cleanup_coverage() { find . -maxdepth 1 -type f -name '.coverage*' -delete; }; trap cleanup_coverage EXIT; \
    COVERAGE_CORE={{coverage_core}} AGM_TEST_REPORT_TOP={{test_report_top}} AGM_TEST_MAX_CPU_SECONDS={{test_cpu_budget}} \
    uv run python -m pytest tests/ -q -n auto --dist worksteal --cov=agm --cov=stdlib/std --cov-branch --cov-fail-under=0 --cov-report=

# Re-run the suite with neutrally named temp directories, so any assertion that
# passes only because pytest names tmp_path after the test itself fails
test-neutral-tmp *args:
    AGM_TEST_NEUTRAL_TMP_PATH=1 uv run python -m pytest tests/ -q -n auto --dist worksteal --no-cov {{args}}

# Lint and check formatting with ruff
lint:
    uv run ruff check src/ tests/ stdlib/
    uv run ruff format --check src/ tests/ stdlib/

# Check for dead code in the application package
vulture:
    uv run vulture src/agm/ --min-confidence 80

# Install the AgL Emacs mode into the user's Emacs (requires emacs)
setup-emacs *args:
    uv run python tools/install_emacs_mode.py {{args}}

# Install the Emacs mode when emacs is available, else skip with a notice
[private]
setup-emacs-optional *args:
    if command -v emacs >/dev/null 2>&1; then \
        just setup-emacs {{args}}; \
    else \
        echo "emacs not found; skipping the AgL Emacs mode install"; \
    fi

# Run the Emacs mode's ERT suite (requires emacs)
test-emacs:
    emacs --batch -L config/emacs -l ert \
        $(for f in config/emacs/tests/*.el; do printf ' -l %s' "$f"; done) \
        -f ert-run-tests-batch-and-exit

# Run the micro syntax rules' test suite (requires go)
#
# -count=1 defeats the test cache: agl.yaml is data the harness reads, not a Go
# dependency, so a cached pass would otherwise survive editing the rules.
test-micro:
    cd config/micro/tests && go test -count=1 ./...

# Type-check with mypy
typecheck:
    MYPYPATH=src:stubs uv run mypy src/agm/ --strict --python-version 3.12

# Run type-checking, linting, dead-code checks, tests
check: typecheck lint vulture test

# Install the agm CLI into an isolated environment
install-agm:
    uv tool install --reinstall "{{justfile_directory()}}"

install *args:
    test -d "{{prompts_dir}}"
    install_prefix=""; \
    for arg in {{args}}; do \
        if [[ "$arg" != -* ]]; then install_prefix="$arg"; break; fi; \
    done; \
    if [[ -n "$install_prefix" ]]; then \
        UV_TOOL_BIN_DIR="$install_prefix/bin" uv tool install --reinstall "{{justfile_directory()}}"; \
    else \
        uv tool install --reinstall "{{justfile_directory()}}"; \
    fi
    uv run python tools/install_agm_config.py {{args}}
    just setup-emacs-optional {{args}}
