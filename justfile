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
    uv venv .venv --python 3.12
    uv pip install --python .venv/bin/python -e ".[dev]" --group dev

# Run the test suite
check_coverage := "100"
test:
    uv run python -m pytest tests/ -q -n auto --cov=agm --cov-branch --cov-fail-under={{check_coverage}} --cov-report=term:skip-covered

# Lint and check formatting with ruff
lint:
    uv run ruff check src/ tests/
    uv run ruff format --check src/ tests/

# Check for dead code in the application package
vulture:
    uv run vulture src/agm/ --min-confidence 80

# Run the Emacs mode's ERT suite (requires emacs)
test-emacs:
    emacs --batch -L config/emacs -l ert \
        $(for f in config/emacs/tests/*.el; do printf ' -l %s' "$f"; done) \
        -f ert-run-tests-batch-and-exit

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
