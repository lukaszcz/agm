set shell := ["bash", "-euo", "pipefail", "-c"]

default_prefix := env_var_or_default("HOME", "") + "/.local"
prefix := default_prefix

# Create the virtualenv and install the project with dev dependencies
setup:
    uv venv .venv --python 3.12
    uv pip install --python .venv/bin/python -e ".[dev]" --group dev

# Run the test suite
test:
    uv run python -m pytest tests/ -q

# Lint with ruff
lint:
    uv run ruff check src/ tests/

# Type-check with mypy
typecheck:
    MYPYPATH=src:stubs uv run mypy src/agm/ --strict --python-version 3.12

# Run linting, tests, and type-checking
check: lint test typecheck

# Install the agm CLI into an isolated environment
install-agm:
    uv tool install --reinstall "{{justfile_directory()}}"

install *args: install-agm
    uv run python tools/install_agm_config.py {{args}}
