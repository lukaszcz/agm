set shell := ["bash", "-euo", "pipefail", "-c"]

default_prefix := env_var_or_default("HOME", "") + "/.local"
prefix := default_prefix

# Run the test suite
test:
    .venv/bin/python -m pytest tests/ -v

# Type-check with mypy
typecheck:
    MYPYPATH=src:stubs mypy src/agm/ --strict --python-version 3.12

# Run both tests and type-checking
check: test typecheck

# Install the agm CLI into an isolated environment
install-agm:
    uv tool install --reinstall "{{justfile_directory()}}"

install p=prefix: install-agm
    @bin_dir="{{p}}/bin"; \
    sandbox_dir="$HOME/.sandbox"; \
    mkdir -p "$bin_dir"; \
    mkdir -p "$sandbox_dir"; \
    for file in scripts/*.sh; do \
      script=$(basename $file) \
      src="{{justfile_directory()}}/scripts/$script"; \
      dst="$bin_dir/$script"; \
      install -m 0755 "$src" "$dst"; \
      echo "Installed $dst"; \
    done; \
    for sandbox_src in "{{justfile_directory()}}"/sandbox/*; do \
      [ -f "$sandbox_src" ] || continue; \
      sandbox_dst="$sandbox_dir/$(basename "$sandbox_src")"; \
      install -m 0644 "$sandbox_src" "$sandbox_dst"; \
      echo "Installed $sandbox_dst"; \
    done
