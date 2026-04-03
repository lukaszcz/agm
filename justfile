set shell := ["bash", "-euo", "pipefail", "-c"]

default_prefix := env_var_or_default("HOME", "") + "/.local"
prefix := default_prefix
symlink := "false"

install p=prefix s=symlink:
    @bin_dir="{{p}}/bin"; \
    sandbox_dir="$HOME/.sandbox"; \
    mkdir -p "$bin_dir"; \
    mkdir -p "$sandbox_dir"; \
    for file in scripts/*.sh; do \
      script=$(basename $file) \
      src="{{justfile_directory()}}/scripts/$script"; \
      dst="$bin_dir/$script"; \
      if [ "{{s}}" = "1" ] || [ "{{s}}" = "true" ] || [ "{{s}}" = "yes" ]; then \
        ln -sfn "$src" "$dst"; \
        echo "Symlinked $dst -> $src"; \
      else \
        install -m 0755 "$src" "$dst"; \
        echo "Installed $dst"; \
      fi; \
    done; \
    for sandbox_src in "{{justfile_directory()}}"/sandbox/*; do \
      [ -f "$sandbox_src" ] || continue; \
      sandbox_dst="$sandbox_dir/$(basename "$sandbox_src")"; \
      if [ "{{s}}" = "1" ] || [ "{{s}}" = "true" ] || [ "{{s}}" = "yes" ]; then \
        ln -sfn "$sandbox_src" "$sandbox_dst"; \
        echo "Symlinked $sandbox_dst -> $sandbox_src"; \
      else \
        install -m 0644 "$sandbox_src" "$sandbox_dst"; \
        echo "Installed $sandbox_dst"; \
      fi; \
    done
