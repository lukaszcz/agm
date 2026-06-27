# Configuration

AGM is configured through layered TOML files. Configuration is resolved relative to a *config context* — the home directory, the current project directory, and the invocation directory — and merged so that more specific scopes override more general ones. Commands read only the sections they need.

## Config Context

A config context locates the directories that contribute configuration. The project directory is discovered from an environment variable when set, otherwise by walking up the filesystem from the invocation directory until a recognizable project layout is found (see [workspaces.md](workspaces.md)). This context is what lets the same command behave correctly whether run from the main workspace or a branch worktree.

## Layering and Precedence

General configuration merges across scopes, from least to most specific:

1. the installation prefix's `.agm/config.toml`
2. the user's `~/.agm/config.toml`
3. the project's config directory
4. the workspace-local `.agm/config.toml`

Later layers override earlier ones; table-valued sections merge by key rather than wholesale replacement. Path-valued settings are resolved relative to the config file that defined them, with a sensible fallback to the invocation directory.

## Sections and Per-Command Overrides

Configuration is organized into sections consumed by specific features — for example loop, run, exec, and module-root settings. Some commands additionally support per-command override sections (such as a per-command review or revise table) that merge over the base section, so a default can be set once and specialized for a particular command.

For AgL execution, three sources combine with a defined precedence: a program's in-source `config` pragmas, the `[exec]` config section, and CLI flags. CLI flags win, then source pragmas, then config files. This is described in [agl/repl.md](agl/repl.md).

## Sandbox Configuration

Sandbox settings for `agm run` follow their own discovery and merge chain across the same install/home/project/workspace scopes, selecting a per-command settings file with a default fallback and patching project-specific write paths in. See [sandbox.md](sandbox.md).

## Code Entry Points

- `src/agm/config/context.py` defines the config context and project-directory discovery.
- `src/agm/config/general.py` loads and merges the layered config and exposes the per-feature config readers.
- `src/agm/config/command_config.py` resolves per-command override sections.
- `src/agm/config/module_roots.py` resolves AgL module search roots from the `[modules]` config.
- `src/agm/config/sandbox/` discovers and merges SRT sandbox settings.
- `config/` (repository root) holds the default config templates installed into `~/.agm/`.
