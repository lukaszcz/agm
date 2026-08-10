# Configuration

AGM is configured through layered TOML files. Configuration is resolved relative to a *config context* — the home directory, the current project directory, and the invocation directory — and merged so that more specific scopes override more general ones. Commands read only the sections they need.

## Config Context

A config context locates the directories that contribute configuration. The project directory is discovered from an environment variable when set, otherwise by walking up the filesystem from the invocation directory until a recognizable project layout is found (see [workspaces.md](workspaces.md)). This context is what lets the same command behave correctly whether run from the main workspace or a branch worktree.

## Home Directory Overrides

The runtime selects one AGM home for config, prompts, sandbox settings, the AgL global library, and the package store. `AGM_HOME` selects it explicitly; otherwise AGM uses a populated `<install-prefix>/.agm` beside the executable, then falls back to `~/.agm`. This gives an explicit prefix installation one reachable runtime tree while preserving the user-home default. `AGM_STDLIB` overrides just the AgL standard-library root, taking precedence over every other stdlib candidate and skipping the store-version check described below (the deliberate escape hatch for synthetic or in-progress trees). Both environment overrides accept a leading `~`; project and workspace layers remain path-discovered as usual.

Stdlib resolution uses `AGM_STDLIB` as an unchecked escape hatch, then the active immutable `std` store package, then AGM's shipped fallback. A shared locator selects the wheel-bundled `agm/stdlib` tree in an installed artifact and the repository-root `stdlib/` tree in a source checkout. The active package must exactly match the running AGM version; a missing selected tree falls back to the shipped tree, while malformed activation or corrupt present tree fails cleanly and a version mismatch names both versions and directs the user to `just install`. `just install` force-refreshes and activates the managed lockstep `std` store tree, refusing symlinks and pruning stale files. With no prefix it honors `AGM_HOME`; an explicit prefix installs both the executable link and runtime tree under that prefix so runtime prefix discovery reaches the same package store. Unlike `config.toml`, sandbox templates, and prompts, this tree is not user-editable.

## Layering and Precedence

General configuration merges across scopes, from least to most specific:

1. the installation prefix's `.agm/config.toml`
2. the selected AGM home's `config.toml` when distinct
3. the project's config directory
4. the workspace-local `.agm/config.toml`

Later layers override earlier ones; table-valued sections merge by key rather than wholesale replacement. Path-valued settings resolve relative to the config file that defined them, with a sensible fallback to the invocation directory. They interpolate `%{VAR}` from the environment and expand `~`. Unresolved or malformed holes remain verbatim: one config file serves every command, so an irrelevant section must not prevent loading; any resulting failure follows the consuming command's normal path semantics.

## Sections and Per-Command Overrides

Configuration is organized into sections consumed by specific features — for example loop, run, exec, module-root, and package-pin settings. `[packages]` maps package names to exact semantic versions; its merged pins override the global package activation selection for that invocation, and malformed pins fail when package roots are selected without making unrelated sections strict. Some commands additionally support per-command override sections (such as a per-command review or revise table) that merge over the base section, so a default can be set once and specialized for a particular command.

For AgL execution, four sources combine with a defined precedence:

- **Engine settings** (`default-agent`, `log`, `strict-json`, `max-iters`, `log-file`, `timeout`) — the `std/config` `builtin var` bindings:
  `source write (std/config::X := e) > CLI flag > qualified program table > [exec].X > engine default`
  Their names, value kinds, and consuming side come from the pure shared catalog in `config/engine_keys.py`, also consumed by AgL semantics, deep IR validation, and the AgL evaluator/REPL.
- **Param values** (`param NAME`):
  `agm exec` inventories the selected program module and its transitive imports, resolving every inventory param as `CLI flag > qualified config table > source default > required error`. The REPL uses those qualified config values for imported-module params; params declared directly at the prompt remain default-or-required.

`[exec]` holds global engine defaults with kebab field names (`default-agent`, `strict-json`, `max-iters`, `log-file`). `default-agent` is a quoted AgL `Agent` literal; it remains raw configuration data until `exec` or `repl` lazily parse and typecheck it. Qualified lookup in `config/qualified_keys.py` uses a module suffix plus declaration scope path, preserves layer provenance, and rejects a config leaf that maps to multiple routes or conflicting spellings for one key. AGM sections cannot route to modules. A loose file entry uses its stem as its module component, while a file executed directly from a development package retains its package-qualified module path; reserved loose-file stems are rejected only when that entry declares params. Engine keys live at a selected program's qualified table, while params use their own declaring module path. Inline `-c` params are CLI-only.

## Sandbox Configuration

Sandbox settings for `agm run` follow their own discovery and merge chain across the same install/home/project/workspace scopes, selecting a per-command settings file with a default fallback and patching project-specific write paths in. See [sandbox.md](sandbox.md).

## Code Entry Points

- `src/agm/config/context.py` defines the config context and project-directory discovery.
- `src/agm/config/general.py` loads path-normalized config layers, retains their precedence order alongside the conventional merged view, and exposes the per-feature config readers.
- `src/agm/config/command_config.py` resolves per-command override sections.
- `src/agm/config/engine_keys.py` is the pure data-leaf catalog of engine keys: each key's name, value kind, config accessor, host default, and consuming side (runtime-live — backed by a live interpreter field — versus host-consumed registers). Shared with host seed/default resolution, the AgL engine-key type registry that maps each kind to an AgL type, and the evaluator/REPL, which route a write by its consuming side. Its named trace-register projection and trace coupling helper keep the `log`/`log-file` pair explicit; `default-agent` remains a register-only `Agent` value with no host default.
- `src/agm/config/qualified_keys.py` is the pure qualified AgL config-key resolver. It consumes the retained general-config layers so qualified aliases retain their file provenance, enforcing unambiguous config-key suffixes while keeping AGM sections out of module matching.
- `src/agm/config/module_roots.py` resolves AgL module search roots from the `[modules]` config.
- `src/agm/stdlib_locator.py` locates the same shipped stdlib in source-checkout and wheel layouts.
- `src/agm/config/general.py` retains `[packages]` through the conventional merged view; `packages/activation.py` consumes its exact-version pins when selecting installed package roots.
- `src/agm/config/sandbox/` discovers and merges SRT sandbox settings.
- `config/` (repository root) holds the default config templates installed into `~/.agm/`.
