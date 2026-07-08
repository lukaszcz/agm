"""AGM configuration reader for AgL module roots.

Each configured root path retains the *origin directory* — the directory of
the config file that declared it — so the assembler can resolve relative paths
against the right base.

Config schema (in any of the layered ``config.toml`` files):

    [modules]
    lib_root = "~/.agm/lib"   # optional; overrides the AGM_HOME-relative default
    roots = [                  # optional; additional search roots
        "/absolute/path",
        "relative/to/config",
    ]

Layering follows the same order as all other AGM config:
    AGM home config.toml  →  project config/config.toml  →  cwd/.agm/config.toml

For ``lib_root``, later layers override earlier ones (last-write-wins).
For ``roots``, entries from *all* layers are accumulated (union).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from agm.config.general import agm_home_dir, agm_path_candidates, config_file_candidates
from agm.core.env import resolve_env
from agm.core.toml import load_toml_file, toml_dict


@dataclass(frozen=True)
class ModuleRootsConfig:
    """Resolved module-roots configuration from all config layers.

    Attributes
    ----------
    lib_root:
        The configured global library root as ``(raw_path_str, origin_dir)``,
        or ``None`` if no config file sets ``[modules] lib_root``.  Use
        :func:`resolve_lib_root` to convert this into a resolved ``Path``
        (applying ``~`` expansion and resolving relative paths against
        *origin_dir*).  Relative *raw_path_str* values that start with ``~``
        must be expanded before the is-absolute check.
    extra:
        Additional configured roots as ``(raw_path_str, origin_dir)`` pairs,
        accumulated across all config layers.  Relative paths resolve against
        their respective *origin_dir*.
    """

    lib_root: tuple[str, Path] | None
    extra: tuple[tuple[str, Path], ...]


def load_module_roots(
    *,
    home: Path,
    proj_dir: Path | None,
    cwd: Path,
) -> ModuleRootsConfig:
    """Read module-root configuration from all AGM config layers.

    Visits each config file in layering order (home → project → cwd).  For
    ``[modules] lib_root``, later files override earlier ones.  For
    ``[modules] roots``, entries from all files are accumulated.

    Each path retains the directory of the config file that declared it as its
    *origin_dir*, enabling the assembler to resolve relative paths correctly.
    """
    lib_root: tuple[str, Path] | None = None
    extra: list[tuple[str, Path]] = []

    for config_path in config_file_candidates(home=home, proj_dir=proj_dir, cwd=cwd):
        if not config_path.is_file():
            continue
        origin_dir = config_path.parent
        raw = load_toml_file(config_path)
        modules_section = raw.get("modules")
        if not isinstance(modules_section, dict):
            continue
        table = toml_dict(modules_section)

        # lib_root: last layer that sets it wins
        lib_root_raw = table.get("lib_root")
        if isinstance(lib_root_raw, str) and lib_root_raw.strip():
            lib_root = (lib_root_raw, origin_dir)

        # roots: accumulated across layers
        roots_raw = table.get("roots")
        if isinstance(roots_raw, list):
            for item in roots_raw:
                if isinstance(item, str) and item.strip():
                    extra.append((item, origin_dir))

    return ModuleRootsConfig(lib_root=lib_root, extra=tuple(extra))


def resolve_lib_root(
    mr_config: ModuleRootsConfig,
    *,
    home: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the ``lib_root`` from config into an absolute ``Path``.

    Expands ``~`` via :func:`os.path.expanduser` before checking whether the
    path is absolute, so a configured value like ``"~/mylib"`` is treated as
    absolute (rooted at the user's home directory) rather than relative to the
    config file's directory.

    Parameters
    ----------
    mr_config:
        The loaded module-roots configuration.

    Returns
    -------
    Path
        Resolved ``lib_root`` path (not yet canonicalized; ``assemble_roots``
        applies ``expanduser`` + ``resolve`` on its own paths).  When no
        ``lib_root`` is configured, returns the default AGM home ``lib``
        directory, honouring ``AGM_HOME`` when *home* is supplied (or when the
        process environment is consulted with the implicit ``Path.home()``).
    """
    if mr_config.lib_root is not None:
        raw_str, origin_dir = mr_config.lib_root
        raw_path = Path(os.path.expanduser(raw_str))
        return raw_path if raw_path.is_absolute() else origin_dir / raw_path
    default_home = Path.home() if home is None else home
    return agm_home_dir(home=default_home, env=env) / "lib"


def _has_legacy_constructor_syntax(stdlib_root: Path) -> bool:
    """Return whether *stdlib_root* is from before ``Type::Ctor`` syntax."""
    core = stdlib_root / "std" / "core.agl"
    return core.is_file() and "ParsePolicy.Abort" in core.read_text(encoding="utf-8")


def resolve_stdlib_root(*, home: Path, env: Mapping[str, str] | None = None) -> Path:
    """Return the selected AgL standard-library module root.

    The stdlib is a normal module tree installed under ``.agm/stdlib``.  An
    explicit ``AGM_STDLIB`` environment override wins outright.  Otherwise a
    user-writable home stdlib wins when present (honouring ``AGM_HOME``), then
    an installation-prefix stdlib, then the repository ``stdlib/`` tree for
    source-checkout workflows.  A legacy installed stdlib that still uses old
    constructor syntax is skipped when a source-checkout stdlib is available.
    If none exists yet, return the home destination so diagnostics mention the
    path that ``just install`` populates.
    """
    override = resolve_env(env).get("AGM_STDLIB")
    if override is not None and override.strip():
        return Path(os.path.expanduser(override))
    candidates = agm_path_candidates(home=home, relative_path=Path("stdlib"), env=env)
    repo_stdlib = Path(__file__).resolve().parents[3] / "stdlib"
    for candidate in reversed(candidates):
        if candidate.is_dir() and not (
            repo_stdlib.is_dir() and _has_legacy_constructor_syntax(candidate)
        ):
            return candidate
    if repo_stdlib.is_dir():
        return repo_stdlib
    return candidates[-1]
