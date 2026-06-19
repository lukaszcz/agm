"""AGM configuration reader for AgL module roots.

Each configured root path retains the *origin directory* — the directory of
the config file that declared it — so the assembler can resolve relative paths
against the right base.

Config schema (in any of the layered ``config.toml`` files):

    [modules]
    lib_root = "~/.agm/lib"   # optional; overrides the default library root
    roots = [                  # optional; additional search roots
        "/absolute/path",
        "relative/to/config",
    ]

Layering follows the same order as all other AGM config:
    home/.agm/config.toml  →  project config/config.toml  →  cwd/.agm/config.toml

For ``lib_root``, later layers override earlier ones (last-write-wins).
For ``roots``, entries from *all* layers are accumulated (union).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agm.config.general import config_file_candidates
from agm.core.toml import load_toml_file, toml_dict


@dataclass(frozen=True)
class ModuleRootsConfig:
    """Resolved module-roots configuration from all config layers.

    Attributes
    ----------
    lib_root:
        The configured global library root as ``(raw_path_str, origin_dir)``,
        or ``None`` if no config file sets ``[modules] lib_root``.  The caller
        (``assemble_roots``) is responsible for applying the default
        ``~/.agm/lib`` when this is ``None``.  Relative *raw_path_str* values
        must be resolved against *origin_dir*.
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
