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

from agm.config.general import (
    agm_home_dir,
    agm_path_candidates,
    config_file_candidates,
    expand_env_root,
)
from agm.core.env import resolve_env
from agm.core.toml import load_toml_file, toml_dict
from agm.util.interp import interp_preserving


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
            lib_root = (interp_preserving(lib_root_raw, os.environ)[0], origin_dir)

        # roots: accumulated across layers
        roots_raw = table.get("roots")
        if isinstance(roots_raw, list):
            for item in roots_raw:
                if isinstance(item, str) and item.strip():
                    extra.append((interp_preserving(item, os.environ)[0], origin_dir))

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


STDLIB_CONTRACT_MARKER_NAME = "STDLIB_CONTRACT"

# The contract id the running code expects a selected stdlib tree to declare
# in its top-level ``STDLIB_CONTRACT`` marker file. Bump this whenever a
# shipped stdlib change (a builtin type, a runtime-checked field, ...) is
# incompatible with an older installed tree, and update ``stdlib/STDLIB_CONTRACT``
# in the same change so a fresh ``just install`` ships a matching marker.
STDLIB_CONTRACT_ID = "1"


class StaleStdlibError(RuntimeError):
    """Raised when no candidate AgL standard-library root matches the current contract.

    Every default search-chain candidate (home, installation prefix, and the
    repository checkout) either was absent or carried a missing/mismatched
    ``STDLIB_CONTRACT`` marker, so returning one anyway would defer an opaque
    frontend failure to program load time instead of failing fast here.
    """

    def __init__(self, stale_path: Path) -> None:
        self.stale_path = stale_path
        super().__init__(
            f"AgL standard library at {stale_path} is out of date. Re-run `just install` "
            "to refresh it."
        )


def _stdlib_contract_matches(stdlib_root: Path) -> bool:
    """Return whether *stdlib_root* declares the expected ``STDLIB_CONTRACT`` id."""
    marker = stdlib_root / STDLIB_CONTRACT_MARKER_NAME
    if not marker.is_file():
        return False
    return marker.read_text(encoding="utf-8").strip() == STDLIB_CONTRACT_ID


def resolve_stdlib_root(*, home: Path, env: Mapping[str, str] | None = None) -> Path:
    """Return the selected AgL standard-library module root.

    The stdlib is a normal module tree installed under ``.agm/stdlib``.  An
    explicit ``AGM_STDLIB`` environment override wins outright (a leading ``~``
    is expanded and a relative override is anchored to the current directory)
    and is deliberately never marker-checked: it is the escape hatch for
    pointing at a synthetic or in-progress tree, which tests and manual
    debugging both rely on.

    Otherwise a user-writable home stdlib wins when present (honouring
    ``AGM_HOME``), then an installation-prefix stdlib, then the repository
    ``stdlib/`` tree for source-checkout workflows. Every one of those
    candidates is checked against :data:`STDLIB_CONTRACT_ID` via its
    top-level ``STDLIB_CONTRACT`` marker file, regardless of which other
    candidates exist; a candidate whose marker is missing or does not match
    is skipped in favour of the next one in precedence order. If none exists
    yet, return the home destination so diagnostics mention the path that
    ``just install`` populates. If at least one candidate directory exists
    but none is compatible, raise :class:`StaleStdlibError` naming the
    highest-precedence stale path instead of silently returning it.
    """
    override = resolve_env(env).get("AGM_STDLIB")
    if override is not None and override.strip():
        return expand_env_root(override)
    candidates = agm_path_candidates(home=home, relative_path=Path("stdlib"), env=env)
    repo_stdlib = Path(__file__).resolve().parents[3] / "stdlib"
    search_order = [*reversed(candidates), repo_stdlib]

    stale_path: Path | None = None
    for candidate in search_order:
        if not candidate.is_dir():
            continue
        if _stdlib_contract_matches(candidate):
            return candidate
        if stale_path is None:
            stale_path = candidate

    if stale_path is not None:
        raise StaleStdlibError(stale_path)
    return candidates[-1]
