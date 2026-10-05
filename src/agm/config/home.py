"""AGM home directory resolution; a leaf module ``agm.project.layout`` can import."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from agm.core.env import agm_installation_prefix, resolve_env
from agm.packages.layout import activation_index_path


def expand_env_root(override: str) -> Path:
    """Expand ``~`` in an environment root override and force it absolute.

    ``AGM_HOME`` / ``AGM_STDLIB`` relocate the whole AGM tree, so a relative
    value must be anchored to the current directory once (via ``abspath``)
    rather than silently re-resolving against wherever ``agm`` happens to be
    invoked from.  A leading ``~`` / ``~user`` is expanded first.
    """
    return Path(os.path.abspath(os.path.expanduser(override)))


def installation_agm_home() -> Path | None:
    """Return ``<installation prefix>/.agm`` when the prefix holds an activation index."""
    install_prefix = agm_installation_prefix()
    if install_prefix is None:
        return None
    installation_home = install_prefix / ".agm"
    if activation_index_path(installation_home).is_file():
        return installation_home
    return None


def agm_home_dir(*, home: Path, env: Mapping[str, str] | None = None) -> Path:
    """Return the AGM home directory (the ``.agm`` data/config root).

    ``AGM_HOME`` selects the directory explicitly. Otherwise, an installed
    package activation index beneath the executable's installation prefix
    identifies a complete prefix-local runtime tree. The fallback is
    ``home/.agm``.

    A leading ``~`` in ``AGM_HOME`` is expanded and a relative override is
    anchored to the current directory, so the resolved home is always
    absolute.
    """
    override = resolve_env(env).get("AGM_HOME")
    if override is not None and override.strip():
        return expand_env_root(override)

    installation_home = installation_agm_home()
    if installation_home is not None:
        return installation_home

    return home / ".agm"


def agm_path_candidates(
    *, home: Path, relative_path: Path, env: Mapping[str, str] | None = None
) -> list[Path]:
    """Return AGM-home-rooted candidates for *relative_path*, most-general first."""
    candidates: list[Path] = []
    install_prefix = agm_installation_prefix()
    if install_prefix is not None:
        candidates.append(install_prefix / ".agm" / relative_path)
    candidates.append(agm_home_dir(home=home, env=env) / relative_path)
    return list({path: None for path in candidates})
