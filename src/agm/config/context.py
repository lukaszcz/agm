"""Current process context for config loading."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from agm.project.layout import current_project_dir, is_project_dir


@dataclass(frozen=True)
class ConfigContext:
    """Resolved context used to load AGM configuration."""

    home: Path
    proj_dir: Path | None
    cwd: Path


def current_config_context(
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> ConfigContext:
    """Return config-loading paths for the current command invocation."""

    resolved_env = os.environ if env is None else env
    resolved_cwd = Path.cwd().resolve() if cwd is None else cwd.resolve()
    home = Path(resolved_env.get("HOME", "~"))

    raw_proj_dir = resolved_env.get("PROJ_DIR")
    if raw_proj_dir:
        proj_dir = Path(raw_proj_dir)
    else:
        try:
            discovered = current_project_dir(resolved_cwd)
        except SystemExit:
            proj_dir = None
        else:
            proj_dir = (
                discovered
                if discovered.resolve(strict=False) != resolved_cwd.resolve(strict=False)
                or is_project_dir(discovered)
                else None
            )

    return ConfigContext(home=home, proj_dir=proj_dir, cwd=resolved_cwd)
