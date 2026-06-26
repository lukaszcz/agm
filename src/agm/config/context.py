"""Current process context for config loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from agm.core.env import resolve_env
from agm.project.layout import discover_current_project_dir


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

    resolved_env = resolve_env(env)
    resolved_cwd = Path.cwd().resolve() if cwd is None else cwd.resolve()
    home = Path(resolved_env.get("HOME", "~"))

    try:
        proj_dir = discover_current_project_dir(resolved_cwd, env=resolved_env)
    except SystemExit:
        proj_dir = None

    return ConfigContext(home=home, proj_dir=proj_dir, cwd=resolved_cwd)
