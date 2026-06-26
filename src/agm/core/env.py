"""Environment sourcing helpers."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values

from agm.core.process import exit_with_output


def resolve_env(env: Mapping[str, str] | None = None) -> Mapping[str, str]:
    """Return *env*, or the process environment when *env* is None (no copy)."""
    return os.environ if env is None else env


def clone_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a fresh mutable copy of *env* (process environment when None)."""
    return dict(os.environ if env is None else env)


_SHELL_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SHELL_ENV_ASSIGNMENT_SKIP_NAMES = frozenset(
    {
        "_",
        "BASHOPTS",
        "BASHPID",
        "DISPLAY",
        "EUID",
        "LOGNAME",
        "OLDPWD",
        "PPID",
        "PWD",
        "SHELL",
        "SHELLOPTS",
        "SHLVL",
        "TERM",
        "TMUX",
        "UID",
        "USER",
    }
)


def is_shell_identifier(name: str) -> bool:
    """Return whether *name* has shell variable identifier syntax."""

    return _SHELL_IDENTIFIER_RE.fullmatch(name) is not None


def is_safe_shell_env_assignment_name(name: str) -> bool:
    """Return whether AGM may safely emit or inject *name* as a shell env assignment."""

    return is_shell_identifier(name) and name not in _SHELL_ENV_ASSIGNMENT_SKIP_NAMES


def agm_installation_prefix() -> Path | None:
    """Return the AGM installation prefix inferred from the executable on PATH."""

    agm_executable = shutil.which("agm")
    if agm_executable is None:
        return None
    return Path(agm_executable).resolve().parent.parent


def source_env_files(
    paths: list[Path],
    env: dict[str, str] | None = None,
    *,
    cwd: Path | None = None,
) -> dict[str, str]:
    """Source bash files in order and return the resulting environment.

    The files are sourced in a single shell so later files can observe shell
    variables created by earlier files. Missing files are ignored. Any shell
    side effects still happen while sourcing.
    """

    base_env = clone_env(env)
    existing_paths = [str(path) for path in paths if path.is_file()]
    if not existing_paths:
        return base_env

    command = [
        "bash",
        "-c",
        'set -euo pipefail; for file in "$@"; do [[ -f "$file" ]] || continue; '
        'source "$file" >/dev/null; done; env -0',
        "agm-capture-env",
        *existing_paths,
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        cwd=cwd,
        env=base_env,
        check=False,
    )
    if result.returncode != 0:
        exit_with_output(
            result.returncode,
            result.stdout.decode("utf-8", errors="replace"),
            result.stderr.decode("utf-8", errors="replace"),
        )

    sourced_env: dict[str, str] = {}
    for entry in result.stdout.split(b"\0"):
        if not entry:
            continue
        key, _, value = entry.partition(b"=")
        sourced_env[key.decode("utf-8")] = value.decode("utf-8", errors="surrogateescape")
    return sourced_env

def load_dotenv_file(path: Path) -> dict[str, str]:
    """Load dotenv assignments from *path* without executing shell code."""

    if not path.is_file():
        return {}

    parsed = dotenv_values(path, encoding="utf-8")
    return {
        key: value if value is not None else ""
        for key, value in parsed.items()
    }


def load_dotenv_files(
    paths: list[Path],
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return *env* updated with dotenv values loaded from *paths* in order."""

    resolved_env = clone_env(env)
    for path in paths:
        resolved_env.update(load_dotenv_file(path))
    return resolved_env


def load_config_dotenv_files(
    config_dirs: list[Path],
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return *env* updated from ``.env`` and ``.env.local`` in each config dir."""

    resolved_env = clone_env(env)
    for config_dir in config_dirs:
        resolved_env = load_dotenv_files(
            [config_dir / ".env", config_dir / ".env.local"],
            resolved_env,
        )
    return resolved_env


def source_env_file(
    path: Path,
    env: dict[str, str],
    *,
    cwd: Path | None = None,
) -> dict[str, str]:
    """Source one env file if it exists."""

    return source_env_files([path], env, cwd=cwd)
