"""Shell and environment helpers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TextIO


def _write_stream(stream: TextIO, data: str) -> None:
    if data:
        stream.write(data)
        stream.flush()


def exit_with_output(returncode: int, stdout: str = "", stderr: str = "") -> None:
    """Forward captured output and exit with *returncode*."""
    _write_stream(sys.stdout, stdout)
    _write_stream(sys.stderr, stderr)
    raise SystemExit(returncode)


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

    base_env = dict(os.environ if env is None else env)
    existing_paths = [str(path) for path in paths if path.is_file()]
    if not existing_paths:
        return base_env

    command = [
        "bash",
        "-c",
        'set -euo pipefail; for file in "$@"; do [[ -f "$file" ]] || continue; source "$file"; done; env -0',
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


def source_env_file(
    path: Path,
    env: dict[str, str],
    *,
    cwd: Path | None = None,
) -> dict[str, str]:
    """Source one env file if it exists."""

    return source_env_files([path], env, cwd=cwd)


def run_foreground(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Run a command inheriting stdio."""

    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=os.environ if env is None else env,
        check=False,
    )
    return result.returncode


def run_capture(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """Run a command and capture stdout/stderr."""

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd,
        env=os.environ if env is None else env,
        check=False,
    )
    return result.returncode, result.stdout, result.stderr


def require_success(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run a command in the foreground and exit if it fails."""

    returncode = run_foreground(cmd, cwd=cwd, env=env)
    if returncode != 0:
        raise SystemExit(returncode)


def require_capture(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """Run a command, return stdout, and exit if it fails."""

    returncode, stdout, stderr = run_capture(cmd, cwd=cwd, env=env)
    if returncode != 0:
        exit_with_output(returncode, stdout, stderr)
    return stdout
