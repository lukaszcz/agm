"""Process execution helpers."""

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
