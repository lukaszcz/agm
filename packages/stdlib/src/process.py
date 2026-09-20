"""Process operations for ``std/process``."""

import os
import socket

from agl import nominals

from agm.agl.runtime.host_text import scalar_host_text

EncodingError = nominals.std.errors.EncodingError

_MIN_EXIT_CODE = 0
_MAX_EXIT_CODE = 255


def exit(code: int = 0) -> None:
    """End the AgL host process with portable status *code*."""
    if not _MIN_EXIT_CODE <= code <= _MAX_EXIT_CODE:
        raise ValueError(f"exit code must be in {_MIN_EXIT_CODE}..{_MAX_EXIT_CODE}")
    raise SystemExit(code)


def cwd() -> str:
    """Return the current working directory."""
    return scalar_host_text(os.getcwd(), EncodingError)


def pid() -> int:
    """Return this process identifier."""
    return os.getpid()


def hostname() -> str:
    """Return this host's name."""
    return scalar_host_text(socket.gethostname(), EncodingError)


__all__ = ["cwd", "exit", "hostname", "pid"]
