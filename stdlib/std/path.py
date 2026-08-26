"""Path manipulation externs for ``std/path``."""

from __future__ import annotations

import os
from pathlib import Path

from agl import array, nominals

Option = nominals.std.option.Option


def join(parts: list[str]) -> str:
    """Join path *parts* using the host platform's separator and ``""`` as its identity."""
    return os.path.join(*parts) if parts else ""


def dirname(path: str) -> str:
    """Return the directory component of *path*."""
    return os.path.dirname(path)


def basename(path: str) -> str:
    """Return the final component of *path*."""
    return os.path.basename(path)


def extension_option(path: str) -> object:
    """Return *path*'s extension, including its leading dot, when it has one."""
    _, extension = os.path.splitext(path)
    return Option.Some(value=extension) if extension else getattr(Option, "None")()


def with_extension(path: str, extension: str) -> str:
    """Replace *path*'s extension with *extension*."""
    return str(Path(path).with_suffix(extension))


def absolute(path: str) -> str:
    """Return an absolute lexical form of *path*."""
    return os.path.abspath(path)


def normalize(path: str) -> str:
    """Collapse redundant separators and dot components in *path*."""
    return os.path.normpath(path)


def relative(path: str, base: str) -> str:
    """Return the lexical path from *base* to *path*."""
    return os.path.relpath(path, base)


def is_absolute(path: str) -> bool:
    """Return whether *path* is absolute on the host platform."""
    return os.path.isabs(path)


def parts(path: str) -> object:
    """Return the non-empty path components of *path*."""
    return array(Path(path).parts)


def home() -> str:
    """Return the current user's home directory."""
    return str(Path.home())


__all__ = [
    "absolute",
    "basename",
    "dirname",
    "extension_option",
    "home",
    "is_absolute",
    "join",
    "normalize",
    "parts",
    "relative",
    "with_extension",
]
