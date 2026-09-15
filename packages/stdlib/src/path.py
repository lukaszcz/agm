"""Path manipulation externs for ``std/path``."""

from __future__ import annotations

import os
from pathlib import Path, PurePath

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


def stem(path: str) -> str:
    """Return *path*'s final component without its extension."""
    return os.path.splitext(os.path.basename(path))[0]


def extension(path: str) -> object:
    """Return *path*'s extension, including its leading dot, when it has one."""
    _, extension = os.path.splitext(path)
    return Option.Some(value=extension) if extension else getattr(Option, "None")()


def with_extension(path: str, extension: str) -> str:
    """Replace *path*'s extension with *extension*."""
    return str(Path(path).with_suffix(extension))


def with_name(path: str, name: str) -> str:
    """Replace *path*'s final component with *name*."""
    return os.path.join(os.path.dirname(path), name)


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


def is_under(path: str, base: str) -> bool:
    """Return whether *path* lexically resolves inside *base*, or is *base* itself."""
    return PurePath(os.path.normpath(path)).is_relative_to(os.path.normpath(base))


def parts(path: str) -> object:
    """Return the non-empty path components of *path*."""
    return array(Path(path).parts)


def expand_user(path: str) -> str:
    """Expand a leading ``~`` in *path* to the current user's home directory."""
    return os.path.expanduser(path)


def common_prefix(paths: list[str]) -> object:
    """Return the longest directory prefix shared by *paths*.

    An empty sequence, or absolute mixed with relative, has none.
    """
    try:
        return Option.Some(value=os.path.commonpath(paths))
    except ValueError:
        return getattr(Option, "None")()


def home() -> str:
    """Return the current user's home directory."""
    return str(Path.home())


__all__ = [
    "absolute",
    "basename",
    "common_prefix",
    "dirname",
    "expand_user",
    "extension",
    "home",
    "is_absolute",
    "is_under",
    "join",
    "normalize",
    "parts",
    "relative",
    "stem",
    "with_extension",
    "with_name",
]
