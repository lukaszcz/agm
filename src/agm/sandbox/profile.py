"""Sandbox profile-name derivation from an argv."""

from __future__ import annotations

import shlex
from pathlib import Path


def profile_name(argv0: str) -> str:
    """Return the sandbox profile name for an executable path or bare name."""

    return Path(argv0).name or argv0


def profile_name_for_shell(command: str) -> str | None:
    """Return the sandbox profile name selected by a shell command's first word.

    Derived from ``shlex.split(command, posix=True)[0]`` through
    :func:`profile_name`. ``None`` when *command* is unsplittable (unbalanced
    quotes), empty, or its first word is empty -- the caller then falls
    through to the unqualified default chain, exactly as ``agm run`` does for
    an unrecognized name. An assignment prefix (``VAR=1 make``) or a leading
    ``(``/``{`` therefore selects that literal token as the profile name, an
    "unknown name" that falls through the same chain: never special-cased.
    """

    try:
        words = shlex.split(command, posix=True)
    except ValueError:
        return None
    if not words or not words[0]:
        return None
    return profile_name(words[0])
