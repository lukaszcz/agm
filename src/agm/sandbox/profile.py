"""Sandbox profile-name derivation from an argv."""

from __future__ import annotations

from pathlib import Path


def profile_name(argv0: str) -> str:
    """Return the sandbox profile name for an executable path or bare name."""

    return Path(argv0).name or argv0
