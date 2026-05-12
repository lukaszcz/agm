"""Helpers for interpreting agent command responses."""

from __future__ import annotations


def last_response_line(output: str) -> str:
    """Return the final response line with surrounding whitespace removed."""

    lines = output.splitlines()
    if not lines:
        return output.strip()
    return lines[-1].strip()
