"""Python ``re`` operations for ``std/regex``."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import NoReturn

from agl import AglException, array, dict, nominals, option_none, option_some

Match = nominals.std.regex.Match
RegexError = nominals.std.regex.RegexError


def _regex_error(pattern: str) -> NoReturn:
    raise AglException(RegexError(message="Could not compile regular expression.", pattern=pattern))


@lru_cache(maxsize=256)
def _compile(pattern: str) -> re.Pattern[str]:
    """Compile one Python ``re`` pattern, retaining a bounded companion-local cache."""
    try:
        return re.compile(pattern)
    except re.error:
        _regex_error(pattern)


def _match(match: re.Match[str]) -> object:
    groups = array(
        [option_some(value) if value is not None else option_none() for value in match.groups()]
    )
    named_groups = dict(
        {name: value for name, value in match.groupdict().items() if value is not None}
    )
    return Match(
        matched=match.group(),
        start=match.start(),
        end=match.end(),
        groups=groups,
        **{"named-groups": named_groups},
    )


def test(pattern: str, s: str) -> bool:
    """Return whether *pattern* occurs anywhere in *s*."""
    return _compile(pattern).search(s) is not None


def find(pattern: str, s: str) -> object:
    """Return the first match as ``Option``, if one occurs."""
    match = _compile(pattern).search(s)
    return option_some(_match(match)) if match is not None else option_none()


def find_all(pattern: str, s: str) -> object:
    """Return every non-overlapping match in left-to-right order."""
    return array([_match(match) for match in _compile(pattern).finditer(s)])


def replace(pattern: str, s: str, replacement: str) -> str:
    """Replace all matches using Python ``re`` replacement/backreference syntax."""
    return _compile(pattern).sub(replacement, s)


def split(pattern: str, s: str) -> object:
    """Split using Python ``re`` semantics, retaining captured separators."""
    return array([part if part is not None else "" for part in _compile(pattern).split(s)])


def escape(s: str) -> str:
    """Escape literal text for use as a Python ``re`` pattern."""
    return re.escape(s)


__all__ = ["escape", "find", "find_all", "replace", "split", "test"]
