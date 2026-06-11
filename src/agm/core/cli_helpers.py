"""General CLI helper utilities shared across AGM commands.

These helpers are reusable across multiple commands and do not depend on any
DSL-specific types.
"""

from __future__ import annotations


def parse_key_value(item: str) -> tuple[str, str]:
    """Parse a ``key=value`` string into a ``(key, value)`` pair.

    The key is everything before the first ``=``; the value is everything after.
    Both leading and trailing whitespace in the key are stripped; the value is
    taken verbatim.

    Raises ``ValueError`` if the string contains no ``=`` or if the key is empty.
    """
    if "=" not in item:
        raise ValueError(
            f"Invalid key=value pair {item!r}: missing '='. "
            "Expected the form 'NAME=VALUE'."
        )
    key, _, value = item.partition("=")
    key = key.strip()
    if not key:
        raise ValueError(
            f"Invalid key=value pair {item!r}: key is empty. "
            "Expected the form 'NAME=VALUE'."
        )
    return key, value


def parse_inputs(items: list[str]) -> dict[str, str]:
    """Parse a list of ``key=value`` strings into a dictionary.

    Raises ``ValueError`` if any item is malformed or if the same key appears
    more than once.
    """
    result: dict[str, str] = {}
    for item in items:
        key, value = parse_key_value(item)
        if key in result:
            raise ValueError(
                f"Duplicate input key {key!r}. Each --input key must appear at most once."
            )
        result[key] = value
    return result
