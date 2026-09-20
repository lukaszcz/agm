"""Unicode scalar-value helpers for every point where text enters AgL.

AgL ``text`` holds only Unicode scalar values: a surrogate code point
(U+D800-U+DFFF) never appears in one.  A Python ``str`` can hold a surrogate,
so each intake point -- an escape, a decoder, an OS name, a JSON document --
rejects one here rather than letting it surface as an encoding traceback at
whichever sink first writes bytes.

A standard-library leaf, so the lexer, the value-syntax reader, ``core``, the
runtime and the stdlib companions can all import it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import cast

__all__ = [
    "LoneSurrogateError",
    "holds_surrogate",
    "loads_json",
    "require_scalar_text",
    "surrogate_index",
    "visible_text",
]

# A JSON escape that can denote a surrogate: ``\uD800``-``\uDFFF``.
_SURROGATE_ESCAPE = re.compile(r"\\u[dD][89a-fA-F]")


class LoneSurrogateError(ValueError):
    """Text held the surrogate code point at *index*.

    The message spells the code point (``U+DCFF``) and never carries the
    character itself, so reporting the failure cannot fail in turn.
    """

    def __init__(self, code_point: int, index: int) -> None:
        super().__init__(f"lone surrogate U+{code_point:04X} at index {index}")
        self.index = index


def surrogate_index(text: str) -> int | None:
    """Return the index of the first surrogate in *text*, or ``None``.

    UTF-8 encoding is the surrogate test: it is the only code-point range the
    codec rejects, and it costs a fraction of a scan.  ASCII text -- the
    overwhelming majority -- answers from a cached flag without touching the
    characters at all.
    """
    if text.isascii():
        return None
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        return exc.start
    return None


def require_scalar_text(text: str) -> str:
    """Return *text*, or raise :exc:`LoneSurrogateError` if it holds a surrogate."""
    index = surrogate_index(text)
    if index is None:
        return text
    raise LoneSurrogateError(ord(text[index]), index)


def holds_surrogate(obj: object) -> bool:
    """Return whether *obj* holds a surrogate anywhere a decoded document can.

    Walks the containers JSON and TOML produce -- dict keys and values, lists,
    tuples -- and ignores every other leaf, none of which carries text.
    """
    if isinstance(obj, str):
        return surrogate_index(obj) is not None
    if isinstance(obj, dict):
        mapping = cast("dict[object, object]", obj)
        return any(holds_surrogate(key) or holds_surrogate(value) for key, value in mapping.items())
    if isinstance(obj, (list, tuple)):
        sequence = cast("Sequence[object]", obj)
        return any(holds_surrogate(item) for item in sequence)
    return False


def visible_text(text: str) -> str:
    """Return *text* with every surrogate spelled out (``\\udcff``).

    The result is encodable, so a diagnostic built from it always prints.  For
    diagnostics only -- error messages, host errors, trace records -- never for
    data, which must stay byte-exact or be rejected.
    """
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


def loads_json(
    doc: str,
    *,
    parse_float: Callable[[str], object] | None = None,
    parse_constant: Callable[[str], object] | None = None,
    object_pairs_hook: Callable[[list[tuple[str, object]]], object] | None = None,
) -> object:
    """Parse *doc* as JSON, rejecting a ``\\uD800``-style lone surrogate escape.

    Python's decoder combines an adjacent high+low escape pair into one scalar
    and keeps every other surrogate escape as a lone surrogate.  Since *doc* is
    itself scalar text -- every source of AgL text is checked -- a surrogate in
    the result can only have come from such an escape, so the escape pattern
    gates the walk and the walk decides.

    Raises :exc:`json.JSONDecodeError`, which every caller already maps to its
    own typed error.
    """
    result: object = json.loads(
        doc,
        parse_float=parse_float,
        parse_constant=parse_constant,
        object_pairs_hook=object_pairs_hook,
    )
    match = _SURROGATE_ESCAPE.search(doc)
    if match is not None and holds_surrogate(result):
        raise json.JSONDecodeError("lone surrogate escape", doc, match.start())
    return result
