"""Output codecs for the AgL runtime.

``OutputCodec`` is a protocol that every codec must satisfy.  In M1 the only
codec is ``TextCodec`` (passthrough).  The ``JsonCodec`` lands in M2.

Codec names (``TextCodec.name == "text"``) are the values used in
``HostCapabilities.codec_kinds`` and ``OutputContractSpec.codec_name``.
"""

from __future__ import annotations

from typing import Protocol

from agm.agl.eval.values import TextValue, Value
from agm.agl.typecheck.types import TextType, Type

# ---------------------------------------------------------------------------
# ParseResult — outcome of codec.parse()
# ---------------------------------------------------------------------------


class ParseResult:
    """The result of parsing a raw agent-response string through a codec.

    ``ok``         — True iff parsing and validation succeeded.
    ``value``      — The typed Value on success; ``None`` on failure.
    ``error_msg``  — A human-readable failure description (empty on success).
    """

    __slots__ = ("ok", "value", "error_msg")

    def __init__(self, *, ok: bool, value: Value | None, error_msg: str) -> None:
        self.ok = ok
        self.value = value
        self.error_msg = error_msg

    @classmethod
    def success(cls, value: Value) -> "ParseResult":
        return cls(ok=True, value=value, error_msg="")

    @classmethod
    def failure(cls, msg: str) -> "ParseResult":
        return cls(ok=False, value=None, error_msg=msg)


# ---------------------------------------------------------------------------
# OutputCodec protocol
# ---------------------------------------------------------------------------


class OutputCodec(Protocol):
    """Protocol for AgL output codecs.

    Every codec exposes:
    - ``name`` — the codec identifier (e.g. ``"text"``, ``"json"``).
    - ``supports_type(t)`` — True iff this codec can handle the given type.
    - ``parse(raw, target_type, strict_json)`` — parse a raw string.
    """

    @property
    def name(self) -> str: ...

    def supports_type(self, t: Type) -> bool: ...

    def parse(self, raw: str, target_type: Type, *, strict_json: bool = False) -> ParseResult: ...


# ---------------------------------------------------------------------------
# TextCodec — M1 passthrough codec
# ---------------------------------------------------------------------------


class TextCodec:
    """The built-in ``text`` codec: passthrough, no parsing needed.

    For a ``text`` target, the raw agent response is returned as-is, wrapped
    in a ``TextValue``.  ``strict_json`` is ignored (inapplicable).
    """

    @property
    def name(self) -> str:
        return "text"

    def supports_type(self, t: Type) -> bool:
        return isinstance(t, TextType)

    def parse(self, raw: str, target_type: Type, *, strict_json: bool = False) -> ParseResult:
        # Text codec: always succeeds; the raw string is the value.
        return ParseResult.success(TextValue(raw))
