"""Host text crossing into AgL: the scalar check the companions share.

An OS path, name or variable reaches a companion as a Python ``str`` that the
host decoded with ``surrogateescape``, so it can hold a surrogate that AgL
``text`` must never carry.  The companions that return such text check it here
and raise their module's encoding exception rather than each repeating the
carrier construction.
"""

from __future__ import annotations

from typing import Protocol

from agm.agl.runtime.boundary import AglException
from agm.util.unicode import LoneSurrogateError, require_scalar_text, visible_text

__all__ = ["EncodingErrorFactory", "scalar_host_text"]


class EncodingErrorFactory(Protocol):
    """A synthesized ``std/errors::EncodingError`` class, as a companion sees it."""

    def __call__(self, *, message: str, raw: str) -> object: ...


def scalar_host_text(text: str, encoding_error: EncodingErrorFactory) -> str:
    """Return *text*, raising *encoding_error* when the host text is not valid Unicode.

    ``raw`` carries the escaped rendering, so reporting the failure cannot fail
    in turn.
    """
    try:
        return require_scalar_text(text)
    except LoneSurrogateError as exc:
        raise AglException(
            encoding_error(message=f"not valid Unicode: {exc}", raw=visible_text(text))
        ) from exc
