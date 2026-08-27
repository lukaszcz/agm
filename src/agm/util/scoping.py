"""Pure, generic context-variable scoping for AGM (stdlib-only, zero agm imports).

Provides:
- ``ScopedVar`` — bind a :class:`~contextvars.ContextVar` for a ``with`` block.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from contextvars import ContextVar, Token

T = TypeVar("T")


class ScopedVar(AbstractContextManager[None], Generic[T]):
    """Bind *var* to *value* for the ``with`` block, restoring it on exit.

    A plain object rather than a ``@contextmanager`` generator: the ambient
    sinks and per-call publications that use this sit on paths hot enough that
    the generator machinery is worth skipping. One instance covers one entry;
    build a fresh one per ``with``.
    """

    __slots__ = ("_token", "_value", "_var")

    _token: Token[T]

    def __init__(self, var: ContextVar[T], value: T) -> None:
        self._var = var
        self._value = value

    def __enter__(self) -> None:
        self._token = self._var.set(self._value)

    def __exit__(self, *_exc: object) -> None:
        self._var.reset(self._token)
