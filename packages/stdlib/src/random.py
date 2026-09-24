"""Seedable random operations and UUID generation for ``std/random``."""

from __future__ import annotations

import random
import uuid as uuid_module
from decimal import Decimal
from typing import Protocol

from agl import nominals, runtime

from agm.agl.runtime.boundary import raise_index_error

IndexError = nominals.std.errors.IndexError


def _random() -> random.Random:
    """Return the pseudo-random source for the active AgL interpreter."""
    return runtime.state("std/random", random.Random)


class _MutableSequence(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> object: ...

    def __setitem__(self, index: int, value: object) -> None: ...


def seed(value: int) -> None:
    """Reset this module's pseudo-random sequence to *value*."""
    _random().seed(value)


def below(upper: int) -> int:
    """Return an integer in the half-open interval ``[0, upper)``."""
    return _random().randrange(upper)


def between(lower: int, upper: int) -> int:
    """Return an integer in the closed interval ``[lower, upper]``."""
    return _random().randint(lower, upper)


def uniform() -> Decimal:
    """Return a decimal in the half-open interval ``[0, 1)``."""
    return Decimal(str(_random().random()))


def choice(values: _MutableSequence) -> object:
    """Return one selected element, raising AgL's ``IndexError`` when empty."""
    if not values:
        raise_index_error(IndexError, "cannot choose from an empty array", 0, 0)
    return values[_random().randrange(len(values))]


def shuffle_in_place(values: _MutableSequence) -> None:
    """Shuffle a live AgL array view in place with Fisher-Yates."""
    for index in range(len(values) - 1, 0, -1):
        selected = _random().randrange(index + 1)
        values[index], values[selected] = values[selected], values[index]


def uuid() -> str:
    """Return a cryptographically random UUIDv4, independent of ``seed``."""
    return str(uuid_module.uuid4())


__all__ = ["below", "between", "choice", "seed", "shuffle_in_place", "uniform", "uuid"]
