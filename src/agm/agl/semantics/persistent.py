"""Mutable mapping handles backed by structurally shared immutable maps."""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from typing import Generic, TypeVar

from immutables import Map

K = TypeVar("K")
V = TypeVar("V")


class PersistentDict(MutableMapping[K, V], Generic[K, V]):
    """A mutable interface whose updates replace an immutable map snapshot."""

    def __init__(self, values: Map[K, V] | None = None) -> None:
        self._values: Map[K, V] = Map() if values is None else values
        self._changed: set[K] = set()

    def fork(self) -> PersistentDict[K, V]:
        """Create an independent mutable handle sharing this mapping's snapshot."""
        return PersistentDict(self._values)

    def __getitem__(self, key: K) -> V:
        return self._values[key]

    def __setitem__(self, key: K, value: V) -> None:
        self._values = self._values.set(key, value)
        self._changed.add(key)

    def __delitem__(self, key: K) -> None:
        self._values = self._values.delete(key)
        self._changed.add(key)

    def changed_values(self) -> tuple[V, ...]:
        """Return values introduced or replaced through this mutable handle."""
        return tuple(self._values[key] for key in self._changed if key in self._values)

    def __iter__(self) -> Iterator[K]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)
