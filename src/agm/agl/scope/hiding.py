"""What ``hiding`` removes, by declaration identity.

Every way a declaration or scope is reached carries the declarations that way's
``hiding`` removes. Only :class:`~agm.agl.scope.lookup._Walk` decides the hidden
verdict from them; consumers outside it filter with the same predicate
(:func:`removed`). This leaf holds the types and the predicates.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from agm.agl.scope.symbols import DeclarationKey, QName, qname_declaration

__all__ = [
    "NOT_HIDDEN",
    "DeclarationNames",
    "Hiding",
    "Origin",
    "beneath_hiding",
    "hidden_keys",
    "removed",
    "removes",
    "unremoved",
]

Hiding: TypeAlias = frozenset[frozenset[DeclarationKey]]
"""What the ``hiding`` on each way a declaration is reached removes, by identity.

A declaration every way removes -- it or one above it -- is reached no way.
"""

#: Reached one way, hiding nothing.
NOT_HIDDEN: Hiding = frozenset({frozenset()})


class DeclarationNames(Protocol):
    """What declaration a key names."""

    def identity(self, key: DeclarationKey) -> DeclarationKey:
        """The declaration *key* names: a renaming alias's is its target's."""
        ...

    def denotes(self, key: DeclarationKey) -> object:
        """What *key* names in an ambiguity: its :meth:`identity`, or the type an alias denotes.

        Two aliases denoting one type are one, whatever their declarations.
        """
        ...


def removes(hiding: Hiding, key: DeclarationKey, names: DeclarationNames) -> bool:
    """Whether every way of *hiding* removes the declaration *key* names, or one above it.

    *names* names it: an alias denoting the type a removed alias denotes is
    removed too.
    """
    if hiding == NOT_HIDDEN:
        return False
    named = names.identity(key)
    denoted = names.denotes(key)
    return all(
        any(
            _beneath(named, hidden) or (denoted != named and names.denotes(hidden) == denoted)
            for hidden in way
        )
        for way in hiding
    )


def _beneath(key: DeclarationKey, above: DeclarationKey) -> bool:
    """Whether declaration *key* is *above*, or lies beneath it."""
    module, path, name = key
    return above[0] == module and (*path, name)[: len(above[1]) + 1] == (*above[1], above[2])


@dataclass(frozen=True, slots=True)
class Origin:
    """A scope or type a source reaches as a qualifier, with what the ways reaching it hide."""

    qname: QName
    hiding: Hiding = NOT_HIDDEN

    @property
    def key(self) -> DeclarationKey | None:
        """The declaration it names; ``None`` for a module's root, which declares nothing."""
        return qname_declaration(self.qname) if self.qname[1] else None


def removed(hiding: Hiding, key: DeclarationKey | None, names: DeclarationNames) -> bool:
    """Whether every way of *hiding* removes the declaration *key* (:func:`removes`).

    A ``None`` *key*, a module's root or a constructor without a path, is never removed.
    """
    return key is not None and removes(hiding, key, names)


def unremoved(origins: Iterable[Origin], names: DeclarationNames) -> frozenset[QName]:
    """The scopes and types of *origins* no ``hiding`` removes."""
    return frozenset(
        origin.qname for origin in origins if not removed(origin.hiding, origin.key, names)
    )


def beneath_hiding(
    owner: Hiding,
    owner_key: DeclarationKey | None,
    hiding: Hiding,
    key: DeclarationKey | None,
    names: DeclarationNames,
) -> Hiding:
    """What a declaration *key* reached with *hiding* hides, beneath an owner reached as *owner*.

    It is reached the ways the owner is; a way removing the owner removes it.
    """
    if owner == NOT_HIDDEN:
        return hiding
    removing = frozenset(
        way
        for way in owner
        if owner_key is not None and removes(frozenset({way}), owner_key, names)
    )
    return frozenset(
        way | also | ({key} if way in removing and key is not None else frozenset())
        for way in owner
        for also in hiding
    )


def hidden_keys[T](
    items: Iterable[T],
    read: Callable[[T], Iterable[QName]],
    removed_with: Callable[[QName], frozenset[DeclarationKey]],
) -> frozenset[DeclarationKey]:
    """The declarations a ``hiding``'s *items* remove: what each *read* names, and beneath it."""
    return frozenset(key for item in items for qname in read(item) for key in removed_with(qname))
