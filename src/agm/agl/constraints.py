"""The structural type-constraint kinds (`Eq`, `Hashable`), shared vocabulary leaf.

A constraint block (`{Hashable K}`, `{Eq T}`) names one of these kinds on a
type parameter. Like ``zones``/``keywords``, this is a dependency-free leaf so
both ``syntax`` (parsing the block) and ``semantics`` (interpreting it, see
``semantics.type_table.satisfies``) can import it
without importing each other.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from types import MappingProxyType

__all__ = [
    "CONSTRAINT_SPELLINGS",
    "ConstraintBounds",
    "ConstraintKind",
    "close_constraints",
    "constraints_by_strength",
]


class ConstraintKind(enum.Enum):
    """A language-level structural constraint, spelled contextually like a type name."""

    EQ = "Eq"
    HASHABLE = "Hashable"


#: Spelling -> kind, the closed set of contextual names a constraint block recognizes.
CONSTRAINT_SPELLINGS: Mapping[str, ConstraintKind] = MappingProxyType(
    {kind.value: kind for kind in ConstraintKind}
)

#: In-scope type variables' constraints, by name — what a bare type variable
#: needs to satisfy ``semantics.type_table.satisfies``.
#: An absent name has no bound and satisfies neither; ``None`` in place of this
#: mapping instead means open-world mode, where a type variable anywhere
#: (along with the bottom and inference-variable types) counts as satisfied.
ConstraintBounds = Mapping[str, frozenset[ConstraintKind]]


def close_constraints(kinds: frozenset[ConstraintKind]) -> frozenset[ConstraintKind]:
    """Return *kinds* closed under implication: ``Hashable`` implies ``Eq``."""
    if ConstraintKind.HASHABLE in kinds:
        return kinds | {ConstraintKind.EQ}
    return kinds


#: Every kind, strongest first: each kind implies all that follow it (see
#: :func:`close_constraints`).
_STRENGTH_ORDER = (ConstraintKind.HASHABLE, ConstraintKind.EQ)


def constraints_by_strength(kinds: frozenset[ConstraintKind]) -> tuple[ConstraintKind, ...]:
    """Return *kinds* strongest first, so iteration never depends on hash order.

    A diagnostic that names the first unmet kind of an implication-closed set
    thereby names the one whose absence is the real obstacle.
    """
    return tuple(kind for kind in _STRENGTH_ORDER if kind in kinds)
