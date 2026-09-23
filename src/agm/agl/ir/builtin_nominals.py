"""The host's source-selected built-in nominal and enum-member table.

Every host-minted value (a raised built-in exception, a structured ``exec``
``ExecResult``, an ``ask-request`` ``AgentRequest``, a nested ``Option`` member, ...) needs a
``NominalId`` to stamp on the value it constructs, and a spelling to display
for it. Rather than each minting site hardcoding
``NominalId(require_reserved_nominal_id("SomeType"))`` and the bare name,
it reads both from this table, keyed by the type's bare declared name.

This module holds only the table's shape and lookup, so ``ir/program.py``
(the typeless IR's data root) can carry a table without depending on the
typed ``semantics`` layer. Building a table's ``declared`` map from a
program's own ``builtin`` declarations is
``agm.agl.lower.lowerer.builtin_nominals_from_declarations``, which does need
that layer.
"""

from __future__ import annotations

import types
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from agm.agl.ir.ids import NominalId
from agm.agl.ir.reserved_nominals import (
    RESERVED_ENUM_MEMBER_IDS,
    RESERVED_NOMINAL_IDS,
    require_reserved_enum_member_id,
    require_reserved_nominal_id,
)

__all__ = [
    "NO_BUILTIN_DECLARATIONS",
    "BuiltinNominals",
    "DeclaredNominal",
    "resolve_standard_member_name",
]


@dataclass(frozen=True, slots=True)
class DeclaredNominal:
    """One ``builtin`` declaration's identity and declared source spelling.

    ``display_name`` is the scoped spelling the declaration was written
    under (e.g. ``"A::E"`` for a ``builtin exception E`` inside ``scope A``),
    distinct from the bare name it is keyed by in :attr:`BuiltinNominals.declared`.
    """

    nominal: NominalId
    display_name: str


@dataclass(frozen=True, slots=True)
class BuiltinNominals:
    """Source-selected identities for host-known types and enum members.

    ``declared`` and ``members`` follow the selected top-level builtin
    declarations. ``standard_members`` carries members used inside fixed
    standard host representations. Missing entries use reserved fallbacks.
    """

    declared: Mapping[str, DeclaredNominal]
    members: Mapping[tuple[str, str], DeclaredNominal] = field(default_factory=dict)
    standard_members: Mapping[tuple[str, str], DeclaredNominal] = field(default_factory=dict)

    def resolve(self, name: str) -> DeclaredNominal:
        """Return the identity and spelling a host mints for the built-in type *name*.

        A name present in :attr:`declared` answers with its declaration's own
        identity and its own declared scoped spelling. A name the program
        declares nothing for answers with the shipped standard library's own
        reserved identity for it (see ``ir.reserved_nominals``) and its bare
        name. Reserved ids are fallback identities, not identities assigned
        to a parsed standard-library declaration.

        Identity and spelling are resolved together so a value can never be
        stamped with one declaration's identity and another's spelling.
        """
        declared = self.declared.get(name)
        if declared is not None:
            return declared
        return DeclaredNominal(
            nominal=NominalId(require_reserved_nominal_id(name)), display_name=name
        )

    def nominal(self, name: str) -> NominalId:
        """Return the ``NominalId`` a host mints for the built-in type *name*."""
        return self.resolve(name).nominal

    def resolve_member(self, enum_name: str, member_name: str) -> DeclaredNominal:
        """Resolve a member of the selected top-level builtin enum."""
        declared = self.members.get((enum_name, member_name))
        if declared is not None:
            return declared
        return self._fallback_member(enum_name, member_name)

    def resolve_standard_member(self, enum_name: str, member_name: str) -> DeclaredNominal:
        """Resolve a member supplied through a standard nested host contract."""
        declared = self.standard_members.get((enum_name, member_name))
        if declared is not None:
            return declared
        return self._fallback_member(enum_name, member_name)

    @staticmethod
    def _fallback_member(enum_name: str, member_name: str) -> DeclaredNominal:
        return DeclaredNominal(
            nominal=NominalId(require_reserved_enum_member_id(enum_name, member_name)),
            display_name=f"{enum_name}::{member_name}",
        )

    def reverse(self, nominal: NominalId) -> tuple[str, str | None] | None:
        """Return the name(s) whose :meth:`resolve`/:meth:`resolve_standard_member` mint *nominal*.

        The inverse lookup: given an identity this table minted, returns the
        same ``name`` (with ``member_name`` ``None``) or ``(enum_name,
        member_name)`` pair another table's matching call would take, so a
        value carrying this table's identity can be restamped onto another
        table's (see :func:`~agm.agl.runtime.engine_config.restamp_engine_setting`).
        Returns ``None`` for an identity this table has no name for at all.
        """
        for name, entry in self.declared.items():
            if entry.nominal == nominal:
                return (name, None)
        for (enum_name, member_name), entry in self.members.items():
            if entry.nominal == nominal:
                return (enum_name, member_name)
        for (enum_name, member_name), entry in self.standard_members.items():
            if entry.nominal == nominal:
                return (enum_name, member_name)
        for name, reserved_id in RESERVED_NOMINAL_IDS.items():
            if reserved_id == nominal.value:
                return (name, None)
        for (enum_name, member_name), reserved_id in RESERVED_ENUM_MEMBER_IDS.items():
            if reserved_id == nominal.value:
                return (enum_name, member_name)
        return None


#: The table for a program with no ``builtin`` declarations of its own: every
#: name resolves to the shipped standard library's own identity.
NO_BUILTIN_DECLARATIONS = BuiltinNominals(
    declared=types.MappingProxyType({}),
    members=types.MappingProxyType({}),
    standard_members=types.MappingProxyType({}),
)


def resolve_standard_member_name(
    nominal: NominalId,
    enum_name: str,
    member_names: Iterable[str],
    nominals: BuiltinNominals,
) -> str | None:
    """Return which of *member_names* on *enum_name* carries *nominal*'s identity.

    Checks only *nominals*, the identity table of the program that produced
    the value: a value in a different table's identity (e.g. a REPL-persisted
    engine setting in the reserved fallback identity) is restamped onto
    *nominals* before reaching here. Returns ``None`` when *nominal* matches
    none of *member_names*.
    """
    for name in member_names:
        if nominal == nominals.resolve_standard_member(enum_name, name).nominal:
            return name
    return None
