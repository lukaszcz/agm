"""The host's built-in nominal table: bare built-in type name -> ``NominalId``.

Every host-minted value (a raised built-in exception, a structured ``exec``
``ExecResult``, an ``ask-request`` ``AgentRequest``, ...) needs a
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
from collections.abc import Mapping
from dataclasses import dataclass

from agm.agl.ir.ids import NominalId
from agm.agl.ir.reserved_nominals import require_reserved_nominal_id

__all__ = ["NO_BUILTIN_DECLARATIONS", "BuiltinNominals", "DeclaredNominal"]


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
    """Bare built-in type name -> the ``DeclaredNominal`` its ``builtin`` declaration has.

    ``declared`` holds only the names a program's own ``builtin``
    declarations gave an identity to.
    """

    declared: Mapping[str, DeclaredNominal]

    def nominal(self, name: str) -> NominalId:
        """Return the ``NominalId`` a host mints for the built-in type *name*.

        A name present in :attr:`declared` answers with its declaration's own
        identity. A name the program declares nothing for answers with the
        shipped standard library's own reserved identity for it (see
        ``ir.reserved_nominals``), which is the correct identity for it, not
        a placeholder for a missing lookup.
        """
        declared = self.declared.get(name)
        if declared is not None:
            return declared.nominal
        return NominalId(require_reserved_nominal_id(name))

    def display_name(self, name: str) -> str:
        """Return the source spelling a host mints for the built-in type *name*.

        A name present in :attr:`declared` answers with its own declared
        scoped spelling. A name the program declares nothing for answers with
        its own bare name — the shipped standard library's own declaration of
        a reserved name is always written bare, at ``std/core``'s root.
        """
        declared = self.declared.get(name)
        return declared.display_name if declared is not None else name

    def accumulate(self, update: "BuiltinNominals") -> "BuiltinNominals":
        """Return this table with *update*'s declarations folded in, *update* winning.

        A REPL session lowers one compile unit (entry) at a time, but reuses
        one link image across the whole session: an earlier entry's
        ``builtin`` declaration must stay live for a later entry that does
        not redeclare it, so the link image ACCUMULATES declarations across
        calls rather than being replaced wholesale by each call's own
        (necessarily partial) table. When the same name is declared again —
        e.g. a later entry redeclares it at a different scope path — *update*
        is the newer one and wins.
        """
        if not update.declared:
            return self
        merged = dict(self.declared)
        merged.update(update.declared)
        return BuiltinNominals(declared=types.MappingProxyType(merged))


#: The table for a program with no ``builtin`` declarations of its own: every
#: name resolves to the shipped standard library's own identity.
NO_BUILTIN_DECLARATIONS = BuiltinNominals(declared=types.MappingProxyType({}))
