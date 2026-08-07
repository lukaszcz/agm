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

    def resolve(self, name: str) -> DeclaredNominal:
        """Return the identity and spelling a host mints for the built-in type *name*.

        A name present in :attr:`declared` answers with its declaration's own
        identity and its own declared scoped spelling. A name the program
        declares nothing for answers with the shipped standard library's own
        reserved identity for it (see ``ir.reserved_nominals``) and its bare
        name — the shipped library's own declaration of a reserved name is
        always written bare, at ``std/core``'s root. That is the correct
        answer for such a name, not a placeholder for a missing lookup.

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


#: The table for a program with no ``builtin`` declarations of its own: every
#: name resolves to the shipped standard library's own identity.
NO_BUILTIN_DECLARATIONS = BuiltinNominals(declared=types.MappingProxyType({}))
