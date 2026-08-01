"""The host's built-in nominal table: bare built-in type name -> ``NominalId``.

Every host-minted value (a raised built-in exception, a structured ``exec``
``ExecResult``, an ``ask-request`` ``AgentRequest``, ...) needs a
``NominalId`` to stamp on the value it constructs. Rather than each minting
site hardcoding ``NominalId(PRELUDE_ID, "SomeType")``, it reads the identity
from this table, keyed by the type's bare declared name.

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
from agm.agl.modules.ids import PRELUDE_ID

__all__ = ["NO_BUILTIN_DECLARATIONS", "BuiltinNominals"]


@dataclass(frozen=True, slots=True)
class BuiltinNominals:
    """Bare built-in type name -> the ``NominalId`` its ``builtin`` declaration has.

    ``declared`` holds only the names a program's own ``builtin``
    declarations gave an identity to.
    """

    declared: Mapping[str, NominalId]

    def nominal(self, name: str) -> NominalId:
        """Return the ``NominalId`` a host mints for the built-in type *name*.

        A name present in :attr:`declared` answers with its declaration's own
        identity. A name the program declares nothing for answers with the
        shipped standard library's own declaration of that name —
        ``NominalId(PRELUDE_ID, name)`` — which is the correct identity for
        it, not a placeholder for a missing lookup.
        """
        declared = self.declared.get(name)
        return declared if declared is not None else NominalId(PRELUDE_ID, name)


#: The table for a program with no ``builtin`` declarations of its own: every
#: name resolves to the shipped standard library's own identity.
NO_BUILTIN_DECLARATIONS = BuiltinNominals(declared=types.MappingProxyType({}))
