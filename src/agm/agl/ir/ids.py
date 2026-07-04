"""IR identity types for the AgL execution IR.

All types are immutable frozen dataclasses with slots.

Note: ``SourceId`` here is a linker-allocated integer handle into
``ExecutableProgram.sources``.  It is **distinct** from
``agm.agl.syntax.spans.SourceId`` (a label string); do not conflate or import
that one here.
"""

from __future__ import annotations

from dataclasses import dataclass

from agm.agl.modules.ids import ModuleId

__all__ = [
    "ContractId",
    "FunctionId",
    "Location",
    "NominalId",
    "SourceId",
    "SymbolId",
]


# ---------------------------------------------------------------------------
# Linker-allocated integer handles
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceId:
    """Linker-allocated integer handle into ``ExecutableProgram.sources``.

    Distinct from ``agm.agl.syntax.spans.SourceId`` (a label string).
    Unique within a single ``ExecutableProgram``.
    """

    value: int


@dataclass(frozen=True, slots=True)
class SymbolId:
    """Linker-allocated integer handle for a named binding (let/var/param).

    Unique within a single ``ExecutableProgram``.
    """

    value: int


@dataclass(frozen=True, slots=True)
class FunctionId:
    """Linker-allocated integer handle for a function definition.

    Unique within a single ``ExecutableProgram``.
    """

    value: int


@dataclass(frozen=True, slots=True)
class ContractId:
    """Linker-allocated integer handle for a contract (pre/post condition).

    Unique within a single ``ExecutableProgram``.
    """

    value: int


# ---------------------------------------------------------------------------
# Nominal identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NominalId:
    """Identity key for a named nominal type (record, enum, or exception).

    Allocated by the linker.  Equality and hash are by field values
    (``module_id`` + ``declared_name``), making two descriptors from the same
    module with the same declared name structurally equal — which is the
    correct behaviour since module + declared name uniquely identifies a type.

    """

    module_id: ModuleId
    declared_name: str


# ---------------------------------------------------------------------------
# Source location
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Location:
    """Source location carried by every IR node.

    ``source_id`` is a handle into ``ExecutableProgram.sources``.
    Offsets are byte (character) positions in the normalised source text;
    ``start_line`` and ``start_col`` are 1-based and 0-based respectively
    (matching the conventions used by the lexer/parser).
    """

    source_id: SourceId
    start_offset: int
    end_offset: int
    start_line: int
    start_col: int
