"""AgL type-checking pass (Component 5).

Public API
----------
- :func:`check` — minimal-but-principled M1 type pass:
  ``ResolvedProgram × HostCapabilities → CheckedProgram``.
- :class:`CheckedProgram` — frozen dataclass with ``node_types``,
  ``contract_specs``, and ``warnings``.
- :class:`OutputContractSpec` — per-call codec + target-type record.
- :class:`AglTypeError` — fatal type error (span-aware ``AglError`` subclass).
"""

from __future__ import annotations

from agm.agl.typecheck.checker import check
from agm.agl.typecheck.env import (
    AglTypeError,
    CheckedProgram,
    OutputContractSpec,
)
from agm.agl.typecheck.types import (
    BoolType,
    DecimalType,
    DictType,
    EnumType,
    ExceptionType,
    IntType,
    JsonType,
    ListType,
    RecordType,
    TextType,
    Type,
)

__all__ = [
    "AglTypeError",
    "BoolType",
    "CheckedProgram",
    "DecimalType",
    "DictType",
    "EnumType",
    "ExceptionType",
    "IntType",
    "JsonType",
    "ListType",
    "OutputContractSpec",
    "RecordType",
    "TextType",
    "Type",
    "check",
]
