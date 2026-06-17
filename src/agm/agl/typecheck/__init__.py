"""AgL type-checking pass (Component 5).

Public API
----------
- :func:`check` — full type pass:
  ``ResolvedProgram × HostCapabilities → CheckedProgram``.
- :class:`CheckedProgram` — frozen dataclass with ``node_types``,
  ``contract_specs``, ``warnings``, and ``function_signatures``.
- :class:`OutputContractSpec` — per-call codec + target-type record.
- :class:`AglTypeError` — fatal type error (span-aware ``AglError`` subclass).
"""

from agm.agl.typecheck.checker import check
from agm.agl.typecheck.env import (
    AglTypeError,
    CallSiteRecord,
    CheckedProgram,
    FunctionSignature,
    OutputContractSpec,
    TypeEnvironment,
)
from agm.agl.typecheck.types import (
    AgentType,
    BoolType,
    BottomType,
    DecimalType,
    DictType,
    EnumType,
    ExceptionType,
    FunctionType,
    IntType,
    JsonType,
    ListType,
    RecordType,
    TextType,
    Type,
    UnitType,
)

__all__ = [
    "AgentType",
    "AglTypeError",
    "BoolType",
    "BottomType",
    "CallSiteRecord",
    "CheckedProgram",
    "DecimalType",
    "DictType",
    "EnumType",
    "ExceptionType",
    "FunctionSignature",
    "FunctionType",
    "IntType",
    "JsonType",
    "ListType",
    "OutputContractSpec",
    "RecordType",
    "TextType",
    "Type",
    "TypeEnvironment",
    "UnitType",
    "check",
]
