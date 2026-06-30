"""AgL type-checking pass (Component 5).

Public API
----------
- :func:`check` — single-module type pass:
  ``ResolvedProgram × HostCapabilities → CheckedProgram``.
- :func:`check_graph` — graph-wide type pass:
  ``ResolvedModuleGraph × HostCapabilities → CheckedModuleGraph``.
- :class:`CheckedProgram` — frozen dataclass with ``node_types``,
  ``contract_specs``, ``warnings``, and ``function_signatures``.
- :class:`CheckedModuleGraph` — graph output: per-module ``CheckedModule`` dict
  plus shared ``graph_type_table``.
- :class:`CheckedModule` — per-module analogue of ``CheckedProgram``.
- :class:`OutputContractSpec` — per-call codec + target-type record.
- :class:`AglTypeError` — fatal type error (span-aware ``AglError`` subclass).
"""

from agm.agl.semantics.types import (
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
    TypeVarType,
    UnitType,
    contains_type_var,
    free_type_vars,
    substitute,
)
from agm.agl.typecheck.checker import check
from agm.agl.typecheck.env import (
    AglTypeError,
    CallSiteRecord,
    CheckedProgram,
    ConstructorSignature,
    FunctionSignature,
    GenericTypeDef,
    OutputContractSpec,
    ParamSpec,
    TypeEnvironment,
)
from agm.agl.typecheck.graph import CheckedModule, CheckedModuleGraph, check_graph

__all__ = [
    "AgentType",
    "AglTypeError",
    "BoolType",
    "BottomType",
    "CallSiteRecord",
    "CheckedModule",
    "CheckedModuleGraph",
    "CheckedProgram",
    "ConstructorSignature",
    "DecimalType",
    "DictType",
    "EnumType",
    "ExceptionType",
    "FunctionSignature",
    "FunctionType",
    "GenericTypeDef",
    "IntType",
    "JsonType",
    "ListType",
    "OutputContractSpec",
    "ParamSpec",
    "RecordType",
    "TextType",
    "Type",
    "TypeEnvironment",
    "TypeVarType",
    "UnitType",
    "check",
    "contains_type_var",
    "free_type_vars",
    "substitute",
    "check_graph",
]
