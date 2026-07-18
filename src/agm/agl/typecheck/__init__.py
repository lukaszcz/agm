"""AgL type-checking pass.

Public API
----------
- :func:`check_module` — per-module type pass:
  ``ModuleResolution × HostCapabilities → CheckedModule``.
- :func:`check_program` — whole-program type pass:
  ``ResolvedProgram × HostCapabilities → CheckedProgram``.
- :class:`CheckedModule` — frozen dataclass with ``node_types``,
  ``contract_specs``, ``warnings``, and ``function_signatures``.
- :class:`CheckedProgram` — program output: per-module ``CheckedModule`` dict
  plus shared ``program_type_table``.
- :class:`OutputContractSpec` — per-call codec + target-type record.
- :class:`AglTypeError` — fatal type error (span-aware ``AglError`` subclass).
"""

from agm.agl.semantics.types import (
    AgentType,
    BoolType,
    BottomType,
    DecimalType,
    DictType,
    EnumOwnerForm,
    EnumOwnerFormKind,
    EnumType,
    ExceptionType,
    FunctionType,
    IntType,
    JsonType,
    ListType,
    RecordType,
    TextType,
    Type,
    TypeTemplate,
    TypeTemplateMatch,
    TypeVarType,
    UnitType,
    contains_type_var,
    free_type_vars,
    substitute,
)
from agm.agl.typecheck.checker import check_module
from agm.agl.typecheck.env import (
    AglTypeError,
    CallSiteRecord,
    CheckedModule,
    ConstructorSignature,
    FunctionSignature,
    GenericTypeDef,
    OutputContractSpec,
    ParamSpec,
    PartialCallSpec,
    TypeEnvironment,
    assert_checked_module_closed,
)
from agm.agl.typecheck.program import (
    CheckedProgram,
    assert_checked_program_closed,
    check_program,
)

__all__ = [
    "AgentType",
    "AglTypeError",
    "BoolType",
    "BottomType",
    "CallSiteRecord",
    "CheckedModule",
    "CheckedProgram",
    "ConstructorSignature",
    "DecimalType",
    "DictType",
    "EnumOwnerForm",
    "EnumOwnerFormKind",
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
    "PartialCallSpec",
    "RecordType",
    "TextType",
    "Type",
    "TypeEnvironment",
    "TypeTemplate",
    "TypeTemplateMatch",
    "TypeVarType",
    "UnitType",
    "assert_checked_program_closed",
    "assert_checked_module_closed",
    "check_module",
    "contains_type_var",
    "free_type_vars",
    "substitute",
    "check_program",
]
