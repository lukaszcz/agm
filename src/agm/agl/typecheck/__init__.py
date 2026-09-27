"""AgL type-checking pass.

Public API
----------
- :func:`check_program` — whole-program type pass:
  ``ResolvedProgram × HostCapabilities → CheckedProgram``.
- :class:`CheckedModule` — frozen dataclass with ``node_types``,
  ``contract_specs``, ``warnings``, and ``function_signatures``.
- :class:`CheckedProgram` — program output: per-module ``CheckedModule`` dict
  plus shared ``program_type_table``.
- :class:`OutputContractSpec` — per-call codec + target-type record.
- :class:`AglTypeError` — fatal type error (span-aware ``AglError`` subclass).
- :class:`UnappliedGenericTypeError` — an ``AglTypeError`` for a bare generic
  type reference missing its type arguments; carries the resolved
  ``GenericTypeDef`` and display name.
"""

from agm.agl.semantics.types import (
    ArrayType,
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
    UnappliedGenericTypeError,
    assert_checked_module_closed,
)
from agm.agl.typecheck.program import (
    CheckedProgram,
    assert_checked_program_closed,
    check_program,
)

__all__ = [
    "AglTypeError",
    "ArrayType",
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
    "UnappliedGenericTypeError",
    "UnitType",
    "assert_checked_program_closed",
    "assert_checked_module_closed",
    "contains_type_var",
    "free_type_vars",
    "substitute",
    "check_program",
]
