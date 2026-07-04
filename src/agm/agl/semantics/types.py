"""Shared resolved semantic type model for AgL.

These are *resolved* nominal types distinct from the syntactic ``TypeExpr``
hierarchy in ``agm.agl.syntax.types``.  Aliases are resolved transparently
to their target — ``TypeAlias`` nodes never appear here.

Type hierarchy
--------------
- ``TextType`` — the ``text`` primitive.
- ``JsonType`` — the ``json`` primitive (any JSON-shaped value).
- ``BoolType`` — the ``bool`` primitive.
- ``IntType`` — the ``int`` primitive (arbitrary-precision integer).
- ``DecimalType`` — the ``decimal`` primitive (exact fixed-point).
- ``ListType(elem)`` — ``list[T]``.
- ``DictType(value)`` — ``dict[text, V]`` (keys are always ``text`` in AgL).
- ``RecordType(name, fields)`` — a ``record`` nominal type.
- ``EnumType(name, variants)`` — an ``enum`` nominal type.
- ``ExceptionType(name, fields)`` — a built-in exception type.
- ``UnitType`` — the ``unit`` type (AgL; single value ``()``).
- ``AgentType`` — the opaque ``agent`` type (AgL).
- ``FunctionType(params, result)`` — a first-class function type (AgL),
  positional only; named/optional arguments are erased from the value type.
- ``TypeVarType(name)`` — a rigid type variable bound by an enclosing generic
  declaration.

``Type`` is the closed union of all semantic types.

Single coercion rule
-------------------------------------
``int → decimal`` widening is the **only** implicit type coercion.  Use
:func:`is_assignable` to check assignability with this single coercion
applied.

Type-kind strings (for codec capability lookup)
------------------------------------------------
Each ``Type`` exposes a ``kind`` property — a lower-cased string identifying
the type's kind in the ``HostCapabilities.codec_kinds`` maps.  E.g.
``TextType().kind == "text"``, ``RecordType(...).kind == "record"``.
"""

from __future__ import annotations

import enum as _enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import assert_never

from agm.agl.modules.ids import ENTRY_ID, PRELUDE_ID, STD_CORE_ID, ModuleId

# ---------------------------------------------------------------------------
# Primitive types (singletons-by-construction; frozen dataclasses)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextType:
    """The ``text`` built-in type."""

    @property
    def kind(self) -> str:
        return "text"

    def __repr__(self) -> str:
        return "text"


@dataclass(frozen=True, slots=True)
class JsonType:
    """The ``json`` built-in type (any JSON value)."""

    @property
    def kind(self) -> str:
        return "json"

    def __repr__(self) -> str:
        return "json"


@dataclass(frozen=True, slots=True)
class BoolType:
    """The ``bool`` built-in type."""

    @property
    def kind(self) -> str:
        return "bool"

    def __repr__(self) -> str:
        return "bool"


@dataclass(frozen=True, slots=True)
class IntType:
    """The ``int`` built-in type."""

    @property
    def kind(self) -> str:
        return "int"

    def __repr__(self) -> str:
        return "int"


@dataclass(frozen=True, slots=True)
class DecimalType:
    """The ``decimal`` built-in type (exact fixed-point)."""

    @property
    def kind(self) -> str:
        return "decimal"

    def __repr__(self) -> str:
        return "decimal"


# ---------------------------------------------------------------------------
# Parameterised container types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ListType:
    """``list[T]`` — a homogeneous list."""

    elem: Type

    @property
    def kind(self) -> str:
        return "list"

    def __repr__(self) -> str:
        return f"list[{self.elem!r}]"


@dataclass(frozen=True, slots=True)
class DictType:
    """``dict[text, V]`` — string-keyed dict."""

    value: Type

    @property
    def kind(self) -> str:
        return "dict"

    def __repr__(self) -> str:
        return f"dict[text, {self.value!r}]"


# ---------------------------------------------------------------------------
# Nominal types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecordType:
    """A ``record`` nominal type.

    ``fields`` maps field name → field type.  Fields are excluded from
    equality and hashing so that two instantiations of the same generic type
    compare equal iff their ``name``, ``type_args``, and ``module_id`` match.
    ``type_args`` holds the resolved type arguments for a generic instantiation
    (empty tuple for non-generic records).
    ``module_id`` is the owning module (defaults to ``ENTRY_ID`` so existing
    single-program paths and built-in/prelude types are unaffected).
    Identity is ``(module_id, name, type_args)`` — ``fields`` is excluded from
    equality and hashing so that a shell (empty fields) and its built form
    compare equal.
    """

    name: str
    fields: Mapping[str, Type] = field(compare=False)
    type_args: tuple[Type, ...] = ()
    module_id: ModuleId = field(default_factory=lambda: ENTRY_ID)

    @property
    def kind(self) -> str:
        return "record"

    def __repr__(self) -> str:
        prefix = "" if self.module_id.is_entry else f"{self.module_id.dotted()}::"
        if self.type_args:
            args_str = ", ".join(repr(a) for a in self.type_args)
            return f"{prefix}{self.name}[{args_str}]"
        return f"{prefix}{self.name}"


@dataclass(frozen=True, slots=True)
class EnumType:
    """An ``enum`` nominal type.

    ``variants`` maps variant name → mapping of field names → field types.
    Variants are excluded from equality and hashing so that two instantiations
    of the same generic type compare equal iff their ``name``, ``type_args``,
    and ``module_id`` match.  ``type_args`` holds the resolved type arguments
    for a generic instantiation (empty tuple for non-generic enums).
    ``module_id`` is the owning module (defaults to ``ENTRY_ID``).
    Identity is ``(module_id, name, type_args)`` — ``variants`` is excluded
    from equality and hashing so that a shell and its built form compare equal.
    """

    name: str
    variants: Mapping[str, Mapping[str, Type]] = field(compare=False)
    type_args: tuple[Type, ...] = ()
    module_id: ModuleId = field(default_factory=lambda: ENTRY_ID)

    @property
    def kind(self) -> str:
        return "enum"

    def __repr__(self) -> str:
        prefix = "" if self.module_id.is_entry else f"{self.module_id.dotted()}::"
        if self.type_args:
            args_str = ", ".join(repr(a) for a in self.type_args)
            return f"{prefix}{self.name}[{args_str}]"
        return f"{prefix}{self.name}"


@dataclass(frozen=True, slots=True)
class ExceptionType:
    """A built-in exception type.

    ``fields`` maps field name → field type.
    The abstract ``Exception`` base is represented as an ``ExceptionType``
    with name ``"Exception"``, only ``message``/``trace_id`` fields, and
    ``abstract=True`` (it is catchable as the hierarchy root but not
    constructible).
    """

    name: str
    fields: Mapping[str, Type] = field(default_factory=dict)
    abstract: bool = False

    @property
    def kind(self) -> str:
        return "exception"

    def __repr__(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# AgL value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UnitType:
    """The ``unit`` type — has a single value written ``()``.

    Side-effecting expressions (``print``, ``:=``, ``if`` with no ``else``,
    loops) yield ``unit``.
    """

    @property
    def kind(self) -> str:
        return "unit"

    def __repr__(self) -> str:
        return "unit"


@dataclass(frozen=True, slots=True)
class AgentType:
    """The opaque ``agent`` type.

    Agent values are first-class capability handles.  They are not
    JSON-shaped, not renderable, and have no equality operator.
    """

    @property
    def kind(self) -> str:
        return "agent"

    def __repr__(self) -> str:
        return "agent"


@dataclass(frozen=True, slots=True)
class FunctionType:
    """A first-class function value type.

    Positional only — named and optional argument information is erased from
    the value type. Structural equality is derived from the
    frozen ``params`` tuple and ``result`` field.

    ``params``  — positional parameter types, in declaration order.
    ``result``  — the function's return type.
    """

    params: tuple[Type, ...]
    result: Type

    @property
    def kind(self) -> str:
        return "function"

    def __repr__(self) -> str:
        param_str = ", ".join(repr(p) for p in self.params)
        return f"({param_str}) -> {self.result!r}"


@dataclass(frozen=True, slots=True)
class BottomType:
    """Internal bottom type for ``raise`` expressions.

    Assignable to ANY target; nothing is assignable to it except itself.
    Not JSON-shaped, not comparable, not user-writable (no TypeExpr yields it).
    """

    @property
    def kind(self) -> str:
        return "bottom"

    def __repr__(self) -> str:
        return "bottom"


@dataclass(frozen=True, slots=True)
class TypeVarType:
    """A rigid type variable bound by an enclosing generic declaration.

    ``TypeVarType`` is used during type resolution and type checking of
    generic definitions.  It is never user-visible at the value level
    — generic instantiation substitutes all type variables before a value
    is constructed.

    Capability notes:
    - Not JSON-shaped (``is_json_shaped`` returns ``False``).
    - Not comparable (``comparable_types`` returns ``False`` for either side).
    - Assignable only to an identical ``TypeVarType`` (same name); ``json``
      does NOT absorb it; ``BottomType`` is still assignable to it.
    """

    name: str

    @property
    def kind(self) -> str:
        return "typevar"

    def __repr__(self) -> str:
        return self.name


# Closed union of all semantic types.
Type = (
    TextType
    | JsonType
    | BoolType
    | IntType
    | DecimalType
    | ListType
    | DictType
    | RecordType
    | EnumType
    | ExceptionType
    | UnitType
    | AgentType
    | FunctionType
    | BottomType
    | TypeVarType
)


# ---------------------------------------------------------------------------
# Assignability helper (single coercion: int → decimal)
# ---------------------------------------------------------------------------


def is_json_shaped(value_type: Type) -> bool:
    """Return ``True`` if ``value_type`` is JSON-shaped.

    JSON-shaped types are the values that may inhabit a ``json`` slot:
    ``null``/``json``, ``bool``, ``int``, ``decimal``, ``text``, and
    ``list``/``dict`` whose element/value types are themselves JSON-shaped.
    Records, enums, and exceptions are **not** JSON-shaped — to embed one in a
    ``json`` value they must first be rendered to text (e.g. via a ``let`` binding).

    AgL: ``UnitType``, ``AgentType``, and ``FunctionType`` are also NOT
    JSON-shaped; function and agent values render only as opaque handles.
    """
    if isinstance(value_type, (TextType, JsonType, BoolType, IntType, DecimalType)):
        return True
    if isinstance(value_type, ListType):
        return is_json_shaped(value_type.elem)
    if isinstance(value_type, DictType):
        return is_json_shaped(value_type.value)
    # RecordType, EnumType, ExceptionType, UnitType, AgentType, FunctionType
    # are not JSON-shaped.
    return False


def _has_no_value_equality(t: Type) -> bool:
    """True if ``t`` is, or transitively contains, a type with no value equality.

    Function, agent, and unit values are opaque / identity-only and AgL gives
    them no ``=``/``!=`` operator; ``unit`` has a single value but no equality
    operator.  A list, dict, record, enum, or exception that transitively holds
    such a type is therefore itself not comparable.  Recursive types are
    rejected, so this recursion terminates.
    """
    match t:
        case FunctionType() | AgentType() | UnitType():
            return True
        case ListType():
            return _has_no_value_equality(t.elem)
        case DictType():
            return _has_no_value_equality(t.value)
        case RecordType():
            return any(_has_no_value_equality(ft) for ft in t.fields.values())
        case EnumType():
            return any(
                _has_no_value_equality(ft)
                for variant in t.variants.values()
                for ft in variant.values()
            )
        case ExceptionType():
            return any(_has_no_value_equality(ft) for ft in t.fields.values())
        case (TextType() | JsonType() | BoolType() | IntType() | DecimalType()
              | BottomType() | TypeVarType()):
            return False
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def comparable_types(left: Type, right: Type) -> bool:
    """Return ``True`` if ``left`` and ``right`` may be compared.

    Equality (``=``, ``!=``) and ordering comparisons require both operands to
    have the **same** type after the single ``int → decimal`` widening.  Unlike
    :func:`is_assignable`, ``json`` does **not** absorb JSON-shaped scalars here:
    ``json = json`` is allowed but ``json`` vs any non-``json`` type is a static
    error (rule 4 as written).  Records/enums/exceptions compare only with their
    own exact type.

    AgL: ``AgentType``, ``FunctionType``, and ``UnitType`` operands are
    NON-comparable — using ``=``/``!=``/``<`` on them is a static error. Agents
    have no equality in AgL; function values are opaque.
    This rule is **transitive**: a ``list``, ``dict``, ``record``, ``enum``, or
    ``exception`` that (at any depth) contains a function, agent, or ``unit``
    value likewise has no equality and cannot be compared with ``=``/``!=``.
    """
    # Function/agent/unit values — and any container/record/enum that transitively
    # holds one — have no value equality.
    if _has_no_value_equality(left) or _has_no_value_equality(right):
        return False
    # Bare type variables and the bottom type are never comparable here (the
    # checker additionally rejects bare type variables at the comparison site).
    if isinstance(left, (BottomType, TypeVarType)) or isinstance(right, (BottomType, TypeVarType)):
        return False
    if left == right:
        return True
    # The only cross-type comparison is numeric int↔decimal (either direction).
    numeric = (IntType, DecimalType)
    return isinstance(left, numeric) and isinstance(right, numeric)


def is_assignable(value_type: Type, target_type: Type) -> bool:
    """Return ``True`` if ``value_type`` is assignable to ``target_type``.

    Implicit coercions:

    1. ``int → decimal`` widening is the only scalar coercion.
    2. ``json`` accepts any JSON-shaped value (rule 3): ``null``/``json``,
       ``bool``, ``int``, ``decimal``, ``text``, and ``list``/``dict`` of
       JSON-shaped types.  Records/enums/exceptions are rejected.

    All other assignments require exact structural equality.

    AgL: ``UnitType``, ``AgentType``, and ``FunctionType`` assignability is
    exact-only — no widening and no variance.  The
    ``value_type == target_type`` check below handles them: ``UnitType`` and
    ``AgentType`` are parameter-free singletons so equality is trivial;
    ``FunctionType`` uses structural tuple equality on ``params`` + ``result``.

    AgL: ``BottomType`` (the type of ``raise``) is assignable to any target.
    """
    # Bottom type is assignable to any target (raise can appear anywhere).
    if isinstance(value_type, BottomType):
        return True
    if value_type == target_type:
        return True
    # Single scalar coercion: int can widen to decimal.
    if isinstance(value_type, IntType) and isinstance(target_type, DecimalType):
        return True
    # json accepts any JSON-shaped value (records/enums/exceptions excluded;
    # UnitType/AgentType/FunctionType also excluded via is_json_shaped).
    if isinstance(target_type, JsonType):
        return is_json_shaped(value_type)
    return False


# ---------------------------------------------------------------------------
# Generic type helpers
# ---------------------------------------------------------------------------


def free_type_vars(t: Type) -> frozenset[str]:
    """Recursively collect free type-variable names in *t*."""
    if isinstance(t, TypeVarType):
        return frozenset({t.name})
    if isinstance(t, ListType):
        return free_type_vars(t.elem)
    if isinstance(t, DictType):
        return free_type_vars(t.value)
    if isinstance(t, FunctionType):
        result: frozenset[str] = frozenset()
        for p in t.params:
            result = result | free_type_vars(p)
        return result | free_type_vars(t.result)
    if isinstance(t, RecordType):
        result = frozenset()
        for ta in t.type_args:
            result = result | free_type_vars(ta)
        for ft in t.fields.values():
            result = result | free_type_vars(ft)
        return result
    if isinstance(t, EnumType):
        result = frozenset()
        for ta in t.type_args:
            result = result | free_type_vars(ta)
        for vfields in t.variants.values():
            for ft in vfields.values():
                result = result | free_type_vars(ft)
        return result
    # Primitives, ExceptionType, UnitType, AgentType, BottomType: no type vars.
    return frozenset()


def substitute(t: Type, subst: Mapping[str, Type]) -> Type:
    """Capture-free substitution: replace ``TypeVarType(n)`` with ``subst[n]``."""
    if isinstance(t, TypeVarType):
        return subst.get(t.name, t)
    if isinstance(t, ListType):
        return ListType(elem=substitute(t.elem, subst))
    if isinstance(t, DictType):
        return DictType(value=substitute(t.value, subst))
    if isinstance(t, FunctionType):
        return FunctionType(
            params=tuple(substitute(p, subst) for p in t.params),
            result=substitute(t.result, subst),
        )
    if isinstance(t, RecordType):
        new_type_args = tuple(substitute(ta, subst) for ta in t.type_args)
        new_fields = {k: substitute(v, subst) for k, v in t.fields.items()}
        return RecordType(
            name=t.name, fields=new_fields, type_args=new_type_args, module_id=t.module_id
        )
    if isinstance(t, EnumType):
        new_type_args = tuple(substitute(ta, subst) for ta in t.type_args)
        new_variants = {
            vname: {k: substitute(v, subst) for k, v in vfields.items()}
            for vname, vfields in t.variants.items()
        }
        return EnumType(
            name=t.name, variants=new_variants, type_args=new_type_args, module_id=t.module_id
        )
    # Primitives, ExceptionType, UnitType, AgentType, BottomType: unchanged.
    return t


def contains_type_var(t: Type) -> bool:
    """Return ``True`` if *t* contains any free type variable.

    Short-circuits on the first ``TypeVarType`` found instead of collecting the
    full free-variable set (this is called per-argument in the inference loops).
    """
    if isinstance(t, TypeVarType):
        return True
    if isinstance(t, ListType):
        return contains_type_var(t.elem)
    if isinstance(t, DictType):
        return contains_type_var(t.value)
    if isinstance(t, FunctionType):
        return any(contains_type_var(p) for p in t.params) or contains_type_var(t.result)
    if isinstance(t, RecordType):
        return any(contains_type_var(ta) for ta in t.type_args) or any(
            contains_type_var(ft) for ft in t.fields.values()
        )
    if isinstance(t, EnumType):
        return any(contains_type_var(ta) for ta in t.type_args) or any(
            contains_type_var(ft) for vfields in t.variants.values() for ft in vfields.values()
        )
    return False


# ---------------------------------------------------------------------------
# Built-in exception types
# ---------------------------------------------------------------------------

# Abstract base: only message + trace_id fields.
EXCEPTION_BASE = ExceptionType(
    name="Exception",
    fields={
        "message": TextType(),
        "trace_id": TextType(),
    },
    abstract=True,
)

# Concrete built-in exceptions — exact .
# Every exception includes the base fields (message, trace_id) plus the
# additional fields listed in the design's "Additional fields" section.
#
# Changes from prior draft vs :
#   - AgentCallError: added agent: text and metadata: json
#   - UndefinedVariableError: added with name: text
#   - ImmutableBindingError: added with name: text, operation: text
#   - ValidationError: REMOVED — ;
#     agm.agl.runtime.request.ValidationError is a Python-level record shape
#     embedded in AgentParseError.validation_errors, not an AgL type.
BUILTIN_EXCEPTIONS: dict[str, ExceptionType] = {
    "Exception": EXCEPTION_BASE,
    # : agent/cause/metadata.
    "AgentCallError": ExceptionType(
        name="AgentCallError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
            "agent": TextType(),
            "cause": TextType(),
            "metadata": JsonType(),
        },
    ),
    "AgentParseError": ExceptionType(
        name="AgentParseError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
            "agent": TextType(),
            "target_type": TextType(),
            "expected_schema": JsonType(),
            "raw": TextType(),
            "normalized_raw": TextType(),
            "validation_errors": JsonType(),
            "attempts": IntType(),
            "metadata": JsonType(),
        },
    ),
    "ExecError": ExceptionType(
        name="ExecError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
            "command": TextType(),
            "exit_code": IntType(),
            "stdout": TextType(),
            "stderr": TextType(),
            "timed_out": BoolType(),
        },
    ),
    # Raised for every runtime failure crossing an extern (Python FFI) call:
    # the Python callable raising, a return-contract violation (including a
    # seal violation), or an argument-conversion failure.  ``python_type`` is
    # the raising Python exception's class name, or empty for a contract
    # violation (no Python exception was involved).
    "ExternError": ExceptionType(
        name="ExternError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
            "function": TextType(),
            "python_type": TextType(),
        },
    ),
    "MaxIterationsExceeded": ExceptionType(
        name="MaxIterationsExceeded",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
            "limit": IntType(),
            "condition": TextType(),
            "last_condition_value": BoolType(),
            "metadata": JsonType(),
        },
    ),
    "MatchError": ExceptionType(
        name="MatchError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
            "scrutinee_type": TextType(),
            "scrutinee": JsonType(),
        },
    ),
    "IndexError": ExceptionType(
        name="IndexError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
            "index": IntType(),
            "length": IntType(),
        },
    ),
    "KeyError": ExceptionType(
        name="KeyError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
            "key": TextType(),
        },
    ),
    "TypeError": ExceptionType(
        name="TypeError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
        },
    ),
    "ArithmeticError": ExceptionType(
        name="ArithmeticError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
            "operation": TextType(),
        },
    ),
    # Statically prevented by scope/typecheck (assignment to immutable bindings
    # and undeclared names), but still listed as catchable runtime
    # exceptions for any runtime paths that bypass the static passes.
    "UndefinedVariableError": ExceptionType(
        name="UndefinedVariableError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
            "name": TextType(),
        },
    ),
    "ImmutableBindingError": ExceptionType(
        name="ImmutableBindingError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
            "name": TextType(),
            "operation": TextType(),
        },
    ),
    "Abort": ExceptionType(
        name="Abort",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
        },
    ),
    # AgL: RecursionError raised when the call-depth limit is exceeded.
    "RecursionError": ExceptionType(
        name="RecursionError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
            "limit": IntType(),
        },
    ),
    "CastError": ExceptionType(
        name="CastError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
            "source_type": TextType(),
            "target_type": TextType(),
            "raw": TextType(),
        },
    ),
    "JsonParseError": ExceptionType(
        name="JsonParseError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
            "raw": TextType(),
        },
    ),
    "RangeError": ExceptionType(
        name="RangeError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
        },
    ),
}

# Names of built-in exception types (cannot be redeclared as records/enums/aliases).
BUILTIN_EXCEPTION_NAMES: frozenset[str] = frozenset(BUILTIN_EXCEPTIONS)


# ---------------------------------------------------------------------------
# Built-in prelude types
#
# These are registered into every fresh TypeEnvironment alongside the built-in
# exceptions and are non-shadowable.  Their runtime semantics are implemented
# in the eval/runtime stages.
# ---------------------------------------------------------------------------

# ``ExecResult`` — the structured result of an ``exec`` call when the target
# type is ``ExecResult``.  Mirrors the field shape of ``ExecError``.
_EXEC_RESULT_TYPE = RecordType(
    name="ExecResult",
    fields={
        "stdout": TextType(),
        "exit_code": IntType(),
        "stderr": TextType(),
        "timed_out": BoolType(),
    },
    module_id=PRELUDE_ID,
)

# ``ParsePolicy`` — controls ``ask``/``exec`` error handling.
# ``Abort`` — abort on parse error (no fields).
# ``Retry(n: int)`` — retry up to ``n`` times.
_PARSE_POLICY_TYPE = EnumType(
    name="ParsePolicy",
    variants={
        "Abort": {},
        "Retry": {"n": IntType()},
    },
    module_id=PRELUDE_ID,
)

_OPTION_TEXT_TYPE = EnumType(
    name="Option",
    variants={
        "None": {},
        "Some": {"value": TextType()},
    },
    type_args=(TextType(),),
    module_id=STD_CORE_ID,
)

# Public alias for the ``Option[text]`` type — the single source of truth
# shared with engine_keys and any other module that needs this type.
OPTION_TEXT_TYPE: EnumType = _OPTION_TEXT_TYPE

_OPTION_JSON_TYPE = EnumType(
    name="Option",
    variants={
        "None": {},
        "Some": {"value": JsonType()},
    },
    type_args=(JsonType(),),
    module_id=STD_CORE_ID,
)

_OUTPUT_CONTRACT_TYPE = RecordType(
    name="OutputContract",
    fields={
        "target_type": TextType(),
        "codec_name": TextType(),
        "strict_json": JsonType(),
        "format_instructions": TextType(),
        "json_schema": JsonType(),
        "structured_exec": BoolType(),
    },
    module_id=PRELUDE_ID,
)

_OUTPUT_CONTRACT_OPTION_TYPE = EnumType(
    name="OutputContractOption",
    variants={
        "None": {},
        "Some": {"value": _OUTPUT_CONTRACT_TYPE},
    },
    module_id=PRELUDE_ID,
)

# ``AgentRequest`` — the request that the corresponding ``ask`` call would
# dispatch to its agent, surfaced as an AgL value by ``ask-request``.  This is
# the first-attempt request: ``attempt`` is always ``0`` and there is no
# retry context (no ``previous_invalid_output`` / ``validation_errors``),
# because ``ask-request`` never invokes the agent.
_AGENT_REQUEST_TYPE = RecordType(
    name="AgentRequest",
    fields={
        "agent": TextType(),
        "prompt": TextType(),
        "target_type": _OPTION_TEXT_TYPE,
        "format_instructions": _OPTION_TEXT_TYPE,
        "json_schema": _OPTION_JSON_TYPE,
        "attempt": IntType(),
        "previous_error": _OPTION_TEXT_TYPE,
        "metadata": JsonType(),
    },
    module_id=PRELUDE_ID,
)

BUILTIN_PRELUDE_TYPES: dict[str, Type] = {
    "ExecResult": _EXEC_RESULT_TYPE,
    "ParsePolicy": _PARSE_POLICY_TYPE,
    "OutputContract": _OUTPUT_CONTRACT_TYPE,
    "OutputContractOption": _OUTPUT_CONTRACT_OPTION_TYPE,
    "AgentRequest": _AGENT_REQUEST_TYPE,
}

# Names of built-in prelude types (non-shadowable, like built-in exceptions).
BUILTIN_PRELUDE_TYPE_NAMES: frozenset[str] = frozenset(BUILTIN_PRELUDE_TYPES)

# Legacy built-in types kept for compatibility with already-compiled tests and
# internal APIs.  They remain available as nominal types, but their constructors
# are not exported into source scope because std.core replaces this surface.
COMPATIBILITY_PRELUDE_TYPE_NAMES: frozenset[str] = frozenset(
    {"OutputContract", "OutputContractOption"}
)


# ---------------------------------------------------------------------------
# Cast classification
# ---------------------------------------------------------------------------


class CastKind(_enum.Enum):
    """Classification of a cast operation from cast_classification()."""

    TOTAL_NOOP = "TOTAL_NOOP"      # source already assignable to target (no-op/widen)
    TOTAL_RENDER = "TOTAL_RENDER"  # render data value to text
    TOTAL_JSON = "TOTAL_JSON"      # canonicalize JSON-shaped value to json
    FALLIBLE = "FALLIBLE"          # runtime-fallible conversion
    STATIC_ERROR = "STATIC_ERROR"  # statically impossible — raise AglTypeError


@dataclass(frozen=True, slots=True)
class CastSpec:
    """Resolved runtime cast descriptor stored in CheckedProgram.cast_specs."""

    target_type: Type
    kind: CastKind


def cast_classification(source: Type, target: Type) -> CastKind:
    """Classify a cast from source to target type.

    Returns the CastKind for the (source, target) pair.
    """
    # Bottom is a valid source because a raise expression never reaches the
    # conversion. Other non-data sources and all non-data targets are invalid.
    if isinstance(source, (UnitType, AgentType, FunctionType)) or isinstance(
        target, (UnitType, AgentType, FunctionType, BottomType)
    ):
        return CastKind.STATIC_ERROR
    # ExceptionType as target is not in the matrix
    if isinstance(target, ExceptionType):
        return CastKind.STATIC_ERROR

    # Handle is_assignable cases first (no-op / widen / json-absorb).
    # Note: is_assignable(X, TextType) is true only when X is TextType itself
    # (no implicit widening to text), so the only assignable-to-text case is noop.
    # is_assignable(X, JsonType) is true for all json-shaped types.
    if is_assignable(source, target):
        if isinstance(target, JsonType):
            # json → json: noop; all other json-shaped sources → canonicalize
            if isinstance(source, JsonType):
                return CastKind.TOTAL_NOOP
            return CastKind.TOTAL_JSON
        # All other assignable cases are no-ops (including int→decimal widen,
        # same-type identity, etc.)
        return CastKind.TOTAL_NOOP

    # Now source is NOT assignable to target.
    _text_or_json = (TextType, JsonType)

    if isinstance(target, TextType):
        # Every data value renders to text. Non-data sources (unit/agent/function)
        # are filtered at the top, and json-shaped/exact-type sources are handled by
        # the is_assignable block above, so any source reaching here is a renderable
        # data type (json/bool/int/decimal/list/dict/record/enum/exception).
        return CastKind.TOTAL_RENDER

    if isinstance(target, JsonType):
        # All json-shaped sources are assignable to json (handled above), so
        # anything reaching here is NOT json-shaped.
        # Nominal types (record/enum/exception) support an explicit structural JSON cast.
        if isinstance(source, (RecordType, EnumType, ExceptionType)):
            return CastKind.TOTAL_JSON
        return CastKind.STATIC_ERROR

    if isinstance(target, (BoolType, IntType, DecimalType)):
        # decimal → int is a narrowing cast (fallible); text/json → numeric is fallible.
        if isinstance(source, _text_or_json) or (
            isinstance(target, IntType) and isinstance(source, DecimalType)
        ):
            return CastKind.FALLIBLE
        return CastKind.STATIC_ERROR

    if isinstance(target, (ListType, DictType, RecordType, EnumType)):
        if isinstance(source, _text_or_json):
            return CastKind.FALLIBLE
        return CastKind.STATIC_ERROR

    # All target types are covered above; this is a safety fallback.
    return CastKind.STATIC_ERROR  # pragma: no cover
