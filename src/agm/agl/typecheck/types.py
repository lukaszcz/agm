"""Semantic type model for the AgL type checker.

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
- ``DictType(value)`` — ``dict[text, V]`` (keys are always ``text`` in v1).
- ``RecordType(name, fields)`` — a ``record`` nominal type.
- ``EnumType(name, variants)`` — an ``enum`` nominal type.
- ``ExceptionType(name, fields)`` — a built-in exception type.
- ``UnitType`` — the ``unit`` type (AgL v2; single value ``()``).
- ``AgentType`` — the opaque ``agent`` type (AgL v2).
- ``FunctionType(params, result)`` — a first-class function type (AgL v2),
  positional only; named/optional arguments are erased from the value type.

``Type`` is the closed union of all semantic types.

Single coercion rule (design §5.8)
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

    ``fields`` maps field name → field type.
    """

    name: str
    fields: Mapping[str, Type]

    @property
    def kind(self) -> str:
        return "record"

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class EnumType:
    """An ``enum`` nominal type.

    ``variants`` maps variant name → mapping of field names → field types.
    """

    name: str
    variants: Mapping[str, Mapping[str, Type]]

    @property
    def kind(self) -> str:
        return "enum"

    def __repr__(self) -> str:
        return self.name


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
# AgL v2 value types (plan R6, R7, R9)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UnitType:
    """The ``unit`` type — has a single value written ``()`` (AgL v2, R9).

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
    """The opaque ``agent`` type (AgL v2, R6, D7).

    Agent values are first-class capability handles.  They are not
    JSON-shaped, not renderable, and have no equality in v1 (D7).
    """

    @property
    def kind(self) -> str:
        return "agent"

    def __repr__(self) -> str:
        return "agent"


@dataclass(frozen=True, slots=True)
class FunctionType:
    """A first-class function value type (AgL v2, R7).

    Positional only — named and optional argument information is erased from
    the value type per plan R7.  Structural equality is derived from the
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
)


# ---------------------------------------------------------------------------
# Assignability helper (single coercion: int → decimal)
# ---------------------------------------------------------------------------


def is_json_shaped(value_type: Type) -> bool:
    """Return ``True`` if ``value_type`` is JSON-shaped (design §5.8 rule 3).

    JSON-shaped types are the values that may inhabit a ``json`` slot:
    ``null``/``json``, ``bool``, ``int``, ``decimal``, ``text``, and
    ``list``/``dict`` whose element/value types are themselves JSON-shaped.
    Records, enums, and exceptions are **not** JSON-shaped — to embed one in a
    ``json`` value they must first be rendered to text (e.g. via a ``let`` binding).

    AgL v2: ``UnitType``, ``AgentType``, and ``FunctionType`` are also NOT
    JSON-shaped (plan D9 — function/agent values have no rendering).
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


def comparable_types(left: Type, right: Type) -> bool:
    """Return ``True`` if ``left`` and ``right`` may be compared (design §5.8 r4).

    Equality (``=``, ``!=``) and ordering comparisons require both operands to
    have the **same** type after the single ``int → decimal`` widening.  Unlike
    :func:`is_assignable`, ``json`` does **not** absorb JSON-shaped scalars here:
    ``json = json`` is allowed but ``json`` vs any non-``json`` type is a static
    error (rule 4 as written).  Records/enums/exceptions compare only with their
    own exact type.

    AgL v2: ``AgentType``, ``FunctionType``, and ``UnitType`` operands are
    NON-comparable — using ``=``/``!=``/``<`` on them is a static error (plan
    D7: agents have no equality in v1; plan D9: function values are opaque).
    """
    # Guard: agent, function, unit, and bottom values are never comparable.
    if isinstance(left, (AgentType, FunctionType, UnitType, BottomType)):
        return False
    if isinstance(right, (AgentType, FunctionType, UnitType, BottomType)):
        return False
    if left == right:
        return True
    # The only cross-type comparison is numeric int↔decimal (either direction).
    numeric = (IntType, DecimalType)
    return isinstance(left, numeric) and isinstance(right, numeric)


def is_assignable(value_type: Type, target_type: Type) -> bool:
    """Return ``True`` if ``value_type`` is assignable to ``target_type``.

    Implicit coercions (design §5.8):

    1. ``int → decimal`` widening is the only scalar coercion.
    2. ``json`` accepts any JSON-shaped value (rule 3): ``null``/``json``,
       ``bool``, ``int``, ``decimal``, ``text``, and ``list``/``dict`` of
       JSON-shaped types.  Records/enums/exceptions are rejected.

    All other assignments require exact structural equality.

    AgL v2: ``UnitType``, ``AgentType``, and ``FunctionType`` assignability is
    exact-only — no widening and no variance (plan R7, D7, D9).  The
    ``value_type == target_type`` check below handles them: ``UnitType`` and
    ``AgentType`` are parameter-free singletons so equality is trivial;
    ``FunctionType`` uses structural tuple equality on ``params`` + ``result``.

    AgL v2: ``BottomType`` (the type of ``raise``) is assignable to any target.
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
# Built-in exception types (design §8.1)
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

# Concrete built-in exceptions — exact §8.1 table.
# Every exception includes the base fields (message, trace_id) plus the
# additional fields listed in the design's "Additional fields" section.
#
# Changes from prior draft vs §8.1:
#   - AgentCallError: added agent: text and metadata: json (§8.1 §0 resolution 11)
#   - UndefinedVariableError: added with name: text (§8.1)
#   - ImmutableBindingError: added with name: text, operation: text (§8.1)
#   - ValidationError: REMOVED — §8.1 does not list it as a catchable exception;
#     agm.agl.runtime.request.ValidationError is a Python-level record shape
#     embedded in AgentParseError.validation_errors (design §7.5), not an AgL type.
BUILTIN_EXCEPTIONS: dict[str, ExceptionType] = {
    "Exception": EXCEPTION_BASE,
    # §8.1 AgentCallError: agent/cause/metadata (§0 resolution 11: cause is
    # enumerated "spawn_failure"|"nonzero_exit"|"timeout"; metadata carries
    # exit code, stderr tail, elapsed — all stored in the json field).
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
    # §8.1: statically prevented in v1 (scope/typecheck reject assignment to immutable
    # bindings and undeclared names), but still listed as catchable runtime
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
    # AgL v2: RecursionError raised when the call-depth limit is exceeded (plan D8).
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
}

# Names of built-in exception types (cannot be redeclared as records/enums/aliases).
BUILTIN_EXCEPTION_NAMES: frozenset[str] = frozenset(BUILTIN_EXCEPTIONS)


# ---------------------------------------------------------------------------
# Built-in prelude types (AgL v2; plan D10, D11)
#
# These are registered into every fresh TypeEnvironment alongside the built-in
# exceptions and are non-shadowable.  Their runtime semantics are implemented
# in the eval/runtime stages (S4/S5).
# ---------------------------------------------------------------------------

# ``ExecResult`` — the structured result of an ``exec`` call when the target
# type is ``ExecResult`` (plan D10).  Mirrors the field shape of ``ExecError``.
_EXEC_RESULT_TYPE = RecordType(
    name="ExecResult",
    fields={
        "stdout": TextType(),
        "exit_code": IntType(),
        "stderr": TextType(),
        "timed_out": BoolType(),
    },
)

# ``ParsePolicy`` — controls ``ask``/``exec`` error handling (plan D11).
# ``Abort`` — abort on parse error (no fields).
# ``Retry(n: int)`` — retry up to ``n`` times.
_PARSE_POLICY_TYPE = EnumType(
    name="ParsePolicy",
    variants={
        "Abort": {},
        "Retry": {"n": IntType()},
    },
)

# ``OutputContract`` — the materialized output contract of an agent/exec call
# site, surfaced as an AgL value by ``ask-request``.  ``strict_json`` and
# ``json_schema`` use ``json`` because they are nullable (``null`` when the
# codec is not JSON-based / when no schema applies).
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
)

# ``OutputContractOption`` — an explicit optional output contract.  ``unit``
# agent calls use ``None`` because their response is intentionally discarded;
# all parsed-output calls use ``Some``.
_OUTPUT_CONTRACT_OPTION_TYPE = EnumType(
    name="OutputContractOption",
    variants={
        "None": {},
        "Some": {"value": _OUTPUT_CONTRACT_TYPE},
    },
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
        "attempt": IntType(),
        "output_contract": _OUTPUT_CONTRACT_OPTION_TYPE,
    },
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


# ---------------------------------------------------------------------------
# Cast classification (M3b)
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
    # Non-data types never participate
    _non_data = (UnitType, AgentType, FunctionType, BottomType)
    if isinstance(source, _non_data) or isinstance(target, _non_data):
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
        # renderable types: json, bool, int, decimal, list, dict, record, enum
        # ExceptionType NOT renderable via casts; non-data types filtered above.
        if isinstance(source, (JsonType, BoolType, IntType, DecimalType, ListType, DictType,
                               RecordType, EnumType)):
            return CastKind.TOTAL_RENDER
        return CastKind.STATIC_ERROR

    if isinstance(target, JsonType):
        # All json-shaped sources are assignable to json (handled above), so
        # anything reaching here is NOT json-shaped → static error.
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
