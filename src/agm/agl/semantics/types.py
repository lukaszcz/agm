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
- ``RecordType(name, type_args, module_id)`` — a ``record`` nominal type
  handle; field shapes live in the shared ``TypeTable``
  (``semantics.type_table``), keyed by ``(module_id, name)``.
- ``EnumType(name, type_args, module_id)`` — an ``enum`` nominal type handle;
  variant shapes live in the shared ``TypeTable``.
- ``ExceptionType(name, module_id)`` — an exception nominal type handle
  (never generic); field shapes and hierarchy (``abstract``, ``base``) live
  in the shared ``TypeTable``.
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
    """A ``record`` nominal type handle.

    A ``RecordType`` carries no field data — it is a lightweight handle whose
    identity is ``(module_id, name, type_args)``.  Field types are looked up
    by handle in the shared ``TypeTable`` (``semantics.type_table.TypeTable
    .record_fields``).  ``type_args`` holds the resolved type arguments for a
    generic instantiation (empty tuple for non-generic records).
    ``module_id`` is the owning module (defaults to ``ENTRY_ID`` so existing
    single-program paths and built-in/prelude types are unaffected).
    """

    name: str
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
    """An ``enum`` nominal type handle.

    An ``EnumType`` carries no variant data — it is a lightweight handle
    whose identity is ``(module_id, name, type_args)``.  Variant shapes are
    looked up by handle in the shared ``TypeTable``
    (``semantics.type_table.TypeTable.enum_variants``).  ``type_args`` holds
    the resolved type arguments for a generic instantiation (empty tuple for
    non-generic enums).  ``module_id`` is the owning module (defaults to
    ``ENTRY_ID``).
    """

    name: str
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
    """An exception nominal type handle.

    An ``ExceptionType`` carries no field data — it is a lightweight handle
    whose identity is ``(module_id, name)``; exceptions are never generic, so
    there is no ``type_args`` component (unlike ``RecordType``/``EnumType``).
    Field shapes and hierarchy metadata (``abstract``, ``base``) are looked
    up by handle in the shared ``TypeTable``
    (``semantics.type_table.TypeTable.exception_fields``/``exception_def``).
    ``module_id`` is the owning module (defaults to ``ENTRY_ID``, like
    ``RecordType``/``EnumType``); built-in exceptions carry ``PRELUDE_ID``.

    The abstract ``Exception`` root is the ``TypeDef`` registered under name
    ``"Exception"`` with ``abstract=True`` and only ``message``/``trace_id``
    fields — it is catchable as the hierarchy root but not constructible.
    """

    name: str
    module_id: ModuleId = field(default_factory=lambda: ENTRY_ID)

    @property
    def kind(self) -> str:
        return "exception"

    def __repr__(self) -> str:
        # Built-in/prelude exceptions always render as the bare name (matching
        # today's user-visible diagnostics); a module-owned user exception
        # matches the record/enum qualification style.
        if self.module_id.is_entry or self.module_id == PRELUDE_ID:
            return self.name
        return f"{self.module_id.dotted()}::{self.name}"


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
        return _format_function_type(self)


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
    - Not comparable (``semantics.type_table.comparable_types`` returns
      ``False`` for either side).
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


def _format_type(typ: Type, *, parenthesize_function: bool = False) -> str:
    if isinstance(typ, FunctionType):
        rendered = _format_function_type(typ)
        if parenthesize_function:
            return f"({rendered})"
        return rendered
    return repr(typ)


def _format_function_type(typ: FunctionType) -> str:
    if not typ.params:
        params = "()"
    elif len(typ.params) == 1:
        params = _format_type(typ.params[0], parenthesize_function=True)
    else:
        params = f"({', '.join(_format_type(param) for param in typ.params)})"
    return f"{params} -> {_format_type(typ.result)}"


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
    if isinstance(t, (RecordType, EnumType)):
        result = frozenset()
        for ta in t.type_args:
            result = result | free_type_vars(ta)
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
        return RecordType(name=t.name, type_args=new_type_args, module_id=t.module_id)
    if isinstance(t, EnumType):
        new_type_args = tuple(substitute(ta, subst) for ta in t.type_args)
        return EnumType(name=t.name, type_args=new_type_args, module_id=t.module_id)
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
    if isinstance(t, (RecordType, EnumType)):
        return any(contains_type_var(ta) for ta in t.type_args)
    return False


# ---------------------------------------------------------------------------
# Built-in exception types
#
# These are pure handles — ``module_id=PRELUDE_ID`` — carrying no field data
# of their own; the shapes are the single source of truth defined once as
# ``TypeDef`` literals in ``semantics.type_table.BUILTIN_EXCEPTION_TYPE_DEFS``
# (registered into every fresh ``TypeTable`` by ``create_seeded_type_table``).
# ---------------------------------------------------------------------------

# Abstract base: the hierarchy root, catchable but not constructible.
EXCEPTION_BASE = ExceptionType(name="Exception", module_id=PRELUDE_ID)

BUILTIN_EXCEPTIONS: dict[str, ExceptionType] = {
    "Exception": EXCEPTION_BASE,
    "AgentCallError": ExceptionType(name="AgentCallError", module_id=PRELUDE_ID),
    "AgentParseError": ExceptionType(name="AgentParseError", module_id=PRELUDE_ID),
    "ExecError": ExceptionType(name="ExecError", module_id=PRELUDE_ID),
    # Raised for every runtime failure crossing an extern (Python FFI) call:
    # the Python callable raising, a return-contract violation (including a
    # seal violation), or an argument-conversion failure.
    "ExternError": ExceptionType(name="ExternError", module_id=PRELUDE_ID),
    "MaxIterationsExceeded": ExceptionType(name="MaxIterationsExceeded", module_id=PRELUDE_ID),
    "MatchError": ExceptionType(name="MatchError", module_id=PRELUDE_ID),
    "IndexError": ExceptionType(name="IndexError", module_id=PRELUDE_ID),
    "KeyError": ExceptionType(name="KeyError", module_id=PRELUDE_ID),
    "TypeError": ExceptionType(name="TypeError", module_id=PRELUDE_ID),
    "ArithmeticError": ExceptionType(name="ArithmeticError", module_id=PRELUDE_ID),
    # Statically prevented by scope/typecheck (assignment to immutable bindings
    # and undeclared names), but still listed as catchable runtime
    # exceptions for any runtime paths that bypass the static passes.
    "UndefinedVariableError": ExceptionType(name="UndefinedVariableError", module_id=PRELUDE_ID),
    "ImmutableBindingError": ExceptionType(name="ImmutableBindingError", module_id=PRELUDE_ID),
    "Abort": ExceptionType(name="Abort", module_id=PRELUDE_ID),
    # AgL: RecursionError raised when the call-depth limit is exceeded.
    "RecursionError": ExceptionType(name="RecursionError", module_id=PRELUDE_ID),
    "CastError": ExceptionType(name="CastError", module_id=PRELUDE_ID),
    "JsonParseError": ExceptionType(name="JsonParseError", module_id=PRELUDE_ID),
    "RangeError": ExceptionType(name="RangeError", module_id=PRELUDE_ID),
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
# These prelude constants are pure handles — their field/variant shapes are
# defined once as explicit ``TypeDef`` literals in
# ``semantics.type_table.BUILTIN_PRELUDE_TYPE_DEFS``.
_EXEC_RESULT_TYPE = RecordType(name="ExecResult", module_id=PRELUDE_ID)

# ``ParsePolicy`` — controls ``ask``/``exec`` error handling.
# ``Abort`` — abort on parse error (no fields).
# ``Retry(n: int)`` — retry up to ``n`` times.
_PARSE_POLICY_TYPE = EnumType(name="ParsePolicy", module_id=PRELUDE_ID)

_OPTION_TEXT_TYPE = EnumType(name="Option", type_args=(TextType(),), module_id=STD_CORE_ID)

# Public alias for the ``Option[text]`` type — the single source of truth
# shared with engine_keys and any other module that needs this type.
OPTION_TEXT_TYPE: EnumType = _OPTION_TEXT_TYPE

_OPTION_JSON_TYPE = EnumType(name="Option", type_args=(JsonType(),), module_id=STD_CORE_ID)

_OUTPUT_CONTRACT_TYPE = RecordType(name="OutputContract", module_id=PRELUDE_ID)

_OUTPUT_CONTRACT_OPTION_TYPE = EnumType(name="OutputContractOption", module_id=PRELUDE_ID)

# ``AgentRequest`` — the request that the corresponding ``ask`` call would
# dispatch to its agent, surfaced as an AgL value by ``ask-request``.  This is
# the first-attempt request: ``attempt`` is always ``0`` and there is no
# retry context (no ``previous_invalid_output`` / ``validation_errors``),
# because ``ask-request`` never invokes the agent.
_AGENT_REQUEST_TYPE = RecordType(name="AgentRequest", module_id=PRELUDE_ID)

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
