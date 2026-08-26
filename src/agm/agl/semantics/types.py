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
- ``ArrayType(elem)`` — ``array[T]``.
- ``DictType(value)`` — ``dict[text, V]`` (keys are always ``text`` in AgL).
- ``RecordType(name, type_args, module_id, decl_id)`` — a ``record`` nominal
  type handle whose identity is ``decl_id``; field shapes live in the shared
  ``TypeTable`` (``semantics.type_table``), keyed by declaration identity.
- ``EnumType(name, type_args, module_id, decl_id)`` — an ``enum`` nominal type
  handle; variant shapes live in the shared ``TypeTable``.
- ``ExceptionType(name, module_id, decl_id)`` — an exception nominal type
  handle (never generic); field shapes and hierarchy (``abstract``, ``base``)
  live in the shared ``TypeTable``.
- ``UnitType`` — the ``unit`` type (AgL; single value ``()``).
- ``FunctionType(params, result)`` — a first-class function type (AgL),
  positional only; named/optional arguments are erased from the value type.
- ``TypeVarType(name)`` — a rigid type variable bound by an enclosing generic
  declaration.
- ``InferenceVarType(display_hint)`` — an internal, identity-based flexible
  variable owned by a checker inference region; it is not source-spellable.

``Type`` is the closed union of all semantic types.

Implicit coercion rules
-------------------------------------
``int → decimal`` widening, and absorbing a *scalar* JSON-shaped value into
``json``, are the only implicit type coercions — both are leaf conversions
that never rebuild a structure. Use :func:`is_assignable` to check
assignability with these coercions applied.

Type-kind strings (for codec capability lookup)
------------------------------------------------
Each ``Type`` exposes a ``kind`` property — a lower-cased string identifying
the type's kind in the ``HostCapabilities.codec_kinds`` maps.  E.g.
``TextType().kind == "text"``, ``RecordType(...).kind == "record"``.
"""

from __future__ import annotations

import enum as _enum
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from itertools import count
from typing import assert_never

from agm.agl.ir.reserved_nominals import NO_DECL_ID, reserved_nominal_id
from agm.agl.ir.reserved_nominals import require_reserved_nominal_id as _reserved_id
from agm.agl.modules.ids import ENTRY_ID, STD_CORE_ID, ModuleId

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
class ArrayType:
    """``array[T]`` — a homogeneous array."""

    elem: Type

    @property
    def kind(self) -> str:
        return "array"

    def __repr__(self) -> str:
        return f"array[{self.elem!r}]"


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
    identity is the declaration it names (``decl_id``), plus ``type_args`` for
    a generic instantiation. Field types are looked up by handle in the
    shared ``TypeTable`` (``semantics.type_table.TypeTable.record_fields``).
    ``type_args`` holds the resolved type arguments for a generic
    instantiation (empty tuple for non-generic records). ``module_id`` is the
    owning module (defaults to ``ENTRY_ID`` so existing module paths and
    built-in/prelude types are unaffected).

    ``decl_id`` is the identity of the declaration this handle names, or
    ``NO_DECL_ID`` when no declaration identity is attached. It participates
    in equality/hashing alongside ``type_args``, so two declarations sharing
    one name path are distinct types; ``name``/``module_id``/``scope_path``
    remain in equality too — they are consistent with ``decl_id`` for every
    real declaration — and are what resolution and display use.
    """

    name: str
    type_args: tuple[Type, ...] = ()
    module_id: ModuleId = field(default_factory=lambda: ENTRY_ID)
    scope_path: tuple[str, ...] = ()
    decl_id: int = NO_DECL_ID

    @property
    def kind(self) -> str:
        return "record"

    def __repr__(self) -> str:
        prefix = "" if spells_bare(self.module_id, self.name) else f"{self.module_id.path_str()}::"
        scoped_name = "::".join((*self.scope_path, self.name))
        if self.type_args:
            args_str = ", ".join(repr(a) for a in self.type_args)
            return f"{prefix}{scoped_name}[{args_str}]"
        return f"{prefix}{scoped_name}"


@dataclass(frozen=True, slots=True)
class EnumType:
    """An ``enum`` nominal type handle.

    An ``EnumType`` carries no variant data — it is a lightweight handle
    whose identity is the declaration it names (``decl_id``), plus
    ``type_args`` for a generic instantiation. Variant shapes are looked up
    by handle in the shared ``TypeTable``
    (``semantics.type_table.TypeTable.enum_members``). ``type_args`` holds
    the resolved type arguments for a generic instantiation (empty tuple for
    non-generic enums).  ``module_id`` is the owning module (defaults to
    ``ENTRY_ID``).

    ``decl_id`` is the identity of the declaration this handle names, or
    ``NO_DECL_ID`` when no declaration identity is attached. It participates
    in equality/hashing alongside ``type_args``, so two declarations sharing
    one name path are distinct types; ``name``/``module_id``/``scope_path``
    remain in equality too — they are consistent with ``decl_id`` for every
    real declaration — and are what resolution and display use.
    """

    name: str
    type_args: tuple[Type, ...] = ()
    module_id: ModuleId = field(default_factory=lambda: ENTRY_ID)
    scope_path: tuple[str, ...] = ()
    decl_id: int = NO_DECL_ID

    @property
    def kind(self) -> str:
        return "enum"

    def __repr__(self) -> str:
        prefix = "" if spells_bare(self.module_id, self.name) else f"{self.module_id.path_str()}::"
        scoped_name = "::".join((*self.scope_path, self.name))
        if self.type_args:
            args_str = ", ".join(repr(a) for a in self.type_args)
            return f"{prefix}{scoped_name}[{args_str}]"
        return f"{prefix}{scoped_name}"


@dataclass(frozen=True, slots=True)
class ExceptionType:
    """An exception nominal type handle.

    An ``ExceptionType`` carries no field data — it is a lightweight handle
    whose identity is the declaration it names (``decl_id``); exceptions are
    never generic, so there is no ``type_args`` component (unlike
    ``RecordType``/``EnumType``). Field shapes and hierarchy metadata
    (``abstract``, ``base``) are looked up by handle in the shared
    ``TypeTable`` (``semantics.type_table.TypeTable.exception_fields``/
    ``exception_def``). ``module_id`` is the owning module (defaults to
    ``ENTRY_ID``, like ``RecordType``/``EnumType``); a built-in exception's
    declaring module is the shipped standard library's own module
    (``STD_CORE_ID``) unless a program declares its own ``builtin exception``
    of that name, in which case it carries that program's module instead.

    The abstract ``Exception`` root is the ``TypeDef`` registered under name
    ``"Exception"`` with ``abstract=True`` and only a ``message`` field. It is
    not constructible; the source catch spelling ``Exception`` is the catch-all form.

    ``decl_id`` is the identity of the declaration this handle names, or
    ``NO_DECL_ID`` when no declaration identity is attached. It participates
    in equality/hashing; ``name``/``module_id``/``scope_path`` remain in
    equality too — they are consistent with ``decl_id`` for every real
    declaration — and are what resolution and display use.
    """

    name: str
    module_id: ModuleId = field(default_factory=lambda: ENTRY_ID)
    scope_path: tuple[str, ...] = ()
    decl_id: int = NO_DECL_ID

    @property
    def kind(self) -> str:
        return "exception"

    def __repr__(self) -> str:
        # Built-in exceptions and entry-module exceptions render as bare
        # names; other exceptions follow the record/enum qualification style.
        scoped_name = "::".join((*self.scope_path, self.name))
        if spells_bare(self.module_id, self.name):
            return scoped_name
        return f"{self.module_id.path_str()}::{scoped_name}"


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
class FunctionType:
    """A first-class function value type.

    Positional only — named and optional argument information is erased from
    the value type. The semantic type descriptor compares structurally by its
    frozen ``params`` tuple and ``result`` field; AgL function values cannot
    be compared with ``==`` or ``!=``.

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
    - Not JSON-shaped (``is_json_shaped`` returns ``False``), and never
      convertible to ``json``: a cast is compiled once with type arguments
      erased, so the conversion could not know what the variable stands for
      (``semantics.type_table.is_json_convertible``).
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


_inference_var_ids = count()


@dataclass(frozen=True, slots=True)
class InferenceVarType:
    """An internal flexible type variable identified by a fresh solver id.

    Unlike rigid :class:`TypeVarType`, this form has no source spelling and is
    never emitted from checking. ``display_hint`` is solver-only diagnostic
    metadata: equality and hashing use the private fresh identity alone.
    """

    display_hint: str = field(default="", compare=False, hash=False)
    _id: int = field(default_factory=lambda: next(_inference_var_ids), init=False, repr=False)

    @property
    def kind(self) -> str:
        return "inferencevar"

    def __repr__(self) -> str:
        return "<inference-var>"


# Closed union of all semantic types.
Type = (
    TextType
    | JsonType
    | BoolType
    | IntType
    | DecimalType
    | ArrayType
    | DictType
    | RecordType
    | EnumType
    | ExceptionType
    | UnitType
    | FunctionType
    | BottomType
    | TypeVarType
    | InferenceVarType
)


@dataclass(frozen=True, slots=True)
class TypeTemplate:
    """Resolved semantic type template with its declared inference variables."""

    template: Type
    type_params: tuple[str, ...] = ()

    def match(self, concrete: Type) -> TypeTemplateMatch | None:
        """Match this template exactly against one concrete semantic type."""
        return match_type_template(self.template, concrete, self.type_params)


@dataclass(frozen=True, slots=True)
class TypeTemplateMatch:
    """Exact one-sided match of declared type parameters to a concrete type."""

    bindings: tuple[tuple[str, Type], ...]

    @property
    def type_arguments(self) -> tuple[Type, ...]:
        """Return inferred arguments in declared parameter order."""
        return tuple(argument for _, argument in self.bindings)


def match_type_template(
    template: Type,
    concrete: Type,
    type_params: tuple[str, ...],
) -> TypeTemplateMatch | None:
    """Match ``template`` exactly against ``concrete`` from one side.

    Only ``TypeVarType`` names declared in ``type_params`` are inference
    variables. Repeated occurrences must agree, nominal identity is exact,
    and every declared parameter must be inferred. The result is immutable
    and ordered like ``type_params`` so alias argument reordering and fixed
    subterms require no caller-specific logic.
    """
    parameters = frozenset(type_params)
    inferred: dict[str, Type] = {}

    def visit_nominal(pattern: RecordType | EnumType, actual: RecordType | EnumType) -> bool:
        return (
            pattern.module_id == actual.module_id
            and pattern.scope_path == actual.scope_path
            and pattern.name == actual.name
            and len(pattern.type_args) == len(actual.type_args)
            and all(
                visit(pattern_arg, actual_arg)
                for pattern_arg, actual_arg in zip(pattern.type_args, actual.type_args, strict=True)
            )
        )

    def visit(pattern: Type, actual: Type) -> bool:
        if isinstance(pattern, TypeVarType) and pattern.name in parameters:
            previous = inferred.get(pattern.name)
            if previous is None:
                inferred[pattern.name] = actual
                return True
            return previous == actual
        if isinstance(pattern, ArrayType):
            return isinstance(actual, ArrayType) and visit(pattern.elem, actual.elem)
        if isinstance(pattern, DictType):
            return isinstance(actual, DictType) and visit(pattern.value, actual.value)
        if isinstance(pattern, FunctionType):
            return (
                isinstance(actual, FunctionType)
                and len(pattern.params) == len(actual.params)
                and all(
                    visit(pattern_param, actual_param)
                    for pattern_param, actual_param in zip(
                        pattern.params, actual.params, strict=True
                    )
                )
                and visit(pattern.result, actual.result)
            )
        if isinstance(pattern, RecordType):
            return isinstance(actual, RecordType) and visit_nominal(pattern, actual)
        if isinstance(pattern, EnumType):
            return isinstance(actual, EnumType) and visit_nominal(pattern, actual)
        return pattern == actual

    if not visit(template, concrete) or any(parameter not in inferred for parameter in type_params):
        return None
    return TypeTemplateMatch(tuple((parameter, inferred[parameter]) for parameter in type_params))


class EnumOwnerFormKind(_enum.Enum):
    """Checked source forms capable of owning an enum constructor spelling."""

    LOCAL = "local"
    SELF = "self"
    OPEN_IMPORT = "open_import"
    QUALIFIED_IMPORT = "qualified_import"


@dataclass(frozen=True, slots=True)
class EnumOwnerForm:
    """One immutable checked enum-owner source form.

    Source identity and template metadata are excluded from display equality;
    they retain the checked resolution needed to validate a concrete enum
    without reinterpreting import syntax downstream. This type describes only
    an owner spelling; which variants a module route makes ambiguous under
    that spelling is variant-level data carried alongside forms, not on them.
    """

    owner_name: str | None
    module_qualifier: tuple[str, ...] | None
    bare: bool = False
    qualifier_anchored: bool = False
    kind: EnumOwnerFormKind | None = field(default=None, compare=False)
    source_module_id: ModuleId | None = field(default=None, compare=False, repr=False)
    source_name: str | None = field(default=None, compare=False, repr=False)
    type_template: TypeTemplate | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.owner_name is None and self.module_qualifier is not None:
            raise ValueError("a bare constructor spelling cannot have a module qualifier")
        if self.bare and self.owner_name is not None:
            raise ValueError("a type-qualified constructor spelling cannot be bare")
        if self.owner_name is None:
            return
        kind = self.kind
        if kind is None:
            if self.module_qualifier is None:
                kind = EnumOwnerFormKind.LOCAL
            elif self.module_qualifier:
                kind = EnumOwnerFormKind.QUALIFIED_IMPORT
            else:
                kind = EnumOwnerFormKind.SELF
            object.__setattr__(self, "kind", kind)
        if kind in (EnumOwnerFormKind.LOCAL, EnumOwnerFormKind.OPEN_IMPORT):
            if self.module_qualifier is not None:
                raise ValueError("an unqualified enum owner form cannot have an import handle")
        elif kind is EnumOwnerFormKind.SELF:
            if self.module_qualifier != ():
                raise ValueError("a self-qualified enum owner form requires an empty qualifier")
        elif not self.module_qualifier:
            raise ValueError("a qualified-import enum owner form requires an import handle")
        if self.qualifier_anchored and not self.module_qualifier:
            raise ValueError("only a non-empty module qualifier can be anchored")

    def match(self, concrete: Type) -> TypeTemplateMatch | None:
        """Match this checked owner form against one concrete semantic type.

        Enum-owner aliases may carry phantom parameters, which cannot be
        inferred from a scrutinee but do not affect the enum they denote.
        """
        if self.type_template is None:
            return None
        return match_nominal_owner_template(self.type_template, concrete)


def match_nominal_owner_template(
    template: TypeTemplate, concrete: Type
) -> TypeTemplateMatch | None:
    """Match a nominal owner template while permitting uninferred phantom parameters."""
    bindings = match_type_template(template.template, concrete, ())
    if bindings is not None:
        return bindings

    # ``match_type_template`` needs declared variables to compare variable
    # occurrences. Give phantom variables a concrete sentinel, then omit them
    # from the resulting match; only variables present in the template matter.
    occurring = free_type_vars(template.template)
    parameters = tuple(parameter for parameter in template.type_params if parameter in occurring)
    if not parameters:
        return TypeTemplateMatch(()) if template.template == concrete else None
    return match_type_template(template.template, concrete, parameters)


def type_children(t: Type) -> tuple[Type, ...]:
    """Return *t*'s direct structural children, if any.

    This is the single constructor walk shared by type traversals. Nominal
    handles expose only their explicit type arguments; their declaration
    shapes remain owned by ``TypeTable`` and are never expanded here.
    """
    match t:
        case ArrayType(elem=elem):
            return (elem,)
        case DictType(value=value):
            return (value,)
        case FunctionType(params=params, result=result):
            return (*params, result)
        case RecordType(type_args=type_args) | EnumType(type_args=type_args):
            return type_args
        case (
            TextType()
            | JsonType()
            | BoolType()
            | IntType()
            | DecimalType()
            | ExceptionType()
            | UnitType()
            | BottomType()
            | TypeVarType()
            | InferenceVarType()
        ):
            return ()
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def iter_type(t: Type) -> Iterator[Type]:
    """Yield *t* and every nested structural type in pre-order."""
    yield t
    for child in type_children(t):
        yield from iter_type(child)


def iter_nominal_types(t: Type) -> Iterator[RecordType | EnumType | ExceptionType]:
    """Yield every nominal handle reachable from *t*, including *t* itself."""
    for part in iter_type(t):
        if isinstance(part, (RecordType, EnumType, ExceptionType)):
            yield part


def replace_type_children(t: Type, children: tuple[Type, ...]) -> Type:
    """Return *t* rebuilt with its direct structural *children*."""
    match t:
        case ArrayType():
            return ArrayType(children[0])
        case DictType():
            return DictType(children[0])
        case FunctionType(params=params):
            return FunctionType(params=children[: len(params)], result=children[-1])
        case RecordType(name=name, module_id=module_id, scope_path=scope_path, decl_id=decl_id):
            return RecordType(
                name=name,
                type_args=children,
                module_id=module_id,
                scope_path=scope_path,
                decl_id=decl_id,
            )
        case EnumType(name=name, module_id=module_id, scope_path=scope_path, decl_id=decl_id):
            return EnumType(
                name=name,
                type_args=children,
                module_id=module_id,
                scope_path=scope_path,
                decl_id=decl_id,
            )
        case (
            TextType()
            | JsonType()
            | BoolType()
            | IntType()
            | DecimalType()
            | ExceptionType()
            | UnitType()
            | BottomType()
            | TypeVarType()
            | InferenceVarType()
        ):
            return t
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def transform_type(t: Type, transform: Callable[[Type], Type]) -> Type:
    """Recursively rebuild *t*, applying ``transform`` bottom-up to every node."""
    children = tuple(transform_type(child, transform) for child in type_children(t))
    return transform(replace_type_children(t, children))


def _reserved_or_absent(name: str) -> int:
    """Return *name*'s reserved declaration identity, or ``NO_DECL_ID``."""
    reserved = reserved_nominal_id(name)
    return NO_DECL_ID if reserved is None else reserved


def reroot_type(
    t: Type,
    prefix: tuple[str, ...],
    remap_module: tuple[ModuleId, ModuleId] | None = None,
) -> Type:
    """Return *t* re-rooted onto a canonical frame, for shape comparison.

    A declaration inside a named scope region resolves its own nominal
    references (a record/enum field, an exception's base) under that same
    region path. Comparing such a reference against a canonical shape defined
    at scope path ``()`` therefore needs the two re-rooted onto the same
    frame first: this strips *prefix* from a ``RecordType``/``EnumType``/
    ``ExceptionType`` node's ``scope_path`` wherever it starts with *prefix*,
    leaving every other node (including a nominal reference declared
    elsewhere, whose ``scope_path`` does not start with *prefix*) unchanged.

    *remap_module*, when given as ``(from_module, to_module)``, additionally
    rewrites a nominal handle's ``module_id`` from *from_module* to
    *to_module* wherever it matches — a reference naming a sibling declared
    in the same module as the declaration being re-rooted names *that*
    declaration's own module, which must map onto the canonical shape's
    module the same way; a reference to a type from any other module is left
    alone, so a genuine cross-module mismatch is still rejected. The two
    adjustments are independent: a handle already at scope path ``()`` still
    needs its module remapped, and a handle outside *prefix* still needs
    nothing stripped. Callers that only need the scope-path adjustment (an
    empty *prefix* has nothing to strip) omit *remap_module*.

    A reference's ``decl_id`` denotes the *specific* declaration it names,
    which necessarily differs between an arbitrary declaration and the
    canonical ``std/core`` one being compared against, even when the two
    denote the same host type. So whenever a reference's ``module_id`` is
    remapped onto the canonical module, its ``decl_id`` is normalized too: to
    the reserved identity for its name when that name is a host-known
    reserved nominal (the canonical shape's own handles carry exactly that
    identity), or to ``NO_DECL_ID`` otherwise. A reference whose module is
    left alone keeps its ``decl_id`` unchanged.
    """

    def strip(node: Type) -> Type:
        if not isinstance(node, (RecordType, EnumType, ExceptionType)):
            return node
        scope_path = node.scope_path
        module_id = node.module_id
        decl_id = node.decl_id
        if scope_path[: len(prefix)] == prefix:
            scope_path = scope_path[len(prefix) :]
        if remap_module is not None and module_id == remap_module[0]:
            module_id = remap_module[1]
            decl_id = _reserved_or_absent(node.name)
        if (
            scope_path == node.scope_path
            and module_id == node.module_id
            and decl_id == node.decl_id
        ):
            return node
        return replace(node, scope_path=scope_path, module_id=module_id, decl_id=decl_id)

    return transform_type(t, strip)


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
# Assignability helpers
# ---------------------------------------------------------------------------


_SCALAR_JSON_SHAPED_TYPES: tuple[type[Type], ...] = (
    TextType,
    JsonType,
    BoolType,
    IntType,
    DecimalType,
)


def is_scalar_json_shaped(value_type: Type) -> bool:
    """Return ``True`` if ``value_type`` is a *scalar* JSON-shaped type.

    The scalar JSON-shaped types are ``null``/``json``, ``bool``, ``int``,
    ``decimal``, and ``text`` — the leaf types an implicit coercion may absorb
    into a ``json`` slot without rebuilding any structure. ``array``/``dict``
    are JSON-shaped (see :func:`is_json_shaped`) but not scalar: absorbing one
    into ``json`` would require rebuilding the whole container, which is an
    implicit deep copy, so it requires an explicit ``as json`` cast instead.
    """
    return isinstance(value_type, _SCALAR_JSON_SHAPED_TYPES)


def is_json_shaped(value_type: Type) -> bool:
    """Return ``True`` if ``value_type`` is JSON-shaped.

    JSON-shaped types are the values that may inhabit a ``json`` slot:
    ``null``/``json``, ``bool``, ``int``, ``decimal``, ``text``, and
    ``array``/``dict`` whose element/value types are themselves JSON-shaped.
    Records, enums, and exceptions are **not** JSON-shaped — explicitly cast
    one with ``as json`` to convert it to its structural JSON representation.

    AgL: ``UnitType`` and ``FunctionType`` are also NOT
    JSON-shaped; function values render only as opaque handles.

    Three predicates answer three different questions and must not be
    conflated: this one decides ``json``-slot *inhabitation*,
    :func:`is_scalar_json_shaped` decides what an *implicit* coercion absorbs
    into a ``json`` slot, and
    :func:`~agm.agl.semantics.type_table.is_json_convertible` — wider than
    both, since a nominal does have a JSON representation — decides what an
    explicit ``as json`` cast accepts.
    """
    if is_scalar_json_shaped(value_type):
        return True
    if isinstance(value_type, ArrayType):
        return is_json_shaped(value_type.elem)
    if isinstance(value_type, DictType):
        return is_json_shaped(value_type.value)
    if isinstance(value_type, InferenceVarType):
        return False
    # RecordType, EnumType, ExceptionType, UnitType, FunctionType,
    # BottomType, and TypeVarType are not JSON-shaped.
    return False


def is_assignable(value_type: Type, target_type: Type) -> bool:
    """Return ``True`` if ``value_type`` is assignable to ``target_type``.

    Implicit coercions:

    1. ``int → decimal`` widening.
    2. ``json`` accepts any *scalar* JSON-shaped value (rule 3): ``null``/
       ``json``, ``bool``, ``int``, ``decimal``, ``text``. An ``array`` or
       ``dict`` source — even one that is JSON-shaped — is rejected here: an
       implicit coercion never rebuilds a structure, so absorbing a container
       into ``json`` requires an explicit ``as json`` cast. Records/enums/
       exceptions are rejected outright (not JSON-shaped at all).

    All other assignments require exact structural equality.

    AgL: ``UnitType`` and ``FunctionType`` assignability is
    exact-only — no widening and no variance.  The
    ``value_type == target_type`` check below handles them; ``FunctionType``
    uses structural tuple equality on ``params`` + ``result``.

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
    # json accepts a scalar JSON-shaped value only; array/dict sources require
    # an explicit `as json` cast (see is_scalar_json_shaped).
    if isinstance(target_type, JsonType):
        return is_scalar_json_shaped(value_type)
    return False


# ---------------------------------------------------------------------------
# Generic type helpers
# ---------------------------------------------------------------------------


def free_type_vars(t: Type) -> frozenset[str]:
    """Collect free rigid source type-variable names in *t*."""
    return frozenset(node.name for node in iter_type(t) if isinstance(node, TypeVarType))


def substitute(t: Type, subst: Mapping[str, Type]) -> Type:
    """Capture-free substitution of rigid ``TypeVarType`` names only."""

    def replace_rigid(node: Type) -> Type:
        if isinstance(node, TypeVarType):
            return subst.get(node.name, node)
        return node

    return transform_type(t, replace_rigid)


def contains_type_var(t: Type) -> bool:
    """Return whether *t* contains a rigid source declaration variable."""
    return any(isinstance(node, TypeVarType) for node in iter_type(t))


def contains_inference_var(t: Type) -> bool:
    """Return whether *t* contains a solver-owned flexible variable."""
    return any(isinstance(node, InferenceVarType) for node in iter_type(t))


# ---------------------------------------------------------------------------
# Built-in exception types
#
# These are pure handles — ``module_id=STD_CORE_ID``, the shipped standard
# library's own declaring module — carrying no field data of their own; the
# shapes are the single source of truth defined once as ``TypeDef`` literals
# in ``semantics.type_table.BUILTIN_EXCEPTION_TYPE_DEFS`` (registered into
# every fresh ``TypeTable`` by ``create_seeded_type_table``). A program that
# declares its own ``builtin exception`` of one of these names gets its own
# distinct handle instead, carrying that program's module.
# ---------------------------------------------------------------------------


def _builtin_exception(name: str) -> ExceptionType:
    """Build the built-in exception handle for *name*, carrying its reserved identity.

    Each name is spelled once, here, and stamped with its own reserved
    identity, so a handle can never be paired with another name's identity.
    """
    return ExceptionType(name=name, module_id=STD_CORE_ID, decl_id=_reserved_id(name))


# Abstract base: the hierarchy root, catchable but not constructible.
EXCEPTION_BASE = _builtin_exception("Exception")

BUILTIN_EXCEPTIONS: dict[str, ExceptionType] = {
    name: _builtin_exception(name)
    for name in (
        "Exception",
        "AgentCallError",
        "AgentParseError",
        "ExecError",
        # Raised for every runtime failure crossing an extern (Python FFI) call:
        # the Python callable raising, a return-contract violation (including a
        # seal violation), or an argument-conversion failure.
        "ExternError",
        "MaxIterationsExceeded",
        "MatchError",
        "IndexError",
        "KeyError",
        "TypeError",
        "ArithmeticError",
        # Statically prevented by scope/typecheck (assignment to immutable bindings
        # and undeclared names), but still listed as catchable runtime
        # exceptions for any runtime paths that bypass the static passes.
        "UndefinedVariableError",
        "ImmutableBindingError",
        "Abort",
        # AgL: RecursionError raised when the call-depth limit is exceeded.
        "RecursionError",
        "CastError",
        "JsonParseError",
        "RangeError",
        # Reference semantics makes cyclic array/dict values constructible; raised
        # when rendering or JSON conversion re-enters a container already on its
        # path. Extern array/dict arguments cross as lazy views; repr of a view or
        # FFI view rendering that reaches a cycle raises this exception instead.
        "CyclicValueError",
    )
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
_EXEC_RESULT_TYPE = RecordType(
    name="ExecResult", module_id=STD_CORE_ID, decl_id=_reserved_id("ExecResult")
)

# ``ParsePolicy`` — controls ``ask``/``exec`` error handling.
# ``Abort`` — abort on parse error (no fields).
# ``Retry(n: int)`` — retry up to ``n`` times.
_PARSE_POLICY_TYPE = EnumType(
    name="ParsePolicy", module_id=STD_CORE_ID, decl_id=_reserved_id("ParsePolicy")
)

# ``Agent`` — a plain enum data value that specifies an agent backend.
_AGENT_TYPE = EnumType(name="Agent", module_id=STD_CORE_ID, decl_id=_reserved_id("Agent"))

_OPTION_TEXT_TYPE = EnumType(
    name="Option",
    type_args=(TextType(),),
    module_id=STD_CORE_ID,
    decl_id=_reserved_id("Option"),
)

# Public alias for the ``Option[text]`` type — the single source of truth
# shared with engine_keys and any other module that needs this type.
OPTION_TEXT_TYPE: EnumType = _OPTION_TEXT_TYPE

_OPTION_JSON_TYPE = EnumType(
    name="Option",
    type_args=(JsonType(),),
    module_id=STD_CORE_ID,
    decl_id=_reserved_id("Option"),
)

_OUTPUT_CONTRACT_TYPE = RecordType(
    name="OutputContract", module_id=STD_CORE_ID, decl_id=_reserved_id("OutputContract")
)

_OUTPUT_CONTRACT_OPTION_TYPE = EnumType(
    name="OutputContractOption",
    module_id=STD_CORE_ID,
    decl_id=_reserved_id("OutputContractOption"),
)

# ``AgentRequest`` — the request that the corresponding ``ask`` call would
# dispatch to its agent, surfaced as an AgL value by ``ask-request``.  This is
# the first-attempt request: ``attempt`` is always ``0`` and there is no
# retry context (no ``previous_invalid_output`` / ``validation_errors``),
# because ``ask-request`` never invokes the agent.
_AGENT_REQUEST_TYPE = RecordType(
    name="AgentRequest", module_id=STD_CORE_ID, decl_id=_reserved_id("AgentRequest")
)

_SESSION_TRANSPORT_TYPE = EnumType(
    name="SessionTransport", module_id=STD_CORE_ID, decl_id=_reserved_id("SessionTransport")
)

_SESSION_TYPE = RecordType(name="Session", module_id=STD_CORE_ID, decl_id=_reserved_id("Session"))

_SESSION_STATS_TYPE = RecordType(
    name="SessionStats", module_id=STD_CORE_ID, decl_id=_reserved_id("SessionStats")
)

_SESSION_ERROR_TYPE = ExceptionType(
    name="SessionError", module_id=STD_CORE_ID, decl_id=_reserved_id("SessionError")
)

# These records represent host resources rather than source-constructible data.
HOST_MINTED_PRELUDE_TYPE_NAMES: frozenset[str] = frozenset({"Session"})
HOST_MINTED_PRELUDE_TYPE_IDS: frozenset[int] = frozenset(
    _reserved_id(name) for name in HOST_MINTED_PRELUDE_TYPE_NAMES
)

BUILTIN_PRELUDE_TYPES: dict[str, Type] = {
    "ExecResult": _EXEC_RESULT_TYPE,
    "ParsePolicy": _PARSE_POLICY_TYPE,
    "Agent": _AGENT_TYPE,
    "OutputContract": _OUTPUT_CONTRACT_TYPE,
    "OutputContractOption": _OUTPUT_CONTRACT_OPTION_TYPE,
    "AgentRequest": _AGENT_REQUEST_TYPE,
    "SessionTransport": _SESSION_TRANSPORT_TYPE,
    "Session": _SESSION_TYPE,
    "SessionStats": _SESSION_STATS_TYPE,
    "SessionError": _SESSION_ERROR_TYPE,
}

# Names of built-in prelude types (non-shadowable, like built-in exceptions).
BUILTIN_PRELUDE_TYPE_NAMES: frozenset[str] = frozenset(BUILTIN_PRELUDE_TYPES)

# Every bare name the host recognizes as a built-in exception or prelude
# record/enum — used by ``spells_bare`` to recognize the shipped standard
# library's own declaration of one of them, as opposed to an ordinary,
# non-builtin declaration in the same module (e.g. ``Option``).
_BUILTIN_HOST_NAMES: frozenset[str] = BUILTIN_EXCEPTION_NAMES | BUILTIN_PRELUDE_TYPE_NAMES


def terminal_name(display_name: str) -> str:
    """Return the last segment of a ``::``-qualified nominal display name.

    Display names carry the module route and scope path a reader would write;
    a host that keys on the declaration alone wants only that final segment.
    """
    return display_name.rsplit("::", maxsplit=1)[-1]


def spells_bare(module_id: ModuleId, name: str) -> bool:
    """Return whether a nominal owned by *module_id* named *name* spells bare.

    True for the entry module — a program's own declarations never need a
    qualifier — and for the shipped standard library's own declaration of one
    of its built-in names, so a built-in exception or prelude record/enum
    reads the same in diagnostics whether or not a program declares its own
    ``builtin`` alias for it. Any other module — including an ordinary,
    non-builtin declaration in the standard library itself, such as
    ``Option`` — still qualifies, matching how a reader would write it.

    Shared by ``RecordType``/``EnumType``/``ExceptionType.__repr__`` and
    ``semantics.type_table.qualified_decl_name``.
    """
    return module_id.is_entry or (module_id == STD_CORE_ID and name in _BUILTIN_HOST_NAMES)


# Legacy built-in types kept for compatibility with already-compiled tests and
# internal APIs.  They remain available as nominal types, but their constructors
# are not exported into source scope because std/core replaces this surface.
COMPATIBILITY_PRELUDE_TYPE_NAMES: frozenset[str] = frozenset(
    {"OutputContract", "OutputContractOption"}
)


class CastKind(_enum.Enum):
    """Classification of a cast operation.

    Assigned by ``semantics.type_table.cast_classification``, which lives
    beside the declaration table because deciding a cast to ``json`` needs the
    table's non-data-reachability flags.
    """

    TOTAL_NOOP = "TOTAL_NOOP"  # source already assignable to target (no-op/widen)
    TOTAL_RENDER = "TOTAL_RENDER"  # render data value to text; a cyclic walk can fail
    TOTAL_JSON = "TOTAL_JSON"  # convert to json; a cyclic walk can fail
    IDENTITY_UPCAST = "IDENTITY_UPCAST"  # member record → containing enum
    NOMINAL_DOWNCAST = "NOMINAL_DOWNCAST"  # enum → one of its member records
    FALLIBLE = "FALLIBLE"  # runtime-fallible conversion
    STATIC_ERROR = "STATIC_ERROR"  # statically impossible — raise AglTypeError


@dataclass(frozen=True, slots=True)
class CastSpec:
    """Resolved runtime cast descriptor stored in CheckedModule.cast_specs."""

    target_type: Type
    kind: CastKind
