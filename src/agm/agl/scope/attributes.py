"""Recognition of the attributes a module's declarations carry.

The parser keeps every attribute exactly as written. This step is where an
attribute acquires meaning: one walk over a module's declarations checks each
one against :mod:`agm.agl.attributes` — the name exists, the declaration is an
admitted target, the arguments match the declared schema, the attribute is not
repeated or contradicted — and turns the surviving attributes into typed
side-table entries.

One fact is built: a parameter's zone, from the ``@arg-*`` attribute an entry
or its owning declaration carries. The walk is the seam a further attribute
meaning joins through — a new fact reads the attributes the walk already
hands it and fills a table of its own, so it costs one more builder, never one
more traversal.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from agm.agl.attributes import (
    BUILTIN_ATTRIBUTES,
    ZONE_ATTRIBUTES,
    AttributeArguments,
    AttributeSpec,
    AttributeTarget,
)
from agm.agl.scope.symbols import AglScopeError
from agm.agl.syntax.nodes import (
    Attribute,
    BuiltinVarDecl,
    EnumDef,
    ExceptionDef,
    FuncDef,
    Lambda,
    LetDecl,
    Param,
    Program,
    RecordDef,
    StringLit,
    TypeAlias,
    VarDecl,
    VariantDef,
)
from agm.agl.syntax.visitor import walk
from agm.agl.zones import ParamZone

__all__ = ["AttributeFacts", "recognize_attributes"]


@dataclass(frozen=True, slots=True)
class AttributeFacts:
    """The typed facts one module's recognized attributes carry.

    ``param_zones``
        The zone of every function, lambda, record, exception, and enum-member
        entry, keyed by its ``Param.node_id``.
    """

    param_zones: dict[int, ParamZone]


#: Entry order: positional-only, then standard, then named-only.
_ZONE_ORDER: Mapping[ParamZone, int] = {
    ParamZone.POSITIONAL_ONLY: 0,
    ParamZone.STANDARD: 1,
    ParamZone.NAMED_ONLY: 2,
}

_ZONE_BY_ORDER: Mapping[int, ParamZone] = {order: zone for zone, order in _ZONE_ORDER.items()}

#: How each zone is named in a diagnostic.
_ZONE_LABEL: Mapping[ParamZone, str] = {
    ParamZone.POSITIONAL_ONLY: "positional-only",
    ParamZone.STANDARD: "standard",
    ParamZone.NAMED_ONLY: "named-only",
}

#: How each target kind is named in a diagnostic.
_TARGET_LABEL: Mapping[AttributeTarget, str] = {
    AttributeTarget.FUNCTION: "function declaration",
    AttributeTarget.PROGRAM: "program declaration",
    AttributeTarget.EXTERN: "extern declaration",
    AttributeTarget.BUILTIN_FUNCTION: "builtin function declaration",
    AttributeTarget.RECORD: "record declaration",
    AttributeTarget.ENUM: "enum declaration",
    AttributeTarget.ENUM_MEMBER: "enum member",
    AttributeTarget.EXCEPTION: "exception declaration",
    AttributeTarget.TYPE_ALIAS: "type alias",
    AttributeTarget.BINDING: "binding",
    AttributeTarget.PARAMETER: "parameter",
    AttributeTarget.FIELD: "field",
    AttributeTarget.PROGRAM_PARAMETER: "program parameter",
}


def _function_target(node: FuncDef) -> AttributeTarget:
    """Return the target kind of one ``def`` flavor."""
    if node.is_program:
        return AttributeTarget.PROGRAM
    if node.is_extern:
        return AttributeTarget.EXTERN
    if node.is_builtin:
        return AttributeTarget.BUILTIN_FUNCTION
    return AttributeTarget.FUNCTION


def recognize_attributes(
    program: Program, *, declares_receiver: Callable[[FuncDef], bool]
) -> AttributeFacts:
    """Validate *program*'s declaration attributes and build their facts.

    *declares_receiver* reports scope's method classification for a ``def``:
    a classified method's first parameter is its ``self`` receiver, which is
    positional-only and admits no zone attribute of its own.
    """
    recognizer = _Recognizer(declares_receiver)
    walk(program, recognizer.visit)
    return AttributeFacts(param_zones=recognizer.param_zones)


class _Recognizer:
    """One module's attribute walk, accumulating facts as it goes."""

    def __init__(self, declares_receiver: Callable[[FuncDef], bool]) -> None:
        self._declares_receiver = declares_receiver
        self.param_zones: dict[int, ParamZone] = {}

    def visit(self, node: object) -> None:
        """Recognize the attributes of *node*, if it is a defining declaration."""
        if isinstance(node, FuncDef):
            self._check(node.attributes, _function_target(node))
            self._entries(
                node.params,
                node.attributes,
                target=(
                    AttributeTarget.PROGRAM_PARAMETER
                    if node.is_program
                    else AttributeTarget.PARAMETER
                ),
                form_default=(ParamZone.NAMED_ONLY if node.is_program else ParamZone.STANDARD),
                has_receiver=self._declares_receiver(node),
                owner_name=node.name,
            )
        elif isinstance(node, Lambda):
            self._entries(node.params, (), target=AttributeTarget.PARAMETER)
        elif isinstance(node, RecordDef):
            self._check(node.attributes, AttributeTarget.RECORD)
            self._entries(node.fields, node.attributes, target=AttributeTarget.FIELD)
        elif isinstance(node, ExceptionDef):
            self._check(node.attributes, AttributeTarget.EXCEPTION)
            self._entries(node.fields, node.attributes, target=AttributeTarget.FIELD)
        elif isinstance(node, VariantDef):
            self._check(node.attributes, AttributeTarget.ENUM_MEMBER)
            self._entries(node.fields, node.attributes, target=AttributeTarget.FIELD)
        elif isinstance(node, EnumDef):
            self._check(node.attributes, AttributeTarget.ENUM)
        elif isinstance(node, TypeAlias):
            self._check(node.attributes, AttributeTarget.TYPE_ALIAS)
        elif isinstance(node, (LetDecl, VarDecl, BuiltinVarDecl)):
            self._check(node.attributes, AttributeTarget.BINDING)

    # ------------------------------------------------------------------
    # Catalog validation
    # ------------------------------------------------------------------

    def _check(self, attributes: tuple[Attribute, ...], target: AttributeTarget) -> None:
        """Validate one declaration's whole attribute prefix."""
        seen: set[str] = set()
        for attribute in attributes:
            spec = BUILTIN_ATTRIBUTES.get(attribute.name)
            if spec is None:
                raise AglScopeError(f"Unknown attribute '@{attribute.name}'.", span=attribute.span)
            if target not in spec.targets:
                raise AglScopeError(
                    f"Attribute '@{attribute.name}' cannot be attached to a "
                    f"{_TARGET_LABEL[target]}.",
                    span=attribute.span,
                )
            _check_arguments(attribute, spec)
            if attribute.name in seen and not spec.repeatable:
                raise AglScopeError(
                    f"Attribute '@{attribute.name}' cannot be repeated.", span=attribute.span
                )
            for other in spec.conflicts:
                if other in seen:
                    raise AglScopeError(
                        f"Attribute '@{attribute.name}' conflicts with '@{other}'.",
                        span=attribute.span,
                    )
            seen.add(attribute.name)

    # ------------------------------------------------------------------
    # Fact builder: parameter zones
    # ------------------------------------------------------------------

    def _entries(
        self,
        entries: tuple[Param, ...],
        owner_attributes: tuple[Attribute, ...],
        *,
        target: AttributeTarget,
        form_default: ParamZone = ParamZone.STANDARD,
        has_receiver: bool = False,
        owner_name: str | None = None,
    ) -> None:
        """Validate and zone every entry of one parameter or field list.

        An entry's own ``@arg-*`` attribute wins; otherwise the owning
        declaration's sets the list default; otherwise *form_default* applies.
        A classified method's receiver is positional-only regardless. Entries
        then have to run positional-only, standard, named-only.
        """
        declared_default = _zone_attribute(owner_attributes)
        list_default = form_default if declared_default is None else declared_default
        highest = _ZONE_ORDER[ParamZone.POSITIONAL_ONLY]
        for index, entry in enumerate(entries):
            self._check(entry.attributes, target)
            declared = _zone_attribute(entry.attributes)
            if has_receiver and index == 0:
                if declared is not None:
                    raise AglScopeError(
                        f"Receiver '{entry.name}' of '{owner_name}' is positional-only and "
                        "cannot carry a zone attribute.",
                        span=entry.span,
                    )
                zone = ParamZone.POSITIONAL_ONLY
            else:
                zone = list_default if declared is None else declared
            order = _ZONE_ORDER[zone]
            if order < highest:
                raise AglScopeError(
                    f"{entry.name!r} is {_ZONE_LABEL[zone]} but follows a "
                    f"{_ZONE_LABEL[_ZONE_BY_ORDER[highest]]} entry; entries are ordered "
                    "positional-only, then standard, then named-only.",
                    span=entry.span,
                )
            highest = order
            self.param_zones[entry.node_id] = zone


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_arguments(attribute: Attribute, spec: AttributeSpec) -> None:
    """Reject arguments a built-in attribute's literal schema does not admit."""
    if attribute.named_args:
        raise AglScopeError(
            f"Attribute '@{attribute.name}' takes no named argument.", span=attribute.span
        )
    if spec.arguments is AttributeArguments.NONE:
        if attribute.args:
            raise AglScopeError(
                f"Attribute '@{attribute.name}' takes no arguments.", span=attribute.span
            )
        return
    if len(attribute.args) != 1:
        raise AglScopeError(
            f"Attribute '@{attribute.name}' takes exactly one text argument.",
            span=attribute.span,
        )
    if not isinstance(attribute.args[0], StringLit):
        raise AglScopeError(
            f"Attribute '@{attribute.name}' requires a plain text literal argument.",
            span=attribute.span,
        )


def _zone_attribute(attributes: tuple[Attribute, ...]) -> ParamZone | None:
    """Return the zone the ``@arg-*`` attribute among *attributes* selects."""
    for attribute in attributes:
        zone = ZONE_ATTRIBUTES.get(attribute.name)
        if zone is not None:
            return zone
    return None
