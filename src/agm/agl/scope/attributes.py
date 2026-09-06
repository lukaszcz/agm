"""Recognition of the attributes a module's declarations carry.

The parser keeps every attribute exactly as written. This step is where an
attribute acquires meaning: one walk over a module's declarations checks each
one against :mod:`agm.agl.attributes` — the name exists, the declaration is an
admitted target, the arguments match the declared schema, the attribute is not
repeated or contradicted — and turns the surviving attributes into typed
side-table entries.

Four facts are built: a parameter's zone, from the ``@arg-*`` attribute an
entry or its owning declaration carries; an ``extern def``'s Python companion
name, from ``@extern-name`` — the walk sees every extern of a module, so it is
also where their companion names are held apart; a ``program def`` parameter's
command-line presentation, from the ``@opt-*`` attributes; and a declaration's
documentation text, from ``@doc``. The walk is the seam a further attribute
meaning joins through — a new fact reads the attributes the walk already hands
it and fills a table of its own, so it costs one more builder, never one more
traversal.
"""

from __future__ import annotations

import keyword
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from agm.agl.attributes import (
    BUILTIN_ATTRIBUTES,
    DOC_ATTRIBUTE,
    EXTERN_NAME_ATTRIBUTE,
    NAME_ADDRESSED_OPTION_ATTRIBUTES,
    OPTION_ENV_ATTRIBUTE,
    OPTION_HIDDEN_ATTRIBUTE,
    OPTION_METAVAR_ATTRIBUTE,
    OPTION_NAME_ATTRIBUTE,
    OPTION_SHORT_ATTRIBUTE,
    ZONE_ATTRIBUTES,
    AttributeArguments,
    AttributeSpec,
    AttributeTarget,
    ProgramOptionSpec,
)
from agm.agl.scope.symbols import AglScopeError, AttributeFacts
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
class _Recognized:
    """One declaration's validated attribute prefix, ready for a fact builder.

    ``nodes`` locates every attribute the declaration carries by name — for a
    builder reporting one that may not sit where it does — and ``texts`` the
    already-validated argument of every attribute whose schema takes one.
    """

    nodes: dict[str, Attribute]
    texts: dict[str, str]

    def text_of(self, name: str) -> str | None:
        """Return the text argument attribute *name* carries, if it is present."""
        return self.texts.get(name)


#: The order entries run in: positional-only, then standard, then named-only.
_ZONE_SEQUENCE: tuple[ParamZone, ...] = (
    ParamZone.POSITIONAL_ONLY,
    ParamZone.STANDARD,
    ParamZone.NAMED_ONLY,
)

#: Declarations carrying both their own attributes and an attributed field list.
_FIELD_OWNERS: Mapping[type[object], AttributeTarget] = {
    RecordDef: AttributeTarget.RECORD,
    ExceptionDef: AttributeTarget.EXCEPTION,
    VariantDef: AttributeTarget.ENUM_MEMBER,
}

#: Declarations whose attributes attach to the declaration and nothing else.
_PLAIN_TARGETS: Mapping[type[object], AttributeTarget] = {
    EnumDef: AttributeTarget.ENUM,
    TypeAlias: AttributeTarget.TYPE_ALIAS,
    LetDecl: AttributeTarget.BINDING,
    VarDecl: AttributeTarget.BINDING,
    BuiltinVarDecl: AttributeTarget.BINDING,
}

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
    return AttributeFacts(
        param_zones=recognizer.param_zones,
        extern_names=recognizer.extern_names,
        program_options=recognizer.program_options,
        docs=recognizer.docs,
    )


class _Recognizer:
    """One module's attribute walk, accumulating facts as it goes."""

    def __init__(self, declares_receiver: Callable[[FuncDef], bool]) -> None:
        self._declares_receiver = declares_receiver
        self.param_zones: dict[int, ParamZone] = {}
        self.extern_names: dict[int, str] = {}
        self.program_options: dict[int, ProgramOptionSpec] = {}
        self.docs: dict[int, str] = {}
        self._companion_owners: dict[str, str] = {}

    def visit(self, node: object) -> None:
        """Recognize the attributes of *node*, if it is a defining declaration."""
        if isinstance(node, FuncDef):
            recognized = self._check(node.attributes, _function_target(node), node.node_id)
            if node.is_extern:
                self._extern_name(node, recognized)
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
        elif isinstance(node, (RecordDef, ExceptionDef, VariantDef)):
            self._check(node.attributes, _FIELD_OWNERS[type(node)], node.node_id)
            self._entries(node.fields, node.attributes, target=AttributeTarget.FIELD)
        elif isinstance(node, (EnumDef, TypeAlias, LetDecl, VarDecl, BuiltinVarDecl)):
            self._check(node.attributes, _PLAIN_TARGETS[type(node)], node.node_id)

    # ------------------------------------------------------------------
    # Catalog validation
    # ------------------------------------------------------------------

    def _check(
        self, attributes: tuple[Attribute, ...], target: AttributeTarget, node_id: int
    ) -> _Recognized:
        """Validate the whole attribute prefix of the declaration *node_id*.

        Returns every attribute the declaration carries and the text argument
        of those whose schema takes one, keyed by attribute name, so a fact
        builder reads an already-validated argument instead of re-inspecting
        the raw node.
        Documentation text, admitted on every declaration kind, is filed here
        rather than in a builder of its own.
        """
        nodes: dict[str, Attribute] = {}
        texts: dict[str, str] = {}
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
            if attribute.name in nodes:
                raise AglScopeError(
                    f"Attribute '@{attribute.name}' cannot be repeated.", span=attribute.span
                )
            for other in spec.conflicts:
                if other in nodes:
                    raise AglScopeError(
                        f"Attribute '@{attribute.name}' conflicts with '@{other}'.",
                        span=attribute.span,
                    )
            text = _check_arguments(attribute, spec)
            if text is not None:
                texts[attribute.name] = text
            nodes[attribute.name] = attribute
        recognized = _Recognized(nodes=nodes, texts=texts)
        documentation = recognized.text_of(DOC_ATTRIBUTE)
        if documentation is not None:
            self.docs[node_id] = documentation
        return recognized

    # ------------------------------------------------------------------
    # Fact builder: extern companion names
    # ------------------------------------------------------------------

    def _extern_name(self, node: FuncDef, recognized: _Recognized) -> None:
        """Record the Python companion name one ``extern def`` resolves to.

        *recognized* is the declaration's validated attribute prefix; its
        ``@extern-name`` argument names the companion when it carries one, and
        without it the declared member name is the companion name verbatim.
        The companion module has to define a Python function spelled exactly
        this way, so the effective name must be a valid identifier that is not
        a hard Python keyword — soft keywords (``match``, ``type``, …) are
        legal Python ``def`` names and pass — and a module's externs, wherever
        they are declared, must each claim a different one of its companion's
        functions.
        """
        supplied = recognized.nodes.get(EXTERN_NAME_ATTRIBUTE)
        name = node.name if supplied is None else recognized.texts[EXTERN_NAME_ATTRIBUTE]
        span = node.span if supplied is None else supplied.span
        if not name.isidentifier() or keyword.iskeyword(name):
            remedy = (
                ""
                if supplied is not None
                else f" Supply one with '@{EXTERN_NAME_ATTRIBUTE}(\"python_name\")'."
            )
            raise AglScopeError(
                f"Extern companion name '{name}' must be a valid Python identifier and not a "
                "Python keyword, because the companion module must define a Python function "
                f"with exactly this name.{remedy}",
                span=span,
            )
        owner = self._companion_owners.get(name)
        if owner is not None:
            raise AglScopeError(
                f"Extern declarations '{owner}' and '{node.name}' both map to companion "
                f"function '{name}'; a module's externs need distinct companion names, "
                f"supplied with '@{EXTERN_NAME_ATTRIBUTE}'.",
                span=span,
            )
        self._companion_owners[name] = node.name
        self.extern_names[node.node_id] = name

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
        highest = _ZONE_SEQUENCE[0]
        for index, entry in enumerate(entries):
            recognized = self._check(entry.attributes, target, entry.node_id)
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
            if _ZONE_SEQUENCE.index(zone) < _ZONE_SEQUENCE.index(highest):
                raise AglScopeError(
                    f"{entry.name!r} is {_ZONE_LABEL[zone]} but follows a "
                    f"{_ZONE_LABEL[highest]} entry; entries are ordered "
                    "positional-only, then standard, then named-only.",
                    span=entry.span,
                )
            highest = zone
            self.param_zones[entry.node_id] = zone
            if target is AttributeTarget.PROGRAM_PARAMETER:
                self._program_option(entry, recognized, zone)

    # ------------------------------------------------------------------
    # Fact builder: program option presentation
    # ------------------------------------------------------------------

    def _program_option(self, entry: Param, recognized: _Recognized, zone: ParamZone) -> None:
        """Record how one ``program def`` parameter presents itself to a host.

        Every program parameter gets an entry: without any ``@opt-*``
        attribute it is addressed by its declared name and carries nothing
        else. A positional-only parameter is addressed by position alone, so
        the attributes that only mean something for a named one are rejected
        rather than silently ignored, and the two argument shapes a host has
        to spell — a one-letter short option and a flag word it can also
        negate — are checked here, where the attribute is still in view.
        """
        if zone is ParamZone.POSITIONAL_ONLY:
            for name in NAME_ADDRESSED_OPTION_ATTRIBUTES:
                attribute = recognized.nodes.get(name)
                if attribute is not None:
                    raise AglScopeError(
                        f"Attribute '@{name}' cannot be attached to positional-only parameter "
                        f"{entry.name!r}, which a host addresses by position and never by name.",
                        span=attribute.span,
                    )
        external = recognized.text_of(OPTION_NAME_ATTRIBUTE)
        self.program_options[entry.node_id] = ProgramOptionSpec(
            name=entry.name if external is None else external,
            short=recognized.text_of(OPTION_SHORT_ATTRIBUTE),
            env=recognized.text_of(OPTION_ENV_ATTRIBUTE),
            metavar=recognized.text_of(OPTION_METAVAR_ATTRIBUTE),
            hidden=OPTION_HIDDEN_ATTRIBUTE in recognized.nodes,
            doc=recognized.text_of(DOC_ATTRIBUTE),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_arguments(attribute: Attribute, spec: AttributeSpec) -> str | None:
    """Reject arguments a built-in attribute's literal schema does not admit.

    Returns the text an attribute taking one argument carries, and ``None``
    for an attribute whose schema takes none. A schema narrowing that text to
    a spelling a host has to form (``AttributeSpec.pattern``) is enforced
    here too, so a fact builder downstream reads an argument already known to
    be well shaped.
    """
    if attribute.named_args:
        raise AglScopeError(
            f"Attribute '@{attribute.name}' takes no named argument.", span=attribute.span
        )
    if spec.arguments is AttributeArguments.NONE:
        if attribute.args:
            raise AglScopeError(
                f"Attribute '@{attribute.name}' takes no arguments.", span=attribute.span
            )
        return None
    if len(attribute.args) != 1:
        raise AglScopeError(
            f"Attribute '@{attribute.name}' takes exactly one text argument.",
            span=attribute.span,
        )
    argument = attribute.args[0]
    if not isinstance(argument, StringLit):
        raise AglScopeError(
            f"Attribute '@{attribute.name}' requires a plain text literal argument.",
            span=attribute.span,
        )
    if spec.pattern is not None and spec.pattern.fullmatch(argument.value) is None:
        raise AglScopeError(
            f"Attribute '@{attribute.name}' {spec.expected} {argument.value!r}.",
            span=attribute.span,
        )
    return argument.value


def _zone_attribute(attributes: tuple[Attribute, ...]) -> ParamZone | None:
    """Return the zone the ``@arg-*`` attribute among *attributes* selects."""
    for attribute in attributes:
        zone = ZONE_ATTRIBUTES.get(attribute.name)
        if zone is not None:
            return zone
    return None
