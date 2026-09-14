"""Recognition of the attributes a module's declarations carry.

The parser keeps every attribute exactly as written. This step is where an
attribute acquires meaning: one walk over a module's declarations checks each
one against :mod:`agm.agl.attributes` — the name exists, the declaration is an
admitted target, the arguments match the declared schema, the attribute is not
repeated or contradicted — and turns the surviving attributes into typed
side-table entries.

Six facts are built: a parameter's zone, from the ``@arg-*`` attribute an
entry or its owning declaration carries; an ``extern def``'s Python companion
name, from ``@extern-name`` — the walk sees every extern of a module, so it is
also where their companion names are held apart; a ``program def`` parameter's
command-line presentation, from the ``@opt-*`` attributes; the package command
a ``program def`` registers itself as, from ``@command`` and its prose; a
declaration's documentation text, from ``@doc``; and a field, inline enum
member, or record declaration's external spellings, from ``@name``/
``@json-name``. The walk is the seam a further attribute meaning joins
through — a new fact reads the attributes the walk already hands it and fills
a table of its own, so it costs one more builder, never one more traversal.
"""

from __future__ import annotations

import keyword
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from agm.agl.attributes import (
    BUILTIN_ATTRIBUTES,
    COMMAND_ATTRIBUTE,
    COMMAND_PROSE_ATTRIBUTES,
    DESCRIPTION_ATTRIBUTE,
    DOC_ATTRIBUTE,
    EXTERN_NAME_ATTRIBUTE,
    HELP_ATTRIBUTE,
    JSON_NAME_ATTRIBUTE,
    NAME_ADDRESSED_OPTION_ATTRIBUTES,
    NAME_ATTRIBUTE,
    OPTION_ENV_ATTRIBUTE,
    OPTION_HIDDEN_ATTRIBUTE,
    OPTION_METAVAR_ATTRIBUTE,
    OPTION_NAME_ATTRIBUTE,
    OPTION_SHORT_ATTRIBUTE,
    ZONE_ATTRIBUTES,
    AttributeArguments,
    AttributeSpec,
    AttributeTarget,
    ProgramCommandSpec,
    ProgramOptionSpec,
    invalid_external_name,
    invalid_json_name,
    invalid_program_command_path,
)
from agm.agl.scope.symbols import AglScopeError, AttributeFacts
from agm.agl.semantics.external_names import ExternalName
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

__all__ = ["AttributeFacts", "recognize_attributes", "recognize_program_command"]


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


def recognize_program_command(node: FuncDef) -> ProgramCommandSpec | None:
    """Return the package command a ``program def`` registers, or ``None``.

    Validates *node*'s attribute prefix against the catalog exactly as the
    scope walk does — this is the one place both share, so a parse-only
    caller (package command discovery) and the full scope pass raise the
    same diagnostics for the same source. The caller is responsible for
    passing a ``program def``; this function trusts that and does not check it.
    """
    recognized = _validate_attribute_prefix(node.attributes, AttributeTarget.PROGRAM)
    return _program_command_spec(node, recognized)


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
        command_registrations=recognizer.command_registrations,
        docs=recognizer.docs,
        external_names=recognizer.external_names,
    )


class _Recognizer:
    """One module's attribute walk, accumulating facts as it goes."""

    def __init__(self, declares_receiver: Callable[[FuncDef], bool]) -> None:
        self._declares_receiver = declares_receiver
        self.param_zones: dict[int, ParamZone] = {}
        self.extern_names: dict[int, str] = {}
        self.program_options: dict[int, ProgramOptionSpec] = {}
        self.command_registrations: dict[int, ProgramCommandSpec] = {}
        self.docs: dict[int, str] = {}
        self.external_names: dict[int, ExternalName] = {}
        self._companion_owners: dict[str, str] = {}

    def visit(self, node: object) -> None:
        """Recognize the attributes of *node*, if it is a defining declaration."""
        if isinstance(node, FuncDef):
            recognized = self._check(node.attributes, _function_target(node), node.node_id)
            if node.is_extern:
                self._extern_name(node, recognized)
            if node.is_program:
                self._command_registration(node, recognized)
            self._entries(
                node.params,
                _zone_attribute(recognized),
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
            self._entries(node.params, None, target=AttributeTarget.PARAMETER)
        elif isinstance(node, (RecordDef, ExceptionDef, VariantDef)):
            recognized = self._check(node.attributes, _FIELD_OWNERS[type(node)], node.node_id)
            self._entries(node.fields, _zone_attribute(recognized), target=AttributeTarget.FIELD)
        elif isinstance(node, (EnumDef, TypeAlias, LetDecl, VarDecl, BuiltinVarDecl)):
            self._check(node.attributes, _PLAIN_TARGETS[type(node)], node.node_id)

    # ------------------------------------------------------------------
    # Catalog validation
    # ------------------------------------------------------------------

    def _check(
        self, attributes: tuple[Attribute, ...], target: AttributeTarget, node_id: int
    ) -> _Recognized:
        """Validate the whole attribute prefix of the declaration *node_id*.

        Delegates to :func:`_validate_attribute_prefix` and additionally files
        ``@doc`` text, admitted on every declaration kind, since documentation
        has no fact builder of its own.
        """
        recognized = _validate_attribute_prefix(attributes, target)
        documentation = recognized.text_of(DOC_ATTRIBUTE)
        if documentation is not None:
            self.docs[node_id] = documentation
        self._external_name(node_id, recognized)
        return recognized

    # ------------------------------------------------------------------
    # Fact builder: field/member/record external spellings
    # ------------------------------------------------------------------

    def _external_name(self, node_id: int, recognized: _Recognized) -> None:
        """Record a field/enum-member/record's ``@name``/``@json-name`` spellings.

        Only these three target kinds admit either attribute (enforced by the
        catalog before *recognized* exists), so this runs unconditionally for
        every declaration :meth:`_check` sees and simply finds nothing to
        file for the rest.
        """
        texts: dict[str, str | None] = {}
        for attribute, invalid_text in (
            (NAME_ATTRIBUTE, invalid_external_name),
            (JSON_NAME_ATTRIBUTE, invalid_json_name),
        ):
            attribute_node = recognized.nodes.get(attribute)
            text = recognized.text_of(attribute)
            if text is not None:
                assert attribute_node is not None
                invalid = invalid_text(text)
                if invalid is not None:
                    raise AglScopeError(
                        f"Attribute '@{attribute}' argument {text!r} {invalid}.",
                        span=attribute_node.span,
                    )
            texts[attribute] = text
        if texts[NAME_ATTRIBUTE] is None and texts[JSON_NAME_ATTRIBUTE] is None:
            return
        self.external_names[node_id] = ExternalName(
            name=texts[NAME_ATTRIBUTE], json_name=texts[JSON_NAME_ATTRIBUTE]
        )

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
    # Fact builder: package command registrations
    # ------------------------------------------------------------------

    def _command_registration(self, node: FuncDef, recognized: _Recognized) -> None:
        """Record the package command one ``program def`` registers itself as."""
        spec = _program_command_spec(node, recognized)
        if spec is not None:
            self.command_registrations[node.node_id] = spec

    # ------------------------------------------------------------------
    # Fact builder: parameter zones
    # ------------------------------------------------------------------

    def _entries(
        self,
        entries: tuple[Param, ...],
        owner_zone: ParamZone | None,
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
        list_default = form_default if owner_zone is None else owner_zone
        highest = _ZONE_SEQUENCE[0]
        for index, entry in enumerate(entries):
            recognized = self._check(entry.attributes, target, entry.node_id)
            declared = _zone_attribute(recognized)
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


def _validate_attribute_prefix(
    attributes: tuple[Attribute, ...], target: AttributeTarget
) -> _Recognized:
    """Validate one declaration's whole attribute prefix against the catalog.

    Checks that each attribute is known, admits *target*, is not repeated,
    conflicts with nothing else present, and carries an argument matching its
    schema. This is the one place both the full scope walk (:meth:`_Recognizer._check`)
    and a parse-only caller (:func:`recognize_program_command`) validate an
    attribute prefix, so both raise the same diagnostics for the same source.
    """
    nodes: dict[str, Attribute] = {}
    texts: dict[str, str] = {}
    for attribute in attributes:
        spec = BUILTIN_ATTRIBUTES.get(attribute.name)
        if spec is None:
            raise AglScopeError(f"Unknown attribute '@{attribute.name}'.", span=attribute.span)
        if target not in spec.targets:
            raise AglScopeError(
                f"Attribute '@{attribute.name}' cannot be attached to a {_TARGET_LABEL[target]}.",
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
    return _Recognized(nodes=nodes, texts=texts)


def _program_command_spec(node: FuncDef, recognized: _Recognized) -> ProgramCommandSpec | None:
    """Return the package command *node* registers via its attribute prefix, or ``None``.

    Only a program carrying ``@command`` registers anything, so the prose
    attributes — which describe a registration rather than a program — are
    rejected without it rather than silently dropped. The path is held to
    the rule a package manifest's command paths answer to, since both
    register into the same command tree; whether the path reaches a CLI at
    all is a package fact, so a program outside a package is simply never
    asked for its registration.
    """
    command = recognized.nodes.get(COMMAND_ATTRIBUTE)
    if command is None:
        for name in COMMAND_PROSE_ATTRIBUTES:
            attribute = recognized.nodes.get(name)
            if attribute is not None:
                raise AglScopeError(
                    f"Attribute '@{name}' describes a command registration, so program "
                    f"{node.name!r} needs a '@{COMMAND_ATTRIBUTE}' attribute beside it.",
                    span=attribute.span,
                )
        return None
    path = recognized.texts[COMMAND_ATTRIBUTE]
    invalid = invalid_program_command_path(path)
    if invalid is not None:
        raise AglScopeError(
            f"Command path {path!r} {invalid}.",
            span=command.span,
        )
    return ProgramCommandSpec(
        path=path,
        description=recognized.text_of(DESCRIPTION_ATTRIBUTE),
        help=recognized.text_of(HELP_ATTRIBUTE),
    )


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


def _zone_attribute(recognized: _Recognized) -> ParamZone | None:
    """Return the zone the recognized ``@arg-*`` attribute selects."""
    for name in recognized.nodes:
        zone = ZONE_ATTRIBUTES.get(name)
        if zone is not None:
            return zone
    return None
