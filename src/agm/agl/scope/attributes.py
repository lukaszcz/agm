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
import re
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
    OPTION_NAME_PATTERN,
    OPTION_SHORT_ATTRIBUTE,
    OPTION_SHORT_PATTERN,
    ZONE_ATTRIBUTES,
    AttributeArguments,
    AttributeSpec,
    AttributeTarget,
    ProgramOptionSpec,
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
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.visitor import walk
from agm.agl.zones import ParamZone

__all__ = ["AttributeFacts", "recognize_attributes"]


@dataclass(frozen=True, slots=True)
class AttributeFacts:
    """The typed facts one module's recognized attributes carry.

    ``param_zones``
        The zone of every function, lambda, record, exception, and enum-member
        entry, keyed by its ``Param.node_id``.
    ``extern_names``
        The Python companion name of every ``extern def``, keyed by its
        ``FuncDef.node_id``.
    ``program_options``
        The command-line presentation of every ``program def`` parameter,
        keyed by its ``Param.node_id``.
    ``docs``
        The ``@doc`` text of every declaration carrying one — parameters and
        fields included — keyed by that declaration's node id.
    """

    param_zones: dict[int, ParamZone]
    extern_names: dict[int, str]
    program_options: dict[int, ProgramOptionSpec]
    docs: dict[int, str]


@dataclass(frozen=True, slots=True)
class _TextArgument:
    """One validated single-text attribute argument, with its own span."""

    attribute: Attribute
    text: str


@dataclass(frozen=True, slots=True)
class _Recognized:
    """One declaration's validated attribute prefix, ready for a fact builder.

    ``spans`` locates every attribute the declaration carries by name, so a
    builder can report an attribute that takes no argument; ``texts`` holds
    the already-validated argument of every attribute whose schema takes one.
    """

    spans: dict[str, SourceSpan]
    texts: dict[str, _TextArgument]

    def text_of(self, name: str) -> str | None:
        """Return the text argument attribute *name* carries, if it is present."""
        argument = self.texts.get(name)
        return None if argument is None else argument.text


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
                self._extern_name(node, recognized.texts.get(EXTERN_NAME_ATTRIBUTE))
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
            self._check(node.attributes, AttributeTarget.RECORD, node.node_id)
            self._entries(node.fields, node.attributes, target=AttributeTarget.FIELD)
        elif isinstance(node, ExceptionDef):
            self._check(node.attributes, AttributeTarget.EXCEPTION, node.node_id)
            self._entries(node.fields, node.attributes, target=AttributeTarget.FIELD)
        elif isinstance(node, VariantDef):
            self._check(node.attributes, AttributeTarget.ENUM_MEMBER, node.node_id)
            self._entries(node.fields, node.attributes, target=AttributeTarget.FIELD)
        elif isinstance(node, EnumDef):
            self._check(node.attributes, AttributeTarget.ENUM, node.node_id)
        elif isinstance(node, TypeAlias):
            self._check(node.attributes, AttributeTarget.TYPE_ALIAS, node.node_id)
        elif isinstance(node, (LetDecl, VarDecl, BuiltinVarDecl)):
            self._check(node.attributes, AttributeTarget.BINDING, node.node_id)

    # ------------------------------------------------------------------
    # Catalog validation
    # ------------------------------------------------------------------

    def _check(
        self, attributes: tuple[Attribute, ...], target: AttributeTarget, node_id: int
    ) -> _Recognized:
        """Validate the whole attribute prefix of the declaration *node_id*.

        Returns every attribute's span and the text argument of those whose
        schema takes one, keyed by attribute name, so a fact builder reads an
        already-validated argument instead of re-inspecting the raw node.
        Documentation text, admitted on every declaration kind, is filed here
        rather than in a builder of its own.
        """
        seen: set[str] = set()
        spans: dict[str, SourceSpan] = {}
        texts: dict[str, _TextArgument] = {}
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
            text = _check_arguments(attribute, spec)
            if text is not None:
                texts[attribute.name] = _TextArgument(attribute=attribute, text=text)
            spans[attribute.name] = attribute.span
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
        recognized = _Recognized(spans=spans, texts=texts)
        documentation = recognized.text_of(DOC_ATTRIBUTE)
        if documentation is not None:
            self.docs[node_id] = documentation
        return recognized

    # ------------------------------------------------------------------
    # Fact builder: extern companion names
    # ------------------------------------------------------------------

    def _extern_name(self, node: FuncDef, supplied: _TextArgument | None) -> None:
        """Record the Python companion name one ``extern def`` resolves to.

        *supplied* is the recognized ``@extern-name`` argument, if the
        declaration carries one; without it the declared member name is the
        companion name verbatim. The companion module has to define a Python
        function spelled exactly this way, so the effective name must be a
        valid identifier that is not a hard Python keyword — soft keywords
        (``match``, ``type``, …) are legal Python ``def`` names and pass — and
        a module's externs, wherever they are declared, must each claim a
        different one of its companion's functions.
        """
        name = node.name if supplied is None else supplied.text
        span = node.span if supplied is None else supplied.attribute.span
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
        highest = _ZONE_ORDER[ParamZone.POSITIONAL_ONLY]
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
                span = recognized.spans.get(name)
                if span is not None:
                    raise AglScopeError(
                        f"Attribute '@{name}' cannot be attached to positional-only parameter "
                        f"{entry.name!r}, which a host addresses by position and never by name.",
                        span=span,
                    )
        short = _shaped_text(
            recognized,
            OPTION_SHORT_ATTRIBUTE,
            OPTION_SHORT_PATTERN,
            "takes exactly one ASCII letter, not",
        )
        external = _shaped_text(
            recognized,
            OPTION_NAME_ATTRIBUTE,
            OPTION_NAME_PATTERN,
            "takes a flag word — ASCII letters, digits and hyphens, beginning with a "
            "letter or digit — not",
        )
        self.program_options[entry.node_id] = ProgramOptionSpec(
            name=entry.name if external is None else external,
            short=short,
            env=recognized.text_of(OPTION_ENV_ATTRIBUTE),
            metavar=recognized.text_of(OPTION_METAVAR_ATTRIBUTE),
            hidden=OPTION_HIDDEN_ATTRIBUTE in recognized.spans,
            doc=recognized.text_of(DOC_ATTRIBUTE),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _shaped_text(
    recognized: _Recognized, name: str, pattern: re.Pattern[str], expected: str
) -> str | None:
    """Return the argument of attribute *name*, rejecting one *pattern* refuses.

    ``None`` when the declaration carries no such attribute. *expected*
    describes the shape the attribute admits; the diagnostic anchors on the
    attribute itself, which is what the writer has to change.
    """
    argument = recognized.texts.get(name)
    if argument is None:
        return None
    if pattern.fullmatch(argument.text) is None:
        raise AglScopeError(
            f"Attribute '@{name}' {expected} {argument.text!r}.", span=argument.attribute.span
        )
    return argument.text


def _check_arguments(attribute: Attribute, spec: AttributeSpec) -> str | None:
    """Reject arguments a built-in attribute's literal schema does not admit.

    Returns the text an attribute taking one argument carries, and ``None``
    for an attribute whose schema takes none.
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
    return argument.value


def _zone_attribute(attributes: tuple[Attribute, ...]) -> ParamZone | None:
    """Return the zone the ``@arg-*`` attribute among *attributes* selects."""
    for attribute in attributes:
        zone = ZONE_ATTRIBUTES.get(attribute.name)
        if zone is not None:
            return zone
    return None
