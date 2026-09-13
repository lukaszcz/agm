"""The catalog of built-in declaration attributes.

An AgL declaration may carry an attribute prefix (``@doc("…")``,
``@arg-named``). The AST keeps attributes raw; recognition happens later,
against this catalog: it says which attributes exist, which declaration kinds
each one may sit on, what arguments it takes, whether it may repeat, and which
other attributes it excludes. Everything here is data — the diagnostics for an
unknown, misplaced, malformed, duplicate, or conflicting attribute belong to
the pass that consults the catalog. The typed shapes an attribute's meaning
takes — :class:`ProgramOptionSpec`, the command-line presentation the
``@opt-*`` attributes describe, and :class:`ProgramCommandSpec`, the package
command registration the ``@command`` family describes — live here too, so a
host reads one without reaching into a pass.

It is a top-level leaf sitting on ``zones``, whose ``ParamZone`` the ``@arg-*``
attributes name, and on the pure command catalog, whose command-path rule
``@command`` shares with a package manifest, and on nothing else, so any layer
may name an attribute without pulling a pass in with it.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from agm.agl.zones import ParamZone
from agm.command_catalog import invalid_command_path

__all__ = [
    "BUILTIN_ATTRIBUTES",
    "COMMAND_ATTRIBUTE",
    "COMMAND_PROSE_ATTRIBUTES",
    "DESCRIPTION_ATTRIBUTE",
    "DOC_ATTRIBUTE",
    "EXTERN_NAME_ATTRIBUTE",
    "HELP_ATTRIBUTE",
    "NAME_ADDRESSED_OPTION_ATTRIBUTES",
    "OPTION_ENV_ATTRIBUTE",
    "OPTION_HIDDEN_ATTRIBUTE",
    "OPTION_METAVAR_ATTRIBUTE",
    "OPTION_NAME_ATTRIBUTE",
    "OPTION_NAME_PATTERN",
    "OPTION_SHORT_ATTRIBUTE",
    "OPTION_SHORT_PATTERN",
    "PARAM_ATTRIBUTE",
    "ZONE_ATTRIBUTES",
    "AttributeArguments",
    "AttributeSpec",
    "AttributeTarget",
    "ProgramCommandSpec",
    "ProgramOptionSpec",
    "invalid_program_command_path",
]


class AttributeTarget(enum.Enum):
    """The kind of declaration an attribute is attached to.

    Values are stable strings for debuggability; no code should branch on
    the string values.
    """

    FUNCTION = "function"
    PROGRAM = "program"
    EXTERN = "extern"
    BUILTIN_FUNCTION = "builtin_function"
    RECORD = "record"
    ENUM = "enum"
    ENUM_MEMBER = "enum_member"
    EXCEPTION = "exception"
    TYPE_ALIAS = "type_alias"
    BINDING = "binding"
    PARAM_BINDING = "param_binding"
    BUILTIN_VAR = "builtin_var"
    PARAMETER = "parameter"
    FIELD = "field"
    PROGRAM_PARAMETER = "program_parameter"


class AttributeArguments(enum.Enum):
    """The argument shape a built-in attribute accepts.

    Every built-in attribute takes literal constants only, in one of two
    shapes: nothing at all, or a single text literal.
    """

    NONE = "none"
    ONE_TEXT = "one_text"


@dataclass(frozen=True, slots=True)
class AttributeSpec:
    """What one attribute name admits.

    ``targets`` are the declaration kinds it may sit on, ``arguments`` its
    argument shape, and ``conflicts`` the names it may not appear beside.
    ``pattern`` narrows a text argument further — the spelling a host has to
    be able to form from it — and ``expected`` phrases that shape for the
    diagnostic an argument failing it raises. Both are absent for an
    attribute whose text is arbitrary prose.
    """

    name: str
    targets: frozenset[AttributeTarget]
    arguments: AttributeArguments
    conflicts: tuple[str, ...] = ()
    pattern: re.Pattern[str] | None = None
    expected: str | None = None


#: Every declaration kind: ``@doc`` may sit on all of them.
_EVERY_TARGET: frozenset[AttributeTarget] = frozenset(AttributeTarget)

#: A parameter or field, plus the declarations whose entry lists they form: a
#: zone attribute sets one entry's zone or the whole list's default.
_ZONED_TARGETS: frozenset[AttributeTarget] = frozenset(
    {
        AttributeTarget.PARAMETER,
        AttributeTarget.PROGRAM_PARAMETER,
        AttributeTarget.FIELD,
        AttributeTarget.FUNCTION,
        AttributeTarget.PROGRAM,
        AttributeTarget.EXTERN,
        AttributeTarget.BUILTIN_FUNCTION,
        AttributeTarget.RECORD,
        AttributeTarget.EXCEPTION,
        AttributeTarget.ENUM_MEMBER,
    }
)

#: The zone each ``@arg-*`` attribute selects. This is the one place the zone
#: attributes are named: the catalog derives their mutual conflicts from these
#: keys, and recognition reads the zone an attribute stands for from here.
ZONE_ATTRIBUTES: Mapping[str, ParamZone] = MappingProxyType(
    {
        "arg-pos": ParamZone.POSITIONAL_ONLY,
        "arg-std": ParamZone.STANDARD,
        "arg-named": ParamZone.NAMED_ONLY,
    }
)

#: The attribute naming an extern's Python companion function. Named here so
#: the pass computing an extern's effective companion name, and the diagnostics
#: it raises to name the remedy, reach it through this constant.
EXTERN_NAME_ATTRIBUTE = "extern-name"

#: The attribute carrying human-readable prose for a declaration. Named here
#: because the pass that recognizes attributes files its text in a table of
#: its own, and every host surface showing documentation reads that table.
DOC_ATTRIBUTE = "doc"

#: The attributes registering a ``program def`` as a package command. The path
#: ``@command`` names is the command a reader invokes; ``@description`` and
#: ``@help`` are the prose the registration carries, the namesake fields of a
#: package manifest's command table. The prose attributes describe a
#: registration rather than a program, so neither means anything without
#: ``@command`` beside it.
COMMAND_ATTRIBUTE = "command"
DESCRIPTION_ATTRIBUTE = "description"
HELP_ATTRIBUTE = "help"

#: The command attributes carrying prose: legal only beside ``@command``.
COMMAND_PROSE_ATTRIBUTES: tuple[str, ...] = (DESCRIPTION_ATTRIBUTE, HELP_ATTRIBUTE)

#: The attributes shaping how a host-facing parameter is presented.
OPTION_NAME_ATTRIBUTE = "opt-name"
OPTION_SHORT_ATTRIBUTE = "opt-short"
OPTION_ENV_ATTRIBUTE = "opt-env"
OPTION_METAVAR_ATTRIBUTE = "opt-metavar"
OPTION_HIDDEN_ATTRIBUTE = "opt-hidden"

#: The marker that exposes a static binding as a host-facing parameter.
PARAM_ATTRIBUTE = "param"

#: The option attributes that address a parameter by name: they rename its
#: flag, give it a short spelling, name an environment fallback, or keep that
#: name out of help. A positional-only parameter is never addressed by name,
#: so carrying one of these is an error; ``@opt-metavar`` and ``@doc`` still
#: apply, since they describe the value and its meaning rather than the name
#: it is addressed by.
NAME_ADDRESSED_OPTION_ATTRIBUTES: tuple[str, ...] = (
    OPTION_NAME_ATTRIBUTE,
    OPTION_SHORT_ATTRIBUTE,
    OPTION_ENV_ATTRIBUTE,
    OPTION_HIDDEN_ATTRIBUTE,
)

#: The shape of an ``@opt-name`` argument: a flag word of ASCII letters,
#: digits and hyphens that begins with a letter or digit and follows every
#: hyphen with one, so a host can form both ``--<name>`` and the derived
#: ``--no-<name>`` from it. Anything else — a leading, trailing or doubled
#: hyphen, whitespace, ``=``, other punctuation, the empty text — is rejected.
OPTION_NAME_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9]))*")
_OPTION_NAME_EXPECTED = (
    "takes a flag word — ASCII letters, digits and hyphens, beginning with a letter or digit — not"
)

#: The shape of an ``@opt-short`` argument: exactly one ASCII letter, which a
#: host spells ``-<short>``.
OPTION_SHORT_PATTERN = re.compile(r"[A-Za-z]")
_OPTION_SHORT_EXPECTED = "takes exactly one ASCII letter, not"

_HOST_PARAMETER_TARGETS: frozenset[AttributeTarget] = frozenset(
    {AttributeTarget.PROGRAM_PARAMETER, AttributeTarget.PARAM_BINDING}
)


@dataclass(frozen=True, slots=True)
class ProgramOptionSpec:
    """How one host-facing parameter presents itself to a host.

    ``name`` is the external spelling every host surface uses for the
    parameter — its flag, the derived negative, the config key, completion —
    and defaults to the declared name. ``short`` is a one-letter alternative
    spelling, ``env`` an environment variable read when no value is supplied,
    ``metavar`` the placeholder standing for the value in usage text,
    ``hidden`` whether the parameter's own name entry is kept out of help and
    completion, and ``doc`` its help prose. Every field but ``name`` is absent
    unless an attribute supplies it.
    """

    name: str
    short: str | None = None
    env: str | None = None
    metavar: str | None = None
    hidden: bool = False
    doc: str | None = None


@dataclass(frozen=True, slots=True)
class ProgramCommandSpec:
    """The package command one ``program def`` registers itself as.

    ``path`` is the command a reader invokes, in the space-separated spelling a
    package manifest uses. ``description`` and ``help`` are the registration's
    prose, absent unless their attribute supplies them; they are distinct from
    the program's own ``@doc``, which describes the program wherever it is run.
    """

    path: str
    description: str | None = None
    help: str | None = None


def invalid_program_command_path(path: str) -> str | None:
    """Describe why *path* cannot name a registered command, or ``None``.

    A ``@command`` path extends the same command tree a package manifest
    registers into, so it answers to the same rule; this wrapper is the name
    the passes reach it by, so no pass imports the command catalog itself.
    """

    return invalid_command_path(path)


def _program_spec(name: str) -> AttributeSpec:
    return AttributeSpec(
        name=name,
        targets=frozenset({AttributeTarget.PROGRAM}),
        arguments=AttributeArguments.ONE_TEXT,
    )


def _zone_spec(name: str) -> AttributeSpec:
    return AttributeSpec(
        name=name,
        targets=_ZONED_TARGETS,
        arguments=AttributeArguments.NONE,
        conflicts=tuple(other for other in ZONE_ATTRIBUTES if other != name),
    )


def _option_spec(
    name: str,
    arguments: AttributeArguments,
    *,
    pattern: re.Pattern[str] | None = None,
    expected: str | None = None,
) -> AttributeSpec:
    return AttributeSpec(
        name=name,
        targets=_HOST_PARAMETER_TARGETS,
        arguments=arguments,
        pattern=pattern,
        expected=expected,
    )


_SPECS: tuple[AttributeSpec, ...] = (
    *(_zone_spec(name) for name in ZONE_ATTRIBUTES),
    AttributeSpec(
        name=EXTERN_NAME_ATTRIBUTE,
        targets=frozenset({AttributeTarget.EXTERN}),
        arguments=AttributeArguments.ONE_TEXT,
    ),
    AttributeSpec(
        name=PARAM_ATTRIBUTE,
        targets=frozenset({AttributeTarget.BINDING}),
        arguments=AttributeArguments.NONE,
    ),
    _option_spec(
        OPTION_SHORT_ATTRIBUTE,
        AttributeArguments.ONE_TEXT,
        pattern=OPTION_SHORT_PATTERN,
        expected=_OPTION_SHORT_EXPECTED,
    ),
    _option_spec(
        OPTION_NAME_ATTRIBUTE,
        AttributeArguments.ONE_TEXT,
        pattern=OPTION_NAME_PATTERN,
        expected=_OPTION_NAME_EXPECTED,
    ),
    _option_spec(OPTION_ENV_ATTRIBUTE, AttributeArguments.ONE_TEXT),
    _option_spec(OPTION_METAVAR_ATTRIBUTE, AttributeArguments.ONE_TEXT),
    _option_spec(OPTION_HIDDEN_ATTRIBUTE, AttributeArguments.NONE),
    _program_spec(COMMAND_ATTRIBUTE),
    *(_program_spec(name) for name in COMMAND_PROSE_ATTRIBUTES),
    AttributeSpec(
        name=DOC_ATTRIBUTE,
        targets=_EVERY_TARGET,
        arguments=AttributeArguments.ONE_TEXT,
    ),
)

#: The built-in attributes, keyed by the name written after ``@``.
BUILTIN_ATTRIBUTES: Mapping[str, AttributeSpec] = MappingProxyType(
    {spec.name: spec for spec in _SPECS}
)
