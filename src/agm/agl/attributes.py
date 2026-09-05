"""The catalog of built-in declaration attributes.

An AgL declaration may carry an attribute prefix (``@doc("…")``,
``@arg-named``). The AST keeps attributes raw; recognition happens later,
against this catalog: it says which attributes exist, which declaration kinds
each one may sit on, what arguments it takes, whether it may repeat, and which
other attributes it excludes. Everything here is data — the diagnostics for an
unknown, misplaced, malformed, duplicate, or conflicting attribute belong to
the pass that consults the catalog.

It lives in its own dependency-free top-level leaf, alongside ``zones`` and
``modules.ids``, so any layer may name an attribute without pulling a pass in
with it.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

__all__ = [
    "BUILTIN_ATTRIBUTES",
    "AttributeArguments",
    "AttributeSpec",
    "AttributeTarget",
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
    argument shape, ``repeatable`` whether one declaration may carry it more
    than once, and ``conflicts`` the names it may not appear beside.
    """

    name: str
    targets: frozenset[AttributeTarget]
    arguments: AttributeArguments
    repeatable: bool = False
    conflicts: tuple[str, ...] = ()


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

#: The three zone attributes; each excludes the other two.
_ZONE_ATTRIBUTE_NAMES: tuple[str, ...] = ("arg-pos", "arg-std", "arg-named")

_PROGRAM_PARAMETER_ONLY: frozenset[AttributeTarget] = frozenset({AttributeTarget.PROGRAM_PARAMETER})


def _zone_spec(name: str) -> AttributeSpec:
    return AttributeSpec(
        name=name,
        targets=_ZONED_TARGETS,
        arguments=AttributeArguments.NONE,
        conflicts=tuple(other for other in _ZONE_ATTRIBUTE_NAMES if other != name),
    )


def _option_spec(name: str, arguments: AttributeArguments) -> AttributeSpec:
    return AttributeSpec(
        name=name,
        targets=_PROGRAM_PARAMETER_ONLY,
        arguments=arguments,
    )


_SPECS: tuple[AttributeSpec, ...] = (
    *(_zone_spec(name) for name in _ZONE_ATTRIBUTE_NAMES),
    AttributeSpec(
        name="extern-name",
        targets=frozenset({AttributeTarget.EXTERN}),
        arguments=AttributeArguments.ONE_TEXT,
    ),
    _option_spec("opt-short", AttributeArguments.ONE_TEXT),
    _option_spec("opt-name", AttributeArguments.ONE_TEXT),
    _option_spec("opt-env", AttributeArguments.ONE_TEXT),
    _option_spec("opt-metavar", AttributeArguments.ONE_TEXT),
    _option_spec("opt-hidden", AttributeArguments.NONE),
    AttributeSpec(
        name="doc",
        targets=_EVERY_TARGET,
        arguments=AttributeArguments.ONE_TEXT,
    ),
)

#: The built-in attributes, keyed by the name written after ``@``.
BUILTIN_ATTRIBUTES: Mapping[str, AttributeSpec] = MappingProxyType(
    {spec.name: spec for spec in _SPECS}
)
