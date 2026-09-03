"""Type-directed CLI projection shared by engine keys and program parameters.

:func:`project_option` maps one name and AgL type onto its CLI surface:
``bool`` becomes a bare ``--x``/``--no-x`` flag pair (no value); ``Option[T]``
becomes a value-taking ``--x VALUE``/``--no-x`` pair, where ``VALUE`` decodes
as ``T`` and is wrapped ``Some``, and ``--no-x`` supplies ``None``; ``text``
takes ``VALUE`` verbatim; every other type takes one strict-JSON ``VALUE``.
Whether a type is the standard ``Option`` is decided by nominal provenance
(``agm.agl.semantics.types.is_standard_option_enum``), not by name, so a
user-declared ``enum Option[T]`` of its own projects as an ordinary JSON-form
value, never as the ``Option`` shape. This is the single derivation shared by
an engine key's CLI flags (:func:`engine_key_flags`, folded into
:data:`RESERVED_FLAGS`) and a program parameter's own flags
(:class:`ProgramOptionMap`), so the reserved set and the parameter flags can
never disagree.

:class:`ProgramOptionMap` builds a whole program's CLI surface from its
parameter signature: positional slots (``POSITIONAL_ONLY``/``STANDARD``
parameters, in declaration order), options (``STANDARD``/``NAMED_ONLY``
parameters, each projected), a reservation check against the host's flag
inventory and against every other parameter's own projected flags, token
parsing, usage/help rendering, and shell-completion items. It is pure and
host-agnostic — it neither knows about ``agm exec`` nor performs any decoding
itself; :meth:`ProgramOptionMap.parse_tokens` produces the raw
:class:`~agm.agl.runtime.arguments.ProgramArguments` that
``runtime.arguments.decode_param_value`` later decodes.

``RESERVED_FLAGS`` is the host's own flag inventory: the flags ``agm exec``
declares itself (``--help``, ``--program``, ``--agent``, …) union every
engine-key flag. A program parameter can never be projected onto one of
these — :func:`build_program_option_map` reports the collision instead.

A leading ``--`` end-of-options marker: this module treats a bare ``--``
token as ending option parsing, ordinary CLI convention for a pure token
parser. The CLI framework invoking this module may itself already consume a
leading ``--`` before tokens reach here (``agm exec``'s Click command sets
``allow_extra_args``/``ignore_unknown_options``, which strips the first
``--``), in which case this module's own ``--`` handling is reached only by
a second, doubled separator.

Nested ``Option`` handling
---------------------------

Only the outermost type gets special CLI treatment. A ``text``/JSON-form
flag's ``VALUE`` token passes through verbatim, exactly like a positional
token: ``decode_param_value`` already parses a raw string against the
parameter's own decoder (JSON-form included), so parsing it again here would
duplicate that step and lose its diagnostic, anchored at the parameter's
declaration, on a malformed value.

An ``Option[T]`` flag is the one exception. Its raw value is a
``{"$case": "Some"/"None", "value": ...}`` envelope, not a bare string —
``decode_param_value``'s enum-decode walk needs the ``"value"`` payload to
already be a native Python object, not JSON text standing for one — so the
``Some`` payload is built here: verbatim when ``T`` is ``text``, otherwise
eagerly strict-JSON-parsed. This rule is applied once, uniformly, with no
further special case, so it also defines nested behavior: ``Option[bool]``'s
``VALUE`` is a JSON boolean literal (``--x true`` / ``--x false``; ``--no-x``
for ``None``), and ``Option[Option[T]]``'s ``VALUE`` is a JSON literal of the
*whole* inner ``Option`` shape (``--x '{"$case": "Some", "value": ...}'`` /
``--x '{"$case": "None"}'``; ``--no-x`` for the outer ``None``). Unlike a
top-level ``text``/JSON-form flag, a malformed ``Option`` payload is
therefore a plain usage error raised by :meth:`ProgramOptionMap.parse_tokens`
itself, not a decode diagnostic.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agm.agl.ir.zones import ParamZone
from agm.agl.runtime.arguments import ProgramArguments
from agm.agl.runtime.convert import StrictJsonParseError, parse_json_strict
from agm.agl.semantics.engine_keys import ENGINE_KEY_TYPES
from agm.agl.semantics.types import BoolType, TextType, is_standard_option_enum

if TYPE_CHECKING:
    from agm.agl.runtime.types import ProgramDeclInfo, ProgramParamInfo
    from agm.agl.semantics.types import Type as AglType

__all__ = [
    "DuplicateOptionFlagError",
    "ProgramOptionError",
    "ProgramOptionMap",
    "ProjectedOption",
    "RESERVED_FLAGS",
    "ReservedFlagError",
    "ValueForm",
    "build_program_option_map",
    "engine_key_flags",
    "option_none_raw",
    "option_some_raw",
    "project_option",
    "render_program_arguments_help",
]

# Zones whose parameter fills a positional CLI slot.
_POSITIONAL_ZONES = (ParamZone.POSITIONAL_ONLY, ParamZone.STANDARD)

# Zones whose parameter is addressable by a ``--name`` flag.
_NAME_ADDRESSABLE_ZONES = (ParamZone.STANDARD, ParamZone.NAMED_ONLY)


class ValueForm(enum.Enum):
    """How a projected option's ``VALUE`` becomes a raw decode input."""

    #: ``text``: the token, verbatim.
    TEXT = "text"
    #: ``Option[T]``: a ``{"$case": ...}`` envelope wrapping ``T``'s own value.
    OPTION = "option"
    #: ``bool``: a native Python bool, no ``VALUE`` token.
    BOOL = "bool"
    #: Every other type: one strict-JSON token.
    JSON = "json"


@dataclass(frozen=True, slots=True)
class ProjectedOption:
    """One type's CLI surface, as :func:`project_option` derives it.

    ``option_inner`` is ``T`` when ``value_form`` is :attr:`ValueForm.OPTION`
    and ``None`` for every other form — computed once here rather than
    re-derived from the parameter's type on every token, so the raw-value
    builder never has to represent the impossible combination.
    """

    flags: tuple[str, ...]
    negative_flags: tuple[str, ...]
    takes_value: bool
    value_form: ValueForm
    option_inner: "AglType | None" = None


def _option_inner(type_: "AglType") -> "AglType | None":
    """Return ``T`` when *type_* is the standard ``Option[T]``, else ``None``."""
    if is_standard_option_enum(type_) and type_.type_args:
        return type_.type_args[0]
    return None


def project_option(name: str, type_: "AglType") -> ProjectedOption:
    """Project one *name*/*type_* pair (a program parameter or an engine key) onto its CLI flags."""
    flag = f"--{name}"
    negative = f"--no-{name}"
    if isinstance(type_, BoolType):
        return ProjectedOption(
            flags=(flag,), negative_flags=(negative,), takes_value=False, value_form=ValueForm.BOOL
        )
    inner = _option_inner(type_)
    if inner is not None:
        return ProjectedOption(
            flags=(flag,),
            negative_flags=(negative,),
            takes_value=True,
            value_form=ValueForm.OPTION,
            option_inner=inner,
        )
    if isinstance(type_, TextType):
        return ProjectedOption(
            flags=(flag,), negative_flags=(), takes_value=True, value_form=ValueForm.TEXT
        )
    return ProjectedOption(
        flags=(flag,), negative_flags=(), takes_value=True, value_form=ValueForm.JSON
    )


def engine_key_flags() -> frozenset[str]:
    """Return the CLI flags the engine-key catalog reserves.

    Every engine key runs through :func:`project_option` against its AgL
    type, exactly like a program parameter — the single derivation shared
    with :data:`RESERVED_FLAGS`, so the two can never disagree. Iterates
    ``semantics.engine_keys.ENGINE_KEY_TYPES``, whose keys are exactly the
    engine-key names, so the lookup is total by construction.
    """
    flags: set[str] = set()
    for name, agl_type in ENGINE_KEY_TYPES.items():
        projected = project_option(name, agl_type)
        flags.update(projected.flags)
        flags.update(projected.negative_flags)
    return frozenset(flags)


# Flags ``agm exec`` declares that the engine-key catalog does not spell.
# Every flag the command declares must appear either here or in
# ``engine_key_flags()``: an unreserved flag is worse than a rejected
# program parameter, because Click binds the token to the built-in option and
# the parameter advertised under that flag silently keeps its default. The
# declarations live in ``agm.cli``, a layer above this one, so they are
# mirrored here and cross-checked by the tests.
_BUILTIN_EXEC_FLAGS: frozenset[str] = frozenset(
    {
        "--command",
        "-c",
        "--program",
        "-p",
        "--module-path",
        "-I",
        "--max-call-depth",
        # ``agm exec`` turns off Click's built-in help option so that program
        # ``--param`` tokens pass through to it, and recognises these itself.
        "--help",
        "-h",
        "--dry-run",
        "--no-stdlib",
        # The CLI spelling of the ``default-agent`` engine key.
        "--agent",
    }
)

# Reserved flag strings: declared built-ins UNION engine-key flags (both
# polarities). Collision checking is verbatim — no underscore/hyphen
# normalisation.
RESERVED_FLAGS: frozenset[str] = _BUILTIN_EXEC_FLAGS | engine_key_flags()


@dataclass(frozen=True, slots=True)
class ReservedFlagError:
    """One program parameter's projected flag collides with a reserved host flag."""

    parameter: str
    flag: str


@dataclass(frozen=True, slots=True)
class DuplicateOptionFlagError:
    """Two program parameters project onto the same CLI flag.

    AgL identifiers are kebab-case, so e.g. ``cache: bool`` and
    ``no-cache: bool`` are both plausible parameter names, yet the first's
    negative flag (``--no-cache``) and the second's positive flag spell the
    same string. ``first_parameter``/``second_parameter`` are the colliding
    parameters' names, in declaration order.
    """

    first_parameter: str
    second_parameter: str
    flag: str


type ProgramOptionError = ReservedFlagError | DuplicateOptionFlagError


def _value_from_token(flag: str, inner_type: "AglType", token: str) -> object:
    """Return the raw ``Some`` payload for one ``Option[T]`` flag's ``VALUE`` token.

    Verbatim when ``T`` is ``text``; otherwise strict-JSON-parsed, so a
    nested ``Option``/``bool``/... inner type round-trips through
    ``decode_param_value``'s enum-decode walk, which needs a native Python
    object already, not another string to parse. *flag* names the option in
    a malformed-JSON error, so the message identifies which flag failed.
    """
    if isinstance(inner_type, TextType):
        return token
    try:
        return parse_json_strict(token)
    except StrictJsonParseError as exc:
        raise ValueError(
            f"Option {flag!r} value {token!r} is not valid JSON: {exc.message}"
        ) from exc


def option_some_raw(value: object) -> object:
    """Return the ``Some`` envelope ``decode_param_value`` expects for an ``Option[T]`` raw value.

    The single place this shape (``{"$case": "Some", "value": ...}``) is
    built — shared by :func:`_positive_raw`, for a CLI flag's already-decoded
    ``VALUE`` token, and by the config-table path (``commands.exec_program``),
    for an already-native TOML/JSON value read from a program's qualified
    table.
    """
    return {"$case": "Some", "value": value}


def option_none_raw() -> object:
    """Return the ``None`` envelope ``decode_param_value`` expects for an ``Option[T]`` value."""
    return {"$case": "None"}


def _positive_raw(projected: ProjectedOption, flag: str, token: str) -> object:
    """Build the raw value for one value-taking flag's ``VALUE`` token.

    Only ``TEXT``/``JSON``/``OPTION`` forms reach here — the caller
    (:meth:`ProgramOptionMap.parse_tokens`) never calls this for a ``BOOL``
    form, whose ``takes_value`` is ``False``.
    """
    if projected.option_inner is not None:
        return option_some_raw(_value_from_token(flag, projected.option_inner, token))
    return token


def _negative_raw(projected: ProjectedOption) -> object:
    """Build the raw value for one negated (``--no-x``) flag."""
    if projected.value_form is ValueForm.BOOL:
        return False
    return option_none_raw()


@dataclass(frozen=True, slots=True)
class ProgramOptionMap:
    """One program's whole CLI surface, built by :func:`build_program_option_map`.

    ``positional`` lists positional-capable parameters (``POSITIONAL_ONLY``,
    ``STANDARD``) in declaration order. ``options`` pairs every
    name-addressable parameter (``STANDARD``, ``NAMED_ONLY``) with its
    projected CLI option; a ``STANDARD`` parameter appears in both, since it
    accepts either a positional token or ``--name``.
    """

    positional: tuple["ProgramParamInfo", ...]
    options: tuple[tuple["ProgramParamInfo", ProjectedOption], ...]

    def parse_tokens(self, tokens: Sequence[str]) -> ProgramArguments:
        """Parse *tokens* into raw positional/named host values.

        Every non-flag token is collected positionally, unconditionally: a
        token collected positionally does not need to know which
        positional-capable parameter it will fill, or even whether one
        remains — the shared zone binder (``semantics.arguments.
        bind_arguments``, run later by ``runtime.arguments.
        bind_program_arguments``) pairs positional values with parameters
        left to right and diagnoses an excess positional argument itself,
        with a message that depends on whether the program declares any
        named-only parameters. Deferring to it, rather than capping here,
        keeps that one diagnosis in one place.

        A token that does not start with ``--`` is always positional,
        including one that starts with a single ``-`` — a negative-number
        JSON value such as ``-5``, or any other single-dash spelling; this
        parser recognizes no short-option forms.

        :raises ValueError: for an unknown flag, a value-taking flag missing
            its value, a value given to a flag that takes none, a flag
            supplied more than once, or a malformed ``Option`` ``VALUE``
            (see the module docstring).
        """
        positive_index = {
            flag: (param, projected)
            for param, projected in self.options
            for flag in projected.flags
        }
        negative_index = {
            flag: (param, projected)
            for param, projected in self.options
            for flag in projected.negative_flags
        }

        positional: list[object] = []
        named: dict[str, object] = {}
        options_ended = False
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if not options_ended and token == "--":
                options_ended = True
                index += 1
                continue
            if options_ended or not token.startswith("--"):
                positional.append(token)
                index += 1
                continue

            flag, has_equals, inline_value = token.partition("=")
            if flag in negative_index:
                param, projected = negative_index[flag]
                if has_equals:
                    raise ValueError(f"Option {flag!r} does not take a value")
                self._store(named, param.name, _negative_raw(projected))
                index += 1
                continue
            if flag not in positive_index:
                raise ValueError(f"Unknown option: {flag!r}")
            param, projected = positive_index[flag]
            if not projected.takes_value:
                if has_equals:
                    raise ValueError(f"Option {flag!r} does not take a value")
                self._store(named, param.name, True)
                index += 1
                continue
            if has_equals:
                value_token = inline_value
                index += 1
            else:
                if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                    raise ValueError(f"Option {token!r} requires a value")
                value_token = tokens[index + 1]
                index += 2
            self._store(named, param.name, _positive_raw(projected, flag, value_token))

        return ProgramArguments(positional=tuple(positional), named=named)

    @staticmethod
    def _store(named: dict[str, object], name: str, value: object) -> None:
        """Record *value* for *name*, rejecting a flag supplied more than once."""
        if name in named:
            raise ValueError(f"Option {f'--{name}'!r} specified more than once")
        named[name] = value

    def usage_line(self, program_name: str) -> str:
        """Render one usage line: the program name, its positional slots, then options."""
        parts = [program_name]
        for param in self.positional:
            parts.append(f"[{param.name}]" if param.has_default else f"<{param.name}>")
        if self.options:
            parts.append("[OPTIONS]")
        return " ".join(parts)

    def render_help_section(self) -> str:
        """Render an ``Options:`` section, one line per name-addressable parameter.

        Each line shows the parameter's flag(s) — both polarities joined
        together when it has a negative form (``bool``/``Option``) — its
        ``VALUE`` placeholder when it takes one (every form but ``bool``),
        its type, and whether it is required or has a default; no
        per-parameter help text is rendered.
        """
        if not self.options:
            return ""
        lines = ["Options:"]
        for param, projected in self.options:
            flag_str = "/".join((*projected.flags, *projected.negative_flags))
            if projected.takes_value:
                flag_str = f"{flag_str} VALUE"
            status = "(optional, has default)" if param.has_default else "(required)"
            lines.append(f"  {flag_str}  {param.type!r}  {status}")
        return "\n".join(lines) + "\n"

    def completion_items(self) -> tuple[str, ...]:
        """Return every completable CLI token: each option's flags and negative forms."""
        items: list[str] = []
        for _param, projected in self.options:
            items.extend(projected.flags)
            items.extend(projected.negative_flags)
        return tuple(items)


def build_program_option_map(
    signature: "tuple[ProgramParamInfo, ...]",
) -> "ProgramOptionMap | ProgramOptionError":
    """Build *signature*'s option map, checking flag reservation first.

    Reservation is checked against :data:`RESERVED_FLAGS` — the host's full
    inventory, engine-key flags (:func:`engine_key_flags`) already included —
    so a program parameter can never shadow either an engine key or a
    built-in host option (``--help``, ``--program``, ``--agent``, …); and
    against every other parameter's own projected flags, so one parameter's
    ``--no-<name>`` negation can never silently steal a different parameter
    literally named ``no-<name>``. Positional-only parameters never enter the
    flag namespace and so are never checked. Returns a
    :class:`ReservedFlagError` or :class:`DuplicateOptionFlagError` for the
    caller to render on the first collision found, in declaration order.
    """
    positional = tuple(p for p in signature if p.kind in _POSITIONAL_ZONES)
    options = tuple(
        (p, project_option(p.name, p.type)) for p in signature if p.kind in _NAME_ADDRESSABLE_ZONES
    )
    seen: dict[str, str] = {}
    for param, projected in options:
        for flag in (*projected.flags, *projected.negative_flags):
            if flag in RESERVED_FLAGS:
                return ReservedFlagError(parameter=param.name, flag=flag)
            if flag in seen:
                return DuplicateOptionFlagError(
                    first_parameter=seen[flag], second_parameter=param.name, flag=flag
                )
            seen[flag] = param.name
    return ProgramOptionMap(positional=positional, options=options)


def render_program_arguments_help(
    entry_programs: "tuple[ProgramDeclInfo, ...]", *, selected: "ProgramDeclInfo | None"
) -> str:
    """Render the ``Program arguments:`` section of ``agm exec --help``.

    *entry_programs* and *selected* come from
    ``cli_support.program_discovery.select_entry_program``, the one place a
    requested program name is matched against the entry module's own
    declarations, so this renderer only formats a selection already made —
    it never re-derives one. When *selected* names one program, its usage
    line and full ``Options:`` section are rendered; otherwise (several entry
    programs and none selected, including a requested name matching none of
    them) every entry program's usage line alone is listed, so the reader can
    pick one with ``-p``.
    """
    if not entry_programs:
        return ""
    candidates = (selected,) if selected is not None else entry_programs
    lines: list[str] = ["Program arguments:"]
    for candidate in candidates:
        option_map_result = build_program_option_map(candidate.parameters)
        if not isinstance(option_map_result, ProgramOptionMap):
            continue
        lines.append(f"  Usage: {option_map_result.usage_line(candidate.declaration_path)}")
        if selected is not None:
            help_section = option_map_result.render_help_section()
            if help_section:
                lines.extend(f"  {line}" for line in help_section.splitlines())
    if len(lines) == 1:
        return ""
    return "\n".join(lines) + "\n"
