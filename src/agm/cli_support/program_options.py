"""Type-directed CLI projection shared by engine keys and program parameters.

:func:`project_option` maps one name and AgL type onto its CLI surface:
``bool`` becomes a bare ``--x``/``--no-x`` flag pair (no value); ``Option[T]``
becomes a value-taking ``--x VALUE``/``--no-x`` pair, where ``VALUE`` decodes
as ``T`` and is wrapped ``Some``, and ``--no-x`` supplies ``None``; ``text``
takes ``VALUE`` verbatim; standard ``Agent`` uses host Agent syntax; every
other type takes one strict-JSON ``VALUE``.
Whether a type is the standard ``Option`` is decided by nominal provenance
(``agm.agl.semantics.types.is_standard_option_enum``), not by name, so a
user-declared ``enum Option[T]`` of its own projects as an ordinary JSON-form
value, never as the ``Option`` shape. This is the single derivation shared by
an engine key's CLI flags (:func:`engine_key_flags`, folded into
:data:`EXEC_RESERVED_FLAGS`) and a program parameter's own flags
(:class:`ProgramCommand`), so the reserved set and the parameter flags can
never disagree.

:func:`build_program_command` builds a whole program's CLI surface from its
declaration: one ``click.Command`` carrying a catch-all positional argument
and one option per name-addressable parameter, plus a reservation check
against the invoking surface's flag inventory and against every other
parameter's own spellings. Click owns every token convention — short options, bundling,
attached values, ``--x=V``, the ``--`` end-of-options marker, and
``-h``/``--help`` — while zone pairing, missing required arguments and excess
positionals stay with the shared binder in ``agm.agl.runtime.arguments``, the
one place they are diagnosed. A parameter supplied more than once is
diagnosed here instead: the binder takes a name-keyed mapping and so is
entitled to assume its caller already rejected repetition.

The command also owns its own help. :func:`program_help_requested` asks Click
whether a token stream requests it — so a ``-h`` bundled into a short group is
a help request while one standing where a value-taking option expects its
value is that value — and :func:`render_program_help` renders it: the usage
line of the invocation the reader typed, the program's ``@doc`` as the
description (or whatever description its host prefers), one entry per visible
option with its own ``@doc`` and metavar, and no entry at all for a hidden
one. :meth:`ProgramCommand.parse` raises :class:`ProgramHelpRequested` for a
help request its caller has not already recognized, so the tokens are read
once, by Click, however the invocation reaches this module.

A parameter's *external* name (its declared name, or whatever ``@opt-name``
renames it to) governs every host surface: the flag, its derived negative,
the qualified config key, completion and the undeclared-config-key warning.
The external name stops here: :meth:`ProgramCommand.parse` maps it back to
the declared name, so the shared binder only ever sees declared names.

Each surface a program runs under reserves its own flag inventory.
``EXEC_RESERVED_FLAGS`` is ``agm exec``'s: the flags it declares itself
(``--help``, ``-p``, ``--program``, …) union every engine-key
flag. ``REGISTERED_RESERVED_FLAGS`` is a package-registered command's: its
``agm exec`` run-time options (``REGISTERED_RUN_FLAGS``) and ``--dry-run``
beside the help flags every program command owns. A program
parameter can never be projected onto its surface's reserved flags —
:func:`build_program_command` reports the collision instead.

The same inventory decides which tail token is ``agm exec``'s FILE argument
and which belong to the program: :func:`split_exec_tail` is the one place
that split is derived, and :func:`retain_end_of_options` is what keeps the
end-of-options marker visible to it across Click's own parse.

Nested ``Option`` handling
---------------------------

Only the outermost type gets special CLI treatment. A non-``Option`` flag's
``VALUE`` token passes through verbatim, exactly like a positional token:
``decode_param_value`` already reads a raw string against the parameter's
own decoder (host Agent syntax, strict JSON, or AgL value syntax alike), so
reading it again here would duplicate that step and lose its diagnostic,
anchored at the parameter's declaration, on a malformed value.

An ``Option[T]`` flag is the one boxing exception. Its raw value is an
:class:`~agm.agl.runtime.arguments.OptionSome` box around the token, built by
:func:`option_some_raw` — ``decode_param_value``'s deferred-decode path reads
a boxed ``Some`` payload against ``T``'s own field schema (host-text dispatch
when it is a string, as native data otherwise) and wraps the result into the
enum's own ``{"$case": "Some", "value": ...}`` JSON shape itself, so no
decoding happens in this module. This rule is applied once, uniformly, with
no further special case, so it also defines nested behavior:
``Option[bool]``'s ``VALUE`` is a JSON boolean literal (``--x true`` /
``--x false``; ``--no-x`` for ``None``), and ``Option[Option[T]]``'s
``VALUE`` is a JSON or value-syntax literal of the *whole* inner ``Option``
shape (``--no-x`` for the outer ``None``). A malformed ``Option`` payload —
like any other malformed value — surfaces as a decode diagnostic anchored at
the parameter's declaration, not as a Click usage error.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import click
from click.core import ParameterSource

from agm.agl.runtime.arguments import OptionSome, ProgramArguments
from agm.agl.runtime.serialize import dumps_exact
from agm.agl.runtime.types import ParamBindingInfo, ProgramParamInfo
from agm.agl.semantics.engine_keys import ENGINE_KEY_TYPES
from agm.agl.semantics.types import (
    BoolType,
    JsonType,
    TextType,
    is_standard_agent_enum,
    is_standard_option_enum,
)
from agm.agl.zones import ParamZone

if TYPE_CHECKING:
    from agm.agl.ir.static_keys import StaticBindingKey
    from agm.agl.runtime.types import ProgramDeclInfo
    from agm.agl.semantics.types import Type as AglType
    from agm.cli_support.param_surface import ParamSurface, ParamSurfaceEntry

__all__ = [
    "DuplicateOptionFlagError",
    "END_OF_OPTIONS",
    "EXEC_RESERVED_FLAGS",
    "ExecTail",
    "ProgramCommand",
    "ProgramHelpRequested",
    "ProgramOptionError",
    "ParsedTail",
    "ProjectedOption",
    "REGISTERED_RESERVED_FLAGS",
    "REGISTERED_RUN_FLAGS",
    "ReservedFlagError",
    "ValueForm",
    "build_program_command",
    "contains_help_flag",
    "engine_key_flags",
    "exec_program_help",
    "exec_program_name",
    "native_raw_value",
    "option_none_raw",
    "option_value_map",
    "option_some_raw",
    "program_command_for",
    "program_help_requested",
    "protect_potential_program_values",
    "protect_host_option_values",
    "project_option",
    "render_program_help",
    "retain_end_of_options",
    "split_exec_tail",
]

# Zones whose parameter fills a positional CLI slot.
_POSITIONAL_ZONES = (ParamZone.POSITIONAL_ONLY, ParamZone.STANDARD)

# Zones whose parameter is addressable by a ``--name`` flag.
_NAME_ADDRESSABLE_ZONES = (ParamZone.STANDARD, ParamZone.NAMED_ONLY)

# The command parameter every positional token lands in. Click keys its parsed
# values by parameter name, so this and the per-option names below are internal
# identifiers only — a parameter's user-visible spelling lives in its flags.
_POSITIONAL_DEST = "_positionals"


# The flags a program command reserves for its own help, in both spellings.
HELP_FLAGS = ("--help", "-h")


def contains_help_flag(tokens: "Sequence[str]") -> bool:
    """Return whether *tokens* spell a help flag at all, in either spelling.

    A cheap pre-filter a host applies before discovering a program, matching
    only the common spelling: a hit is worth building the program's command
    for. A help flag bundled into a short group, as in ``-vh``, is missed
    here and recognized later by the program command itself while parsing.
    """
    return any(token in HELP_FLAGS for token in tokens)


class _HelpSignal(Exception):
    """One program command's help option, read while parsing its tokens.

    Raised by the help callback, which knows only that the flag was read.
    The command being parsed re-raises it as :class:`ProgramHelpRequested`,
    attaching itself, so every instance that escapes this module carries the
    command a handler needs to render.
    """


class ProgramHelpRequested(_HelpSignal):
    """Raised when a token stream asks a program command for its help.

    ``command`` is the command whose help was asked for, and ``program`` the
    selected declaration it was built from — attached by the host that holds
    it, so a caller can render the help its own invocation calls for (an
    ``agm exec`` usage line, or a registered command's) without discovering
    the program a second time.
    """

    def __init__(
        self,
        command: "ProgramCommand",
        program: "ProgramDeclInfo | None" = None,
    ) -> None:
        super().__init__("program help requested")
        self.command = command
        self.program = program


def _help_callback(ctx: click.Context, param: click.Parameter, value: object) -> None:
    """Signal a help request instead of printing, leaving rendering to the caller."""
    del param
    if value and not ctx.resilient_parsing:
        raise _HelpSignal("program help requested")


def _help_option() -> click.Option:
    """Return the ``-h``/``--help`` option every program command carries."""
    return click.Option(
        [*HELP_FLAGS],
        is_flag=True,
        is_eager=True,
        expose_value=False,
        callback=_help_callback,
        help="Show this message and exit.",
    )


def _positional_argument() -> click.Argument:
    """Return the catch-all argument every non-option token lands in."""
    return click.Argument([_POSITIONAL_DEST], nargs=-1)


def _positive_dest(index: int) -> str:
    """Return the command-parameter name holding option *index*'s positive value."""
    return f"_opt{index}"


def _negative_dest(index: int) -> str:
    """Return the command-parameter name holding option *index*'s ``--no-`` flag."""
    return f"_neg{index}"


class ValueForm(enum.Enum):
    """How a projected option's ``VALUE`` becomes a raw decode input."""

    #: ``text``: the token, verbatim.
    TEXT = "text"
    #: ``Option[T]``: a ``{"$case": ...}`` envelope wrapping ``T``'s own value.
    OPTION = "option"
    #: ``bool``: a native Python bool, no ``VALUE`` token.
    BOOL = "bool"
    #: Standard ``Agent``: compact syntax, command text, a constructor call, or
    #: canonical JSON.
    AGENT = "agent"
    #: ``json``: one JSON token.
    JSON = "json"
    #: Every other type: one strict-JSON-or-AgL-value-syntax token.
    VALUE = "value"


@dataclass(frozen=True, slots=True)
class ProjectedOption:
    """One type's CLI surface, as :func:`project_option` derives it.

    ``value_form`` is the one stored discriminant: ``negative_flags`` and
    ``takes_value`` both follow from it, so :attr:`takes_value` is derived
    rather than stored and cannot disagree with the form.

    ``option_inner``/``option_inner_form`` are ``Option[T]``'s own ``T`` and
    ``T``'s bare value form when ``value_form`` is :attr:`ValueForm.OPTION`,
    and ``None`` for every other form — computed once here rather than
    re-derived from the parameter's type on every token, so the raw-value
    builder and the metavar never have to re-classify ``T`` themselves.
    """

    flags: tuple[str, ...]
    negative_flags: tuple[str, ...]
    value_form: ValueForm
    option_inner: "AglType | None" = None
    option_inner_form: ValueForm | None = None

    @property
    def takes_value(self) -> bool:
        """Return whether this option consumes a following ``VALUE`` token."""
        return self.value_form is not ValueForm.BOOL


def _option_inner(type_: "AglType") -> "AglType | None":
    """Return ``T`` when *type_* is the standard ``Option[T]``, else ``None``."""
    if is_standard_option_enum(type_) and type_.type_args:
        return type_.type_args[0]
    return None


def _bare_value_form(type_: "AglType") -> ValueForm:
    """Return the CLI value form for a bare type: never ``bool`` or ``Option``.

    Shared by a parameter's own type, when it is neither, and by
    ``Option[T]``'s inner ``T``, so the type-to-form mapping is written once.
    """
    if isinstance(type_, TextType):
        return ValueForm.TEXT
    if is_standard_agent_enum(type_):
        return ValueForm.AGENT
    if isinstance(type_, JsonType):
        return ValueForm.JSON
    return ValueForm.VALUE


def project_option(name: str, type_: "AglType") -> ProjectedOption:
    """Project one *name*/*type_* pair (a program parameter or an engine key) onto its CLI flags.

    *name* is the option's external spelling: an engine key's own name, or a
    program parameter's ``@opt-name`` (defaulting to its declared name).
    """
    inner = _option_inner(type_)
    inner_form: ValueForm | None = None
    if isinstance(type_, BoolType):
        form = ValueForm.BOOL
    elif inner is not None:
        form = ValueForm.OPTION
        inner_form = _bare_value_form(inner)
    else:
        form = _bare_value_form(type_)
    negated = form in (ValueForm.BOOL, ValueForm.OPTION)
    return ProjectedOption(
        flags=(f"--{name}",),
        negative_flags=(f"--no-{name}",) if negated else (),
        value_form=form,
        option_inner=inner if form is ValueForm.OPTION else None,
        option_inner_form=inner_form,
    )


def engine_key_flags() -> frozenset[str]:
    """Return the CLI flags the engine-key catalog reserves.

    Every engine key runs through :func:`project_option` against its AgL
    type, exactly like a program parameter — the single derivation shared
    with :data:`EXEC_RESERVED_FLAGS`, so the two can never disagree. Iterates
    ``semantics.engine_keys.ENGINE_KEY_TYPES``, whose keys are exactly the
    engine-key names, so the lookup is total by construction.
    """
    flags: set[str] = set()
    for name, agl_type in ENGINE_KEY_TYPES.items():
        projected = project_option(name, agl_type)
        flags.update(projected.flags)
        flags.update(projected.negative_flags)
    return frozenset(flags)


# Flags ``agm exec`` declares that the engine-key catalog does not spell,
# short spellings included: with ``@opt-short`` a program parameter can claim
# a single-dash spelling of its own, so the host's own short flags are
# reserved beside its long ones.
# Every flag the command declares must appear either here or in
# ``engine_key_flags()``: an unreserved flag is worse than a rejected
# program parameter, because Click binds the token to the built-in option and
# the parameter advertised under that flag silently keeps its default. The
# declarations live in ``agm.cli``, a layer above this one, so they are
# mirrored here; the tests derive ``agm exec``'s own option spellings from
# that command and assert every one of them is reserved, so a flag added
# there and forgotten here fails.
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
    }
)

# ``agm exec``'s reserved flag strings: declared built-ins UNION engine-key
# flags (both polarities). Collision checking is verbatim — no
# underscore/hyphen normalisation.
EXEC_RESERVED_FLAGS: frozenset[str] = _BUILTIN_EXEC_FLAGS | engine_key_flags()

# ``agm exec``'s run-time options a package-registered command parses too: every
# engine-key flag and ``--max-call-depth``. ``agm.cli_dispatch`` takes the options
# themselves from ``agm exec``'s command by these spellings.
REGISTERED_RUN_FLAGS: frozenset[str] = engine_key_flags() | {"--max-call-depth"}

# A package-registered command's reserved flag strings: its run-time options,
# ``--dry-run`` and help. Its program may claim ``agm exec``'s other spellings
# (``-p``, ``--module-path``, …), which select sources a registration fixes.
REGISTERED_RESERVED_FLAGS: frozenset[str] = (
    frozenset({*HELP_FLAGS, "--dry-run"}) | REGISTERED_RUN_FLAGS
)

# The end-of-options marker. ``agm exec`` consumes one only when it is what
# names the FILE; any other marker belongs to the program and reaches it.
END_OF_OPTIONS = "--"


@dataclass(frozen=True, slots=True)
class ExecTail:
    """``agm exec``'s source selector and the tokens the program itself reads.

    ``file`` is the FILE argument — a path or an installed reference — or
    ``None`` when the invocation names none (``-c/--command``, or nothing at
    all). ``tokens`` is every remaining tail token in written order, for the
    selected program's own command to read.
    """

    file: str | None
    tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParsedTail:
    """Raw program arguments and module-parameter values from one CLI tail."""

    arguments: ProgramArguments
    params: dict["StaticBindingKey", object]

    @property
    def positional(self) -> tuple[object, ...]:
        """Return the parsed program positional values."""
        return self.arguments.positional

    @property
    def named(self) -> Mapping[str, object]:
        """Return the parsed program named values."""
        return self.arguments.named


def _is_option_token(token: str) -> bool:
    """Return whether *token* is written as an option rather than as a value.

    A lone ``-`` is a value, exactly as Click reads it.
    """
    return len(token) > 1 and token.startswith("-")


def _is_host_option(token: str) -> bool:
    """Return whether *token* spells one of ``agm exec``'s own options.

    The host's inventory is :data:`_BUILTIN_EXEC_FLAGS`, and a long option's
    ``--x=V`` form spells the same option as ``--x``.
    """
    return token.partition("=")[0] in _BUILTIN_EXEC_FLAGS


def _file_index(tokens: "Sequence[str]") -> int | None:
    """Return the position of the FILE token in *tokens*, or ``None`` for no FILE.

    Only the host's own options may precede the FILE, and Click has already
    consumed the ones it declares, so what is left before the FILE is a help
    flag (which ``agm exec`` declares but does not let Click intercept) or a
    program option written out of position. A program option's arity is
    unknown until the program is known — which is what the FILE selects — so
    a value token directly after one is read as that option's value rather
    than as the FILE, unless the option already carries its value inline.
    """
    claims_next = False
    for index, token in enumerate(tokens):
        if _is_option_token(token):
            claims_next = not _is_host_option(token) and "=" not in token
            continue
        if claims_next:
            claims_next = False
            continue
        return index
    return None


def _file_index_from_program(
    tokens: "Sequence[str]", program_command_for_file: "Callable[[str], ProgramCommand | None]"
) -> int | None:
    """Find a FILE candidate whose selected program supplies its option arity.

    A pre-FILE program flag cannot be classified from spelling alone: it may
    be a flag, take a separate value, or take an attached short value. The
    source it selects provides that information. Candidates are considered
    from right to left because the FILE follows the program tokens in this
    otherwise ambiguous form. A resolver returning ``None`` means the token
    does not select one usable program, so it cannot settle the ambiguity.
    """
    for index in range(len(tokens) - 1, -1, -1):
        token = tokens[index]
        if not _is_option_token(token) and program_command_for_file(token) is not None:
            return index
    return None


type OptionValueMap = Mapping[str, bool] | Collection[str]


def _option_value_map(options: OptionValueMap) -> Mapping[str, bool]:
    """Normalize an option inventory to flag → takes-separate-value metadata."""
    if isinstance(options, Mapping):
        return options
    return {flag: False for flag in options}


def _option_token_ownership(token: str, options: OptionValueMap) -> tuple[bool, bool]:
    """Return whether *token* is owned and claims the next token by *options*.

    The short-option walk follows Click's attached-value convention: a
    value-taking ``-p`` owns both ``-p VALUE`` and ``-pVALUE``. The second
    result is true only for the separate form.
    """
    option_values = _option_value_map(options)
    if token.startswith("--"):
        flag, separator, _value = token.partition("=")
        takes_value = option_values.get(flag)
        return (takes_value is not None, bool(takes_value and not separator))
    if not token.startswith("-") or token == "-":
        return False, False
    recognized = False
    short_group = token[1:]
    for offset, short in enumerate(short_group):
        takes_value = option_values.get(f"-{short}")
        if takes_value is None:
            continue
        recognized = True
        if takes_value:
            return True, offset + 1 == len(short_group)
    return recognized, False


def option_value_map(params: Sequence[click.Option]) -> dict[str, bool]:
    """Return Click option spellings and whether each consumes a value."""
    return {
        flag: not param.is_flag for param in params for flag in (*param.opts, *param.secondary_opts)
    }


def retain_end_of_options(
    args: "Sequence[str]", host_options: OptionValueMap = frozenset()
) -> list[str]:
    """Return *args* with the host's own end-of-options marker doubled.

    Click's parser removes the first bare ``--`` that is not already the
    value of a host option, so the tail it hands over no longer shows where
    host option scanning ended. Doubling that marker keeps both readings: Click removes
    the copy it would have removed anyway — stopping option parsing at
    exactly the same token — and the survivor is the marker the reader wrote.
    For ``agm exec`` that survivor marks the boundary for
    :func:`split_exec_tail`, which consumes it only when it is what named the
    FILE and otherwise forwards it; a registered package command names no
    FILE, so its survivor always reaches the program. Either way the
    program's own parser reads it.
    """
    tokens = list(args)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == END_OF_OPTIONS:
            return [*tokens[:index], END_OF_OPTIONS, *tokens[index:]]
        _owned, claims_next = _option_token_ownership(token, host_options)
        index += 2 if claims_next and index + 1 < len(tokens) else 1
    return tokens


def split_exec_tail(
    tail: "Sequence[str]",
    *,
    program_command_for_file: "Callable[[str], ProgramCommand | None] | None" = None,
) -> ExecTail:
    """Derive ``agm exec``'s FILE argument and the program's own tokens from *tail*.

    The single place a tail token is classified. *tail* is what Click leaves
    after parsing the host's own options: the FILE, the program's tokens, the
    help flags, and the end-of-options marker :func:`retain_end_of_options`
    kept. The marker ends host option scanning: when the tokens before it name
    no FILE, the token after it is the FILE however it is spelled — this is
    how a file named like an option is named — and the host consumes that
    marker. A marker that named no FILE was written for the program instead,
    and is forwarded in its own position for the program's parser to apply.
    Everything else keeps its written order too, so the program's command
    reads exactly the tokens the reader wrote for it. A tail whose program
    flags precede the FILE cannot be split by spelling alone — a program
    flag's arity is not known until the program is, and the FILE is what
    selects it — so when *program_command_for_file* is given, the rightmost
    pre-marker token naming one usable program settles it. That scan is
    inference where a marker is an explicit statement, so a marker naming a
    usable program wins over it. Without a resolver (inline ``-c`` source,
    which names no FILE) the inexpensive spelling-only scan stands.
    """
    tokens = list(tail)
    marker = tokens.index(END_OF_OPTIONS) if END_OF_OPTIONS in tokens else None
    before_marker = tokens if marker is None else tokens[:marker]
    index = _file_index(before_marker)
    needs_program_arity = index is None or any(
        _is_option_token(token) and not _is_host_option(token) for token in before_marker[:index]
    )
    if needs_program_arity and program_command_for_file is not None:
        # Preserve the inexpensive spelling-only answer when it actually
        # names a usable program. Otherwise its unknown arity may have put
        # the FILE one token too early or too late, so consult the marker
        # first — the reader wrote it to say which token is the FILE, where
        # the pre-marker scan only infers one — and let the target's program
        # signature settle it only when the marker names no usable program.
        if index is None or program_command_for_file(before_marker[index]) is None:
            marker_names_file = (
                marker is not None
                and marker + 1 < len(tokens)
                and program_command_for_file(tokens[marker + 1]) is not None
            )
            index = (
                None
                if marker_names_file
                else _file_index_from_program(before_marker, program_command_for_file)
            )
    consumed: set[int] = set()
    if index is None and marker is not None and marker + 1 < len(tokens):
        # The marker is what named the FILE, so the host consumes it. A marker
        # that named no FILE was written for the program instead, and stays in
        # its own position for the program's own parser to read.
        index = marker + 1
        consumed.add(marker)
    if index is not None:
        consumed.add(index)
    return ExecTail(
        file=None if index is None else tokens[index],
        tokens=tuple(token for position, token in enumerate(tokens) if position not in consumed),
    )


def protect_host_option_values(
    tokens: Sequence[str], program_command: ProgramCommand | None, host_options: OptionValueMap
) -> tuple[list[str], dict[str, str]]:
    """Hide host-looking values until an outer Click command has parsed.

    Click otherwise consumes a known host flag even when a selected program's
    value-taking option owns that following token. Replacing just those
    tokens with unique non-options preserves ordinary parsing, after which a
    host restores the original program tail.
    """
    if program_command is None:
        return list(tokens), {}
    protected = program_command.value_token_indexes(tokens, host_options=host_options)
    replacements: dict[str, str] = {}
    parsed = list(tokens)
    unavailable = set(tokens)
    for index in protected:
        value = tokens[index]
        owned_by_host, _claims_next = _option_token_ownership(value, host_options)
        if value != END_OF_OPTIONS and not owned_by_host:
            continue
        replacement = f"agm-program-value-{index}"
        suffix = 1
        while replacement in unavailable:
            replacement = f"agm-program-value-{index}-{suffix}"
            suffix += 1
        parsed[index] = replacement
        replacements[replacement] = value
        unavailable.add(replacement)
    return parsed, replacements


def protect_potential_program_values(
    tokens: Sequence[str], host_options: OptionValueMap
) -> list[str]:
    """Hide possible program values for an advisory outer parse.

    Before a source has been discovered its unknown options have unknown
    arity. Treat a separate token after one as a possible value only for the
    preview that locates that source. The authoritative pass later uses the
    discovered :class:`ProgramCommand` and restores ordinary host ownership.
    """
    parsed = list(tokens)
    unavailable = set(tokens)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == END_OF_OPTIONS:
            break
        owned_by_host, host_claims_next = _option_token_ownership(token, host_options)
        if owned_by_host:
            index += 2 if host_claims_next and index + 1 < len(tokens) else 1
            continue
        possible_program_option = token.startswith("--") and "=" not in token
        possible_program_option = possible_program_option or (
            token.startswith("-") and token != "-" and len(token) == 2
        )
        if not possible_program_option or index + 1 >= len(tokens):
            index += 1
            continue
        value_index = index + 1
        value = tokens[value_index]
        value_owned_by_host, _claims_next = _option_token_ownership(value, host_options)
        if value == END_OF_OPTIONS or value_owned_by_host:
            replacement = f"agm-program-preview-{value_index}"
            suffix = 1
            while replacement in unavailable:
                replacement = f"agm-program-preview-{value_index}-{suffix}"
                suffix += 1
            parsed[value_index] = replacement
            unavailable.add(replacement)
        index += 2
    return parsed


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
    parameters' declared names, in declaration order.
    """

    first_parameter: str
    second_parameter: str
    flag: str


type ProgramOptionError = ReservedFlagError | DuplicateOptionFlagError


def option_some_raw(value: object) -> OptionSome:
    """Box one raw ``Option[T]`` "Some" value for ``decode_param_value`` to decode.

    The single place this box is built — shared by :func:`_positive_raw`,
    for a CLI flag's raw ``VALUE`` token, and by the config-table path
    (``commands.exec_program``), for an already-native TOML/JSON value read
    from a program's qualified table. Decoding *value* (host-text dispatch
    when it is a string, as is otherwise) and wrapping it into the enum's
    ``{"$case": "Some", "value": ...}`` JSON shape happens at decode time
    against the ``Some`` variant's own field type, not here.
    """
    return OptionSome(value)


def option_none_raw() -> object:
    """Return the ``None`` envelope ``decode_param_value`` expects for an ``Option[T]`` value."""
    return {"$case": "None"}


def native_raw_value(projected: ProjectedOption, raw: object) -> object:
    """Project one already-native host value onto *projected*'s raw argument shape.

    The config-table counterpart of :func:`_positive_raw`, which applies the
    same ``Option`` rule to a CLI ``VALUE`` token. A present value for an
    ``Option[T]`` parameter is wrapped ``Some``: a config table has no
    ``--no-x`` equivalent, so an absent key supplies nothing at all and the
    parameter falls back to its own default. A native string for a slot whose
    type is ``json`` (the parameter's own type, or ``Option[T]``'s inner
    ``T``) is encoded into JSON text, preserving its distinction from AgL
    value syntax; every other native string is host text, decoded the same
    way a CLI token or ``@opt-env`` value is; a non-string native is already
    decoded data. Keeping this beside :func:`_positive_raw` is what stops the
    envelope rule from being spelled once per host surface.
    """
    if projected.value_form is ValueForm.OPTION:
        value = raw
        if isinstance(raw, str) and projected.option_inner_form is ValueForm.JSON:
            value = dumps_exact(raw, indent=None)
        return option_some_raw(value)
    if projected.value_form is ValueForm.JSON and isinstance(raw, str):
        return dumps_exact(raw, indent=None)
    return raw


def _positive_raw(projected: ProjectedOption, token: str) -> object:
    """Build the raw value for one value-taking flag's ``VALUE`` token.

    Only ``TEXT``/``JSON``/``VALUE``/``AGENT``/``OPTION`` forms reach here —
    the caller (:meth:`ProgramCommand.parse`) never calls this for a ``BOOL``
    form, whose ``takes_value`` is ``False``. Every form but ``OPTION`` takes
    its token verbatim: ``decode_param_value`` reads a string raw value
    through the shared host-text dispatch, so parsing it here would duplicate
    that step and lose its diagnostic. An ``Option[T]`` token is boxed for the
    same deferred decode, against ``T``'s own field type.
    """
    if projected.option_inner is not None:
        return option_some_raw(token)
    return token


def _reject_repetition(
    occurrences: Sequence[object], flag: str, negative_flag: str | None = None
) -> None:
    """Reject a parameter supplied by more than one CLI token.

    Every projected option is ``multiple`` (:class:`_ProgramOption`), so Click
    records one entry per occurrence and anything past the first is the
    duplicate. The message names the offending token's own polarity:
    *negative_flag* is given only for a ``bool`` pair, whose two polarities
    share one Click parameter and whose occurrences carry their own polarity
    (``True`` for *flag*), so a repeated ``--no-x`` is reported as ``--no-x``
    rather than as the ``--x`` the user never typed. A parameter's long flag
    stands in for its ``@opt-short`` spelling, which shares the same Click
    parameter and leaves no separate trace. An ``@opt-env`` value is always a
    single occurrence, so it never collides with the token that overrides it.
    """
    if len(occurrences) < 2:
        return
    offender = occurrences[1]
    spelling = flag if negative_flag is None or offender else negative_flag
    raise ValueError(f"Option {spelling!r} specified more than once")


def _short_flag(param: "ProgramParamInfo | ParamBindingInfo") -> str | None:
    """Return *param*'s ``-x`` short spelling, or ``None`` when it declares none."""
    short = param.cli.short
    return None if short is None else f"-{short}"


def _spellings(param: "ProgramParamInfo", projected: ProjectedOption) -> list[str]:
    """Return every flag one parameter claims: its own, its negative, and its short.

    The one enumeration of a parameter's CLI spellings, so the reservation
    check (:func:`_check_reservation`) and completion
    (:meth:`ProgramCommand.option_spellings`) can never disagree about what a
    parameter occupies.
    """
    spellings = [*projected.flags, *projected.negative_flags]
    short = _short_flag(param)
    if short is not None:
        spellings.append(short)
    return spellings


class _ProgramOption(click.Option):
    """A program parameter's Click option: every occurrence recorded, separately.

    Every projected option is ``multiple``, so Click records one entry per
    token rather than collapsing repetitions to the last value — a parameter
    supplied twice is a usage error (:meth:`ProgramCommand.parse` diagnoses
    it), not a silent overwrite, and the shared binder downstream is entitled
    to assume duplicates were already caught.

    An ``@opt-env`` value stays exactly one occurrence: Click otherwise reads
    a ``multiple`` option's environment value as a ``os.path.pathsep``-joined
    list, which would make ``VAR=a:b`` look like a repeated flag. Click
    supplies an environment value only when it is non-empty, so ``VAR=""``
    reads as unset and an environment fallback can never deliver an empty
    ``text``.
    """

    def value_from_envvar(self, ctx: click.Context) -> list[str] | None:
        """Return this option's environment value as a single occurrence."""
        raw = self.resolve_envvar_value(ctx)
        return None if raw is None else [raw]


class _EnvironmentFallbackOption(_ProgramOption):
    """An environment-only program parameter fallback with no CLI declaration."""

    def add_to_parser(self, parser: object, ctx: click.Context) -> None:
        """Keep the synthetic declaration out of Click's accepted option table."""
        del parser, ctx


def _value_form_metavar(form: ValueForm) -> str:
    """Return the placeholder for a bare (non-``Option``) value form."""
    if form is ValueForm.TEXT:
        return "TEXT"
    if form is ValueForm.AGENT:
        return "AGENT"
    if form is ValueForm.JSON:
        return "JSON"
    return "VALUE"


def _default_metavar(
    param: "ProgramParamInfo | ParamBindingInfo", projected: ProjectedOption
) -> str:
    """Return the placeholder standing for a value-taking option's own VALUE.

    A ``path`` parameter (directly or as ``Option[path]``) announces a path.
    Otherwise it names how the token is read rather than the declared type: a
    ``text`` parameter takes its token verbatim, standard ``Agent`` uses host
    syntax, ``json`` parses one JSON value, and every other type reads strict
    JSON or AgL value syntax. An ``Option[T]`` follows ``T``.
    """
    if isinstance(param, ProgramParamInfo) and param.is_path:
        return "PATH"
    if projected.value_form is ValueForm.OPTION:
        return _value_form_metavar(cast(ValueForm, projected.option_inner_form))
    return _value_form_metavar(projected.value_form)


def _click_params(
    index: int,
    param: "ProgramParamInfo | ParamBindingInfo",
    projected: ProjectedOption,
    *,
    names: Sequence[str] | None = None,
) -> list[click.Parameter]:
    """Build the Click parameter(s) carrying one program parameter's CLI surface.

    A ``bool`` parameter is a single ``--x/--no-x`` flag, whose recorded
    occurrences carry their own polarity; an ``Option[T]`` parameter is a
    value-taking ``--x`` plus an independent ``--no-x`` flag, whose
    exclusivity :meth:`ProgramCommand.parse` checks once both are parsed;
    every other form is one value-taking option. Only the positive option
    carries the environment fallback, so a variable is read once, and only it
    carries the parameter's ``@doc`` help, which describes the parameter
    rather than either polarity of its flag.
    """
    spec = param.cli
    option_names = (param.cli.name,) if names is None else tuple(names)
    positive_flags = tuple(f"--{name}" for name in option_names)
    negative_flags = tuple(f"--no-{name}" for name in option_names)
    short = _short_flag(param)
    shorts: list[str] = [] if short is None else [short]
    if projected.value_form is ValueForm.BOOL:
        return [
            _ProgramOption(
                [
                    _positive_dest(index),
                    *(
                        f"{positive}/{negative}"
                        for positive, negative in zip(positive_flags, negative_flags, strict=True)
                    ),
                    *shorts,
                ],
                type=click.BOOL,
                default=None,
                multiple=True,
                envvar=spec.env,
                help=spec.doc,
                hidden=spec.hidden,
            )
        ]
    positive = _ProgramOption(
        [_positive_dest(index), *positive_flags, *shorts],
        default=None,
        multiple=True,
        metavar=spec.metavar or _default_metavar(param, projected),
        envvar=spec.env,
        help=spec.doc,
        hidden=spec.hidden,
    )
    if projected.value_form is not ValueForm.OPTION:
        return [positive]
    negative = _ProgramOption(
        [_negative_dest(index), *negative_flags],
        is_flag=True,
        type=click.BOOL,
        default=None,
        multiple=True,
        hidden=spec.hidden,
    )
    return [positive, negative]


def _module_click_params(
    index: int, entry: "ParamSurfaceEntry", projected: ProjectedOption
) -> list[click.Parameter]:
    """Build one module parameter's Click options from its resolved flag table."""
    spec = entry.param.cli
    positive = (*entry.positive_option_spellings, *entry.short_option_spellings)
    negative = entry.negative_option_spellings
    if projected.value_form is ValueForm.BOOL:
        result: list[click.Parameter] = []
        if positive:
            result.append(
                _ProgramOption(
                    [_positive_dest(index), *positive],
                    is_flag=True,
                    default=None,
                    multiple=True,
                    envvar=spec.env,
                    help=spec.doc,
                    hidden=spec.hidden,
                )
            )
        elif spec.env is not None:
            result.append(_module_environment_fallback(index, entry, projected))
        if negative:
            result.append(
                _ProgramOption(
                    [_negative_dest(index), *negative],
                    is_flag=True,
                    default=None,
                    multiple=True,
                    hidden=spec.hidden,
                )
            )
        return result
    result = []
    if positive:
        result.append(
            _ProgramOption(
                [_positive_dest(index), *positive],
                default=None,
                multiple=True,
                metavar=spec.metavar or _default_metavar(entry.param, projected),
                envvar=spec.env,
                help=spec.doc,
                hidden=spec.hidden,
            )
        )
    elif spec.env is not None:
        result.append(_module_environment_fallback(index, entry, projected))
    if negative:
        result.append(
            _ProgramOption(
                [_negative_dest(index), *negative],
                is_flag=True,
                type=click.BOOL,
                default=None,
                multiple=True,
                hidden=spec.hidden,
            )
        )
    return result


def _module_environment_fallback(
    index: int, entry: "ParamSurfaceEntry", projected: ProjectedOption
) -> _EnvironmentFallbackOption:
    """Build a module parameter's environment-only positive-polarity source."""
    spec = entry.param.cli
    assert spec.env is not None
    declarations = [_positive_dest(index), f"--_module-env-{index}"]
    if projected.value_form is ValueForm.BOOL:
        return _EnvironmentFallbackOption(
            declarations,
            is_flag=True,
            default=None,
            multiple=True,
            envvar=spec.env,
            hidden=True,
        )
    return _EnvironmentFallbackOption(
        declarations,
        default=None,
        multiple=True,
        metavar=spec.metavar or _default_metavar(entry.param, projected),
        envvar=spec.env,
        hidden=True,
    )


def _module_help_record(entry: "ParamSurfaceEntry", projected: ProjectedOption) -> tuple[str, str]:
    """Render one module parameter under its shortest resolving option spelling."""
    spelling = next(
        iter(
            (
                *entry.positive_option_spellings,
                *entry.short_option_spellings,
                *entry.negative_option_spellings,
            )
        )
    )
    if projected.takes_value and spelling not in entry.negative_option_spellings:
        spelling = (
            f"{spelling} {entry.param.cli.metavar or _default_metavar(entry.param, projected)}"
        )
    help_text = entry.param.cli.doc or ""
    if entry.param.cli.env is not None:
        environment = f"[env var: {entry.param.cli.env}]"
        help_text = f"{help_text}  {environment}" if help_text else environment
    return spelling, help_text


def _module_primary_flag(entry: "ParamSurfaceEntry") -> str:
    """Return an available spelling, or the logical positive flag for diagnostics."""
    if entry.positive_option_spellings:
        return entry.positive_option_spellings[0]
    if entry.negative_option_spellings:
        return entry.negative_option_spellings[0]
    return f"--{entry.param.cli.name}"


class _ProgramClickCommand(click.Command):
    """A program's Click command, whose usage line names its positional slots.

    Every positional token lands in one catch-all argument whose internal
    name says nothing to a reader, so the usage line is spelled from the
    program's own positional parameters instead, in declaration order; see
    :func:`_usage_slots` for how one slot is spelled.
    """

    def __init__(
        self,
        name: str,
        *,
        params: list[click.Parameter],
        description: str | None,
        usage_slots: tuple[str, ...],
        parameter_sections: tuple[tuple[str, tuple[click.Parameter, ...]], ...] = (),
        module_help_sections: tuple[
            tuple[str, tuple[tuple["ParamSurfaceEntry", ProjectedOption], ...]], ...
        ] = (),
        ambiguous_options: Mapping[str, tuple["ParamBindingInfo", ...]] | None = None,
        surface_entries: Mapping["ParamBindingInfo", "ParamSurfaceEntry"] | None = None,
        context_settings: dict[str, bool] | None = None,
    ) -> None:
        settings: dict[str, bool] = {} if context_settings is None else dict(context_settings)
        super().__init__(
            name=name,
            params=params,
            help=description,
            add_help_option=False,
            context_settings=settings,
        )
        self.usage_slots = usage_slots
        self.parameter_sections = parameter_sections
        self.module_help_sections = module_help_sections
        self.ambiguous_options = {} if ambiguous_options is None else ambiguous_options
        self.surface_entries = {} if surface_entries is None else surface_entries

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Turn Click's invalid ambiguity value syntax into the candidate diagnostic."""
        ambiguous_short = self._ambiguous_short_option(args)
        if ambiguous_short is not None:
            short_candidates = self.ambiguous_options[ambiguous_short]
            raise click.UsageError(
                _ambiguous_option_message(ambiguous_short, short_candidates, self.surface_entries),
                ctx,
            )
        try:
            return super().parse_args(ctx, args)
        except click.BadOptionUsage as exc:
            candidates = self.ambiguous_options.get(exc.option_name)
            if candidates is None:
                raise
            raise click.UsageError(
                _ambiguous_option_message(exc.option_name, candidates, self.surface_entries), ctx
            ) from exc

    def _ambiguous_short_option(self, args: Sequence[str]) -> str | None:
        """Return an ambiguous short spelling Click reaches before any parse error.

        Click rejects the suffix of ``-tVALUE`` after accepting the flag
        ``-t`` when that flag is a rejecting ambiguity sentinel.  Detect that
        spelling before Click reaches its suffix so both the direct command
        and :meth:`ProgramCommand.parse` report the candidates.  The walk
        deliberately stops at a value-taking option: in ``-xt``, ``t`` is
        the attached value for ``-x``, not another short option.
        """
        option_values = option_value_map(cast(Sequence[click.Option], self.params[1:]))
        index = 0
        while index < len(args):
            token = args[index]
            if token == END_OF_OPTIONS:
                return None
            if token.startswith("--"):
                spelling, separator, _value = token.partition("=")
                if option_values.get(spelling, False) and not separator:
                    index += 2
                    continue
                index += 1
                continue
            if not token.startswith("-") or token == "-":
                index += 1
                continue
            for offset, short in enumerate(token[1:]):
                spelling = f"-{short}"
                if spelling in self.ambiguous_options:
                    return spelling
                takes_value = option_values.get(spelling)
                if takes_value is None:
                    index += 1
                    break
                if takes_value:
                    if offset + 2 == len(token) and index + 1 < len(args):
                        index += 2
                    else:
                        index += 1
                    break
            else:
                index += 1
                continue
            continue
        return None

    def collect_usage_pieces(self, ctx: click.Context) -> list[str]:
        """Return the usage pieces after the command name."""
        del ctx
        options = [] if self.options_metavar is None else [self.options_metavar]
        return [*options, *self.usage_slots]

    def format_options(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Render program options, then module-parameter sections."""
        section_params = {
            id(param) for _title, params in self.parameter_sections for param in params
        }
        own_options = [
            record
            for param in self.get_params(ctx)
            if id(param) not in section_params
            if (record := param.get_help_record(ctx)) is not None
        ]
        with formatter.section("Options"):
            formatter.write_dl(own_options)
        for title, entries in self.module_help_sections:
            records = [
                _module_help_record(entry, projected)
                for entry, projected in entries
                if not entry.hidden
            ]
            if records:
                with formatter.section(f"Parameters of {title}"):
                    formatter.write_dl(records)


def _build_click_command(
    name: str,
    *,
    params: Sequence[click.Parameter],
    description: str | None,
    usage_slots: tuple[str, ...],
    extra_options: Sequence[click.Parameter] = (),
    parameter_sections: tuple[tuple[str, tuple[click.Parameter, ...]], ...] = (),
    module_help_sections: tuple[
        tuple[str, tuple[tuple["ParamSurfaceEntry", ProjectedOption], ...]], ...
    ] = (),
    ambiguous_options: Mapping[str, tuple["ParamBindingInfo", ...]] | None = None,
    surface_entries: Mapping["ParamBindingInfo", "ParamSurfaceEntry"] | None = None,
    context_settings: dict[str, bool] | None = None,
) -> _ProgramClickCommand:
    """Assemble one program command: its own parameters, then *extra_options*, then help.

    The single constructor for the command that parses a program's tokens,
    the one that renders its help under a host's own invocation name, and the
    stand-in for a program that could not be built, so none of them can
    advertise different options.
    """
    return _ProgramClickCommand(
        name,
        params=[*params, *extra_options, _help_option()],
        description=description,
        usage_slots=usage_slots,
        parameter_sections=parameter_sections,
        module_help_sections=module_help_sections,
        ambiguous_options=ambiguous_options,
        surface_entries=surface_entries,
        context_settings=context_settings,
    )


def _usage_slots(positional: "tuple[ProgramParamInfo, ...]") -> tuple[str, ...]:
    """Return one usage slot per positional-capable parameter, in declaration order.

    A slot is named by the parameter's own ``@opt-metavar`` when it declares
    one and by its external name otherwise, so a positional parameter — never
    addressed by a flag — still gets the placeholder its declaration asks for.

    ``@opt-hidden`` is about how a parameter is addressed *by name*, so it
    hides the parameter's ``--name`` entry alone: every positional-capable
    parameter keeps its slot here, because the shared binder still fills it
    from the positional tokens a reader types.
    """
    slots: list[str] = []
    for param in positional:
        slot = param.cli.metavar or param.cli.name
        slots.append(f"[{slot}]" if param.has_default else f"<{slot}>")
    return tuple(slots)


@dataclass(frozen=True, slots=True)
class ProgramCommand:
    """One program's whole CLI surface: a ``click.Command`` and what it stands for.

    ``command`` parses tokens; ``positional`` lists positional-capable
    parameters (``POSITIONAL_ONLY``, ``STANDARD``) in declaration order; and
    ``options`` pairs every name-addressable parameter (``STANDARD``,
    ``NAMED_ONLY``) with its projected CLI option, in the same order the
    command's own options were built from. A ``STANDARD`` parameter appears
    in both, since it accepts either a positional token or ``--name``.

    ``options`` also carries the external↔declared correspondence: each entry
    holds the parameter's declared name and the external name its flags were
    spelled from (``ProgramParamInfo.cli.name``).

    ``params`` are the Click parameters ``command`` parses with, kept so the
    help rendering (:meth:`render_help`) can present the same surface under a
    host's own invocation name. The positional slots its usage line names
    live on ``command`` itself.
    """

    command: _ProgramClickCommand
    positional: tuple["ProgramParamInfo", ...]
    options: tuple[tuple["ProgramParamInfo", ProjectedOption], ...]
    module_options: tuple[tuple["ParamSurfaceEntry", ProjectedOption], ...]
    surface: "ParamSurface"
    params: tuple[click.Parameter, ...]

    def parse(self, tokens: Sequence[str]) -> ParsedTail:
        """Parse *tokens* into raw positional/named host values, keyed by declared name.

        Click owns the token conventions: short options and their bundles,
        attached and ``--x=V`` values, the ``--`` end-of-options marker, and
        an option's value being whatever token follows it. Every non-option
        token lands in the catch-all positional slot unconditionally — the
        shared zone binder (``runtime.arguments.bind_program_arguments``)
        pairs positional values with parameters left to right and diagnoses
        an excess positional argument itself, with a message that depends on
        whether the program declares any named-only parameters, so capping
        here would pre-empt that one diagnosis.

        A parameter whose value came from neither a token nor its
        ``@opt-env`` variable is left out of the result entirely, so a host
        can tell "supplied" from "defaulted" and fall back to its own
        configuration layer.

        :raises ProgramHelpRequested: when the tokens ask for this command's
            help, so a caller that has not already recognized the request —
            a ``-h`` bundled into a short group, say — renders it rather than
            running the program.
        :raises ValueError: for any Click usage error (unknown option,
            missing value, a value given to a flag), for a parameter supplied
            more than once, or for an ``Option[T]`` parameter given both
            polarities on the command line. A malformed ``VALUE`` — including
            a malformed ``Option`` payload (see the module docstring) — is
            not raised here: it reaches this method's caller as a deferred
            decode diagnostic from ``decode_param_value``, anchored at the
            parameter's declaration.
        """
        try:
            ctx = self.command.make_context(self.command.name, list(tokens))
        except _HelpSignal as exc:
            raise ProgramHelpRequested(self) from exc
        except click.UsageError as exc:
            raise ValueError(exc.format_message()) from exc
        values = cast(dict[str, object], ctx.params)
        positional = cast(tuple[str, ...], values[_POSITIONAL_DEST])
        positional_names = self.positionally_filled_names(len(positional))

        named: dict[str, object] = {}
        for index, (param, projected) in enumerate(self.options):
            flag = projected.flags[0]
            positive = cast(tuple[object, ...], values[_positive_dest(index)])
            # A STANDARD parameter can be filled by either its positional
            # slot or its name. A positional CLI token outranks an envvar
            # fallback just as a named CLI token does, but a named CLI token
            # remains a duplicate for the shared binder to reject.
            positional_overrides_environment = (
                param.name in positional_names
                and self._from_environment(ctx, _positive_dest(index))
            )
            if projected.value_form is ValueForm.BOOL:
                # Both polarities fill one Click parameter, so this single
                # check covers same- and mixed-polarity repetition alike.
                _reject_repetition(positive, flag, projected.negative_flags[0])
                if positive and not positional_overrides_environment:
                    named[param.name] = cast(bool, positive[0])
                continue
            _reject_repetition(positive, flag)
            if projected.value_form is not ValueForm.OPTION:
                if positive and not positional_overrides_environment:
                    named[param.name] = _positive_raw(projected, cast(str, positive[0]))
                continue
            negative_flag = projected.negative_flags[0]
            negative = cast(tuple[object, ...], values[_negative_dest(index)])
            _reject_repetition(negative, negative_flag)
            # An ``Option[T]``'s two polarities are two Click parameters, so
            # only their sources tell a supplied token from an ``@opt-env``
            # value. Both polarities typed together is the contradiction; a
            # ``--no-x`` token against an environment-supplied positive is
            # not, since a CLI token outranks the environment.
            if self._from_commandline(ctx, _positive_dest(index)) and self._from_commandline(
                ctx, _negative_dest(index)
            ):
                raise ValueError(f"Options {flag!r} and {negative_flag!r} cannot both be supplied")
            if negative:
                named[param.name] = option_none_raw()
            elif positive and not positional_overrides_environment:
                named[param.name] = _positive_raw(projected, cast(str, positive[0]))
        module_params: dict[StaticBindingKey, object] = {}
        for offset, (entry, projected) in enumerate(self.module_options, start=len(self.options)):
            module_param = entry.param
            positive_flag = _module_primary_flag(entry)
            positive = cast(tuple[object, ...], values.get(_positive_dest(offset), ()))
            if projected.value_form is ValueForm.BOOL:
                _reject_repetition(positive, positive_flag)
                negative = cast(tuple[object, ...], values.get(_negative_dest(offset), ()))
                negative_flag = next(iter(entry.negative_option_spellings), positive_flag)
                _reject_repetition(negative, negative_flag)
                if (
                    positive
                    and negative
                    and self._from_commandline(ctx, _positive_dest(offset))
                    and self._from_commandline(ctx, _negative_dest(offset))
                ):
                    raise ValueError(
                        f"Options {positive_flag!r} and {negative_flag!r} cannot both be supplied"
                    )
                if negative:
                    module_params[module_param.key] = False
                elif positive:
                    module_params[module_param.key] = cast(bool, positive[0])
                continue
            _reject_repetition(positive, positive_flag)
            if projected.value_form is not ValueForm.OPTION:
                if positive:
                    module_params[module_param.key] = _positive_raw(
                        projected, cast(str, positive[0])
                    )
                continue
            negative_flag = next(iter(entry.negative_option_spellings), positive_flag)
            negative = cast(tuple[object, ...], values.get(_negative_dest(offset), ()))
            _reject_repetition(negative, negative_flag)
            if (
                positive
                and negative
                and self._from_commandline(ctx, _positive_dest(offset))
                and self._from_commandline(ctx, _negative_dest(offset))
            ):
                raise ValueError(
                    f"Options {positive_flag!r} and {negative_flag!r} cannot both be supplied"
                )
            if negative:
                module_params[module_param.key] = option_none_raw()
            elif positive:
                module_params[module_param.key] = _positive_raw(projected, cast(str, positive[0]))
        return ParsedTail(
            arguments=ProgramArguments(positional=positional, named=named), params=module_params
        )

    def value_token_indexes(
        self, tokens: Sequence[str], *, host_options: OptionValueMap = frozenset()
    ) -> frozenset[int]:
        """Return indexes that this program reads as separate option values.

        Hosts use this before their outer Click command runs. Host options
        and their own separate values are skipped as one unit. A token in one
        of these positions belongs to the program even when it happens to
        spell a host option, so the outer command must leave it in the raw
        tail. Inline long and attached short values are part of their option
        token and therefore have no separate index to report.
        """
        return self._read_tokens(tokens, host_options)[0]

    def next_positional(
        self, tokens: Sequence[str], token: str, *, host_options: OptionValueMap = frozenset()
    ) -> "ProgramParamInfo | None":
        """Return the positional slot *token*, written after *tokens*, fills.

        ``None`` when *token* fills none: option tokens, their separate values,
        and *host_options* with their values fill no slot, nor does any token
        once every positional-capable parameter is filled. Every token after
        ``--`` fills one.
        """
        count = self._read_tokens(tokens, host_options)[1]
        if count >= len(self.positional) or (
            self._read_tokens([*tokens, token], host_options)[1] == count
        ):
            return None
        return self.positional[count]

    def value_options(self) -> tuple[tuple["ProgramParamInfo", tuple[str, ...]], ...]:
        """Return each value-taking parameter with the spellings its value follows.

        Its long flag and ``@opt-short`` spelling; a ``--no-x`` negative
        never takes a value.
        """
        result: list[tuple["ProgramParamInfo", tuple[str, ...]]] = []
        for param, projected in self.options:
            if not projected.takes_value:
                continue
            short = _short_flag(param)
            result.append((param, (projected.flags[0], *(() if short is None else (short,)))))
        return tuple(result)

    def _read_tokens(
        self, tokens: Sequence[str], host_options: OptionValueMap
    ) -> tuple[frozenset[int], int]:
        """Return the separate option-value indexes in *tokens* and its positional token count."""
        long_options: dict[str, bool] = {}
        short_options: dict[str, bool] = {}
        for param, projected in self.options:
            long_options[projected.flags[0]] = projected.takes_value
            for flag in projected.negative_flags:
                long_options[flag] = False
            short = _short_flag(param)
            if short is not None:
                short_options[short] = projected.takes_value
        for entry, projected in self.module_options:
            for spelling in entry.option_spellings:
                if spelling.startswith("--"):
                    long_options[spelling] = projected.takes_value and not spelling.startswith(
                        "--no-"
                    )
                else:
                    short_options[spelling] = projected.takes_value

        values: set[int] = set()
        positional = 0
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token == END_OF_OPTIONS:
                positional += len(tokens) - index - 1
                break
            owned_by_host, host_claims_next = _option_token_ownership(token, host_options)
            if owned_by_host:
                index += 2 if host_claims_next and index + 1 < len(tokens) else 1
                continue
            if token.startswith("--"):
                flag, separator, _value = token.partition("=")
                if separator or not long_options.get(flag, False):
                    index += 1
                    continue
                if index + 1 < len(tokens):
                    values.add(index + 1)
                    index += 2
                    continue
            elif token.startswith("-") and token != "-":
                short_group = token[1:]
                for offset, short in enumerate(short_group):
                    takes_value = short_options.get(f"-{short}")
                    if takes_value is None:
                        continue
                    if takes_value and offset + 1 == len(short_group) and index + 1 < len(tokens):
                        values.add(index + 1)
                        index += 2
                        break
                    if takes_value:
                        index += 1
                        break
                else:
                    index += 1
                continue
            else:
                positional += 1
            index += 1
        return frozenset(values), positional

    @staticmethod
    def _from_commandline(ctx: click.Context, dest: str) -> bool:
        """Return whether *dest* was filled by a CLI token rather than the environment.

        Click records where each parameter's value came from, which is the
        only way to tell a token from an environment fallback: both arrive as
        an ordinary parsed value.
        """
        return ctx.get_parameter_source(dest) is ParameterSource.COMMANDLINE

    @staticmethod
    def _from_environment(ctx: click.Context, dest: str) -> bool:
        """Return whether *dest* was filled by its environment fallback."""
        return ctx.get_parameter_source(dest) is ParameterSource.ENVIRONMENT

    def render_help(
        self,
        program_name: str,
        *,
        description: str | None = None,
        extra_options: Sequence[click.Parameter] = (),
    ) -> str:
        """Render this command's help as the invocation *program_name* spells it.

        The usage line names *program_name* and the program's own positional
        slots; the description is the program's ``@doc`` unless *description*
        supplies one of the host's own (a package manifest's, say); and the
        options are this program's visible flags with their own ``@doc`` and
        metavars, followed by *extra_options* — flags the host adds around the
        program, such as ``--dry-run`` — and the help flags themselves.
        """
        command = _build_click_command(
            program_name,
            params=self.params,
            description=self.command.help if description is None else description,
            usage_slots=self.command.usage_slots,
            extra_options=extra_options,
            parameter_sections=self.command.parameter_sections,
            module_help_sections=self.command.module_help_sections,
            ambiguous_options=self.command.ambiguous_options,
            surface_entries=self.command.surface_entries,
        )
        return _format_help(command, program_name)

    def option_spellings(self) -> tuple[str, ...]:
        """Return every completable option spelling this command accepts.

        Each visible parameter's long flag, its derived negative and its
        ``@opt-short`` spelling, then the help flags the command owns. A
        hidden parameter contributes none, so completion offers exactly the
        ``--name`` entries the help lists.
        """
        spellings: list[str] = []
        for param, projected in self.options:
            if param.cli.hidden:
                continue
            spellings.extend(_spellings(param, projected))
        for entry, _projected in self.module_options:
            if not entry.hidden:
                spellings.extend(entry.option_spellings)
        spellings.extend(HELP_FLAGS)
        return tuple(spellings)

    def positionally_filled_names(self, count: int) -> frozenset[str]:
        """Return the declared names *count* positional tokens fill.

        The shared binder (``runtime.arguments.bind_program_arguments``) pairs
        positional values with positional-capable parameters left to right,
        without skipping, so the first *count* of them are the ones a token
        supplied. Held here so every surface deciding whether a positional
        token outranks a lower-precedence layer — an ``@opt-env`` fallback in
        :meth:`parse`, a configured value in ``commands.exec_program`` — reads
        the binder's pairing rule from one place.
        """
        return frozenset(param.name for param in self.positional[:count])

    def positional_only_names(self) -> frozenset[str]:
        """Return the external names of this program's positional-only parameters.

        A positional-only parameter exposes no name in the CLI/config
        namespace — it fills an ``ARG`` slot only, never a ``--name`` flag or
        a config-table key — so callers that report on the name-addressable
        surface (config-key diagnostics, completions) use this to tell those
        parameters apart from a genuinely undeclared name.
        """
        return frozenset(
            param.cli.name for param in self.positional if param.kind is ParamZone.POSITIONAL_ONLY
        )


def _check_reservation(
    options: "tuple[tuple[ProgramParamInfo, ProjectedOption], ...]",
    reserved_flags: frozenset[str],
) -> ProgramOptionError | None:
    """Return the first flag collision among *options*, or ``None`` when there is none.

    Every spelling a parameter claims — its ``--name``, the derived
    ``--no-name`` of a negatable form, and its ``-x`` short — is checked
    against *reserved_flags* (the invoking surface's inventory) and against
    every other parameter's own spellings, in declaration order.
    """
    seen: dict[str, str] = {}
    for param, projected in options:
        for flag in _spellings(param, projected):
            if flag in reserved_flags:
                return ReservedFlagError(parameter=param.name, flag=flag)
            if flag in seen:
                return DuplicateOptionFlagError(
                    first_parameter=seen[flag], second_parameter=param.name, flag=flag
                )
            seen[flag] = param.name
    return None


def _ambiguous_option_callback(
    spelling: str,
    candidates: tuple["ParamBindingInfo", ...],
    entries: Mapping["ParamBindingInfo", "ParamSurfaceEntry"],
) -> Callable[[click.Context, click.Parameter, object], None]:
    """Return the usage callback for one ambiguous module-parameter flag."""

    def reject(ctx: click.Context, param: click.Parameter, value: object) -> None:
        del ctx, param
        if value:
            raise click.UsageError(_ambiguous_option_message(spelling, candidates, entries))

    return reject


def _ambiguous_option_message(
    spelling: str,
    candidates: tuple["ParamBindingInfo", ...],
    entries: Mapping["ParamBindingInfo", "ParamSurfaceEntry"],
) -> str:
    """Render one ambiguous option's declaration-aware usage diagnostic."""
    descriptions = "; ".join(
        f"{candidate.declaration_path} ({', '.join(entries[candidate].option_spellings)})"
        for candidate in candidates
    )
    return f"ambiguous option {spelling!r}: {descriptions}"


def _ambiguous_click_options(
    ambiguous: Mapping[str, tuple["ParamBindingInfo", ...]],
    entries: Mapping["ParamBindingInfo", "ParamSurfaceEntry"],
) -> list[click.Parameter]:
    """Build hidden rejecting options for every ambiguous module spelling."""
    return [
        click.Option(
            [f"_ambiguous{index}", spelling],
            is_flag=True,
            is_eager=True,
            expose_value=False,
            hidden=True,
            callback=_ambiguous_option_callback(spelling, candidates, entries),
        )
        for index, (spelling, candidates) in enumerate(ambiguous.items())
    ]


def build_program_command(
    program: "ProgramDeclInfo",
    reserved_flags: frozenset[str],
    params: Sequence["ParamBindingInfo"] = (),
) -> "ProgramCommand | ProgramOptionError":
    """Build *program*'s Click command, checking flag reservation first.

    *reserved_flags* is the invoking surface's inventory —
    :data:`EXEC_RESERVED_FLAGS` or :data:`REGISTERED_RESERVED_FLAGS`. A
    program parameter can never shadow one of those, nor another
    parameter's own spelling — so one parameter's ``--no-<name>`` negation
    can never silently steal a different parameter literally named
    ``no-<name>``, and two parameters can never claim the same short.
    Positional-only parameters never enter the flag namespace and so are
    never checked. Returns a :class:`ReservedFlagError` or
    :class:`DuplicateOptionFlagError` for the caller to render on the first
    collision found, in declaration order.

    The returned command owns ``-h``/``--help`` as well: it carries the
    program's own ``@doc`` prose and each parameter's, and its help is the
    help a reader of this program sees.
    """
    signature = program.parameters
    positional = tuple(p for p in signature if p.kind in _POSITIONAL_ZONES)
    options = tuple(
        (p, project_option(p.cli.name, p.type))
        for p in signature
        if p.kind in _NAME_ADDRESSABLE_ZONES
    )
    collision = _check_reservation(options, reserved_flags)
    if collision is not None:
        return collision
    from agm.cli_support.param_surface import build_param_surface

    surface = build_param_surface(reserved_flags, program, params)
    module_options = tuple(
        (entry, project_option(entry.param.cli.name, entry.param.type))
        for entry in surface.entries
        if entry.option_spellings or entry.param.cli.env is not None
    )
    click_params: list[click.Parameter] = [_positional_argument()]
    for index, (param, projected) in enumerate(options):
        click_params.extend(_click_params(index, param, projected))
    sections: dict[str, list[click.Parameter]] = {}
    help_sections: dict[str, list[tuple[ParamSurfaceEntry, ProjectedOption]]] = {}
    for offset, (entry, projected) in enumerate(module_options, start=len(options)):
        entry_params = _module_click_params(offset, entry, projected)
        click_params.extend(entry_params)
        sections.setdefault(entry.section, []).extend(entry_params)
        if entry.option_spellings:
            help_sections.setdefault(entry.section, []).append((entry, projected))
    entries = {entry.param: entry for entry in surface.entries}
    click_params.extend(_ambiguous_click_options(surface.ambiguous_options, entries))
    command = _build_click_command(
        program.declaration_path,
        params=click_params,
        description=program.doc,
        usage_slots=_usage_slots(positional),
        parameter_sections=tuple(
            (section, tuple(section_params)) for section, section_params in sections.items()
        ),
        module_help_sections=tuple(
            (section, tuple(section_entries)) for section, section_entries in help_sections.items()
        ),
        ambiguous_options=surface.ambiguous_options,
        surface_entries=entries,
    )
    return ProgramCommand(
        command=command,
        positional=positional,
        options=options,
        module_options=module_options,
        surface=surface,
        params=tuple(click_params),
    )


def program_command_for(
    program: "ProgramDeclInfo | None",
    reserved_flags: frozenset[str],
    params: Sequence["ParamBindingInfo"] = (),
) -> "ProgramCommand | None":
    """Build *program*'s command, degrading a missing program or a collision to ``None``.

    For the advisory surfaces (help, completion, ``-h`` disambiguation), which
    show no program-argument flags rather than raising the host diagnostic the
    execution path takes from :func:`build_program_command`: no program
    selected and a colliding projection are the same outcome there.
    """
    if program is None:
        return None
    result = build_program_command(program, reserved_flags, params)
    return result if isinstance(result, ProgramCommand) else None


def _fallback_command() -> click.Command:
    """Return a command that owns ``-h``/``--help`` and treats every other token as positional.

    The stand-in for a program whose command could not be built — its source
    does not compile, no single program is selected, or its flags collide —
    so a help request is still recognized by Click's own parsing rather than
    by a token scan of the host's own.
    """
    return _build_click_command(
        "",
        params=[_positional_argument()],
        description=None,
        usage_slots=(),
        context_settings={"ignore_unknown_options": True},
    )


def program_help_requested(
    tokens: "Sequence[str]", program_command: "ProgramCommand | None"
) -> bool:
    """Return whether *tokens* ask *program_command* for its help.

    Click decides, so every convention it owns applies: a ``-h`` bundled into
    a short group is a help request, one standing where a value-taking option
    expects its value is that value, and one past the end-of-options marker is
    a positional argument. A token stream Click rejects outright (an unknown
    option before the help flag) is no help request either — the host runs it
    and reports the usage error.
    """
    command = _fallback_command() if program_command is None else program_command.command
    try:
        command.make_context(command.name, list(tokens))
    except _HelpSignal:
        return True
    except click.ClickException:
        return False
    return False


def _format_help(command: click.Command, program_name: str) -> str:
    """Render *command*'s help as it reads under the invocation *program_name*."""
    return command.get_help(click.Context(command, info_name=program_name)) + "\n"


def render_program_help(
    program_command: "ProgramCommand | None",
    *,
    program_name: str,
    description: str | None = None,
    extra_options: "Sequence[click.Parameter]" = (),
) -> str:
    """Render the help of the command *program_name* invokes.

    Without a usable *program_command* — an undiscoverable program, or one
    whose flags collide — the same surface is rendered without any program
    flags, so a host that can still describe the command (its own options and
    a description of its own) presents one help layout, not two.
    """
    if program_command is not None:
        return program_command.render_help(
            program_name, description=description, extra_options=extra_options
        )
    command = _build_click_command(
        program_name,
        params=(),
        description=description,
        usage_slots=(),
        extra_options=extra_options,
    )
    return _format_help(command, program_name)


def exec_program_help(
    program_command: "ProgramCommand | None", *, file: str | None, program: str | None
) -> str:
    """Render the help of the ``agm exec`` invocation *program_command* was selected by.

    The one rendering shared by the pre-parse help path (``cli._exec_print_help``)
    and the one reached through :class:`ProgramHelpRequested`
    (``commands.exec``), so a help request answered before parsing and one
    recognized while parsing produce the same page.
    """
    return render_program_help(
        program_command, program_name=exec_program_name(file=file, program=program)
    )


def exec_program_name(*, file: str | None, program: str | None) -> str:
    """Return the ``agm exec`` invocation a selected program's usage line is spelled with.

    Names the source the reader gave — a path or an installed reference, or
    the ``-c`` option for inline text, whose program text would not read as a
    usage line — and the ``-p`` selection when one was made, so the usage
    line stands for the command that was actually run.
    """
    parts = ["agm", "exec", "-c COMMAND" if file is None else file]
    if program is not None:
        parts.extend(("-p", program))
    return " ".join(parts)
