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
(:class:`ProgramCommand`), so the reserved set and the parameter flags can
never disagree.

:func:`build_program_command` builds a whole program's CLI surface from its
declaration: one ``click.Command`` carrying a catch-all positional argument
and one option per name-addressable parameter, plus a reservation check
against the host's flag inventory and against every other parameter's own
spellings. Click owns every token convention — short options, bundling,
attached values, ``--x=V``, the ``--`` end-of-options marker — while zone
pairing, missing required arguments and excess positionals stay with the
shared binder in ``agm.agl.runtime.arguments``, the one place they are
diagnosed. A parameter supplied more than once is diagnosed here instead: the
binder takes a name-keyed mapping and so is entitled to assume its caller
already rejected repetition.

A parameter's *external* name (its declared name, or whatever ``@opt-name``
renames it to) governs every host surface: the flag, its derived negative,
the qualified config key, completion and the undeclared-config-key warning.
The external name stops here: :meth:`ProgramCommand.parse` maps it back to
the declared name, so the shared binder only ever sees declared names.

``RESERVED_FLAGS`` is the host's own flag inventory: the flags ``agm exec``
declares itself (``--help``, ``-p``, ``--program``, ``--agent``, …) union
every engine-key flag. A program parameter can never be projected onto one
of these — :func:`build_program_command` reports the collision instead.

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
therefore a plain usage error raised by :meth:`ProgramCommand.parse` itself,
not a decode diagnostic.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import click
from click.core import ParameterSource

from agm.agl.runtime.arguments import ProgramArguments
from agm.agl.runtime.convert import StrictJsonParseError, parse_json_strict
from agm.agl.runtime.serialize import dumps_exact
from agm.agl.semantics.engine_keys import ENGINE_KEY_TYPES
from agm.agl.semantics.types import BoolType, TextType, is_standard_option_enum
from agm.agl.zones import ParamZone

if TYPE_CHECKING:
    from agm.agl.runtime.types import ProgramDeclInfo, ProgramParamInfo
    from agm.agl.semantics.types import Type as AglType

__all__ = [
    "DuplicateOptionFlagError",
    "ProgramCommand",
    "ProgramOptionError",
    "ProjectedOption",
    "RESERVED_FLAGS",
    "ReservedFlagError",
    "ValueForm",
    "build_program_command",
    "engine_key_flags",
    "native_raw_value",
    "option_none_raw",
    "option_some_raw",
    "program_command_for",
    "program_value_taking_flags",
    "project_option",
    "render_program_arguments_help",
    "short_help_requested",
]

# Zones whose parameter fills a positional CLI slot.
_POSITIONAL_ZONES = (ParamZone.POSITIONAL_ONLY, ParamZone.STANDARD)

# Zones whose parameter is addressable by a ``--name`` flag.
_NAME_ADDRESSABLE_ZONES = (ParamZone.STANDARD, ParamZone.NAMED_ONLY)

# The command parameter every positional token lands in. Click keys its parsed
# values by parameter name, so this and the per-option names below are internal
# identifiers only — a parameter's user-visible spelling lives in its flags.
_POSITIONAL_DEST = "_positionals"


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
    #: Every other type: one strict-JSON token.
    JSON = "json"


@dataclass(frozen=True, slots=True)
class ProjectedOption:
    """One type's CLI surface, as :func:`project_option` derives it.

    ``value_form`` is the one stored discriminant: ``negative_flags`` and
    ``takes_value`` both follow from it, so :attr:`takes_value` is derived
    rather than stored and cannot disagree with the form.

    ``option_inner`` is ``T`` when ``value_form`` is :attr:`ValueForm.OPTION`
    and ``None`` for every other form — computed once here rather than
    re-derived from the parameter's type on every token, so the raw-value
    builder never has to represent the impossible combination.
    """

    flags: tuple[str, ...]
    negative_flags: tuple[str, ...]
    value_form: ValueForm
    option_inner: "AglType | None" = None

    @property
    def takes_value(self) -> bool:
        """Return whether this option consumes a following ``VALUE`` token."""
        return self.value_form is not ValueForm.BOOL


def _option_inner(type_: "AglType") -> "AglType | None":
    """Return ``T`` when *type_* is the standard ``Option[T]``, else ``None``."""
    if is_standard_option_enum(type_) and type_.type_args:
        return type_.type_args[0]
    return None


def project_option(name: str, type_: "AglType") -> ProjectedOption:
    """Project one *name*/*type_* pair (a program parameter or an engine key) onto its CLI flags.

    *name* is the option's external spelling: an engine key's own name, or a
    program parameter's ``@opt-name`` (defaulting to its declared name).
    """
    inner = _option_inner(type_)
    if isinstance(type_, BoolType):
        form = ValueForm.BOOL
    elif inner is not None:
        form = ValueForm.OPTION
    elif isinstance(type_, TextType):
        form = ValueForm.TEXT
    else:
        form = ValueForm.JSON
    negated = form in (ValueForm.BOOL, ValueForm.OPTION)
    return ProjectedOption(
        flags=(f"--{name}",),
        negative_flags=(f"--no-{name}",) if negated else (),
        value_form=form,
        option_inner=inner if form is ValueForm.OPTION else None,
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
    parameters' declared names, in declaration order.
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


def native_raw_value(projected: ProjectedOption, raw: object) -> object:
    """Project one already-native host value onto *projected*'s raw argument shape.

    The config-table counterpart of :func:`_positive_raw`, which applies the
    same ``Option`` rule to a CLI ``VALUE`` token. A present value for an
    ``Option[T]`` parameter is wrapped ``Some``: a config table has no
    ``--no-x`` equivalent, so an absent key supplies nothing at all and the
    parameter falls back to its own default. A native string for a JSON-form
    parameter is encoded into JSON text, preserving its distinction from a
    serialized CLI token; every other value form is already a native
    TOML/JSON value that ``decode_param_value`` decodes directly. Keeping this beside
    :func:`_positive_raw` is what stops the envelope rule from being spelled
    once per host surface.
    """
    if projected.value_form is ValueForm.OPTION:
        return option_some_raw(raw)
    if projected.value_form is ValueForm.JSON and isinstance(raw, str):
        return dumps_exact(raw, indent=None)
    return raw


def _positive_raw(projected: ProjectedOption, flag: str, token: str) -> object:
    """Build the raw value for one value-taking flag's ``VALUE`` token.

    Only ``TEXT``/``JSON``/``OPTION`` forms reach here — the caller
    (:meth:`ProgramCommand.parse`) never calls this for a ``BOOL`` form,
    whose ``takes_value`` is ``False``.
    """
    if projected.option_inner is not None:
        return option_some_raw(_value_from_token(flag, projected.option_inner, token))
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


def _short_flag(param: "ProgramParamInfo") -> str | None:
    """Return *param*'s ``-x`` short spelling, or ``None`` when it declares none."""
    short = param.cli.short
    return None if short is None else f"-{short}"


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


def _click_params(
    index: int, param: "ProgramParamInfo", projected: ProjectedOption
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
    short = _short_flag(param)
    shorts: list[str] = [] if short is None else [short]
    if projected.value_form is ValueForm.BOOL:
        return [
            _ProgramOption(
                [_positive_dest(index), f"--{spec.name}/--no-{spec.name}", *shorts],
                type=click.BOOL,
                default=None,
                multiple=True,
                envvar=spec.env,
                help=spec.doc,
                hidden=spec.hidden,
            )
        ]
    positive = _ProgramOption(
        [_positive_dest(index), f"--{spec.name}", *shorts],
        default=None,
        multiple=True,
        metavar=spec.metavar,
        envvar=spec.env,
        help=spec.doc,
        hidden=spec.hidden,
    )
    if projected.value_form is not ValueForm.OPTION:
        return [positive]
    negative = _ProgramOption(
        [_negative_dest(index), f"--no-{spec.name}"],
        is_flag=True,
        type=click.BOOL,
        default=None,
        multiple=True,
        hidden=spec.hidden,
    )
    return [positive, negative]


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
    """

    command: click.Command
    positional: tuple["ProgramParamInfo", ...]
    options: tuple[tuple["ProgramParamInfo", ProjectedOption], ...]

    def parse(self, tokens: Sequence[str]) -> ProgramArguments:
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

        :raises ValueError: for any Click usage error (unknown option,
            missing value, a value given to a flag), for a parameter supplied
            more than once, for an ``Option[T]`` parameter given both
            polarities on the command line, or for a malformed ``Option``
            ``VALUE`` (see the module docstring).
        """
        try:
            ctx = self.command.make_context(self.command.name, list(tokens))
        except click.UsageError as exc:
            raise ValueError(exc.format_message()) from exc
        values = cast(dict[str, object], ctx.params)

        named: dict[str, object] = {}
        for index, (param, projected) in enumerate(self.options):
            flag = projected.flags[0]
            positive = cast(tuple[object, ...], values[_positive_dest(index)])
            if projected.value_form is ValueForm.BOOL:
                # Both polarities fill one Click parameter, so this single
                # check covers same- and mixed-polarity repetition alike.
                _reject_repetition(positive, flag, projected.negative_flags[0])
                if positive:
                    named[param.name] = cast(bool, positive[0])
                continue
            _reject_repetition(positive, flag)
            if projected.value_form is not ValueForm.OPTION:
                if positive:
                    named[param.name] = _positive_raw(projected, flag, cast(str, positive[0]))
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
            elif positive:
                named[param.name] = _positive_raw(projected, flag, cast(str, positive[0]))
        positional = cast(tuple[str, ...], values[_POSITIONAL_DEST])
        return ProgramArguments(positional=positional, named=named)

    @staticmethod
    def _from_commandline(ctx: click.Context, dest: str) -> bool:
        """Return whether *dest* was filled by a CLI token rather than the environment.

        Click records where each parameter's value came from, which is the
        only way to tell a token from an environment fallback: both arrive as
        an ordinary parsed value.
        """
        return ctx.get_parameter_source(dest) is ParameterSource.COMMANDLINE

    def usage_line(self, program_name: str) -> str:
        """Render one usage line: the program name, its positional slots, then options."""
        parts = [program_name]
        for param in self.positional:
            name = param.cli.name
            parts.append(f"[{name}]" if param.has_default else f"<{name}>")
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
        described = self.option_lines()
        if not described:
            return ""
        return "\n".join(("Options:", *(f"  {line}" for line in described))) + "\n"

    def option_lines(self) -> tuple[str, ...]:
        """Return one description line per name-addressable parameter, unindented.

        The body of :meth:`render_help_section` without its ``Options:``
        header, for a caller that supplies a header of its own — a
        registered command lists its parameters directly under its own
        section, where a nested ``Options:`` would only repeat the heading
        above it.
        """
        described: list[str] = []
        for param, projected in self.options:
            flag_str = "/".join((*projected.flags, *projected.negative_flags))
            if projected.takes_value:
                flag_str = f"{flag_str} VALUE"
            status = "(optional, has default)" if param.has_default else "(required)"
            described.append(f"{flag_str}  {param.type!r}  {status}")
        return tuple(described)

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

    def completion_items(self) -> tuple[str, ...]:
        """Return every completable CLI token: each option's flags and negative forms."""
        items: list[str] = []
        for _param, projected in self.options:
            items.extend(projected.flags)
            items.extend(projected.negative_flags)
        return tuple(items)

    def value_taking_flags(self) -> frozenset[str]:
        """Return every spelling of an option that consumes a following ``VALUE`` token.

        Both the long flag and any ``@opt-short`` spelling. Used to
        disambiguate a bare ``-h`` token from a plausible ``VALUE`` supplied
        to a preceding value-taking flag (a ``text``/JSON-form argument can
        legitimately be the literal string ``-h``) — the *value_flags*
        :func:`short_help_requested` checks against.
        """
        flags: set[str] = set()
        for param, projected in self.options:
            if not projected.takes_value:
                continue
            flags.update(projected.flags)
            short = _short_flag(param)
            if short is not None:
                flags.add(short)
        return frozenset(flags)


def _check_reservation(
    options: "tuple[tuple[ProgramParamInfo, ProjectedOption], ...]",
) -> ProgramOptionError | None:
    """Return the first flag collision among *options*, or ``None`` when there is none.

    Every spelling a parameter claims — its ``--name``, the derived
    ``--no-name`` of a negatable form, and its ``-x`` short — is checked
    against :data:`RESERVED_FLAGS` (the host's full inventory, engine-key
    flags included) and against every other parameter's own spellings, in
    declaration order.
    """
    seen: dict[str, str] = {}
    for param, projected in options:
        spellings = [*projected.flags, *projected.negative_flags]
        short = _short_flag(param)
        if short is not None:
            spellings.append(short)
        for flag in spellings:
            if flag in RESERVED_FLAGS:
                return ReservedFlagError(parameter=param.name, flag=flag)
            if flag in seen:
                return DuplicateOptionFlagError(
                    first_parameter=seen[flag], second_parameter=param.name, flag=flag
                )
            seen[flag] = param.name
    return None


def build_program_command(
    program: "ProgramDeclInfo",
) -> "ProgramCommand | ProgramOptionError":
    """Build *program*'s Click command, checking flag reservation first.

    A program parameter can never shadow an engine key or a built-in host
    option (``--help``, ``--program``, ``-p``, ``--agent``, …), nor another
    parameter's own spelling — so one parameter's ``--no-<name>`` negation
    can never silently steal a different parameter literally named
    ``no-<name>``, and two parameters can never claim the same short.
    Positional-only parameters never enter the flag namespace and so are
    never checked. Returns a :class:`ReservedFlagError` or
    :class:`DuplicateOptionFlagError` for the caller to render on the first
    collision found, in declaration order.

    Help is not rendered from the returned command: it carries the
    program's own ``@doc`` prose and each parameter's, so a host can, but
    ``add_help_option`` stays off — ``--help``/``-h`` belong to ``agm exec``
    itself, which must see them before program tokens are parsed.
    """
    signature = program.parameters
    positional = tuple(p for p in signature if p.kind in _POSITIONAL_ZONES)
    options = tuple(
        (p, project_option(p.cli.name, p.type))
        for p in signature
        if p.kind in _NAME_ADDRESSABLE_ZONES
    )
    collision = _check_reservation(options)
    if collision is not None:
        return collision
    params: list[click.Parameter] = [click.Argument([_POSITIONAL_DEST], nargs=-1)]
    for index, (param, projected) in enumerate(options):
        params.extend(_click_params(index, param, projected))
    command = click.Command(
        name=program.declaration_path,
        params=params,
        help=program.doc,
        add_help_option=False,
    )
    return ProgramCommand(command=command, positional=positional, options=options)


def program_command_for(program: "ProgramDeclInfo | None") -> "ProgramCommand | None":
    """Build *program*'s command, degrading a missing program or a collision to ``None``.

    For the advisory surfaces (help, completion, ``-h`` disambiguation), which
    show no program-argument flags rather than raising the host diagnostic the
    execution path takes from :func:`build_program_command`: no program
    selected and a colliding projection are the same outcome there.
    """
    if program is None:
        return None
    result = build_program_command(program)
    return result if isinstance(result, ProgramCommand) else None


def program_value_taking_flags(program: "ProgramDeclInfo | None") -> frozenset[str]:
    """Return *program*'s value-taking flags, or none when it has no usable command.

    The :func:`short_help_requested` *value_flags* argument, derived in one
    place so every caller that disambiguates a bare ``-h`` against a selected
    program does it identically.
    """
    command = program_command_for(program)
    return frozenset() if command is None else command.value_taking_flags()


def render_program_arguments_help(
    entry_programs: "tuple[ProgramDeclInfo, ...]",
    *,
    selected: "ProgramDeclInfo | None",
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
        command = program_command_for(candidate)
        if command is None:
            continue
        lines.append(f"  Usage: {command.usage_line(candidate.declaration_path)}")
        if selected is not None:
            help_section = command.render_help_section()
            if help_section:
                lines.extend(f"  {line}" for line in help_section.splitlines())
    if len(lines) == 1:
        return ""
    return "\n".join(lines) + "\n"


def short_help_requested(tokens: Sequence[str], *, value_flags: frozenset[str]) -> bool:
    """Return whether an unconsumed ``-h`` occurs in *tokens*.

    *value_flags* names every spelling of a flag that consumes a following
    ``VALUE`` token — a selected program's own
    (:meth:`ProgramCommand.value_taking_flags`) — so a value legitimately
    spelled ``-h`` for one of them is recognized as consumed, not as a
    short-help request.

    A bare ``--`` ends option parsing, so every token from there on is
    positional and a program can legitimately receive ``-h`` as one of its
    own arguments after it.
    """
    consume_value = False
    options_ended = False
    for token in tokens:
        if not options_ended and token == "--":
            options_ended = True
            continue
        if options_ended:
            continue
        if consume_value and not token.startswith("--"):
            consume_value = False
            continue
        consume_value = False
        if token == "-h":
            return True
        if token in value_flags:
            consume_value = True
    return False
