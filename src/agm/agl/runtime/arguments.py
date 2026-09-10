"""Runtime program-argument binding: host-supplied raw values -> checked ``Value``s.

``ProgramArguments`` is what a host (``agm exec``, a registered package
command, the config layer, the e2e harness) hands the pipeline: the raw
positional and named values it collected from its own surface (CLI tokens,
config-table entries, …). :func:`bind_program_arguments` runs those against a
program's parameter signature through the shared zone binder
(``agm.agl.semantics.arguments``), decodes each supplied value through its
``ParamDecoder``, and reports every violation as a pre-execution
:class:`~agm.agl.diagnostics.Diagnostic`, anchored at the offending
parameter's declaration span — or the program's own span when no single
parameter is at fault (an unknown name, or an excess/misdirected positional
argument). A structural violation (unknown name, duplicate, positional
overflow, a positional-only parameter supplied by name) stops binding at the
first one found, since it makes the rest of the binding meaningless; a
missing required parameter and a decode failure both accumulate, so every
offending parameter is reported in one pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never

from agm.agl.diagnostics import Diagnostic, diagnostic_from_span
from agm.agl.ir.nodes import UseDefault
from agm.agl.semantics.arguments import (
    ArgumentBindingError,
    ArgumentBindingErrorKind,
    BindParam,
    bind_arguments,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agm.agl.ir.contracts import ParamDecoder
    from agm.agl.ir.program import ExecutableProgram, IrProgramParam
    from agm.agl.runtime.types import ProgramDeclInfo, ProgramParamInfo
    from agm.agl.semantics.types import Type as AglType
    from agm.agl.semantics.values import Value
    from agm.agl.syntax.spans import SourceSpan
    from agm.agl.zones import ParamZone

__all__ = [
    "ProgramArguments",
    "ProgramParameter",
    "ProgramSignature",
    "bind_program_arguments",
    "bind_program_arguments_for",
    "decode_param_value",
    "default_program_arguments",
]


def _missing_required_message(name: str) -> str:
    """The diagnostic message for a missing required program argument.

    The sole place this wording is generated: every diagnostic that reports
    a missing required program argument, however it is reached, calls this.
    """
    return f"Missing required program argument: {name!r}"


@dataclass(frozen=True, slots=True)
class ProgramArguments:
    """Raw host-supplied values for one program invocation.

    ``positional`` is host-supplied in source/token order; ``named`` maps a
    parameter name to its host-supplied value. Values are plain Python
    objects — JSON-shaped values, strings, or native TOML values — not yet
    decoded against any parameter type.
    """

    positional: tuple[object, ...]
    named: "Mapping[str, object]"


@dataclass(frozen=True, slots=True)
class ProgramParameter:
    """One ``program def`` value parameter, fused for binding and diagnostics.

    ``name``, ``kind``, and ``has_default`` drive zone binding; ``type`` is
    the parameter's checked type; ``span`` is its declaration span, the
    anchor for a diagnostic naming this parameter; ``decoder`` decodes one
    raw host-supplied value into this parameter's checked type.
    """

    name: str
    kind: "ParamZone"
    type: "AglType"
    has_default: bool
    span: "SourceSpan"
    decoder: "ParamDecoder"


@dataclass(frozen=True, slots=True)
class ProgramSignature:
    """A program's parameter signature, ready for binding and diagnostics.

    ``span`` anchors a diagnostic that names no single parameter (an unknown
    name, or an excess/misdirected positional argument). Build with
    :meth:`fuse` from the two independent descriptions a host has on hand —
    never pair ``IrProgramParam``/``ProgramParamInfo`` tuples by hand.
    """

    parameters: "tuple[ProgramParameter, ...]"
    span: "SourceSpan"

    @classmethod
    def fuse(
        cls,
        params: "tuple[IrProgramParam, ...]",
        infos: "tuple[ProgramParamInfo, ...]",
        span: "SourceSpan",
    ) -> "ProgramSignature":
        """Pair each executable parameter with its declaration info, by name.

        *params* (an executable's per-program decoders) and *infos* (a
        program declaration's spans and types) describe the same parameters
        independently. Pairing them by name — rather than assuming matching
        order — means a reordering between the two sources can never produce
        a wrong pairing. Both are built from the same declaration, so a name
        present in one and absent from the other is a compiler bug.
        """
        infos_by_name = {info.name: info for info in infos}
        parameters = []
        for param in params:
            info = infos_by_name.pop(param.name, None)
            assert info is not None, (
                f"compiler bug: program parameter {param.name!r} has a signature "
                "decoder but no declaration info"
            )
            parameters.append(
                ProgramParameter(
                    name=param.name,
                    kind=param.kind,
                    type=info.type,
                    has_default=info.has_default,
                    span=info.span,
                    decoder=param.external_decoder,
                )
            )
        assert not infos_by_name, (
            "compiler bug: program declaration info has parameters absent from its signature: "
            f"{sorted(infos_by_name)}"
        )
        return cls(parameters=tuple(parameters), span=span)


def agent_raw_value(text: str) -> object:
    """Convert Agent host text to the raw tagged shape used by AgL decoding."""
    from agm.agent.values import agent_spec_shape, parse_agent_text
    from agm.agl.runtime.convert import StrictJsonParseError, parse_json_strict

    try:
        parsed = parse_json_strict(text)
    except StrictJsonParseError:
        parsed = None
    return parsed if isinstance(parsed, dict) else agent_spec_shape(parse_agent_text(text))


def decode_param_value(decoder: "ParamDecoder", raw: object) -> "Value":
    """Decode a raw host param value against *decoder* into a typed ``Value``.

    The single decode path shared by program-argument binding
    (:func:`bind_program_arguments`) and the host engine-config decode path
    (``runtime.engine_config.convert_host_value``). ``text`` params are taken
    verbatim. Standard ``Agent`` params accept canonical tagged JSON, compact
    native-agent syntax, or a command string. Every other value crosses the
    canonical JSON boundary (strict parse, integral-decimal normalization,
    JSON-Schema validation, then the typeless ``decode_value`` walk).

    :raises StrictJsonParseError: if a textual/native value is not strict JSON.
    :raises ValueError: on a type/shape mismatch or schema-validation failure.
    """
    from agm.agl.runtime.convert import (
        _clean_validation_message,
        decode_value,
        normalize_integral_decimals,
        parse_json_strict,
        validator_for_schema,
    )
    from agm.agl.runtime.serialize import dumps_exact

    if decoder.text_verbatim:
        if not isinstance(raw, str):
            raise ValueError(f"expected a text value (str), got {type(raw).__name__}")
        obj: object = raw
    elif decoder.agent_text and isinstance(raw, str):
        obj = agent_raw_value(raw)
    elif isinstance(raw, str):
        obj = parse_json_strict(raw)
    elif _is_json_shaped(raw):
        # Native host values cross the same canonical JSON boundary as textual
        # values. In particular, Python floats become Decimal through
        # parse_float=Decimal before typed decoding.
        obj = parse_json_strict(dumps_exact(raw, indent=None))
    else:
        raise ValueError(f"expected a JSON-compatible value, got {type(raw).__name__}")
    normalized = normalize_integral_decimals(obj)
    validation_errors = list(validator_for_schema(decoder.json_schema).iter_errors(normalized))
    if validation_errors:
        raise ValueError(_clean_validation_message(validation_errors[0]))
    return decode_value(decoder.decode, normalized, dict(decoder.defs))


@dataclass(frozen=True, slots=True)
class _Supplied:
    """Box a raw host argument so ``None`` keeps meaning "not supplied".

    ``bind_arguments`` is generic over an opaque item type and returns
    ``None`` for a parameter slot the caller must default; a host may
    legitimately *supply* ``None`` itself (JSON ``null`` for a ``json``-typed
    parameter), so every raw value is boxed before it enters the pure binder
    and unboxed on the way out.  An absent ``Option[T]`` is not such a case:
    it crosses as the ``{"$case": "None"}`` envelope, never a bare ``None``.
    """

    value: object


def bind_program_arguments(
    signature: ProgramSignature, arguments: ProgramArguments
) -> "tuple[tuple[Value | UseDefault, ...], tuple[Diagnostic, ...]]":
    """Bind and decode *arguments* against *signature*.

    Runs the shared zone binder (positional-greedy; a host supplies plain
    values rather than AST expressions, so no item is ever treated as a bare
    name) with every parameter's default suppressed, so a missing required
    parameter never stops the binder early; it is instead reported below,
    alongside every other missing required parameter. A structural
    violation (unknown name, duplicate, positional overflow, or a
    positional-only parameter supplied by name) still stops binding
    immediately, since it makes the rest of the binding meaningless.

    Returns ``(values, ())`` on success, where *values* is ready for
    ``IrInterpreter.run``'s ``arguments`` (one entry per parameter, in
    declaration order: a decoded ``Value``, or ``UseDefault`` for an omitted
    defaulted parameter) — a zero-parameter program returns ``((), ())``, so
    a caller must key success off the diagnostics tuple, not off whether
    values is non-empty. Returns ``((), diagnostics)`` on any binding or
    decode failure — one diagnostic for a structural violation, or one per
    parameter that is missing-and-required or failed to decode (both keep
    checking every parameter rather than stopping at the first).
    """
    from agm.agl.runtime.convert import StrictJsonParseError

    bind_params = [
        BindParam(name=p.name, kind=p.kind, has_default=True) for p in signature.parameters
    ]
    boxed_positional = [_Supplied(value) for value in arguments.positional]
    boxed_named = [(name, _Supplied(value)) for name, value in arguments.named.items()]
    try:
        bound = bind_arguments(bind_params, boxed_positional, boxed_named)
    except ArgumentBindingError as exc:
        return (), (_diagnose_binding_error(exc, signature),)

    values: "list[Value | UseDefault]" = []
    diagnostics: list[Diagnostic] = []
    for index, (param, box) in enumerate(zip(signature.parameters, bound, strict=True)):
        if box is None:
            if param.has_default:
                values.append(UseDefault(param_index=index))
            else:
                diagnostics.append(
                    diagnostic_from_span(_missing_required_message(param.name), param.span)
                )
            continue
        try:
            values.append(decode_param_value(param.decoder, box.value))
        except (StrictJsonParseError, ValueError) as exc:
            diagnostics.append(
                diagnostic_from_span(
                    f"Program argument {param.name!r}: could not parse as "
                    f"{param.decoder.target_type_label}: {exc}",
                    param.span,
                )
            )
    if diagnostics:
        return (), tuple(diagnostics)
    return tuple(values), ()


def bind_program_arguments_for(
    executable: "ExecutableProgram", program: "ProgramDeclInfo", arguments: ProgramArguments
) -> "tuple[tuple[Value | UseDefault, ...], tuple[Diagnostic, ...]]":
    """Fuse *program*'s declaration info with *executable*'s signature, then bind *arguments*.

    The shared tail of ``PipelineDriver.preflight_arguments`` and
    ``commands.exec_program.run``: both hold an already-lowered
    ``ExecutableProgram`` and the ``ProgramDeclInfo`` for the same selected
    program, and both need a :class:`ProgramSignature` fused from them before
    calling :func:`bind_program_arguments` — this is the one place that
    fusing happens, so the two hosts can never pair the two descriptions
    differently.
    """
    program_symbol = executable.program_symbols[program.node_id]
    signature = ProgramSignature.fuse(
        executable.program_signatures[program_symbol], program.parameters, program.span
    )
    return bind_program_arguments(signature, arguments)


def default_program_arguments(
    signature: "tuple[IrProgramParam, ...]",
) -> "tuple[tuple[Value | UseDefault, ...], tuple[Diagnostic, ...]]":
    """Derive an argument list for a program invoked with no host-supplied values.

    Used when a caller runs a selected program without going through
    :func:`bind_program_arguments` — no ``ProgramArguments`` was ever
    collected, only the lowered signature is on hand. Every parameter defers
    to its own default (``UseDefault``); a parameter with none is reported
    with the same message :func:`bind_program_arguments` uses for an
    explicitly missing argument, anchored at line 1 since no declaration span
    is reachable from an ``IrProgramParam`` alone.

    Returns ``(values, ())`` on success — one entry per parameter, in
    declaration order — or ``((), diagnostics)``, one diagnostic per missing
    required parameter.
    """
    diagnostics = tuple(
        Diagnostic(message=_missing_required_message(param.name), line=1)
        for param in signature
        if param.required
    )
    if diagnostics:
        return (), diagnostics
    return tuple(UseDefault(param_index=i) for i in range(len(signature))), ()


def _diagnose_binding_error(exc: ArgumentBindingError, signature: ProgramSignature) -> Diagnostic:
    """Translate one structural zone-binding violation into a pre-execution diagnostic."""
    match exc.kind:
        case ArgumentBindingErrorKind.MISSING_REQUIRED:  # pragma: no cover
            # Unreachable: `bind_program_arguments` always passes
            # `has_default=True`, so the binder never short-circuits on a
            # missing required parameter — they accumulate and are reported
            # per parameter there instead. Listed only for exhaustiveness.
            raise AssertionError("binder never short-circuits on a missing required parameter")
        case ArgumentBindingErrorKind.UNKNOWN_NAME:
            assert exc.name is not None, "binder always names an unknown argument"
            return diagnostic_from_span(f"Unknown program argument: {exc.name!r}", signature.span)
        case ArgumentBindingErrorKind.POSITIONAL_ONLY_BY_NAME:
            assert exc.name is not None, "binder always names a positional-only argument"
            return diagnostic_from_span(
                f"Program argument {exc.name!r} is positional-only and cannot be supplied by name",
                _span_for(signature, exc.name),
            )
        case ArgumentBindingErrorKind.DUPLICATE:
            assert exc.name is not None, "binder always names a duplicate argument"
            return diagnostic_from_span(
                f"Program argument {exc.name!r} supplied more than once",
                _span_for(signature, exc.name),
            )
        case ArgumentBindingErrorKind.TOO_MANY_POSITIONAL:
            return diagnostic_from_span("Too many positional program arguments", signature.span)
        case ArgumentBindingErrorKind.POSITIONAL_IN_NAMED_ONLY:
            return diagnostic_from_span(
                "Remaining program arguments must be supplied by name", signature.span
            )
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _span_for(signature: ProgramSignature, name: str) -> "SourceSpan":
    """Return the declaration span of the parameter named *name*."""
    return next(p for p in signature.parameters if p.name == name).span


def _is_json_shaped(obj: object) -> bool:
    """Return ``True`` iff *obj* is a JSON-compatible Python value.

    The closed set: ``None``, ``bool``, ``int``, ``float``,
    ``decimal.Decimal``, ``str``, ``list`` (elements recursively JSON-shaped),
    and ``dict`` (str keys, values recursively JSON-shaped).

    Used by :func:`decode_param_value` to detect non-JSON-shaped host objects
    (e.g. sets or custom classes) before attempting serialisation, so the
    caller can emit a clean diagnostic instead of a cryptic traceback.
    """
    import decimal as _decimal_mod

    if obj is None or isinstance(obj, (bool, int, float, str, _decimal_mod.Decimal)):
        return True
    if isinstance(obj, list):
        return all(_is_json_shaped(e) for e in obj)
    if isinstance(obj, dict):
        return all(isinstance(k, str) and _is_json_shaped(v) for k, v in obj.items())
    return False
