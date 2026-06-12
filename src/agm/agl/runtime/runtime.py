"""WorkflowRuntime — the public façade for the AgL host runtime.

Drives the full ``parse → scope → typecheck → host-prep → eval`` pipeline:
registers agents/codecs/renderers, validates host inputs, materializes output
contracts, and executes the program (or stops after static checking for
``agm exec --dry-run``).  Structured outputs use the JSON codec with
lenient-by-default recovery (design §2.8).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agm.agl.diagnostics import Diagnostic
from agm.agl.runtime.agents import AgentFn

if TYPE_CHECKING:
    from pathlib import Path

    from agm.agl.eval.values import ExceptionValue, Value
    from agm.agl.runtime.codec import OutputCodec
    from agm.agl.runtime.render import RendererFn
    from agm.agl.typecheck.env import CheckedProgram as CheckedProgramType
    from agm.agl.typecheck.types import Type as AglType

# Reserved agent names: cannot be registered by callers.
_RESERVED_AGENT_NAMES: frozenset[str] = frozenset({"prompt", "exec"})

# The full v1 vocabulary of semantic type *kinds* (matches ``Type.kind`` for
# every member of the ``Type`` union in ``agm.agl.typecheck.types``).  Used to
# validate a renderer's ``supported_types`` capability descriptor (F6, §9.1).
ALL_TYPE_KINDS: frozenset[str] = frozenset(
    {
        "text",
        "json",
        "bool",
        "int",
        "decimal",
        "list",
        "dict",
        "record",
        "enum",
        "exception",
    }
)


@dataclass(frozen=True, slots=True)
class CallSiteInfo:
    """Static summary of one agent-call or exec site (--dry-run inventory).

    ``callee``        Agent or executor name (``"prompt"``, ``"exec"``, or a
                      registered agent name).
    ``target_type``   The target type name (e.g. ``"text"``, ``"Review"``).
    ``codec_name``    Selected codec (``"text"`` or ``"json"``).
    ``has_schema``    ``True`` when the contract carries a JSON Schema.
    ``parse_policy``  ``"abort"`` / ``"retry[N]"`` / ``"default"``.
    ``line``          1-based source line of the call site.
    ``col``           1-based source column of the call site.
    """

    callee: str
    target_type: str
    codec_name: str
    has_schema: bool
    parse_policy: str
    line: int
    col: int


@dataclass(frozen=True, slots=True)
class RunError:
    """Structured representation of an uncaught AgL exception.

    ``type_name`` is the exception's declared type name (e.g. ``"AgentParseError"``).
    ``fields`` is a mapping from field names to JSON-shaped Python values.
    ``line`` is the 1-based source line of the raise site when known (design
    §12.6: source location is part of runtime error reporting); ``None`` when
    the span was not threaded through (e.g. arithmetic errors inside expressions).
    ``col`` is the 1-based source column of the raise site; ``None`` when unknown.
    """

    type_name: str
    fields: dict[str, object]
    line: int | None = None
    col: int | None = None


@dataclass(slots=True)
class RunResult:
    """Result of a ``WorkflowRuntime.run`` call.

    ``ok``
        ``True`` iff there are no error-severity ``diagnostics`` **and** no
        uncaught AgL exception.  ``warnings`` never affect ``ok``.
    ``diagnostics``
        Pre-execution FAILURES only: error-severity items from
        lex/parse/scope/typecheck/input-validation.  Each entry has a
        ``.message`` (str) and a ``.line`` (int, 1-based).  Warnings are a
        SEPARATE channel and NEVER appear here; on a successful run this list is
        empty.
    ``warnings``
        Advisory warning-severity diagnostics (e.g. non-exhaustive ``case``)
        surfaced on EVERY path — success, static failure, input-validation
        failure, and uncaught exception.  Same ``Diagnostic`` type as
        ``diagnostics`` but with ``.severity == "warning"``.  Reported to the
        user but never cause the run to fail (never affect ``ok``).
    ``error``
        The uncaught AgL exception, or ``None``.  Set only when the program
        *started* executing but ended with an unhandled exception (exit code 2
        per the CLI contract).  ``None`` for pre-execution failures and for
        successful runs.
    ``bindings``
        Root-scope bindings after a successful run (name → Value).  Empty for
        failed runs.
    ``call_sites``
        Static call-site inventory populated when ``check_only=True``
        (``agm exec --dry-run``).  One entry per agent-call/exec site in
        source order.  Empty for ordinary runs.
    ``trace_path``
        Path of the JSONL trace file written during this run, or ``None``
        when logging was disabled (``--no-log``) or the run was a dry-run.
        This is the handle referred to in plan §8.3.
    """

    ok: bool
    diagnostics: list[Diagnostic]
    error: RunError | None
    warnings: list[Diagnostic] = field(default_factory=list)
    bindings: dict[str, Value] = field(default_factory=dict)
    call_sites: tuple[CallSiteInfo, ...] = field(default_factory=tuple)
    trace_path: Path | None = field(default=None)


class WorkflowRuntime:
    """Host API for the AgL interpreter.

    Constructor parameters
    ----------------------
    default_loop_limit : int
        Default iteration bound for ``do[N]`` loops (design §2.11).
    default_strict_json : bool
        When ``True`` the JSON codec defaults to strict parsing (only a bare
        JSON value with surrounding whitespace is accepted).  The default
        ``False`` enables lenient JSON recovery (design §2.8, Q3).
    default_agent : callable or None
        The callable used for the built-in ``prompt`` agent.  ``None`` means
        no default agent is configured (only explicitly registered agents will
        be available).
    shell_exec_timeout : float or None
        Idle timeout (in seconds) applied to every ``exec`` shell call (design
        §4.12, §11.13).  ``None`` means no timeout (the shell command may run
        indefinitely).  This is the ``[exec] timeout`` config value, threaded
        in from the CLI (plan §10.3).
    """

    def __init__(
        self,
        *,
        default_loop_limit: int = 5,
        default_strict_json: bool = False,
        default_agent: AgentFn | None = None,
        shell_exec_timeout: float | None = None,
    ) -> None:
        self._default_loop_limit = default_loop_limit
        self._default_strict_json = default_strict_json
        self._default_agent = default_agent
        self._shell_exec_timeout = shell_exec_timeout
        self._agents: dict[str, AgentFn] = {}
        # Extra codecs/renderers registered by the host (beyond the built-ins).
        self._extra_codecs: dict[str, "OutputCodec"] = {}
        self._extra_renderers: dict[str, "RendererFn"] = {}
        # Per-renderer supported type kinds (``None`` → type-agnostic / all kinds).
        self._extra_renderer_kinds: dict[str, frozenset[str] | None] = {}

    def register_agent(self, name: str, fn: AgentFn) -> None:
        """Register a named agent callable.

        Raises ``ValueError`` if ``name`` is a reserved name (``prompt`` or
        ``exec``) or if an agent with that name has already been registered.
        """
        if name in _RESERVED_AGENT_NAMES:
            raise ValueError(
                f"Cannot register agent with reserved name {name!r}. "
                f"Reserved names: {sorted(_RESERVED_AGENT_NAMES)}"
            )
        if name in self._agents:
            raise ValueError(
                f"An agent named {name!r} is already registered. "
                "Duplicate registrations are not allowed."
            )
        self._agents[name] = fn

    def register_codec(self, codec: "OutputCodec") -> None:
        """Register a custom output codec.

        The codec's ``name`` property is the registration key.  The built-in
        codec names (``"text"`` and ``"json"``) are reserved and cannot be
        overridden.  Duplicate registrations (same name, regardless of
        implementation) are rejected.

        The codec must expose ``supported_kinds: frozenset[str]``; those kinds
        are surfaced in ``HostCapabilities.codec_kinds`` so the type-checker
        can validate ``format`` options at a call site (design §7.6 / CARRY-IN 1).

        Raises ``ValueError`` for reserved or duplicate names.
        """
        from agm.agl.runtime.codec import JsonCodec, TextCodec

        _reserved_codec_names: frozenset[str] = frozenset({
            TextCodec().name, JsonCodec().name
        })
        name = codec.name
        if name in _reserved_codec_names:
            raise ValueError(
                f"Cannot register codec with reserved name {name!r}. "
                f"Reserved codec names: {sorted(_reserved_codec_names)}"
            )
        if name in self._extra_codecs:
            raise ValueError(
                f"A codec named {name!r} is already registered. "
                "Duplicate codec registrations are not allowed."
            )
        self._extra_codecs[name] = codec

    def register_renderer(
        self,
        name: str,
        fn: "RendererFn",
        *,
        supported_types: "frozenset[str] | None" = None,
    ) -> None:
        """Register a custom renderer function (§2.12 / design §7.6, plan §9.1).

        *name* is the ``as <name>`` identifier used in prompt interpolation
        segments.  Built-in renderer names (``"default"``, ``"raw"``,
        ``"json"``, ``"bullets"``) are reserved.  Duplicate names are
        rejected.

        *fn* must match the ``RendererFn`` signature:
        ``Callable[[Value, str | None], str]``.

        *supported_types* is the renderer's capability descriptor: a frozenset
        of semantic type **kinds** (the same vocabulary as a codec's
        ``supported_kinds`` — ``"text"``, ``"int"``, ``"list"``, ``"record"``,
        …).  The type-checker rejects ``${x as <name>}`` when ``x``'s kind is
        not in this set (F6, plan §9.1).  ``None`` (the default) means the
        renderer is type-agnostic and accepts every kind, matching the built-in
        renderers.

        Raises ``ValueError`` for reserved or duplicate names, or for an
        unknown type kind in *supported_types*.
        """
        from agm.agl.runtime.render import RENDERER_NAMES

        if name in RENDERER_NAMES:
            raise ValueError(
                f"Cannot register renderer with reserved name {name!r}. "
                f"Reserved renderer names: {sorted(RENDERER_NAMES)}"
            )
        if name in self._extra_renderers:
            raise ValueError(
                f"A renderer named {name!r} is already registered. "
                "Duplicate renderer registrations are not allowed."
            )
        if supported_types is not None:
            unknown = supported_types - ALL_TYPE_KINDS
            if unknown:
                raise ValueError(
                    f"Renderer {name!r} declares unknown type kind(s) "
                    f"{sorted(unknown)}. Valid kinds: {sorted(ALL_TYPE_KINDS)}."
                )
        self._extra_renderers[name] = fn
        self._extra_renderer_kinds[name] = supported_types

    def run(
        self,
        source: str,
        *,
        inputs: Mapping[str, object] | None = None,
        check_only: bool = False,
        log_file: "Path | None" = None,
    ) -> RunResult:
        """Parse, analyse, and (unless ``check_only``) execute an AgL program.

        Pipeline:
            parse → resolve → check (with HostCapabilities) →
            validate inputs → materialize contracts → eval

        When ``check_only`` is ``True`` (``agm exec --dry-run``) the runtime
        runs the full static pipeline, input validation, and contract
        materialization, then STOPS before executing any statement: a clean
        program returns ``ok=True`` with no bindings and produces no program
        output; static/input errors still return ``ok=False``.  On a clean
        ``check_only`` run the §10.1 static call-site inventory is populated on
        ``RunResult.call_sites`` (printed by ``agm exec --dry-run``).

        ``log_file`` is the path of the JSONL trace file to write.  When
        ``None`` (the default) no trace is written.  Dry-run (``check_only``)
        never writes a trace regardless of *log_file* (plan §10.1: dry-run
        is side-effect-free).

        Returns a ``RunResult`` capturing the outcome.
        """
        if inputs is None:
            inputs = {}

        # ----------------------------------------------------------------
        # Build HostCapabilities from registrations.
        # (CARRY-IN 1: codec_kinds and renderer_names are derived from the
        # actual codec/renderer registries, not from duplicated constants.)
        # ----------------------------------------------------------------
        from agm.agl.capabilities import HostCapabilities
        from agm.agl.runtime.agents import AgentRegistry
        from agm.agl.runtime.codec import JsonCodec, TextCodec
        from agm.agl.runtime.render import builtin_renderers

        text_codec = TextCodec()
        json_codec = JsonCodec()

        # Merge built-in codecs with any host-registered extras.
        all_codecs: dict[str, "OutputCodec"] = {
            text_codec.name: text_codec,
            json_codec.name: json_codec,
            **self._extra_codecs,
        }

        # Merge built-in renderers with any host-registered extras.  This single
        # table is the authoritative source for BOTH the static capability
        # descriptors (names + supported kinds) AND the interpolation rendering
        # at eval time, so a registered renderer is actually invoked (F1, M3b).
        all_renderers: dict[str, "RendererFn"] = {
            **builtin_renderers(),
            **self._extra_renderers,
        }
        # Built-in renderers are type-agnostic (``None`` → all kinds); custom
        # renderers carry the kinds declared at registration (F6, plan §9.1).
        renderer_kinds: dict[str, frozenset[str] | None] = {
            name: None for name in builtin_renderers()
        }
        renderer_kinds.update(self._extra_renderer_kinds)

        registry = AgentRegistry(
            named={name: fn for name, fn in self._agents.items()},
            default_agent=self._default_agent,
        )
        capabilities = HostCapabilities(
            agent_names=registry.agent_names,
            has_fallback_agent=registry.has_fallback,
            has_default_agent=registry.has_default_agent,
            supports_shell_exec=True,
            codec_kinds={
                name: codec.supported_kinds for name, codec in all_codecs.items()
            },
            renderer_names=frozenset(all_renderers),
            renderer_kinds=renderer_kinds,
        )

        # ----------------------------------------------------------------
        # [1] Parse
        # ----------------------------------------------------------------
        from agm.agl.parser import AglSyntaxError, parse_program

        try:
            program = parse_program(source)
        except AglSyntaxError as exc:
            return RunResult(
                ok=False,
                diagnostics=[exc.to_diagnostic()],
                error=None,
            )
        except Exception as exc:
            return RunResult(
                ok=False,
                diagnostics=[Diagnostic(message=str(exc), line=1)],
                error=None,
            )

        # ----------------------------------------------------------------
        # [2] Scope / name resolution
        # ----------------------------------------------------------------
        from agm.agl.scope import AglScopeError, resolve

        try:
            resolved = resolve(program)
        except AglScopeError as exc:
            return RunResult(
                ok=False,
                diagnostics=[exc.to_diagnostic()],
                error=None,
            )
        except Exception as exc:
            return RunResult(
                ok=False,
                diagnostics=[Diagnostic(message=f"Scope error: {exc}", line=1)],
                error=None,
            )

        # ----------------------------------------------------------------
        # [3] Type checking
        # ----------------------------------------------------------------
        from agm.agl.typecheck import AglTypeError, check

        try:
            checked = check(resolved, capabilities)
        except AglTypeError as exc:
            return RunResult(
                ok=False,
                diagnostics=[exc.to_diagnostic()],
                error=None,
            )
        except Exception as exc:
            return RunResult(
                ok=False,
                diagnostics=[Diagnostic(message=f"Type error: {exc}", line=1)],
                error=None,
            )

        # Collect warnings from typecheck.
        warnings: list[Diagnostic] = list(checked.warnings)

        # ----------------------------------------------------------------
        # [4] Validate host inputs against input declarations
        # ----------------------------------------------------------------
        from agm.agl.syntax.nodes import InputDecl

        # Build declared input map.  Read the exact binding type recorded by the
        # checker (keyed by the InputDecl node_id) rather than re-resolving the
        # annotation here — the checker is the single source of truth and already
        # handles compound types (list/dict/record/enum) correctly.
        declared_inputs: dict[str, AglType] = {}  # name → declared Type
        for stmt in program.body:
            if isinstance(stmt, InputDecl):
                input_type = checked.type_env.get_binding_type(stmt.node_id)
                if input_type is None:
                    raise AssertionError(
                        f"Input {stmt.name!r} has no recorded binding type; "
                        "checker invariant violated."
                    )
                declared_inputs[stmt.name] = input_type

        # Validate: check for missing and undeclared.
        input_errors: list[Diagnostic] = []
        provided_keys = set(inputs.keys())
        declared_keys = set(declared_inputs.keys())

        for name in declared_keys - provided_keys:
            input_errors.append(
                Diagnostic(
                    message=f"Missing declared input: {name!r}",
                    line=1,
                )
            )
        for name in provided_keys - declared_keys:
            input_errors.append(
                Diagnostic(
                    message=f"Undeclared input: {name!r} was provided but not declared",
                    line=1,
                )
            )

        if input_errors:
            return RunResult(
                ok=False,
                diagnostics=input_errors,
                error=None,
                warnings=list(warnings),
            )

        # ----------------------------------------------------------------
        # [5] Materialize output contracts (text codec only in M1)
        # ----------------------------------------------------------------
        from agm.agl.runtime.contract import materialize_contract

        # Reuse the merged codec map built for HostCapabilities above.
        codecs = all_codecs
        contracts: dict[int, object] = {}
        contract_errors: list[Diagnostic] = []

        for node_id, spec in checked.contract_specs.items():
            try:
                contracts[node_id] = materialize_contract(spec, codecs)
            except ValueError as exc:
                contract_errors.append(
                    Diagnostic(message=f"Contract error: {exc}", line=1)
                )

        if contract_errors:
            return RunResult(
                ok=False,
                diagnostics=contract_errors,
                error=None,
                warnings=list(warnings),
            )

        # ----------------------------------------------------------------
        # [6] Build root scope from inputs + type-check declarations
        # ----------------------------------------------------------------
        from agm.agl.eval.scope import Scope

        root_scope = Scope(parent=None)
        input_bind_errors: list[Diagnostic] = []

        for stmt in program.body:
            if isinstance(stmt, InputDecl):
                raw_val = inputs[stmt.name]
                input_type_obj = declared_inputs[stmt.name]
                # Convert/validate the raw value.
                try:
                    typed_val = _convert_input(stmt.name, raw_val, input_type_obj)
                except ValueError as exc:
                    input_bind_errors.append(
                        Diagnostic(message=str(exc), line=stmt.span.start_line)
                    )
                    continue
                root_scope.define(
                    stmt.name, typed_val, mutable=False, decl_span=stmt.span
                )

        if input_bind_errors:
            return RunResult(
                ok=False,
                diagnostics=input_bind_errors,
                error=None,
                warnings=list(warnings),
            )

        # ----------------------------------------------------------------
        # [check_only] --dry-run stop: the full static pipeline, input
        # validation, and contract materialization have all succeeded.  Stop
        # before executing any statement (no program output, no side effects).
        # Build the §10.1 static call-site inventory before returning.
        # Dry-run is side-effect-free: no trace is written (plan §10.1).
        # ----------------------------------------------------------------
        if check_only:
            inventory = _build_call_inventory(checked, contracts)
            return RunResult(
                ok=True,
                diagnostics=[],
                error=None,
                warnings=list(warnings),
                bindings={},
                call_sites=tuple(inventory),
                trace_path=None,
            )

        # ----------------------------------------------------------------
        # [7] Build and run the interpreter
        # ----------------------------------------------------------------
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.runtime.contract import OutputContract
        from agm.agl.runtime.trace import TraceStore

        # Create the trace store for this run.  When log_file is None the
        # store is a no-op and no file is touched.
        trace = TraceStore(path=log_file)
        if log_file is not None:
            from agm.core.fs import mkdir

            mkdir(log_file.parent, parents=True, exist_ok=True)
        trace.run_start()

        typed_contracts: dict[int, OutputContract] = {
            nid: c for nid, c in contracts.items() if isinstance(c, OutputContract)
        }

        interp = Interpreter(
            checked=checked,
            registry=registry,
            contracts=typed_contracts,
            type_env=checked.type_env,
            renderers=all_renderers,
            loop_limit=self._default_loop_limit,
            strict_json=self._default_strict_json,
            source=source,
            shell_exec_timeout=self._shell_exec_timeout,
            trace=trace,
        )

        try:
            interp.execute(root_scope)
        except AglRaise as exc:
            # Uncaught AgL exception (exit code 2 per the CLI contract).
            # ONLY the AgL exception carrier is caught here: an unexpected Python
            # exception is an interpreter bug and must propagate (crash loudly)
            # rather than masquerade as a user-facing pre-execution diagnostic.
            error = _exception_value_to_run_error(exc.exc, span=exc.span)
            # Record the uncaught exception in the trace (design §12.6: include
            # the source span when the raise site threaded it through AglRaise).
            trace_id = str(error.fields.get("trace_id", ""))
            trace.exception(
                type_name=error.type_name,
                message=str(error.fields.get("message", "")),
                trace_id=trace_id,
                span=exc.span,
            )
            trace.run_end(ok=False)
            return RunResult(
                ok=False,
                diagnostics=[],
                error=error,
                warnings=list(warnings),
                bindings={},
                trace_path=log_file,
            )

        # Successful run: snapshot root bindings.
        root_bindings = root_scope.snapshot()
        trace.run_end(ok=True)

        return RunResult(
            ok=True,
            diagnostics=[],
            error=None,
            warnings=list(warnings),
            bindings=root_bindings,
            trace_path=log_file,
        )

    @property
    def default_loop_limit(self) -> int:
        """Default iteration bound for ``do[N]`` loops."""
        return self._default_loop_limit

    @property
    def default_strict_json(self) -> bool:
        """Whether strict JSON parsing is the default."""
        return self._default_strict_json

    @property
    def shell_exec_timeout(self) -> float | None:
        """Idle timeout in seconds for ``exec`` shell calls (``None`` = no timeout)."""
        return self._shell_exec_timeout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _convert_input(name: str, raw: object, type_obj: "AglType") -> "Value":
    """Convert a raw host input value to the declared AgL type.

    Supported types:
    - ``text``: verbatim (the value must already be a ``str``).
    - ``int``/``decimal``/``bool``/``json``: parsed via stdlib ``json`` with
      ``parse_float=Decimal`` (design §5.1: no binary floats) and validated.
    - ``list``/``dict``/``record``/``enum``: parsed from a JSON string via the
      ``JsonCodec`` (M2+).
    """
    import decimal as _decimal

    from agm.agl.eval.values import (
        BoolValue,
        DecimalValue,
        IntValue,
        JsonValue,
        TextValue,
    )
    from agm.agl.typecheck.types import (
        BoolType,
        DecimalType,
        DictType,
        EnumType,
        IntType,
        JsonType,
        ListType,
        RecordType,
        TextType,
    )

    # Text: verbatim.
    if isinstance(type_obj, TextType):
        if not isinstance(raw, str):
            raise ValueError(
                f"Input {name!r}: expected a text value (str), got {type(raw).__name__}"
            )
        return TextValue(raw)

    # Structured types (list/dict/record/enum): delegate to JsonCodec.
    if isinstance(type_obj, (ListType, DictType, RecordType, EnumType)):
        import decimal as _decimal_mod

        from agm.agl.runtime.codec import JsonCodec
        from agm.agl.runtime.schema import derive_schema

        # Accept either a JSON string or a Python native object (list/dict)
        # that was already parsed from JSON (e.g. by the e2e test harness).
        if isinstance(raw, str):
            json_str = raw
        elif isinstance(raw, (dict, list, bool, int, _decimal_mod.Decimal, float)):
            # Serialise native Python object to JSON string so the codec can
            # validate and convert it using the full type-aware path.
            json_str = json.dumps(raw, default=str)
        else:
            raise ValueError(
                f"Input {name!r} has type {type_obj!r}; structured inputs must be "
                "provided as a JSON string or a JSON-compatible Python value."
            )
        codec = JsonCodec()
        # Precompute schema once (CARRY-IN 2: avoids re-derivation inside parse).
        schema = derive_schema(type_obj)
        # Host-supplied --input values are not chatty agent output: they must be
        # exactly one bare JSON value (F7).  Strict parsing avoids json-repair
        # silently "fixing" user typos.
        result = codec.parse(json_str, type_obj, strict_json=True, schema=schema)
        if not result.ok or result.value is None:
            raise ValueError(
                f"Input {name!r}: could not parse as {type_obj!r}; structured "
                f"inputs must be exactly one valid JSON value: {result.error_msg}"
            )
        return result.value

    # Scalar non-text (int/decimal/bool/json): parse from JSON if given as string.
    if not isinstance(type_obj, (IntType, DecimalType, BoolType, JsonType)):
        raise ValueError(
            f"Input {name!r} has unsupported type {type_obj!r}."
        )

    value = raw
    if isinstance(value, str):
        try:
            value = json.loads(value, parse_float=_decimal.Decimal)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Input {name!r}: could not parse as JSON: {exc}"
            ) from exc

    if isinstance(type_obj, IntType):
        if isinstance(value, int) and not isinstance(value, bool):
            return IntValue(value)
        if isinstance(value, _decimal.Decimal) and value == int(value):
            return IntValue(int(value))
        raise ValueError(
            f"Input {name!r}: expected an integer, got {type(value).__name__} {value!r}"
        )

    if isinstance(type_obj, DecimalType):
        if isinstance(value, _decimal.Decimal):
            return DecimalValue(value)
        if isinstance(value, int) and not isinstance(value, bool):
            return DecimalValue(_decimal.Decimal(value))
        raise ValueError(
            f"Input {name!r}: expected a decimal, got {type(value).__name__} {value!r}"
        )

    if isinstance(type_obj, BoolType):
        if isinstance(value, bool):
            return BoolValue(value)
        raise ValueError(
            f"Input {name!r}: expected a bool, got {type(value).__name__} {value!r}"
        )

    # JsonType: accept any parsed JSON value.
    return JsonValue(value)


def _exception_value_to_run_error(
    exc: "ExceptionValue",
    *,
    span: "object" = None,  # SourceSpan | None — avoids import cycle
) -> RunError:
    """Convert an ``ExceptionValue`` to a ``RunError`` for ``RunResult``.

    Field values are converted via the shared serializer, which preserves
    ``Decimal`` exactness (never routed through binary ``float``; design §5.1).

    *span* is the optional raise-site source span threaded from ``AglRaise``
    (design §12.6); when present, ``RunError.line`` and ``RunError.col`` are
    populated from it so the CLI can include the source location in its
    exit-2 error output.
    """
    from agm.agl.runtime.serialize import value_to_json_obj
    from agm.agl.syntax.spans import SourceSpan

    fields: dict[str, object] = {
        k: value_to_json_obj(v) for k, v in exc.fields.items()
    }
    line: int | None = None
    col: int | None = None
    if isinstance(span, SourceSpan):
        line = span.start_line
        col = span.start_col
    return RunError(type_name=exc.type_name, fields=fields, line=line, col=col)


def _build_call_inventory(
    checked: CheckedProgramType,
    contracts: dict[int, object],
) -> list[CallSiteInfo]:
    """Build the §10.1 static call-site inventory from the checked program.

    The inventory is derived entirely from the checker's work: each
    ``CallSiteRecord`` (recorded in source order while type-checking) supplies the
    callee, parse policy, and span; ``contract_specs`` supplies the codec and
    target type; and the materialized ``contracts`` table supplies schema
    presence.  No second AST walk is performed.

    Returns one ``CallSiteInfo`` per agent-call/exec site, in source order.
    """
    from agm.agl.runtime.contract import OutputContract

    inventory: list[CallSiteInfo] = []

    for record in checked.call_sites:
        # The checker records a CallSiteRecord and an OutputContractSpec together
        # in ``_check_agent_call`` (both keyed by the call's node_id), so every
        # recorded call site has a spec.  A missing spec is a checker-invariant
        # violation, not a normal skip.
        spec = checked.contract_specs[record.node_id]

        contract = contracts.get(record.node_id)
        has_schema = (
            isinstance(contract, OutputContract) and contract.json_schema is not None
        )

        inventory.append(
            CallSiteInfo(
                callee=record.callee,
                target_type=repr(spec.target_type),
                codec_name=spec.codec_name,
                has_schema=has_schema,
                parse_policy=record.parse_policy,
                line=record.line,
                col=record.col,
            )
        )

    return inventory
