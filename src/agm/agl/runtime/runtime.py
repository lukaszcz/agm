"""WorkflowRuntime — the public façade for the AgL host runtime.

M1 implementation: full parse → scope → typecheck → eval pipeline.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agm.agl.diagnostics import Diagnostic
from agm.agl.runtime.agents import AgentFn

if TYPE_CHECKING:
    from agm.agl.eval.values import ExceptionValue, Value
    from agm.agl.typecheck.types import Type as AglType

# Reserved agent names: cannot be registered by callers.
_RESERVED_AGENT_NAMES: frozenset[str] = frozenset({"prompt", "exec"})


@dataclass(frozen=True, slots=True)
class RunError:
    """Structured representation of an uncaught AgL exception.

    ``type_name`` is the exception's declared type name (e.g. ``"AgentParseError"``).
    ``fields`` is a mapping from field names to JSON-shaped Python values.
    """

    type_name: str
    fields: dict[str, object]


@dataclass(slots=True)
class RunResult:
    """Result of a ``WorkflowRuntime.run`` call.

    ``ok``
        ``True`` iff there are no error-severity diagnostics **and** no
        uncaught AgL exception.  Warning-severity diagnostics (see
        ``Diagnostic.severity``) may be present while ``ok`` is ``True``.
    ``diagnostics``
        Pre-execution diagnostics (lex/parse/scope/typecheck/input-validation).
        Each entry has a ``.message`` (str), a ``.line`` (int, 1-based) and a
        ``.severity`` (``"error"`` or ``"warning"``).  When ``ok`` is ``True``
        any entries are warnings only.
    ``error``
        The uncaught AgL exception, or ``None``.  Set only when the program
        *started* executing but ended with an unhandled exception (exit code 2
        per the CLI contract).  ``None`` for pre-execution failures and for
        successful runs.
    ``bindings``
        Root-scope bindings after a successful run (name → Value).  Empty for
        failed runs.
    """

    ok: bool
    diagnostics: list[Diagnostic]
    error: RunError | None
    bindings: dict[str, Value] = field(default_factory=dict)


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
    """

    def __init__(
        self,
        *,
        default_loop_limit: int = 5,
        default_strict_json: bool = False,
        default_agent: AgentFn | None = None,
    ) -> None:
        self._default_loop_limit = default_loop_limit
        self._default_strict_json = default_strict_json
        self._default_agent = default_agent
        self._agents: dict[str, AgentFn] = {}

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

    def run(
        self,
        source: str,
        *,
        inputs: Mapping[str, object] | None = None,
        check_only: bool = False,
    ) -> RunResult:
        """Parse, analyse, and (unless ``check_only``) execute an AgL program.

        Pipeline:
            parse → resolve → check (with HostCapabilities) →
            validate inputs → materialize contracts → eval

        When ``check_only`` is ``True`` (``agm exec --dry-run``) the runtime
        runs the full static pipeline, input validation, and contract
        materialization, then STOPS before executing any statement: a clean
        program returns ``ok=True`` with no bindings and produces no program
        output; static/input errors still return ``ok=False``.
        TODO(M2): emit the §10.1 static call-site inventory here.

        Returns a ``RunResult`` capturing the outcome.
        """
        if inputs is None:
            inputs = {}

        # ----------------------------------------------------------------
        # Build HostCapabilities from registrations.
        # ----------------------------------------------------------------
        from agm.agl.capabilities import HostCapabilities
        from agm.agl.runtime.agents import AgentRegistry
        from agm.agl.runtime.codec import TextCodec

        text_codec = TextCodec()
        registry = AgentRegistry(
            named={name: fn for name, fn in self._agents.items()},
            default_agent=self._default_agent,
        )
        capabilities = HostCapabilities(
            agent_names=registry.agent_names,
            has_fallback_agent=registry.has_fallback,
            has_default_agent=registry.has_default_agent,
            codec_kinds={text_codec.name: frozenset({"text"})},
            # The renderers/codecs below are exactly those implemented in M1
            # (see ``render.py`` ``_RENDERERS`` and ``codec.py``).  This catalog
            # is the single source of truth for host capabilities.
            renderer_names=frozenset({"default", "raw", "json", "bullets"}),
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
                ok=False, diagnostics=list(warnings) + input_errors, error=None
            )

        # ----------------------------------------------------------------
        # [5] Materialize output contracts (text codec only in M1)
        # ----------------------------------------------------------------
        from agm.agl.runtime.contract import materialize_contract

        codecs = {text_codec.name: text_codec}
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
                ok=False, diagnostics=list(warnings) + contract_errors, error=None
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
                ok=False, diagnostics=list(warnings) + input_bind_errors, error=None
            )

        # ----------------------------------------------------------------
        # [check_only] --dry-run stop: the full static pipeline, input
        # validation, and contract materialization have all succeeded.  Stop
        # before executing any statement (no program output, no side effects).
        # TODO(M2): print the §10.1 static call-site inventory here.
        # ----------------------------------------------------------------
        if check_only:
            return RunResult(
                ok=True,
                diagnostics=list(warnings),
                error=None,
                bindings={},
            )

        # ----------------------------------------------------------------
        # [7] Build and run the interpreter
        # ----------------------------------------------------------------
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.runtime.contract import OutputContract

        typed_contracts: dict[int, OutputContract] = {
            nid: c for nid, c in contracts.items() if isinstance(c, OutputContract)
        }

        interp = Interpreter(
            checked=checked,
            registry=registry,
            contracts=typed_contracts,
            type_env=checked.type_env,
            loop_limit=self._default_loop_limit,
            strict_json=self._default_strict_json,
        )

        try:
            interp.execute(root_scope)
        except AglRaise as exc:
            # Uncaught AgL exception (exit code 2 per the CLI contract).
            # ONLY the AgL exception carrier is caught here: an unexpected Python
            # exception is an interpreter bug and must propagate (crash loudly)
            # rather than masquerade as a user-facing pre-execution diagnostic.
            error = _exception_value_to_run_error(exc.exc)
            return RunResult(
                ok=False,
                diagnostics=list(warnings),
                error=error,
                bindings={},
            )

        # Successful run: snapshot root bindings.
        root_bindings = root_scope.snapshot()

        return RunResult(
            ok=True,
            diagnostics=list(warnings),
            error=None,
            bindings=root_bindings,
        )

    @property
    def default_loop_limit(self) -> int:
        """Default iteration bound for ``do[N]`` loops."""
        return self._default_loop_limit

    @property
    def default_strict_json(self) -> bool:
        """Whether strict JSON parsing is the default."""
        return self._default_strict_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _convert_input(name: str, raw: object, type_obj: "AglType") -> "Value":
    """Convert a raw host input value to the declared AgL type.

    Scalars are supported in M1:
    - ``text``: verbatim (the value must already be a ``str``).
    - ``int``/``decimal``/``bool``/``json``: parsed via stdlib ``json`` with
      ``parse_float=Decimal`` (design §5.1: no binary floats) and validated.

    Non-scalar declared types (``list``/``dict``/``record``/``enum`` — anything
    the M1 host cannot parse from JSON into a typed ``Value``) are rejected with
    a clear pre-execution diagnostic; the JSON codec lands in M2.
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
        IntType,
        JsonType,
        TextType,
    )

    # Text: verbatim.
    if isinstance(type_obj, TextType):
        if not isinstance(raw, str):
            raise ValueError(
                f"Input {name!r}: expected a text value (str), got {type(raw).__name__}"
            )
        return TextValue(raw)

    # Reject non-scalar declared types: the M1 host has no codec to build a
    # typed Value from JSON for these.
    if not isinstance(type_obj, (IntType, DecimalType, BoolType, JsonType)):
        raise ValueError(
            f"Input {name!r} has type {type_obj!r}; non-scalar JSON-typed inputs "
            "land with the json codec (M2). M1 supports only text, int, decimal, "
            "bool, and json inputs."
        )

    # Scalar non-text: parse from JSON if given as a string.
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


def _exception_value_to_run_error(exc: "ExceptionValue") -> RunError:
    """Convert an ``ExceptionValue`` to a ``RunError`` for ``RunResult``.

    Field values are converted via the shared serializer, which preserves
    ``Decimal`` exactness (never routed through binary ``float``; design §5.1).
    """
    from agm.agl.runtime.serialize import value_to_json_obj

    fields: dict[str, object] = {
        k: value_to_json_obj(v) for k, v in exc.fields.items()
    }
    return RunError(type_name=exc.type_name, fields=fields)
