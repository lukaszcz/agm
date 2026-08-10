"""IR evaluator for the AgL typeless execution IR.

``IrInterpreter`` executes an ``ExecutableProgram`` using the per-invocation
frame / let-by-value / var-by-cell model.

Allowed imports:
- ``agm.agl.ir.*``
- ``agm.agl.semantics.values`` (all value types, Cell, Frame)
- ``agm.agl.semantics.exceptions`` (AglRaise, make_builtin_exception)
- ``agm.agl.semantics.copying`` (deep_copy_value, shallow_copy_value)
- ``agm.agl.eval._decimal`` (shared pinned decimal context)
- ``agm.agl.runtime.serialize`` (value_to_json_obj for ToJson and direct JSON
  construction)
- ``agm.config.engine_keys`` (the canonical engine-key catalog data leaf)

NOT allowed: ``agm.agl.syntax``, ``agm.agl.scope``, ``agm.agl.typecheck``.
"""

from __future__ import annotations

import decimal
import inspect
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, assert_never, cast

from agm.agl.eval._decimal import AGL_DECIMAL_CONTEXT
from agm.agl.eval.arith import (
    AglDivisionByZero,
    add,
    contains,
    div,
    logical_not,
    mul,
    negate,
    order,
    sub,
    value_eq,
)
from agm.agl.eval.conversions import AglCastConversion, run_recipe
from agm.agl.eval.effects import EffectHandlers
from agm.agl.eval.indexing import AglIndexOutOfRange, AglMissingKey, index_get, index_set
from agm.agl.ir.contracts import ContractRequest, ConversionFailureMode
from agm.agl.ir.ids import ContractId, FunctionId, Location, NominalId, SymbolId
from agm.agl.ir.nodes import (
    IrAnd,
    IrArith,
    IrAsk,
    IrAskRequest,
    IrAssign,
    IrBind,
    IrBlock,
    IrBreak,
    IrBuiltinLoad,
    IrBuiltinStore,
    IrCase,
    IrCoerce,
    IrCompare,
    IrConstBool,
    IrConstDecimal,
    IrConstInt,
    IrConstJsonNull,
    IrConstText,
    IrConstUnit,
    IrContains,
    IrContinue,
    IrConvert,
    IrCopyValue,
    IrDirectCall,
    IrEnumCaseKey,
    IrExec,
    IrExpr,
    IrField,
    IrFieldMode,
    IrFunctionParam,
    IrIf,
    IrIndex,
    IrIndexSet,
    IrIndirectCall,
    IrIterHasNext,
    IrIterInit,
    IrIterNext,
    IrLiteralCaseKey,
    IrLiteralKind,
    IrLoad,
    IrLoop,
    IrMakeArray,
    IrMakeClosure,
    IrMakeConstructor,
    IrMakeDict,
    IrMakeEnum,
    IrMakeException,
    IrMakeJsonArray,
    IrMakeJsonObject,
    IrMakeRecord,
    IrOr,
    IrParseJson,
    IrPrint,
    IrRaise,
    IrRenderTemplate,
    IrRenderValue,
    IrResource,
    IrReturn,
    IrSequence,
    IrTemplateText,
    IrTemplateValue,
    IrTry,
    IrUnary,
    IrUpdateRecord,
    IrVariantIs,
    UseDefault,
)
from agm.agl.ir.operations import (
    ArithOp,
    CmpOp,
    Coercion,
    CopyKind,
    IntToDecimal,
    ToJson,
    UnaryOp,
)
from agm.agl.ir.program import (
    ExecutableProgram,
    ExternFunctionBody,
    FunctionDescriptor,
    IrFunctionBody,
)
from agm.agl.ir.validate import InvalidIrError
from agm.agl.modules.ids import ModuleId
from agm.agl.runtime.agents import AgentFn
from agm.agl.runtime.codec import ParseResult, _parse_contract_output
from agm.agl.runtime.convert import StrictJsonParseError, parse_json_strict
from agm.agl.runtime.externs import ExternRegistry
from agm.agl.runtime.option import none_value, option_text, some_value
from agm.agl.runtime.params import engine_default_settings
from agm.agl.runtime.render import render_value
from agm.agl.runtime.serialize import value_to_json_obj
from agm.agl.runtime.trace import TraceStore, noop_trace
from agm.agl.semantics.copying import deep_copy_value, shallow_copy_value
from agm.agl.semantics.cycles import AglCyclicValue, cyclic_value_raise
from agm.agl.semantics.exceptions import AglRaise
from agm.agl.semantics.exceptions import make_builtin_exception as _make_exc_value
from agm.agl.semantics.values import (
    UNIT_VALUE,
    VOID_VALUE,
    ArrayValue,
    BoolValue,
    Cell,
    ConstructorValue,
    DecimalValue,
    DictValue,
    EnumValue,
    ExceptionValue,
    Frame,
    IntValue,
    IrClosureValue,
    IteratorValue,
    JsonValue,
    RecordValue,
    TextValue,
    Value,
)
from agm.config.engine_keys import (
    HOST_CONSUMED_ENGINE_KEYS,
    RUNTIME_LIVE_ENGINE_KEYS,
    TRACE_ENGINE_KEYS,
    trace_write_implies_enabled,
)
from agm.core.parse import format_timeout as _format_timeout
from agm.core.parse import parse_timeout as _parse_timeout

if TYPE_CHECKING:
    from agm.agl.runtime.contract import OutputContract
    from agm.agl.runtime.host_settings import HostSettingsReconfigurer

__all__ = ["HostConfigurationError", "IrInterpreter", "_apply_coercion", "_make_exc_value"]


class HostConfigurationError(Exception):
    """The materialized ``default-agent`` value cannot be dispatched.

    Raised by :class:`IrInterpreter`'s constructor when the winning
    ``default-agent`` value — a host seed (``[exec] runner``, already
    validated before this point) or a declared/spliced ``builtin var``
    default (``--agent``, ``[exec]``/qualified program-table ``default-agent``, or
    ``std/config``'s own default) — is an ``AgentCommand`` whose command text
    does not shell-split (see :func:`agm.agent.runner.parse_command`).

    This only covers the value materialized at construction time, before any
    statement of the entry runs: a later ``std/config::default-agent := ...``
    source write is never checked here, so a malformed command written at
    runtime stays an ordinary AgL runtime error raised from the ``ask`` call
    site that actually dispatches it, not a host-configuration failure.
    """


class _FlexibleParse(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> ParseResult: ...


def _call_custom_codec_parse(
    host_contract: "OutputContract",
    request: ContractRequest,
    raw: str,
    *,
    effective_strict: bool,
    schema: dict[str, object] | None,
) -> ParseResult:
    """Call a custom codec parse hook, accepting legacy signatures."""
    kwargs: dict[str, object] = {
        "strict_json": effective_strict,
        "schema": schema,
        "decode": host_contract.decode,
        "defs": dict(host_contract.defs),
        "type_table": None,
    }
    parse = cast(_FlexibleParse, host_contract.codec.parse)
    try:
        params = inspect.signature(host_contract.codec.parse).parameters
    except (TypeError, ValueError):
        return parse(raw, **kwargs)

    accepts_var_kw = any(param.kind.name == "VAR_KEYWORD" for param in params.values())
    accepted_kwargs = {
        name: value for name, value in kwargs.items() if accepts_var_kw or name in params
    }
    positional = [
        param
        for param in params.values()
        if param.kind.name in {"POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD"}
    ]
    accepts_varargs = any(param.kind.name == "VAR_POSITIONAL" for param in params.values())
    if accepts_varargs or len(positional) >= 2:
        from agm.agl.runtime.contract import _target_type_for_request

        return parse(raw, _target_type_for_request(request), **accepted_kwargs)
    return parse(raw, **accepted_kwargs)


# Engine-key defaults, built on first use.  The evaluator owns no default of its
# own: a register falls back to the ``builtin var`` declaration's initializer and
# then to these host-runtime values.  They are fixed data, so they are built once
# and shared.
_ENGINE_DEFAULT_SETTINGS: dict[str, Value] = {}


def _engine_default_settings() -> Mapping[str, Value]:
    """Return the host runtime's engine-key defaults."""
    if not _ENGINE_DEFAULT_SETTINGS:
        _ENGINE_DEFAULT_SETTINGS.update(engine_default_settings())
    return _ENGINE_DEFAULT_SETTINGS


# Memo for :func:`_literal_key_value`, keyed by the frozen, hashable case key.
_LITERAL_KEY_VALUES: dict[IrLiteralCaseKey, Value] = {}


def _literal_key_value(key: IrLiteralCaseKey) -> Value:
    """Materialize the runtime value represented by one typeless scalar key.

    Memoized on *key*: a literal case arm always materializes the same immutable
    ``Value``, so the hot case-dispatch path reuses one instance instead of
    reallocating per arm per evaluation.
    """
    cached = _LITERAL_KEY_VALUES.get(key)
    if cached is not None:
        return cached
    value = _make_literal_key_value(key)
    _LITERAL_KEY_VALUES[key] = value
    return value


def _make_literal_key_value(key: IrLiteralCaseKey) -> Value:
    """Build the runtime value for one typeless scalar key (uncached)."""
    if key.kind is IrLiteralKind.NUMERIC:
        assert isinstance(key.scalar_value, decimal.Decimal)
        return DecimalValue(key.scalar_value)
    if key.kind is IrLiteralKind.BOOL:
        assert isinstance(key.scalar_value, bool)
        return BoolValue(key.scalar_value)
    if key.kind is IrLiteralKind.TEXT:
        assert isinstance(key.scalar_value, str)
        return TextValue(key.scalar_value)
    assert key.scalar_value is None
    return JsonValue(None)


def _project_nominal_field(
    value: Value, nominal: NominalId, field: str, mode: IrFieldMode
) -> Value:
    """Read one declared field from a nominal runtime value.

    Records, enum payloads, and exceptions all retain their nominal identity and
    field mapping at runtime, so this is the one typeless projection mechanism.
    Exact projections require identity equality; upper-bound projections do not
    check identity because the static layer already proved the field exists on
    every value admitted by the bound.
    """
    if not isinstance(value, (RecordValue, EnumValue, ExceptionValue)):
        raise InvalidIrError(
            "IrField: expected RecordValue, EnumValue, or ExceptionValue, "
            f"got {type(value).__name__}"
        )
    match mode:
        case IrFieldMode.EXACT:
            if value.nominal != nominal:
                raise InvalidIrError(
                    f"IrField: expected nominal {nominal!r}, got {value.nominal!r}"
                )
        case IrFieldMode.UPPER_BOUND:
            pass
        case _ as _unreachable_mode:  # pragma: no cover
            assert_never(_unreachable_mode)
    try:
        return value.fields[field]
    except KeyError:
        raise InvalidIrError(f"IrField: nominal value lacks field {field!r}") from None


# The tree-walking evaluator descends through several Python stack frames per AgL
# call, so Python's own recursion limit — not the AgL call-depth guard — is what
# a deep recursion hits first at the default ceiling.  Before running, ``run()``
# raises Python's limit so the AgL guard (``max_call_depth``) trips first and its
# ``RecursionError`` stays catchable.  This budget is a generous upper bound on
# the Python frames one AgL call spans; ``_MAX_PYTHON_RECURSION_LIMIT`` caps how
# high the limit is pushed, beyond which a genuine Python ``RecursionError`` is
# converted to a catchable AgL ``RecursionError`` instead of escaping as a crash.
_PYTHON_FRAMES_PER_AGL_CALL = 80
_BASE_RECURSION_HEADROOM = 4000
_MAX_PYTHON_RECURSION_LIMIT = 1_000_000


# ---------------------------------------------------------------------------
# Internal loop-control signals (not AglRaise — bypass IrTry catch handlers)
# ---------------------------------------------------------------------------


class _BreakSignal(Exception):
    """Raised by ``IrBreak`` evaluation; caught only by the enclosing ``IrLoop``.

    Propagates through ``IrTry`` bodies unchanged because those catch only
    ``AglRaise``.  This ensures a ``break`` inside a ``try`` block exits the
    loop, not the ``try``.
    """


class _ContinueSignal(Exception):
    """Raised by ``IrContinue`` evaluation; caught only by the enclosing ``IrLoop``.

    Propagates through ``IrTry`` bodies unchanged.  The ``IrLoop`` evaluator
    catches this and executes ``continue`` on its Python ``while True`` loop to
    start the next iteration.
    """


class _ReturnSignal(Exception):
    """Raised by ``IrReturn`` evaluation; caught at the function-call boundary."""

    def __init__(self, value: Value) -> None:
        super().__init__()
        self.value = value


# ---------------------------------------------------------------------------
# Coercion helper — module-level for test access
# ---------------------------------------------------------------------------


def _apply_coercion(value: Value, coercion: Coercion) -> Value:
    """Apply a resolved ``Coercion`` to *value* and return the result.

    Switches on the closed ``Coercion`` union — no runtime type
    sniffing of *value*; the coercion op is pre-resolved by the lowerer.
    Every coercion is a scalar leaf conversion; neither arm rebuilds a
    structure. In particular ``ToJson`` is only ever compiled for a scalar
    source (see ``lower.coercions.compile_coercion``) — a container literal
    typed ``json`` lowers to ``IrMakeJsonArray``/``IrMakeJsonObject`` instead,
    so ``value`` here is never already a ``JsonValue``, an ``ArrayValue``, or
    a ``DictValue``.

    Raises ``InvalidIrError`` when the value tag does not match the coercion
    (cannot occur in well-lowered IR; defensive check only).
    """
    match coercion:
        case IntToDecimal():
            if not isinstance(value, IntValue):
                raise InvalidIrError(
                    f"IntToDecimal coercion requires IntValue, got {type(value).__name__}"
                )
            return DecimalValue(decimal.Decimal(value.value))

        case ToJson():
            return JsonValue(value_to_json_obj(value))

        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


# ---------------------------------------------------------------------------
# IrInterpreter
# ---------------------------------------------------------------------------


def _static_initializer_symbols(initializer: IrExpr) -> tuple[SymbolId, ...]:
    """Return the binding symbols materialized by one module initializer."""
    match initializer:
        case IrBind(symbol=sym):
            return (sym,)
        case IrSequence(items=items):
            return tuple(symbol for item in items for symbol in _static_initializer_symbols(item))
        case _:
            return ()


class IrInterpreter:
    """Evaluates an ``ExecutableProgram`` using the frame/cell model.

    The entry module starts with a root frame; function calls, closures, and loop
    iterations allocate additional frames as needed.

    ``run()`` executes the entry module's initializers in order and returns
    its public non-function bindings. When explicitly invoked, a synthetic
    inline ``main`` also contributes its direct public bindings.
    """

    DEFAULT_MAX_CALL_DEPTH: int = 256

    def __init__(
        self,
        program: ExecutableProgram,
        *,
        trace: TraceStore | None = None,
        max_call_depth: int = DEFAULT_MAX_CALL_DEPTH,
        param_values: Mapping[SymbolId, Value] | None = None,
        agent_dispatcher: AgentFn | None = None,
        strict_json: bool = False,
        loop_limit: int | None = None,
        shell_exec_timeout: float | None = None,
        host_contracts: Mapping[ContractId, "OutputContract"] | None = None,
        base_frame: Frame | None = None,
        extern_registry: ExternRegistry | None = None,
        host_reconfigurer: "HostSettingsReconfigurer | None" = None,
        builtin_host_settings: Mapping[str, Value] | None = None,
    ) -> None:
        self._program = program
        self._frames: list[Frame] = [base_frame if base_frame is not None else {}]
        self.initializer_values: list[Value] = []
        self.module_initializer_values: dict[ModuleId, list[Value]] = {}
        self.entry_param_symbols_installed: set[SymbolId] = set()
        self._static_bindings: dict[SymbolId, tuple[ModuleId, IrExpr]] = {
            symbol: (module.module_id, initializer)
            for module in program.modules.values()
            for initializer in module.initializers
            for symbol in _static_initializer_symbols(initializer)
        }
        self._evaluated_static_binding_ids: set[int] = set()
        self._resolving_param_defaults = False
        self._synthetic_main_frame: Frame | None = None
        self._call_depth: int = 0
        self._trace: TraceStore = trace if trace is not None else noop_trace()
        self._max_call_depth: int = max_call_depth
        self._param_values: Mapping[SymbolId, Value] = (
            param_values if param_values is not None else {}
        )
        self._agent_dispatcher = agent_dispatcher
        # Bootstrap the setting fields so declared defaults can be evaluated by
        # the ordinary, typeless evaluator. Constant defaults cannot read a
        # setting or invoke a host operation, so this temporary state is never
        # observable by their evaluation.
        self._strict_json = False
        self._loop_limit: int | None = None
        self._shell_exec_timeout: float | None = None
        self._timeout_setting = none_value()
        self._builtin_host_settings: dict[str, Value] = {}
        self._host_reconfigurer = host_reconfigurer

        defaults = dict(_engine_default_settings())
        defaults.update(
            {
                key: self._eval(value)
                for key, value in self._program.builtin_setting_defaults.items()
            }
        )
        seed = builtin_host_settings if builtin_host_settings is not None else {}

        # Runtime-live settings use an explicit host seed when present.  Their
        # driver arguments remain compatibility fallbacks: an absent false/None
        # must not suppress a declaration default.  Bootstrap through the same
        # effect path as a source write so host-invalid declared values become
        # normal AgL runtime errors.
        strict_setting = seed.get("strict-json")
        if strict_setting is None:
            strict_default = defaults["strict-json"]
            assert isinstance(strict_default, BoolValue)
            strict_setting = BoolValue(strict_json or strict_default.value)
        assert isinstance(strict_setting, BoolValue)
        self._apply_config_effect("strict-json", strict_setting)

        max_iters_setting = seed.get("max-iters")
        if max_iters_setting is None:
            max_iters_default = defaults["max-iters"]
            assert isinstance(max_iters_default, IntValue)
            max_iters_setting = IntValue(
                loop_limit if loop_limit is not None else max_iters_default.value
            )
        assert isinstance(max_iters_setting, IntValue)
        self._apply_config_effect("max-iters", max_iters_setting)

        timeout_setting = seed.get("timeout")
        if timeout_setting is None:
            timeout_setting = (
                some_value(TextValue(_format_timeout(shell_exec_timeout)))
                if shell_exec_timeout is not None
                else defaults["timeout"]
            )
        assert isinstance(timeout_setting, EnumValue)
        self._timeout_setting = timeout_setting
        self._apply_config_effect("timeout", timeout_setting)

        # Host-consumed registers use the host seed when one is provided and
        # otherwise the ``builtin var`` declaration's default. A key with
        # neither gets no register at all rather than a fabricated value;
        # reading it is then a hard error (see ``_load_builtin_setting``).
        effective = {**defaults, **seed}
        self._builtin_host_settings = {
            key: effective[key] for key in HOST_CONSUMED_ENGINE_KEYS if key in effective
        }
        default_agent = self._builtin_host_settings.get("default-agent")
        if isinstance(default_agent, EnumValue):
            self._check_default_agent_dispatchable(default_agent)
        if self._host_reconfigurer is not None:
            self._reconfigure_host_service()
        self._host_contracts: Mapping[ContractId, OutputContract] = (
            host_contracts if host_contracts is not None else {}
        )
        self._extern_registry: ExternRegistry = (
            extern_registry if extern_registry is not None else ExternRegistry()
        )
        self._effects = EffectHandlers(self)

    def _parse_host_output(
        self, raw: str, contract_id: ContractId, *, effective_strict: bool
    ) -> ParseResult:
        contract = self._program.contracts[contract_id]
        host_contract = self._host_contracts.get(contract_id)
        if host_contract is None or contract.codec_name in {"text", "json"}:
            return _parse_contract_output(raw, contract, effective_strict=effective_strict)
        schema = host_contract.json_schema if isinstance(host_contract.json_schema, dict) else None
        return _call_custom_codec_parse(
            host_contract,
            contract,
            raw,
            effective_strict=effective_strict,
            schema=schema,
        )

    @property
    def _frame(self) -> Frame:
        """Return the current (top-of-stack) frame."""
        return self._frames[-1]

    # ------------------------------------------------------------------
    # Post-run engine-setting accessors
    # ------------------------------------------------------------------

    @property
    def strict_json(self) -> bool:
        """Current strict-JSON setting (may have been updated by a ``builtin var`` write)."""
        return self._strict_json

    @property
    def loop_limit(self) -> int | None:
        """Current global max-iters valve (``None`` means off)."""
        return self._loop_limit

    @property
    def timeout_setting(self) -> EnumValue:
        """Current raw ``Option[text]`` timeout value."""
        return self._timeout_setting

    @property
    def shell_exec_timeout(self) -> float | None:
        """Current shell-exec timeout (may have been updated by a ``builtin var`` write)."""
        return self._shell_exec_timeout

    @property
    def builtin_host_settings(self) -> dict[str, Value]:
        """Current host-consumed register values.

        A snapshot copy of the registers backing the host-consumed ``builtin
        var`` engine settings, reflecting any writes made during the run.  A key
        with neither a host seed nor a declared default is absent.  Hosts that
        persist settings across runs (the REPL) read this back after a run to
        seed the next one.
        """
        return dict(self._builtin_host_settings)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _eval_expecting_bool(self, expr: IrExpr, context: str) -> bool:
        """Evaluate ``expr`` and require its result to be a ``BoolValue``."""
        value = self._eval(expr)
        if not isinstance(value, BoolValue):
            raise InvalidIrError(f"{context} expected BoolValue, got {type(value).__name__}")
        return value.value

    def _eval_render_bool_option(self, expr: IrExpr, option_name: str) -> bool:
        return self._eval_expecting_bool(expr, f"IrRenderValue: {option_name}")

    def _index_failure(self, err: AglIndexOutOfRange | AglMissingKey) -> AglRaise:
        """Convert an index/key sentinel into an ``AglRaise`` with the appropriate fields.

        Centralises the identical sentinel-wrapping sites (the ``IrIndex`` and
        ``IrIndexSet`` handlers) so the exception message and field shapes are
        defined exactly once.
        """
        match err:
            case AglIndexOutOfRange():
                return AglRaise(
                    _make_exc_value(
                        "IndexError",
                        f"Array index {err.index} out of range for length {err.length}",
                        nominals=self._program.builtin_nominals,
                        index=IntValue(err.index),
                        length=IntValue(err.length),
                    ),
                )
            case AglMissingKey():
                return AglRaise(
                    _make_exc_value(
                        "KeyError",
                        f"Dict key {err.key!r} is missing",
                        nominals=self._program.builtin_nominals,
                        key=TextValue(err.key),
                    ),
                )
            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)

    def _cyclic_failure(self) -> AglRaise:
        """Convert a detected reference cycle into an ``AglRaise(CyclicValueError)``.

        Mirrors ``_index_failure``: centralizes the sentinel-to-exception
        conversion so every ``render``/``as json``/coercion site that can
        reach a cyclic array or dict raises identical exception fields.
        """
        return cyclic_value_raise(nominals=self._program.builtin_nominals)

    def _render_or_raise(
        self, value: Value, *, pretty: bool = False, quote_strings: bool = False
    ) -> str:
        """Render a value, converting only the rendering cycle sentinel."""
        try:
            return render_value(value, pretty=pretty, quote_strings=quote_strings)
        except AglCyclicValue:
            raise self._cyclic_failure()

    def _on_cast_failure(
        self, failure_mode: ConversionFailureMode, exc: AglCastConversion
    ) -> BoolValue:
        """Handle a fallible-cast failure per the conversion failure mode.

        ``RAISE_CAST_ERROR`` (``as``) raises a ``CastError`` matching the legacy
        field shapes; ``RETURN_BOOL`` (``as?``) yields ``BoolValue(False)``.
        """
        match failure_mode:
            case ConversionFailureMode.RAISE_CAST_ERROR:
                raise AglRaise(
                    _make_exc_value(
                        "CastError",
                        exc.message,
                        nominals=self._program.builtin_nominals,
                        source_type=TextValue(exc.source_label),
                        target_type=TextValue(exc.target_label),
                        raw=TextValue(exc.raw),
                    ),
                )
            case ConversionFailureMode.RETURN_BOOL:
                return BoolValue(False)
            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)

    def _get_closure_for(self, fn_id: FunctionId) -> IrClosureValue:
        """Look up a direct-call closure in its lexical evaluation frame."""
        desc = self._program.functions[fn_id]
        slot = next(
            (
                frame[desc.function_symbol]
                for frame in reversed(self._frames)
                if desc.function_symbol in frame
            ),
            None,
        )
        if slot is None:
            raise InvalidIrError(
                f"IrDirectCall: function_symbol for fn_id={fn_id!r} not in any evaluation frame"
            )
        val = slot.value if isinstance(slot, Cell) else slot
        if not isinstance(val, IrClosureValue):
            raise InvalidIrError(
                f"IrDirectCall: function_symbol slot is not IrClosureValue,"
                f" got {type(val).__name__}"
            )
        return val

    def _bind_and_invoke(
        self,
        desc: "FunctionDescriptor",
        body: IrExpr,
        closure_val: IrClosureValue,
        bound_values: list[Value],
        *,
        retain_frame: bool = False,
    ) -> Value:
        """Build a call frame, push it, evaluate the function body, pop the frame, return result.

        Shared by ``_execute_direct_call`` and ``_execute_indirect_call``.

        Function parameters are immutable in AgL (they can never be the target of an
        assignment), so they are bound by value — never boxed in a Cell.
        """
        call_frame: Frame = dict(closure_val.captures)
        for param, val in zip(desc.params, bound_values, strict=True):
            call_frame[param.symbol] = val

        self._frames.append(call_frame)
        self._call_depth += 1
        try:
            try:
                result = self._eval(body)
            except _ReturnSignal as signal:
                result = signal.value
        finally:
            self._call_depth -= 1
            if retain_frame:
                self._synthetic_main_frame = call_frame
            self._frames.pop()

        return result

    def _install_function_closures(self) -> None:
        """Pre-install every module's zero-capture function closures.

        Runs before parameter defaults so a default can call a function from
        its declaring module, including an imported module whose initializers
        have not yet run.
        """
        for module in self._program.modules.values():
            for node in module.initializers:
                match node:
                    case IrBind(
                        symbol=sym,
                        value=IrMakeClosure(function_id=fn_id, captures=()) as closure_node,
                    ):
                        desc = self._program.functions.get(fn_id)
                        if desc is None or desc.function_symbol != sym:
                            continue
                        value = self._eval(closure_node)
                        self._frames[0][sym] = value
                    case _:
                        continue

    def _recursion_error(self) -> AglRaise:
        """Build the catchable AgL ``RecursionError`` for an exceeded call depth.

        Shared by the ``max_call_depth`` guard and the backstop that converts a
        Python ``RecursionError`` (raised when Python's own limit is hit before
        the guard) into the same catchable AgL exception.
        """
        return AglRaise(
            _make_exc_value(
                "RecursionError",
                f"Maximum call depth ({self._max_call_depth}) exceeded",
                nominals=self._program.builtin_nominals,
                limit=IntValue(self._max_call_depth),
            )
        )

    def _check_call_depth(self) -> None:
        if self._call_depth >= self._max_call_depth:
            raise self._recursion_error()

    def _eval_default_in_frame(self, param: "IrFunctionParam", frame: Frame) -> Value:
        """Evaluate an omitted argument's default expression in *frame*."""
        assert param.default is not None, "arg omitted but param has no default (lowerer bug)"
        self._frames.append(frame)
        try:
            return self._eval(param.default)
        finally:
            self._frames.pop()

    def _eval_extern_default(self, param: "IrFunctionParam") -> Value:
        """Evaluate an omitted extern argument's default expression.

        Extern closures never capture anything, so the default is evaluated
        in a fresh empty frame — reads fall through to module (base-frame)
        scope, mirroring how an ordinary closure's captures frame chains to
        module scope for its own defaults.
        """
        return self._eval_default_in_frame(param, {})

    def _execute_direct_call(
        self,
        fn_id: FunctionId,
        arguments: "tuple[IrExpr | UseDefault, ...]",
        location: Location | None,
        *,
        retain_frame: bool = False,
    ) -> Value:
        """Execute a direct call to a named user function or an extern.

        An extern ``function_id`` skips the AgL body entirely and crosses
        into the companion Python module via the effects layer, mirroring
        the host-op dispatch pattern (no call-depth accounting — there is no
        AgL frame to recurse into).  Otherwise: depth check → evaluate
        arguments (``UseDefault`` uses a captures frame) → ``_bind_and_invoke``.
        """
        desc = self._program.functions[fn_id]
        match desc.impl:
            case ExternFunctionBody() as extern:
                extern_bound_values: list[Value] = []
                for param, arg in zip(desc.params, arguments, strict=True):
                    val = (
                        self._eval_extern_default(param)
                        if isinstance(arg, UseDefault)
                        else self._eval(arg)
                    )
                    extern_bound_values.append(val)
                return self._effects.eval_extern_call(desc.module_id, extern, extern_bound_values)
            case IrFunctionBody(body=body):
                self._check_call_depth()
                closure_val = self._get_closure_for(fn_id)

                bound_values: list[Value] = []
                for param, arg in zip(desc.params, arguments, strict=True):
                    val = (
                        self._eval_default_in_frame(param, dict(closure_val.captures))
                        if isinstance(arg, UseDefault)
                        else self._eval(arg)
                    )
                    bound_values.append(val)

                return self._bind_and_invoke(
                    desc, body, closure_val, bound_values, retain_frame=retain_frame
                )
            case other:  # pragma: no cover
                assert_never(other)

    def _execute_indirect_call(
        self,
        callee_expr: IrExpr,
        arguments: "tuple[IrExpr, ...]",
        location: Location,
    ) -> Value:
        """Execute an indirect (value) call.

        Evaluation order (mirrors ``_apply_closure``):
        1. Evaluate the callee in the current frame.
        2. Depth-limit check (AFTER callee eval, BEFORE arg binding).
        3. Evaluate each positional arg in the caller frame, NO coercion.
        4. Defensive: use ``desc.params[i].default`` for omitted trailing params
           (evaluated in a captures frame).
        5. ``_bind_and_invoke``.
        """
        callee_val = self._eval(callee_expr)
        if isinstance(callee_val, ConstructorValue):
            constructor_desc = self._program.nominals[callee_val.nominal]
            field_names = (
                constructor_desc.fields
                if callee_val.variant is None
                else next(
                    v.fields for v in constructor_desc.variants if v.name == callee_val.variant
                )
            )
            fields = {
                name: self._eval(argument)
                for name, argument in zip(field_names, arguments, strict=True)
            }
            if callee_val.variant is None:
                return RecordValue(
                    nominal=callee_val.nominal,
                    display_name=callee_val.display_name,
                    fields=fields,
                )
            return EnumValue(
                nominal=callee_val.nominal,
                display_name=callee_val.display_name,
                variant=callee_val.variant,
                fields=fields,
            )
        if not isinstance(callee_val, IrClosureValue):
            raise InvalidIrError(
                f"IrIndirectCall: callee evaluated to {type(callee_val).__name__},"
                " expected IrClosureValue"
            )

        desc = self._program.functions[callee_val.function_id]
        match desc.impl:
            case ExternFunctionBody() as extern:
                extern_bound_values: list[Value] = []
                for i, param in enumerate(desc.params):
                    if i < len(arguments):
                        val = self._eval(arguments[i])
                    elif param.default is not None:
                        val = self._eval_extern_default(param)
                    else:
                        raise InvalidIrError(
                            f"IrIndirectCall: missing argument for parameter {i!r}"
                            " and no default available (lowerer bug)"
                        )
                    extern_bound_values.append(val)
                return self._effects.eval_extern_call(desc.module_id, extern, extern_bound_values)
            case IrFunctionBody(body=body):
                self._check_call_depth()

                # Evaluate each positional argument in the CALLER frame (no coercion).
                bound_values: list[Value] = []
                for i, param in enumerate(desc.params):
                    if i < len(arguments):
                        val = self._eval(arguments[i])
                    elif param.default is not None:
                        # Defensive: evaluate default in a captures frame.
                        val = self._eval_default_in_frame(param, dict(callee_val.captures))
                    else:
                        raise InvalidIrError(
                            f"IrIndirectCall: missing argument for parameter {i!r}"
                            " and no default available (lowerer bug)"
                        )
                    bound_values.append(val)

                return self._bind_and_invoke(desc, body, callee_val, bound_values)
            case other:  # pragma: no cover
                assert_never(other)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def _program_entry_location(self, symbol: SymbolId) -> Location:
        """Return the selected ``program def`` body's source location."""
        descriptor = self._program.functions[self._program.program_functions[symbol]]
        assert isinstance(descriptor.impl, IrFunctionBody)
        return descriptor.impl.body.location

    def _invoke_program(self, symbol: SymbolId) -> Value:
        """Invoke a selected linked ``program def`` with an entry-point error span."""
        location = self._program_entry_location(symbol)
        try:
            return self._execute_direct_call(
                self._program.program_functions[symbol],
                (),
                location,
                retain_frame=symbol == self._program.synthetic_main_symbol,
            )
        except AglRaise as exc:
            if exc.span is None:
                exc.span = location
            raise

    def run(self, *, program_symbol: SymbolId | None = None) -> dict[str, Value]:
        """Execute all modules in order and return the entry module's public bindings.

        Installs every linked module param into the base frame BEFORE any module
        initializer runs, then iterates over all modules in insertion order
        (library modules first, entry last) executing each module's initializers.
        When *program_symbol* is provided, invokes that selected ``program def``
        before leaving the same managed execution boundary. All evaluation runs
        under the pinned AgL decimal context.

        Python's recursion limit is raised for the duration so the AgL
        ``max_call_depth`` guard is reached before Python's own limit; a Python
        ``RecursionError`` that still escapes (its limit is capped) is converted
        to a catchable AgL ``RecursionError`` rather than crashing the host.
        """
        needed = _BASE_RECURSION_HEADROOM + self._max_call_depth * _PYTHON_FRAMES_PER_AGL_CALL
        target = min(needed, _MAX_PYTHON_RECURSION_LIMIT)
        previous_limit = sys.getrecursionlimit()
        # Never lower an already-higher limit (e.g. a nested run); only raise it.
        sys.setrecursionlimit(max(previous_limit, target))
        try:
            with decimal.localcontext(AGL_DECIMAL_CONTEXT):
                # Closures install before params: a failing param default must
                # still let already-installed closures (and any declarations
                # completed earlier in the entry) be promoted. Promotion itself
                # is driven by ``lowered.promotion_plan.completed_declaration_ids``
                # (see ``entry_pipeline.completed_declaration_ids``), which is
                # conservative on its own — it excludes params whose symbols were
                # never installed and applies the declaration-dependency
                # fixpoint — so no separate "did the entry frame start" gate is
                # needed here.
                self._install_function_closures()
                self._resolving_param_defaults = True
                try:
                    self._install_params()
                finally:
                    self._resolving_param_defaults = False

                for mod in self._program.modules.values():
                    for node in mod.initializers:
                        if id(node) not in self._evaluated_static_binding_ids:
                            self._eval_and_record_initializer(mod.module_id, node)
                if program_symbol is not None:
                    try:
                        self._invoke_program(program_symbol)
                    except RecursionError:
                        error = self._recursion_error()
                        error.span = self._program_entry_location(program_symbol)
                        raise error from None
            return self._collect_results()
        except RecursionError:
            raise self._recursion_error() from None
        finally:
            sys.setrecursionlimit(previous_limit)

    def _eval_and_record_initializer(self, module_id: ModuleId, node: IrExpr) -> None:
        """Evaluate one initializer, retaining its result for result collection."""
        try:
            value = self._eval_initializer(node)
        except AglRaise as exc:
            if exc.span is None:
                exc.span = node.location
            raise
        self.initializer_values.append(value)
        self.module_initializer_values.setdefault(module_id, []).append(value)

    def _install_params(self) -> None:
        """Install resolved module parameters before evaluating initializers."""
        for ir_param in self._program.params:
            if ir_param.symbol in self._param_values:
                self._frames[0][ir_param.symbol] = self._param_values[ir_param.symbol]
            elif ir_param.default is not None:
                self._frames[0][ir_param.symbol] = self._eval(ir_param.default)
            else:
                raise InvalidIrError(
                    f"Required param {ir_param.public_name!r} has no value;"
                    " the host must supply a value for required params before calling run()"
                )
            self.entry_param_symbols_installed.add(ir_param.symbol)

    def _eval_static_binding(self, module_id: ModuleId, initializer: IrExpr) -> None:
        """Evaluate a parameter-default dependency in the module base frame."""
        self._frames.append(self._frames[0])
        try:
            self._eval_and_record_initializer(module_id, initializer)
        finally:
            self._frames.pop()
        self._evaluated_static_binding_ids.add(id(initializer))

    def _eval_initializer(self, node: IrExpr) -> Value:
        match node:
            case IrBind(symbol=sym, value=IrMakeClosure(function_id=fn_id)):
                desc = self._program.functions.get(fn_id)
                if desc is not None and desc.function_symbol == sym:
                    slot = self._frames[0].get(sym)
                    if slot is not None:
                        return slot.value if isinstance(slot, Cell) else slot
            case _:
                pass
        return self._eval(node)

    # ------------------------------------------------------------------
    # Expression evaluator (closed IrExpr dispatch)
    # ------------------------------------------------------------------

    def _eval(self, node: IrExpr) -> Value:
        """Evaluate *node* in the current frame and return its value.

        Dispatches over the closed ``IrExpr`` union with a structural ``match``
        whose final arm is ``assert_never`` so mypy exhaustiveness makes a
        missing case a compile-time error.
        """
        match node:
            case IrConstInt(value=v):
                return IntValue(v)

            case IrConstDecimal(value=v):
                return DecimalValue(v)

            case IrConstBool(value=v):
                return BoolValue(v)

            case IrConstText(value=v):
                return TextValue(v)

            case IrResource(path=path):
                return TextValue(path)

            case IrConstUnit():
                return UNIT_VALUE

            case IrConstJsonNull():
                return JsonValue(None)

            case IrMakeArray(items=items):
                return ArrayValue([self._eval(item) for item in items])

            case IrMakeDict(entries=entries):
                result: dict[str, Value] = {}
                for key_expr, val_expr in entries:
                    key_val = self._eval(key_expr)
                    if not isinstance(key_val, TextValue):
                        raise InvalidIrError(
                            f"IrMakeDict key must evaluate to TextValue,"
                            f" got {type(key_val).__name__}"
                        )
                    result[key_val.value] = self._eval(val_expr)
                return DictValue(result)

            # `IrMakeJsonArray`/`IrMakeJsonObject`: every item/value here is already
            # scalar or `JsonValue` (never a raw array/dict): the checker requires an
            # explicit `as json` cast to embed a container in a json literal, and that
            # cast's own `IrConvert` handling is what detects a cyclic source. So
            # `value_to_json_obj` in both arms below is a leaf conversion, never a walk
            # that could re-enter a container — no cycle guard needed in either.
            case IrMakeJsonArray(items=json_items):
                return JsonValue([value_to_json_obj(self._eval(item)) for item in json_items])

            case IrMakeJsonObject(entries=json_entries):
                json_result: dict[str, object] = {}
                for key_expr, val_expr in json_entries:
                    key_val = self._eval(key_expr)
                    if not isinstance(key_val, TextValue):
                        raise InvalidIrError(
                            f"IrMakeJsonObject key must evaluate to TextValue,"
                            f" got {type(key_val).__name__}"
                        )
                    json_result[key_val.value] = value_to_json_obj(self._eval(val_expr))
                return JsonValue(json_result)

            case IrLoad(symbol=sym):
                slot = next(
                    (frame[sym] for frame in reversed(self._frames) if sym in frame),
                    None,
                )
                if slot is None and self._resolving_param_defaults:
                    module_id, initializer = self._static_bindings.pop(sym)
                    self._eval_static_binding(module_id, initializer)
                    slot = self._frames[0][sym]
                if slot is None:
                    raise InvalidIrError(
                        f"IrLoad: symbol_id={sym.value!r} is not bound in the frame"
                    )
                if isinstance(slot, Cell):
                    return slot.value
                return slot

            case IrBind(symbol=sym, value=val_expr):
                value = self._eval(val_expr)
                desc = self._program.symbols.get(sym)
                if desc is not None and desc.mutable:
                    self._frame[sym] = Cell(value)
                else:
                    self._frame[sym] = value
                return value

            case IrAssign(symbol=sym, value=val_expr):
                slot = self._frame.get(sym)
                if slot is None and self._frames[0] is not self._frame:
                    # Module vars live in the base frame and are intentionally
                    # not closure captures.
                    slot = self._frames[0].get(sym)
                if slot is None and self._resolving_param_defaults:
                    module_id, initializer = self._static_bindings.pop(sym)
                    self._eval_static_binding(module_id, initializer)
                    slot = self._frames[0][sym]
                if not isinstance(slot, Cell):
                    desc = self._program.symbols.get(sym)
                    if desc is None:
                        raise InvalidIrError(
                            f"IrAssign: symbol_id={sym.value!r} is not in program.symbols"
                        )
                    raise InvalidIrError(
                        f"IrAssign: symbol_id={sym.value!r}"
                        f" (public_name={desc.public_name!r}) is not a mutable var"
                    )
                # A simple var-cell store.  An assignment statement yields unit;
                # the mutation is the side effect.
                slot.value = self._eval(val_expr)
                return VOID_VALUE

            case IrIndexSet(container=container_expr, kind=kind, index=idx_expr, value=val_expr):
                # Plain left-to-right evaluation order: container, then index,
                # then the right-hand side, then the checked in-place store.
                container = self._eval(container_expr)
                index_val = self._eval(idx_expr)
                new_value = self._eval(val_expr)
                try:
                    index_set(kind, container, index_val, new_value)
                except (AglIndexOutOfRange, AglMissingKey) as e:
                    raise self._index_failure(e)
                return VOID_VALUE

            case IrCoerce(value=val_expr, operation=op):
                value = self._eval(val_expr)
                return _apply_coercion(value, op)

            case IrSequence(items=items) | IrBlock(items=items):
                last: Value = VOID_VALUE
                for item in items:
                    last = self._eval(item)
                return last

            case IrArith(op=arith_op, kind=kind, lhs=lhs_expr, rhs=rhs_expr):
                lhs_val = self._eval(lhs_expr)
                rhs_val = self._eval(rhs_expr)
                try:
                    match arith_op:
                        case ArithOp.ADD:
                            return add(kind, lhs_val, rhs_val)
                        case ArithOp.SUB:
                            return sub(kind, lhs_val, rhs_val)
                        case ArithOp.MUL:
                            return mul(kind, lhs_val, rhs_val)
                        case ArithOp.DIV:
                            return div(lhs_val, rhs_val)
                        case _ as unreachable:  # pragma: no cover
                            assert_never(unreachable)
                except AglDivisionByZero:
                    raise AglRaise(
                        _make_exc_value(
                            "ArithmeticError",
                            "Division by zero",
                            nominals=self._program.builtin_nominals,
                            operation=TextValue("/"),
                        )
                    )

            case IrCompare(op=cmp_op, kind=_kind, lhs=lhs_expr, rhs=rhs_expr):
                lhs_val = self._eval(lhs_expr)
                rhs_val = self._eval(rhs_expr)
                match cmp_op:
                    case CmpOp.EQ:
                        return BoolValue(value_eq(lhs_val, rhs_val))
                    case CmpOp.NEQ:
                        return BoolValue(not value_eq(lhs_val, rhs_val))
                    case CmpOp.LT | CmpOp.LE | CmpOp.GT | CmpOp.GE:
                        return BoolValue(order(cmp_op, lhs_val, rhs_val))
                    case _ as _unreachable_cmp:  # pragma: no cover
                        assert_never(_unreachable_cmp)

            case IrContains(kind=kind, item=item_expr, container=container_expr):
                item_val = self._eval(item_expr)
                container_val = self._eval(container_expr)
                return BoolValue(contains(kind, item_val, container_val))

            case IrAnd(lhs=lhs_expr, rhs=rhs_expr):
                lhs_val = self._eval(lhs_expr)
                if not isinstance(lhs_val, BoolValue):
                    raise InvalidIrError(
                        f"IrAnd: lhs is not BoolValue, got {type(lhs_val).__name__}"
                    )
                if not lhs_val.value:
                    return BoolValue(False)
                rhs_val = self._eval(rhs_expr)
                if not isinstance(rhs_val, BoolValue):
                    raise InvalidIrError(
                        f"IrAnd: rhs is not BoolValue, got {type(rhs_val).__name__}"
                    )
                return BoolValue(rhs_val.value)

            case IrOr(lhs=lhs_expr, rhs=rhs_expr):
                lhs_val = self._eval(lhs_expr)
                if not isinstance(lhs_val, BoolValue):
                    raise InvalidIrError(
                        f"IrOr: lhs is not BoolValue, got {type(lhs_val).__name__}"
                    )
                if lhs_val.value:
                    return BoolValue(True)
                rhs_val = self._eval(rhs_expr)
                if not isinstance(rhs_val, BoolValue):
                    raise InvalidIrError(
                        f"IrOr: rhs is not BoolValue, got {type(rhs_val).__name__}"
                    )
                return BoolValue(rhs_val.value)

            case IrUnary(op=unary_op, kind=kind, value=val_expr):
                val = self._eval(val_expr)
                match unary_op:
                    case UnaryOp.NOT:
                        if not isinstance(val, BoolValue):
                            raise InvalidIrError(
                                f"IrUnary NOT: expected BoolValue, got {type(val).__name__}"
                            )
                        return logical_not(val)
                    case UnaryOp.NEG:
                        if kind is None:
                            raise InvalidIrError("IrUnary NEG: kind must not be None")
                        if not isinstance(val, (IntValue, DecimalValue)):
                            raise InvalidIrError(
                                f"IrUnary NEG: expected numeric, got {type(val).__name__}"
                            )
                        return negate(kind, val)
                    case _ as _unreachable_unary:  # pragma: no cover
                        assert_never(_unreachable_unary)

            case IrField(value=val_expr, nominal=nominal, field=field_name, mode=mode):
                return _project_nominal_field(self._eval(val_expr), nominal, field_name, mode)

            case IrUpdateRecord(value=val_expr, updates=updates):
                target = self._eval(val_expr)
                if not isinstance(target, (RecordValue, ExceptionValue)):
                    raise InvalidIrError(
                        "IrUpdateRecord: expected RecordValue or ExceptionValue, "
                        f"got {type(target).__name__}"
                    )
                updated_fields: dict[str, Value] = dict(target.fields)
                for fname, fexpr in updates:
                    updated_fields[fname] = self._eval(fexpr)
                if isinstance(target, RecordValue):
                    return RecordValue(
                        nominal=target.nominal,
                        display_name=target.display_name,
                        fields=updated_fields,
                    )
                return ExceptionValue(
                    nominal=target.nominal,
                    display_name=target.display_name,
                    fields=updated_fields,
                )

            case IrIndex(kind=kind, value=val_expr, index=idx_expr):
                container = self._eval(val_expr)
                index_val = self._eval(idx_expr)
                try:
                    return index_get(kind, container, index_val)
                except (AglIndexOutOfRange, AglMissingKey) as e:
                    raise self._index_failure(e)

            case IrRenderTemplate(segments=segs):
                # The lexer has already applied the shared ``%{...}`` surface
                # rules, so splicing here is plain concatenation.
                parts: list[str] = []
                for seg in segs:
                    match seg:
                        case IrTemplateText(text=text):
                            parts.append(text)
                        case IrTemplateValue(value=value_expr):
                            parts.append(self._render_or_raise(self._eval(value_expr)))
                        case _ as unreachable_seg:  # pragma: no cover
                            assert_never(unreachable_seg)
                return TextValue("".join(parts))

            case IrMakeRecord(nominal=nominal, display_name=display_name, fields=fields):
                record_fields: dict[str, Value] = {
                    fname: self._eval(fexpr) for fname, fexpr in fields
                }
                return RecordValue(
                    nominal=nominal,
                    display_name=display_name,
                    fields=record_fields,
                )

            case IrMakeEnum(
                nominal=nominal, display_name=display_name, variant=variant, fields=fields
            ):
                enum_fields: dict[str, Value] = {
                    fname: self._eval(fexpr) for fname, fexpr in fields
                }
                return EnumValue(
                    nominal=nominal,
                    display_name=display_name,
                    variant=variant,
                    fields=enum_fields,
                )

            case IrMakeException(nominal=nominal, display_name=display_name, fields=fields):
                exc_fields: dict[str, Value] = {
                    fname: self._eval(field_expr) for fname, field_expr in fields
                }
                return ExceptionValue(
                    nominal=nominal,
                    display_name=display_name,
                    fields=exc_fields,
                )

            case IrMakeConstructor(nominal=nominal, display_name=display_name, variant=variant):
                return ConstructorValue(
                    nominal=nominal,
                    display_name=display_name,
                    variant=variant,
                )

            case IrVariantIs(variant=variant, value=val_expr, negated=negated):
                value = self._eval(val_expr)
                if not isinstance(value, EnumValue):
                    raise InvalidIrError(
                        f"IrVariantIs: value is not EnumValue, got {type(value).__name__}"
                    )
                return BoolValue((value.variant == variant) != negated)

            case IrConvert(value=val_expr, recipe=recipe, failure_mode=failure_mode):
                source_value = self._eval(val_expr)
                try:
                    converted = run_recipe(recipe, source_value)
                except AglCastConversion as exc:
                    return self._on_cast_failure(failure_mode, exc)
                except AglCyclicValue:
                    # `as` (RAISE_CAST_ERROR) raises the catchable CyclicValueError.
                    # `as?` (RETURN_BOOL) is a trial conversion: a cyclic value fails
                    # the same as any other unconvertible source, so it yields False
                    # rather than raising — see the lowerer's `as?` short-circuit
                    # comment for why RENDER/JSON casts reach here instead of
                    # skipping straight to True.
                    if failure_mode is ConversionFailureMode.RETURN_BOOL:
                        return BoolValue(False)
                    raise self._cyclic_failure()
                if failure_mode is ConversionFailureMode.RETURN_BOOL:
                    return BoolValue(True)
                return converted

            case IrIf(branches=branches, has_else=has_else):
                for branch in branches:
                    if branch.cond is None:
                        # Else branch — always taken.
                        branch_val = self._eval(branch.body)
                        return branch_val if has_else else VOID_VALUE
                    cond_val = self._eval(branch.cond)
                    if not isinstance(cond_val, BoolValue):
                        raise InvalidIrError(
                            f"IrIf: branch condition evaluated to"
                            f" {type(cond_val).__name__}, expected BoolValue"
                        )
                    if cond_val.value:
                        branch_val = self._eval(branch.body)
                        return branch_val if has_else else VOID_VALUE
                # No branch matched and no else: return the non-printable unit.
                return VOID_VALUE

            case IrRaise(exc=exc_expr):
                exc_val = self._eval(exc_expr)
                if not isinstance(exc_val, ExceptionValue):
                    raise InvalidIrError(
                        f"IrRaise: exc evaluated to {type(exc_val).__name__},"
                        " expected ExceptionValue"
                    )
                raise AglRaise(exc_val, span=node.location)

            case IrReturn(value=value_expr):
                raise _ReturnSignal(self._eval(value_expr))

            case IrTry(body=body_expr, handlers=handlers):
                try:
                    return self._eval(body_expr)
                except RecursionError:
                    # Python's limit was hit before the AgL guard (its limit is
                    # capped); surface it as the same catchable AgL exception so
                    # a `catch RecursionError` handler still fires.
                    pending = self._recursion_error()
                except AglRaise as exc:
                    pending = exc
                for handler in handlers:
                    if handler.nominal is None or handler.nominal == pending.exc.nominal:
                        if handler.symbol is not None:
                            self._frame[handler.symbol] = pending.exc
                        return self._eval(handler.body)
                raise pending

            case IrCase(subject=subject_expr, arms=arms, default=default):
                subject_val = self._eval(subject_expr)
                for arm in arms:
                    key = arm.key
                    if isinstance(key, IrEnumCaseKey):
                        selected = (
                            isinstance(subject_val, EnumValue)
                            and subject_val.nominal == key.nominal
                            and subject_val.variant == key.variant
                        )
                    else:
                        selected = value_eq(subject_val, _literal_key_value(key))
                    if not selected:
                        continue
                    if arm.field_bindings:
                        if not isinstance(subject_val, EnumValue):
                            raise InvalidIrError(
                                "IrCase: selected payload arm for a non-enum subject"
                            )
                        assert isinstance(key, IrEnumCaseKey)
                        for field_name, symbol in arm.field_bindings:
                            self._frame[symbol] = _project_nominal_field(
                                subject_val,
                                key.nominal,
                                field_name,
                                IrFieldMode.EXACT,
                            )
                    return self._eval(arm.body)
                if default is not None:
                    return self._eval(default)
                raise InvalidIrError("IrCase has no matching key and no default")

            case IrLoop(body=body_expr, guarded=guarded):
                # Unconditional repeat — all loop logic (bound checks, until
                # guards, for/while clauses) is desugared into the body by the
                # lowerer.  The only exits are IrBreak (leave the loop)
                # and IrContinue (next iteration).  Both signals propagate through
                # IrTry bodies (which catch only AglRaise) to reach this handler.
                #
                # The host's global max-iters valve applies ONLY to unguarded
                # loops (no [n] bound, no for clause): a self-bounded loop carries
                # its own termination and must never be cut short by this safety
                # net, which exists to catch runaway while/do-until loops.
                iterations = 0
                while True:
                    if (
                        not guarded
                        and self._loop_limit is not None
                        and iterations >= self._loop_limit
                    ):
                        raise AglRaise(
                            _make_exc_value(
                                "MaxIterationsExceeded",
                                f"Loop exhausted after {self._loop_limit} iterations",
                                nominals=self._program.builtin_nominals,
                                limit=IntValue(self._loop_limit),
                                condition=TextValue("loop limit"),
                                last_condition_value=BoolValue(False),
                                metadata=JsonValue(None),
                            )
                        )
                    try:
                        self._eval(body_expr)
                    except _BreakSignal:
                        return VOID_VALUE
                    except _ContinueSignal:
                        iterations += 1
                        continue
                    iterations += 1

            case IrBreak():
                raise _BreakSignal()

            case IrContinue():
                raise _ContinueSignal()

            case IrIterInit(collection=collection_expr):
                coll = self._eval(collection_expr)
                if isinstance(coll, ArrayValue):
                    # The ArrayValue's own element list, by reference: no copy,
                    # so an in-place element mutation is visible to the cursor.
                    return IteratorValue(elements=coll.elements)
                if isinstance(coll, DictValue):
                    # The key set is fixed for the collection's lifetime, so a
                    # one-time tuple of keys is sound even though the values
                    # behind those keys may still be mutated.
                    return IteratorValue(elements=tuple(TextValue(k) for k in coll.entries))
                if isinstance(coll, TextValue):
                    return IteratorValue(elements=tuple(TextValue(ch) for ch in coll.value))
                raise InvalidIrError(  # pragma: no cover
                    f"IrIterInit: unexpected collection type {type(coll)!r}"
                )

            case IrIterHasNext(iterator=iter_expr):
                it = self._eval(iter_expr)
                if not isinstance(it, IteratorValue):  # pragma: no cover
                    raise InvalidIrError(f"IrIterHasNext: expected IteratorValue, got {type(it)!r}")
                return BoolValue(it.pos < len(it.elements))

            case IrIterNext(iterator=iter_expr):
                it = self._eval(iter_expr)
                if not isinstance(it, IteratorValue):  # pragma: no cover
                    raise InvalidIrError(f"IrIterNext: expected IteratorValue, got {type(it)!r}")
                elem = it.elements[it.pos]
                it.pos += 1
                return elem

            case IrMakeClosure(function_id=fn_id, captures=captures):
                cap_slots: list[tuple[SymbolId, Value | Cell]] = []
                for cap in captures:
                    slot = self._frame.get(cap.symbol)
                    if slot is None:
                        raise InvalidIrError(
                            f"IrMakeClosure: capture symbol_id={cap.symbol.value!r} not in frame"
                        )
                    if cap.by_cell:
                        if not isinstance(slot, Cell):
                            raise InvalidIrError(
                                f"IrMakeClosure: by_cell capture symbol_id={cap.symbol.value!r}"
                                " but slot is not Cell"
                            )
                        cap_slots.append((cap.symbol, slot))
                    else:
                        val = slot.value if isinstance(slot, Cell) else slot
                        cap_slots.append((cap.symbol, val))
                function_desc = self._program.functions.get(fn_id)
                assert function_desc is not None, (
                    f"IrMakeClosure references unknown function id {fn_id.value!r}"
                )
                return IrClosureValue(
                    function_id=fn_id,
                    captures=tuple(cap_slots),
                    param_labels=function_desc.param_labels,
                    arity=len(function_desc.params),
                    result_label=function_desc.result_label,
                )

            case IrDirectCall() | IrIndirectCall():
                # A call unwinding an AglRaise surfaces its own site's location
                # when the error does not already carry a more specific span.
                # Both call kinds share this defaulting; keeping it in one arm
                # also avoids an extra Python stack frame per call, which the
                # recursive hot path (see DEFAULT_MAX_CALL_DEPTH) cannot spare.
                try:
                    if isinstance(node, IrDirectCall):
                        return self._execute_direct_call(
                            node.function_id, node.arguments, node.location
                        )
                    return self._execute_indirect_call(node.callee, node.arguments, node.location)
                except AglRaise as exc:
                    if exc.span is None:
                        exc.span = node.location
                    raise

            case IrPrint(value=val_expr):
                rendered = self._render_or_raise(self._eval(val_expr))
                print(rendered)
                self._trace.print_stmt(rendered=rendered, span=node.location)
                return VOID_VALUE

            case IrRenderValue(
                value=val_expr,
                pretty=pretty_expr,
                quote_strings=quote_strings_expr,
            ):
                pretty = (
                    self._eval_render_bool_option(pretty_expr, "pretty")
                    if pretty_expr is not None
                    else True
                )
                quote_strings = (
                    self._eval_render_bool_option(quote_strings_expr, "quote_strings")
                    if quote_strings_expr is not None
                    else True
                )
                return TextValue(
                    self._render_or_raise(
                        self._eval(val_expr), pretty=pretty, quote_strings=quote_strings
                    )
                )

            case IrParseJson(value=val_expr):
                val = self._eval(val_expr)
                if not isinstance(val, TextValue):
                    raise InvalidIrError(
                        f"IrParseJson: expected TextValue, got {type(val).__name__}"
                    )
                try:
                    obj = parse_json_strict(val.value)
                except StrictJsonParseError as exc:
                    raise AglRaise(
                        _make_exc_value(
                            "JsonParseError",
                            exc.message,
                            nominals=self._program.builtin_nominals,
                            raw=TextValue(val.value),
                        ),
                        span=node.location,
                    ) from exc
                return JsonValue(obj)

            case IrCopyValue(kind=kind, value=val_expr):
                value = self._eval(val_expr)
                return (
                    deep_copy_value(value) if kind is CopyKind.DEEP else shallow_copy_value(value)
                )

            case IrAsk(
                agent=agent_expr,
                prompt=prompt_expr,
                contract_id=contract_id,
                max_attempts=max_attempts,
            ):
                try:
                    return self._effects.eval_ir_ask(
                        node, agent_expr, prompt_expr, contract_id, max_attempts
                    )
                except AglRaise as exc:
                    if exc.span is None:
                        exc.span = node.location
                    raise

            case IrAskRequest(agent=agent_expr, prompt=prompt_expr):
                return self._effects.eval_ir_ask_request(node, agent_expr, prompt_expr)

            case IrExec(
                command=command_expr,
                contract_id=contract_id,
                max_attempts=max_attempts,
            ):
                try:
                    return self._effects.eval_ir_exec(node, command_expr, contract_id, max_attempts)
                except AglRaise as exc:
                    if exc.span is None:
                        exc.span = node.location
                    raise

            case IrBuiltinLoad(key=key):
                return self._load_builtin_setting(key)

            case IrBuiltinStore(key=key, value=value_expr):
                stored = self._eval(value_expr)
                try:
                    self._store_builtin_setting(key, stored)
                except AglRaise as exc:
                    exc.span = node.location
                    raise
                return VOID_VALUE

            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)

    def _check_default_agent_dispatchable(self, value: EnumValue) -> None:
        """Eagerly validate a materialized ``default-agent`` value's command shape.

        Only the ``AgentCommand`` variant needs this: its command text is
        host-supplied and must shell-split, exactly like a configured runner
        command.  The other ``Agent`` variants build their argv from typed
        fields with no parsing step, so decoding and building argv for them
        here is cheap and can never fail — reusing
        :func:`~agm.agl.runtime.agents.decode_agent_value` keeps this in sync
        with the variant shapes dispatch itself relies on, rather than
        duplicating them.
        """
        from agm.agl.runtime.agents import decode_agent_value

        try:
            decode_agent_value(value).argv()
        except ValueError as exc:
            raise HostConfigurationError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Builtin-var register access
    # ------------------------------------------------------------------

    def _load_builtin_setting(self, key: str) -> Value:
        """Return the current value of the ``builtin var`` engine setting *key*.

        The runtime-live keys read the live interpreter fields; the
        host-consumed keys read their register in ``_builtin_host_settings``.
        A host-consumed key with neither a host seed nor a declared default has
        no register, and no value to produce.

        :raises InvalidIrError: if *key* has no host-consumed register.
        """
        if key == "strict-json":
            return BoolValue(self._strict_json)
        if key == "max-iters":
            return IntValue(0 if self._loop_limit is None else self._loop_limit)
        if key == "timeout":
            return self._timeout_setting
        if key not in self._builtin_host_settings:
            raise InvalidIrError(
                f"builtin var {key!r} has no host-consumed register value: it was neither "
                "seeded by the host nor given a declaration default"
            )
        return self._builtin_host_settings[key]

    def _store_builtin_setting(self, key: str, value: Value) -> None:
        """Store *value* into the ``builtin var`` engine setting *key*.

        The three runtime-live keys route through ``_apply_config_effect`` so the
        live effect (loop cap, strict-json mode, shell timeout) takes hold from
        the write onward; the host-consumed keys update their register.
        Writes to the ``log``/``log-file`` trace-register pair additionally
        reconfigure the live trace service when a host reconfigurer is present;
        ``default-agent`` remains a register-only value.
        """
        if key in RUNTIME_LIVE_ENGINE_KEYS:
            self._apply_config_effect(key, value)
            if key == "timeout":
                assert isinstance(value, EnumValue)
                self._timeout_setting = value
            return

        previous = dict(self._builtin_host_settings)
        self._builtin_host_settings[key] = value
        if trace_write_implies_enabled(
            key, isinstance(value, EnumValue) and value.variant == "Some"
        ):
            self._builtin_host_settings["log"] = BoolValue(True)
        if self._host_reconfigurer is None or key not in TRACE_ENGINE_KEYS:
            return
        try:
            self._reconfigure_host_service()
        except Exception:
            self._builtin_host_settings = previous
            raise

    def _reconfigure_host_service(self) -> None:
        """Reflect a host-consumed register write into the live host service.

        ``log``/``log-file`` recompute the trace destination from the current
        register pair (either write repoints the same trace store).
        """
        assert self._host_reconfigurer is not None
        log = self._builtin_host_settings["log"]
        assert isinstance(log, BoolValue)
        log_file_reg = self._builtin_host_settings["log-file"]
        assert isinstance(log_file_reg, EnumValue)
        self._host_reconfigurer.reconfigure_trace(
            enabled=log.value, log_file=option_text(log_file_reg)
        )

    # ------------------------------------------------------------------
    # Engine-setting effect
    # ------------------------------------------------------------------

    def _apply_config_effect(self, public_name: str, config_value: Value) -> None:
        """Apply the live engine-setting effect for a runtime-live engine key.

        Only ``strict-json``, ``max-iters``, and ``timeout`` update live
        interpreter state; all other keys are inert here.
        """
        if public_name == "strict-json":
            assert isinstance(config_value, BoolValue)
            self._strict_json = config_value.value
        elif public_name == "max-iters":
            assert isinstance(config_value, IntValue)
            if config_value.value < 0:
                raise AglRaise(
                    _make_exc_value(
                        "TypeError",
                        "invalid max-iters: expected a non-negative integer",
                        nominals=self._program.builtin_nominals,
                    )
                )
            self._loop_limit = config_value.value or None
        else:
            assert public_name == "timeout"
            assert isinstance(config_value, EnumValue)
            raw = option_text(config_value)
            if raw is None:
                self._shell_exec_timeout = None
            else:
                try:
                    self._shell_exec_timeout = _parse_timeout(raw)
                except ValueError as exc:
                    raise AglRaise(
                        _make_exc_value(
                            "TypeError",
                            f"invalid timeout: {exc}",
                            nominals=self._program.builtin_nominals,
                        )
                    ) from exc

    # ------------------------------------------------------------------
    # Result collection
    # ------------------------------------------------------------------

    def _collect_results(self) -> dict[str, Value]:
        """Return file constants and, for wrapped input, final main locals.

        A file run exposes entry-module bindings other than function closures.
        Wrapped input additionally exposes the synthesized ``main`` frame
        retained at return time.
        """
        entry_id: ModuleId = self._program.entry_module
        function_symbols = {
            function.function_symbol for function in self._program.functions.values()
        }
        results: dict[str, Value] = {}

        def collect(frame: Frame, *, module_only: bool) -> None:
            for sym_id, desc in self._program.symbols.items():
                if (
                    desc.public_name is None
                    or sym_id in function_symbols
                    or (module_only and desc.owner != entry_id)
                ):
                    continue
                slot = frame.get(sym_id)
                if slot is not None:
                    results[desc.public_name] = slot.value if isinstance(slot, Cell) else slot

        collect(self._frames[0], module_only=True)
        if self._synthetic_main_frame is not None:
            collect(self._synthetic_main_frame, module_only=False)
        return results
