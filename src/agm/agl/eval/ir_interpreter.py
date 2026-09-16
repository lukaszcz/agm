"""IR evaluator for the AgL typeless execution IR.

``IrInterpreter`` executes an ``ExecutableProgram`` using the per-invocation
frame / let-by-value / var-by-cell model.

Allowed imports:
- ``agm.agl.ir.*``
- ``agm.agl.semantics.values`` (all value types, Cell, Frame)
- ``agm.agl.semantics.exceptions`` (AglRaise, make_builtin_exception)
- ``agm.agl.semantics.copying`` (deep_copy_value, shallow_copy_value)
- ``agm.agl.eval._decimal`` (shared pinned decimal context)
- ``agm.agl.runtime.serialize`` (untyped coercion and static direct JSON
  construction)
- ``agm.config.engine_keys`` (the canonical engine-key catalog data leaf)
- ``agm.agent.spec`` (AGENT_SPECS, for the Agent enum's member names)

NOT allowed: ``agm.agl.syntax``, ``agm.agl.scope``, ``agm.agl.typecheck``.
"""

from __future__ import annotations

import decimal
import inspect
import sys
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, ContextManager, Protocol, TypeVar, assert_never, cast

from agm.agent.spec import AGENT_SPECS
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
from agm.agl.ir.builtin_nominals import NO_BUILTIN_DECLARATIONS, BuiltinNominals
from agm.agl.ir.builtin_vars import BuiltinVarKey, builtin_var_key, is_engine_builtin_var_key
from agm.agl.ir.contracts import (
    ContractRequest,
    ConversionFailureMode,
    EncodePlan,
    ScalarEncode,
)
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
    IrExec,
    IrExpr,
    IrField,
    IrFieldMode,
    IrFieldSet,
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
    IrMakeException,
    IrMakeJsonArray,
    IrMakeJsonObject,
    IrMakeRecord,
    IrNominalCaseKey,
    IrNominalCast,
    IrNominalIs,
    IrOr,
    IrPrint,
    IrRaise,
    IrRenderTemplate,
    IrRenderValue,
    IrResource,
    IrReturn,
    IrSequence,
    IrSessionAsk,
    IrSessionDefault,
    IrSessionOp,
    IrSessionOpen,
    IrTemplateText,
    IrTemplateValue,
    IrTry,
    IrUnary,
    IrUpdateRecord,
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
    ValueDescriptors,
)
from agm.agl.ir.static_keys import StaticBindingKey
from agm.agl.ir.validate import InvalidIrError
from agm.agl.modules.ids import STD_CONFIG_ID, STD_ENV_ID, ModuleId
from agm.agl.runtime.agents import AgentFn
from agm.agl.runtime.codec import ParseResult, _parse_contract_output
from agm.agl.runtime.engine_config import engine_default_settings
from agm.agl.runtime.externs import (
    AglCallableProxy,
    ExternCallWindow,
    ExternRegistry,
    ExternRuntimeState,
)
from agm.agl.runtime.option import none_value, option_text, some_value
from agm.agl.runtime.render import render_value
from agm.agl.runtime.serialize import encode_value
from agm.agl.runtime.sessions import AgentDispatcherSessionHost
from agm.agl.runtime.trace import TraceStore, noop_trace
from agm.agl.semantics.copying import deep_copy_value, shallow_copy_value
from agm.agl.semantics.cycles import AglCyclicValue, cyclic_value_raise
from agm.agl.semantics.exceptions import AglRaise
from agm.agl.semantics.exceptions import make_builtin_exception as _make_exc_value
from agm.agl.semantics.values import (
    UNIT_VALUE,
    ArrayValue,
    BoolValue,
    Cell,
    ConstructorValue,
    DecimalValue,
    DictValue,
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
    ENGINE_KEYS,
    HOST_CONSUMED_ENGINE_KEYS,
    RUNTIME_LIVE_ENGINE_KEYS,
    TRACE_ENGINE_KEYS,
    EngineKeyKind,
    trace_write_implies_enabled,
)
from agm.core.cleanup import preserve_primary_error
from agm.core.parse import format_timeout as _format_timeout
from agm.core.parse import parse_timeout as _parse_timeout

if TYPE_CHECKING:
    from agm.agl.runtime.contract import OutputContract
    from agm.agl.runtime.host_settings import HostSettingsReconfigurer
    from agm.agl.runtime.sessions import SessionHost

__all__ = [
    "HostConfigurationError",
    "MissingBuiltinVarSeedError",
    "IrInterpreter",
    "_apply_coercion",
    "_make_exc_value",
]


_SCALAR_ENCODE_PLAN = EncodePlan(ScalarEncode())

_ArgT = TypeVar("_ArgT")


def _engine_key_shape(kind: EngineKeyKind) -> tuple[str, tuple[str, ...]] | None:
    """Return the ``(enum name, member names)`` an engine key *kind* restamps, if any."""
    if kind is EngineKeyKind.AGENT:
        return ("Agent", tuple(AGENT_SPECS))
    if kind is EngineKeyKind.OPTION_TEXT:
        return ("Option", ("None", "Some"))
    return None


#: Engine key name -> ``(enum name, member names)``, built once from ``ENGINE_KEYS``.
_ENGINE_KEY_ENUM_SHAPES: dict[str, tuple[str, tuple[str, ...]]] = {
    spec.name: shape for spec in ENGINE_KEYS if (shape := _engine_key_shape(spec.kind)) is not None
}


def _engine_key_enum_shape(key: str) -> tuple[str, tuple[str, ...]] | None:
    """Return the ``(enum name, member names)`` an enum-backed engine key restamps.

    ``None`` for a key whose kind carries no host-enum identity.
    """
    return _ENGINE_KEY_ENUM_SHAPES.get(key)


def _restamp_host_enum_member(
    value: RecordValue,
    *,
    enum_name: str,
    member_names: tuple[str, ...],
    from_table: BuiltinNominals,
    to_table: BuiltinNominals,
) -> RecordValue:
    """Restamp *value*'s identity from *from_table* to *to_table*, if it is an *enum_name* member.

    Used both to bind a persisted engine-setting seed (reserved fallback
    identity) onto this program's own nominal table, and to persist a
    post-run engine-setting value (this program's identity) back onto the
    reserved fallback table so it survives past this program's own lifetime.
    """
    for member_name in member_names:
        source = from_table.resolve_standard_member(enum_name, member_name)
        if value.nominal == source.nominal:
            target = to_table.resolve_standard_member(enum_name, member_name)
            return RecordValue(nominal=target.nominal, fields=value.fields)
    return value


def _restamp_engine_setting(
    key: str, value: Value, *, from_table: BuiltinNominals, to_table: BuiltinNominals
) -> Value:
    """Restamp *value* onto *to_table*'s identity when *key* is enum-backed."""
    if not isinstance(value, RecordValue):
        return value
    shape = _engine_key_enum_shape(key)
    if shape is None:
        return value
    enum_name, member_names = shape
    return _restamp_host_enum_member(
        value,
        enum_name=enum_name,
        member_names=member_names,
        from_table=from_table,
        to_table=to_table,
    )


class HostConfigurationError(Exception):
    """A host-backed value cannot be materialized as a valid AgL binding.

    The evaluator raises this for a malformed startup ``default-agent`` and
    for a read of a non-engine ``builtin var`` that has neither a host seed nor
    a declared default. Pipeline hosts translate it to an ordinary language
    diagnostic rather than leaking a Python implementation exception.
    """


class MissingBuiltinVarSeedError(HostConfigurationError):
    """A non-engine host-backed binding was read without a value source."""

    def __init__(self, key: BuiltinVarKey) -> None:
        module_id, scope_path, name = key
        scoped_name = "::".join((*scope_path, name))
        super().__init__(
            f"builtin var '{module_id.path_str()}::{scoped_name}' has no value: "
            "the host did not seed it and its declaration has no default"
        )


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
    value: Value,
    nominal: NominalId,
    field: str,
    mode: IrFieldMode,
) -> Value:
    """Read one declared field from a nominal runtime value.

    Records, enum payloads, and exceptions all retain their nominal identity and
    field mapping at runtime, so this is the one typeless projection mechanism.
    Exact projections require identity equality; upper-bound projections do not
    check identity because the static layer already proved the field exists on
    every value admitted by the bound.
    """
    if not isinstance(value, (RecordValue, ExceptionValue)):
        raise InvalidIrError(
            f"IrField: expected RecordValue or ExceptionValue, got {type(value).__name__}"
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


def _noop() -> None:
    """Provide an inert cleanup action when this interpreter owns no sessions."""


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
            return JsonValue(encode_value(_SCALAR_ENCODE_PLAN, value))

        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


# ---------------------------------------------------------------------------
# IrInterpreter
# ---------------------------------------------------------------------------


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
        agent_dispatcher: AgentFn | None = None,
        session_host: "SessionHost | None" = None,
        close_sessions: bool = True,
        strict_json: bool = False,
        shell_exec_timeout: float | None = None,
        host_contracts: Mapping[ContractId, "OutputContract"] | None = None,
        base_frame: Frame | None = None,
        extern_registry: ExternRegistry | None = None,
        host_reconfigurer: "HostSettingsReconfigurer | None" = None,
        builtin_host_settings: Mapping[str | BuiltinVarKey, Value] | None = None,
        param_seeds: Mapping[StaticBindingKey, Value] | None = None,
        process_environment: Mapping[str, str] | None = None,
    ) -> None:
        self._program = program
        self._descriptors = ValueDescriptors.from_program(program)
        self._frames: list[Frame] = [base_frame if base_frame is not None else {}]
        self.initializer_values: list[Value] = []
        self.module_initializer_values: dict[ModuleId, list[Value]] = {}
        self.module_completed_initializer_indices: dict[ModuleId, set[int]] = {}
        self._initializer_indices = {
            id(initializer): index
            for module in program.modules.values()
            for index, initializer in enumerate(module.initializers)
        }
        self._synthetic_main_frame: Frame | None = None
        self._call_depth: int = 0
        self._trace: TraceStore = trace if trace is not None else noop_trace()
        self._max_call_depth: int = max_call_depth
        self._agent_dispatcher = agent_dispatcher
        self._session_host: SessionHost = (
            session_host
            if session_host is not None
            else AgentDispatcherSessionHost(agent_dispatcher)
        )
        self._close_sessions = close_sessions
        # Bootstrap the setting fields so declared defaults can be evaluated by
        # the ordinary, typeless evaluator. Constant defaults cannot read a
        # setting or invoke a host operation, so this temporary state is never
        # observable by their evaluation.
        self._strict_json = False
        self._shell_exec_timeout: float | None = None
        self._timeout_setting = none_value(nominals=self._program.builtin_nominals)
        self._builtin_host_settings: dict[str, Value] = {}
        self._builtin_vars: dict[BuiltinVarKey, Value] = {}
        self._seeded_symbols: dict[SymbolId, Value] = {
            symbol: value
            for key, value in (param_seeds or {}).items()
            if (symbol := self._program.param_bindings.get(key)) is not None
        }
        self._host_reconfigurer = host_reconfigurer

        defaults: dict[BuiltinVarKey, Value] = {
            builtin_var_key(STD_CONFIG_ID, (), key): value
            for key, value in _engine_default_settings().items()
        }
        defaults.update(
            {
                self._builtin_var_key(key): self._eval(value)
                for key, value in self._program.builtin_setting_defaults.items()
            }
        )
        seed: dict[BuiltinVarKey, Value] = {
            (builtin_var_key(STD_CONFIG_ID, (), key) if isinstance(key, str) else key): value
            for key, value in (builtin_host_settings or {}).items()
        }
        if process_environment is not None:
            environ_seed = self._process_environ_seed(process_environment)
            if environ_seed is not None:
                seed.setdefault(environ_seed[0], environ_seed[1])

        # Runtime-live settings use an explicit host seed when present.  Their
        # driver arguments remain compatibility fallbacks: an absent false/None
        # must not suppress a declaration default.  Bootstrap through the same
        # effect path as a source write so host-invalid declared values become
        # normal AgL runtime errors.
        strict_key = builtin_var_key(STD_CONFIG_ID, (), "strict-json")
        strict_setting = seed.get(strict_key)
        if strict_setting is None:
            strict_default = defaults[strict_key]
            assert isinstance(strict_default, BoolValue)
            strict_setting = BoolValue(strict_json or strict_default.value)
        assert isinstance(strict_setting, BoolValue)
        self._apply_config_effect("strict-json", strict_setting)

        timeout_key = builtin_var_key(STD_CONFIG_ID, (), "timeout")
        timeout_setting = seed.get(timeout_key)
        if timeout_setting is None:
            timeout_setting = (
                some_value(
                    TextValue(_format_timeout(shell_exec_timeout)),
                    nominals=self._program.builtin_nominals,
                )
                if shell_exec_timeout is not None
                else defaults[timeout_key]
            )
        assert isinstance(timeout_setting, RecordValue)
        timeout_setting = cast(
            RecordValue,
            _restamp_engine_setting(
                "timeout",
                timeout_setting,
                from_table=NO_BUILTIN_DECLARATIONS,
                to_table=self._program.builtin_nominals,
            ),
        )
        self._timeout_setting = timeout_setting
        self._apply_config_effect("timeout", timeout_setting)

        # Host-consumed registers use the host seed when one is provided and
        # otherwise the ``builtin var`` declaration's default. A key with
        # neither gets no register at all rather than a fabricated value;
        # reading it is then a hard error (see ``_load_builtin_setting``).
        effective = {**defaults, **seed}
        self._builtin_host_settings = {
            key: effective[builtin_var_key(STD_CONFIG_ID, (), key)]
            for key in HOST_CONSUMED_ENGINE_KEYS
            if builtin_var_key(STD_CONFIG_ID, (), key) in effective
        }
        self._builtin_vars = {
            key: value for key, value in effective.items() if not is_engine_builtin_var_key(key)
        }
        default_agent = self._builtin_host_settings.get("default-agent")
        if isinstance(default_agent, RecordValue):
            default_agent = cast(
                RecordValue,
                _restamp_engine_setting(
                    "default-agent",
                    default_agent,
                    from_table=NO_BUILTIN_DECLARATIONS,
                    to_table=self._program.builtin_nominals,
                ),
            )
            self._builtin_host_settings["default-agent"] = default_agent
            self._check_default_agent_dispatchable(default_agent)
        log_file = self._builtin_host_settings.get("log-file")
        # ``log-file`` always has a declared default (unlike ``default-agent``),
        # so it is always present here.
        assert isinstance(log_file, RecordValue)
        self._builtin_host_settings["log-file"] = _restamp_engine_setting(
            "log-file",
            log_file,
            from_table=NO_BUILTIN_DECLARATIONS,
            to_table=self._program.builtin_nominals,
        )
        if self._host_reconfigurer is not None:
            self._reconfigure_host_service()
        self._host_contracts: Mapping[ContractId, OutputContract] = (
            host_contracts if host_contracts is not None else {}
        )
        self._extern_registry: ExternRegistry = (
            extern_registry if extern_registry is not None else ExternRegistry()
        )
        self._extern_call_window_guard = ExternCallWindow()
        self._extern_runtime_state = ExternRuntimeState()
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
    def timeout_setting(self) -> RecordValue:
        """Current raw ``Option[text]`` timeout value, in reserved fallback identity.

        A host that persists this past the run that produced it (the REPL)
        needs it recognizable once this run's own compiled program is gone --
        see :func:`_restamp_host_enum_member`.
        """
        value = _restamp_engine_setting(
            "timeout",
            self._timeout_setting,
            from_table=self._program.builtin_nominals,
            to_table=NO_BUILTIN_DECLARATIONS,
        )
        assert isinstance(value, RecordValue)
        return value

    @property
    def shell_exec_timeout(self) -> float | None:
        """Current shell-exec timeout (may have been updated by a ``builtin var`` write)."""
        return self._shell_exec_timeout

    @property
    def builtin_vars(self) -> dict[BuiltinVarKey, Value]:
        """Current non-engine host-backed bindings for incremental hosts."""
        return dict(self._builtin_vars)

    @property
    def builtin_host_settings(self) -> dict[str, Value]:
        """Current host-consumed register values, in reserved fallback identity.

        A snapshot of the registers backing the host-consumed ``builtin var``
        engine settings, reflecting any writes made during the run.  A key
        with neither a host seed nor a declared default is absent.  Hosts that
        persist settings across runs (the REPL) read this back after a run to
        seed the next one; an enum-backed value (``Option``/``Agent``) is
        already restamped onto the reserved fallback identity, so such a host
        needs no program-specific nominal table of its own -- see
        :func:`_restamp_host_enum_member`.
        """
        return {
            key: _restamp_engine_setting(
                key,
                value,
                from_table=self._program.builtin_nominals,
                to_table=NO_BUILTIN_DECLARATIONS,
            )
            for key, value in self._builtin_host_settings.items()
        }

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
                        str(err),
                        nominals=self._program.builtin_nominals,
                        fields={
                            "index": IntValue(err.index),
                            "length": IntValue(err.length),
                        },
                    ),
                )
            case AglMissingKey():
                return AglRaise(
                    _make_exc_value(
                        "KeyError",
                        f"Dict key {err.key!r} is missing",
                        nominals=self._program.builtin_nominals,
                        fields={
                            "key": TextValue(err.key),
                        },
                    ),
                )
            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)

    def _cyclic_failure(self) -> AglRaise:
        """Convert a detected reference cycle into an ``AglRaise(CyclicValueError)``.

        Mirrors ``_index_failure``: centralizes the sentinel-to-exception
        conversion so every ``render``/``as json``/coercion site that can
        reach a cyclic value raises identical exception fields.
        """
        return cyclic_value_raise(nominals=self._program.builtin_nominals)

    def _render_or_raise(
        self, value: Value, *, pretty: bool = False, quote_strings: bool = False
    ) -> str:
        """Render a value, converting only the rendering cycle sentinel."""
        try:
            return render_value(
                value, self._descriptors, pretty=pretty, quote_strings=quote_strings
            )
        except AglCyclicValue:
            raise self._cyclic_failure()

    def _on_cast_failure(
        self, failure_mode: ConversionFailureMode, exc: AglCastConversion
    ) -> Value:
        """Handle a fallible-cast failure per the conversion failure mode."""
        match failure_mode:
            case ConversionFailureMode.RAISE_CAST_ERROR:
                raise self._cast_conversion_raise("CastError", exc)
            case ConversionFailureMode.RAISE_VALUE_PARSE_ERROR:
                raise self._cast_conversion_raise("ValueParseError", exc)
            case ConversionFailureMode.RETURN_OPTION:
                return self._option_none()
            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)

    def _option_some(self, value: Value) -> Value:
        """Build the ``Option::Some`` an ``as?`` conversion evaluates to."""
        return some_value(value, nominals=self._program.builtin_nominals, declared=True)

    def _option_none(self) -> Value:
        """Build the ``Option::None`` a failed ``as?`` conversion evaluates to."""
        return none_value(nominals=self._program.builtin_nominals, declared=True)

    def _cast_conversion_raise(self, exception_name: str, exc: AglCastConversion) -> AglRaise:
        """Build the ``AglRaise`` for a failed cast-like conversion, by exception name.

        Shared by ``RAISE_CAST_ERROR`` (``CastError``) and
        ``RAISE_VALUE_PARSE_ERROR`` (``ValueParseError``): both exceptions carry
        the identical ``source-type``/``target-type``/``raw`` field shape.
        """
        return AglRaise(
            _make_exc_value(
                exception_name,
                exc.message,
                nominals=self._program.builtin_nominals,
                fields={
                    "source-type": TextValue(exc.source_label),
                    "target-type": TextValue(exc.target_label),
                    "raw": TextValue(exc.raw),
                },
            ),
        )

    def _cast_raw(self, value: Value) -> str:
        """Render a failed nominal cast without letting a cycle mask CastError."""
        try:
            return render_value(value, self._descriptors)
        except AglCyclicValue:
            return "<cyclic value>"

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
                        self.module_completed_initializer_indices.setdefault(
                            module.module_id, set()
                        ).add(self._initializer_indices[id(node)])
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
                fields={
                    "limit": IntValue(self._max_call_depth),
                },
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

    def _extern_call_window(self) -> ContextManager[None]:
        """Open this interpreter's callback window for one extern invocation."""
        return self._extern_call_window_guard.active()

    def _make_extern_callable_proxy(self, closure: IrClosureValue) -> AglCallableProxy:
        """Wrap one AgL closure for a companion's synchronous callback."""

        def invoke(args: tuple[Value, ...]) -> Value:
            return self._invoke_crossed_closure(closure, args)

        return AglCallableProxy(
            arity=len(self._program.functions[closure.function_id].params),
            closure=closure,
            require_active_window=self._extern_call_window_guard.require_active,
            invoke=invoke,
        )

    def _invoke_crossed_closure(self, closure: IrClosureValue, args: tuple[Value, ...]) -> Value:
        """Re-enter this interpreter to execute an AgL callback from an extern."""
        desc = self._program.functions[closure.function_id]
        match desc.impl:
            case ExternFunctionBody() as extern:
                return self._effects.eval_extern_call(desc.module_id, extern, args)
            case IrFunctionBody(body=body):
                self._check_call_depth()
                return self._bind_and_invoke(desc, body, closure, list(args))
            case other:  # pragma: no cover
                assert_never(other)

    def _resolve_defaults_and_invoke(
        self,
        desc: "FunctionDescriptor",
        body: IrExpr,
        closure_val: IrClosureValue,
        arguments: "tuple[object, ...]",
        eval_arg: "Callable[[_ArgT], Value]",
        *,
        retain_frame: bool = False,
    ) -> Value:
        """Resolve each argument against the callee, in one positional pass, then invoke.

        The tail shared by the IR direct-call path (:meth:`_execute_direct_call`)
        and the program entry point (:meth:`_invoke_program`). A single loop
        walks *arguments* in parameter order: a plain argument is turned into a
        value by *eval_arg* (``self._eval`` in the caller frame for a direct
        call, or the identity function for the program entry point's
        already-evaluated arguments), and a ``UseDefault`` argument evaluates
        that parameter's default expression in the callee's captures frame —
        interleaved in argument order, as an omitted argument does. Every
        non-``UseDefault`` element of *arguments* is an ``_ArgT``, matching
        *eval_arg*'s own parameter type; the caller's own signature enforces
        that, so the per-element cast below only restates it for the type
        checker.
        """
        bound_values: list[Value] = []
        for param, arg in zip(desc.params, arguments, strict=True):
            val = (
                self._eval_default_in_frame(param, dict(closure_val.captures))
                if isinstance(arg, UseDefault)
                else eval_arg(cast(_ArgT, arg))
            )
            bound_values.append(val)
        return self._bind_and_invoke(
            desc, body, closure_val, bound_values, retain_frame=retain_frame
        )

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
        AgL frame to recurse into).  Otherwise: depth check → single
        positional pass over *arguments*, evaluating each plain one in the
        caller frame and each ``UseDefault`` one against the callee's
        captures frame (:meth:`_resolve_defaults_and_invoke`).
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
                return self._resolve_defaults_and_invoke(
                    desc, body, closure_val, arguments, self._eval, retain_frame=retain_frame
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
            fields = {
                name: self._eval(argument)
                for name, argument in zip(constructor_desc.fields, arguments, strict=True)
            }
            return RecordValue(nominal=callee_val.nominal, fields=fields)
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

    def _program_entry(self, symbol: SymbolId) -> "tuple[FunctionDescriptor, IrFunctionBody]":
        """Look up a selected ``program def``'s descriptor and body."""
        descriptor = self._program.functions[self._program.program_functions[symbol]]
        assert isinstance(descriptor.impl, IrFunctionBody)
        return descriptor, descriptor.impl

    def _program_entry_location(self, symbol: SymbolId) -> Location:
        """Return the selected ``program def`` body's source location."""
        return self._program_entry(symbol)[1].body.location

    def _invoke_program(
        self, symbol: SymbolId, arguments: "tuple[Value | UseDefault, ...]"
    ) -> Value:
        """Invoke a selected linked ``program def`` with pre-evaluated arguments.

        Reuses :meth:`_resolve_defaults_and_invoke`, the same tail an ordinary
        direct call uses, so the entry point binds exactly like a call to the
        same function descriptor, with an entry-point error span attached to
        any depth-limit or body error.
        """
        desc, impl = self._program_entry(symbol)
        location = impl.body.location
        try:
            self._check_call_depth()
            closure_val = self._get_closure_for(desc.function_id)
            return self._resolve_defaults_and_invoke(
                desc,
                impl.body,
                closure_val,
                arguments,
                lambda value: value,
                retain_frame=symbol == self._program.synthetic_main_symbol,
            )
        except AglRaise as exc:
            if exc.span is None:
                exc.span = location
            raise

    def run(
        self,
        *,
        program_symbol: SymbolId | None = None,
        arguments: "tuple[Value | UseDefault, ...]" = (),
    ) -> dict[str, Value]:
        """Execute all modules in order and return the entry module's public bindings.

        Iterates over all modules in insertion order (library modules first,
        entry last), executing each module's initializers. When
        *program_symbol* is provided, invokes that selected ``program def``
        with *arguments* — its own pre-evaluated parameter values, positionally
        matching its signature, with ``UseDefault`` in place of an omitted
        defaulted parameter — before leaving the same managed execution
        boundary. Arguments are bound after every module initializer has run,
        so a defaulted parameter's default may read module bindings. All
        evaluation runs under the pinned AgL decimal context.

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
        cleanup = self._session_host.close_all if self._close_sessions else _noop
        try:
            with preserve_primary_error(cleanup, label="agent session cleanup"):
                with decimal.localcontext(AGL_DECIMAL_CONTEXT):
                    self._install_function_closures()
                    for mod in self._program.modules.values():
                        for node in mod.initializers:
                            self._eval_and_record_initializer(mod.module_id, node)
                    if program_symbol is not None:
                        try:
                            self._invoke_program(program_symbol, arguments)
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
        self.module_completed_initializer_indices.setdefault(module_id, set()).add(
            self._initializer_indices[id(node)]
        )

    def _eval_initializer(self, node: IrExpr) -> Value:
        match node:
            case IrBind(symbol=sym) if sym in self._seeded_symbols:
                value = self._seeded_symbols[sym]
                seed_desc = self._program.symbols.get(sym)
                if seed_desc is not None and seed_desc.mutable:
                    self._frame[sym] = Cell(value)
                else:
                    self._frame[sym] = value
                return value
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

    def _eval_session_effect(
        self, node: IrSessionOpen | IrSessionDefault | IrSessionAsk | IrSessionOp
    ) -> Value:
        try:
            if isinstance(node, IrSessionOpen):
                return self._effects.eval_ir_session_open(node)
            if isinstance(node, IrSessionDefault):
                return self._effects.eval_ir_session_default(
                    node, self._load_builtin_setting("default-agent")
                )
            if isinstance(node, IrSessionAsk):
                return self._effects.eval_ir_session_ask(node)
            return self._effects.eval_ir_session_op(node)
        except AglRaise as exc:
            if exc.span is None:
                exc.span = node.location
            raise

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
            # The scalar encode plan in both arms below is a leaf conversion, never a
            # walk that could re-enter a container — no cycle guard needed in either.
            case IrMakeJsonArray(items=json_items):
                return JsonValue(
                    [encode_value(_SCALAR_ENCODE_PLAN, self._eval(item)) for item in json_items]
                )

            case IrMakeJsonObject(entries=json_entries):
                json_result: dict[str, object] = {}
                for key_expr, val_expr in json_entries:
                    key_val = self._eval(key_expr)
                    if not isinstance(key_val, TextValue):
                        raise InvalidIrError(
                            f"IrMakeJsonObject key must evaluate to TextValue,"
                            f" got {type(key_val).__name__}"
                        )
                    json_result[key_val.value] = encode_value(
                        _SCALAR_ENCODE_PLAN, self._eval(val_expr)
                    )
                return JsonValue(json_result)

            case IrLoad(symbol=sym):
                slot = next(
                    (frame[sym] for frame in reversed(self._frames) if sym in frame),
                    None,
                )
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
                return UNIT_VALUE

            case IrFieldSet(value=value_expr, nominal=nominal, field=field, new=new_expr):
                # Evaluate the receiver before the replacement. The identity
                # guard protects superseded same-named declarations from
                # writes; validation already proved the field is declared, so
                # matching identity is all the store needs.
                value = self._eval(value_expr)
                if not isinstance(value, RecordValue):
                    raise InvalidIrError(
                        f"IrFieldSet: expected RecordValue, got {type(value).__name__}"
                    )
                if value.nominal != nominal:
                    raise InvalidIrError(
                        f"IrFieldSet: expected nominal {nominal!r}, got {value.nominal!r}"
                    )
                value.fields[field] = self._eval(new_expr)
                return UNIT_VALUE

            # Plain left-to-right evaluation order: container, then index,
            # then the right-hand side, then the checked in-place store.
            case IrIndexSet(container=container_expr, kind=kind, index=idx_expr, value=val_expr):
                container = self._eval(container_expr)
                index_val = self._eval(idx_expr)
                new_value = self._eval(val_expr)
                try:
                    index_set(kind, container, index_val, new_value)
                except (AglIndexOutOfRange, AglMissingKey) as e:
                    raise self._index_failure(e)
                return UNIT_VALUE

            case IrCoerce(value=val_expr, operation=op):
                value = self._eval(val_expr)
                return _apply_coercion(value, op)

            case IrSequence(items=items) | IrBlock(items=items):
                last: Value = UNIT_VALUE
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
                            fields={
                                "operation": TextValue("/"),
                            },
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
                value = self._eval(val_expr)
                return _project_nominal_field(value, nominal, field_name, mode)

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
                    return RecordValue(nominal=target.nominal, fields=updated_fields)
                return ExceptionValue(nominal=target.nominal, fields=updated_fields)

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

            case IrMakeRecord(nominal=nominal, fields=fields):
                record_fields: dict[str, Value] = {
                    fname: self._eval(fexpr) for fname, fexpr in fields
                }
                return RecordValue(nominal=nominal, fields=record_fields)

            case IrMakeException(nominal=nominal, fields=fields):
                exc_fields: dict[str, Value] = {
                    fname: self._eval(field_expr) for fname, field_expr in fields
                }
                return ExceptionValue(nominal=nominal, fields=exc_fields)

            case IrMakeConstructor(nominal=nominal):
                return ConstructorValue(nominal=nominal)

            case IrNominalCast(
                nominal=nominal,
                value=val_expr,
                test_only=test_only,
                source_label=source_label,
                target_label=target_label,
            ):
                value = self._eval(val_expr)
                if not isinstance(value, RecordValue):
                    raise InvalidIrError(
                        f"IrNominalCast: value is not a record, got {type(value).__name__}"
                    )
                if value.nominal == nominal:
                    return self._option_some(value) if test_only else value
                if test_only:
                    return self._option_none()
                raise AglRaise(
                    _make_exc_value(
                        "CastError",
                        f"cannot cast '{source_label}' to '{target_label}'",
                        nominals=self._program.builtin_nominals,
                        fields={
                            "source-type": TextValue(source_label),
                            "target-type": TextValue(target_label),
                            "raw": TextValue(self._cast_raw(value)),
                        },
                    )
                )

            case IrNominalIs(nominal=nominal, value=val_expr, negated=negated):
                value = self._eval(val_expr)
                if not isinstance(value, (RecordValue, ExceptionValue)):
                    raise InvalidIrError(
                        f"IrNominalIs: value is not nominal, got {type(value).__name__}"
                    )
                return BoolValue((value.nominal == nominal) != negated)

            case IrConvert(value=val_expr, recipe=recipe, failure_mode=failure_mode):
                source_value = self._eval(val_expr)
                try:
                    converted = run_recipe(recipe, source_value, self._descriptors)
                except AglCastConversion as exc:
                    return self._on_cast_failure(failure_mode, exc)
                except AglCyclicValue:
                    # A conversion test reports a cycle as failure; ordinary
                    # `as` still reports the catchable CyclicValueError.
                    if failure_mode is ConversionFailureMode.RETURN_OPTION:
                        return self._option_none()
                    raise self._cyclic_failure()
                if failure_mode is ConversionFailureMode.RETURN_OPTION:
                    return self._option_some(converted)
                return converted

            case IrIf(branches=branches, has_else=has_else):
                for branch in branches:
                    if branch.cond is None:
                        # Else branch — always taken.
                        branch_val = self._eval(branch.body)
                        return branch_val if has_else else UNIT_VALUE
                    cond_val = self._eval(branch.cond)
                    if not isinstance(cond_val, BoolValue):
                        raise InvalidIrError(
                            f"IrIf: branch condition evaluated to"
                            f" {type(cond_val).__name__}, expected BoolValue"
                        )
                    if cond_val.value:
                        branch_val = self._eval(branch.body)
                        return branch_val if has_else else UNIT_VALUE
                # No branch matched and no else: return unit.
                return UNIT_VALUE

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
                # The subject's identity is invariant across the arm scan, so it
                # is read once here rather than per arm.
                subject_nominal = (
                    subject_val.nominal
                    if isinstance(subject_val, (RecordValue, ExceptionValue))
                    else None
                )
                for arm in arms:
                    key = arm.key
                    if isinstance(key, IrNominalCaseKey):
                        selected = subject_nominal == key.nominal
                    else:
                        selected = value_eq(subject_val, _literal_key_value(key))
                    if not selected:
                        continue
                    if arm.field_bindings:
                        if not isinstance(subject_val, (RecordValue, ExceptionValue)):
                            raise InvalidIrError(
                                "IrCase: selected payload arm for a non-nominal subject"
                            )
                        assert isinstance(key, IrNominalCaseKey)
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

            case IrLoop(body=body_expr):
                # Unconditional repeat — all loop logic (bound checks, until
                # guards, for/while clauses) is desugared into the body by the
                # lowerer.  The only exits are IrBreak (leave the loop)
                # and IrContinue (next iteration).  Both signals propagate through
                # IrTry bodies (which catch only AglRaise) to reach this handler.
                while True:
                    try:
                        self._eval(body_expr)
                    except _BreakSignal:
                        return UNIT_VALUE
                    except _ContinueSignal:
                        continue

            case IrBreak():
                raise _BreakSignal()

            case IrContinue():
                raise _ContinueSignal()

            case IrIterInit(collection=collection_expr):
                coll = self._eval(collection_expr)
                if isinstance(coll, ArrayValue):
                    # Keep the live element list so mutations ahead of the
                    # cursor remain visible. IteratorValue captures its entry
                    # length so structural growth cannot extend the loop.
                    return IteratorValue(elements=coll.elements)
                if isinstance(coll, DictValue):
                    # The key set is fixed for the collection's lifetime, so a
                    # one-time tuple of keys is sound even though the values
                    # behind those keys may still be mutated.
                    return IteratorValue(elements=tuple(TextValue(k) for k in coll.entries))
                if isinstance(coll, TextValue):
                    return IteratorValue(elements=coll.value)
                raise InvalidIrError(  # pragma: no cover
                    f"IrIterInit: unexpected collection type {type(coll)!r}"
                )

            case IrIterHasNext(iterator=iter_expr):
                it = self._eval(iter_expr)
                if not isinstance(it, IteratorValue):  # pragma: no cover
                    raise InvalidIrError(f"IrIterHasNext: expected IteratorValue, got {type(it)!r}")
                return BoolValue(it.pos < it.entry_length and it.pos < len(it.elements))

            case IrIterNext(iterator=iter_expr):
                it = self._eval(iter_expr)
                if not isinstance(it, IteratorValue):  # pragma: no cover
                    raise InvalidIrError(f"IrIterNext: expected IteratorValue, got {type(it)!r}")
                elem = it.elements[it.pos]
                it.pos += 1
                return TextValue(elem) if isinstance(elem, str) else elem

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
                return IrClosureValue(function_id=fn_id, captures=tuple(cap_slots))

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
                return UNIT_VALUE

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

            case IrSessionOpen() | IrSessionDefault() | IrSessionAsk() | IrSessionOp():
                return self._eval_session_effect(node)

            case IrAskRequest(
                agent=agent_expr,
                prompt=prompt_expr,
                contract_id=contract_id,
                max_attempts=max_attempts,
            ):
                return self._effects.eval_ir_ask_request(
                    node, agent_expr, prompt_expr, contract_id, max_attempts
                )

            case IrExec(
                command=command_expr,
                env=env_expr,
                cwd=cwd_expr,
                timeout=timeout_expr,
                contract_id=contract_id,
                max_attempts=max_attempts,
            ):
                try:
                    return self._effects.eval_ir_exec(
                        node,
                        command_expr,
                        env_expr,
                        cwd_expr,
                        timeout_expr,
                        contract_id,
                        max_attempts,
                    )
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
                return UNIT_VALUE

            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)

    def _check_default_agent_dispatchable(self, value: RecordValue) -> None:
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
            decode_agent_value(value, self._program.builtin_nominals).argv()
        except ValueError as exc:
            raise HostConfigurationError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Builtin-var register access
    # ------------------------------------------------------------------

    def _process_environ_seed(
        self, process_environment: Mapping[str, str]
    ) -> tuple[BuiltinVarKey, RecordValue] | None:
        """Build ``std/env::environ`` from an immutable host environment snapshot."""
        descriptor = next(
            (
                descriptor
                for descriptor in self._program.nominals.values()
                if descriptor.module_id == STD_ENV_ID and descriptor.declared_name == "Environ"
            ),
            None,
        )
        if descriptor is None:
            return None
        return (
            builtin_var_key(STD_ENV_ID, (), "environ"),
            RecordValue(
                nominal=descriptor.nominal,
                fields={
                    "vars": DictValue(
                        {name: TextValue(value) for name, value in process_environment.items()}
                    )
                },
            ),
        )

    @staticmethod
    def _builtin_var_key(key: BuiltinVarKey | str) -> BuiltinVarKey:
        """Normalize legacy engine-only IR keys to their ``std/config`` owner."""
        return builtin_var_key(STD_CONFIG_ID, (), key) if isinstance(key, str) else key

    def _load_builtin_setting(self, key: BuiltinVarKey | str) -> Value:
        """Return the current value of the host-backed binding *key*.

        The runtime-live keys read the live interpreter fields; the
        host-consumed keys read their register in ``_builtin_host_settings``.
        A host-consumed key with neither a host seed nor a declared default has
        no register, and no value to produce.

        :raises InvalidIrError: if *key* has no host-consumed register.
        """
        key = self._builtin_var_key(key)
        _, _, name = key
        if not is_engine_builtin_var_key(key):
            try:
                return self._builtin_vars[key]
            except KeyError as exc:
                raise MissingBuiltinVarSeedError(key) from exc
        if name == "strict-json":
            return BoolValue(self._strict_json)
        if name == "timeout":
            return self._timeout_setting
        if name not in self._builtin_host_settings:
            raise InvalidIrError(
                f"builtin var {name!r} has no host-consumed register value: it was neither "
                "seeded by the host nor given a declaration default"
            )
        return self._builtin_host_settings[name]

    def _store_builtin_setting(self, key: BuiltinVarKey | str, value: Value) -> None:
        """Store *value* into the host-backed binding *key*.

        The runtime-live keys route through ``_apply_config_effect`` so the
        live effect (strict-json mode, shell timeout) takes hold from
        the write onward; the host-consumed keys update their register.
        Writes to the ``log``/``log-file`` trace-register pair additionally
        reconfigure the live trace service when a host reconfigurer is present;
        ``default-agent`` remains a register-only value.
        """
        key = self._builtin_var_key(key)
        _, _, name = key
        if not is_engine_builtin_var_key(key):
            self._builtin_vars[key] = value
            return
        if name in RUNTIME_LIVE_ENGINE_KEYS:
            self._apply_config_effect(name, value)
            if name == "timeout":
                assert isinstance(value, RecordValue)
                self._timeout_setting = value
            return

        previous = dict(self._builtin_host_settings)
        self._builtin_host_settings[name] = value
        if trace_write_implies_enabled(
            name,
            isinstance(value, RecordValue)
            and option_text(value, nominals=self._program.builtin_nominals) is not None,
        ):
            self._builtin_host_settings["log"] = BoolValue(True)
        if self._host_reconfigurer is None or name not in TRACE_ENGINE_KEYS:
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
        assert isinstance(log_file_reg, RecordValue)
        self._host_reconfigurer.reconfigure_trace(
            enabled=log.value,
            log_file=option_text(log_file_reg, nominals=self._program.builtin_nominals),
        )

    # ------------------------------------------------------------------
    # Engine-setting effect
    # ------------------------------------------------------------------

    def _apply_config_effect(self, public_name: str, config_value: Value) -> None:
        """Apply the live engine-setting effect for a runtime-live engine key.

        Only ``strict-json`` and ``timeout`` update live
        interpreter state; all other keys are inert here.
        """
        if public_name == "strict-json":
            assert isinstance(config_value, BoolValue)
            self._strict_json = config_value.value
        else:
            assert public_name == "timeout"
            assert isinstance(config_value, RecordValue)
            raw = option_text(config_value, nominals=self._program.builtin_nominals)
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
