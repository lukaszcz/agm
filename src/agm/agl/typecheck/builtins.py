"""Built-in call (print/render/parse_json/ask/ask-request/exec) type-checking collaborator.

Driven by ``_Checker`` via the narrow ``BuiltinCheckCtx`` Protocol.  All logic
lives here; the host checker instantiates ``BuiltinCallChecker(self)`` and
delegates the six built-in dispatch branches to the public entry points.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from agm.agl.capabilities import HostCapabilities
from agm.agl.diagnostics import Diagnostic
from agm.agl.semantics.types import (
    AgentType,
    BoolType,
    FunctionType,
    JsonType,
    TextType,
    Type,
    UnitType,
    contains_inference_var,
    contains_type_var,
    free_type_vars,
)
from agm.agl.syntax.nodes import (
    BoolLit,
    Call,
    Expr,
    IntLit,
    NamedArg,
    StringLit,
    VarRef,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck.env import (
    AglTypeError,
    CallSiteRecord,
    OutputContractSpec,
    TypeEnvironment,
)

# ---------------------------------------------------------------------------
# Deferred built-in obligations
# ---------------------------------------------------------------------------


class BuiltinObligationKind(StrEnum):
    """The built-in operation whose concrete output metadata is pending."""

    ASK = "ask"
    ASK_REQUEST = "ask-request"
    EXEC = "exec"


@dataclass(frozen=True, slots=True)
class PendingBuiltinObligation:
    """Syntax-derived data needed to materialize one built-in contract later.

    ``target_type`` and ``result_type`` may contain solver-owned variables while
    the enclosing inference region is open.  The checker zonks both before
    handing this record back to :class:`BuiltinCallChecker` at region close.
    No callback captures checker state, so a failed region can discard its
    whole obligation list without publishing partial side-table data.
    """

    node_id: int
    target_type: Type
    result_type: Type
    span: SourceSpan
    kind: BuiltinObligationKind
    format_name: str | None
    strict_json: bool | None
    parse_policy: str
    has_parse_error_option: bool
    has_parse_shaping_option: bool
    has_agent_argument: bool


# ---------------------------------------------------------------------------
# Narrow context Protocol
# ---------------------------------------------------------------------------


class BuiltinCheckCtx(Protocol):
    """The minimal _Checker surface the built-in call checker needs."""

    _env: TypeEnvironment
    _caps: HostCapabilities
    def _record_contract_spec(self, node_id: int, spec: OutputContractSpec) -> None: ...

    def _append_call_site(self, call_site: CallSiteRecord) -> None: ...

    def _append_warning(self, warning: Diagnostic) -> None: ...
    _current_type_vars: frozenset[str]

    def _check_expr(self, expr: Expr, *, expected: Type | None) -> Type: ...

    def _assert_assignable(
        self, value_type: Type, target_type: Type, span: SourceSpan
    ) -> None: ...

    def _type_is_wire_serializable(self, typ: Type) -> bool: ...

    def _register_builtin_obligation(self, obligation: PendingBuiltinObligation) -> None: ...


# ---------------------------------------------------------------------------
# Collaborator class
# ---------------------------------------------------------------------------


class BuiltinCallChecker:
    """Type-checking collaborator for built-in call nodes.

    Instantiated once per ``_Checker`` instance (``self._builtins``).
    All built-in dispatch in ``_check_call`` is delegated here.
    """

    _ASK_ALLOWED_NAMED_ARGS: frozenset[str] = frozenset(
        {"agent", "format", "strict_json", "on_parse_error"}
    )

    _EXEC_ALLOWED_NAMED_ARGS: frozenset[str] = frozenset(
        {"format", "strict_json", "on_parse_error"}
    )

    def __init__(self, ctx: BuiltinCheckCtx) -> None:
        self._ctx = ctx

    # --- print ---

    def check_print(self, node: Call) -> Type:
        if len(node.args) != 1 or node.named_args:
            raise AglTypeError(
                "print() requires exactly one positional argument.",
                span=node.span,
            )
        self._ctx._check_expr(node.args[0], expected=None)
        return UnitType()

    # --- render ---

    def check_render(self, node: Call) -> Type:
        if len(node.args) != 1:
            raise AglTypeError(
                "render() requires exactly one positional argument.",
                span=node.span,
            )
        allowed = {"pretty", "quote_strings"}
        for named in node.named_args:
            if named.name not in allowed:
                raise AglTypeError(
                    f"render() got unknown named argument {named.name!r}.",
                    span=named.span,
                )
            option_type = self._ctx._check_expr(named.value, expected=BoolType())
            self._ctx._assert_assignable(option_type, BoolType(), named.value.span)
        self._ctx._check_expr(node.args[0], expected=None)
        return TextType()

    # --- parse_json ---

    def check_parse_json(self, node: Call) -> Type:
        if len(node.args) != 1 or node.named_args:
            raise AglTypeError(
                "parse_json() requires exactly one positional text argument.",
                span=node.span,
            )
        arg_type = self._ctx._check_expr(node.args[0], expected=TextType())
        self._ctx._assert_assignable(arg_type, TextType(), node.args[0].span)
        return JsonType()

    # --- ask ---

    def check_ask(self, node: Call, *, expected: Type | None) -> Type:
        # Target type: explicit type argument overrides context.
        explicit = self._resolve_explicit_target(node, "ask")
        target_type: Type = explicit if explicit is not None else (
            expected if expected is not None else TextType()
        )
        self._reject_type_var_target(target_type, node.span)
        self._register_ask_like_obligation(
            node,
            target_type=target_type,
            result_type=target_type,
            kind=BuiltinObligationKind.ASK,
        )
        return target_type

    # --- ask-request ---

    def check_ask_request(self, node: Call) -> Type:
        """Type-check ``ask-request(prompt, ...)`` — the side-effect-free twin of ``ask``.

        Like ``ask`` it builds an output contract from a target type and the
        parse-shaping named args (``format`` / ``strict_json`` /
        ``on_parse_error``), and accepts an ``agent:`` named arg. But it never
        dispatches to the agent: it yields the ``AgentRequest`` record that the
        corresponding ``ask`` call would pass to ``AgentRegistry.dispatch`` on
        its first attempt.

        The target type is taken from the explicit type argument
        (``ask-request::[Review](...)``) when present, and defaults to ``text``
        otherwise (``ask-request(...)``).  Because the result type is fixed to
        ``AgentRequest``, the contextual ``expected`` type is ignored — unlike
        ``ask``, the target type is not inferred from context.
        """
        agent_request_type = self._ctx._env.get_type("AgentRequest")
        assert agent_request_type is not None, "AgentRequest prelude type missing"

        # Target type: explicit type argument, else text default.
        explicit = self._resolve_explicit_target(node, "ask-request")
        target_type = explicit if explicit is not None else TextType()
        self._reject_type_var_target(target_type, node.span)

        # Build the same output contract spec an ``ask`` call would, so the
        # materialized contract (and thus the returned request) matches exactly.
        # ``ask-request`` never dispatches, so a missing ``agent:`` is allowed.
        self._register_ask_like_obligation(
            node,
            target_type=target_type,
            result_type=agent_request_type,
            kind=BuiltinObligationKind.ASK_REQUEST,
        )
        return agent_request_type

    def _register_ask_like_obligation(
        self,
        node: Call,
        *,
        target_type: Type,
        result_type: Type,
        kind: BuiltinObligationKind,
    ) -> None:
        """Check target-independent syntax, then queue contract materialization."""
        callee = kind.value
        named = self._validate_ask_like_arguments(node, callee)
        format_name, strict_json, parse_policy = self._parse_options(named)
        self._ctx._register_builtin_obligation(
            PendingBuiltinObligation(
                node_id=node.node_id,
                target_type=target_type,
                result_type=result_type,
                span=node.span,
                kind=kind,
                format_name=format_name,
                strict_json=strict_json,
                parse_policy=parse_policy,
                has_parse_error_option="on_parse_error" in named,
                has_parse_shaping_option=any(
                    name in named for name in ("format", "strict_json", "on_parse_error")
                ),
                has_agent_argument="agent" in named,
            )
        )

    def _validate_ask_like_arguments(self, node: Call, callee: str) -> dict[str, NamedArg]:
        """Check syntax and value arguments that do not need the target type."""
        named = {na.name: na for na in node.named_args}
        for arg_name, na in named.items():
            if arg_name not in self._ASK_ALLOWED_NAMED_ARGS:
                raise AglTypeError(f"{callee}: unknown argument '{arg_name}'.", span=na.span)
        if not node.args:
            raise AglTypeError(f"{callee}() requires a prompt argument.", span=node.span)
        if len(node.args) > 1:
            raise AglTypeError(
                f"{callee}: too many positional arguments (expected 1).", span=node.span
            )
        prompt_type = self._ctx._check_expr(node.args[0], expected=TextType())
        self._ctx._assert_assignable(prompt_type, TextType(), node.args[0].span)
        if "agent" in named:
            agent_na = named["agent"]
            agent_type = self._ctx._check_expr(agent_na.value, expected=AgentType())
            self._ctx._assert_assignable(agent_type, AgentType(), agent_na.value.span)
        return named

    def finalize(self, obligation: PendingBuiltinObligation) -> None:
        """Materialize one fully zonked built-in obligation at region close."""
        target_type = obligation.target_type
        if contains_inference_var(target_type) or contains_inference_var(obligation.result_type):
            raise AglTypeError(
                "Cannot infer a concrete target type for this built-in call.", span=obligation.span
            )
        self._reject_type_var_target(target_type, obligation.span)
        if isinstance(target_type, (FunctionType, AgentType)):
            raise AglTypeError(
                "cannot parse agent or exec output into a function/agent value.",
                span=obligation.span,
            )
        if obligation.kind is BuiltinObligationKind.EXEC:
            self._finalize_exec(obligation)
        else:
            self._finalize_ask_like(obligation)

    def _finalize_ask_like(self, obligation: PendingBuiltinObligation) -> None:
        target_type = obligation.target_type
        callee = obligation.kind.value
        if (
            obligation.kind is BuiltinObligationKind.ASK
            and not obligation.has_agent_argument
            and not self._ctx._caps.has_default_agent
            and isinstance(target_type, TextType)
        ):
            raise AglTypeError(
                "No default agent is configured; the built-in 'ask' call cannot run. "
                "Register a default agent, or run via `agm exec`, which provides one.",
                span=obligation.span,
            )
        if isinstance(target_type, UnitType):
            if obligation.has_parse_shaping_option:
                raise AglTypeError(
                    f"{callee} returning unit does not accept parse options; unit responses "
                    "are ignored and have no output contract.",
                    span=obligation.span,
                )
            codec_name = "none"
            parse_policy = "default"
        else:
            codec_name, effective_strict = self._resolve_codec(obligation)
            self._check_schema_compilable(
                target_type, codec_name, obligation.span, use="an agent output type"
            )
            spec = OutputContractSpec(target_type, codec_name, effective_strict)
            assert not contains_inference_var(spec.target_type)
            self._ctx._record_contract_spec(obligation.node_id, spec)
            parse_policy = obligation.parse_policy
            self._warn_noop_parse_error_on_text(obligation)
        self._append_call_site(obligation, codec_name, parse_policy)

    def _warn_noop_parse_error_on_text(self, obligation: PendingBuiltinObligation) -> None:
        """Warn when ``on_parse_error`` is set on a text target, where it can never fire."""
        if not (
            obligation.has_parse_error_option and isinstance(obligation.target_type, TextType)
        ):
            return
        self._ctx._append_warning(
            Diagnostic(
                message=(
                    "'on_parse_error' has no effect on a text target: a text result "
                    "never fails parsing, so the policy can never fire."
                ),
                line=obligation.span.start_line,
                column=obligation.span.start_col,
                end_line=obligation.span.end_line,
                end_column=obligation.span.end_col,
                severity="warning",
            )
        )

    def _append_call_site(
        self, obligation: PendingBuiltinObligation, codec_name: str, parse_policy: str
    ) -> None:
        assert not contains_inference_var(obligation.target_type)
        self._ctx._append_call_site(
            CallSiteRecord(
                node_id=obligation.node_id,
                callee=obligation.kind.value,
                target_type=obligation.target_type,
                codec_name=codec_name,
                parse_policy=parse_policy,
                line=obligation.span.start_line,
                col=obligation.span.start_col,
            )
        )

    # --- exec ---

    def check_exec(self, node: Call, *, expected: Type | None) -> Type:
        if not self._ctx._caps.supports_shell_exec:
            raise AglTypeError(
                "The host does not support 'exec' (shell) calls.", span=node.span
            )

        exec_result_type = self._ctx._env.get_type("ExecResult")
        target_type: Type
        # Explicit type argument overrides context.
        explicit = self._resolve_explicit_target(node, "exec")
        if explicit is not None:
            target_type = explicit
        elif expected is not None:
            target_type = expected
        else:
            assert exec_result_type is not None
            target_type = exec_result_type
        self._reject_type_var_target(target_type, node.span)
        named = {na.name: na for na in node.named_args}
        for arg_name, na in named.items():
            if arg_name not in self._EXEC_ALLOWED_NAMED_ARGS:
                raise AglTypeError(f"exec: unknown argument '{arg_name}'.", span=na.span)
        if not node.args:
            raise AglTypeError("exec() requires a command argument.", span=node.span)
        if len(node.args) > 1:
            raise AglTypeError("exec: too many positional arguments (expected 1).", span=node.span)
        cmd_type = self._ctx._check_expr(node.args[0], expected=TextType())
        self._ctx._assert_assignable(cmd_type, TextType(), node.args[0].span)
        format_name, strict_json, parse_policy = self._parse_options(named)
        self._ctx._register_builtin_obligation(
            PendingBuiltinObligation(
                node_id=node.node_id,
                target_type=target_type,
                result_type=target_type,
                span=node.span,
                kind=BuiltinObligationKind.EXEC,
                format_name=format_name,
                strict_json=strict_json,
                parse_policy=parse_policy,
                has_parse_error_option="on_parse_error" in named,
                has_parse_shaping_option=any(
                    name in named for name in ("format", "strict_json", "on_parse_error")
                ),
                has_agent_argument=False,
            )
        )
        return target_type

    # --- shared explicit-target resolver for --

    def _resolve_explicit_target(self, node: Call, builtin_name: str) -> Type | None:
        """Resolve the explicit type argument of an ask/ask-request/exec call.

        Returns the resolved ``Type`` when ``node.type_args`` is non-empty, or
        ``None`` when there are no explicit type arguments (caller falls back to
        its contextual/default target logic).

        Raises ``AglTypeError`` when more than one type argument is provided
        (arity error). The  type-variable guard is applied by the caller to
        the *final* target type (see :meth:`_reject_type_var_target`), so it
        covers both the explicit and the contextual/inferred target paths.
        """
        if not node.type_args:
            return None
        if len(node.type_args) > 1:
            raise AglTypeError(
                f"{builtin_name} expects at most one explicit type argument; "
                f"got {len(node.type_args)}.",
                span=node.span,
            )
        return self._ctx._env.resolve_type_expr(
            node.type_args[0], span=node.span, type_vars=self._ctx._current_type_vars
        )

    def _reject_type_var_target(self, target_type: Type, span: SourceSpan) -> None:
        """an ask/exec/ask-request target type may not contain a type variable.

        Applied to the final resolved target — whether it came from an explicit
        ``::[…]`` argument or was inferred from the contextual expected type
        (e.g. a generic ``def``'s return type) — so a type variable never reaches
        codec selection or schema generation (which cannot serialise one).
        """
        if contains_type_var(target_type):
            tv = next(iter(free_type_vars(target_type)))
            raise AglTypeError(
                f"agent/exec target type cannot contain a type variable ('{tv}').",
                span=span,
            )

    def _check_schema_compilable(
        self, target_type: Type, codec_name: str, span: SourceSpan, *, use: str
    ) -> None:
        """Reject *target_type* if lowering will schema-compile it but cannot.

        Shared by ``ask``/``ask-request`` finalization and
        ``exec``. The lowerer derives schema/decode metadata only for the
        built-in JSON codec, so custom codecs are responsible for their own
        output format and parsing behavior. Text (and unit/structured-exec)
        outputs do not build a schema.
        """
        if codec_name != "json":
            return
        message = self._ctx._env.type_table.no_finite_schema_message(target_type, use=use)
        if message is not None:
            raise AglTypeError(message, span=span)
        if not self._ctx._type_is_wire_serializable(target_type):
            raise AglTypeError(
                f"{use.capitalize()} '{target_type!r}' is not JSON-serializable; "
                "use a JSON-serializable data type.",
                span=span,
            )

    # --- shared parse-option handling (ask / exec) ---

    def _parse_options(
        self, named: dict[str, NamedArg]
    ) -> tuple[str | None, bool | None, str]:
        """Validate static option syntax without selecting a target-dependent codec."""
        format_name: str | None = None
        if "format" in named:
            format_na = named["format"]
            if not isinstance(format_na.value, StringLit):
                raise AglTypeError(
                    "'format' must be a static text literal (codec name).", span=format_na.span
                )
            format_name = format_na.value.value
        strict_json: bool | None = None
        if "strict_json" in named:
            strict_na = named["strict_json"]
            if not isinstance(strict_na.value, BoolLit):
                raise AglTypeError(
                    "'strict_json' must be a static bool literal.", span=strict_na.span
                )
            strict_json = strict_na.value.value
        parse_policy = "default"
        if "on_parse_error" in named:
            parse_na = named["on_parse_error"]
            parse_policy = self._extract_parse_policy_str(parse_na.value, parse_na.span)
        return format_name, strict_json, parse_policy

    def _resolve_codec(self, obligation: PendingBuiltinObligation) -> tuple[str, bool | None]:
        """Select and validate the codec after the target type is concrete."""
        if obligation.format_name is None:
            codec_name = self._select_codec(obligation.target_type, obligation.span)
        else:
            codec_name = self._validate_format_option(
                obligation.format_name, obligation.target_type, obligation.span
            )
        if obligation.strict_json is not None and codec_name != "json":
            raise AglTypeError(
                f"'strict_json' is only valid when the codec is 'json'; the selected codec "
                f"for this call is '{codec_name}'.",
                span=obligation.span,
            )
        return codec_name, obligation.strict_json if codec_name == "json" else None

    def _finalize_exec(self, obligation: PendingBuiltinObligation) -> None:
        exec_result_type = self._ctx._env.get_type("ExecResult")
        is_structured = exec_result_type is not None and obligation.target_type == exec_result_type
        if is_structured:
            if obligation.has_parse_shaping_option:
                raise AglTypeError(
                    "exec returning ExecResult does not accept parse options; those options "
                    "apply only when parsing stdout into a typed value.",
                    span=obligation.span,
                )
            spec = OutputContractSpec(
                obligation.target_type, "text", None, structured_exec=True
            )
            parse_policy = "default"
        else:
            codec_name, effective_strict = self._resolve_codec(obligation)
            self._check_schema_compilable(
                obligation.target_type, codec_name, obligation.span, use="an exec output type"
            )
            spec = OutputContractSpec(obligation.target_type, codec_name, effective_strict)
            parse_policy = obligation.parse_policy
            self._warn_noop_parse_error_on_text(obligation)
        assert not contains_inference_var(spec.target_type)
        self._ctx._record_contract_spec(obligation.node_id, spec)
        self._append_call_site(obligation, spec.codec_name, parse_policy)

    # --- on_parse_error policy extraction ---

    def _extract_parse_policy_str(self, arg: Expr, span: SourceSpan) -> str:
        """Extract a static ``ParsePolicy`` constructor as an inventory string."""
        if isinstance(arg, Call) and isinstance(arg.callee, VarRef):
            if arg.callee.module_qualifier is not None:
                if arg.callee.module_qualifier.segments not in ((), ("ParsePolicy",)):
                    raise AglTypeError(
                        "'on_parse_error' must be a static ParsePolicy constructor "
                        "(Abort or Retry(n: <int>)).",
                        span=span,
                    )
            return self._extract_parse_policy_variant(arg.callee.name, arg.named_args, span)
        # Bare VarRef: ``Abort`` or ``ParsePolicy::Abort`` (no parens) is also accepted.
        if isinstance(arg, VarRef) and arg.name == "Abort":
            if (
                arg.module_qualifier is None
                or arg.module_qualifier.segments in ((), ("ParsePolicy",))
            ):
                return "abort"
        raise AglTypeError(
            "'on_parse_error' must be a static ParsePolicy constructor "
            "(Abort or Retry(n: <int>)).",
            span=span,
        )

    def _extract_parse_policy_variant(
        self, name: str, named_args: tuple[NamedArg, ...], span: SourceSpan
    ) -> str:
        """Extract Abort or Retry variant from ParsePolicy call."""
        if name == "Abort":
            if named_args:
                raise AglTypeError(
                    "'on_parse_error' must be a static ParsePolicy constructor "
                    "(Abort or Retry(n: <int>)).",
                    span=span,
                )
            return "abort"
        if name == "Retry":
            n_arg = next((a for a in named_args if a.name == "n"), None)
            if n_arg is None or not isinstance(n_arg.value, IntLit):
                raise AglTypeError(
                    "'on_parse_error' must be a static ParsePolicy constructor "
                    "(Abort or Retry(n: <int>)).",
                    span=span,
                )
            return f"retry[{n_arg.value.value}]"
        raise AglTypeError(
            "'on_parse_error' must be a static ParsePolicy constructor "
            "(Abort or Retry(n: <int>)).",
            span=span,
        )

    # --- codec helpers ---

    def _select_codec(self, target_type: Type, span: SourceSpan) -> str:
        kind = target_type.kind
        for codec_name, supported_kinds in self._ctx._caps.codec_kinds.items():
            if kind in supported_kinds:
                return codec_name
        raise AglTypeError(
            f"No registered codec supports type '{target_type!r}'. "
            f"(Type kind '{kind}' is not handled by any available codec.)",
            span=span,
        )

    def _validate_format_option(
        self, format_name: str, target_type: Type, span: SourceSpan
    ) -> str:
        if format_name not in self._ctx._caps.codec_kinds:
            known = sorted(self._ctx._caps.codec_kinds)
            raise AglTypeError(
                f"Unknown codec '{format_name}' in 'format' option. "
                f"Known codecs: {known}.",
                span=span,
            )
        supported_kinds = self._ctx._caps.codec_kinds[format_name]
        if target_type.kind not in supported_kinds:
            raise AglTypeError(
                f"Codec '{format_name}' does not support target type '{target_type!r}'. "
                f"(Supported kinds: {sorted(supported_kinds)}.)",
                span=span,
            )
        return format_name
