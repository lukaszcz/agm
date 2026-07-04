"""Built-in call (print/render/parse_json/ask/ask-request/exec) type-checking collaborator.

Driven by ``_Checker`` via the narrow ``BuiltinCheckCtx`` Protocol.  All logic
lives here; the host checker instantiates ``BuiltinCallChecker(self)`` and
delegates the six built-in dispatch branches to the public entry points.
"""

from __future__ import annotations

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
    contains_type_var,
    free_type_vars,
)
from agm.agl.syntax.nodes import (
    BoolLit,
    Call,
    Expr,
    FieldAccess,
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
# Narrow context Protocol
# ---------------------------------------------------------------------------


class BuiltinCheckCtx(Protocol):
    """The minimal _Checker surface the built-in call checker needs."""

    _env: TypeEnvironment
    _caps: HostCapabilities
    _contract_specs: dict[int, OutputContractSpec]
    _call_sites: list[CallSiteRecord]
    _warnings: list[Diagnostic]
    _current_type_vars: frozenset[str]

    def _check_expr(self, expr: Expr, *, expected: Type | None) -> Type: ...

    def _assert_assignable(
        self, value_type: Type, target_type: Type, span: SourceSpan
    ) -> None: ...


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

        # reject function/agent targets.
        if isinstance(target_type, (FunctionType, AgentType)):
            raise AglTypeError(
                "cannot parse agent output into a function/agent value.",
                span=node.span,
            )

        self._finish_ask_like(node, target_type, callee="ask", require_default_agent=True)
        return target_type

    # --- ask-request ---

    def check_ask_request(self, node: Call) -> Type:
        """Type-check ``ask-request(prompt, ...)`` — the side-effect-free twin of ``ask``.

        Like ``ask`` it builds an output contract from a target type and the
        parse-shaping named args (``format`` / ``strict_json`` /
        ``on_parse_error``), and accepts an ``agent:`` named arg.  But it never
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

        # reject function/agent targets.
        if isinstance(target_type, (FunctionType, AgentType)):
            raise AglTypeError(
                "cannot build an output contract for a function/agent target.",
                span=node.span,
            )

        # Build the same output contract spec an ``ask`` call would, so the
        # materialized contract (and thus the returned request) matches exactly.
        # ``ask-request`` never dispatches, so a missing ``agent:`` is allowed.
        self._finish_ask_like(
            node, target_type, callee="ask-request", require_default_agent=False
        )
        return agent_request_type

    def _finish_ask_like(
        self, node: Call, target_type: Type, *, callee: str, require_default_agent: bool
    ) -> None:
        """Shared tail for ``ask`` / ``ask-request``: validate args, record the contract.

        Both builtins accept the same named args, a single prompt positional, and
        an optional ``agent:`` value, and build an identical output-contract spec +
        call-site record.  They differ only in ``callee`` (woven into diagnostics)
        and whether a missing ``agent:`` requires a configured default agent
        (``ask`` dispatches; ``ask-request`` does not).
        """
        named = {na.name: na for na in node.named_args}

        # Reject unknown named args.
        for arg_name, na in named.items():
            if arg_name not in self._ASK_ALLOWED_NAMED_ARGS:
                raise AglTypeError(
                    f"{callee}: unknown argument '{arg_name}'.",
                    span=na.span,
                )

        # Prompt (first positional arg — reject extra positionals).
        if not node.args:
            raise AglTypeError(f"{callee}() requires a prompt argument.", span=node.span)
        if len(node.args) > 1:
            raise AglTypeError(
                f"{callee}: too many positional arguments (expected 1).",
                span=node.span,
            )
        prompt_type = self._ctx._check_expr(node.args[0], expected=TextType())
        self._ctx._assert_assignable(prompt_type, TextType(), node.args[0].span)

        # agent: named arg.
        if "agent" in named:
            agent_na = named["agent"]
            agent_type = self._ctx._check_expr(agent_na.value, expected=None)
            if not isinstance(agent_type, AgentType):
                raise AglTypeError(
                    f"'agent:' argument must be of type agent; got '{agent_type!r}'.",
                    span=agent_na.span,
                )
        elif require_default_agent and not self._ctx._caps.has_default_agent:
            raise AglTypeError(
                "No default agent is configured; the built-in 'ask' call "
                "cannot run. Register a default agent, or run via `agm exec`, "
                "which provides one.",
                span=node.span,
            )

        if isinstance(target_type, UnitType):
            self._reject_unit_parse_options(named, callee=callee)
            codec_name = "none"
            parse_policy_str = "default"
        else:
            codec_name, effective_strict, parse_policy_str = self._resolve_parse_options(
                node, target_type, named
            )
            spec = OutputContractSpec(
                target_type=target_type,
                codec_name=codec_name,
                strict_json=effective_strict,
            )
            self._ctx._contract_specs[node.node_id] = spec
        self._ctx._call_sites.append(
            CallSiteRecord(
                node_id=node.node_id,
                callee=callee,
                target_type=target_type,
                codec_name=codec_name,
                parse_policy=parse_policy_str,
                line=node.span.start_line,
                col=node.span.start_col,
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

        # reject function/agent targets.
        if isinstance(target_type, (FunctionType, AgentType)):
            raise AglTypeError(
                "cannot parse exec output into a function/agent value.",
                span=node.span,
            )

        named = {na.name: na for na in node.named_args}

        # Reject unknown named args (exec has no 'agent:' argument).
        for arg_name, na in named.items():
            if arg_name not in self._EXEC_ALLOWED_NAMED_ARGS:
                raise AglTypeError(
                    f"exec: unknown argument '{arg_name}'.",
                    span=na.span,
                )

        # Command (first positional arg — reject extra positionals).
        if not node.args:
            raise AglTypeError("exec() requires a command argument.", span=node.span)
        if len(node.args) > 1:
            raise AglTypeError(
                "exec: too many positional arguments (expected 1).",
                span=node.span,
            )
        cmd_type = self._ctx._check_expr(node.args[0], expected=TextType())
        self._ctx._assert_assignable(cmd_type, TextType(), node.args[0].span)

        # Determine codec.
        is_exec_result = exec_result_type is not None and target_type == exec_result_type
        parse_policy_str = "default"

        if is_exec_result:
            # Structured form: reject parse-shaping options — they are meaningless
            # when exec returns the raw ExecResult record.
            for shaping_arg in ("format", "strict_json", "on_parse_error"):
                if shaping_arg in named:
                    raise AglTypeError(
                        f"exec returning ExecResult does not accept '{shaping_arg}'; "
                        "those options apply only when parsing stdout into a typed value.",
                        span=named[shaping_arg].span,
                    )
            spec = OutputContractSpec(
                target_type=target_type,
                codec_name="text",
                strict_json=None,
                structured_exec=True,
            )
        else:
            codec_name, effective_strict, parse_policy_str = self._resolve_parse_options(
                node, target_type, named
            )
            spec = OutputContractSpec(
                target_type=target_type,
                codec_name=codec_name,
                strict_json=effective_strict,
            )
        self._ctx._contract_specs[node.node_id] = spec
        self._ctx._call_sites.append(
            CallSiteRecord(
                node_id=node.node_id,
                callee="exec",
                target_type=target_type,
                codec_name=spec.codec_name,
                parse_policy=parse_policy_str,
                line=node.span.start_line,
                col=node.span.start_col,
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

    # --- shared parse-option handling (ask / exec) ---

    def _reject_unit_parse_options(
        self, named: dict[str, NamedArg], *, callee: str
    ) -> None:
        for option in ("format", "strict_json", "on_parse_error"):
            if option in named:
                raise AglTypeError(
                    f"{callee} returning unit does not accept '{option}'; "
                    "unit responses are ignored and have no output contract.",
                    span=named[option].span,
                )

    def _resolve_parse_options(
        self, node: Call, target_type: Type, named: dict[str, NamedArg]
    ) -> tuple[str, bool | None, str]:
        """Resolve the format/strict_json/on_parse_error named args shared by ask and exec.

        Returns ``(codec_name, effective_strict, parse_policy_str)``.
        """
        if "format" in named:
            format_na = named["format"]
            fmt_expr = format_na.value
            if not isinstance(fmt_expr, StringLit):
                raise AglTypeError(
                    "'format' must be a static text literal (codec name).",
                    span=format_na.span,
                )
            codec_name = self._validate_format_option(fmt_expr.value, target_type, format_na.span)
        else:
            codec_name = self._select_codec(target_type, node.span)

        strict_json: bool | None = None
        if "strict_json" in named:
            sj_na = named["strict_json"]
            sj_expr = sj_na.value
            if not isinstance(sj_expr, BoolLit):
                raise AglTypeError(
                    "'strict_json' must be a static bool literal.",
                    span=sj_na.span,
                )
            if codec_name != "json":
                raise AglTypeError(
                    f"'strict_json' is only valid when the codec is 'json'; "
                    f"the selected codec for this call is '{codec_name}'.",
                    span=sj_na.span,
                )
            strict_json = sj_expr.value

        parse_policy_str = "default"
        if "on_parse_error" in named:
            ope_na = named["on_parse_error"]
            parse_policy_str = self._extract_parse_policy_str(ope_na.value, ope_na.span)
            # Warn: no-op on text target.
            if isinstance(target_type, TextType):
                self._ctx._warnings.append(
                    Diagnostic(
                        message=(
                            "'on_parse_error' has no effect on a text target: a text "
                            "result never fails parsing, so the policy can never fire."
                        ),
                        line=node.span.start_line,
                        column=node.span.start_col,
                        end_line=node.span.end_line,
                        end_column=node.span.end_col,
                        severity="warning",
                    )
                )

        effective_strict = strict_json if codec_name == "json" else None
        return codec_name, effective_strict, parse_policy_str

    # --- on_parse_error policy extraction ---

    def _extract_parse_policy_str(self, arg: Expr, span: SourceSpan) -> str:
        """Extract a static ``ParsePolicy`` constructor as an inventory string."""
        if isinstance(arg, Call) and isinstance(arg.callee, FieldAccess):
            qualifier = arg.callee.obj
            if not (isinstance(qualifier, VarRef) and qualifier.name == "ParsePolicy"):
                raise AglTypeError(
                    "'on_parse_error' must be a static ParsePolicy constructor "
                    "(Abort or Retry(n: <int>)).",
                    span=span,
                )
            return self._extract_parse_policy_variant(arg.callee.field, arg.named_args, span)
        if isinstance(arg, Call) and isinstance(arg.callee, VarRef):
            return self._extract_parse_policy_variant(arg.callee.name, arg.named_args, span)
        # Bare VarRef: ``Abort`` (no parens) is also accepted as abort policy.
        if isinstance(arg, VarRef) and arg.name == "Abort":
            return "abort"
        # Bare FieldAccess: ``ParsePolicy.Abort`` (no parens) is also accepted.
        if isinstance(arg, FieldAccess) and arg.field == "Abort":
            qualifier = arg.obj
            if isinstance(qualifier, VarRef) and qualifier.name == "ParsePolicy":
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
