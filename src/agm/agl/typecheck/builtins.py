"""Built-in call (print/render/copy/shallow_copy/parse_json/ask/ask-request/exec)
type-checking collaborator.

Driven by ``_Checker`` via the narrow ``BuiltinCheckCtx`` Protocol.  All logic
lives here; the host checker instantiates ``BuiltinCallChecker(self)`` and
delegates the built-in dispatch branches to the public entry points.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from agm.agl.capabilities import HostCapabilities
from agm.agl.diagnostics import Diagnostic
from agm.agl.ir.reserved_nominals import reserved_nominal_id
from agm.agl.modules.ids import spell_declaration
from agm.agl.scope.symbols import ConstructorRef
from agm.agl.semantics.analyses import nominal_references
from agm.agl.semantics.type_table import OPTION_TYPE_DEF, DeclId
from agm.agl.semantics.types import (
    BUILTIN_PRELUDE_TYPES,
    BoolType,
    EnumType,
    ExceptionType,
    FunctionType,
    JsonType,
    RecordType,
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
    ParamKind,
    QualifierAnchor,
    StringLit,
    VarRef,
)
from agm.agl.syntax.resources import ResourceError, resource_path
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck.arguments import bind_call_args
from agm.agl.typecheck.env import (
    AglTypeError,
    CallSiteRecord,
    OutputContractSpec,
    ParamSpec,
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
    # (name, span) of every parse-shaping named arg present, in canonical order.
    # Retained so region-close diagnostics point at the offending argument rather
    # than the whole call span.
    parse_option_spans: tuple[tuple[str, SourceSpan], ...]

    @property
    def has_parse_shaping_option(self) -> bool:
        """Whether any of ``format`` / ``strict_json`` / ``on_parse_error`` is set."""
        return bool(self.parse_option_spans)

    @property
    def has_parse_error_option(self) -> bool:
        """Whether the ``on_parse_error`` option is set."""
        return any(name == "on_parse_error" for name, _ in self.parse_option_spans)

    def strict_json_span(self) -> SourceSpan:
        """Return the span of the ``strict_json`` option.

        Precondition: :attr:`strict_json` is not ``None`` (the option was
        supplied), which guarantees a matching entry is present.
        """
        return next(span for name, span in self.parse_option_spans if name == "strict_json")

    def first_parse_option(self) -> tuple[str, SourceSpan]:
        """Return the first parse-shaping option's ``(name, span)``.

        Precondition: :attr:`has_parse_shaping_option` is true.
        """
        return self.parse_option_spans[0]


# ---------------------------------------------------------------------------
# Narrow context Protocol
# ---------------------------------------------------------------------------


class BuiltinCheckCtx(Protocol):
    """The minimal _Checker surface the built-in call checker needs."""

    _env: TypeEnvironment
    _caps: HostCapabilities

    def _record_contract_spec(self, node_id: int, spec: OutputContractSpec) -> None: ...

    def _record_explicit_builtin_target(self, node_id: int, target_type: Type) -> None: ...

    def _append_call_site(self, call_site: CallSiteRecord) -> None: ...

    def _append_warning(self, warning: Diagnostic) -> None: ...

    _current_type_vars: frozenset[str]

    def _check_expr(self, expr: Expr, *, expected: Type | None) -> Type: ...

    def _assert_assignable_from(
        self, value_type: Type, target_type: Type, span: SourceSpan, expr: Expr
    ) -> None: ...

    def _type_is_wire_serializable(self, typ: Type) -> bool: ...

    def _register_builtin_obligation(self, obligation: PendingBuiltinObligation) -> None: ...

    def _constructor_ref_for(self, node_id: int) -> ConstructorRef | None: ...


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
        # Contracts already found coherent (:meth:`_check_host_contract_coherent`).
        # The verdict depends only on the declaration and the type table, not
        # on the call site, and every obligation is checked after registration
        # settles -- so a contract reached from many call sites is walked once
        # rather than once per site. Only successes are recorded; the first
        # incoherent contract raises.
        self._coherent_contracts: set[DeclId] = set()

    # --- print ---

    def check_print(self, node: Call) -> Type:
        if len(node.args) != 1 or node.named_args:
            raise AglTypeError(
                "print() requires exactly one positional argument.",
                span=node.span,
            )
        self._check_arg_with_optional_explicit_target(node, node.args[0], "print")
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
            self._ctx._assert_assignable_from(
                option_type, BoolType(), named.value.span, named.value
            )
        self._check_arg_with_optional_explicit_target(node, node.args[0], "render")
        return TextType()

    # --- copy / shallow_copy ---

    def check_copy(self, node: Call) -> Type:
        if len(node.args) != 1 or node.named_args:
            raise AglTypeError("copy() requires exactly one positional argument.", span=node.span)
        return self._check_arg_with_optional_explicit_target(node, node.args[0], "copy")

    def check_shallow_copy(self, node: Call) -> Type:
        if len(node.args) != 1 or node.named_args:
            raise AglTypeError(
                "shallow_copy() requires exactly one positional argument.", span=node.span
            )
        return self._check_arg_with_optional_explicit_target(node, node.args[0], "shallow_copy")

    def _check_arg_with_optional_explicit_target(
        self, node: Call, arg_expr: Expr, name: str
    ) -> Type:
        """Check *arg_expr*, honoring an optional explicit ``::[T]`` type argument.

        With no explicit type argument, checks *arg_expr* with no contextual
        expectation and returns its own inferred type. With one
        (``name::[T](...)``), *arg_expr* must be assignable to ``T``, and the
        return is ``T``. Shared by every built-in whose sole type parameter
        needs no other contextual ``expected`` type: ``print``, ``render``,
        ``copy``, and ``shallow_copy``.
        """
        explicit = self._resolve_explicit_target(node, name)
        if explicit is None:
            return self._ctx._check_expr(arg_expr, expected=None)
        arg_type = self._ctx._check_expr(arg_expr, expected=explicit)
        self._ctx._assert_assignable_from(arg_type, explicit, arg_expr.span, arg_expr)
        return explicit

    # --- Session statics ---

    def check_session_open(self, node: Call) -> Type:
        """Type-check ``Session::open(agent, transport?, name?)``."""
        session_transport = self._builtin_contract_type("SessionTransport")
        assert isinstance(session_transport, EnumType)
        transport = OPTION_TYPE_DEF.handle((session_transport,))
        return self._check_static_call(
            node,
            "Session::open",
            (
                ParamSpec(
                    name="agent",
                    type=self._builtin_contract_type("Agent"),
                    kind=ParamKind.STANDARD,
                    has_default=False,
                ),
                ParamSpec(
                    name="transport",
                    type=transport,
                    kind=ParamKind.STANDARD,
                    has_default=True,
                ),
                ParamSpec(
                    name="name",
                    type=TextType(),
                    kind=ParamKind.STANDARD,
                    has_default=True,
                ),
            ),
            self._builtin_contract_type("Session"),
        )

    def check_session_default(self, node: Call) -> Type:
        """Type-check the nullary ``Session::default()`` static."""
        return self._check_static_call(
            node,
            "Session::default",
            (),
            self._builtin_contract_type("Session"),
        )

    def _check_static_call(
        self,
        node: Call,
        name: str,
        params: tuple[ParamSpec, ...],
        result: Type,
    ) -> Type:
        """Bind and check one fixed-signature type-scoped builtin call."""
        if node.type_args:
            raise AglTypeError(
                f"{name} does not accept type arguments.",
                span=node.span,
            )
        binding = bind_call_args(
            params,
            node.args,
            node.named_args,
            call_span=node.span,
            context_desc=f"call to '{name}'",
        )
        for param, argument in zip(params, binding, strict=True):
            if argument is None:
                continue
            argument_type = self._ctx._check_expr(argument, expected=param.type)
            self._ctx._assert_assignable_from(argument_type, param.type, argument.span, argument)
        return result

    # --- resources ---

    def check_resource(self, node: Call) -> Type:
        try:
            resource_path(node, is_directory=False)
        except ResourceError as exc:
            raise AglTypeError(str(exc), span=node.span) from exc
        return TextType()

    def check_resource_dir(self, node: Call) -> Type:
        try:
            resource_path(node, is_directory=True)
        except ResourceError as exc:
            raise AglTypeError(str(exc), span=node.span) from exc
        return TextType()

    # --- parse_json ---

    def check_parse_json(self, node: Call) -> Type:
        if len(node.args) != 1 or node.named_args:
            raise AglTypeError(
                "parse_json() requires exactly one positional text argument.",
                span=node.span,
            )
        arg_type = self._ctx._check_expr(node.args[0], expected=TextType())
        self._ctx._assert_assignable_from(arg_type, TextType(), node.args[0].span, node.args[0])
        return JsonType()

    # --- ask ---

    def check_ask(
        self, node: Call, *, expected: Type | None, receiver_type: Type | None = None
    ) -> Type:
        """Type-check ``ask``. *receiver_type* is set only for ``x.ask(...)``."""
        # Target type: explicit type argument overrides context.
        explicit = self._resolve_explicit_target(node, "ask")
        target_type: Type = (
            explicit if explicit is not None else (expected if expected is not None else TextType())
        )
        self._reject_type_var_target(target_type, node.span)
        self._register_ask_like_obligation(
            node,
            target_type=target_type,
            result_type=target_type,
            kind=BuiltinObligationKind.ASK,
            receiver_type=receiver_type,
        )
        return target_type

    # --- ask-request ---

    def check_ask_request(self, node: Call, *, receiver_type: Type | None = None) -> Type:
        """Type-check the fixed-text, side-effect-free ``ask-request`` builder."""
        agent_request_type = self._resolve_host_record_contract("AgentRequest", span=node.span)

        if node.type_args:
            raise AglTypeError(
                "ask-request does not accept type arguments; it always builds a text request.",
                span=node.span,
            )
        # This reuses prompt and Agent argument type validation, without exposing
        # ask's parse-shaping options. The obligation still records the fixed text
        # target so the call site is reported like any other agent call site;
        # lowering builds the request record itself and allocates no contract.
        # The contract resolved above is threaded through so the coherence walk
        # does not run a second time for this call site.
        self._validate_ask_like_arguments(
            node,
            "ask-request",
            allowed_named=frozenset() if receiver_type is not None else frozenset({"agent"}),
            receiver_type=receiver_type,
            agent_request_type=agent_request_type,
        )
        self._ctx._register_builtin_obligation(
            PendingBuiltinObligation(
                node_id=node.node_id,
                target_type=TextType(),
                result_type=agent_request_type,
                span=node.span,
                kind=BuiltinObligationKind.ASK_REQUEST,
                format_name=None,
                strict_json=None,
                parse_policy="default",
                parse_option_spans=(),
            )
        )
        return agent_request_type

    def _register_ask_like_obligation(
        self,
        node: Call,
        *,
        target_type: Type,
        result_type: Type,
        kind: BuiltinObligationKind,
        receiver_type: Type | None,
    ) -> None:
        """Check target-independent syntax, then queue contract materialization."""
        callee = kind.value
        named = self._validate_ask_like_arguments(
            node,
            callee,
            allowed_named=self._ASK_ALLOWED_NAMED_ARGS
            - ({"agent"} if receiver_type is not None else set()),
            receiver_type=receiver_type,
        )
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
                parse_option_spans=self._collect_parse_option_spans(named),
            )
        )

    def _validate_ask_like_arguments(
        self,
        node: Call,
        callee: str,
        *,
        allowed_named: frozenset[str],
        receiver_type: Type | None = None,
        agent_request_type: RecordType | None = None,
    ) -> dict[str, NamedArg]:
        """Check syntax and value arguments that do not need the target type.

        *allowed_named* is the caller's permitted named-argument set: ``ask``
        offers its parse-shaping options, ``ask-request`` only ``agent``, and a
        receiver call drops ``agent`` because the receiver already supplies it.

        Every ``ask``/``ask-request`` call resolves the ``AgentRequest``
        contract unconditionally, whether or not ``agent`` is itself supplied:
        the host builds an ``AgentRequest`` (directly for ``ask-request``, via
        its retry machinery for ``ask``) either way, filling a missing
        ``agent`` from the canonical default agent, so the contract must be
        host-coherent (:meth:`_resolve_host_record_contract`) regardless.
        *agent_request_type* lets a caller that already resolved the contract
        itself (``check_ask_request``, which needs it before this method runs
        anyway) pass the resolved handle through instead of paying the
        coherence walk a second time; when omitted (``ask``'s path), this
        method resolves it itself, at this same point in the check order.

        The agent the request is built with is that resolved contract's own
        ``agent`` FIELD type -- the type the value is actually stored as --
        rather than an independently resolved ``Agent`` type, so the two can
        never name different declarations of the same bare name. That single
        expected type governs both ways of supplying the agent: the ``agent``
        named argument, and *receiver_type* for a receiver call
        (``x.ask(...)``), whose receiver IS the agent.
        """
        named = {na.name: na for na in node.named_args}
        for arg_name, na in named.items():
            if arg_name not in allowed_named:
                raise AglTypeError(f"{callee}: unknown argument '{arg_name}'.", span=na.span)
        if not node.args:
            raise AglTypeError(f"{callee}() requires a prompt argument.", span=node.span)
        if len(node.args) > 1:
            raise AglTypeError(
                f"{callee}: too many positional arguments (expected 1).", span=node.span
            )
        prompt_type = self._ctx._check_expr(node.args[0], expected=TextType())
        self._ctx._assert_assignable_from(prompt_type, TextType(), node.args[0].span, node.args[0])
        if agent_request_type is None:
            agent_request_type = self._resolve_host_record_contract("AgentRequest", span=node.span)
        expected_agent_type = self._ctx._env.type_table.record_fields(agent_request_type)["agent"]
        if receiver_type is not None:
            self._ctx._assert_assignable_from(receiver_type, expected_agent_type, node.span, node)
        if "agent" in named:
            agent_na = named["agent"]
            agent_type = self._ctx._check_expr(agent_na.value, expected=expected_agent_type)
            self._ctx._assert_assignable_from(
                agent_type,
                expected_agent_type,
                agent_na.value.span,
                agent_na.value,
            )
        return named

    def finalize(self, obligation: PendingBuiltinObligation) -> None:
        """Materialize one fully zonked built-in obligation at region close."""
        target_type = obligation.target_type
        if contains_inference_var(target_type) or contains_inference_var(obligation.result_type):
            raise AglTypeError(
                "Cannot infer a concrete target type for this built-in call.", span=obligation.span
            )
        self._reject_type_var_target(target_type, obligation.span)
        if isinstance(target_type, FunctionType):
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
        if isinstance(target_type, UnitType):
            if obligation.has_parse_shaping_option:
                option_name, offending_span = obligation.first_parse_option()
                raise AglTypeError(
                    f"{callee} returning unit does not accept '{option_name}'; unit responses "
                    "are ignored and have no output contract.",
                    span=offending_span,
                )
            codec_name = "none"
            parse_policy = "default"
        else:
            spec = self._record_parsed_contract(obligation, use="an agent output type")
            codec_name = spec.codec_name
            parse_policy = obligation.parse_policy
        self._append_call_site(obligation, codec_name, parse_policy)

    def _warn_noop_parse_error_on_text(self, obligation: PendingBuiltinObligation) -> None:
        """Warn when ``on_parse_error`` is set on a text target, where it can never fire."""
        if not (obligation.has_parse_error_option and isinstance(obligation.target_type, TextType)):
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
            raise AglTypeError("The host does not support 'exec' (shell) calls.", span=node.span)

        target_type: Type
        # Explicit type argument overrides context.
        explicit = self._resolve_explicit_target(node, "exec")
        if explicit is not None:
            target_type = explicit
        elif expected is not None:
            target_type = expected
        else:
            target_type = self._builtin_contract_type("ExecResult")
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
        self._ctx._assert_assignable_from(cmd_type, TextType(), node.args[0].span, node.args[0])
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
                parse_option_spans=self._collect_parse_option_spans(named),
            )
        )
        return target_type

    def _builtin_contract_type(self, name: str) -> RecordType | EnumType | ExceptionType:
        """Return the type this program's own ``builtin`` declaration of *name* names.

        The single source of truth every checker-side resolution of a
        host-contract built-in nominal (``ExecResult``, ``AgentRequest``,
        ``Agent``, ``ParsePolicy``) goes through: a program's own ``builtin
        record``/``builtin enum`` declaration of *name* — at whatever scope
        path it is written, so a scoped declaration is recognized at its own
        path rather than the root — when the shared ``TypeTable`` has one
        registered; the seeded canonical (root) handle otherwise. A program
        loaded without the standard library declares no such ``builtin`` of
        its own for a name it does not itself define, so that case falls
        back to the seeded canonical handle, which is always registered.
        This is also exactly the declaration the host's own minting table
        (``lower.lowerer.builtin_nominals_from_declarations``) is derived
        from, so a call's static type and the identity the host actually
        mints can never disagree.
        """
        declared = self._ctx._env.type_table.builtin_declaration(name)
        if declared is not None:
            return declared.handle()
        canonical = BUILTIN_PRELUDE_TYPES.get(name)
        assert isinstance(canonical, (RecordType, EnumType, ExceptionType)), (
            f"{name!r} has no canonical builtin prelude type"
        )
        return canonical

    def _resolve_host_record_contract(self, name: str, *, span: SourceSpan) -> RecordType:
        """Resolve *name*'s host record contract, rejecting it if incoherent.

        Wraps :meth:`_builtin_contract_type` for the two RECORD contracts a
        host call mints directly (``AgentRequest``, ``ExecResult``) with the
        one coherence check every such resolution needs
        (:meth:`_check_host_contract_coherent`). *name* always names a
        ``builtin record`` in the canonical prelude, so the resolved handle is
        always a ``RecordType``.
        """
        contract_type = self._builtin_contract_type(name)
        assert isinstance(contract_type, RecordType), f"{name!r} is not a builtin record contract"
        self._check_host_contract_coherent(contract_type, span=span)
        return contract_type

    def check_caught_exception_contract(self, exc_type: ExceptionType, *, span: SourceSpan) -> None:
        """Reject a ``catch`` clause naming an incoherent or shadowed host exception.

        A ``builtin exception`` is raised by the host, which fills its fields
        itself — ``AgentCallError``/``AgentParseError`` carry an ``agent``
        field the host always fills with the standard ``Agent``. Catching one
        is the only way a program reads such a field, so this is where a
        host-raised exception's contract is held to the same requirement a
        host-minted record's is (:meth:`_check_host_contract_coherent`). An
        ordinary, non-``builtin`` exception is never host-raised: its fields
        always hold whatever the source that constructed it put there, so it
        carries no requirement of its own and is skipped.

        Independently of field coherence, a ``catch`` clause must also name
        the declaration the host actually mints under this bare name — the
        live registered ``builtin`` declaration
        (:meth:`~agm.agl.semantics.type_table.TypeTable.builtin_declaration`),
        the same resolution :meth:`_builtin_contract_type` and the host's own
        minting table (``lower.lowerer.builtin_nominals_from_declarations``)
        both use. A program's own ``builtin exception`` of a reserved name
        shadows the standard declaration for that name everywhere, not only
        inside the declaration's own scope region; a ``catch`` clause written
        outside that region still resolves the bare name to the standard
        declaration by ordinary scope rules, which the host can then never
        actually raise there — a handler that is statically well-typed but
        provably dead. When nothing shadows this bare name at all (no
        redeclaration anywhere, or a program loaded without the standard
        library), :meth:`~agm.agl.semantics.type_table.TypeTable.builtin_declaration`
        returns ``None`` and this is skipped.
        """
        table = self._ctx._env.type_table
        typedef = table.get_by_id(exc_type.decl_id)
        if typedef is not None and typedef.is_builtin:
            self._check_host_contract_coherent(exc_type, span=span)
        live_declaration = table.builtin_declaration(exc_type.name)
        if live_declaration is None:
            return
        live_handle = live_declaration.handle()
        assert isinstance(live_handle, ExceptionType), (
            f"{exc_type.name!r} is registered as a builtin exception name but its live "
            "declaration is not an exception"
        )
        if live_handle == exc_type:
            return
        caught_spelling = spell_declaration(
            exc_type.module_id, (*exc_type.scope_path, exc_type.name)
        )
        live_spelling = spell_declaration(
            live_handle.module_id, (*live_handle.scope_path, live_handle.name)
        )
        raise AglTypeError(
            f"'catch {exc_type.name}' names '{caught_spelling}', but the host raises "
            f"'{exc_type.name}' here as '{live_spelling}' — this program's own "
            f"'builtin exception {exc_type.name}' declaration shadows the standard one, "
            "so this clause can never match.",
            span=span,
        )

    def _check_host_contract_coherent(
        self, contract_type: RecordType | ExceptionType, *, span: SourceSpan
    ) -> None:
        """Reject *contract_type* if the host cannot produce a value of its shape.

        The host produces a value of *contract_type* itself — a record a
        built-in call mints (``ask-request``, ``ask``, ``exec``) or an
        exception it raises — filling every nominal-typed field with a value
        that carries a fixed, host-known identity (the canonical ``Agent`` for
        ``AgentRequest.agent`` and ``AgentCallError.agent``, the canonical
        ``Option`` for ``AgentRequest``'s ``Option``-typed fields — see
        ``eval/effects.py``, ``runtime/agents.py`` and ``runtime/option.py``)
        — never whatever declaration *contract_type*'s own field type happens
        to name. When a field's type does not itself carry that fixed
        identity — checked recursively through type arguments, so
        ``Option[text]``'s own ``Option`` is checked too — the checker would
        type the field one way while the host produces another, so such a
        contract is rejected here (an ordinary, user-facing static error)
        rather than left to crash the evaluator.
        """
        if contract_type.decl_id in self._coherent_contracts:
            return
        table = self._ctx._env.type_table
        field_types = (
            table.record_fields(contract_type)
            if isinstance(contract_type, RecordType)
            else table.exception_fields(contract_type)
        )
        for field_name, field_type in field_types.items():
            for nominal in nominal_references(field_type):
                # A name with no reserved identity at all (``reserved_id is
                # None``) is also incoherent: ``!=`` against ``None`` is
                # always true, so it is rejected here exactly like a name
                # that has one but whose declaration doesn't carry it.
                reserved_id = reserved_nominal_id(nominal.name)
                if nominal.decl_id != reserved_id:
                    raise AglTypeError(
                        f"{contract_type.name}'s field '{field_name}' is typed "
                        f"'{field_type!r}', which names this program's own "
                        f"'{nominal.name}' declaration rather than the standard one. "
                        f"{contract_type.name} is produced directly by the host, which "
                        f"always fills '{field_name}' with the standard "
                        f"'{nominal.name}' identity, so this contract cannot be used "
                        "here.",
                        span=span,
                    )
        self._coherent_contracts.add(contract_type.decl_id)

    # --- shared explicit-target resolver for --

    def _resolve_explicit_target(self, node: Call, builtin_name: str) -> Type | None:
        """Resolve a built-in call with an explicit ``::[T]`` argument.

        Returns the resolved ``Type`` when ``node.type_args`` is non-empty, or
        ``None`` when there are no explicit type arguments (caller falls back to
        its contextual/default target logic).

        Raises ``AglTypeError`` when more than one type argument is provided
        (arity error). The ask/ask-request/exec callers apply the
        type-variable guard (see :meth:`_reject_type_var_target`) to the
        *final* target type, covering both the explicit and the
        contextual/inferred target paths; ``print``, ``render``, ``copy``,
        and ``shallow_copy`` do not apply that guard, because none of them
        schema-compile their target — a bare type variable in
        ``copy::[T](v)`` is exactly what makes it work inside a generic
        ``def``.

        Every resolution is also recorded into the checked module's
        ``explicit_builtin_targets`` side table, keyed by ``node.node_id`` —
        the lowerer's authoritative source for a call's explicit target type,
        needed by ``print``/``render`` whose own checked result type discards
        it (see ``CheckedModule.explicit_builtin_targets``). ``ask-request``
        has no output target and therefore does not use this helper.
        """
        if not node.type_args:
            return None
        if len(node.type_args) > 1:
            raise AglTypeError(
                f"{builtin_name} expects at most one explicit type argument; "
                f"got {len(node.type_args)}.",
                span=node.span,
            )
        resolved = self._ctx._env.resolve_type_expr(
            node.type_args[0], span=node.span, type_vars=self._ctx._current_type_vars
        )
        self._ctx._record_explicit_builtin_target(node.node_id, resolved)
        return resolved

    def _reject_type_var_target(self, target_type: Type, span: SourceSpan) -> None:
        """An ask/exec target type may not contain a type variable.

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

        Shared by ``ask`` finalization and ``exec``. The lowerer derives
        schema/decode metadata only for the built-in JSON codec, so custom
        codecs are responsible for their own output format and parsing behavior.
        Text (and unit/structured-exec)
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

    def _parse_options(self, named: dict[str, NamedArg]) -> tuple[str | None, bool | None, str]:
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

    @staticmethod
    def _collect_parse_option_spans(
        named: dict[str, NamedArg],
    ) -> tuple[tuple[str, SourceSpan], ...]:
        """Capture the spans of the parse-shaping named args for later diagnostics."""
        return tuple(
            (name, named[name].span)
            for name in ("format", "strict_json", "on_parse_error")
            if name in named
        )

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
                span=obligation.strict_json_span(),
            )
        return codec_name, obligation.strict_json if codec_name == "json" else None

    def _record_parsed_contract(
        self, obligation: PendingBuiltinObligation, *, use: str
    ) -> OutputContractSpec:
        """Resolve the codec, validate the schema, and record the parsed output contract."""
        codec_name, effective_strict = self._resolve_codec(obligation)
        self._check_schema_compilable(obligation.target_type, codec_name, obligation.span, use=use)
        spec = OutputContractSpec(obligation.target_type, codec_name, effective_strict)
        assert not contains_inference_var(spec.target_type)
        self._ctx._record_contract_spec(obligation.node_id, spec)
        self._warn_noop_parse_error_on_text(obligation)
        return spec

    def _finalize_exec(self, obligation: PendingBuiltinObligation) -> None:
        target_type = obligation.target_type
        if isinstance(target_type, UnitType):
            if obligation.has_parse_shaping_option:
                option_name, offending_span = obligation.first_parse_option()
                raise AglTypeError(
                    f"exec returning unit does not accept '{option_name}'; unit output is "
                    "discarded and has no output contract.",
                    span=offending_span,
                )
            spec = OutputContractSpec(target_type, "none", None, structured_exec=False)
        else:
            # ``ExecResult`` is identified as the type the program's own
            # `builtin record ExecResult` declaration names (see
            # `_builtin_contract_type`), so a scoped declaration is
            # recognized at its own path rather than the root canonical one.
            is_structured = target_type == self._builtin_contract_type("ExecResult")
            if not is_structured:
                spec = self._record_parsed_contract(obligation, use="an exec output type")
                self._append_call_site(obligation, spec.codec_name, obligation.parse_policy)
                return
            # ``exec`` will mint an ``ExecResult`` record directly: apply the
            # same host-coherence check as ``ask``/``ask-request`` (currently
            # vacuous, since none of ``ExecResult``'s own fields are nominal,
            # but applied uniformly so a future field cannot regress silently).
            assert isinstance(target_type, RecordType)
            self._check_host_contract_coherent(target_type, span=obligation.span)
            if obligation.has_parse_shaping_option:
                option_name, offending_span = obligation.first_parse_option()
                raise AglTypeError(
                    f"exec returning ExecResult does not accept '{option_name}'; those options "
                    "apply only when parsing stdout into a typed value.",
                    span=offending_span,
                )
            spec = OutputContractSpec(target_type, "text", None, structured_exec=True)

        self._ctx._record_contract_spec(obligation.node_id, spec)
        self._append_call_site(obligation, spec.codec_name, "default")

    # --- on_parse_error policy extraction ---

    def _extract_parse_policy_str(self, arg: Expr, span: SourceSpan) -> str:
        """Extract a static ``ParsePolicy`` constructor as an inventory string.

        *arg* must actually RESOLVE (through the same constructor-identity
        mechanism ordinary expression-checking uses,
        :meth:`BuiltinCheckCtx._constructor_ref_for`) to a genuine
        constructor — not merely share a constructor's bare spelling. A local
        binding that shadows the name (``let Abort = ParsePolicy::Retry(n =
        3)``) resolves to that binding, not any constructor, so
        ``_constructor_ref_for`` returns ``None`` for it and it is rejected
        here exactly like any other non-constructor expression, matching how
        the surrounding checker treats a shadowed constructor name everywhere
        else (e.g. ``_check_builtin_var``'s constant-expression check).
        """
        if isinstance(arg, Call) and isinstance(arg.callee, VarRef):
            callee = arg.callee
            if not self._accepts_as_parse_policy_constructor(callee):
                raise AglTypeError(
                    "'on_parse_error' must be a static ParsePolicy constructor "
                    "(Abort or Retry(n: <int>)).",
                    span=span,
                )
            return self._extract_parse_policy_variant(callee.name, arg.named_args, span)
        # Bare VarRef: ``Abort`` or ``ParsePolicy::Abort`` (no parens) is also accepted.
        if (
            isinstance(arg, VarRef)
            and arg.name == "Abort"
            and self._accepts_as_parse_policy_constructor(arg)
        ):
            return "abort"
        raise AglTypeError(
            "'on_parse_error' must be a static ParsePolicy constructor (Abort or Retry(n: <int>)).",
            span=span,
        )

    def _accepts_as_parse_policy_constructor(self, ref: VarRef) -> bool:
        """Whether *ref* denotes an accepted ``ParsePolicy`` constructor spelling.

        Resolves *ref* through :meth:`BuiltinCheckCtx._constructor_ref_for` so
        a local binding that shadows the name is never mistaken for the
        constructor it shadows.

        Unqualified (and current-module-anchored, ``::Retry``) spellings are
        accepted whenever they resolve to ANY constructor — not necessarily
        one this program's own ``ParsePolicy`` declares. The built-in
        ``Abort`` EXCEPTION and ``ParsePolicy``'s nullary ``Abort`` variant
        share that one bare root spelling, and ordinary name resolution picks
        one of them (the exception, today); accepting either is unambiguous
        in this position, since only a ``ParsePolicy`` constructor is ever a
        legal ``on_parse_error`` value, and it preserves the unqualified
        spelling's existing leniency while still closing the actual
        shadowing hole (a binding that resolves to no constructor at all).

        A qualified spelling (other than the current-module anchor) must
        instead resolve to the exact constructor of this program's own
        ``ParsePolicy`` (:meth:`_builtin_contract_type`) that its final
        segment names — the bare root ``ParsePolicy::`` prefix when the
        program declares none of its own, or that declaration's own scope
        path (e.g. ``A::ParsePolicy::``) when it does — so a scoped
        ``ParsePolicy`` is recognized at its own path exactly like the root
        one, and an unrelated same-named constructor is rejected. A qualifier
        segment carrying an explicit type argument (``ParsePolicy[int]::``)
        is never accepted, regardless of what it would otherwise resolve to:
        ``ParsePolicy`` is not generic, so a type argument there can only be
        a mistake.
        """
        chain = ref.qualifier
        if chain is None or chain.anchor is QualifierAnchor.CURRENT_MODULE:
            return self._ctx._constructor_ref_for(ref.node_id) is not None
        if any(segment.type_args is not None for segment in chain.segments):
            return False
        parse_policy_type = self._builtin_contract_type("ParsePolicy")
        assert isinstance(parse_policy_type, EnumType), "ParsePolicy is always an enum contract"
        ctor_ref = self._ctx._constructor_ref_for(ref.node_id)
        return ctor_ref is not None and ctor_ref.matches(parse_policy_type, ref.name)

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
            "'on_parse_error' must be a static ParsePolicy constructor (Abort or Retry(n: <int>)).",
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

    def _validate_format_option(self, format_name: str, target_type: Type, span: SourceSpan) -> str:
        if format_name not in self._ctx._caps.codec_kinds:
            known = sorted(self._ctx._caps.codec_kinds)
            raise AglTypeError(
                f"Unknown codec '{format_name}' in 'format' option. Known codecs: {known}.",
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
