"""AgL tree-walking interpreter (Component 6).

``Interpreter`` evaluates a ``CheckedProgram`` using a ``WorkflowRuntime``
and a root ``Scope``.  It implements all M1 statements and expressions.

Control flow for AgL exceptions uses Python exceptions (``AglRaise``).
"""

from __future__ import annotations

import decimal
from typing import TYPE_CHECKING, TypeVar, assert_never

from agm.agl._text import normalize_newlines
from agm.agl.eval.exceptions import AglRaise
from agm.agl.eval.scope import Scope
from agm.agl.eval.values import (
    BoolValue,
    DecimalValue,
    DictValue,
    EnumValue,
    ExceptionValue,
    IntValue,
    JsonValue,
    ListValue,
    RecordValue,
    TextValue,
    Value,
)
from agm.agl.runtime.serialize import value_to_json_obj
from agm.agl.syntax.nodes import (
    AgentCall,
    AgentDecl,
    BinaryOp,
    BinOp,
    BoolLit,
    CaseExpr,
    CaseStmt,
    CatchClause,
    Constructor,
    DecimalLit,
    DictLit,
    DoUntil,
    ElseSentinel,
    EnumDef,
    Expr,
    ExprStmt,
    FieldAccess,
    IfStmt,
    InputDecl,
    InterpSegment,
    IntLit,
    IsTest,
    LetDecl,
    ListLit,
    NullLit,
    PassStmt,
    Pattern,
    PrintStmt,
    Raise,
    RecordDef,
    SetStmt,
    Stmt,
    StringLit,
    Template,
    TextSegment,
    TryCatch,
    TypeAlias,
    UnaryNeg,
    UnaryNot,
    VarDecl,
    VarPattern,
    VarRef,
    WildcardPattern,
)
from agm.agl.typecheck.types import (
    DecimalType,
    DictType,
    EnumType,
    JsonType,
    ListType,
    RecordType,
    TextType,
    Type,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from agm.agl.runtime.agents import AgentRegistry
    from agm.agl.runtime.codec import ParseResult
    from agm.agl.runtime.contract import OutputContract
    from agm.agl.runtime.request import ValidationError
    from agm.agl.runtime.trace import TraceStore
    from agm.agl.syntax.spans import SourceSpan
    from agm.agl.typecheck.env import CheckedProgram, TypeEnvironment


_Ordered = TypeVar("_Ordered", int, decimal.Decimal, str)

# Pinned decimal context for all AgL arithmetic (F7).  AgL semantics must not
# depend on the host's ambient ``decimal`` context — a host that lowered
# ``getcontext().prec`` would otherwise change results such as ``1 / 3``.  We
# evaluate every program under this explicit context (28-digit precision, the
# stdlib default, with ROUND_HALF_EVEN banker's rounding) via
# ``decimal.localcontext`` in ``Interpreter.execute``.
_AGL_DECIMAL_CONTEXT = decimal.Context(prec=28, rounding=decimal.ROUND_HALF_EVEN)


def _make_exc_value(
    type_name: str, message: str, *, trace_id: str = "", **extra: Value
) -> ExceptionValue:
    """Create an ``ExceptionValue`` with ``message`` and optional extra fields.

    *trace_id* links this exception to the corresponding record in the trace
    file (design §8.1 / §12.6).  When a trace store is active the caller
    passes the event trace_id obtained from the store; otherwise the empty
    string is used (no-log mode).
    """
    fields: dict[str, Value] = {
        "message": TextValue(message),
        "trace_id": TextValue(trace_id),
    }
    fields.update(extra)
    return ExceptionValue(type_name=type_name, fields=fields)


def _make_match_error(subject: Value, *, trace_id: str = "") -> "ExceptionValue":
    """Create a ``MatchError`` ``ExceptionValue`` for a non-matching *subject*.

    Populates ``scrutinee_type`` (the AgL type-name string of the value) and
    ``scrutinee`` (the JSON-shaped representation of the value) per design §8.1.
    *trace_id* links the exception to its eventual ``exception`` trace record
    (design §8.1 / §12.6); the caller mints it via the trace store.
    """
    scrutinee_type = _describe_value(subject)
    scrutinee_json = value_to_json_obj(subject)
    return _make_exc_value(
        "MatchError",
        f"Non-exhaustive case: no pattern matched value of type {scrutinee_type!r}",
        trace_id=trace_id,
        scrutinee_type=TextValue(scrutinee_type),
        scrutinee=JsonValue(scrutinee_json),
    )


class Interpreter:
    """Tree-walking interpreter for a checked AgL program.

    ``checked``   — the type-checked program with side tables.
    ``registry``  — the host agent registry.
    ``contracts`` — materialized ``OutputContract`` per agent-call node_id.
    ``type_env`` — the ``TypeEnvironment`` from the checked program, used to
        resolve record/enum constructors at runtime.
    ``loop_limit`` — default bound for ``do`` loops without an explicit limit.
    ``strict_json`` — default strict-JSON flag for codec operations.
    ``source`` — the normalized program source text.  Threaded in so error
        sites can recover the exact source slice for a node via its span
        offsets (e.g. ``MaxIterationsExceeded.condition``); general and
        reusable for any future error context that wants source text.
    """

    def __init__(
        self,
        checked: "CheckedProgram",
        registry: "AgentRegistry",
        contracts: dict[int, "OutputContract"],
        type_env: "TypeEnvironment",
        *,
        loop_limit: int,
        strict_json: bool,
        source: str = "",
        shell_exec_timeout: float | None = None,
        trace: "TraceStore | None" = None,
    ) -> None:
        from agm.agl.runtime.trace import noop_trace

        self._checked = checked
        self._registry = registry
        self._contracts = contracts
        self._type_env = type_env
        # Span offsets index the *normalized* source (universal newlines; see
        # the scanner module docstring), so normalize here to match before any
        # offset-based slicing.  Shared helper keeps this identical to the
        # scanner's normalization without depending on the lexer (F10).
        self._source = normalize_newlines(source)
        self._loop_limit = loop_limit
        self._strict_json = strict_json
        self._shell_exec_timeout = shell_exec_timeout
        # Trace store: no-op when not provided (no-log mode or direct tests).
        self._trace: "TraceStore" = trace if trace is not None else noop_trace()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute(self, root_scope: Scope) -> None:
        """Execute the program body in *root_scope*.

        All arithmetic runs under a pinned 28-digit ``decimal`` context
        (``_AGL_DECIMAL_CONTEXT``) so AgL semantics never depend on the host's
        ambient context (F7).

        May raise ``AglRaise`` for uncaught AgL exceptions.
        """
        with decimal.localcontext(_AGL_DECIMAL_CONTEXT):
            for stmt in self._checked.resolved.program.body:
                self._exec_stmt(stmt, root_scope)

    # ------------------------------------------------------------------
    # Statement execution
    # ------------------------------------------------------------------

    def _exec_stmt(self, stmt: Stmt, scope: Scope) -> None:
        if isinstance(stmt, LetDecl):
            self._exec_let(stmt, scope)
        elif isinstance(stmt, VarDecl):
            self._exec_var(stmt, scope)
        elif isinstance(stmt, SetStmt):
            self._exec_set(stmt, scope)
        elif isinstance(stmt, PassStmt):
            pass
        elif isinstance(stmt, PrintStmt):
            self._exec_print(stmt, scope)
        elif isinstance(stmt, ExprStmt):
            self._eval_expr(stmt.expr, scope)  # result discarded
        elif isinstance(stmt, DoUntil):
            self._exec_do_until(stmt, scope)
        elif isinstance(stmt, IfStmt):
            self._exec_if(stmt, scope)
        elif isinstance(stmt, CaseStmt):
            self._exec_case_stmt(stmt, scope)
        elif isinstance(stmt, TryCatch):
            self._exec_try_catch(stmt, scope)
        elif isinstance(stmt, Raise):
            self._exec_raise(stmt, scope)
        elif isinstance(stmt, (InputDecl, AgentDecl)):
            pass  # static declarations: resolved before execution, no runtime action
        elif isinstance(stmt, (RecordDef, EnumDef, TypeAlias)):
            pass  # type declarations: no runtime action
        else:
            assert_never(stmt)  # pragma: no cover

    def _exec_let(self, stmt: LetDecl, scope: Scope) -> None:
        value = self._eval_expr(stmt.value, scope)
        # The checker records a declared type for every let binding; coerce the
        # value toward it (int → decimal, element-wise in containers, json
        # boundary wrapping — design §5.8).
        target_type = self._binding_type_for(stmt.node_id)
        scope.define(stmt.name, _coerce(value, target_type), mutable=False, decl_span=stmt.span)

    def _exec_var(self, stmt: VarDecl, scope: Scope) -> None:
        value = self._eval_expr(stmt.value, scope)
        target_type = self._binding_type_for(stmt.node_id)
        scope.define(stmt.name, _coerce(value, target_type), mutable=True, decl_span=stmt.span)

    def _exec_set(self, stmt: SetStmt, scope: Scope) -> None:
        value = self._eval_expr(stmt.value, scope)
        # Coerce toward the mutable binding's declared type (design §5.8).
        ref = self._checked.resolved.resolution[stmt.node_id]
        target_type = self._binding_type_for(ref.decl_node_id)
        coerced = _coerce(value, target_type)
        scope.set_value(stmt.target, coerced)
        self._trace.mutation(name=stmt.target, value=coerced, span=stmt.span)

    def _binding_type_for(self, decl_node_id: int) -> Type:
        """Return the declared type the checker recorded for a binding node.

        The type checker assigns a declared type to every ``let``/``var``/input
        binding before evaluation, so this is always present.
        """
        target_type = self._type_env.get_binding_type(decl_node_id)
        assert target_type is not None, "binding type must be recorded by the checker"
        return target_type

    def _exec_print(self, stmt: PrintStmt, scope: Scope) -> None:
        from agm.agl.runtime.render import render_value

        value = self._eval_expr(stmt.value, scope)
        rendered = render_value(value)
        print(rendered)
        self._trace.print_stmt(rendered=rendered, span=stmt.span)

    def _walk_template(self, expr: Template, scope: Scope) -> str:
        """Walk *expr*'s segments, concatenate text segments verbatim, and
        render interpolation segments with uniform ``render_value`` rendering.
        """
        from agm.agl.runtime.render import render_value

        parts: list[str] = []
        for seg in expr.segments:
            if isinstance(seg, TextSegment):
                parts.append(seg.text)
            elif isinstance(seg, InterpSegment):
                value = self._eval_expr(seg.expr, scope)
                parts.append(render_value(value))
            else:
                assert_never(seg)  # pragma: no cover
        return "".join(parts)

    def _exec_do_until(self, stmt: DoUntil, scope: Scope) -> None:
        limit = stmt.limit if stmt.limit is not None else self._loop_limit
        last_cond = False
        for iteration in range(limit):
            # Each iteration opens a fresh nested scope.
            iter_scope = Scope(parent=scope)
            for s in stmt.body:
                self._exec_stmt(s, iter_scope)
            cond = self._eval_expr(stmt.condition, iter_scope)
            # The checker requires the until-condition to be bool, so this is
            # always a BoolValue.
            last_cond = self._require_bool(cond)
            if last_cond:
                return
        # Exhausted without condition becoming true.  Populate the §8.1 schema:
        # ``condition`` is the until-expression's source text (sliced via span
        # offsets), ``last_condition_value`` is its final evaluation result, and
        # ``metadata`` is an empty json placeholder until the M4 trace store.
        raise AglRaise(
            _make_exc_value(
                "MaxIterationsExceeded",
                f"Loop exhausted after {limit} iterations",
                trace_id=self._trace.new_event_id(),
                limit=IntValue(limit),
                condition=TextValue(self._source_slice(stmt.condition.span)),
                last_condition_value=BoolValue(last_cond),
                metadata=JsonValue(None),
            ),
            span=stmt.span,  # F7: thread raise-site span for §12.6 source location
        )

    def _source_slice(self, span: "SourceSpan") -> str:
        """Return the exact normalized-source text covered by *span*.

        Uses the span's 0-based, end-exclusive character offsets into the
        normalized source threaded into the interpreter.  Returns ``""`` when no
        source was provided (e.g. direct construction in unit tests).
        """
        if not self._source:
            return ""
        return self._source[span.start_offset : span.end_offset]

    def _exec_if(self, stmt: IfStmt, scope: Scope) -> None:
        for branch in stmt.branches:
            # The checker requires every if-condition to be bool (design §4.3),
            # so ``_eval_expr`` always yields a BoolValue here.  An else branch
            # has no condition and is always taken (short-circuits the eval).
            take = isinstance(branch.cond, ElseSentinel) or self._require_bool(
                self._eval_expr(branch.cond, scope)
            )
            if take:
                branch_scope = Scope(parent=scope)
                for s in branch.body:
                    self._exec_stmt(s, branch_scope)
                return
        # No branch matched and no else: do nothing.

    def _exec_case_stmt(self, stmt: CaseStmt, scope: Scope) -> None:
        subject = self._eval_expr(stmt.subject, scope)
        for branch in stmt.branches:
            matched, bindings = _match_pattern(branch.pattern, subject)
            if matched:
                branch_scope = Scope(parent=scope)
                for name, val in bindings.items():
                    branch_scope.define(
                        name, val, mutable=False, decl_span=branch.span
                    )
                for s in branch.body:
                    self._exec_stmt(s, branch_scope)
                return
        # No match: raise MatchError with scrutinee_type and scrutinee fields.
        raise AglRaise(
            _make_match_error(subject, trace_id=self._trace.new_event_id()),
            span=stmt.span,  # F7: thread raise-site span for §12.6 source location
        )

    def _exec_try_catch(self, stmt: TryCatch, scope: Scope) -> None:
        try:
            try_scope = Scope(parent=scope)
            for s in stmt.body:
                self._exec_stmt(s, try_scope)
        except AglRaise as exc:
            for handler in stmt.handlers:
                if _matches_catch(handler, exc.exc):
                    catch_scope = Scope(parent=scope)
                    if handler.binding is not None:
                        catch_scope.define(
                            handler.binding,
                            exc.exc,
                            mutable=False,
                            decl_span=handler.span,
                        )
                    for s in handler.body:
                        self._exec_stmt(s, catch_scope)
                    return
            # No handler matched: re-propagate.
            raise

    def _exec_raise(self, stmt: Raise, scope: Scope) -> None:
        exc_val = self._eval_expr(stmt.exc, scope)
        if isinstance(exc_val, ExceptionValue):
            raise AglRaise(
                exc_val,
                span=stmt.span,  # F7: thread raise-site span for §12.6 source location
            )
        # Unreachable: the checker requires an exception-typed operand for
        # 'raise' (design §8.3).
        raise AssertionError(  # pragma: no cover
            f"raise: expected an ExceptionValue, got {type(exc_val).__name__}"
        )

    # ------------------------------------------------------------------
    # Expression evaluation
    # ------------------------------------------------------------------

    def _eval_expr(self, expr: Expr, scope: Scope) -> Value:
        # Scalar literals
        if isinstance(expr, IntLit):
            return IntValue(expr.value)
        if isinstance(expr, DecimalLit):
            return DecimalValue(expr.value)
        if isinstance(expr, BoolLit):
            return BoolValue(expr.value)
        if isinstance(expr, NullLit):
            return JsonValue(None)
        if isinstance(expr, StringLit):
            return TextValue(expr.value)
        # Name / member access
        if isinstance(expr, VarRef):
            return self._eval_var_ref(expr, scope)
        if isinstance(expr, FieldAccess):
            return self._eval_field_access(expr, scope)
        # Compound expressions (templates, calls, operators, …).
        if isinstance(expr, Template):
            return self._eval_template(expr, scope)
        if isinstance(expr, AgentCall):
            return self._eval_agent_call(expr, scope)
        if isinstance(expr, Constructor):
            return self._eval_constructor(expr, scope)
        if isinstance(expr, BinaryOp):
            return self._eval_binary_op(expr, scope)
        if isinstance(expr, UnaryNot):
            return self._eval_unary_not(expr, scope)
        if isinstance(expr, UnaryNeg):
            return self._eval_unary_neg(expr, scope)
        if isinstance(expr, IsTest):
            return self._eval_is_test(expr, scope)
        if isinstance(expr, CaseExpr):
            return self._eval_case_expr(expr, scope)
        if isinstance(expr, ListLit):
            return self._eval_list_lit(expr, scope)
        if isinstance(expr, DictLit):
            return self._eval_dict_lit(expr, scope)
        assert_never(expr)  # pragma: no cover

    def _eval_unary_not(self, expr: UnaryNot, scope: Scope) -> BoolValue:
        operand = self._eval_expr(expr.operand, scope)
        if isinstance(operand, BoolValue):
            return BoolValue(not operand.value)
        # Unreachable: the checker requires a bool operand for 'not' (design §4.3).
        raise AssertionError(  # pragma: no cover
            f"not: expected bool, got {type(operand).__name__}"
        )

    def _eval_unary_neg(self, expr: UnaryNeg, scope: Scope) -> IntValue | DecimalValue:
        operand = self._eval_expr(expr.operand, scope)
        if isinstance(operand, IntValue):
            return IntValue(-operand.value)
        if isinstance(operand, DecimalValue):
            return DecimalValue(-operand.value)
        # Unreachable: the type checker requires a numeric operand for unary '-'.
        raise RuntimeError(  # pragma: no cover
            f"unary -: expected number, got {type(operand).__name__}"
        )

    def _eval_list_lit(self, expr: ListLit, scope: Scope) -> ListValue:
        elements = tuple(self._eval_expr(e, scope) for e in expr.elements)
        return ListValue(elements=elements)

    def _eval_dict_lit(self, expr: DictLit, scope: Scope) -> DictValue:
        entries: dict[str, Value] = {}
        for entry in expr.entries:
            entries[entry.key.value] = self._eval_expr(entry.value, scope)
        return DictValue(entries=entries)

    def _eval_var_ref(self, expr: VarRef, scope: Scope) -> Value:
        binding = scope.lookup(expr.name)
        if binding is None:  # pragma: no cover
            # Unreachable: name resolution rejects undefined variables statically.
            raise RuntimeError(f"Undefined variable at runtime: {expr.name!r}")
        return binding.value

    def _eval_field_access(self, expr: FieldAccess, scope: Scope) -> Value:
        obj = self._eval_expr(expr.obj, scope)
        # The type checker only admits field access on record and exception
        # values, and it guarantees the named field exists, so missing-field /
        # wrong-kind cases are statically unreachable.
        if isinstance(obj, (RecordValue, ExceptionValue)):
            return obj.fields[expr.field]
        raise RuntimeError(  # pragma: no cover
            f"Field access on non-record/exception: {type(obj).__name__}"
        )

    def _eval_template(self, expr: Template, scope: Scope) -> TextValue:
        """Evaluate a template by rendering each segment and concatenating."""
        return TextValue(self._walk_template(expr, scope))

    def _eval_agent_call(self, expr: AgentCall, scope: Scope) -> Value:
        """Dispatch an agent call and return the typed result."""
        from agm.agl.scope.symbols import CallKind

        call_kind = self._checked.resolved.call_kinds.get(expr.node_id)
        if call_kind == CallKind.shell_exec:
            return self._exec_shell_exec(expr, scope)

        # Determine the agent name for dispatch.
        if call_kind is None or call_kind == CallKind.default_agent:
            agent_name = "prompt"
        else:
            agent_name = expr.agent

        # Get the output contract for this call site.
        contract = self._contracts.get(expr.node_id)
        if contract is None:
            # Fallback for text-target calls without a contract (shouldn't happen
            # after typecheck, but defensive).
            from agm.agl.runtime.codec import TextCodec
            from agm.agl.runtime.contract import OutputContract

            contract = OutputContract(
                target_type=TextType(),
                codec=TextCodec(),
                strict_json=None,
                format_instructions="",
                json_schema=None,
            )

        # Resolve the parse policy (from call options or runtime default).
        parse_policy = expr.options.parse_policy

        # Perform the call with retry support.
        return self._dispatch_agent_call(
            agent_name=agent_name,
            expr=expr,
            scope=scope,
            contract=contract,
            parse_policy=parse_policy,
        )

    def _dispatch_agent_call(
        self,
        agent_name: str,
        expr: AgentCall,
        scope: Scope,
        contract: "OutputContract",
        parse_policy: object,  # ParsePolicy | None
    ) -> Value:
        """Execute an agent call with the given contract and parse policy.

        Re-dispatches per attempt, feeding the previous invalid output and
        validation errors back to the agent (corrective retry).  The per-attempt
        parse and the final ``AgentParseError`` construction are delegated to the
        shared :meth:`_run_parse_attempts` helper; this method owns only the
        agent-specific concern of *source acquisition* (dispatch + the
        ``agent_call_attempt``/``parse_result`` trace records).
        """
        from agm.agl.runtime.request import AgentRequest

        # Render the template once (re-used on retries per design).
        prompt_text = self._eval_template(expr.template, scope).value
        call_span = expr.span

        def acquire(
            attempt: int,
            last_raw: str | None,
            last_errors: tuple[ValidationError, ...],
        ) -> tuple[str, str]:
            # Emit agent_call_attempt record, dispatch, and return (raw, trace_id).
            attempt_trace_id = self._trace.agent_call_attempt(
                agent=agent_name,
                attempt=attempt,
                prompt=prompt_text,
                span=call_span,
            )
            request = AgentRequest(
                agent=agent_name,
                prompt=prompt_text,
                attempt=attempt,
                previous_invalid_output=last_raw,
                validation_errors=list(last_errors),
                output_contract=contract,
            )
            try:
                response = self._registry.dispatch(agent_name, request)
            except AglRaise as exc:
                # F2: ``AgentRegistry.dispatch`` raises ``AglRaise`` (e.g.
                # ``AgentCallError``) without a span.  Attach this call site's
                # span so an uncaught agent-call failure reports its source
                # location (design §12.6).  Only fill a missing span — never
                # overwrite a span the raise site already supplied.
                if exc.span is None:
                    exc.span = call_span
                raise
            return response.content, attempt_trace_id

        def on_parsed(raw: str, result: "ParseResult") -> None:
            # Emit a parse_result record per attempt (agent path only).
            # F4: use result.error_msg as fallback when errors is empty so that
            # totally-unparseable output (no structured validation errors) still
            # produces a non-empty error_summary in the trace record.
            if result.errors:
                error_summary = "; ".join(e.message for e in result.errors)
            else:
                error_summary = result.error_msg
            self._trace.parse_result(
                ok=result.ok and result.value is not None,
                raw=raw,
                normalized_raw=result.normalized_raw or "",
                error_summary=error_summary,
                span=call_span,
            )

        def make_failure_message(last_raw: str | None, max_attempts: int) -> str:
            return (
                f"Agent {agent_name!r} failed to produce a valid "
                f"{contract.target_type!r} after {max_attempts} attempt(s). "
                f"Last output: {last_raw!r}"
            )

        return self._run_parse_attempts(
            acquire=acquire,
            contract=contract,
            parse_policy=parse_policy,
            agent_label=agent_name,
            make_failure_message=make_failure_message,
            on_parsed=on_parsed,
            raise_span=call_span,  # F7: thread call span for §12.6 source location
        )

    def _run_parse_attempts(
        self,
        *,
        acquire: "Callable[[int, str | None, tuple[ValidationError, ...]], tuple[str, str]]",
        contract: "OutputContract",
        parse_policy: object,  # ParsePolicy | None
        agent_label: str,
        make_failure_message: "Callable[[str | None, int], str]",
        on_parsed: "Callable[[str, ParseResult], None] | None" = None,
        raise_span: "SourceSpan | None" = None,
    ) -> Value:
        """Run the shared parse/retry loop and return the parsed value.

        Owns everything common to the agent-call and ``exec`` paths: the
        attempt count derived from ``parse_policy``, the effective ``strict_json``
        resolution, the per-attempt codec parse, and the ``AgentParseError``
        construction when every attempt fails (design §4.12, §7.5/§7.9).

        The caller supplies *acquire*, which performs source acquisition for a
        given attempt — re-dispatching an agent or re-running ``exec`` — and
        returns ``(raw, trace_id)``.
        *make_failure_message* builds the path-specific ``AgentParseError``
        message from ``(last_raw, max_attempts)``.  When provided, *on_parsed* is
        invoked after each parse so the agent path can emit its per-attempt
        ``parse_result`` trace record; ``exec`` passes ``None`` (it emits a single
        ``exec_command`` record outside this loop).
        """
        from agm.agl.syntax.nodes import RetryPolicy

        # Determine max attempts (AbortPolicy or None → one attempt only).
        if isinstance(parse_policy, RetryPolicy):
            max_attempts = 1 + parse_policy.extra
        else:
            max_attempts = 1

        # Determine effective strict_json (per-call override, else runtime default).
        if contract.strict_json is not None:
            effective_strict = contract.strict_json
        else:
            effective_strict = self._strict_json

        # Reuse the schema already materialized on the contract so the codec never
        # re-derives it per attempt (BONUS, M3b).
        schema = contract.json_schema if isinstance(contract.json_schema, dict) else None

        last_raw: str | None = None
        last_normalized: str | None = None
        last_errors: tuple[ValidationError, ...] = ()
        last_trace_id = ""
        for attempt in range(max_attempts):
            raw, last_trace_id = acquire(attempt, last_raw, last_errors)
            result = contract.codec.parse(
                raw,
                contract.target_type,
                strict_json=effective_strict,
                schema=schema,
            )
            if on_parsed is not None:
                on_parsed(raw, result)
            if result.ok and result.value is not None:
                return result.value
            last_raw = raw
            last_normalized = result.normalized_raw
            # F4: when the codec returns no structured errors but does have an
            # error_msg (e.g. "Strict JSON parse failed: …" for totally-
            # unparseable output), synthesize an ``invalid_json`` ValidationError
            # so the failure reason is fed back to the agent on the next retry
            # attempt and recorded in the parse_result trace (design §7.5 ext).
            if result.errors:
                last_errors = result.errors
            elif result.error_msg:
                from agm.agl.runtime.request import ValidationError as _VE

                last_errors = (
                    _VE(
                        category="invalid_json",
                        message=result.error_msg,
                        path="$",
                        field=None,
                    ),
                )
            else:
                last_errors = ()

        # All attempts exhausted → raise AgentParseError.  Validation errors are
        # threaded as JSON-shaped values (a list of per-error objects) so the
        # exception's ``validation_errors`` field (text-renderable) matches the
        # exception field schema (design §7.5 / §7.9).  ``normalized_raw`` carries
        # the recovered/extracted JSON text when the failure was a schema or
        # conversion error (F5); it falls back to the raw output when no value
        # could be recovered at all.  ``trace_id`` links to the last attempt's
        # record so the exception can be cross-referenced (§8.1/§12.6).
        errors_json: list[object] = [e.to_json_obj() for e in last_errors]
        normalized_text = last_normalized if last_normalized is not None else (last_raw or "")
        raise AglRaise(
            _make_exc_value(
                "AgentParseError",
                make_failure_message(last_raw, max_attempts),
                trace_id=last_trace_id,
                raw=TextValue(last_raw or ""),
                normalized_raw=TextValue(normalized_text),
                agent=TextValue(agent_label),
                attempts=IntValue(max_attempts),
                target_type=TextValue(str(contract.target_type)),
                expected_schema=JsonValue(contract.json_schema),
                validation_errors=JsonValue(errors_json),
                metadata=JsonValue(None),
            ),
            span=raise_span,  # F7: thread raise-site span for §12.6 source location
        )

    def _eval_template_for_shell(self, expr: Template, scope: Scope) -> str:
        """Render a template for use as a shell command (design §4.12, §11.13).

        Each interpolated segment is rendered with uniform ``render_value``
        rendering and inserted verbatim — no shell quoting is applied.
        """
        return self._walk_template(expr, scope)

    def _exec_shell_exec(self, expr: AgentCall, scope: Scope) -> Value:
        """Execute an ``exec`` shell call and return the typed result (§11.13).

        Steps:
        1. Render the template with shell-safe interpolation.
        2. Run via ``sh -c <command>`` under the configured idle timeout.
        3. Nonzero exit or timeout → raise ``ExecError``.
        4. Spawn failure (``sh`` itself not found) → raise ``ExecError`` with
           ``exit_code=-1`` (host-broken environment — not a user program error,
           but ``ExecError`` is the closest in-language representation since the
           design §4.12/§11.13 names only ``ExecError`` for shell failures and
           does not define a separate spawn-failure exception for ``exec``).
        5. Success: stdout with trailing newlines stripped, parsed through the
           same codec/parse-policy path as agent output (§4.12 item 4).
        """
        from agm.core.process import run_capture_result

        command = self._eval_template_for_shell(expr.template, scope)

        exec_span = expr.span

        def execute_command() -> tuple[str, str]:
            # ``isolate_process_group=True`` ensures an idle timeout tears down
            # the whole process tree, including grandchildren.
            result = run_capture_result(
                ["sh", "-c", command],
                idle_timeout=self._shell_exec_timeout,
                isolate_process_group=True,
            )

            # Spawn failure: sh itself could not be launched.
            if result.spawn_error is not None:
                self._trace.exec_command(
                    command=command,
                    exit_code=-1,
                    duration=result.elapsed,
                    stdout="",
                    stderr=result.spawn_error,
                    timed_out=False,
                    span=exec_span,
                )
                raise AglRaise(
                    _make_exc_value(
                        "ExecError",
                        f"Failed to spawn shell: {result.spawn_error}",
                        trace_id=self._trace.new_event_id(),
                        command=TextValue(command),
                        exit_code=IntValue(-1),
                        stdout=TextValue(""),
                        stderr=TextValue(""),
                        timed_out=BoolValue(False),
                    ),
                    span=exec_span,
                )

            # Nonzero exit or timeout → ExecError (design §11.13 item 3).
            if result.timed_out or (result.returncode is not None and result.returncode != 0):
                exit_code = result.returncode if result.returncode is not None else -1
                self._trace.exec_command(
                    command=command,
                    exit_code=exit_code,
                    duration=result.elapsed,
                    stdout=result.stdout.rstrip("\n"),
                    stderr=result.stderr.rstrip("\n"),
                    timed_out=result.timed_out,
                    span=exec_span,
                )
                if result.timed_out:
                    message = f"Shell command timed out (idle timeout exceeded): {command!r}"
                else:
                    message = f"Shell command exited with code {exit_code}: {command!r}"
                raise AglRaise(
                    _make_exc_value(
                        "ExecError",
                        message,
                        trace_id=self._trace.new_event_id(),
                        command=TextValue(command),
                        exit_code=IntValue(exit_code),
                        stdout=TextValue(result.stdout.rstrip("\n")),
                        stderr=TextValue(result.stderr.rstrip("\n")),
                        timed_out=BoolValue(result.timed_out),
                    ),
                    span=exec_span,
                )

            stdout = result.stdout.rstrip("\n")
            trace_id = self._trace.exec_command(
                command=command,
                exit_code=result.returncode if result.returncode is not None else 0,
                duration=result.elapsed,
                stdout=stdout,
                stderr=result.stderr.rstrip("\n"),
                timed_out=False,
                span=exec_span,
            )
            return stdout, trace_id

        stdout, exec_trace_id = execute_command()

        # Determine the target type and contract for this call site.
        contract = self._contracts.get(expr.node_id)
        if contract is None:
            # No contract → text target (fallback, mirrors _eval_agent_call).
            return TextValue(stdout)

        # For text targets return verbatim (no parsing needed).
        if isinstance(contract.target_type, TextType):
            return TextValue(stdout)

        # Non-text target: parse + validate via the same codec/parse-policy path
        # as agent output. Retry attempts re-run the command and each invocation
        # emits its own ``exec_command`` trace record.
        def acquire(
            attempt: int,
            _last_raw: str | None,
            _last_errors: tuple[ValidationError, ...],
        ) -> tuple[str, str]:
            if attempt == 0:
                return stdout, exec_trace_id
            return execute_command()

        def make_failure_message(last_raw: str | None, max_attempts: int) -> str:
            return (
                f"exec output failed to parse as {contract.target_type!r} "
                f"after {max_attempts} attempt(s). Last output: {last_raw!r}"
            )

        return self._run_parse_attempts(
            acquire=acquire,
            contract=contract,
            parse_policy=expr.options.parse_policy,
            agent_label="exec",
            make_failure_message=make_failure_message,
            raise_span=exec_span,  # F7: thread exec span for §12.6 source location
        )

    def _eval_constructor(self, expr: Constructor, scope: Scope) -> Value:
        """Evaluate a record, enum-variant, or exception constructor."""
        # Evaluate arguments.  Template expressions are handled by ``_eval_expr``
        # (console rendering, no ``<dsl-value>`` boundary tags).
        arg_values: dict[str, Value] = {}
        for arg in expr.args:
            arg_values[arg.name] = self._eval_expr(arg.value, scope)

        # The checker resolved this constructor's type (records the resolved
        # nominal type in ``node_types``), already resolving any alias qualifier
        # transparently (design §5.4) — the interpreter never re-resolves it.
        typ = self._checked.node_types.get(expr.node_id)

        # The type checker validates that the constructor names a known type and
        # that every supplied field is declared, so each ``arg`` field is present
        # in the corresponding record/variant field table.
        if isinstance(typ, RecordType):
            coerced = {
                fname: _coerce(fval, typ.fields[fname]) for fname, fval in arg_values.items()
            }
            return RecordValue(type_name=typ.name, fields=coerced)

        from agm.agl.typecheck.types import ExceptionType as ExcType

        if isinstance(typ, ExcType):
            # Exception constructor: inject a real ``trace_id`` from the trace
            # store (plan §9.6 / design §8.1). A fresh event-level id is
            # generated so the raised exception can be cross-referenced with
            # any matching record in the trace file.
            exc_trace_id = self._trace.new_event_id()
            fields: dict[str, Value] = {"trace_id": TextValue(exc_trace_id)}
            for fname, fval in arg_values.items():
                field_type = typ.fields.get(fname)
                fields[fname] = _coerce(fval, field_type) if field_type is not None else fval
            return ExceptionValue(type_name=typ.name, fields=fields)

        # Otherwise an enum-variant constructor.  The variant is the constructor's
        # own name (the qualifier, if any, only selects the enum).
        assert isinstance(typ, EnumType), "constructor type must be record, enum, or exception"
        variant_name = expr.name
        variant_fields = typ.variants.get(variant_name, {})
        coerced2 = {
            fname: _coerce(fval, variant_fields[fname]) for fname, fval in arg_values.items()
        }
        return EnumValue(type_name=typ.name, variant=variant_name, fields=coerced2)

    def _eval_binary_op(self, expr: BinaryOp, scope: Scope) -> Value:
        op = expr.op
        left = self._eval_expr(expr.left, scope)

        # Short-circuit for and/or (right operand evaluated lazily). The checker
        # guarantees both operands are bool, so the result is always a BoolValue
        # (design §4.3).
        if op is BinOp.AND or op is BinOp.OR:
            return self._eval_bool_binop(op, left, expr.right, scope)

        right = self._eval_expr(expr.right, scope)

        # Arithmetic.
        if op is BinOp.ADD:
            return _add(left, right)
        if op is BinOp.SUB or op is BinOp.MUL:
            return _arith(left, right, op)
        if op is BinOp.DIV:
            return _div(left, right, trace=self._trace)

        # Comparison.
        if (
            op is BinOp.EQ
            or op is BinOp.NEQ
            or op is BinOp.LT
            or op is BinOp.LE
            or op is BinOp.GT
            or op is BinOp.GE
        ):
            return _compare(left, right, op)

        # Membership.
        if op is BinOp.IN:
            return _in_op(left, right)

        assert_never(op)  # pragma: no cover

    def _eval_bool_binop(
        self, op: BinOp, left: Value, right_expr: Expr, scope: Scope
    ) -> BoolValue:
        """Short-circuit evaluation of ``and``/``or`` returning a BoolValue.

        The checker guarantees both operands are bool. ``and`` short-circuits
        when the left operand is ``False``; ``or`` short-circuits when it is
        ``True`` — in those cases the right operand is NOT evaluated.
        """
        left_bool = self._require_bool(left)
        if op is BinOp.AND:
            if not left_bool:
                return BoolValue(False)
        else:  # BinOp.OR
            if left_bool:
                return BoolValue(True)
        right = self._eval_expr(right_expr, scope)
        return BoolValue(self._require_bool(right))

    @staticmethod
    def _require_bool(value: Value) -> bool:
        if isinstance(value, BoolValue):
            return value.value
        # Unreachable: the checker requires bool operands for 'and'/'or'.
        raise AssertionError(  # pragma: no cover
            f"and/or operand is not a bool: {type(value).__name__}"
        )

    def _eval_is_test(self, expr: IsTest, scope: Scope) -> BoolValue:
        value = self._eval_expr(expr.expr, scope)
        if isinstance(value, EnumValue):
            variant_matches = value.variant == expr.variant
            result = variant_matches != expr.negated
            return BoolValue(result)
        # Unreachable: 'is' tests require an enum-typed operand statically.
        raise RuntimeError(  # pragma: no cover
            f"is test on non-enum value: {type(value).__name__}"
        )

    def _eval_case_expr(self, expr: CaseExpr, scope: Scope) -> Value:
        subject = self._eval_expr(expr.subject, scope)
        for branch in expr.branches:
            matched, bindings = _match_pattern(branch.pattern, subject)
            if matched:
                branch_scope = Scope(parent=scope)
                for name, val in bindings.items():
                    branch_scope.define(
                        name, val, mutable=False, decl_span=branch.span
                    )
                return self._eval_expr(branch.body, branch_scope)
        raise AglRaise(
            _make_match_error(subject, trace_id=self._trace.new_event_id()),
            span=expr.span,  # F7: thread raise-site span for §12.6 source location
        )


# ---------------------------------------------------------------------------
# Helpers: coercion, comparison, arithmetic, pattern matching
# ---------------------------------------------------------------------------


def _coerce(value: Value, target: Type) -> Value:
    """Coerce *value* toward its statically-checked *target* type (design §5.8).

    The checker has already proven the value is assignable to *target*; this
    materializes the implicit coercions in the runtime representation:

    - ``int → decimal`` widening (the single scalar coercion);
    - ``json`` boundary: a JSON-shaped value bound to a ``json`` slot is stored
      in the one canonical ``json`` representation — a :class:`JsonValue`
      wrapping the JSON-shaped object (so ``print``/rendering/equality are
      consistent regardless of the literal it came from);
    - element-wise recursion through ``list``/``dict``/record/enum so e.g.
      ``int`` widens to ``decimal`` and json-wrapping applies inside containers.
    """
    if isinstance(target, JsonType):
        # Already json → leave as-is; otherwise wrap the JSON-shaped object.
        if isinstance(value, JsonValue):
            return value
        return JsonValue(value_to_json_obj(value))
    if isinstance(target, DecimalType) and isinstance(value, IntValue):
        return DecimalValue(decimal.Decimal(value.value))
    if isinstance(target, ListType) and isinstance(value, ListValue):
        return ListValue(elements=tuple(_coerce(e, target.elem) for e in value.elements))
    if isinstance(target, DictType) and isinstance(value, DictValue):
        return DictValue(
            entries={k: _coerce(v, target.value) for k, v in value.entries.items()}
        )
    if isinstance(target, RecordType) and isinstance(value, RecordValue):
        return RecordValue(
            type_name=value.type_name,
            fields={
                k: _coerce(v, target.fields[k]) if k in target.fields else v
                for k, v in value.fields.items()
            },
        )
    if isinstance(target, EnumType) and isinstance(value, EnumValue):
        variant_fields = target.variants.get(value.variant, {})
        return EnumValue(
            type_name=value.type_name,
            variant=value.variant,
            fields={
                k: _coerce(v, variant_fields[k]) if k in variant_fields else v
                for k, v in value.fields.items()
            },
        )
    return value


def _add(left: Value, right: Value) -> Value:
    """Addition: numeric or text concatenation."""
    if isinstance(left, IntValue) and isinstance(right, IntValue):
        return IntValue(left.value + right.value)
    if isinstance(left, (IntValue, DecimalValue)) and isinstance(right, (IntValue, DecimalValue)):
        return DecimalValue(_to_decimal(left) + _to_decimal(right))
    if isinstance(left, TextValue) and isinstance(right, TextValue):
        return TextValue(left.value + right.value)
    raise RuntimeError(f"Cannot add {type(left).__name__} and {type(right).__name__}")


def _arith(left: Value, right: Value, op: BinOp) -> Value:
    """Subtraction and multiplication."""
    if isinstance(left, IntValue) and isinstance(right, IntValue):
        if op == BinOp.SUB:
            return IntValue(left.value - right.value)
        return IntValue(left.value * right.value)
    if isinstance(left, (IntValue, DecimalValue)) and isinstance(right, (IntValue, DecimalValue)):
        ld, rd = _to_decimal(left), _to_decimal(right)
        if op == BinOp.SUB:
            return DecimalValue(ld - rd)
        return DecimalValue(ld * rd)
    raise RuntimeError(f"Cannot perform {op.value} on {type(left).__name__}")


def _div(left: Value, right: Value, *, trace: "TraceStore") -> Value:
    """Division: always yields decimal.

    *trace* is the run's trace store; on division by zero a fresh event id is
    minted from it so the ``ArithmeticError`` links to its eventual
    ``exception`` trace record (design §8.1 / §12.6).  The id is minted only at
    the raise site, so a successful division costs nothing.
    """
    if isinstance(left, (IntValue, DecimalValue)) and isinstance(right, (IntValue, DecimalValue)):
        rd = _to_decimal(right)
        if rd == decimal.Decimal(0):
            raise AglRaise(
                _make_exc_value(
                    "ArithmeticError",
                    "Division by zero",
                    trace_id=trace.new_event_id(),
                    operation=TextValue("/"),
                )
            )
        return DecimalValue(_to_decimal(left) / rd)
    raise RuntimeError(f"Cannot divide {type(left).__name__}")


def _to_decimal(value: Value) -> decimal.Decimal:
    if isinstance(value, IntValue):
        return decimal.Decimal(value.value)
    if isinstance(value, DecimalValue):
        return value.value
    raise RuntimeError(f"Not a numeric value: {type(value).__name__}")


def _value_eq(left: Value, right: Value) -> bool:
    """Value equality with int→decimal widening (design §4.3).

    The single source of truth for ``=`` comparison and ``in`` membership, so
    ``IntValue(1)`` equals ``DecimalValue(1)`` consistently across both passes.
    Two ``json`` values compare their wrapped trees via ``JsonValue.__eq__``,
    which delegates to ``values._json_eq`` for bool-guarded numeric equivalence
    (design §5.8/§11.9).
    """
    if isinstance(left, IntValue) and isinstance(right, DecimalValue):
        return decimal.Decimal(left.value) == right.value
    if isinstance(left, DecimalValue) and isinstance(right, IntValue):
        return left.value == decimal.Decimal(right.value)
    return left == right


def _compare(left: Value, right: Value, op: BinOp) -> BoolValue:
    """Equality and ordering comparison."""
    if op == BinOp.EQ:
        return BoolValue(_value_eq(left, right))
    if op == BinOp.NEQ:
        return BoolValue(not _value_eq(left, right))

    # Widen for ordering comparison.
    if isinstance(left, IntValue) and isinstance(right, DecimalValue):
        left = DecimalValue(decimal.Decimal(left.value))
    elif isinstance(left, DecimalValue) and isinstance(right, IntValue):
        right = DecimalValue(decimal.Decimal(right.value))

    # Ordering: numeric or text operands of the same kind.  ``_compare`` only
    # reaches here with ``op`` in {LT, LE, GT, GE}; EQ/NEQ are handled earlier
    # and no other op flows into ``_compare``.
    if isinstance(left, IntValue) and isinstance(right, IntValue):
        return _order_result(left.value, right.value, op)
    if isinstance(left, DecimalValue) and isinstance(right, DecimalValue):
        return _order_result(left.value, right.value, op)
    if isinstance(left, TextValue) and isinstance(right, TextValue):
        return _order_result(left.value, right.value, op)
    raise RuntimeError(
        f"Cannot compare {type(left).__name__} and {type(right).__name__}"
    )


def _order_result(left: _Ordered, right: _Ordered, op: BinOp) -> BoolValue:
    """Apply an ordering operator to two same-kind comparable keys.

    Reached only with ``op`` in {LT, LE, GT, GE}; the four ordering ops are
    exhaustive, so the final branch is ``GE`` (no unreachable fallback).
    """
    if op == BinOp.LT:
        return BoolValue(left < right)
    if op == BinOp.LE:
        return BoolValue(left <= right)
    if op == BinOp.GT:
        return BoolValue(left > right)
    # Only GE remains.
    return BoolValue(left >= right)


def _in_op(left: Value, right: Value) -> BoolValue:
    """``x in container`` operator.

    List membership uses the same value-equality semantics as ``=`` (incl.
    int→decimal widening), so ``1 in [1.0]`` is true (design §4.3).
    """
    if isinstance(right, ListValue):
        return BoolValue(any(_value_eq(left, elem) for elem in right.elements))
    if isinstance(right, DictValue):
        if isinstance(left, TextValue):
            return BoolValue(left.value in right.entries)
        return BoolValue(False)
    if isinstance(right, TextValue) and isinstance(left, TextValue):
        return BoolValue(left.value in right.value)
    raise RuntimeError(
        f"Cannot use 'in' with {type(left).__name__} and {type(right).__name__}"
    )


def _matches_catch(handler: CatchClause, exc: ExceptionValue) -> bool:
    """Check if *handler* catches an exception of type *exc.type_name*."""
    if handler.exc_type is None:
        # Bare ``catch`` without a type: catches everything.
        return True
    # Match by type name: catches the exact type or any subtype.
    # In v1 the hierarchy is flat: every concrete type is a subtype of
    # "Exception" (the abstract base).
    if handler.exc_type == "_" or handler.exc_type == "Exception":
        return True
    return handler.exc_type == exc.type_name


def _match_pattern(pattern: Pattern, value: Value) -> tuple[bool, dict[str, Value]]:
    """Try to match *pattern* against *value*.

    Returns ``(matched, bindings)`` where ``bindings`` maps new variable
    names to their bound values.
    """
    from agm.agl.syntax.nodes import (
        BoolLit,
        ConstructorPattern,
        DecimalLit,
        IntLit,
        LiteralPattern,
        NullLit,
        StringLit,
    )

    if isinstance(pattern, WildcardPattern):
        return True, {}

    if isinstance(pattern, VarPattern):
        return True, {pattern.name: value}

    if isinstance(pattern, LiteralPattern):
        lit = pattern.literal
        if isinstance(lit, IntLit):
            pat_val: Value = IntValue(lit.value)
        elif isinstance(lit, DecimalLit):
            pat_val = DecimalValue(lit.value)
        elif isinstance(lit, BoolLit):
            pat_val = BoolValue(lit.value)
        elif isinstance(lit, StringLit):
            pat_val = TextValue(lit.value)
        elif isinstance(lit, NullLit):
            pat_val = JsonValue(None)
        else:
            assert_never(lit)  # pragma: no cover
        # Use ``_value_eq`` so a literal pattern matches with int→decimal widening
        # (e.g. ``case 1.0 of | 1 =>``), consistent with ``1 = 1.0`` (F5).
        return _value_eq(value, pat_val), {}

    if isinstance(pattern, ConstructorPattern):
        # The type checker only admits constructor patterns against enum
        # subjects and guarantees each named field exists on the variant, so the
        # subject is always an EnumValue and every field lookup succeeds.
        assert isinstance(value, EnumValue), "constructor pattern on non-enum value"
        if value.variant != pattern.name:
            return False, {}
        bindings: dict[str, Value] = {}
        for field_pat in pattern.fields:
            field_val = value.fields[field_pat.name]
            matched, sub_bindings = _match_pattern(field_pat.pattern, field_val)
            if not matched:
                return False, {}
            bindings.update(sub_bindings)
        return True, bindings

    assert_never(pattern)  # pragma: no cover


def _describe_value(value: Value) -> str:
    """Return the AgL type-name of *value* (design §8.1 ``scrutinee_type``).

    Nominal values (records, enums, exceptions) report their declared type name;
    built-in values map to their AgL type names (``int``, ``text``, ``bool``,
    ``decimal``, ``json``, ``list``, ``dict``) rather than leaking the Python
    runtime class name (F6).
    """
    if isinstance(value, EnumValue):
        return value.type_name
    if isinstance(value, RecordValue):
        return value.type_name
    if isinstance(value, ExceptionValue):
        return value.type_name
    if isinstance(value, TextValue):
        return "text"
    if isinstance(value, IntValue):
        return "int"
    if isinstance(value, DecimalValue):
        return "decimal"
    if isinstance(value, BoolValue):
        return "bool"
    if isinstance(value, JsonValue):
        return "json"
    if isinstance(value, ListValue):
        return "list"
    # DictValue is the only remaining Value member.
    assert isinstance(value, DictValue), f"unexpected value kind: {type(value).__name__}"
    return "dict"
