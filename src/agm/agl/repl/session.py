"""UI-free incremental session core for the AgL REPL (``ReplSession``).

``ReplSession`` keeps a **persistent incremental environment**: each entry is
parsed → resolved → typechecked → evaluated **exactly once** against accumulated
session state (symbols, types, declarations, runtime values).  Agent calls fire
exactly once and are never replayed, because each entry executes ONLY its own
statements — references to earlier bindings read stored runtime ``Value``s.

The driver reproduces ``WorkflowRuntime.run``'s pipeline incrementally, reusing
the shared host-environment assembly, input conversion, and exception-mapping
helpers from :mod:`agm.agl.runtime.runtime` (no duplication).  Promotion into
the session is **atomic**: a runtime raise discards ALL of the entry's in-session
effects — both new ``let``/``var`` bindings and any ``set`` mutation of a PRIOR
session binding is rolled back (the prior binding's value is snapshotted before
evaluation and restored on error).  The only effects that survive a failed entry
are genuinely EXTERNAL ones already issued during evaluation (an agent call or an
``exec`` shell command), which are inherently irreversible.

This module is intentionally UI-free — it returns plain ``EntryResult`` data;
rendering, meta-commands, and the prompt_toolkit console are later milestones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from agm.agl.diagnostics import AglError, Diagnostic

if TYPE_CHECKING:
    from pathlib import Path

    from agm.agl.eval.scope import Scope
    from agm.agl.eval.values import Value
    from agm.agl.runtime.agents import AgentFn
    from agm.agl.runtime.codec import OutputCodec
    from agm.agl.runtime.render import RendererFn
    from agm.agl.runtime.runtime import HostEnvironment, RunError
    from agm.agl.runtime.trace import TraceStore
    from agm.agl.scope.symbols import ResolvedProgram, ScopeNode
    from agm.agl.syntax.nodes import Program
    from agm.agl.typecheck.env import CheckedProgram, TypeEnvironment
    from agm.agl.typecheck.types import Type


EntryKind = Literal["expression", "binding", "declaration", "statement"]

# Layout-only token types that carry no statement to evaluate.
_TRIVIAL_TOKENS: frozenset[str] = frozenset({"_NEWLINE", "_INDENT", "_DEDENT"})


def _set_targets_in_program(program: "Program") -> frozenset[str]:
    """Return the set of variable names targeted by ``set`` statements in *program*.

    Recursively walks all statements in the program, including nested bodies
    inside ``do``/``if``/``case``/``try`` blocks.  Used by
    ``_evaluate_and_promote`` to determine which session bindings need to be
    snapshotted before evaluation (only those names can be mutated in-place by
    a ``set``).
    """
    from agm.agl.syntax.nodes import (
        CaseStmt,
        DoUntil,
        IfStmt,
        SetStmt,
        TryCatch,
    )
    from agm.agl.syntax.nodes import Stmt as StmtType

    targets: set[str] = set()

    def _walk(stmts: "tuple[StmtType, ...]") -> None:
        for stmt in stmts:
            if isinstance(stmt, SetStmt):
                targets.add(stmt.target)
            elif isinstance(stmt, DoUntil):
                _walk(stmt.body)
            elif isinstance(stmt, IfStmt):
                for if_branch in stmt.branches:
                    _walk(if_branch.body)
            elif isinstance(stmt, CaseStmt):
                for case_branch in stmt.branches:
                    _walk(case_branch.body)
            elif isinstance(stmt, TryCatch):
                _walk(stmt.body)
                for handler in stmt.handlers:
                    _walk(handler.body)

    _walk(program.body)
    return frozenset(targets)


def has_runnable_statements(text: str) -> bool:
    """Return ``True`` when *text* contains at least one statement to evaluate.

    Blank, whitespace-only, and comment-only entries (AgL comments run from a
    ``#`` to end of line) have nothing to run.  The check tokenizes *text* with
    the real AgL lexer and looks for any non-trivial token — the lexer skips
    whitespace and comments entirely and emits no tokens for blank/comment-only
    input, while synthetic layout tokens (``_NEWLINE`` / ``_INDENT`` /
    ``_DEDENT``) carry no statement, so they are ignored.  Any lexer error (a
    half-typed entry never reaches here, but be defensive) is treated as
    *runnable* so the entry flows on to ``eval_entry`` and surfaces a real
    diagnostic rather than being silently dropped.

    Shared by the interactive console (blank-line handling) and ``load_file``
    (an empty / comment-only file loads as a benign no-op rather than a parse
    error).
    """
    from agm.agl.lexer import tokenize

    try:
        return any(token.type not in _TRIVIAL_TOKENS for token in tokenize(text))
    except Exception:
        return True


# ---------------------------------------------------------------------------
# EntryResult — pure data describing the outcome of one entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EntryResult:
    """Outcome of evaluating one REPL entry (pure data, no styled strings).

    ``kind``
        Classified by the entry's LAST statement: ``ExprStmt`` → ``"expression"``
        (``value``/``value_type`` set); ``let``/``var`` → ``"binding"``
        (``name``/``value_type``/``value``); ``record``/``enum``/``type``/
        ``input`` → ``"declaration"``; everything else → ``"statement"``.
    ``name``
        The bound/declared name, when meaningful (binding / declaration).
    ``value``
        The echoed runtime value (expression value or new binding value); ``None``
        for declarations, statements, ``check_only`` runs, and failures.
    ``value_type``
        The static type of the echoed value; ``None`` when not applicable.
    ``diagnostics``
        Pre-execution error diagnostics (parse/scope/typecheck/contract/unset
        input).  Empty on success.
    ``warnings``
        Advisory warnings from the type checker (e.g. non-exhaustive ``case``),
        surfaced on every non-parse/scope path.
    ``error``
        The uncaught AgL exception mapped to a ``RunError`` when the entry raised
        during evaluation; ``None`` otherwise.
    ``ok``
        ``True`` iff there are no error diagnostics AND no runtime error.
    ``trace_path``
        Path of the JSONL trace file the entry's records were appended to, or
        ``None`` when tracing is disabled (no ``--log-file``) or for a
        ``check_only`` (dry-run) entry, which writes no trace.
    """

    kind: EntryKind
    name: str | None
    value: "Value | None"
    value_type: "Type | None"
    diagnostics: list[Diagnostic]
    warnings: list[Diagnostic]
    error: "RunError | None"
    ok: bool
    trace_path: "Path | None" = None


# ---------------------------------------------------------------------------
# ReplSession — the persistent incremental driver
# ---------------------------------------------------------------------------


class ReplSession:
    """Persistent incremental AgL evaluation session (UI-free core).

    Constructor parameters mirror ``WorkflowRuntime`` so a host can wire the same
    agent backing.  Registration (``register_agent``/``register_codec``/
    ``register_renderer``) is delegated to an internal ``WorkflowRuntime`` so the
    reserved-name / duplicate validation and host-environment assembly are shared
    rather than duplicated.

    Each entry is promoted into the session **atomically**: if evaluation raises,
    the entry has NO in-session effect — new ``let``/``var`` bindings are discarded
    AND any ``set`` mutation of a prior session binding is rolled back to its
    pre-entry value.  Only external side effects already issued during evaluation
    (agent calls, ``exec`` shell commands) are irreversible.
    """

    def __init__(
        self,
        *,
        default_loop_limit: int = 5,
        default_strict_json: bool = False,
        default_agent: "AgentFn | None" = None,
        shell_exec_timeout: float | None = None,
        trace_path: "Path | None" = None,
    ) -> None:
        from agm.agl.eval.scope import Scope
        from agm.agl.runtime.runtime import WorkflowRuntime
        from agm.agl.scope.symbols import ScopeNode
        from agm.agl.typecheck.env import TypeEnvironment

        self._default_loop_limit = default_loop_limit
        self._default_strict_json = default_strict_json
        self._shell_exec_timeout = shell_exec_timeout
        # Trace destination: when set, each evaluated entry opens a fresh
        # ``TraceStore`` (its own ``run_id``) appending JSONL records to this one
        # file.  ``check_only`` entries write nothing (mirroring ``agm exec``).
        # The COMMAND validates/creates the path up front; the session assumes it
        # is writable but the no-op store tolerates failure (it disables itself).
        self._trace_path = trace_path

        # Internal runtime owns the registrations + host-environment assembly.
        self._runtime = WorkflowRuntime(
            default_loop_limit=default_loop_limit,
            default_strict_json=default_strict_json,
            default_agent=default_agent,
            shell_exec_timeout=shell_exec_timeout,
        )
        self._has_default_agent = default_agent is not None

        # Persistent session environment.
        self._session_scope: ScopeNode = ScopeNode(node_id=-1, parent=None)
        self._type_env: TypeEnvironment = TypeEnvironment()
        self._value_scope: Scope = Scope(parent=None)
        self._next_node_id: int = 0
        # Declared inputs: name → (declared type, current value or None if unset).
        self._declared_inputs: dict[str, tuple[Type, Value | None]] = {}
        # Pending pre-seeded input values (``--input``/``preset_input``) that
        # name an input not yet declared: applied on the input's later
        # declaration (in ``_promote``).  Cleared by ``reset``.
        self._pending_inputs: dict[str, str] = {}
        # Source log of successfully-promoted entries (for dump_source / :save).
        self._source_log: list[str] = []
        # Agents declared by SOURCE ``agent X`` statements in prior promoted
        # entries.  In the REPL, host registration both declares AND backs an
        # agent, so the ambient agent set passed to ``resolve`` is the union of
        # the host-registered names and these cross-entry source declarations.
        # Declarations from a failed/rolled-back entry never land here (merged
        # only on successful promotion).
        self._declared_agents: set[str] = set()

    # ------------------------------------------------------------------
    # Registration (delegated to the internal runtime — shared validation)
    # ------------------------------------------------------------------

    def register_agent(self, name: str, fn: "AgentFn") -> None:
        """Register a named agent (shares ``WorkflowRuntime`` validation)."""
        self._runtime.register_agent(name, fn)

    def register_codec(self, codec: "OutputCodec") -> None:
        """Register a custom output codec (shares ``WorkflowRuntime`` validation)."""
        self._runtime.register_codec(codec)

    def register_renderer(
        self,
        name: str,
        fn: "RendererFn",
        *,
        supported_types: "frozenset[str] | None" = None,
    ) -> None:
        """Register a custom renderer (shares ``WorkflowRuntime`` validation)."""
        self._runtime.register_renderer(name, fn, supported_types=supported_types)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def eval_entry(self, text: str, *, check_only: bool = False) -> EntryResult:
        """Parse → resolve → check → (eval) one entry against the session.

        Promotes the entry's new bindings/declarations into the session ONLY on
        full success (atomic).  ``check_only`` runs the full static pipeline but
        never evaluates, never promotes, and never advances the node-id counter.
        """
        from agm.agl.lexer import tab_warning_collector
        from agm.agl.parser import AglSyntaxError, parse_program_seeded
        from agm.agl.scope import AglScopeError, resolve
        from agm.agl.typecheck import AglTypeError, check

        host_env = self._runtime.host_environment()

        # TAB advisories come from the parse's single lex pass (no separate TAB
        # scan).  The collector is populated even on a failed parse, so they
        # surface on EVERY return path (mirroring ``WorkflowRuntime.prepare``).
        # [1] Parse (seeded so node ids stay globally unique across entries).
        with tab_warning_collector() as tab_sink:
            try:
                program, next_start_id = parse_program_seeded(
                    text, start_id=self._next_node_id
                )
            except AglSyntaxError as exc:
                return self._fail([exc.to_diagnostic()], list(tab_sink))
        tab_warnings: list[Diagnostic] = list(tab_sink)

        # [2] Resolve against the session scope (refs fall through; new decls
        # shadow).  resolve does NOT mutate the parent scope.  Host-registered
        # agents and prior cross-entry ``agent`` declarations are ambient, so a
        # call to them needs no in-entry declaration.
        try:
            resolved = resolve(
                program,
                parent_scope=self._session_scope,
                ambient_agents=self._ambient_agents(host_env),
            )
        except AglScopeError as exc:
            return self._fail([exc.to_diagnostic()], list(tab_warnings))

        # [3] Type-check seeded with the session type env (check COPIES the seed
        # into a fresh env, so self._type_env is not mutated here).
        try:
            checked = check(resolved, host_env.capabilities, seed_env=self._type_env)
        except AglTypeError as exc:
            return self._fail([exc.to_diagnostic()], list(tab_warnings))

        # Surface TAB advisories ahead of the scope and type-checker warnings on
        # every remaining path (typecheck-clean, eval success, or runtime raise).
        # Scope warnings (e.g. an agent declared but never called) are routed the
        # same way the checker's warnings are.
        warnings: list[Diagnostic] = [
            *tab_warnings,
            *resolved.warnings,
            *checked.warnings,
        ]

        # [4] Unset-input guard (conservative, syntactic; before any eval).
        unset_diag = self._check_unset_inputs(resolved)
        if unset_diag is not None:
            return self._fail([unset_diag], warnings)

        # [5] Materialize output contracts for this entry.
        from agm.agl.runtime.contract import materialize_contract

        contracts: dict[int, object] = {}
        contract_errors: list[Diagnostic] = []
        for node_id, spec in checked.contract_specs.items():
            try:
                contracts[node_id] = materialize_contract(spec, host_env.codecs)
            except ValueError as exc:
                contract_errors.append(Diagnostic(message=f"Contract error: {exc}", line=1))
        if contract_errors:
            return self._fail(contract_errors, warnings)

        # [6] check_only: type-only dry run — no eval, no promotion.
        if check_only:
            return self._build_check_only_result(program, checked, warnings)

        # [7] Evaluate ONLY this entry's statements in a fresh child value scope.
        return self._evaluate_and_promote(
            text=text,
            program=program,
            checked=checked,
            contracts=contracts,
            host_env=host_env,
            warnings=warnings,
            next_start_id=next_start_id,
        )

    # ------------------------------------------------------------------
    # eval_entry helpers
    # ------------------------------------------------------------------

    def _ambient_agents(self, host_env: "HostEnvironment") -> frozenset[str]:
        """Agent names valid WITHOUT an in-entry ``agent`` declaration.

        In the REPL, host registration both declares and backs an agent, so the
        authoritative set of host-registered names (``capabilities.agent_names``,
        which excludes the ``ask``/``exec`` built-ins) is ambient, unioned with
        agents declared by ``agent X`` statements in prior promoted entries.
        """
        return host_env.capabilities.agent_names | self._declared_agents

    def _fail(
        self, diagnostics: list[Diagnostic], warnings: list[Diagnostic]
    ) -> EntryResult:
        """Build a clean pre-execution failure result (no promotion)."""
        return EntryResult(
            kind="statement",
            name=None,
            value=None,
            value_type=None,
            diagnostics=diagnostics,
            warnings=warnings,
            error=None,
            ok=False,
        )

    def _check_unset_inputs(self, resolved: "ResolvedProgram") -> Diagnostic | None:
        """Return a diagnostic if the entry references an unset declared input.

        Conservative syntactic check: scans every resolved ``BindingRef``; if one
        is an ``input`` binding naming a declared input whose value is still
        ``None``, the entry is rejected before evaluation (no agent calls).  This
        may flag an input referenced only in a not-taken branch — an acceptable
        v1 limitation.
        """
        from agm.agl.scope.symbols import BinderKind

        for ref in resolved.resolution.values():
            if ref.kind is not BinderKind.param_binding:
                continue
            entry = self._declared_inputs.get(ref.name)
            if entry is not None and entry[1] is None:
                return Diagnostic(
                    message=(
                        f"Input {ref.name!r} is declared but has no value; "
                        f"set it first with :set {ref.name}=<value>."
                    ),
                    line=ref.decl_span.start_line,
                )
        return None

    def _build_check_only_result(
        self,
        program: "Program",
        checked: "CheckedProgram",
        warnings: list[Diagnostic],
    ) -> EntryResult:
        """Build the EntryResult for a ``check_only`` (type-only) run.

        No value, no evaluation, no promotion, no trace.  The value_type for an
        expression entry is the checked node type of the expression; for a binding
        it is the declared binding type.
        """
        kind, name = self._classify(program)
        return EntryResult(
            kind=kind,
            name=name,
            value=None,
            value_type=self._value_type_of_last(program, checked),
            diagnostics=[],
            warnings=warnings,
            error=None,
            ok=True,
        )

    def _evaluate_and_promote(
        self,
        *,
        text: str,
        program: "Program",
        checked: "CheckedProgram",
        contracts: dict[int, object],
        host_env: "HostEnvironment",
        warnings: list[Diagnostic],
        next_start_id: int,
    ) -> EntryResult:
        """Execute the entry in a child scope; promote on success, discard on error."""
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.scope import Scope
        from agm.agl.repl.agents import AgentCancelled
        from agm.agl.repl.echo_interpreter import EchoInterpreter
        from agm.agl.runtime.contract import OutputContract
        from agm.agl.runtime.runtime import exception_value_to_run_error
        from agm.agl.runtime.trace import TraceStore
        from agm.agl.syntax.nodes import ExprStmt

        typed_contracts: dict[int, OutputContract] = {
            nid: c for nid, c in contracts.items() if isinstance(c, OutputContract)
        }

        # Fresh child scope: prior session values resolve via the parent chain,
        # new bindings land in the child and are promoted only on success.
        child_scope = Scope(parent=self._value_scope)

        # Atomicity snapshot: a ``set`` to a PRIOR session binding mutates that
        # binding's ``.value`` in place in the persistent value scope (the child
        # only holds NEW let/var bindings).  New keys never land here and ``set``
        # never adds/removes keys (it only updates an existing binding via
        # ``Scope.set_value``), so a shallow value snapshot of ONLY the binding
        # names targeted by ``set`` statements in this entry is a complete,
        # correct rollback point — Value objects are immutable, so storing the
        # reference suffices.  On a runtime raise we restore each binding's
        # ``.value`` from this snapshot.
        #
        # Optimisation: entries with no ``set`` targeting a prior session binding
        # need no snapshot at all (new let/var bindings live in the child scope
        # and are simply discarded on abort).  We collect targeted names
        # statically from the checked program before evaluation.
        set_targets = _set_targets_in_program(program)
        value_snapshot: dict[str, Value] = {
            name: binding.value
            for name, binding in self._value_scope.bindings.items()
            if name in set_targets
        }

        # One trace run per entry: a fresh ``TraceStore`` (own ``run_id``)
        # appends to the shared file, bracketed by ``run_start``/``run_end``.
        # ``path=None`` (no ``--log-file``) makes every write a silent no-op.
        trace = TraceStore(path=self._trace_path)
        trace.run_start()

        interp = EchoInterpreter(
            checked=checked,
            registry=host_env.registry,
            contracts=typed_contracts,
            type_env=checked.type_env,
            renderers=host_env.renderers,
            loop_limit=self._default_loop_limit,
            strict_json=self._default_strict_json,
            source=text,
            shell_exec_timeout=self._shell_exec_timeout,
            trace=trace,
        )
        # Echo the value of a trailing bare expression (captured during exec).
        if program.body and isinstance(program.body[-1], ExprStmt):
            interp.echo_node_id = program.body[-1].node_id

        try:
            interp.execute(child_scope)
        except AglRaise as exc:
            error = exception_value_to_run_error(exc.exc, span=exc.span)
            trace.exception(
                type_name=error.type_name,
                message=str(error.fields.get("message", "")),
                trace_id=str(error.fields.get("trace_id", "")),
                span=exc.span,
            )
            return self._abort(program, warnings, trace, value_snapshot, error=error)
        except (AgentCancelled, KeyboardInterrupt):
            # A declined confirmation or a Ctrl-C during a live agent call aborts
            # the entry atomically.  The cancellation is a host signal, not an
            # in-language raise, so it surfaces as a diagnostic rather than a
            # mapped AgL exception.
            return self._abort(
                program,
                warnings,
                trace,
                value_snapshot,
                diagnostics=[
                    Diagnostic(message="Agent call cancelled — entry aborted.", line=1)
                ],
            )

        trace.run_end(ok=True)
        captured: Value | None = interp.captured

        # Success — promote atomically, then compute the echo data.
        self._promote(
            text=text,
            program=program,
            checked=checked,
            child_scope=child_scope,
            next_start_id=next_start_id,
        )
        kind, name = self._classify(program)
        value, value_type = self._echo_data(program, checked, captured)
        return EntryResult(
            kind=kind,
            name=name,
            value=value,
            value_type=value_type,
            diagnostics=[],
            warnings=warnings,
            error=None,
            ok=True,
            trace_path=self._trace_path,
        )

    def _abort(
        self,
        program: "Program",
        warnings: list[Diagnostic],
        trace: "TraceStore",
        value_snapshot: dict[str, "Value"],
        *,
        error: "RunError | None" = None,
        diagnostics: list[Diagnostic] | None = None,
    ) -> EntryResult:
        """End the trace, roll back, and build the failed result for an abort.

        Shared by the ``AglRaise`` and cancellation arms of
        ``_evaluate_and_promote``: both end the trace run, restore the value
        scope, and return a failed result differing only in whether the failure
        carries an ``error`` (a mapped AgL raise) or ``diagnostics`` (a
        host-signalled cancellation).
        """
        trace.run_end(ok=False)
        self._rollback(value_snapshot)
        kind, name = self._classify(program)
        return EntryResult(
            kind=kind,
            name=name,
            value=None,
            value_type=None,
            diagnostics=diagnostics if diagnostics is not None else [],
            warnings=warnings,
            error=error,
            ok=False,
            trace_path=self._trace_path,
        )

    def _rollback(self, value_snapshot: dict[str, "Value"]) -> None:
        """Roll the persistent value scope back to *value_snapshot* (atomic abort).

        Shared by the ``AglRaise`` and cancellation paths.  Discarding the entry's
        child scope drops new ``let``/``var`` bindings; restoring each session
        binding's ``.value`` undoes any in-place ``set`` mutation of a prior
        binding.  The snapshot contains ONLY the names that could have been mutated
        (those targeted by ``set`` statements in the entry), and all of them must
        still be present in the session frame (``set`` only updates existing
        bindings, never adds or removes keys).
        """
        assert value_snapshot.keys() <= self._value_scope.bindings.keys()
        for bname, old_value in value_snapshot.items():
            self._value_scope.bindings[bname].value = old_value

    def _promote(
        self,
        *,
        text: str,
        program: "Program",
        checked: "CheckedProgram",
        child_scope: "Scope",
        next_start_id: int,
    ) -> None:
        """Merge the entry's new state into the persistent session (atomic)."""
        from agm.agl.syntax.nodes import ParamDecl

        # Symbols: merge the entry root scope's bindings (overwrite/shadow).
        entry_root = checked.resolved.root_scope
        for bname, ref in entry_root.bindings.items():
            self._session_scope.bindings[bname] = ref

        # Agent declarations: a source ``agent X`` in this entry becomes ambient
        # for later entries (merged only on this successful promotion, so a
        # rolled-back entry's declarations never persist).
        self._declared_agents.update(checked.resolved.declared_agents)

        # Types + binding types: union the entry's checked env into the session.
        self._type_env.seed_from(checked.type_env)

        # Runtime values: copy the child scope's top frame into the session scope.
        for vname, binding in child_scope.bindings.items():
            self._value_scope.bindings[vname] = binding

        # Declared inputs: register any ParamDecl (value None until :set).
        for stmt in program.body:
            if isinstance(stmt, ParamDecl):
                input_type = checked.type_env.get_binding_type(stmt.node_id)
                assert input_type is not None
                # A re-declared input keeps no stale value (shadows fresh):
                # remove any previously-set value from the value scope so that
                # _declared_inputs and _value_scope agree — both report unset.
                self._value_scope.bindings.pop(stmt.name, None)
                self._declared_inputs[stmt.name] = (input_type, None)
                # Apply a pending pre-seeded value (``--input``/``preset_input``)
                # now that the input is declared.  A conversion failure leaves the
                # input unset (the unset-input guard surfaces a clean error if it
                # is later referenced) — pre-seeding must never crash promotion.
                self._apply_pending_input(stmt.name)

        self._source_log.append(text)
        self._next_node_id = next_start_id

    def _classify(self, program: "Program") -> tuple[EntryKind, str | None]:
        """Classify the entry by its last statement; return (kind, name)."""
        from agm.agl.syntax.nodes import (
            EnumDef,
            ExprStmt,
            LetDecl,
            ParamDecl,
            ProgramDecl,
            RecordDef,
            TypeAlias,
            VarDecl,
        )

        # A parsed program always has at least one statement (empty/comment-only
        # input fails parsing earlier).
        last = program.body[-1]
        if isinstance(last, ExprStmt):
            return "expression", None
        if isinstance(last, (LetDecl, VarDecl)):
            return "binding", last.name
        if isinstance(last, (RecordDef, EnumDef, TypeAlias, ParamDecl, ProgramDecl)):
            return "declaration", last.name
        return "statement", None

    def _echo_data(
        self, program: "Program", checked: "CheckedProgram", captured: "Value | None"
    ) -> tuple["Value | None", "Type | None"]:
        """Compute the echoed (value, value_type) from the promoted state.

        *captured* is the value of a trailing bare expression recorded during
        execution (``None`` when the last statement is not an ``ExprStmt``).
        """
        from agm.agl.syntax.nodes import ExprStmt, LetDecl, VarDecl

        # A parsed program always has at least one statement.
        last = program.body[-1]
        value_type = self._value_type_of_last(program, checked)
        if isinstance(last, ExprStmt):
            return captured, value_type
        if isinstance(last, (LetDecl, VarDecl)):
            binding = self._value_scope.lookup(last.name)
            value = binding.value if binding is not None else None
            return value, value_type
        return None, None

    def _value_type_of_last(
        self, program: "Program", checked: "CheckedProgram"
    ) -> "Type | None":
        """Static type carried by the entry's last statement, or ``None``.

        The checked type of the expression for a bare-expression entry, the
        declared binding type for a ``let``/``var``, ``None`` otherwise.  Shared
        by the check-only result builder and the success echo so the two agree
        on how an entry's type is derived.
        """
        from agm.agl.syntax.nodes import ExprStmt, LetDecl, VarDecl

        # A parsed program always has at least one statement (empty/comment-only
        # input fails parsing earlier).
        last = program.body[-1]
        if isinstance(last, ExprStmt):
            return checked.node_types.get(last.expr.node_id)
        if isinstance(last, (LetDecl, VarDecl)):
            return checked.type_env.get_binding_type(last.node_id)
        return None

    # ------------------------------------------------------------------
    # type_of — type without evaluation
    # ------------------------------------------------------------------

    def type_of(self, text: str) -> str:
        """Return the canonical display type of *text* as an expression entry.

        Resolves against the session scope and checks against the session type
        env WITHOUT evaluating, promoting, or advancing the node-id counter.
        Raises the underlying ``AglSyntaxError``/``AglScopeError``/``AglTypeError``
        on failure, or ``AglError`` if *text* is not a single expression.
        """
        from agm.agl.parser import parse_program_seeded
        from agm.agl.scope import resolve
        from agm.agl.syntax.nodes import ExprStmt
        from agm.agl.typecheck import check

        host_env = self._runtime.host_environment()
        # Throwaway ids: type_of never promotes and never advances the session
        # counter, so seeding at ``_next_node_id`` is safe — all promoted ids are
        # strictly below it, making this parse's ids disjoint from the session's.
        program, _ = parse_program_seeded(text, start_id=self._next_node_id)
        if len(program.body) != 1 or not isinstance(program.body[0], ExprStmt):
            raise AglError(
                "':type' expects a single expression, "
                "not a binding, declaration, or statement."
            )
        expr_stmt = program.body[0]
        resolved = resolve(
            program,
            parent_scope=self._session_scope,
            ambient_agents=self._ambient_agents(host_env),
        )
        checked = check(resolved, host_env.capabilities, seed_env=self._type_env)
        typ = checked.node_types.get(expr_stmt.expr.node_id)
        assert typ is not None
        return repr(typ)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def bindings(self) -> list[tuple[str, "Type", "Value"]]:
        """Return promoted user bindings as (name, declared type, current value).

        Excludes declared-but-unset inputs (they have no value).  Set inputs are
        included (their value lives in the value scope).
        """
        result: list[tuple[str, Type, Value]] = []
        for name, ref in self._session_scope.bindings.items():
            binding = self._value_scope.lookup(name)
            if binding is None:
                # A declared-but-unset input is in the symbol scope but has no
                # runtime value yet; skip it (it surfaces via ``inputs()``).
                continue
            typ = self._type_env.get_binding_type(ref.decl_node_id)
            # Every promoted let/var/input binding has a recorded type.
            assert typ is not None
            result.append((name, typ, binding.value))
        return result

    def agents(self) -> list[str]:
        """Return the names of available agents.

        Registered named agents plus ``"ask"`` when a default agent is
        configured.
        """
        host_env = self._runtime.host_environment()
        names = sorted(host_env.registry.agent_names)
        if self._has_default_agent:
            names.append("ask")
        return names

    def inputs(self) -> list[tuple[str, "Type", "Value | None"]]:
        """Return declared inputs as (name, type, current value or None)."""
        return [
            (name, typ, value) for name, (typ, value) in self._declared_inputs.items()
        ]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def set_input(self, name: str, raw: str) -> None:
        """Supply a host value for a declared *input* (the ``:set`` flow).

        Raises ``AglError`` if *name* is not a declared input; converts *raw* to
        the declared type via the shared input-conversion helper and binds the
        value into both the value scope and the declared-inputs table.  Raises
        ``AglError`` on conversion failure.
        """
        entry = self._declared_inputs.get(name)
        if entry is None:
            raise AglError(
                f"{name!r} is not a declared input; declare it with "
                f"'input {name}: <type>' first."
            )
        declared_type, _ = entry
        try:
            self._bind_input(name, raw, declared_type)
        except ValueError as exc:
            raise AglError(str(exc)) from exc

    def _bind_input(self, name: str, raw: str, declared_type: "Type") -> None:
        """Convert *raw* to *declared_type* and bind it as the input's value.

        Shared by ``set_input`` (the ``:set`` flow) and ``_apply_pending_input``
        (the pre-seed flow).  Raises ``ValueError`` on conversion failure; the
        binding tables are left untouched in that case so callers decide how to
        surface or swallow the error.
        """
        from agm.agl.runtime.runtime import convert_input

        value = convert_input(name, raw, declared_type)

        ref = self._session_scope.bindings.get(name)
        assert ref is not None  # a declared input is always a promoted binding
        self._value_scope.bindings.pop(name, None)
        self._value_scope.define(name, value, mutable=False, decl_span=ref.decl_span)
        self._declared_inputs[name] = (declared_type, value)

    def preset_input(self, name: str, raw: str) -> None:
        """Pre-seed a host input value (the ``--input KEY=VALUE`` launch flow).

        If *name* is ALREADY declared, the value is converted and bound
        immediately (reusing the ``:set`` binding path); a bad value leaves the
        input unset.  Otherwise the raw value is stored pending and applied on the
        input's later declaration (in ``_promote``).  Unlike :meth:`set_input`,
        a conversion failure here is swallowed rather than raised — pre-seeding
        is a best-effort launch convenience, and an unset input surfaces a clean
        error only if it is actually referenced.
        """
        if name in self._declared_inputs:
            declared_type, _ = self._declared_inputs[name]
            try:
                self._bind_input(name, raw, declared_type)
            except ValueError:
                # Bad pre-seed value: leave the input unset.
                pass
            return
        self._pending_inputs[name] = raw

    def _apply_pending_input(self, name: str) -> None:
        """Apply (and consume) a pending pre-seeded value for a just-declared input.

        Called from ``_promote`` when an ``input`` declaration is registered.  A
        conversion failure leaves the input unset (the value is consumed either
        way so it is not retried on re-declaration).
        """
        raw = self._pending_inputs.pop(name, None)
        if raw is None:
            return
        declared_type, _ = self._declared_inputs[name]
        try:
            self._bind_input(name, raw, declared_type)
        except ValueError:
            # Bad pre-seed value: leave the input unset (guard surfaces the error
            # only if the input is later referenced).
            pass

    def reset(self) -> None:
        """Clear ALL session state (symbols, types, values, inputs, source, ids)."""
        from agm.agl.eval.scope import Scope
        from agm.agl.scope.symbols import ScopeNode
        from agm.agl.typecheck.env import TypeEnvironment

        self._session_scope = ScopeNode(node_id=-1, parent=None)
        self._type_env = TypeEnvironment()
        self._value_scope = Scope(parent=None)
        self._next_node_id = 0
        self._declared_inputs = {}
        self._pending_inputs = {}
        self._source_log = []
        self._declared_agents = set()

    def load_file(self, path: "Path") -> list[EntryResult]:
        """Evaluate the contents of *path* INCREMENTALLY, one statement per entry.

        Each top-level statement is fed to :meth:`eval_entry` in order, exactly as
        if the user had typed it at the prompt.  This makes redefinition/shadowing
        work on load (within a single entry it would be a duplicate-declaration
        error) so a ``:save`` transcript reliably round-trips through ``:load``.

        The load halts at the FIRST non-``ok`` result (like running a script);
        the returned list holds the results collected so far, including the
        failing one.  Statements that already succeeded remain promoted.

        A syntax error in the file yields a single failed ``EntryResult`` carrying
        the parse diagnostic.  An empty or comment-only file has no statements to
        run and yields an empty list (a benign no-op).
        """
        from agm.agl._text import normalize_newlines
        from agm.agl.parser import AglSyntaxError, parse_program
        from agm.core.fs import read_text

        # Normalize newlines with the SAME helper the lexer/interpreter use so the
        # statement-span char offsets align with the text we slice below.
        normalized = normalize_newlines(read_text(path))

        # A blank / comment-only file has nothing to run — load it as a no-op
        # rather than surfacing the parser's "Unexpected end of input" error.
        if not has_runnable_statements(normalized):
            return []

        # Parse the whole file ONCE only to find top-level statement boundaries;
        # this parse is never promoted (each slice is re-parsed by eval_entry with
        # the session's continuing node-id counter).  start_id=0 is fine here.
        try:
            program = parse_program(normalized)
        except AglSyntaxError as exc:
            return [self._fail([exc.to_diagnostic()], [])]

        results: list[EntryResult] = []
        for stmt in program.body:
            slice_text = normalized[stmt.span.start_offset : stmt.span.end_offset]
            result = self.eval_entry(slice_text)
            results.append(result)
            if not result.ok:
                break  # halt on the first failing statement, like a script
        return results

    def dump_source(self) -> str:
        """Return the accumulated successfully-promoted entry sources (newline-joined)."""
        return "\n".join(self._source_log)
