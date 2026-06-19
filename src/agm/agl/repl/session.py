"""UI-free incremental session core for the AgL REPL (``ReplSession``).

``ReplSession`` keeps a **persistent incremental environment**: each entry is
parsed → resolved → typechecked → evaluated **exactly once** against accumulated
session state (symbols, types, declarations, runtime values).  Agent calls fire
exactly once and are never replayed, because each entry executes ONLY its own
statements — references to earlier bindings read stored runtime ``Value``s.

The driver reproduces ``WorkflowRuntime.run``'s pipeline incrementally, reusing
the shared host-environment assembly, param conversion, and exception-mapping
helpers from :mod:`agm.agl.runtime.runtime` (no duplication).  Promotion into
the session is **atomic**: a runtime raise discards ALL of the entry's in-session
effects — both new ``let``/``var`` bindings and any ``:=`` mutation of a PRIOR
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

from agm.agl.diagnostics import AglError, Diagnostic, diagnostic_from_span

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from agm.agl.eval.scope import Scope
    from agm.agl.eval.values import Value
    from agm.agl.runtime.agents import AgentFn
    from agm.agl.runtime.codec import OutputCodec
    from agm.agl.runtime.runtime import HostEnvironment, RunError
    from agm.agl.runtime.trace import TraceStore
    from agm.agl.scope.symbols import ConstructorRef, ScopeNode
    from agm.agl.syntax.nodes import Program
    from agm.agl.syntax.spans import SourceSpan
    from agm.agl.typecheck.env import CheckedProgram, TypeEnvironment
    from agm.agl.typecheck.types import Type


EntryKind = Literal["expression", "binding", "declaration", "statement"]

# Layout-only token types that carry no statement to evaluate.
_TRIVIAL_TOKENS: frozenset[str] = frozenset({"_NEWLINE", "_INDENT", "_DEDENT"})


def _assign_targets_in_program(program: "Program") -> frozenset[str]:
    """Return the set of variable names targeted by ``:=`` statements in *program*.

    Recursively walks all items in the program block, including nested bodies
    inside ``do``/``if``/``case``/``try`` blocks.  Used by
    ``_evaluate_and_promote`` to determine which session bindings need to be
    snapshotted before evaluation (only those names can be mutated in-place by
    a ``:=``).
    """
    from agm.agl.syntax.nodes import (
        AssignStmt,
        Block,
        Case,
        Do,
        If,
        Item,
        Try,
        assign_target_root_name,
    )

    targets: set[str] = set()

    def _walk_item(item: Item) -> None:
        if isinstance(item, AssignStmt):
            target_name = assign_target_root_name(item.target)
            if target_name is not None:
                targets.add(target_name)
        elif isinstance(item, Block):
            for sub in item.items:
                _walk_item(sub)
        elif isinstance(item, Do):
            _walk_item(item.body)
            _walk_item(item.condition)
        elif isinstance(item, If):
            for if_branch in item.branches:
                _walk_item(if_branch.body)
        elif isinstance(item, Case):
            for case_branch in item.branches:
                _walk_item(case_branch.body)
        elif isinstance(item, Try):
            _walk_item(item.body)
            for handler in item.handlers:
                _walk_item(handler.body)

    for item in program.body.items:
        _walk_item(item)
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
    except Exception:  # defensive: lexer errors are treated as runnable
        return True


# ---------------------------------------------------------------------------
# EntryResult — pure data describing the outcome of one entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EntryResult:
    """Outcome of evaluating one REPL entry (pure data, no styled strings).

    ``kind``
        Classified by the entry's LAST item: a bare ``Expr`` → ``"expression"``
        (``value``/``value_type`` set); ``let``/``var`` → ``"binding"``
        (``name``/``value_type``/``value``); ``record``/``enum``/``type``/
        ``param``/``def``/``agent`` → ``"declaration"``; ``:=`` or side-
        effecting expr (``print``, etc.) → ``"statement"``.
    ``name``
        The bound/declared name, when meaningful (binding / declaration).
    ``value``
        The echoed runtime value (expression value or new binding value); ``None``
        for declarations, statements, ``check_only`` runs, and failures.
    ``value_type``
        The static type of the echoed value; ``None`` when not applicable.
    ``diagnostics``
        Pre-execution error diagnostics (parse/scope/typecheck/contract/unset
        param).  Empty on success.
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
    agent backing.  Registration (``register_agent``/``register_codec``) is
    delegated to an internal ``WorkflowRuntime`` so the reserved-name / duplicate
    validation and host-environment assembly are shared rather than duplicated.

    Each entry is promoted into the session **atomically**: if evaluation raises,
    the entry has NO in-session effect — new ``let``/``var`` bindings are discarded
    AND any ``:=`` mutation of a prior session binding is rolled back to its
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
        params_config_loader: "Callable[[str], dict[str, object]] | None" = None,
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
        self._params_config_loader = params_config_loader

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
        # Ambient agents injected by the scope pass carry a synthetic decl_node_id
        # of -1 (no real AST declaration).  Pre-register AgentType() for this
        # sentinel so the checker can resolve their binding type when they appear
        # as call callees or in expressions.  This is seeded into every entry's
        # fresh TypeEnvironment via seed_from, so it is always available.
        self._type_env.set_binding_type(-1, self._make_agent_type())
        self._value_scope: Scope = Scope(parent=None)
        self._next_node_id: int = 0
        self._program_name: str | None = None
        self._active_config: dict[str, object] = {}
        self._declared_params: dict[str, Type] = {}
        # Source log of successfully-promoted entries (for dump_source / :save).
        self._source_log: list[str] = []
        # Agents declared by SOURCE ``agent X`` statements in prior promoted
        # entries.  In the REPL, host registration both declares AND backs an
        # agent, so the ambient agent set passed to ``resolve`` is the union of
        # the host-registered names and these cross-entry source declarations.
        # Declarations from a failed/rolled-back entry never land here (merged
        # only on successful promotion).
        self._declared_agents: set[str] = set()
        # Constructor candidates from prior promoted entries, keyed by constructor
        # name → ordered tuple of ConstructorRef.  Passed to resolve() as ambient
        # so that subsequent entries can reference constructors from prior entries.
        self._ambient_constructor_candidates: dict[str, tuple[ConstructorRef, ...]] = {}
        # Type names declared in prior promoted entries, for qualified constructor
        # access (``Owner.variant``) across REPL entries.
        self._ambient_type_names: frozenset[str] = frozenset()

    @staticmethod
    def _make_agent_type() -> "Type":
        """Return an ``AgentType`` instance (deferred import, used at init and reset)."""
        from agm.agl.typecheck.types import AgentType

        return AgentType()

    # ------------------------------------------------------------------
    # Registration (delegated to the internal runtime — shared validation)
    # ------------------------------------------------------------------

    def register_agent(self, name: str, fn: "AgentFn") -> None:
        """Register a named agent (shares ``WorkflowRuntime`` validation)."""
        self._runtime.register_agent(name, fn)

    def register_codec(self, codec: "OutputCodec") -> None:
        """Register a custom output codec (shares ``WorkflowRuntime`` validation)."""
        self._runtime.register_codec(codec)

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

        # [1b] Reject config pragmas: they are an exec/program feature and cannot
        # be applied to a live REPL session (session settings come from CLI flags
        # or config files, not source pragmas).  Check here — after parse, before
        # resolve — so no session state is mutated.
        pragma_diag = self._check_no_config_pragmas(program)
        if pragma_diag is not None:
            return self._fail([pragma_diag], tab_warnings)

        # [1c] REPL trailing-binder synthesis: in v2, a block ending in a
        # ``let``/``var`` is a static error (the binder needs a continuation
        # expression).  In the REPL, the continuation is the NEXT entry — so a
        # standalone ``let x = 1`` entry is semantically valid.  Synthesize a
        # ``UnitLit`` continuation appended to the pipeline program so the
        # checker accepts it; the original ``program`` is kept for classification,
        # echo, and promotion (which care about what the user actually typed).
        orig_program = program
        pipeline_program, next_start_id = self._repl_wrap_trailing_binder(
            program, next_start_id
        )

        # [2] Resolve against the session scope (refs fall through; new decls
        # shadow).  resolve does NOT mutate the parent scope.  Host-registered
        # agents and prior cross-entry ``agent`` declarations are ambient, so a
        # call to them needs no in-entry declaration.  Constructor candidates
        # and type names from prior entries are passed as ambient so that
        # subsequent entries can reference constructors declared earlier.
        try:
            resolved = resolve(
                pipeline_program,
                parent_scope=self._session_scope,
                ambient_agents=self._ambient_agents(host_env),
                ambient_constructor_candidates=self._ambient_constructor_candidates,
                ambient_type_names=self._ambient_type_names,
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

        # [4] check_only: type-only dry run — no eval, no promotion.
        if check_only:
            return self._build_check_only_result(orig_program, checked, warnings)

        pre_eval_result, param_values, entry_program_name, entry_active_config = (
            self._pre_eval_param_check(orig_program, checked, warnings)
        )
        if pre_eval_result is not None:
            return pre_eval_result

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

        # [7] Evaluate ONLY this entry's statements in a fresh child value scope.
        return self._evaluate_and_promote(
            text=text,
            orig_program=orig_program,
            checked=checked,
            contracts=contracts,
            host_env=host_env,
            warnings=warnings,
            next_start_id=next_start_id,
            param_values=param_values,
            entry_program_name=entry_program_name,
            entry_active_config=entry_active_config,
        )

    # ------------------------------------------------------------------
    # eval_entry helpers
    # ------------------------------------------------------------------

    def _repl_wrap_trailing_binder(
        self, program: "Program", next_start_id: int
    ) -> "tuple[Program, int]":
        """Append a synthetic ``UnitLit`` when the entry ends with a trailing binder.

        In v2, the type checker rejects a block whose last item is a ``let`` or
        ``var`` declaration (the binder needs a continuation expression).  In the
        REPL, the continuation is the NEXT entry — so standalone ``let x = 1``
        entries must be valid.

        When the last item is a ``LetDecl`` or ``VarDecl``, this helper builds a
        new ``Program`` (and ``Block``) with an appended synthetic ``UnitLit``
        (node id ``next_start_id``), making the block checker-acceptable.
        The synthetic node id is consumed, so ``next_start_id + 1`` is returned.

        The ORIGINAL program is preserved for classification, echo data, and
        promotion (they care about what the user typed, not the pipeline artifact).
        If the last item is NOT a trailing binder, the program is returned
        unchanged and ``next_start_id`` is not advanced.
        """
        from agm.agl.syntax.nodes import Block, LetDecl, Program, UnitLit, VarDecl

        items = program.body.items
        if not items or not isinstance(items[-1], (LetDecl, VarDecl)):
            return program, next_start_id

        # Append a UnitLit continuation — harmless to evaluate, satisfies the
        # checker's "last item must be an expression" invariant.
        synthetic = UnitLit(span=program.body.span, node_id=next_start_id)
        new_block = Block(
            items=(*items, synthetic),
            span=program.body.span,
            node_id=program.body.node_id,
        )
        new_program = Program(
            body=new_block,
            span=program.span,
            node_id=program.node_id,
        )
        return new_program, next_start_id + 1

    def _check_no_config_pragmas(self, program: "Program") -> Diagnostic | None:
        """Return a diagnostic if the entry contains a ``config`` pragma.

        Config pragmas are an exec/program feature: they set options for a batch
        ``agm exec`` run and cannot be applied to a live REPL session, whose
        settings come from CLI flags and config files.  Entering a pragma line
        is rejected with a clear message; the entry has no effect on session
        state.
        """
        from agm.agl.syntax.nodes import ConfigPragma

        for item in program.body.items:
            if isinstance(item, ConfigPragma):
                return diagnostic_from_span(
                    (
                        "config pragmas are not supported in the REPL; "
                        "set options via CLI flags or the config file"
                    ),
                    item.span,
                )
        return None

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

    def _pre_eval_param_check(
        self,
        program: "Program",
        checked: "CheckedProgram",
        warnings: list[Diagnostic],
    ) -> tuple[EntryResult | None, dict[str, Value], str | None, dict[str, object]]:
        """Validate and convert config-backed params without mutating session state."""
        from agm.agl.runtime.runtime import convert_param_value
        from agm.agl.syntax.nodes import ParamDecl, ProgramDecl

        def reject(message: str, span: "SourceSpan") -> EntryResult:
            return self._fail([diagnostic_from_span(message, span)], warnings)

        param_values: dict[str, Value] = {}
        entry_program_name: str | None = None
        effective_config = self._active_config

        for item in program.body.items:
            if isinstance(item, ProgramDecl):
                if self._program_name is not None and self._program_name != item.name:
                    return (
                        reject(
                            f"Program name already set to {self._program_name!r}; "
                            f"cannot redeclare as {item.name!r}. Use :reset first.",
                            item.span,
                        ),
                        {},
                        None,
                        self._active_config,
                    )
                if self._program_name is None:
                    entry_program_name = item.name
                    effective_config = (
                        self._params_config_loader(item.name)
                        if self._params_config_loader is not None
                        else {}
                    )
            elif isinstance(item, ParamDecl):
                raw_config = effective_config.get(item.name)
                if raw_config is not None:
                    declared_type = checked.type_env.get_binding_type(item.node_id)
                    assert declared_type is not None
                    try:
                        param_values[item.name] = convert_param_value(
                            item.name, raw_config, declared_type
                        )
                    except (TypeError, ValueError) as exc:
                        return (
                            reject(
                                f"Config value for param {item.name!r} is invalid: {exc}",
                                item.span,
                            ),
                            {},
                            None,
                            self._active_config,
                        )
                elif item.default is None:
                    effective_program_name = entry_program_name or self._program_name
                    prog_hint = (
                        f" via [params.{effective_program_name}] config"
                        if effective_program_name is not None
                        else ""
                    )
                    return (
                        reject(
                            f"Missing required param {item.name!r}: provide it"
                            f"{prog_hint} or a default expression.",
                            item.span,
                        ),
                        {},
                        None,
                        self._active_config,
                    )

        return None, param_values, entry_program_name, effective_config

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
        orig_program: "Program",
        checked: "CheckedProgram",
        contracts: dict[int, object],
        host_env: "HostEnvironment",
        warnings: list[Diagnostic],
        next_start_id: int,
        param_values: dict[str, Value],
        entry_program_name: str | None,
        entry_active_config: dict[str, object],
    ) -> EntryResult:
        """Execute the entry in a child scope; promote on success, discard on error.

        ``orig_program`` is the program as the user typed it (before any
        trailing-binder synthesis); ``checked.resolved.program`` is the pipeline
        program (potentially with a synthetic UnitLit appended).  Classification,
        echo data, and promotion use ``orig_program`` so the user-visible outcome
        is accurate.
        """
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.scope import Scope
        from agm.agl.repl.agents import AgentCancelled
        from agm.agl.repl.echo_interpreter import EchoInterpreter
        from agm.agl.runtime.contract import OutputContract
        from agm.agl.runtime.runtime import exception_value_to_run_error
        from agm.agl.runtime.trace import TraceStore
        from agm.agl.syntax.nodes import Binder, Declaration

        typed_contracts: dict[int, OutputContract] = {
            nid: c for nid, c in contracts.items() if isinstance(c, OutputContract)
        }

        # Fresh child scope: prior session values resolve via the parent chain,
        # new bindings land in the child and are promoted only on success.
        child_scope = Scope(parent=self._value_scope)

        # Atomicity snapshot: a ``:=`` to a PRIOR session binding mutates that
        # binding's ``.value`` in place in the persistent value scope (the child
        # only holds NEW let/var bindings).  New keys never land here and ``:=``
        # never adds/removes keys (it only updates an existing binding via
        # ``Scope.assign_value``), so a shallow value snapshot of ONLY the binding
        # names targeted by ``:=`` statements in this entry is a complete,
        # correct rollback point — Value objects are immutable, so storing the
        # reference suffices.  On a runtime raise we restore each binding's
        # ``.value`` from this snapshot.
        #
        # Optimisation: entries with no ``:=`` targeting a prior session binding
        # need no snapshot at all (new let/var bindings live in the child scope
        # and are simply discarded on abort).  We collect targeted names
        # statically from the original program before evaluation.
        assign_targets = _assign_targets_in_program(orig_program)
        value_snapshot: dict[str, Value] = {
            name: binding.value
            for name, binding in self._value_scope.bindings.items()
            if name in assign_targets
        }

        # One trace run per entry: a fresh ``TraceStore`` (own ``run_id``)
        # appends to the shared file, bracketed by ``run_start``/``run_end``.
        # ``path=None`` (no ``--log-file``) makes every write a silent no-op.
        trace = TraceStore(path=self._trace_path)
        trace.run_start()

        # The pipeline program (checked.resolved.program) may have a synthetic
        # UnitLit appended for trailing-binder entries; the interpreter runs on
        # that.  Echo capture uses the pipeline program's last item — for a bare
        # expression entry both the original and pipeline programs agree on the
        # last item; for a trailing-binder entry the pipeline's last item is the
        # synthetic UnitLit, but since _classify uses orig_program and returns
        # "binding", _echo_data ignores captured and reads from the value scope.
        pipeline_items = checked.resolved.program.body.items
        interp = EchoInterpreter(
            checked=checked,
            registry=host_env.registry,
            contracts=typed_contracts,
            type_env=checked.type_env,
            loop_limit=self._default_loop_limit,
            strict_json=self._default_strict_json,
            source=text,
            shell_exec_timeout=self._shell_exec_timeout,
            trace=trace,
            param_values=param_values,
        )
        # Echo the value of a trailing bare expression (captured during exec).
        # In v2, a bare Expr (not a Binder or Declaration) is the trailing item.
        last_pipeline_item = pipeline_items[-1] if pipeline_items else None
        if last_pipeline_item is not None and not isinstance(
            last_pipeline_item, (Binder, Declaration)
        ):
            # It's an Expr — set the echo node id to capture it during execution.
            interp.echo_node_id = last_pipeline_item.node_id

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
            return self._abort(orig_program, warnings, trace, value_snapshot, error=error)
        except (AgentCancelled, KeyboardInterrupt):
            # A declined confirmation or a Ctrl-C during a live agent call aborts
            # the entry atomically.  The cancellation is a host signal, not an
            # in-language raise, so it surfaces as a diagnostic rather than a
            # mapped AgL exception.
            return self._abort(
                orig_program,
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
            program=orig_program,
            checked=checked,
            child_scope=child_scope,
            next_start_id=next_start_id,
            entry_program_name=entry_program_name,
            entry_active_config=entry_active_config,
        )
        kind, name = self._classify(orig_program)
        value, value_type = self._echo_data(orig_program, checked, captured)
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
        binding's ``.value`` undoes any in-place ``:=`` mutation of a prior
        binding.  The snapshot contains ONLY the names that could have been mutated
        (those targeted by ``:=`` statements in the entry), and all of them must
        still be present in the session frame (``:=`` only updates existing
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
        entry_program_name: str | None,
        entry_active_config: dict[str, object],
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
        # This includes closures (FuncDef) and AgentValues installed by the
        # interpreter's pre-pass in the child scope.
        for vname, binding in child_scope.bindings.items():
            self._value_scope.bindings[vname] = binding

        # Declared params: register successful ParamDecl entries.
        for item in program.body.items:
            if isinstance(item, ParamDecl):
                param_type = checked.type_env.get_binding_type(item.node_id)
                assert param_type is not None
                self._declared_params[item.name] = param_type

        # Persist constructor candidates so subsequent entries can reference
        # constructors from types declared in this entry.
        for cname, crefs in checked.resolved.constructor_candidates.items():
            existing = list(self._ambient_constructor_candidates.get(cname, ()))
            # Merge: skip duplicates by owner_name.
            for cref in crefs:
                if not any(e.owner_name == cref.owner_name for e in existing):
                    existing.append(cref)
            self._ambient_constructor_candidates[cname] = tuple(existing)
        # Persist type names for qualified constructor access.
        self._ambient_type_names = self._ambient_type_names | checked.resolved.declared_type_names

        if entry_program_name is not None:
            self._program_name = entry_program_name
            self._active_config = entry_active_config

        self._source_log.append(text)
        self._next_node_id = next_start_id

    def _classify(self, program: "Program") -> tuple[EntryKind, str | None]:
        """Classify the entry by its last item; return (kind, name)."""
        from agm.agl.syntax.nodes import (
            AgentDecl,
            AssignStmt,
            Binder,
            Declaration,
            EnumDef,
            FuncDef,
            LetDecl,
            ParamDecl,
            ProgramDecl,
            RecordDef,
            TypeAlias,
            VarDecl,
        )

        # A parsed program always has at least one item (empty/comment-only
        # source fails parsing earlier).
        last = program.body.items[-1]
        # Bare expression (not a binder or declaration) → "expression"
        if not isinstance(last, (Binder, Declaration)):
            return "expression", None
        if isinstance(last, (LetDecl, VarDecl)):
            return "binding", last.name
        if isinstance(
            last,
            (RecordDef, EnumDef, TypeAlias, ParamDecl, ProgramDecl, FuncDef, AgentDecl),
        ):
            return "declaration", last.name
        # AssignStmt → "statement"
        if isinstance(last, AssignStmt):
            return "statement", None
        # Remaining Declaration kinds (ConfigPragma is rejected earlier, but handle
        # defensively).
        return "statement", None  # pragma: no cover

    def _echo_data(
        self, program: "Program", checked: "CheckedProgram", captured: "Value | None"
    ) -> tuple["Value | None", "Type | None"]:
        """Compute the echoed (value, value_type) from the promoted state.

        *captured* is the value of a trailing bare expression recorded during
        execution (``None`` when the last item is not a bare expression).
        """
        from agm.agl.syntax.nodes import Binder, Declaration, LetDecl, VarDecl

        # A parsed program always has at least one item.
        last = program.body.items[-1]
        value_type = self._value_type_of_last(program, checked)
        # Bare expression (not a binder or declaration) → echoed from captured
        if not isinstance(last, (Binder, Declaration)):
            return captured, value_type
        if isinstance(last, (LetDecl, VarDecl)):
            binding = self._value_scope.lookup(last.name)
            value = binding.value if binding is not None else None
            return value, value_type
        return None, None

    def _value_type_of_last(
        self, program: "Program", checked: "CheckedProgram"
    ) -> "Type | None":
        """Static type carried by the entry's last item, or ``None``.

        The checked type of the expression for a bare-expression entry, the
        declared binding type for a ``let``/``var``, ``None`` otherwise.  Shared
        by the check-only result builder and the success echo so the two agree
        on how an entry's type is derived.
        """
        from agm.agl.syntax.nodes import Binder, Declaration, LetDecl, VarDecl

        # A parsed program always has at least one item (empty/comment-only
        # source fails parsing earlier).
        last = program.body.items[-1]
        # Bare expression → node type from checked side table
        if not isinstance(last, (Binder, Declaration)):
            # After narrowing: last is an Expr (not a Binder or Declaration).
            return checked.node_types.get(last.node_id)
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
        from agm.agl.syntax.nodes import Binder, Declaration
        from agm.agl.typecheck import check

        host_env = self._runtime.host_environment()
        # Throwaway ids: type_of never promotes and never advances the session
        # counter, so seeding at ``_next_node_id`` is safe — all promoted ids are
        # strictly below it, making this parse's ids disjoint from the session's.
        program, _ = parse_program_seeded(text, start_id=self._next_node_id)
        items = program.body.items
        if len(items) != 1 or isinstance(items[0], (Binder, Declaration)):
            raise AglError(
                "':type' expects a single expression, "
                "not a binding, declaration, or statement."
            )
        expr_item = items[0]
        resolved = resolve(
            program,
            parent_scope=self._session_scope,
            ambient_agents=self._ambient_agents(host_env),
        )
        checked = check(resolved, host_env.capabilities, seed_env=self._type_env)
        typ = checked.node_types.get(expr_item.node_id)
        assert typ is not None
        return repr(typ)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def bindings(self) -> list[tuple[str, "Type", "Value"]]:
        """Return promoted user bindings as (name, declared type, current value).

        Includes params, which resolve eagerly and live in the value scope.
        Constructor bindings are excluded — they are type-system entities with no
        independent runtime value.
        """
        from agm.agl.scope.symbols import BinderKind

        result: list[tuple[str, Type, Value]] = []
        for name, ref in self._session_scope.bindings.items():
            if ref.kind == BinderKind.constructor_binding:
                continue
            binding = self._value_scope.lookup(name)
            assert binding is not None
            typ = self._type_env.get_binding_type(ref.decl_node_id)
            # Every promoted let/var/param binding has a recorded type.
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

    def declared_params(self) -> list[tuple[str, "Type", "Value"]]:
        """Return declared params as (name, type, resolved value)."""
        result: list[tuple[str, Type, Value]] = []
        for name, typ in self._declared_params.items():
            binding = self._value_scope.lookup(name)
            assert binding is not None
            result.append((name, typ, binding.value))
        return result

    def program_name(self) -> str | None:
        """Return the active program name, if declared."""
        return self._program_name

    def reset(self) -> None:
        """Clear ALL session state (symbols, types, values, params, source, ids)."""
        from agm.agl.eval.scope import Scope
        from agm.agl.scope.symbols import ScopeNode
        from agm.agl.typecheck.env import TypeEnvironment

        self._session_scope = ScopeNode(node_id=-1, parent=None)
        self._type_env = TypeEnvironment()
        # Re-seed the sentinel AgentType for ambient agents (see __init__).
        self._type_env.set_binding_type(-1, self._make_agent_type())
        self._value_scope = Scope(parent=None)
        self._next_node_id = 0
        self._program_name = None
        self._active_config = {}
        self._declared_params = {}
        self._source_log = []
        self._declared_agents = set()
        self._ambient_constructor_candidates = {}
        self._ambient_type_names = frozenset()

    def load_file(self, path: "Path") -> list[EntryResult]:
        """Evaluate the contents of *path* INCREMENTALLY, one item per entry.

        Each top-level item is fed to :meth:`eval_entry` in order, exactly as
        if the user had typed it at the prompt.  This makes redefinition/shadowing
        work on load (within a single entry it would be a duplicate-declaration
        error) so a ``:save`` transcript reliably round-trips through ``:load``.

        The load halts at the FIRST non-``ok`` result (like running a script);
        the returned list holds the results collected so far, including the
        failing one.  Items that already succeeded remain promoted.

        A syntax error in the file yields a single failed ``EntryResult`` carrying
        the parse diagnostic.  An empty or comment-only file has no items to
        run and yields an empty list (a benign no-op).
        """
        from agm.agl._text import normalize_newlines
        from agm.agl.parser import AglSyntaxError, parse_program
        from agm.core.fs import read_text

        # Normalize newlines with the SAME helper the lexer/interpreter use so the
        # item-span char offsets align with the text we slice below.
        normalized = normalize_newlines(read_text(path))

        # A blank / comment-only file has nothing to run — load it as a no-op
        # rather than surfacing the parser's "Unexpected end of input" error.
        if not has_runnable_statements(normalized):
            return []

        # Parse the whole file ONCE only to find top-level item boundaries;
        # this parse is never promoted (each slice is re-parsed by eval_entry with
        # the session's continuing node-id counter).  start_id=0 is fine here.
        try:
            program = parse_program(normalized)
        except AglSyntaxError as exc:
            return [self._fail([exc.to_diagnostic()], [])]

        results: list[EntryResult] = []
        for item in program.body.items:
            slice_text = normalized[item.span.start_offset : item.span.end_offset]
            result = self.eval_entry(slice_text)
            results.append(result)
            if not result.ok:
                break  # halt on the first failing item, like a script
        return results

    def dump_source(self) -> str:
        """Return the accumulated successfully-promoted entry sources (newline-joined)."""
        return "\n".join(self._source_log)
