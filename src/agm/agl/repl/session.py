"""UI-free incremental session core for the AgL REPL (``ReplSession``).

``ReplSession`` keeps a **persistent incremental environment**: each entry is
parsed → resolved → typechecked → evaluated **exactly once** against accumulated
session state (symbols, types, declarations, runtime values).  Agent calls fire
exactly once and are never replayed, because each entry executes ONLY its own
statements — references to earlier bindings read stored runtime ``Value``s.

The driver reproduces ``WorkflowRuntime.run``'s IR pipeline incrementally. A
persistent link image and base frame retain IDs, metadata, closures, values, and
cells across entries. Runtime failure is non-transactional: every initializer
completed before the failure remains visible, while unreached initializers do not.

This module is intentionally UI-free — it returns plain ``EntryResult`` data;
rendering, meta-commands, and the prompt_toolkit console are later milestones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from agm.agl.diagnostics import AglError, Diagnostic, diagnostic_from_span

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from agm.agl.ir.ids import Location
    from agm.agl.modules.ids import ModuleId
    from agm.agl.modules.loader import LoadedModule
    from agm.agl.modules.roots import RootSet
    from agm.agl.runtime.agents import AgentFn
    from agm.agl.runtime.codec import OutputCodec
    from agm.agl.runtime.runtime import HostEnvironment, RunError
    from agm.agl.scope.symbols import ConstructorRef, ScopeNode
    from agm.agl.semantics.values import Frame, Value
    from agm.agl.syntax.nodes import ImportDecl, Program
    from agm.agl.syntax.spans import SourceSpan
    from agm.agl.typecheck.env import CheckedProgram, TypeEnvironment
    from agm.agl.typecheck.graph import CheckedModule, CheckedModuleGraph
    from agm.agl.typecheck.types import Type


EntryKind = Literal["expression", "binding", "declaration", "statement", "type"]

# Layout-only token types that carry no statement to evaluate.
_TRIVIAL_TOKENS: frozenset[str] = frozenset({"_NEWLINE", "_INDENT", "_DEDENT"})


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
        effecting expr (``print``, etc.) → ``"statement"``; a REPL-only bare
        type expression (``int``, a declared type name, ``list[T]``) →
        ``"type"`` (``value_type`` set, no value, no state change).
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
    ``installed``
        Names installed before a failed entry stopped. Empty for pre-execution
        failures and successful entries.
    ``quote_strings``
        Whether REPL echo should quote a top-level text value. This is normally
        ``True``. The only exception is a standalone ``ask`` builtin entry,
        whose response is echoed as display text rather than as an AgL string
        literal.
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
    installed: tuple[str, ...] = ()
    quote_strings: bool = True


# ---------------------------------------------------------------------------
# ReplSession — the persistent incremental driver
# ---------------------------------------------------------------------------


class ReplSession:
    """Persistent incremental AgL evaluation session (UI-free core).

    Constructor parameters mirror ``WorkflowRuntime`` so a host can wire the same
    agent backing.  Registration (``register_agent``/``register_codec``) is
    delegated to an internal ``WorkflowRuntime`` so the reserved-name / duplicate
    validation and host-environment assembly are shared rather than duplicated.

    Each entry is incrementally linked and executed against a persistent IR base
    frame. Completed effects survive a later runtime failure in the same entry.
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
        cwd: "Path | None" = None,
        stdlib_root: "Path | None" = None,
        lib_root: "Path | None" = None,
        configured_roots: "Iterable[tuple[str, Path]]" = (),
        extra_cli_roots: "Iterable[str]" = (),
    ) -> None:
        from pathlib import Path

        from agm.agl.lower import LinkImage
        from agm.agl.runtime.runtime import WorkflowRuntime
        from agm.agl.scope.symbols import ScopeNode
        from agm.agl.typecheck.env import TypeEnvironment
        from agm.config.module_roots import resolve_stdlib_root

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
        self._link_image = LinkImage()
        self._ir_base_frame: Frame = {}
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

        # Module roots configuration (M6 REPL import support).
        # These are stored so _ensure_roots() can assemble the RootSet lazily.
        self._cwd: Path | None = cwd
        self._stdlib_root: Path | None = (
            resolve_stdlib_root(home=Path.home()) if stdlib_root is None else stdlib_root
        )
        self._lib_root: Path | None = lib_root
        self._configured_roots: tuple[tuple[str, Path], ...] = tuple(configured_roots)
        self._extra_cli_roots: tuple[str, ...] = tuple(extra_cli_roots)
        # Lazily assembled RootSet (set directly in tests via s._roots = ...).
        self._roots: RootSet | None = None
        # Cached lib modules from prior REPL graph-mode entries.
        self._loaded_lib_modules: dict[ModuleId, LoadedModule] = {}
        # Accumulated import declarations from prior promoted graph-mode entries.
        # These are prepended to each new entry's program in graph mode so that
        # open imports (e.g. ``import util``) persist across entries.
        self._accumulated_imports: list["ImportDecl"] = []

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
    # Module roots (M6)
    # ------------------------------------------------------------------

    def _ensure_roots(self) -> "RootSet":
        """Build the ``RootSet`` lazily on first import use."""
        if self._roots is not None:
            return self._roots
        from pathlib import Path

        from agm.agl.modules.roots import assemble_roots

        cwd = self._cwd if self._cwd is not None else Path.cwd()
        self._roots = assemble_roots(
            invocation_root=cwd,
            stdlib_root=self._stdlib_root,
            lib_root=self._lib_root,
            configured=self._configured_roots,
            cli=self._extra_cli_roots,
            cwd=cwd,
        )
        return self._roots

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def eval_entry(self, text: str, *, check_only: bool = False) -> EntryResult:
        """Parse → resolve → check → (eval) one entry against the session.

        Completed runtime initializers are promoted even when a later initializer
        fails. ``check_only`` runs the static pipeline without linking, executing,
        promoting, or advancing the node-id counter.

        REPL-only fallback: when the entry fails to evaluate as a program, the
        loop tries to read it as a bare type expression (e.g. ``int``, a declared
        enum/record name, ``list[T]``).  If that resolves to a known type, a
        ``kind == "type"`` result echoing the type is returned instead of the
        original ``'X' is not defined`` error.  Entries that evaluate
        successfully as values are never intercepted, so record constructors and
        bindings keep their normal echo.
        """
        result = self._eval_entry_pipeline(text, check_only=check_only)
        if not result.ok:
            type_result = self._try_type_entry(text)
            if type_result is not None:
                return type_result
        return result

    def _try_type_entry(self, text: str) -> EntryResult | None:
        """Attempt to interpret *text* as a bare type-expression entry.

        Returns a ``kind == "type"`` :class:`EntryResult` echoing the resolved
        type when *text* parses as a single type expression AND resolves to a
        known type in the session type environment; returns ``None`` otherwise
        so the caller keeps the original failure result.

        This is a REPL-only convenience (the language is unchanged): typing a
        type is not a value expression, so previously it surfaced ``'X' is not
        defined.``.  Like :meth:`type_of`, this never evaluates, promotes,
        advances the node-id counter, or mutates session state.  The parse uses
        throwaway node ids; only the resolved :class:`Type` is kept.
        """
        from agm.agl.parser import AglSyntaxError, parse_type_expr
        from agm.agl.typecheck import AglTypeError

        try:
            type_expr = parse_type_expr(text, start_id=self._next_node_id)
        except AglSyntaxError:
            return None
        try:
            typ = self._type_env.resolve_type_expr(type_expr)
        except AglTypeError:
            return None
        return EntryResult(
            kind="type",
            name=None,
            value=None,
            value_type=typ,
            diagnostics=[],
            warnings=[],
            error=None,
            ok=True,
        )

    def _eval_entry_pipeline(self, text: str, *, check_only: bool = False) -> EntryResult:
        """Parse → resolve → check → (eval) one entry against the session (core).

        Completed runtime initializers are promoted even when a later initializer
        fails. ``check_only`` runs the static pipeline without linking, executing,
        promoting, or advancing the node-id counter.
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

        # [1d] Check whether this entry has import declarations or there are
        # already cached library modules from a prior entry.  If so, use the
        # graph-mode pipeline which handles cross-module resolution and execution.
        from agm.agl.syntax.nodes import ImportDecl as _ImportDecl

        has_imports = any(
            isinstance(item, _ImportDecl) for item in pipeline_program.body.items
        )
        use_graph_mode = has_imports or bool(self._loaded_lib_modules)

        if use_graph_mode:
            return self._eval_entry_graph_mode(
                text=text,
                orig_program=orig_program,
                pipeline_program=pipeline_program,
                host_env=host_env,
                tab_warnings=tab_warnings,
                next_start_id=next_start_id,
                check_only=check_only,
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

        # [7] Incrementally link and execute only this entry's IR initializers.
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
            quote_strings=self._quote_strings_for_entry(program),
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
        """Lower and execute an entry against the persistent IR runtime image.

        ``orig_program`` is the program as the user typed it (before any
        trailing-binder synthesis); ``checked.resolved.program`` is the pipeline
        program (potentially with a synthetic UnitLit appended).  Classification,
        echo data, and promotion use ``orig_program`` so the user-visible outcome
        is accurate.
        """
        from agm.agl.eval.ir_interpreter import IrInterpreter
        from agm.agl.lower import lower_repl_entry
        from agm.agl.runtime.request import AgentCancelled
        from agm.agl.runtime.runtime import exception_value_to_run_error
        from agm.agl.runtime.trace import TraceStore
        from agm.agl.semantics.exceptions import AglRaise

        lowered = lower_repl_entry(
            checked,
            image=self._link_image,
            source_text=text,
            source_label=f"<repl:{len(self._source_log) + 1}>",
            validate=True,
        )
        ir_param_values = {
            param.symbol: param_values[param.public_name]
            for param in lowered.program.params
            if param.public_name in param_values
        }
        from agm.agl.runtime.runtime import _materialize_ir_contracts

        host_contracts, _ = _materialize_ir_contracts(lowered.program, host_env.codecs)

        # One trace run per entry: a fresh ``TraceStore`` (own ``run_id``)
        # appends to the shared file, bracketed by ``run_start``/``run_end``.
        # ``path=None`` (no ``--log-file``) makes every write a silent no-op.
        trace = TraceStore(path=self._trace_path)
        trace.run_start()

        base_frame = self._ir_base_frame
        assert isinstance(base_frame, dict)
        interp = IrInterpreter(
            lowered.program,
            registry=host_env.registry,
            loop_limit=self._default_loop_limit,
            strict_json=self._default_strict_json,
            shell_exec_timeout=self._shell_exec_timeout,
            trace=trace,
            param_values=ir_param_values,
            host_contracts=host_contracts,
            base_frame=base_frame,
        )

        try:
            interp.run()
        except AglRaise as exc:
            error = exception_value_to_run_error(exc.exc, span=exc.span)
            trace.exception(
                type_name=error.type_name,
                message=str(error.fields.get("message", "")),
                trace_id=str(error.fields.get("trace_id", "")),
                span=exc.span,
            )
            trace.run_end(ok=False)
            installed = self._promote_ir_state(
                text=text,
                program=orig_program,
                checked=checked,
                next_start_id=next_start_id,
                entry_program_name=entry_program_name,
                entry_active_config=entry_active_config,
                partial=True,
                failure_span=exc.span,
            )
            kind, name = self._classify(orig_program)
            return EntryResult(
                kind=kind,
                name=name,
                value=None,
                value_type=None,
                diagnostics=[],
                warnings=warnings,
                error=error,
                ok=False,
                trace_path=self._trace_path,
                installed=installed,
            )
        except (AgentCancelled, KeyboardInterrupt) as exc:
            cancel_span = exc.span if isinstance(exc, AgentCancelled) else None
            trace.run_end(ok=False)
            installed = self._promote_ir_state(
                text=text,
                program=orig_program,
                checked=checked,
                next_start_id=next_start_id,
                entry_program_name=entry_program_name,
                entry_active_config=entry_active_config,
                partial=True,
                failure_span=cancel_span,
            )
            kind, name = self._classify(orig_program)
            return EntryResult(
                kind=kind,
                name=name,
                value=None,
                value_type=None,
                diagnostics=[
                    Diagnostic(message="Agent call cancelled — entry aborted.", line=1)
                ],
                warnings=warnings,
                error=None,
                ok=False,
                trace_path=self._trace_path,
                installed=installed,
            )

        trace.run_end(ok=True)
        self._promote_ir_state(
            text=text,
            program=orig_program,
            checked=checked,
            next_start_id=next_start_id,
            entry_program_name=entry_program_name,
            entry_active_config=entry_active_config,
            partial=False,
            failure_span=None,
        )
        kind, name = self._classify(orig_program)
        captured = (
            interp.initializer_values[lowered.trailing_expression]
            if lowered.trailing_expression is not None
            else None
        )
        value, value_type = self._echo_data_ir(orig_program, checked, captured)
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
            quote_strings=self._quote_strings_for_entry(orig_program),
        )

    def _promote_ir_state(
        self,
        *,
        text: str,
        program: "Program",
        checked: "CheckedProgram",
        next_start_id: int,
        entry_program_name: str | None,
        entry_active_config: dict[str, object],
        partial: bool,
        failure_span: "SourceSpan | Location | None",
    ) -> tuple[str, ...]:
        """Advance static state in lockstep with installed IR frame symbols."""
        from agm.agl.syntax.nodes import (
            AgentDecl,
            EnumDef,
            FuncDef,
            LetDecl,
            ParamDecl,
            ProgramDecl,
            RecordDef,
            TypeAlias,
            VarDecl,
        )

        entry_root = checked.resolved.root_scope
        named_declarations = (
            AgentDecl,
            EnumDef,
            FuncDef,
            LetDecl,
            ParamDecl,
            ProgramDecl,
            RecordDef,
            TypeAlias,
            VarDecl,
        )
        entry_names = {
            item.name for item in program.body.items if isinstance(item, named_declarations)
        }
        for item in program.body.items:
            if isinstance(item, EnumDef):
                entry_names.update(variant.name for variant in item.variants)
        installed: list[str] = []
        for name, ref in entry_root.bindings.items():
            symbol = self._link_image.symbol_for_decl(ref.decl_node_id)
            declared_before_failure = (
                failure_span is not None
                and ref.decl_span.end_offset <= failure_span.start_offset
            )
            installed_before_failure = symbol in self._ir_base_frame and (
                failure_span is None
                or ref.decl_span.start_offset <= failure_span.start_offset
            )
            if not partial or installed_before_failure or declared_before_failure:
                self._session_scope.bindings[name] = ref
                if partial and name in entry_names:
                    installed.append(name)
        self._type_env.seed_from(checked.type_env)

        def _before_failure(end_offset: int) -> bool:
            return not partial or (
                failure_span is not None and end_offset <= failure_span.start_offset
            )

        promoted_type_names = {
            item.name
            for item in program.body.items
            if isinstance(item, (RecordDef, EnumDef, TypeAlias))
            and _before_failure(item.span.end_offset)
        }
        promoted_agents = {
            item.name
            for item in program.body.items
            if isinstance(item, AgentDecl)
            and _before_failure(item.span.end_offset)
        }
        self._declared_agents.update(promoted_agents)
        if promoted_type_names:
            for cname, crefs in checked.resolved.constructor_candidates.items():
                crefs = tuple(ref for ref in crefs if ref.owner_name in promoted_type_names)
                if crefs:
                    self._ambient_constructor_candidates[cname] = crefs
            self._ambient_type_names |= promoted_type_names
        for item in program.body.items:
            if isinstance(item, ParamDecl):
                symbol = self._link_image.symbol_for_decl(item.node_id)
                # The IR interpreter installs every entry param into the base
                # frame up front (before any initializer runs), so a bare
                # ``symbol in self._ir_base_frame`` check would record a param
                # declared after the failure even though the scope-promotion
                # loop above excluded its binding by source position. Mirror
                # that loop's ``installed_before_failure`` criterion so
                # ``_declared_params`` stays aligned with ``_session_scope``
                # (otherwise ``declared_params()`` raises ``KeyError``).
                installed_before_failure = symbol in self._ir_base_frame and (
                    failure_span is None
                    or item.span.start_offset <= failure_span.start_offset
                )
                if not partial or installed_before_failure:
                    typ = checked.type_env.get_binding_type(item.node_id)
                    assert typ is not None
                    self._declared_params[item.name] = typ
        if entry_program_name is not None and not partial:
            self._program_name = entry_program_name
            self._active_config = entry_active_config
        if not partial:
            self._source_log.append(text)
        self._next_node_id = next_start_id
        return tuple(installed)

    def _echo_data_ir(
        self, program: "Program", checked: "CheckedProgram", captured: "Value | None"
    ) -> tuple["Value | None", "Type | None"]:
        from agm.agl.semantics.values import Cell
        from agm.agl.syntax.nodes import Binder, Declaration, LetDecl, VarDecl

        last = program.body.items[-1]
        value_type = self._value_type_of_last(program, checked)
        if not isinstance(last, (Binder, Declaration)):
            return captured, value_type
        if isinstance(last, (LetDecl, VarDecl)):
            symbol = self._link_image.symbol_for_decl(last.node_id)
            slot = self._ir_base_frame.get(symbol) if symbol is not None else None
            return (slot.value if isinstance(slot, Cell) else slot), value_type
        return None, None

    def _inject_accumulated_imports(self, program: "Program") -> "Program":
        """Return a new program with accumulated session imports prepended.

        Prior graph-mode entries may have imported modules via open import.
        To make those imports persist across entries, we prepend the stored
        ``ImportDecl`` nodes to the current entry's program items.  Nodes
        with already-present module_paths are de-duplicated (if the current
        entry re-imports the same module, the current entry's decl wins).
        """
        from agm.agl.syntax.nodes import Block, ImportDecl, Program

        if not self._accumulated_imports:
            return program

        # Collect (module_path, wildcard) pairs already imported in the current entry.
        current_import_paths: set[tuple[tuple[str, ...], bool]] = set()
        for item in program.body.items:
            if isinstance(item, ImportDecl):
                current_import_paths.add((tuple(item.module_path), item.wildcard))

        # Build the injected preamble: accumulated imports NOT already in the entry.
        preamble = [
            decl
            for decl in self._accumulated_imports
            if (tuple(decl.module_path), decl.wildcard) not in current_import_paths
        ]

        if not preamble:
            return program

        new_items = tuple(preamble) + program.body.items
        new_block = Block(
            items=new_items,
            span=program.body.span,
            node_id=program.body.node_id,
        )
        return Program(body=new_block, span=program.span, node_id=program.node_id)

    def _eval_entry_graph_mode(
        self,
        *,
        text: str,
        orig_program: "Program",
        pipeline_program: "Program",
        host_env: "HostEnvironment",
        tab_warnings: list[Diagnostic],
        next_start_id: int,
        check_only: bool,
    ) -> EntryResult:
        """Graph-mode pipeline for REPL entries that have imports or cached lib modules.

        Builds the module graph from the already-parsed *pipeline_program*, runs
        the full scope/typecheck pass with the session context, then evaluates
        or returns a check-only result.
        """
        from agm.agl.modules.errors import (
            AmbiguousModule,
            ImportEntryError,
            ModuleNotFound,
            ModulePrefixNotFound,
        )
        from agm.agl.modules.ids import ENTRY_ID
        from agm.agl.modules.loader import build_repl_graph
        from agm.agl.parser import AglSyntaxError
        from agm.agl.runtime.contract import materialize_contract
        from agm.agl.scope import AglScopeError
        from agm.agl.scope.graph import resolve_graph
        from agm.agl.typecheck import AglTypeError
        from agm.agl.typecheck.graph import check_graph

        roots = self._ensure_roots()

        # Inject accumulated import declarations from prior entries at the head
        # of the pipeline program so that open imports persist across entries.
        entry_program = self._inject_accumulated_imports(pipeline_program)

        try:
            graph, new_next_id, new_modules = build_repl_graph(
                entry_program,
                next_start_id,
                path=None,
                cached=self._loaded_lib_modules,
                roots=roots,
            )
        except AglSyntaxError as exc:
            return self._fail([exc.to_diagnostic()], tab_warnings)
        except (
            ModuleNotFound,
            AmbiguousModule,
            ModulePrefixNotFound,
            ImportEntryError,
        ) as exc:
            return self._fail([exc.to_diagnostic()], tab_warnings)
        except Exception as exc:
            return self._fail([Diagnostic(message=str(exc), line=1)], tab_warnings)

        try:
            rgraph = resolve_graph(
                graph,
                ambient_agents=self._ambient_agents(host_env),
                entry_parent_scope=self._session_scope,
                entry_repl_session_scope=self._session_scope,
            )
        except AglScopeError as exc:
            return self._fail([exc.to_diagnostic()], tab_warnings)

        try:
            cgraph = check_graph(
                rgraph, host_env.capabilities, entry_seed_env=self._type_env
            )
        except AglTypeError as exc:
            return self._fail([exc.to_diagnostic()], tab_warnings)

        entry_cm = cgraph.modules[ENTRY_ID]

        # Collect warnings from all passes.
        warnings: list[Diagnostic] = [
            *tab_warnings,
            *rgraph.warnings,
            *cgraph.warnings,
        ]

        checked = self._checked_program_from_module(entry_cm)
        if check_only:
            return self._build_check_only_result(orig_program, checked, warnings)

        pre_eval_result, param_values, entry_program_name, entry_active_config = (
            self._pre_eval_param_check(orig_program, checked, warnings)
        )
        if pre_eval_result is not None:
            return pre_eval_result

        # Materialize contracts.
        contracts: dict[int, object] = {}
        contract_errors: list[Diagnostic] = []
        for node_id, spec in checked.contract_specs.items():
            try:
                contracts[node_id] = materialize_contract(spec, host_env.codecs)
            except ValueError as exc:
                contract_errors.append(Diagnostic(message=f"Contract error: {exc}", line=1))
        if contract_errors:
            return self._fail(contract_errors, warnings)

        return self._evaluate_ir_graph_mode(
            text=text,
            orig_program=orig_program,
            checked=checked,
            entry_cm=entry_cm,
            cgraph=cgraph,
            contracts=contracts,
            host_env=host_env,
            warnings=warnings,
            new_next_id=new_next_id,
            new_modules=new_modules,
            param_values=param_values,
            entry_program_name=entry_program_name,
            entry_active_config=entry_active_config,
        )

    @staticmethod
    def _checked_program_from_module(entry: "CheckedModule") -> "CheckedProgram":
        """Adapt entry-module checker output for REPL static-state promotion."""
        from agm.agl.typecheck.env import CheckedProgram

        return CheckedProgram(
            resolved=entry.resolved,
            node_types=entry.node_types,
            contract_specs=entry.contract_specs,
            call_sites=entry.call_sites,
            warnings=entry.warnings,
            type_env=entry.type_env,
            function_signatures=entry.function_signatures,
            cast_specs=entry.cast_specs,
        )

    def _evaluate_ir_graph_mode(
        self,
        *,
        text: str,
        orig_program: "Program",
        checked: "CheckedProgram",
        entry_cm: "CheckedModule",
        cgraph: "CheckedModuleGraph",
        contracts: dict[int, object],
        host_env: "HostEnvironment",
        warnings: list[Diagnostic],
        new_next_id: int,
        new_modules: "dict[ModuleId, LoadedModule]",
        param_values: "dict[str, Value]",
        entry_program_name: str | None,
        entry_active_config: dict[str, object],
    ) -> EntryResult:
        """Lower and execute one graph-mode entry in the persistent IR image."""
        from agm.agl.eval.ir_interpreter import IrInterpreter
        from agm.agl.lower import lower_repl_graph
        from agm.agl.runtime.request import AgentCancelled
        from agm.agl.runtime.runtime import (
            _materialize_ir_contracts,
            exception_value_to_run_error,
        )
        from agm.agl.runtime.trace import TraceStore
        from agm.agl.semantics.exceptions import AglRaise
        from agm.agl.syntax.nodes import ImportDecl

        del contracts
        lowered = lower_repl_graph(
            cgraph, image=self._link_image, source_text=text, validate=True
        )
        ir_params = {
            param.symbol: param_values[param.public_name]
            for param in lowered.program.params
            if param.public_name in param_values
        }
        host_contracts, _ = _materialize_ir_contracts(lowered.program, host_env.codecs)
        trace = TraceStore(path=self._trace_path)
        trace.run_start()
        interp = IrInterpreter(
            lowered.program,
            registry=host_env.registry,
            loop_limit=self._default_loop_limit,
            strict_json=self._default_strict_json,
            shell_exec_timeout=self._shell_exec_timeout,
            trace=trace,
            param_values=ir_params,
            host_contracts=host_contracts,
            base_frame=self._ir_base_frame,
        )
        try:
            interp.run()
        except AglRaise as exc:
            error = exception_value_to_run_error(exc.exc, span=exc.span)
            trace.exception(
                type_name=error.type_name,
                message=str(error.fields.get("message", "")),
                trace_id=str(error.fields.get("trace_id", "")),
                span=exc.span,
            )
            trace.run_end(ok=False)
            installed = self._promote_ir_state(
                text=text,
                program=orig_program,
                checked=checked,
                next_start_id=new_next_id,
                entry_program_name=entry_program_name,
                entry_active_config=entry_active_config,
                partial=True,
                failure_span=exc.span,
            )
            kind, name = self._classify(orig_program)
            return EntryResult(
                kind=kind,
                name=name,
                value=None,
                value_type=None,
                diagnostics=[],
                warnings=warnings,
                error=error,
                ok=False,
                trace_path=self._trace_path,
                installed=installed,
            )
        except (AgentCancelled, KeyboardInterrupt) as exc:
            cancel_span = exc.span if isinstance(exc, AgentCancelled) else None
            trace.run_end(ok=False)
            installed = self._promote_ir_state(
                text=text,
                program=orig_program,
                checked=checked,
                next_start_id=new_next_id,
                entry_program_name=entry_program_name,
                entry_active_config=entry_active_config,
                partial=True,
                failure_span=cancel_span,
            )
            kind, name = self._classify(orig_program)
            return EntryResult(
                kind=kind,
                name=name,
                value=None,
                value_type=None,
                diagnostics=[
                    Diagnostic(message="Agent call cancelled — entry aborted.", line=1)
                ],
                warnings=warnings,
                error=None,
                ok=False,
                trace_path=self._trace_path,
                installed=installed,
            )
        trace.run_end(ok=True)
        self._promote_ir_state(
            text=text,
            program=orig_program,
            checked=checked,
            next_start_id=new_next_id,
            entry_program_name=entry_program_name,
            entry_active_config=entry_active_config,
            partial=False,
            failure_span=None,
        )
        entry_imports = tuple(
            item
            for item in orig_program.body.items
            if isinstance(item, ImportDecl)
        )
        self._loaded_lib_modules.update(new_modules)
        self._link_image.mark_linked(
            mid for mid in cgraph.modules if not mid.is_entry
        )
        import_indexes = {
            (tuple(item.module_path), item.wildcard): index
            for index, item in enumerate(self._accumulated_imports)
        }
        for item in entry_imports:
            key = (tuple(item.module_path), item.wildcard)
            index = import_indexes.get(key)
            if index is None:
                import_indexes[key] = len(self._accumulated_imports)
                self._accumulated_imports.append(item)
            else:
                self._accumulated_imports[index] = item
        marker = lowered.trailing_expression
        captured = (
            interp.module_initializer_values[lowered.program.entry_module][marker]
            if marker is not None
            else None
        )
        kind, name = self._classify(orig_program)
        value, value_type = self._echo_data_ir(orig_program, checked, captured)
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
            quote_strings=self._quote_strings_for_entry(orig_program),
        )

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

    def _quote_strings_for_entry(self, program: "Program") -> bool:
        """Return the top-level text quoting mode for REPL echo.

        Only a syntactically standalone ``ask`` builtin call gets unquoted text
        display. Stored ask results, variables, bindings, and all other
        expressions use normal REPL value display.
        """
        from agm.agl.syntax.nodes import Call, VarRef

        last = program.body.items[-1]
        if isinstance(last, Call) and isinstance(last.callee, VarRef):
            return last.callee.name != "ask"
        return True

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
        from agm.agl.semantics.values import Cell

        result: list[tuple[str, Type, Value]] = []
        for name, ref in self._session_scope.bindings.items():
            if ref.kind == BinderKind.constructor_binding:
                continue
            typ = self._type_env.get_binding_type(ref.decl_node_id)
            # Every promoted let/var/param binding has a recorded type.
            assert typ is not None
            symbol = self._link_image.symbol_for_decl(ref.decl_node_id)
            slot = self._ir_base_frame.get(symbol) if symbol is not None else None
            assert slot is not None
            value = slot.value if isinstance(slot, Cell) else slot
            result.append((name, typ, value))
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
        from agm.agl.semantics.values import Cell

        for name, typ in self._declared_params.items():
            ref = self._session_scope.bindings[name]
            symbol = self._link_image.symbol_for_decl(ref.decl_node_id)
            slot = self._ir_base_frame.get(symbol) if symbol is not None else None
            assert slot is not None
            result.append((name, typ, slot.value if isinstance(slot, Cell) else slot))
        return result

    def program_name(self) -> str | None:
        """Return the active program name, if declared."""
        return self._program_name

    def type_names(self) -> frozenset[str]:
        """Return the names of types declared in prior promoted entries.

        Drives the REPL highlighter's type colouring: a NAME matching one of
        these (or a builtin type spelling) is rendered as a type.  Types declared
        in an entry become available here only after that entry is promoted.
        """
        return self._ambient_type_names

    def constructor_names(self) -> frozenset[str]:
        """Return the constructor names declared in prior promoted entries.

        Drives the REPL highlighter's constructor colouring (enum variants and
        record constructors).  Like :meth:`type_names`, populated on promotion.
        """
        return frozenset(self._ambient_constructor_candidates)

    def reset(self) -> None:
        """Clear ALL session state (symbols, types, values, params, source, ids)."""
        from agm.agl.lower import LinkImage
        from agm.agl.scope.symbols import ScopeNode
        from agm.agl.typecheck.env import TypeEnvironment

        self._session_scope = ScopeNode(node_id=-1, parent=None)
        self._type_env = TypeEnvironment()
        # Re-seed the sentinel AgentType for ambient agents (see __init__).
        self._type_env.set_binding_type(-1, self._make_agent_type())
        self._link_image = LinkImage()
        self._ir_base_frame = {}
        self._next_node_id = 0
        self._program_name = None
        self._active_config = {}
        self._declared_params = {}
        self._source_log = []
        self._declared_agents = set()
        self._ambient_constructor_candidates = {}
        self._ambient_type_names = frozenset()
        # Clear module state (M6).
        self._roots = None
        self._loaded_lib_modules = {}
        self._accumulated_imports = []

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
