"""UI-free incremental session core for the AgL REPL (``ReplSession``).

``ReplSession`` keeps a **persistent incremental environment**: each entry is
parsed → resolved → typechecked → evaluated **exactly once** against accumulated
session state (symbols, types, declarations, runtime values).  Agent calls fire
exactly once and are never replayed, because each entry executes ONLY its own
statements — references to earlier bindings read stored runtime ``Value``s.

The driver reproduces ``PipelineDriver.run``'s IR pipeline incrementally. A
persistent link image and base frame retain IDs, metadata, closures, values, and
cells across entries. Runtime failure is non-transactional: every initializer
completed before the failure remains visible, while unreached initializers do not.

This module is intentionally UI-free — it returns plain ``EntryResult`` data;
rendering, meta-commands, and the prompt_toolkit console are future work.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from agm.agl.diagnostics import AglError, Diagnostic, diagnostic_from_span
from agm.agl.repl.entry import EntryKind, EntryResult
from agm.agl.repl.graph_session import GraphSession

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from agm.agl.ir.ids import Location
    from agm.agl.modules.ids import ModuleId
    from agm.agl.modules.loader import LoadedModule
    from agm.agl.modules.roots import RootSet
    from agm.agl.runtime.agents import AgentFn
    from agm.agl.runtime.codec import OutputCodec
    from agm.agl.runtime.types import HostEnvironment
    from agm.agl.scope.symbols import ConstructorRef, ScopeNode
    from agm.agl.semantics.types import Type
    from agm.agl.semantics.values import Frame, Value
    from agm.agl.syntax.nodes import ImportDecl, InfixAssoc, Program
    from agm.agl.syntax.spans import SourceSpan
    from agm.agl.syntax.types import TypeExpr
    from agm.agl.typecheck.env import CheckedProgram, TypeEnvironment


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
# ReplSession — the persistent incremental driver
# ---------------------------------------------------------------------------


class ReplSession:
    """Persistent incremental AgL evaluation session (UI-free core).

    Constructor parameters mirror ``PipelineDriver`` so a host can wire the same
    agent backing.  Registration (``register_agent``/``register_codec``) is
    delegated to an internal ``PipelineDriver`` so the reserved-name / duplicate
    validation and host-environment assembly are shared rather than duplicated.

    Each entry is incrementally linked and executed against a persistent IR base
    frame. Completed effects survive a later runtime failure in the same entry.
    """

    def __init__(
        self,
        *,
        default_strict_json: bool = False,
        default_loop_limit: int | None = None,
        default_call_depth_limit: int | None = None,
        default_agent: "AgentFn | None" = None,
        shell_exec_timeout: float | None = None,
        trace_path: "Path | None" = None,
        params_config_loader: "Callable[[str], dict[str, object]] | None" = None,
        engine_base: "Mapping[str, Value] | None" = None,
        cwd: "Path | None" = None,
        stdlib_root: "Path | None" = None,
        lib_root: "Path | None" = None,
        configured_roots: "Iterable[tuple[str, Path]]" = (),
        extra_cli_roots: "Iterable[str]" = (),
    ) -> None:
        from pathlib import Path

        from agm.agl.lower import LinkImage
        from agm.agl.pipeline import PipelineDriver
        from agm.agl.scope.symbols import ScopeNode
        from agm.agl.typecheck.env import TypeEnvironment
        from agm.config.module_roots import resolve_stdlib_root

        self._default_strict_json = default_strict_json
        self._default_loop_limit = default_loop_limit
        self._shell_exec_timeout = shell_exec_timeout
        # Capture initial engine defaults for :reset to restore.
        self._initial_loop_limit = default_loop_limit
        self._initial_strict_json = default_strict_json
        self._initial_shell_exec_timeout = shell_exec_timeout
        # The [exec] engine base for all six engine keys, provided by the host
        # command (commands/repl.py).  Used in _build_config_base to supply the
        # runner/log/log-file base values (the three static keys).
        self._engine_base: dict[str, Value] = dict(engine_base) if engine_base is not None else {}
        # Trace destination: when set, each evaluated entry opens a fresh
        # ``TraceStore`` (its own ``run_id``) appending JSONL records to this one
        # file.  ``check_only`` entries write nothing (mirroring ``agm exec``).
        # The COMMAND validates/creates the path up front; the session assumes it
        # is writable but the no-op store tolerates failure (it disables itself).
        self._trace_path = trace_path
        self._params_config_loader = params_config_loader

        # Internal runtime owns the registrations + host-environment assembly.
        self._runtime = PipelineDriver(
            default_strict_json=default_strict_json,
            default_loop_limit=default_loop_limit,
            default_call_depth_limit=default_call_depth_limit,
            default_agent=default_agent,
            shell_exec_timeout=shell_exec_timeout,
        )
        # Reuse the driver's resolved (default-applied) limit for the per-entry
        # interpreters this session builds directly, so the canonical default
        # lives in exactly one place.
        self._default_call_depth_limit = self._runtime.default_call_depth_limit
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
        # access (``Owner::variant``) across REPL entries.
        self._ambient_type_names: frozenset[str] = frozenset()

        # Module roots configuration.
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
        # Resolved user infix fixity declared in prior promoted entries
        # (operator name → ``(priority, associativity)``). Passed to the parser
        # as ambient fixity so an ``infixl``/``infixr`` declaration made in one
        # entry makes the operator usable in later entries.
        self._accumulated_infix: dict[str, tuple[int, "InfixAssoc"]] = {}
        self._graph_session = GraphSession(self)

    @staticmethod
    def _make_agent_type() -> "Type":
        """Return an ``AgentType`` instance (deferred import, used at init and reset)."""
        from agm.agl.semantics.types import AgentType

        return AgentType()

    @staticmethod
    def _type_mentions_entry_nominal(typ: "Type", names: frozenset[str]) -> bool:
        """Return whether *typ* contains an entry-module nominal in *names*."""
        from agm.agl.semantics.types import (
            DictType,
            EnumType,
            ExceptionType,
            FunctionType,
            ListType,
            RecordType,
        )

        if isinstance(typ, (RecordType, EnumType)):
            return (typ.module_id.is_entry and typ.name in names) or any(
                ReplSession._type_mentions_entry_nominal(arg, names) for arg in typ.type_args
            )
        if isinstance(typ, ExceptionType):
            return typ.module_id.is_entry and typ.name in names
        if isinstance(typ, ListType):
            return ReplSession._type_mentions_entry_nominal(typ.elem, names)
        if isinstance(typ, DictType):
            return ReplSession._type_mentions_entry_nominal(typ.value, names)
        if isinstance(typ, FunctionType):
            return any(
                ReplSession._type_mentions_entry_nominal(param, names) for param in typ.params
            ) or ReplSession._type_mentions_entry_nominal(typ.result, names)
        return False

    # ------------------------------------------------------------------
    # Registration (delegated to the internal runtime — shared validation)
    # ------------------------------------------------------------------

    def register_agent(self, name: str, fn: "AgentFn") -> None:
        """Register a named agent (shares ``PipelineDriver`` validation)."""
        self._runtime.register_agent(name, fn)

    def register_codec(self, codec: "OutputCodec") -> None:
        """Register a custom output codec (shares ``PipelineDriver`` validation)."""
        self._runtime.register_codec(codec)

    # ------------------------------------------------------------------
    # Module roots
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
        enum/record name, a bare generic type definition, ``list[T]``).  If
        that resolves to a known type, a ``kind == "type"`` result echoing the
        type is returned instead of the original ``'X' is not defined`` error.
        Entries that evaluate
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
        throwaway node ids; only the resolved :class:`Type` or generic type
        definition display is kept.
        """
        from agm.agl.parser import AglSyntaxError, parse_type_expr
        from agm.agl.typecheck import AglTypeError

        try:
            type_expr = parse_type_expr(text, start_id=self._next_node_id)
        except AglSyntaxError:
            return None

        type_envs = [self._type_env]
        graph_type_env = self._build_type_entry_graph_env()
        if graph_type_env is not None:
            type_envs.append(graph_type_env)

        for type_env in type_envs:
            generic_result = self._try_generic_type_entry(type_expr, type_env)
            if generic_result is not None:
                return generic_result
            try:
                typ = type_env.resolve_type_expr(type_expr)
            except AglTypeError:
                continue
            return EntryResult(
                kind="type",
                name=None,
                value=None,
                value_type=typ,
                diagnostics=[],
                warnings=[],
                error=None,
                ok=True,
                type_table=type_env.type_table,
            )
        return None

    def _try_generic_type_entry(
        self,
        type_expr: "TypeExpr",
        type_env: "TypeEnvironment",
    ) -> EntryResult | None:
        """Return a type-entry result for a bare unapplied generic, if any."""
        from agm.agl.repl.type_display import format_generic_type_def_for_repl
        from agm.agl.syntax.types import NameT
        from agm.agl.typecheck import AglTypeError

        if not isinstance(type_expr, NameT):
            return None
        try:
            if type_expr.module_qualifier is None:
                resolved = type_env.resolve_unapplied_generic_type(
                    type_expr.name,
                    span=type_expr.span,
                )
            else:
                resolved = type_env.resolve_qualified_unapplied_generic_type(
                    type_expr.module_qualifier,
                    type_expr.name,
                    span=type_expr.span,
                )
        except AglTypeError:
            return None
        if resolved is None:
            return None
        display_name, gdef = resolved
        return EntryResult(
            kind="type",
            name=None,
            value=None,
            value_type=None,
            type_display=format_generic_type_def_for_repl(
                display_name, gdef, type_env.type_table
            ),
            diagnostics=[],
            warnings=[],
            error=None,
            ok=True,
        )

    def _build_type_entry_graph_env(self) -> "TypeEnvironment | None":
        """Build a throwaway graph-aware type env for std/imported type entries."""
        from agm.agl.modules.errors import (
            AmbiguousModule,
            ImportEntryError,
            ModuleNotFound,
            ModulePrefixNotFound,
        )
        from agm.agl.modules.ids import ENTRY_ID
        from agm.agl.modules.loader import build_repl_graph
        from agm.agl.parser import AglSyntaxError, parse_program_seeded
        from agm.agl.scope import AglScopeError
        from agm.agl.scope.graph import resolve_graph
        from agm.agl.typecheck import AglTypeError
        from agm.agl.typecheck.graph import check_graph

        host_env = self._runtime.host_environment()
        try:
            program, next_start_id = parse_program_seeded(
                "()",
                start_id=self._next_node_id,
                ambient_infix=self._accumulated_infix,
            )
            program = self._graph_session._inject_accumulated_imports(program)
            graph, _new_next_id, _new_modules = build_repl_graph(
                program,
                next_start_id,
                path=None,
                cached=self._loaded_lib_modules,
                roots=self._ensure_roots(),
            )
            rgraph = resolve_graph(
                graph,
                ambient_agents=self._ambient_agents(host_env),
                entry_ambient_constructor_candidates=self._ambient_constructor_candidates,
                entry_ambient_type_names=self._ambient_type_names,
                entry_parent_scope=self._session_scope,
                entry_repl_session_scope=self._session_scope,
            )
            cgraph = check_graph(
                rgraph,
                host_env.capabilities,
                entry_seed_env=self._type_env,
            )
        except (
            AglSyntaxError,
            AglScopeError,
            AglTypeError,
            ModuleNotFound,
            AmbiguousModule,
            ModulePrefixNotFound,
            ImportEntryError,
        ):
            return None
        return cgraph.modules[ENTRY_ID].type_env

    def _eval_entry_pipeline(self, text: str, *, check_only: bool = False) -> EntryResult:
        """Parse → resolve → check → (eval) one entry against the session (core).

        Completed runtime initializers are promoted even when a later initializer
        fails. ``check_only`` runs the static pipeline without linking, executing,
        promoting, or advancing the node-id counter.
        """
        from agm.agl.lexer import tab_warning_collector
        from agm.agl.parser import AglSyntaxError, parse_program_seeded

        host_env = self._runtime.host_environment()

        # TAB advisories come from the parse's single lex pass (no separate TAB
        # scan).  The collector is populated even on a failed parse, so they
        # surface on EVERY return path (mirroring ``PipelineDriver.prepare``).
        # [1] Parse (seeded so node ids stay globally unique across entries).
        with tab_warning_collector() as tab_sink:
            try:
                program, next_start_id = parse_program_seeded(
                    text, start_id=self._next_node_id, ambient_infix=self._accumulated_infix
                )
            except AglSyntaxError as exc:
                return self._fail([exc.to_diagnostic()], list(tab_sink))
        tab_warnings: list[Diagnostic] = list(tab_sink)

        # [1b] REPL trailing-binder synthesis: in AgL, a block ending in a
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

        # [1d] REPL entries use the graph-mode pipeline by default because that
        # is where the synthetic ``import std.core`` prelude is injected.  This
        # keeps the REPL aligned with ``agm exec``: stdlib names are open unless
        # a host explicitly opts out.
        return self._graph_session.eval_entry_graph_mode(
            text=text,
            orig_program=orig_program,
            pipeline_program=pipeline_program,
            host_env=host_env,
            tab_warnings=tab_warnings,
            next_start_id=next_start_id,
            check_only=check_only,
        )

    # ------------------------------------------------------------------
    # eval_entry helpers
    # ------------------------------------------------------------------

    def _repl_wrap_trailing_binder(
        self, program: "Program", next_start_id: int
    ) -> "tuple[Program, int]":
        """Append a synthetic ``UnitLit`` when the entry ends with a trailing binder.

        In AgL, the type checker rejects a block whose last item is a ``let`` or
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

    def _build_config_base(
        self, effective_config: "dict[str, object]"
    ) -> "dict[str, Value]":
        """Build the ``config_base`` dict for one entry.

        Merges (lowest → highest precedence):
        1. Engine defaults for all six keys (from :func:`build_engine_config_base`).
        2. The session's ``_engine_base`` (exec-config values for the six keys
           provided by the host command at construction; supplies runner/log/log-file
           and the raw-string timeout, e.g. ``"30s"``).
        3. The session's current persisted live settings for ``strict-json`` and
           ``max-iters``, which may have been advanced by prior config declarations.
           NOTE: ``timeout`` is intentionally excluded from step 3 — its base stays
           as-is from ``engine_base``, preserving the raw written value (e.g. "30s"
           not "30.0"). The timeout EFFECT chains correctly via the promoted float
           stored in ``_shell_exec_timeout``.
        4. Program-specific overrides from ``effective_config`` (the raw
           ``[<program>]`` table).  Any engine key present there overrides the
           base from steps 1–3.
        """
        from agm.agl.runtime.params import build_engine_config_base, convert_config_value
        from agm.agl.semantics.engine_keys import ENGINE_KEY_NAMES, get_engine_key_type
        from agm.agl.semantics.values import BoolValue, IntValue

        # Step 1+2: start from engine defaults, then overlay the host engine_base.
        # Engine defaults cover missing keys (e.g. in tests that omit engine_base);
        # engine_base values (from commands/repl.py) override the defaults for the
        # keys it provides, including the raw-string timeout.
        config_base: dict[str, Value] = build_engine_config_base({})
        config_base.update(self._engine_base)

        # Step 3: override the two session-tracked live keys so prior
        # ``config strict-json = true`` or ``config max-iters = N`` entries chain
        # their effect-at-binding into subsequent entries.
        config_base["strict-json"] = BoolValue(self._default_strict_json)
        # When the valve is OFF (None) leave the engine default floor in place;
        # otherwise override with the session-tracked cap.
        if self._default_loop_limit is not None:
            config_base["max-iters"] = IntValue(self._default_loop_limit)

        # Step 4: apply program-specific overrides from [<program>].KEY.
        for key_name in ENGINE_KEY_NAMES:
            raw = effective_config.get(key_name)
            if raw is not None:
                key_type = get_engine_key_type(key_name)
                assert key_type is not None
                config_base[key_name] = convert_config_value(
                    key_name, raw, key_type, self._type_env.type_table
                )

        return config_base

    def _update_engine_settings(
        self,
        *,
        strict_json: bool,
        loop_limit: int | None,
        shell_exec_timeout: float | None,
    ) -> None:
        """Persist the three live engine settings after a successful entry.

        Updates the session's persisted defaults AND the internal
        ``PipelineDriver`` so that subsequent entries start with these values.
        Agent/codec registrations on the driver are preserved.
        """
        self._default_strict_json = strict_json
        self._default_loop_limit = loop_limit
        self._shell_exec_timeout = shell_exec_timeout
        self._runtime.update_defaults(
            strict_json=strict_json,
            loop_limit=loop_limit,
            shell_exec_timeout=shell_exec_timeout,
        )

    def _pre_eval_param_check(
        self,
        program: "Program",
        checked: "CheckedProgram",
        warnings: list[Diagnostic],
    ) -> tuple[EntryResult | None, dict[str, Value], str | None, dict[str, object]]:
        """Validate and convert config-backed params without mutating session state."""
        from agm.agl.runtime.params import convert_param_value
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
                            item.name, raw_config, declared_type, checked.type_env.type_table
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
                        f" via [{effective_program_name}] config"
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
            type_table=checked.type_env.type_table,
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
            ConfigDecl,
            EnumDef,
            ExceptionDef,
            FuncDef,
            LetDecl,
            ParamDecl,
            ProgramDecl,
            RecordDef,
            TypeAlias,
            VarDecl,
        )
        from agm.agl.typecheck.env import TypeEnvironment

        entry_root = checked.resolved.root_scope
        named_declarations = (
            AgentDecl,
            ConfigDecl,
            EnumDef,
            ExceptionDef,
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
        # Config-declared names: on partial failure these bindings are NEVER
        # promoted, keeping the readable binding consistent with the un-promoted
        # engine setting (which is only updated on full success).
        config_decl_names: frozenset[str] = frozenset(
            item.name for item in program.body.items if isinstance(item, ConfigDecl)
        )
        def _before_failure(end_offset: int) -> bool:
            return not partial or (
                failure_span is not None and end_offset <= failure_span.start_offset
            )

        entry_type_names = frozenset(
            item.name
            for item in program.body.items
            if isinstance(item, (RecordDef, EnumDef, ExceptionDef, TypeAlias))
        )
        promoted_type_names = frozenset(
            item.name
            for item in program.body.items
            if isinstance(item, (RecordDef, EnumDef, ExceptionDef, TypeAlias))
            and _before_failure(item.span.end_offset)
        )
        unpromoted_type_names = entry_type_names - promoted_type_names
        stale_binding_names: set[str] = set()
        stale_binding_node_ids: set[int] = set()
        if promoted_type_names:
            for name, ref in self._session_scope.bindings.items():
                typ = self._type_env.resolve_binding(ref)
                if typ is not None and self._type_mentions_entry_nominal(typ, promoted_type_names):
                    stale_binding_names.add(name)
                    stale_binding_node_ids.add(ref.decl_node_id)
            for name in stale_binding_names:
                self._session_scope.bindings.pop(name, None)
            self._declared_params = {
                name: typ
                for name, typ in self._declared_params.items()
                if not self._type_mentions_entry_nominal(typ, promoted_type_names)
            }
            self._ambient_constructor_candidates = {
                cname: tuple(ref for ref in crefs if ref.owner_name not in promoted_type_names)
                for cname, crefs in self._ambient_constructor_candidates.items()
            }
            self._ambient_constructor_candidates = {
                cname: crefs
                for cname, crefs in self._ambient_constructor_candidates.items()
                if crefs
            }

        installed: list[str] = []
        for name, ref in entry_root.bindings.items():
            # Config bindings are promoted only on full success (alongside the
            # engine setting), never on partial failure.
            if partial and name in config_decl_names:
                continue
            symbol = self._link_image.symbol_for_decl(ref.decl_node_id)
            declared_before_failure = (
                failure_span is not None
                and ref.decl_span.end_offset <= failure_span.start_offset
            )
            installed_before_failure = symbol in self._ir_base_frame and (
                failure_span is None
                or ref.decl_span.start_offset <= failure_span.start_offset
            )
            promoted_before_failure = installed_before_failure or (
                symbol is None and declared_before_failure
            )
            if not partial or promoted_before_failure:
                self._session_scope.bindings[name] = ref
                if partial and name in entry_names:
                    installed.append(name)
        previous_type_env = TypeEnvironment()
        previous_type_env.seed_from(self._type_env)
        self._type_env.seed_from(checked.type_env)
        if partial and unpromoted_type_names:
            self._type_env.restore_type_names_from(previous_type_env, unpromoted_type_names)
        self._type_env.remove_binding_types(stale_binding_node_ids)

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
        # Promote infix fixity: declarations that parsed (and, on partial
        # failure, were declared before the failing point) persist so the
        # operator is usable in later entries. Fixity is resolved against the
        # already-accumulated table so relative priorities bind correctly.
        from agm.agl.parser import resolve_infix_fixity
        from agm.agl.syntax.nodes import InfixDecl

        promoted_infix = [
            item
            for item in program.body.items
            if isinstance(item, InfixDecl) and _before_failure(item.span.end_offset)
        ]
        if promoted_infix:
            self._accumulated_infix = resolve_infix_fixity(promoted_infix, self._accumulated_infix)
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

    def _classify(self, program: "Program") -> tuple[EntryKind, str | None]:
        """Classify the entry by its last item; return (kind, name)."""
        from agm.agl.syntax.nodes import (
            AgentDecl,
            AssignStmt,
            Binder,
            ConfigDecl,
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
            (
                RecordDef,
                EnumDef,
                TypeAlias,
                ParamDecl,
                ProgramDecl,
                FuncDef,
                AgentDecl,
                ConfigDecl,
            ),
        ):
            return "declaration", last.name
        # AssignStmt → "statement"
        if isinstance(last, AssignStmt):
            return "statement", None
        # Remaining Declaration kinds (defensive fallback).
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
        program, _ = parse_program_seeded(
            text, start_id=self._next_node_id, ambient_infix=self._accumulated_infix
        )
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
        from agm.agl.repl.type_display import format_type_for_repl

        return format_type_for_repl(typ, checked.type_env.type_table)

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
        """Clear ALL session state (symbols, types, values, params, source, ids).

        Restores the three live engine settings (strict-json/max-iters/timeout)
        to their values at session construction, undoing any effect-at-binding
        from ``config`` declarations entered during the session.
        """
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
        # Restore live engine settings to the session's initial defaults so that
        # promoted ``config`` effects from prior entries do not bleed past :reset.
        self._update_engine_settings(
            strict_json=self._initial_strict_json,
            loop_limit=self._initial_loop_limit,
            shell_exec_timeout=self._initial_shell_exec_timeout,
        )
        # Clear module state.
        self._roots = None
        self._loaded_lib_modules = {}
        self._accumulated_imports = []
        self._accumulated_infix = {}
        # Discard the session's extern (Python FFI) registry like every other
        # session-scoped binding: a companion resolves and imports again on
        # its next use, as though the session were new.
        self._runtime.reset_extern_registry()

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
        from agm.agl.parser import AglSyntaxError, parse_program
        from agm.core.fs import read_text
        from agm.util.text import normalize_newlines

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
