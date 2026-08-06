"""UI-free incremental session core for the AgL REPL (``ReplSession``).

``ReplSession`` keeps a **persistent incremental environment**: each entry is
parsed → resolved → typechecked → match-compiled → lowered → evaluated
**exactly once** against accumulated session state (symbols, types,
declarations, runtime values). Agent calls fire exactly once and are never
replayed, because each entry executes ONLY its own statements — references to
earlier bindings read stored runtime ``Value``s.

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
from agm.agl.repl.entry_pipeline import EntryPipeline
from agm.agl.self_validation import self_validation_enabled
from agm.config.engine_keys import HOST_CONSUMED_ENGINE_KEYS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from agm.agl.ir.ids import SymbolId
    from agm.agl.modules.ids import ModuleId
    from agm.agl.modules.loader import LoadedModule
    from agm.agl.modules.roots import RootSet
    from agm.agl.runtime.agents import AgentFn
    from agm.agl.runtime.codec import OutputCodec
    from agm.agl.runtime.host_settings import HostSettingsPolicy
    from agm.agl.scope.symbols import ConstructorRef, ScopeNode
    from agm.agl.semantics.types import Type
    from agm.agl.semantics.values import BoolValue, EnumValue, Frame, Value
    from agm.agl.syntax.nodes import (
        ImportDecl,
        InfixAssoc,
        OpenDecl,
        Program,
        ScopeRegion,
        TypeAlias,
    )
    from agm.agl.syntax.spans import SourceSpan
    from agm.agl.syntax.types import TypeExpr
    from agm.agl.typecheck.env import CheckedModule, TypeEnvironment


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


def _region_path(region: "ScopeRegion") -> tuple[str, ...]:
    """Return the scope path *region* opens, as the entry spelled it.

    A multi-segment header parses into nested single-segment regions, so a
    region whose only item is another region is spelled back as the equivalent
    multi-segment path; a region that also declares members ends the path.
    """
    from agm.agl.syntax.nodes import ScopeRegion

    path = [region.segment.name]
    current = region
    while len(current.items) == 1 and isinstance(current.items[0], ScopeRegion):
        current = current.items[0]
        path.append(current.segment.name)
    return tuple(path)


# ---------------------------------------------------------------------------
# ReplSession — the persistent incremental driver
# ---------------------------------------------------------------------------


class ReplSession:
    """Persistent incremental AgL evaluation session (UI-free core).

    Constructor parameters mirror ``PipelineDriver`` so a host can wire the same
    value-driven agent dispatcher and codec backing.

    Each entry is incrementally linked and executed against a persistent IR base
    frame. Completed effects survive a later runtime failure in the same entry.
    """

    def __init__(
        self,
        *,
        default_strict_json: bool = False,
        default_loop_limit: int | None = None,
        default_call_depth_limit: int | None = None,
        agent_dispatcher: "AgentFn | None" = None,
        shell_exec_timeout: float | None = None,
        trace_path: "Path | None" = None,
        params_config_loader: "Callable[[str], dict[str, object]] | None" = None,
        engine_base: "Mapping[str, Value] | None" = None,
        host_settings_policy: "HostSettingsPolicy | None" = None,
        cwd: "Path | None" = None,
        stdlib_root: "Path | None" = None,
        lib_root: "Path | None" = None,
        configured_roots: "Iterable[tuple[str, Path]]" = (),
        extra_cli_roots: "Iterable[str]" = (),
        default_stdlib: bool = True,
    ) -> None:
        from pathlib import Path

        from agm.agl.lower import LinkImage
        from agm.agl.pipeline import PipelineDriver
        from agm.agl.runtime.option import some_value
        from agm.agl.scope.symbols import ScopeNode
        from agm.agl.semantics.values import IntValue, TextValue
        from agm.agl.typecheck.env import TypeEnvironment
        from agm.config.module_roots import resolve_stdlib_root
        from agm.core.parse import format_timeout

        self._default_strict_json = default_strict_json
        self._default_stdlib = default_stdlib
        # ``_engine_base`` is the single seed representation for all six engine
        # keys, supplied by the host command (commands/repl.py). Fold the two
        # scalar constructor arguments that would otherwise seed
        # ``max-iters``/``timeout`` through a side channel into it, but only
        # where the host did not seed the key directly: an explicit
        # ``engine_base`` entry keeps the precedence a seed has over a driver
        # argument for every other key. Presence, not truthiness, is what makes
        # a key host-seeded (see :meth:`_has_host_seed`), so a false
        # ``strict-json`` seed still counts as a host control.
        # ``default_strict_json`` is deliberately NOT folded in: a bare
        # ``False`` must stay a driver floor (``_initial_strict_json`` below),
        # or no declared ``std/config`` default could ever be recorded for it.
        self._engine_base: dict[str, Value] = dict(engine_base) if engine_base is not None else {}
        if "max-iters" not in self._engine_base and default_loop_limit is not None:
            self._engine_base["max-iters"] = IntValue(default_loop_limit)
        if "timeout" not in self._engine_base and shell_exec_timeout is not None:
            self._engine_base["timeout"] = some_value(TextValue(format_timeout(shell_exec_timeout)))
        # The live per-entry loop cap follows the normalized seed, so an
        # explicit ``engine_base["max-iters"]`` wins over a differing
        # ``default_loop_limit`` from the first entry onward, not only after
        # :reset. No declared default can exist yet, hence ``fallback=None``.
        self._default_loop_limit: int | None = self._resolve_loop_limit(fallback=None)
        # Driver floor applied only when NEITHER a host seed NOR a declared
        # ``std/config`` default exists for ``strict-json``; unlike
        # ``max-iters``/``timeout`` it is never promoted into ``_engine_base``
        # (see :meth:`_reset_live_engine_settings`).
        self._initial_strict_json = default_strict_json
        # Effective declaration defaults learned when an entry loads ``std/config``.
        # These survive :reset so a bare later entry still starts with the same
        # setting values, while explicit host seeds in ``_engine_base`` retain
        # precedence over them. Empty here, so ``_effective_seed`` below
        # degenerates to the host seed alone -- the shared precedence rule that
        # :meth:`reset` reapplies once declarations have populated this dict.
        self._declared_engine_defaults: dict[str, Value] = {}
        self._persisted_strict_json = self._strict_json_seed()
        # Persisted register values for the three host-consumed engine settings
        # (default-agent/log/log-file). Seeded from explicit host values,
        # threaded into every entry's interpreter, and read
        # back after a successful entry so a ``std/config::KEY := VALUE`` write
        # persists.
        self._persisted_host_settings: dict[str, Value] = self._host_settings_seed()
        self._persisted_timeout_setting = self._timeout_seed()
        # The session's live timeout follows the normalized seed rather than
        # the raw argument. Its only reader (:mod:`agm.agl.repl.entry_pipeline`)
        # consults it just when ``_persisted_timeout_setting`` is ``None``,
        # which the fold above allows only when the raw argument was ``None``
        # too, so the two agree wherever this field is observable.
        self._shell_exec_timeout = self._resolve_timeout_seed(self._persisted_timeout_setting)
        self._host_settings_policy = host_settings_policy
        # Trace destination: when set, each evaluated entry opens a fresh
        # ``TraceStore`` (its own ``run_id``) appending JSONL records to this one
        # file.  ``check_only`` entries write nothing (mirroring ``agm exec``).
        # The COMMAND validates/creates the path up front; the session assumes it
        # is writable but the no-op store tolerates failure (it disables itself).
        self._trace_path = trace_path
        self._initial_trace_path = trace_path
        self._params_config_loader = params_config_loader

        # Internal runtime owns the registrations + host-environment assembly.
        # It never runs an entry on the session's behalf, so it is given none
        # of the three live engine settings: the session owns those (above) and
        # threads them into each per-entry interpreter directly (see
        # :mod:`agm.agl.repl.entry_pipeline`).
        self._runtime = PipelineDriver(
            default_call_depth_limit=default_call_depth_limit,
            agent_dispatcher=agent_dispatcher,
        )
        # Reuse the driver's resolved (default-applied) limit for the per-entry
        # interpreters this session builds directly, so the canonical default
        # lives in exactly one place.
        self._default_call_depth_limit = self._runtime.default_call_depth_limit
        # Persistent session environment.
        self._session_scope: ScopeNode = ScopeNode(node_id=-1, parent=None)
        self._session_scope_nodes: dict[tuple[str, ...], ScopeNode] = {(): self._session_scope}
        # Each retained type-owned path maps to its rendered alias target, or
        # None for a nominal type.
        self._session_type_paths: dict[tuple[str, ...], str | None] = {}
        self._type_env: TypeEnvironment = TypeEnvironment()
        self._type_env.seal()
        self._link_image = LinkImage()
        self._ir_base_frame: Frame = {}
        self._next_node_id: int = 0
        self._program_name: str | None = None
        self._active_config: dict[str, object] = {}
        # Keyed by external key (full path spelling for a scoped param, bare
        # name for a root param); value is (declared type, declaration node id).
        self._declared_params: dict[str, tuple[Type, int]] = {}
        # Source log of successfully-promoted entries (for dump_source / :save).
        self._source_log: list[str] = []
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
        # Cached lib modules from prior REPL program entries.
        self._loaded_lib_modules: dict[ModuleId, LoadedModule] = {}
        # Imports generally persist across successfully promoted program entries,
        # one retained generation per entry, kept as written so a wildcard keeps
        # tracking the module set. Declarations for a module named in a later
        # generation replace earlier ones; the rest are prepended in program
        # context for reuse. Scope opens use the same entry retention model.
        self._accumulated_imports: list[tuple["ImportDecl", ...]] = []
        self._accumulated_opens: list[tuple["OpenDecl | ImportDecl | ScopeRegion", ...]] = []
        # Resolved user infix fixity declared in prior promoted entries
        # (operator name → ``(priority, associativity)``). Passed to the parser
        # as ambient fixity so an ``infixl``/``infixr`` declaration made in one
        # entry makes the operator usable in later entries.
        self._accumulated_infix: dict[str, tuple[int, "InfixAssoc"]] = {}
        self._entry_pipeline = EntryPipeline(self)

    @staticmethod
    def _assert_checked_state_closed(checked: "CheckedModule") -> None:
        """Assert that a checked entry satisfies the shared lowering boundary."""
        from agm.agl.typecheck.env import assert_checked_module_closed

        assert_checked_module_closed(checked)

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
        """Parse → resolve → typecheck → matchcompile → lower/eval one entry.

        Completed runtime initializers are promoted even when a later initializer
        fails. ``check_only`` stops after match compilation without lowering,
        executing, promoting, or advancing the node-id counter.

        REPL-only fallback: when the entry fails to evaluate as a program, the
        loop tries to read it as a bare type expression (e.g. ``int``, a declared
        enum/record name, a bare generic type definition, ``array[T]``).  If
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
        type is not a value expression, so without it the entry would surface
        ``'X' is not defined.``.  Like :meth:`type_of`, this never evaluates,
        promotes, advances the node-id counter, or mutates session state.  The parse uses
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
        program_type_env = self._build_type_entry_program_env()
        if program_type_env is not None:
            type_envs.append(program_type_env)

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
            if type_expr.qualifier is None:
                resolved = type_env.resolve_unapplied_generic_type(
                    type_expr.name,
                    span=type_expr.span,
                )
            else:
                resolved = type_env.resolve_qualified_unapplied_generic_type(
                    type_expr.qualifier,
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
            type_display=format_generic_type_def_for_repl(display_name, gdef, type_env.type_table),
            diagnostics=[],
            warnings=[],
            error=None,
            ok=True,
        )

    def _build_type_entry_program_env(self) -> "TypeEnvironment | None":
        """Build a throwaway program-level type env for std/imported type entries."""
        from agm.agl.modules.errors import (
            AmbiguousModule,
            ImportEntryError,
            ModuleNotFound,
            ModulePrefixNotFound,
        )
        from agm.agl.modules.ids import ENTRY_ID
        from agm.agl.parser import AglSyntaxError, parse_program_seeded
        from agm.agl.scope import AglScopeError
        from agm.agl.typecheck import AglTypeError

        host_env = self._runtime.host_environment()
        try:
            program, next_start_id = parse_program_seeded(
                "()",
                start_id=self._next_node_id,
                ambient_infix=self._accumulated_infix,
            )
            checked_program = self._entry_pipeline.resolve_and_check_program(
                program, next_start_id, host_env
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
        return checked_program.modules[ENTRY_ID].type_env

    def _eval_entry_pipeline(self, text: str, *, check_only: bool = False) -> EntryResult:
        """Run the resolve → typecheck → matchcompile → lower/eval entry core.

        Completed runtime initializers are promoted even when a later initializer
        fails. ``check_only`` stops after match compilation without lowering,
        executing, promoting, or advancing the node-id counter.
        """
        from agm.agl.lexer import spaced_qualifier_collector, tab_warning_collector
        from agm.agl.parser import AglSyntaxError, parse_program_seeded

        host_env = self._runtime.host_environment()

        # TAB advisories come from the parse's single lex pass (no separate TAB
        # scan).  The collector is populated even on a failed parse, so they
        # surface on EVERY return path (mirroring ``PipelineDriver.prepare``).
        # [1] Parse (seeded so node ids stay globally unique across entries).
        with tab_warning_collector() as tab_sink, spaced_qualifier_collector() as spaced_sink:
            try:
                program, next_start_id = parse_program_seeded(
                    text, start_id=self._next_node_id, ambient_infix=self._accumulated_infix
                )
            except AglSyntaxError as exc:
                return self._fail([exc.to_diagnostic()], list(tab_sink))
        tab_warnings: list[Diagnostic] = list(tab_sink)
        spaced_qualifiers = tuple(spaced_sink)

        # [1d] REPL entries use the program pipeline by default because that
        # is where the synthetic ``import std/core`` prelude is injected.  This
        # keeps the REPL aligned with ``agm exec``: stdlib names are open unless
        # a host explicitly opts out.
        return self._entry_pipeline.eval_entry(
            text=text,
            orig_program=program,
            pipeline_program=program,
            host_env=host_env,
            tab_warnings=tab_warnings,
            next_start_id=next_start_id,
            check_only=check_only,
            spaced_qualifiers=spaced_qualifiers,
        )

    def _fail(self, diagnostics: list[Diagnostic], warnings: list[Diagnostic]) -> EntryResult:
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

    def _host_seed(self, key: str) -> "Value | None":
        """Return the explicit host control for *key*, if the host supplied one.

        ``_engine_base`` is the single seed representation, so every key reads
        straight from it with no per-key special case: the constructor folds
        ``default_loop_limit`` and ``shell_exec_timeout`` into it rather than
        leaving them to be consulted separately here.
        """
        return self._engine_base.get(key)

    def _effective_seed(self, key: str) -> "Value | None":
        """Return the host seed if there is one, else the remembered declaration default.

        At construction ``_declared_engine_defaults`` is always empty, so this
        degenerates to :meth:`_host_seed` -- the one reason the initial and
        :meth:`reset` seeding can share this helper.
        """
        seed = self._host_seed(key)
        return seed if seed is not None else self._declared_engine_defaults.get(key)

    def _strict_json_seed(self) -> "BoolValue | None":
        """Return the effective strict-json seed: host control, else declared default."""
        from agm.agl.semantics.values import BoolValue

        seed = self._effective_seed("strict-json")
        assert seed is None or isinstance(seed, BoolValue)
        return seed

    def _timeout_seed(self) -> "EnumValue | None":
        """Return the effective timeout seed: host control, else declared default."""
        from agm.agl.semantics.values import EnumValue

        seed = self._effective_seed("timeout")
        assert seed is None or isinstance(seed, EnumValue)
        return seed

    @staticmethod
    def _resolve_timeout_seed(seed: "EnumValue | None") -> float | None:
        """Unwrap a ``timeout`` seed (``Option[text]``) into seconds, or ``None``.

        ``None`` covers both an absent seed and an explicit ``None`` variant (a
        host or declared "no timeout" control). Shared by construction and
        :meth:`_reset_live_engine_settings` so a timeout ``Option`` is unwrapped
        in exactly one place.
        """
        from agm.agl.semantics.values import TextValue
        from agm.core.parse import parse_timeout

        if seed is None or seed.variant != "Some":
            return None
        timeout_value = seed.fields["value"]
        assert isinstance(timeout_value, TextValue)
        return parse_timeout(timeout_value.value)

    def _host_settings_seed(self) -> "dict[str, Value]":
        """Return the effective seed for each host-consumed (log/log-file/default-agent) key.

        Host seeds win over remembered declared defaults (the shared
        precedence), and the merge is then filtered to the host-consumed keys:
        ``_engine_base`` also carries the runtime-live keys
        (``strict-json``/``max-iters``/``timeout``), which have no register and
        must never enter this seed. A key with neither a seed nor a remembered
        default stays absent until the interpreter has evaluated the ``builtin
        var`` declaration's default; filling it with a host-side floor here
        would incorrectly override that declared default.
        """
        merged = {**self._declared_engine_defaults, **self._engine_base}
        return {key: value for key, value in merged.items() if key in HOST_CONSUMED_ENGINE_KEYS}

    def _record_declared_engine_defaults(
        self, declared_keys: frozenset[str], effective_values: "Mapping[str, Value]"
    ) -> None:
        """Remember unseeded declared defaults before an entry can write them.

        The REPL creates a fresh interpreter for each entry.  Its constructor
        is the one place that evaluates a ``builtin var`` initializer, so the
        entry pipeline gives us its initial effective values before evaluation.
        Keeping them separately from mutable persisted settings lets :reset
        discard source writes without replacing a declaration with a hard-coded
        host fallback.
        """
        for key in declared_keys:
            if not self._has_host_seed(key):
                self._declared_engine_defaults.setdefault(key, effective_values[key])

    def _has_host_seed(self, key: str) -> bool:
        """Return whether *key* has an explicit host control over its declaration."""
        return key in self._engine_base

    def _resolve_loop_limit(self, *, fallback: int | None) -> int | None:
        """Return the effective max-iters cap: the host seed if there is one, else *fallback*.

        Shared by construction (``fallback=None``, since no declared default
        can exist yet) and :meth:`_reset_live_engine_settings` (``fallback`` =
        the remembered declared default, 0-collapsed by that caller), so the
        host-seed-wins precedence is expressed once. A host-seeded ``0`` is
        returned as an explicit ``0`` rather than collapsed to ``None``: it
        means "cap disabled", and ``None`` would instead let a declared nonzero
        default reassert itself when the next interpreter is built.
        """
        from agm.agl.semantics.values import IntValue

        loop_seed = self._host_seed("max-iters")
        if loop_seed is None:
            return fallback
        assert isinstance(loop_seed, IntValue)
        return loop_seed.value

    def _reset_live_engine_settings(self) -> tuple[bool, int | None, float | None]:
        """Return the reset values for settings backed by interpreter fields.

        Strict-json and timeout reapply the shared host-seed-else-declared-
        default precedence via :meth:`_strict_json_seed` / :meth:`_timeout_seed`.
        Max-iters reapplies it via :meth:`_resolve_loop_limit`, which keeps a
        host-seeded ``0`` (an explicit disable) rather than collapsing it the
        way a declared ``0`` is collapsed here.
        """
        from agm.agl.semantics.values import IntValue

        strict_seed = self._strict_json_seed()
        strict_json = self._initial_strict_json if strict_seed is None else strict_seed.value

        declared_limit = self._declared_engine_defaults.get("max-iters")
        declared_fallback = None
        if declared_limit is not None:
            assert isinstance(declared_limit, IntValue)
            declared_fallback = declared_limit.value or None
        loop_limit = self._resolve_loop_limit(fallback=declared_fallback)

        shell_exec_timeout = self._resolve_timeout_seed(self._timeout_seed())
        return strict_json, loop_limit, shell_exec_timeout

    def _update_engine_settings(
        self,
        *,
        strict_json: bool,
        loop_limit: int | None,
        shell_exec_timeout: float | None,
    ) -> None:
        """Persist the three live engine settings an entry ended with.

        Updates the session's persisted defaults so that subsequent entries
        start with these values.  Setting writes are non-transactional, so a
        failed or cancelled entry persists the writes it completed; :meth:`reset`
        calls this too, with the seed values it re-derived.
        """
        self._default_strict_json = strict_json
        if (
            self._persisted_strict_json is not None
            or "strict-json" in self._declared_engine_defaults
        ):
            from agm.agl.semantics.values import BoolValue

            self._persisted_strict_json = BoolValue(strict_json)
        self._default_loop_limit = loop_limit
        self._shell_exec_timeout = shell_exec_timeout

    def _pre_eval_param_check(
        self,
        program: "Program",
        checked: "CheckedModule",
        warnings: list[Diagnostic],
    ) -> tuple[EntryResult | None, dict[str, Value], str | None, dict[str, object]]:
        """Validate and convert config-backed params without mutating session state."""
        from agm.agl.runtime.params import convert_param_value
        from agm.agl.syntax.nodes import ParamDecl, ProgramDecl, scoped_public_name, static_items

        def reject(message: str, span: "SourceSpan") -> EntryResult:
            return self._fail([diagnostic_from_span(message, span)], warnings)

        param_values: dict[str, Value] = {}
        entry_program_name: str | None = None
        effective_config = self._active_config

        for item in static_items(program.body.items):
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
                external_name = scoped_public_name(item.scope_path, item.name)
                raw_config = effective_config.get(external_name)
                if raw_config is not None:
                    declared_type = checked.type_env.get_binding_type(item.node_id)
                    assert declared_type is not None
                    try:
                        param_values[external_name] = convert_param_value(
                            external_name, raw_config, declared_type, checked.type_env.type_table
                        )
                    except (TypeError, ValueError) as exc:
                        return (
                            reject(
                                f"Config value for param {external_name!r} is invalid: {exc}",
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
                            f"Missing required param {external_name!r}: provide it"
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
        checked: "CheckedModule",
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

    def _advance_node_ids(self, next_start_id: int) -> None:
        """Consume node ids for an entry that failed after lowering began."""
        self._next_node_id = next_start_id

    def _promote_ir_state(
        self,
        *,
        text: str,
        program: "Program",
        checked: "CheckedModule",
        next_start_id: int,
        entry_program_name: str | None,
        entry_active_config: dict[str, object],
        partial: bool,
        promoted_declaration_ids: frozenset[int],
    ) -> tuple[str, ...]:
        """Promote declarations whose IR initialization completed in this entry."""
        from agm.agl.parser import resolve_infix_fixity
        from agm.agl.scope.symbols import ScopeNode
        from agm.agl.syntax.nodes import (
            EnumDef,
            ExceptionDef,
            FuncDef,
            InfixDecl,
            LetDecl,
            ParamDecl,
            ProgramDecl,
            RecordDef,
            TypeAlias,
            VarDecl,
            pattern_binder_candidates,
            resolved_public_name,
            scoped_public_name,
            static_items,
        )
        from agm.agl.syntax.types import render_type_expr
        from agm.agl.typecheck.env import TypeEnvironment

        entry_declarations = tuple(static_items(program.body.items))
        if self_validation_enabled():
            self._assert_checked_state_closed(checked)
        entry_root = checked.resolved.root_scope
        named_declarations = (
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
        binding_items = (FuncDef, ParamDecl, VarDecl)
        promotion_bindings = {
            item.name: entry_root.bindings[item.name]
            for item in program.body.items
            if isinstance(item, binding_items)
            and not item.scope_path
            and item.name in entry_root.bindings
        }
        promotion_bindings.update(
            (binding.name, binding)
            for item in program.body.items
            if isinstance(item, LetDecl) and not item.scope_path
            for candidate in pattern_binder_candidates(item.pattern)
            if (binding := checked.pattern_binding_for(candidate.node_id)) is not None
        )
        entry_binding_node_ids = {ref.decl_node_id for ref in promotion_bindings.values()}
        promoted_binding_node_ids = entry_binding_node_ids & promoted_declaration_ids

        # A region-form ``let``'s scope-member ref keys on its binder
        # candidate node id (not the ``LetDecl`` node's own id), same as at
        # the root. Only root binders feed ``promotion_bindings`` — a scoped
        # one must not, since that also drives the root-only
        # ``self._session_scope.bindings`` update below — so its candidate
        # ids are collected separately and only fed into
        # ``entry_declaration_node_ids``.
        scoped_binder_node_ids = {
            candidate.node_id
            for item in static_items(program.body.items)
            if isinstance(item, LetDecl)
            for candidate in pattern_binder_candidates(item.pattern)
        }

        # Declarations this entry introduces, at the root and in its scope
        # regions. Anything outside this set is retained session state, which a
        # partial entry never demotes.
        entry_declaration_node_ids = (
            {item.node_id for item in entry_declarations if isinstance(item, named_declarations)}
            | entry_binding_node_ids
            | scoped_binder_node_ids
        )

        def _is_promoted(node_id: int) -> bool:
            return node_id not in entry_declaration_node_ids or node_id in promoted_declaration_ids

        def type_name_path(
            item: RecordDef | EnumDef | ExceptionDef | TypeAlias,
        ) -> tuple[tuple[str, ...], str]:
            return tuple(segment.name for segment in item.scope_path), item.name

        entry_type_items = tuple(
            item
            for item in entry_declarations
            if isinstance(item, (RecordDef, EnumDef, ExceptionDef, TypeAlias))
        )
        entry_type_name_paths = frozenset(type_name_path(item) for item in entry_type_items)
        promoted_type_name_paths = frozenset(
            type_name_path(item)
            for item in entry_type_items
            if item.node_id in promoted_declaration_ids
        )
        unpromoted_type_name_paths = entry_type_name_paths - promoted_type_name_paths
        unpromoted_type_names = {
            "::".join((*path, name)) for path, name in unpromoted_type_name_paths
        }
        if promoted_type_name_paths:
            # A promoted type declaration supersedes any earlier ambient
            # constructor candidate sharing its name path: retained bindings
            # and scope members keep resolving through their own (possibly
            # superseded) declaration identity, so only the ambient bare-name
            # candidate table -- which drives how a FRESH constructor
            # reference resolves -- needs to move onto the newest owner here.
            self._ambient_constructor_candidates = {
                cname: tuple(
                    ref
                    for ref in crefs
                    if (ref.owner_path, ref.owner_name) not in promoted_type_name_paths
                )
                for cname, crefs in self._ambient_constructor_candidates.items()
            }
            self._ambient_constructor_candidates = {
                cname: crefs
                for cname, crefs in self._ambient_constructor_candidates.items()
                if crefs
            }

        # External keys of params this entry's promotions displace (a `let` /
        # `var` / `def` / `agent` binding that shares a param's public name
        # takes over that name, so the param must stop being a declared
        # param); populated by both the root-binding loop here and the
        # scoped-member loop below, then applied to ``_declared_params`` in
        # one pass.
        displaced_param_keys: set[str] = set()

        for name, ref in promotion_bindings.items():
            if ref.decl_node_id not in promoted_binding_node_ids:
                continue
            displaced_param_keys.add(name)
            self._session_scope.bindings[name] = ref
        installed = (
            self._installed_report(
                program,
                checked,
                promoted_binding_node_ids=promoted_binding_node_ids,
                promoted_type_names=frozenset(
                    name for path, name in promoted_type_name_paths if not path
                ),
            )
            if partial
            else []
        )

        # Named scope paths are namespaces: a region's path is retained whenever
        # it is not a demoted type's own path, and its members are promoted
        # individually below.
        for path, node in checked.resolved.scope_nodes.items():
            if not path or path in self._session_scope_nodes:
                continue
            if path in {(*scope_path, name) for scope_path, name in unpromoted_type_name_paths}:
                continue
            self._session_scope_nodes[path] = ScopeNode(
                node_id=node.node_id,
                parent=self._session_scope_nodes[path[:-1]],
                scope_path=path,
            )
        for path, node in checked.resolved.scope_nodes.items():
            session_node = self._session_scope_nodes.get(path)
            if session_node is None:
                continue
            for name, ref in node.members.items():
                if (ref.scope_path, ref.name) not in unpromoted_type_name_paths and _is_promoted(
                    ref.decl_node_id
                ):
                    if ref.decl_node_id in entry_declaration_node_ids:
                        displaced_param_keys.add(resolved_public_name(path, name))
                    session_node.register_member(name, ref)
        alias_targets = {
            type_name_path(item): render_type_expr(item.type_expr)
            for item in entry_type_items
            if isinstance(item, TypeAlias) and item.node_id in promoted_declaration_ids
        }
        self._session_type_paths.update(
            ((*path, name), alias_targets.get((path, name)))
            for path, name in promoted_type_name_paths
        )

        if not partial:
            # The checked environment already includes the prior sealed session
            # state and is itself sealed at the checked-output boundary. Reuse it
            # directly instead of copying the accumulated session a second time.
            self._type_env = checked.type_env
        else:
            previous_type_env = TypeEnvironment()
            previous_type_env.seed_from(self._type_env)
            # Build the replacement env in a local so a mid-promotion failure
            # leaves the session's still-sealed ``self._type_env`` untouched.
            new_type_env = TypeEnvironment()
            new_type_env.seed_from(checked.type_env)
            if unpromoted_type_names:
                new_type_env.restore_type_names_from(previous_type_env, unpromoted_type_names)
            unpromoted_function_names = (
                item.name
                for item in entry_declarations
                if isinstance(item, FuncDef) and item.node_id not in promoted_declaration_ids
            )
            new_type_env.restore_binding_metadata_from(
                previous_type_env,
                entry_binding_node_ids - promoted_binding_node_ids,
                unpromoted_function_names,
            )
            new_type_env.seal()
            self._type_env = new_type_env

        if promoted_type_name_paths:
            promoted_candidates: dict[str, tuple[ConstructorRef, ...]] = {}
            for (_path, cname), crefs in checked.resolved.constructor_candidates_by_path.items():
                selected = tuple(
                    ref
                    for ref in crefs
                    if (ref.owner_path, ref.owner_name) in promoted_type_name_paths
                )
                if selected:
                    promoted_candidates[cname] = (*promoted_candidates.get(cname, ()), *selected)
            for cname, crefs in promoted_candidates.items():
                self._ambient_constructor_candidates[cname] = (
                    *self._ambient_constructor_candidates.get(cname, ()),
                    *crefs,
                )
            self._ambient_type_names |= frozenset(
                name for path, name in promoted_type_name_paths if not path
            )
        for key in displaced_param_keys:
            self._declared_params.pop(key, None)
        for item in static_items(program.body.items):
            if isinstance(item, ParamDecl) and _is_promoted(item.node_id):
                typ = checked.type_env.get_binding_type(item.node_id)
                assert typ is not None
                self._declared_params[scoped_public_name(item.scope_path, item.name)] = (
                    typ,
                    item.node_id,
                )
        if entry_program_name is not None and not partial:
            self._program_name = entry_program_name
            self._active_config = entry_active_config
        if not partial:
            self._source_log.append(text)
        promoted_infix = [
            item
            for item in program.body.items
            if isinstance(item, InfixDecl) and item.node_id in promoted_declaration_ids
        ]
        if promoted_infix:
            self._accumulated_infix = resolve_infix_fixity(promoted_infix, self._accumulated_infix)
        self._next_node_id = next_start_id
        return tuple(installed)

    @staticmethod
    def _installed_report(
        program: "Program",
        checked: "CheckedModule",
        *,
        promoted_binding_node_ids: set[int],
        promoted_type_names: frozenset[str],
    ) -> list[str]:
        """Return every promoted declaration name for a partial entry's report.

        Ordered by the entry's source items: a promoted value binding (agent /
        function / param / var) or a promoted type declaration (record / enum /
        exception / type alias) contributes its declared name; a promoted ``let``
        contributes each selected binder in pattern order. An enum's variant
        names are never listed separately, only the enum's own declared name.
        """
        from agm.agl.syntax.nodes import (
            EnumDef,
            ExceptionDef,
            FuncDef,
            LetDecl,
            ParamDecl,
            RecordDef,
            TypeAlias,
            VarDecl,
            pattern_binder_candidates,
        )

        binding_items = (FuncDef, ParamDecl, VarDecl)
        type_items = (RecordDef, EnumDef, ExceptionDef, TypeAlias)
        installed: list[str] = []
        for item in program.body.items:
            if isinstance(item, binding_items):
                if item.node_id in promoted_binding_node_ids:
                    installed.append(item.name)
            elif isinstance(item, LetDecl):
                for candidate in pattern_binder_candidates(item.pattern):
                    binding = checked.pattern_binding_for(candidate.node_id)
                    if binding is not None and binding.decl_node_id in promoted_binding_node_ids:
                        installed.append(binding.name)
            elif isinstance(item, type_items) and item.name in promoted_type_names:
                installed.append(item.name)
        return installed

    def frame_value(self, symbol: "SymbolId | None") -> "Value | None":
        """Return the base-frame value bound to *symbol*, unwrapping a mutable cell."""
        from agm.agl.semantics.values import Cell

        slot = self._ir_base_frame.get(symbol) if symbol is not None else None
        return slot.value if isinstance(slot, Cell) else slot

    def _declaration_value(self, decl_node_id: int) -> "Value | None":
        """Return the base-frame value of a promoted declaration."""
        return self.frame_value(self._link_image.symbol_for_decl(decl_node_id))

    def _echo_data_ir(
        self, program: "Program", checked: "CheckedModule", captured: "Value | None"
    ) -> tuple["Value | None", "Type | None"]:
        from agm.agl.syntax.nodes import (
            Binder,
            Declaration,
            LetDecl,
            VarDecl,
            simple_let_pattern_name,
        )

        last = program.body.items[-1]
        value_type = self._value_type_of_last(program, checked)
        if not isinstance(last, (Binder, Declaration)):
            return captured, value_type
        if isinstance(last, LetDecl):
            simple_name = simple_let_pattern_name(last.pattern)
            if simple_name is None or simple_name == "_":
                return captured, value_type
            binding = checked.pattern_binding_for(last.pattern.node_id)
            assert binding is not None, "compiler bug: no selected simple-let binding"
            return self._declaration_value(binding.decl_node_id), value_type
        if isinstance(last, VarDecl):
            return self._declaration_value(last.node_id), value_type
        return None, None

    def _classify(self, program: "Program") -> tuple[EntryKind, str | None]:
        """Classify the entry by its last item; return (kind, name)."""
        from agm.agl.modules.ids import spell_scope_path
        from agm.agl.syntax.nodes import (
            AssignStmt,
            Binder,
            Declaration,
            EnumDef,
            ExceptionDef,
            FuncDef,
            LetDecl,
            ParamDecl,
            ProgramDecl,
            RecordDef,
            ScopeRegion,
            TypeAlias,
            VarDecl,
            is_scoped_declaration,
            scoped_public_name,
            simple_let_pattern_name,
        )

        # A parsed program always has at least one item (empty/comment-only
        # source fails parsing earlier).
        last = program.body.items[-1]
        # Bare expression (not a binder or declaration) → "expression"
        if not isinstance(last, (Binder, Declaration, ScopeRegion)):
            return "expression", None
        if isinstance(last, LetDecl):
            name = simple_let_pattern_name(last.pattern)
            if name is None:
                return "binding", None
            if name == "_":
                return "statement", None
            # A shorthand binder path names the member it declares, so the
            # echo shows the member the way its scope makes it reachable.
            return "binding", scoped_public_name(last.scope_path, name)
        if isinstance(last, VarDecl):
            if last.name == "_":
                return "statement", None
            return "binding", scoped_public_name(last.scope_path, last.name)
        if isinstance(
            last,
            (
                RecordDef,
                EnumDef,
                ExceptionDef,
                TypeAlias,
                ParamDecl,
                ProgramDecl,
                FuncDef,
            ),
        ):
            # A shorthand declaration path names the member it declares, so the
            # echo shows the member the way its scope makes it reachable.
            if is_scoped_declaration(last):
                return "declaration", scoped_public_name(last.scope_path, last.name)
            return "declaration", last.name
        if isinstance(last, ScopeRegion):
            return "declaration", spell_scope_path(_region_path(last))
        # AssignStmt → "statement"
        if isinstance(last, AssignStmt):
            return "statement", None
        # Import, export, open, infix, and builtin declarations name nothing the
        # echo can confirm, so they read as statements.
        return "statement", None

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

    def _value_type_of_last(self, program: "Program", checked: "CheckedModule") -> "Type | None":
        """Static type carried by the entry's final value, or ``None``.

        A bare expression retains its checked type. A trailing ``let``/``var``
        reports the declared binding type for the REPL declaration echo, except
        that an initializer which always exits reports ``bottom``. Shared by the
        check-only result builder and the success echo so the two agree.
        """
        from agm.agl.syntax.nodes import (
            Binder,
            Declaration,
            LetDecl,
            VarDecl,
            simple_let_pattern_name,
        )

        # A parsed program always has at least one item (empty/comment-only
        # source fails parsing earlier).
        last = program.body.items[-1]
        # Bare expression → node type from checked side table
        if not isinstance(last, (Binder, Declaration)):
            # After narrowing: last is an Expr (not a Binder or Declaration).
            return checked.node_types.get(last.node_id)
        if isinstance(last, (LetDecl, VarDecl)):
            from agm.agl.semantics.types import BottomType

            initializer_type = checked.node_types.get(last.value.node_id)
            if isinstance(initializer_type, BottomType):
                return initializer_type
            if isinstance(last, LetDecl):
                if simple_let_pattern_name(last.pattern) == "_":
                    return None
                return checked.let_matched_types.get(last.node_id)
            if last.name == "_":
                return None
            return checked.type_env.get_binding_type(last.node_id)
        return None

    # ------------------------------------------------------------------
    # type_of — type without evaluation
    # ------------------------------------------------------------------

    def type_of(self, text: str) -> str:
        """Return the canonical display type of *text* as an expression entry.

        Resolves against the session scope, typechecks against the session
        environment, and match-compiles before reporting the type. It never
        lowers, evaluates, promotes, or advances the node-id counter. Raises the
        underlying ``AglSyntaxError``/``AglScopeError``/``AglTypeError`` on
        failure, or ``AglError`` for match errors or a non-expression entry.
        """
        from agm.agl.lexer import spaced_qualifier_collector
        from agm.agl.modules.ids import ENTRY_ID
        from agm.agl.parser import parse_program_seeded
        from agm.agl.syntax.nodes import Binder, Declaration

        host_env = self._runtime.host_environment()
        # Throwaway ids: type_of never promotes and never advances the session
        # counter, so seeding at ``_next_node_id`` is safe — all promoted ids are
        # strictly below it, making this parse's ids disjoint from the session's.
        with spaced_qualifier_collector() as spaced_sink:
            program, next_node_id = parse_program_seeded(
                text, start_id=self._next_node_id, ambient_infix=self._accumulated_infix
            )
        items = program.body.items
        if len(items) != 1 or isinstance(items[0], (Binder, Declaration)):
            raise AglError(
                "':type' expects a single expression, not a binding, declaration, or statement."
            )
        expr_item = items[0]
        checked_program = self._entry_pipeline.resolve_and_check_program(
            program, next_node_id, host_env, spaced_qualifiers=tuple(spaced_sink)
        )
        checked = checked_program.modules[ENTRY_ID]
        from agm.agl.matchcompile import compile_program_matches, diagnostics_from_match_issues

        match_result = compile_program_matches(checked_program)
        if match_result.compiled is None:
            diagnostic = diagnostics_from_match_issues(match_result.issues)[0]
            raise AglError(diagnostic.message, span=match_result.issues[0].span)
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
        """
        from agm.agl.semantics.values import Cell

        result: list[tuple[str, Type, Value]] = []
        for name, ref in self._session_scope.bindings.items():
            typ = self._type_env.get_binding_type(ref.decl_node_id)
            # Every promoted let/var/param binding has a recorded type.
            assert typ is not None
            symbol = self._link_image.symbol_for_decl(ref.decl_node_id)
            slot = self._ir_base_frame.get(symbol) if symbol is not None else None
            assert slot is not None
            value = slot.value if isinstance(slot, Cell) else slot
            result.append((name, typ, value))
        return result

    def declared_params(self) -> list[tuple[str, "Type", "Value"]]:
        """Return declared params as (external key, type, resolved value).

        A scoped param's key is its full path spelling, matching how it is
        supplied from the CLI and config; a root param's key is its bare name.
        """
        result: list[tuple[str, Type, Value]] = []
        for name, (typ, decl_node_id) in self._declared_params.items():
            value = self._declaration_value(decl_node_id)
            assert value is not None
            result.append((name, typ, value))
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
        from ``std/config`` writes entered during the session.
        """
        from agm.agl.lower import LinkImage
        from agm.agl.scope.symbols import ScopeNode
        from agm.agl.typecheck.env import TypeEnvironment

        self._session_scope = ScopeNode(node_id=-1, parent=None)
        self._session_scope_nodes = {(): self._session_scope}
        self._session_type_paths = {}
        self._type_env = TypeEnvironment()
        self._type_env.seal()
        self._link_image = LinkImage()
        self._ir_base_frame = {}
        self._next_node_id = 0
        self._program_name = None
        self._active_config = {}
        self._declared_params = {}
        self._source_log = []
        self._ambient_constructor_candidates = {}
        self._ambient_type_names = frozenset()
        # Re-seed the host-consumed registers so a prior
        # ``std/config::default-agent`` (etc.) write does not bleed past
        # :reset. ``_declared_engine_defaults`` is now populated (if
        # ``std/config`` was ever loaded), so these calls apply the same
        # host-seed-else-declared-default precedence used at construction, this
        # time landing on the declared branch where no host seed exists.
        self._persisted_host_settings = self._host_settings_seed()
        self._persisted_strict_json = self._strict_json_seed()
        self._persisted_timeout_setting = self._timeout_seed()
        self._trace_path = self._initial_trace_path
        # Restore the initial host controls or, where none exist, the declared
        # defaults evaluated when ``std/config`` was loaded before :reset.
        strict_json, loop_limit, shell_exec_timeout = self._reset_live_engine_settings()
        self._update_engine_settings(
            strict_json=strict_json,
            loop_limit=loop_limit,
            shell_exec_timeout=shell_exec_timeout,
        )
        # Clear module state.
        self._roots = None
        self._loaded_lib_modules = {}
        self._accumulated_imports = []
        self._accumulated_opens = []
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
