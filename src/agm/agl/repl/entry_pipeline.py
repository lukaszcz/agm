"""Multi-module REPL program pipeline collaborator.

Implements the build_repl_graph → resolve_program → check_program → match
compilation → incremental link/exec pipeline for REPL entries that contain
import declarations or have cached library modules from prior entries. Driven
by ``ReplSession`` via the narrow ``EntryPipelineCtx`` Protocol. Must NOT import
``session`` (no cycle).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from agm.agl.diagnostics import Diagnostic, diagnostic_from_span
from agm.agl.modules.ids import ModuleId
from agm.agl.repl.entry import EntryKind, EntryResult

if TYPE_CHECKING:
    from pathlib import Path

    from agm.agl.eval.ir_interpreter import IrInterpreter
    from agm.agl.ir.builtin_vars import BuiltinVarKey
    from agm.agl.ir.contracts import ContractPayload
    from agm.agl.ir.ids import SymbolId
    from agm.agl.lower import LinkImage
    from agm.agl.matchcompile import MatchCompiledProgram
    from agm.agl.modules.loader import LoadedModule, ModuleGraph
    from agm.agl.modules.roots import RootSet
    from agm.agl.pipeline import RunError
    from agm.agl.runtime.host_settings import HostSettingsPolicy
    from agm.agl.runtime.trace import TraceStore
    from agm.agl.runtime.types import HostEnvironment
    from agm.agl.scope.program import ResolvedModule, ResolvedProgram
    from agm.agl.scope.symbols import ConstructorRef, ScopeNode
    from agm.agl.semantics.types import Type
    from agm.agl.semantics.values import Frame, Value
    from agm.agl.setting_overrides import SettingOverride
    from agm.agl.syntax.advisories import SpacedQualifier
    from agm.agl.syntax.nodes import ImportDecl, InfixAssoc, Item, Program, ScopeRegion
    from agm.agl.typecheck.env import CheckedModule, TypeEnvironment
    from agm.agl.typecheck.program import CheckedProgram


# ---------------------------------------------------------------------------
# Narrow context Protocol
# ---------------------------------------------------------------------------


class EntryPipelineCtx(Protocol):
    """The minimal ReplSession surface the program pipeline needs."""

    _loaded_lib_modules: dict[ModuleId, LoadedModule]
    _retained_resolved_modules: dict[ModuleId, ResolvedModule]
    _retained_checked_modules: dict[ModuleId, CheckedModule]
    _last_match_compilation: MatchCompiledProgram | None
    _accumulated_imports: list[tuple[ImportDecl, ...]]
    _accumulated_scoped_imports: list[tuple[ImportDecl | ScopeRegion, ...]]
    _accumulated_infix: dict[str, tuple[int, InfixAssoc]]
    _link_image: LinkImage
    _ir_base_frame: Frame
    _setting_overrides: dict[str, SettingOverride]
    _session_scope: ScopeNode
    _session_scope_nodes: dict[tuple[str, ...], ScopeNode]
    _session_type_paths: dict[tuple[str, ...], str | None]
    _type_env: TypeEnvironment
    _ambient_constructor_candidates: dict[str, tuple[ConstructorRef, ...]]
    _ambient_bare_constructor_candidates: dict[str, tuple[ConstructorRef, ...]]
    _ambient_type_names: frozenset[str]
    _trace_path: Path | None
    _default_loop_limit: int | None
    _default_call_depth_limit: int
    _default_stdlib: bool
    _shell_exec_timeout: float | None
    _process_environment: dict[str, str] | None
    _builtin_var_seed: dict[BuiltinVarKey, Value]
    _builtin_var_values: dict[BuiltinVarKey, Value]
    # The current-value register for the five engine keys with a ``Value``
    # form (strict-json, timeout, log, log-file, default-agent); a key is
    # present only once a host seed or a learned declared default has made it
    # meaningful. See ``ReplSession._current``.
    _current: dict[str, Value]
    _host_settings_policy: HostSettingsPolicy | None

    @property
    def _default_strict_json(self) -> bool: ...

    def _ensure_roots(self) -> RootSet: ...

    def _fail(self, diagnostics: list[Diagnostic], warnings: list[Diagnostic]) -> EntryResult: ...

    def _build_check_only_result(
        self, program: Program, checked: CheckedModule, warnings: list[Diagnostic]
    ) -> EntryResult: ...

    def _record_declared_engine_defaults(
        self, declared_keys: frozenset[str], interp: IrInterpreter
    ) -> None: ...

    def _update_engine_settings(self, interp: IrInterpreter) -> None: ...

    def _advance_node_ids(self, next_start_id: int) -> None: ...

    def _promote_ir_state(
        self,
        *,
        text: str,
        program: Program,
        checked: CheckedModule,
        next_start_id: int,
        partial: bool,
        promoted_declaration_ids: frozenset[int],
        infix_ambient: Mapping[str, tuple[int, InfixAssoc]],
    ) -> tuple[str, ...]: ...

    def _classify(self, program: Program) -> tuple[EntryKind, str | None]: ...

    def frame_value(self, symbol: SymbolId | None) -> Value | None: ...

    def _echo_data_ir(
        self, program: Program, checked: CheckedModule, captured: Value | None
    ) -> tuple[Value | None, Type | None]: ...

    def _quote_strings_for_entry(self, program: Program) -> bool: ...


class OverrideRejected(Exception):
    """Internal signal: a setting-override splice was rejected.

    Raised by :meth:`EntryPipeline.load_and_check_program` when
    ``_apply_setting_overrides`` returns diagnostics rather than succeeding,
    so that stage can share one raise-on-failure contract with the syntax,
    module-loading, scope, and type-check stages around it. Callers catch it
    and read :attr:`diagnostics` the same way they read a caught
    ``AglError.to_diagnostic()``.
    """

    def __init__(self, diagnostics: list[Diagnostic]) -> None:
        super().__init__("setting override rejected")
        self.diagnostics = diagnostics


@dataclass(frozen=True, slots=True)
class LoadedCheckedProgram:
    """Result of :meth:`EntryPipeline.load_and_check_program`."""

    checked_program: "CheckedProgram"
    new_modules: "dict[ModuleId, LoadedModule]"
    module_adjacency: "dict[ModuleId, tuple[ModuleId, ...]]"
    new_next_id: int
    entry_imports: "tuple[ImportDecl, ...]"
    entry_uses: "tuple[ImportDecl | ScopeRegion, ...]"
    entry_infix_ambient: "dict[str, tuple[int, InfixAssoc]]"


# ---------------------------------------------------------------------------
# Collaborator class
# ---------------------------------------------------------------------------


class EntryPipeline:
    """Program pipeline collaborator for ``ReplSession``.

    Instantiated once per ``ReplSession`` (``self._entry_pipeline``).  Holds
    no state of its own — all session state is borrowed via ``EntryPipelineCtx``.
    """

    def __init__(self, ctx: EntryPipelineCtx) -> None:
        self._ctx = ctx

    def load_and_check_program(
        self,
        *,
        pipeline_program: Program,
        host_env: HostEnvironment,
        next_start_id: int,
        spaced_qualifiers: tuple[SpacedQualifier, ...] = (),
        validate_missing_std_config: bool = False,
    ) -> LoadedCheckedProgram:
        """Build the module graph, splice overrides, resolve, and type-check.

        Shared by :meth:`eval_entry` (which continues on to match
        compilation, lowering, and evaluation) and ``ReplSession.open``
        (which stops here and promotes only the loaded library modules,
        before the session accepts its first entry): builds on the session's
        retained import/use preamble and cached library modules, then
        applies ``setting_overrides`` via
        :func:`~agm.agl.pipeline.apply_setting_overrides` — which owns both
        the splice-once-per-session apply condition and the module-cache
        reconciliation a caller that caches ``new_modules`` needs — before
        resolving and type-checking the result.

        ``validate_missing_std_config`` is ``False`` for an ordinary entry
        (:meth:`eval_entry`): a ``required`` override (e.g. ``--agent``) that
        ``std/config`` never loads for stays unvalidated until whichever
        later entry, if any, first loads it, matching how a non-``required``
        override already behaves. ``ReplSession.open`` passes ``True``
        instead, so a ``required`` override is validated at session-open
        time even when the initial image never loads ``std/config`` (e.g.
        ``--no-stdlib`` with no explicit import) — reported before the
        session accepts its first entry rather than deferred to one that may
        never come.

        Raises the underlying ``AglSyntaxError``/module-loading
        error/``AglScopeError``/``AglTypeError`` on failure, or
        :class:`OverrideRejected` when the override splice itself is
        rejected — callers adapt these to their own failure-reporting shape.
        """
        from agm.agl.modules.loader import build_repl_graph
        from agm.agl.pipeline import apply_setting_overrides
        from agm.agl.typecheck.program import check_program

        roots = self._ctx._ensure_roots()

        entry_program, next_start_id, entry_imports, entry_uses = self._prepare_entry_program(
            pipeline_program, next_start_id, roots
        )
        graph, new_next_id, new_modules = build_repl_graph(
            entry_program,
            next_start_id,
            path=None,
            cached=self._ctx._loaded_lib_modules,
            roots=roots,
            default_stdlib=self._ctx._default_stdlib,
            spaced_qualifiers=spaced_qualifiers,
            session_infix=self._ctx._accumulated_infix,
        )

        graph, new_next_id, override_diagnostics, new_modules = apply_setting_overrides(
            graph,
            new_next_id,
            self._ctx._setting_overrides,
            newly_loaded_modules=new_modules,
            validate_when_absent=validate_missing_std_config,
        )
        if override_diagnostics:
            raise OverrideRejected(override_diagnostics)

        resolved_program = self._resolve_program(graph)
        checked_program = check_program(
            resolved_program,
            host_env.capabilities,
            entry_seed_env=self._ctx._type_env,
            cached_checked_modules=self._ctx._retained_checked_modules,
        )
        self._retain_module_artifacts(resolved_program, checked_program)
        return LoadedCheckedProgram(
            checked_program=checked_program,
            new_modules=new_modules,
            module_adjacency=graph.adjacency,
            new_next_id=new_next_id,
            entry_imports=entry_imports,
            entry_uses=entry_uses,
            entry_infix_ambient=graph.entry_infix_ambient,
        )

    def eval_entry(
        self,
        *,
        text: str,
        orig_program: Program,
        pipeline_program: Program,
        host_env: HostEnvironment,
        tab_warnings: list[Diagnostic],
        next_start_id: int,
        check_only: bool,
        spaced_qualifiers: tuple[SpacedQualifier, ...] = (),
    ) -> EntryResult:
        """Program pipeline for REPL entries that have imports or cached lib modules.

        Builds the module graph from the already-parsed *pipeline_program*, runs
        the full scope/typecheck/match-compilation passes with the session
        context, then returns a check-only result or lowers and evaluates.
        """
        from agm.agl.diagnostics import AglError
        from agm.agl.modules.errors import (
            AmbiguousModule,
            ImportEntryError,
            MissingExternCompanion,
            ModuleNotFound,
            ModulePrefixNotFound,
        )
        from agm.agl.parser import AglSyntaxError
        from agm.agl.scope import AglScopeError
        from agm.agl.typecheck import AglTypeError

        try:
            loaded = self.load_and_check_program(
                pipeline_program=pipeline_program,
                host_env=host_env,
                next_start_id=next_start_id,
                spaced_qualifiers=spaced_qualifiers,
            )
        except AglSyntaxError as exc:
            return self._ctx._fail([exc.to_diagnostic()], tab_warnings)
        except (
            ModuleNotFound,
            AmbiguousModule,
            ModulePrefixNotFound,
            ImportEntryError,
            MissingExternCompanion,
        ) as exc:
            return self._ctx._fail([exc.to_diagnostic()], tab_warnings)
        except OverrideRejected as exc:
            return self._ctx._fail(exc.diagnostics, tab_warnings)
        except AglScopeError as exc:
            return self._ctx._fail([exc.to_diagnostic()], tab_warnings)
        except AglTypeError as exc:
            return self._ctx._fail([exc.to_diagnostic()], tab_warnings)
        except AglError as exc:
            return self._ctx._fail([exc.to_diagnostic()], tab_warnings)
        except Exception as exc:
            return self._ctx._fail([Diagnostic(message=str(exc), line=1)], tab_warnings)

        checked_program = loaded.checked_program
        new_modules = loaded.new_modules
        module_adjacency = loaded.module_adjacency
        new_next_id = loaded.new_next_id
        entry_imports = loaded.entry_imports
        entry_uses = loaded.entry_uses
        entry_infix_ambient = loaded.entry_infix_ambient
        entry_cm = checked_program.modules[checked_program.entry_id]

        # Collect warnings from all passes.
        warnings: list[Diagnostic] = [*tab_warnings, *checked_program.warnings]

        from agm.agl.matchcompile import (
            cached_module_sites,
            compile_program_matches,
            diagnostics_from_match_issues,
        )

        match_result = compile_program_matches(
            checked_program, cached_module_sites(self._ctx._last_match_compilation)
        )
        if match_result.compiled is None:
            return self._ctx._fail(
                list(diagnostics_from_match_issues(match_result.issues)), warnings
            )
        compiled = match_result.compiled
        from agm.agl.matchcompile import MatchCompiledProgram

        assert isinstance(compiled, MatchCompiledProgram)
        # Retained for the next entry: its library modules are the same checked
        # objects, so their compiled sites carry over on identity alone.
        self._ctx._last_match_compilation = compiled

        checked = self._checked_program_from_module(entry_cm)
        if check_only:
            return self._ctx._build_check_only_result(orig_program, checked, warnings)

        from agm.agl.pipeline import _materialize_program_custom_contract_payloads

        contract_payloads, contract_errors = _materialize_program_custom_contract_payloads(
            checked_program,
            host_env.codecs,
        )
        if contract_errors:
            return self._ctx._fail(contract_errors, warnings)

        return self._evaluate_ir_program(
            text=text,
            orig_program=orig_program,
            checked=checked,
            entry_cm=entry_cm,
            checked_program=checked_program,
            compiled=compiled,
            host_env=host_env,
            warnings=warnings,
            new_next_id=new_next_id,
            new_modules=new_modules,
            module_adjacency=module_adjacency,
            entry_imports=entry_imports,
            entry_uses=entry_uses,
            entry_infix_ambient=entry_infix_ambient,
            contract_payloads=contract_payloads,
        )

    def resolve_and_check_program(
        self,
        program: Program,
        next_start_id: int,
        host_env: HostEnvironment,
        *,
        spaced_qualifiers: tuple[SpacedQualifier, ...] = (),
    ) -> CheckedProgram:
        """Prepare, build the module graph, resolve, and typecheck *program*.

        Shared by REPL call sites that only need a checked program — no match
        compilation, lowering, or evaluation — such as ``type_of`` and the
        throwaway std/import type-environment builder. Raises the underlying
        ``AglSyntaxError``/module-loading errors/``AglScopeError``/``AglTypeError``
        on failure; callers that need diagnostics instead of a raised exception
        must catch these themselves.
        """
        from agm.agl.modules.loader import build_repl_graph
        from agm.agl.typecheck.program import check_program

        roots = self._ctx._ensure_roots()
        entry_program, next_start_id, _entry_imports, _entry_uses = self._prepare_entry_program(
            program, next_start_id, roots
        )
        graph, _next_start_id, _new_modules = build_repl_graph(
            entry_program,
            next_start_id,
            path=None,
            cached=self._ctx._loaded_lib_modules,
            roots=roots,
            default_stdlib=self._ctx._default_stdlib,
            spaced_qualifiers=spaced_qualifiers,
            session_infix=self._ctx._accumulated_infix,
        )
        resolved_program = self._resolve_program(graph)
        return check_program(
            resolved_program,
            host_env.capabilities,
            entry_seed_env=self._ctx._type_env,
            cached_checked_modules=self._ctx._retained_checked_modules,
        )

    def _retain_module_artifacts(
        self, resolved_program: "ResolvedProgram", checked_program: "CheckedProgram"
    ) -> None:
        """Retain every non-entry module of this compilation for later entries.

        Scope resolution and type checking of a module depend only on
        that module and the modules it imports, so a later entry that carries
        the same ``Program`` object reuses the artifact instead of rebuilding
        it. The reuse guards live in the passes themselves and key on object
        identity, so a reparsed, spliced or superseded module simply misses.
        """
        for module_id, checked in checked_program.modules.items():
            if module_id == checked_program.entry_id:
                continue
            self._ctx._retained_resolved_modules[module_id] = resolved_program.modules[module_id]
            self._ctx._retained_checked_modules[module_id] = checked

    def _resolve_program(
        self,
        graph: "ModuleGraph",
    ) -> "ResolvedProgram":
        """Resolve one entry against the session's retained resolved scope state."""
        from agm.agl.scope.program import resolve_program

        return resolve_program(
            graph,
            entry_ambient_constructor_candidates=self._ctx._ambient_constructor_candidates,
            entry_ambient_type_names=self._ctx._ambient_type_names,
            entry_ambient_bare_constructor_keys=frozenset(
                (cname, ref.owner_module_id, ref.owner_decl_node_id)
                for cname, crefs in self._ctx._ambient_bare_constructor_candidates.items()
                for ref in crefs
            ),
            entry_parent_scope=self._ctx._session_scope,
            entry_repl_session_scope=self._ctx._session_scope,
            entry_repl_session_scope_nodes=self._ctx._session_scope_nodes,
            entry_repl_session_type_paths=self._ctx._session_type_paths,
            cached_modules=self._ctx._retained_resolved_modules,
        )

    @staticmethod
    def _checked_program_from_module(entry: CheckedModule) -> CheckedModule:
        """Adapt entry-module checker output for REPL static-state promotion."""
        from agm.agl.typecheck.env import CheckedModule

        return CheckedModule(
            resolved=entry.resolved,
            node_types=entry.node_types,
            contract_specs=entry.contract_specs,
            call_sites=entry.call_sites,
            warnings=entry.warnings,
            type_env=entry.type_env,
            function_signatures=entry.function_signatures,
            cast_specs=entry.cast_specs,
            argument_bindings=entry.argument_bindings,
            pattern_classifications=entry.pattern_classifications,
            partial_calls=entry.partial_calls,
            slot_resolution=entry.slot_resolution,
            slot_constructor_refs=entry.slot_constructor_refs,
            is_test_constructor_refs=entry.is_test_constructor_refs,
            let_matched_types=entry.let_matched_types,
            pattern_binding_refs=entry.pattern_binding_refs,
            pattern_constructor_refs=entry.pattern_constructor_refs,
            pattern_constructor_owners=entry.pattern_constructor_owners,
            method_selections=entry.method_selections,
        )

    def _prepare_entry_program(
        self,
        program: Program,
        next_start_id: int,
        roots: RootSet,
    ) -> tuple[Program, int, tuple[ImportDecl, ...], tuple[ImportDecl | ScopeRegion, ...]]:
        """Expand current wildcards, then inject retained imports and uses.

        REPL replacement is finer grained than batch import merging: each
        wildcard expands to exact target modules before retained declarations
        are compared. A new entry replaces only the modules it names, while
        declarations for one module in that entry still union normally.

        Retention keeps each entry's declarations as written, so a retained
        wildcard is re-expanded here against the current roots and picks up
        modules added since. Root imports and region-scoped imports at the
        same region path each keep their own chronological replacement
        decision -- see ``_retained_preamble``.

        The final item order is [retained root imports, this entry's own root
        import/export decls, retained uses/regions, this entry's remaining
        items]. This entry's own root header decls are hoisted ahead of the
        retained uses/regions -- rather than left in their original,
        already-header-legal position within the entry -- because a retained
        region item would otherwise land before them, and the header rule
        ("import and export declarations must appear before any other
        declarations in a module or scope region") applies to the whole
        concatenated root sequence the pipeline resolves, not just to what
        the entry wrote. A root import/export is module-wide and
        order-independent among headers, so hoisting it is semantics-preserving.
        """
        from agm.agl.syntax.nodes import ImportDecl

        entry_imports = tuple(item for item in program.body.items if isinstance(item, ImportDecl))
        entry_uses = self._retained_scoped_import_items(program.body.items)
        expanded, next_start_id, expanded_imports = self._expand_entry_wildcards(
            program, next_start_id, roots
        )
        # The expanded root imports feed the preamble's newest-generation
        # decision, so this entry's wildcards are globbed once, not again.
        import_preamble, use_preamble, next_start_id = self._retained_preamble(
            expanded_imports, entry_uses, roots, next_start_id
        )
        entry_headers, entry_rest = self._partition_entry_root_headers(expanded.body.items)
        preamble: list[Item] = [*import_preamble, *entry_headers, *use_preamble]
        rest_program = (
            expanded if not entry_headers else self._replace_body_items(expanded, entry_rest)
        )
        return (
            self._prepend_items(rest_program, preamble),
            next_start_id,
            entry_imports,
            entry_uses,
        )

    @staticmethod
    def _partition_entry_root_headers(items: tuple[Item, ...]) -> tuple[list[Item], list[Item]]:
        """Split *items* into this entry's own root header decls and everything else.

        A root ``import``/``export`` (empty ``scope_path``) is the entry's
        own header contribution, distinct from the *retained* root imports
        already folded into ``import_preamble`` -- this partition finds the
        current entry's, so ``_prepare_entry_program`` can hoist them ahead
        of retained uses/regions.
        """
        from agm.agl.syntax.nodes import ExportDecl, ImportDecl

        headers: list[Item] = []
        rest: list[Item] = []
        for item in items:
            if isinstance(item, (ImportDecl, ExportDecl)) and not item.scope_path:
                headers.append(item)
            else:
                rest.append(item)
        return headers, rest

    @staticmethod
    def _expand_entry_wildcards(
        program: Program,
        next_start_id: int,
        roots: RootSet,
    ) -> tuple[Program, int, tuple[ImportDecl, ...]]:
        """Expand wildcard imports into distinct exact-module declarations."""
        from agm.agl.syntax.nodes import Block, ImportDecl, Program

        items: list[Item] = []
        imports: list[ImportDecl] = []
        expanded_wildcard = False
        for item in program.body.items:
            if not isinstance(item, ImportDecl):
                items.append(item)
                continue
            expanded_wildcard = expanded_wildcard or item.wildcard
            expanded, next_start_id = EntryPipeline._expand_decls((item,), roots, next_start_id)
            items.extend(expanded)
            imports.extend(expanded)

        if not expanded_wildcard:
            return program, next_start_id, tuple(imports)
        return (
            Program(
                body=Block(
                    items=tuple(items),
                    span=program.body.span,
                    node_id=program.body.node_id,
                ),
                span=program.span,
                node_id=program.node_id,
            ),
            next_start_id,
            tuple(imports),
        )

    @staticmethod
    def _prepend_items(program: Program, preamble: list[Item]) -> Program:
        """Prepend retained header items to *program*'s body."""
        if not preamble:
            return program
        return EntryPipeline._replace_body_items(program, [*preamble, *program.body.items])

    @staticmethod
    def _replace_body_items(program: Program, items: list[Item]) -> Program:
        """Rebuild *program* with its body's item sequence replaced by *items*.

        Preserves the ``Program``/``Block`` node ids and spans exactly;
        callers that reorder or filter a program's root items -- rather than
        merely prepending, which is ``_prepend_items`` -- share this rebuild.
        """
        from agm.agl.syntax.nodes import Block, Program

        return Program(
            body=Block(
                items=tuple(items),
                span=program.body.span,
                node_id=program.body.node_id,
            ),
            span=program.span,
            node_id=program.node_id,
        )

    def _evaluate_ir_program(
        self,
        *,
        text: str,
        orig_program: Program,
        checked: CheckedModule,
        entry_cm: CheckedModule,
        checked_program: CheckedProgram,
        compiled: MatchCompiledProgram,
        host_env: HostEnvironment,
        warnings: list[Diagnostic],
        new_next_id: int,
        new_modules: dict[ModuleId, LoadedModule],
        module_adjacency: dict[ModuleId, tuple[ModuleId, ...]],
        entry_imports: tuple[ImportDecl, ...],
        entry_uses: tuple[ImportDecl | ScopeRegion, ...],
        entry_infix_ambient: Mapping[str, tuple[int, InfixAssoc]],
        contract_payloads: Mapping[int, "ContractPayload"],
    ) -> EntryResult:
        """Lower and execute one program entry in the persistent IR image."""
        from agm.agl.eval.ir_interpreter import HostConfigurationError, IrInterpreter
        from agm.agl.lower import lower_repl_program
        from agm.agl.pipeline import _wire_extern_registry, exception_value_to_run_error
        from agm.agl.recursion import NestingTooDeepError, frontend_recursion_boundary
        from agm.agl.runtime.contract import materialize_ir_contracts
        from agm.agl.runtime.request import AgentCancelled
        from agm.agl.runtime.trace import TraceStore
        from agm.agl.semantics.exceptions import AglRaise
        from agm.agl.syntax.resources import ResourceError

        # Companion paths for every module the checked program can reach: prior
        # entries' cached library modules plus this entry's newly linked ones.
        # ``_wire_extern_registry`` imports/resolves only what is not already
        # cached on ``host_env.extern_registry`` (mutated in place), so a
        # companion imports exactly once per session even across entries.
        companion_paths: dict[ModuleId, Path | None] = {
            mid: lm.companion_path for mid, lm in self._ctx._loaded_lib_modules.items()
        }
        companion_paths.update({mid: lm.companion_path for mid, lm in new_modules.items()})
        # Lowering allocates into the persistent image, so every way this entry
        # can fail from here on rolls back against this snapshot: an entry
        # rejected before anything is promoted discards its whole link delta
        # via ``restore_state``. A partially run entry keeps its delta, caching
        # and marking only dependency-complete library modules, and needs no
        # nominal rollback at all -- the link image's nominal state is rebuilt
        # from the shared type table on every lowering, so it is always current
        # regardless of what this entry did or did not promote.
        link_snapshot = self._ctx._link_image.snapshot_state()
        try:
            with frontend_recursion_boundary():
                lowered = lower_repl_program(
                    compiled,
                    image=self._ctx._link_image,
                    source_text=text,
                    contract_payloads=contract_payloads,
                )
        except (NestingTooDeepError, ResourceError) as exc:
            self._ctx._link_image.restore_state(link_snapshot)
            diagnostic = (
                diagnostic_from_span(str(exc), exc.span)
                if exc.span is not None
                else Diagnostic(message=str(exc), line=1)
            )
            return self._ctx._fail([diagnostic], warnings)
        extern_diagnostics = _wire_extern_registry(
            checked=checked_program,
            capabilities=host_env.capabilities,
            registry=host_env.extern_registry,
            companion_paths=companion_paths,
            nominals=lowered.program.nominals,
        )
        if extern_diagnostics:
            # A pre-execution rejection: nothing ran and nothing promoted, but
            # ``_wire_extern_registry`` registers this entry's nominals with
            # the extern registry before it imports any companion, and that
            # registration is never rolled back (the registry's nominal class
            # cache is insert-only). The node-id range must therefore not be
            # reused -- reusing it would let a later entry's redeclaration
            # collide with an identity this rejected entry already registered
            # under its own, now-abandoned shape.
            self._ctx._link_image.restore_state(link_snapshot)
            self._ctx._advance_node_ids(new_next_id)
            return self._ctx._fail(extern_diagnostics, warnings)
        host_contracts, _ = materialize_ir_contracts(lowered.program, host_env.codecs)
        trace = TraceStore(path=self._ctx._trace_path)
        trace.run_start()
        if self._ctx._host_settings_policy is not None:
            from agm.agl.runtime.host_settings import HostSettingsReconfigurer

            reconfigurer: HostSettingsReconfigurer | None = HostSettingsReconfigurer(
                trace=trace,
                policy=self._ctx._host_settings_policy,
            )
        else:
            reconfigurer = None
        try:
            interp = IrInterpreter(
                lowered.program,
                agent_dispatcher=host_env.agent_dispatcher,
                session_host=host_env.session_host,
                close_sessions=False,
                strict_json=self._ctx._default_strict_json,
                loop_limit=self._ctx._default_loop_limit,
                max_call_depth=self._ctx._default_call_depth_limit,
                shell_exec_timeout=(
                    self._ctx._shell_exec_timeout if "timeout" not in self._ctx._current else None
                ),
                trace=trace,
                host_contracts=host_contracts,
                base_frame=self._ctx._ir_base_frame,
                extern_registry=host_env.extern_registry,
                host_reconfigurer=reconfigurer,
                # The current-value register already carries every key that is
                # meaningful yet (a host seed, or a declared default learned by
                # an earlier entry): a key genuinely absent here is exactly one
                # this interpreter should learn its own ``std/config`` declared
                # default for, rather than have imposed on it.
                builtin_host_settings=self._builtin_host_settings(),
                process_environment=self._ctx._process_environment,
            )
        except AglRaise as exc:
            error = exception_value_to_run_error(
                exc.exc,
                span=exc.span,
                exception_field_encodes=lowered.program.exception_field_encodes,
            )
            trace.exception(
                type_name=error.type_name,
                message=str(error.fields.get("message", "")),
                span=exc.span,
            )
            trace.run_end(ok=False)
            # A rejected declared setting default likewise promotes nothing, so
            # the same discard applies -- but this entry is reported as a run
            # that raised, so it consumes its node-id range like every other
            # entry that reached the interpreter.
            self._ctx._link_image.restore_state(link_snapshot)
            self._ctx._advance_node_ids(new_next_id)
            kind, name = self._ctx._classify(orig_program)
            return EntryResult(
                kind=kind,
                name=name,
                value=None,
                value_type=None,
                diagnostics=[],
                warnings=warnings,
                error=error,
                ok=False,
                trace_path=self._ctx._trace_path,
            )
        except HostConfigurationError as exc:
            # The materialized ``default-agent`` value cannot be dispatched: a
            # pre-execution host-configuration rejection rather than an
            # uncaught AgL exception, so it is reported as an ordinary
            # diagnostic -- like any other per-entry error, this never exits
            # the REPL process.
            trace.run_end(ok=False)
            self._ctx._link_image.restore_state(link_snapshot)
            self._ctx._advance_node_ids(new_next_id)
            kind, name = self._ctx._classify(orig_program)
            return EntryResult(
                kind=kind,
                name=name,
                value=None,
                value_type=None,
                diagnostics=[Diagnostic(message=str(exc), line=1)],
                warnings=warnings,
                error=None,
                ok=False,
                trace_path=self._ctx._trace_path,
            )
        from agm.agl.modules.ids import STD_CONFIG_ID

        self._ctx._record_declared_engine_defaults(
            frozenset(
                name
                for module_id, scope_path, name in cast(
                    "Mapping[BuiltinVarKey, object]", lowered.program.builtin_setting_defaults
                )
                if module_id == STD_CONFIG_ID and not scope_path
            ),
            interp,
        )

        def retain_library_state(module_ids: frozenset[ModuleId]) -> None:
            """Cache one coherent set of initialized library modules."""
            self._ctx._loaded_lib_modules.update(
                (module_id, new_modules[module_id])
                for module_id in module_ids
                if module_id in new_modules
            )
            self._ctx._link_image.mark_linked(module_ids)

        def completed_library_module_ids() -> frozenset[ModuleId]:
            """Return newly loaded modules retained by this entry.

            A module is retained only when its own initializers all ran AND
            every dependency it needs is retained too. An interrupt can abort
            a run between modules, so a module later in initialization order
            can complete while an earlier one it does not depend on does not;
            and a module import cycle makes the loader's initialization order
            arbitrary within the cycle, so a completed module can still
            depend on a later, unretained one. The dependency fixpoint below
            prunes those out.
            """
            candidates: set[ModuleId] = set()
            for module_id in new_modules:
                module = lowered.program.modules[module_id]
                completed_indices = interp.module_completed_initializer_indices.get(
                    module_id, set()
                )
                if completed_indices == set(range(len(module.initializers))):
                    candidates.add(module_id)

            available = set(self._ctx._loaded_lib_modules) | candidates
            while incomplete := {
                module_id
                for module_id in candidates
                if any(
                    dependency != lowered.program.entry_module and dependency not in available
                    for dependency in module_adjacency.get(module_id, ())
                )
            }:
                candidates.difference_update(incomplete)
                available.difference_update(incomplete)
            return frozenset(candidates)

        def promote(*, partial: bool, promoted_declaration_ids: frozenset[int]) -> tuple[str, ...]:
            return self._ctx._promote_ir_state(
                text=text,
                program=orig_program,
                checked=checked,
                next_start_id=new_next_id,
                partial=partial,
                promoted_declaration_ids=promoted_declaration_ids,
                infix_ambient=entry_infix_ambient,
            )

        def partial_failure(
            *, diagnostics: list[Diagnostic], error: "RunError | None"
        ) -> EntryResult:
            """Keep what this entry completed, drop what it did not, and report.

            The promotion plan applies the declaration-dependency fixpoint
            over the entry's completed IR initializers. Unlike a rejected
            entry, this one keeps a partial link delta rather than being
            restored wholesale, and needs no nominal rollback of its own: the
            link image's nominal state, including any ``builtin`` declaration's
            host-mint override, is rebuilt from the shared type table on every
            lowering. Library modules whose initializers did complete are
            retained, provided their dependencies completed too.
            """
            trace.run_end(ok=False)
            self._persist_interpreter_settings(interp, trace)
            completed_module_ids = completed_library_module_ids()
            promoted = lowered.promotion_plan.completed_declaration_ids(
                interp.module_completed_initializer_indices.get(
                    lowered.program.entry_module, set()
                ),
                self._ctx._loaded_lib_modules.keys() | completed_module_ids,
            )
            installed = promote(partial=True, promoted_declaration_ids=promoted)
            retain_library_state(completed_module_ids)
            kind, name = self._ctx._classify(orig_program)
            return EntryResult(
                kind=kind,
                name=name,
                value=None,
                value_type=None,
                diagnostics=diagnostics,
                warnings=warnings,
                error=error,
                ok=False,
                trace_path=self._ctx._trace_path,
                installed=installed,
            )

        try:
            interp.run()
        except SystemExit as exc:
            trace.run_end(ok=exc.code is None or exc.code == 0)
            raise
        except AglRaise as exc:
            error = exception_value_to_run_error(
                exc.exc,
                span=exc.span,
                exception_field_encodes=lowered.program.exception_field_encodes,
            )
            trace.exception(
                type_name=error.type_name,
                message=str(error.fields.get("message", "")),
                span=exc.span,
            )
            return partial_failure(diagnostics=[], error=error)
        except HostConfigurationError as exc:
            # A host-backed library value may be read after earlier entry work
            # completed. Preserve that completed state while reporting the
            # missing/invalid host value as an ordinary language diagnostic.
            return partial_failure(diagnostics=[Diagnostic(message=str(exc), line=1)], error=None)
        except (AgentCancelled, KeyboardInterrupt) as exc:
            cancellation_message = (
                "Agent call cancelled — entry aborted."
                if isinstance(exc, AgentCancelled)
                else "Entry interrupted — entry aborted."
            )
            return partial_failure(
                diagnostics=[Diagnostic(message=cancellation_message, line=1)], error=None
            )
        trace.run_end(ok=True)
        # Setting writes are ordinary non-transactional mutations: persist all
        # effects that completed, on success or before a later runtime failure.
        self._persist_interpreter_settings(interp, trace)
        promote(
            partial=False,
            promoted_declaration_ids=lowered.promotion_plan.completed_declaration_ids(
                range(len(lowered.program.modules[lowered.program.entry_module].initializers)),
                checked_program.modules.keys(),
            ),
        )
        retain_library_state(
            frozenset(
                module_id
                for module_id in checked_program.modules
                if module_id != checked_program.entry_id
            )
        )
        self._retain_import_context(entry_imports, entry_uses)
        marker = lowered.trailing_expression
        initializer_values = interp.module_initializer_values.get(lowered.program.entry_module)
        captured = (
            initializer_values[marker]
            if marker is not None and initializer_values is not None
            else None
        )
        if lowered.trailing_let_value_symbol is not None:
            captured = self._ctx.frame_value(lowered.trailing_let_value_symbol)
        kind, name = self._ctx._classify(orig_program)
        value, value_type = self._ctx._echo_data_ir(orig_program, checked, captured)
        return EntryResult(
            kind=kind,
            name=name,
            value=value,
            value_type=value_type,
            diagnostics=[],
            warnings=warnings,
            error=None,
            ok=True,
            trace_path=self._ctx._trace_path,
            quote_strings=self._ctx._quote_strings_for_entry(orig_program),
            type_table=checked.type_env.type_table,
        )

    def _retained_preamble(
        self,
        entry_imports: tuple[ImportDecl, ...],
        entry_uses: tuple[ImportDecl | ScopeRegion, ...],
        roots: RootSet,
        next_start_id: int,
    ) -> tuple[list[ImportDecl], list[ImportDecl | ScopeRegion], int]:
        """Expand retained entries and keep the newest import generation per module.

        Retained wildcards are re-expanded against the current roots, so a
        module added since the wildcard was written is imported now. A root
        import and a region-scoped import at some region path each own an
        independent chronological generation per module: every declaration
        for a module, at a given region path, in the newest entry that names
        it there is retained, while older declarations for that same module
        at that same region path are removed. An import at one region path
        never replaces a declaration at a different one -- in particular a
        region-scoped import never replaces a root import, or vice versa.
        Resolved ``use`` contributions are retained by session scope nodes,
        so this preamble contains imports only.
        """
        generations: list[tuple[list[ImportDecl], tuple[ImportDecl | ScopeRegion, ...]]] = []
        latest_generation: dict[tuple[tuple[str, ...], tuple[str, ...]], int] = {}
        for retained_root_decls, scoped_items in zip(
            self._ctx._accumulated_imports,
            self._ctx._accumulated_scoped_imports,
            strict=True,
        ):
            expanded_root_decls, next_start_id = self._expand_decls(
                retained_root_decls, roots, next_start_id
            )
            expanded_scoped, next_start_id = self._expand_retained_scoped_imports(
                scoped_items, roots, next_start_id
            )
            index = len(generations)
            generations.append((expanded_root_decls, expanded_scoped))
            generation_imports = (
                *expanded_root_decls,
                *self._scoped_import_decls(expanded_scoped),
            )
            for decl in generation_imports:
                latest_generation[self._generation_key(decl)] = index

        # *entry_imports* arrives already expanded; only the scoped ones still
        # need their module identities resolved. Wildcard expansion has one
        # definition: the node ids minted here are discarded with the
        # rebuilt declarations, since only each declaration's region path
        # and module identity are wanted.
        current_decls, _ = self._expand_decls(
            (*entry_imports, *self._scoped_import_decls(entry_uses)), roots, 0
        )
        current_index = len(generations)
        for decl in current_decls:
            latest_generation[self._generation_key(decl)] = current_index

        retained_root: list[ImportDecl] = []
        retained_scoped: list[ImportDecl | ScopeRegion] = []
        for index, (root_decls, scoped_items) in enumerate(generations):
            retained_root.extend(
                decl
                for decl in root_decls
                if latest_generation[self._generation_key(decl)] == index
            )
            retained_scoped.extend(
                self._filter_retained_scoped_decls(scoped_items, index, latest_generation)
            )
        return retained_root, retained_scoped, next_start_id

    @staticmethod
    def _generation_key(decl: ImportDecl) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return *decl*'s replacement-generation key: its region path and module identity.

        A root import (empty ``scope_path``) and a region-scoped import at
        some region path each own an independent chronological replacement
        decision per module -- keying only on module identity would let an
        import at one region path replace an unrelated declaration at
        another region path, or at the root.
        """
        return (tuple(segment.name for segment in decl.scope_path), tuple(decl.module_path))

    @staticmethod
    def _expand_decls(
        decls: tuple[ImportDecl, ...],
        roots: RootSet,
        next_start_id: int,
    ) -> tuple[list[ImportDecl], int]:
        """Expand every wildcard in *decls* into exact-module declarations."""
        from dataclasses import replace

        from agm.agl.modules.resolver import expand_wildcard

        expanded: list[ImportDecl] = []
        for decl in decls:
            if not decl.wildcard:
                expanded.append(decl)
                continue
            wildcard_origin_node_id = decl.node_id
            for module in expand_wildcard(tuple(decl.module_path), roots, span=decl.span):
                expanded.append(
                    replace(
                        decl,
                        module_path=module.segments,
                        wildcard=False,
                        wildcard_origin_node_id=wildcard_origin_node_id,
                        node_id=next_start_id,
                    )
                )
                next_start_id += 1
        return expanded, next_start_id

    @staticmethod
    def _retained_scoped_import_items(
        items: tuple[Item, ...],
    ) -> tuple[ImportDecl | ScopeRegion, ...]:
        """Extract region-scoped imports, retaining their enclosing regions.

        A region-scoped ``import``'s bare contribution is only meaningful
        nested exactly where it was written -- unlike a root ``import``,
        already retained by the flat, module-wide accumulation channel -- so
        only a scoped one (``item.scope_path`` non-empty, which holds
        precisely when it is a region's own item) is captured here.
        """
        from dataclasses import replace

        from agm.agl.syntax.nodes import ImportDecl, ScopeRegion

        retained: list[ImportDecl | ScopeRegion] = []
        for item in items:
            if isinstance(item, ImportDecl) and item.scope_path:
                retained.append(item)
            elif isinstance(item, ScopeRegion):
                nested = EntryPipeline._retained_scoped_import_items(item.items)
                if nested:
                    retained.append(replace(item, items=nested))
        return tuple(retained)

    @staticmethod
    def _scoped_import_decls(
        items: tuple[ImportDecl | ScopeRegion, ...],
    ) -> tuple[ImportDecl, ...]:
        """Flatten the region-scoped imports retained inside *items*."""
        from agm.agl.syntax.nodes import ImportDecl, static_items

        return tuple(
            item
            for item in static_items(cast("tuple[Item, ...]", items))
            if isinstance(item, ImportDecl)
        )

    @staticmethod
    def _rewrite_scoped_items(
        items: tuple[ImportDecl | ScopeRegion, ...],
        *,
        rewrite_import: Callable[[ImportDecl], Iterable[ImportDecl]],
    ) -> tuple[ImportDecl | ScopeRegion, ...]:
        """Rebuild retained scoped imports, dropping empty region wrappers."""
        from dataclasses import replace

        from agm.agl.syntax.nodes import ImportDecl

        rewritten: list[ImportDecl | ScopeRegion] = []
        for item in items:
            if isinstance(item, ImportDecl):
                rewritten.extend(rewrite_import(item))
            else:
                nested = EntryPipeline._rewrite_scoped_items(
                    cast("tuple[ImportDecl | ScopeRegion, ...]", item.items),
                    rewrite_import=rewrite_import,
                )
                if nested:
                    rewritten.append(replace(item, items=nested))
        return tuple(rewritten)

    @staticmethod
    def _expand_retained_scoped_imports(
        items: tuple[ImportDecl | ScopeRegion, ...],
        roots: RootSet,
        next_start_id: int,
    ) -> tuple[tuple[ImportDecl | ScopeRegion, ...], int]:
        """Expand scoped wildcard imports while preserving their region wrappers."""
        node_id = next_start_id

        def expand(decl: ImportDecl) -> tuple[ImportDecl, ...]:
            nonlocal node_id
            expanded, node_id = EntryPipeline._expand_decls((decl,), roots, node_id)
            return tuple(expanded)

        return EntryPipeline._rewrite_scoped_items(items, rewrite_import=expand), node_id

    @staticmethod
    def _filter_retained_scoped_decls(
        items: tuple[ImportDecl | ScopeRegion, ...],
        generation: int,
        latest_generation: Mapping[tuple[tuple[str, ...], tuple[str, ...]], int],
    ) -> tuple[ImportDecl | ScopeRegion, ...]:
        """Keep the newest retained import at each scoped module path."""

        def keep_newest_import(decl: ImportDecl) -> tuple[ImportDecl, ...]:
            key = EntryPipeline._generation_key(decl)
            return (decl,) if latest_generation[key] == generation else ()

        return EntryPipeline._rewrite_scoped_items(items, rewrite_import=keep_newest_import)

    def _retain_import_context(
        self,
        entry_imports: tuple[ImportDecl, ...],
        entry_uses: tuple[ImportDecl | ScopeRegion, ...],
    ) -> None:
        """Retain one successful entry's aligned root/scoped import generation.

        An entry that declares neither contributes no generation: an empty one
        can never own a module's newest generation nor retain anything, so
        recording it would only lengthen every later entry's replay.
        """
        if not entry_imports and not entry_uses:
            return
        self._ctx._accumulated_imports.append(entry_imports)
        self._ctx._accumulated_scoped_imports.append(entry_uses)

    def _builtin_host_settings(self) -> dict[str | BuiltinVarKey, Value]:
        """Combine persistent engine and standard-library binding state."""
        settings: dict[str | BuiltinVarKey, Value] = {}
        for engine_key, value in self._ctx._current.items():
            settings[engine_key] = value
        for builtin_key, value in self._ctx._builtin_var_values.items():
            settings[builtin_key] = value
        return settings

    def _persist_interpreter_settings(self, interp: "IrInterpreter", trace: "TraceStore") -> None:
        """Persist completed setting writes and the live trace destination."""
        self._ctx._update_engine_settings(interp)
        # A store disabled by a failed write nulls its own path for the rest of
        # the entry; that is a transient I/O condition, not a destination the
        # session should adopt.  Keeping the session path lets the next entry
        # retry at the original destination.  A store that settled into no-log
        # mode deliberately (``std/config::log := false``) does persist ``None``.
        if not trace.disabled:
            self._ctx._trace_path = trace.path
