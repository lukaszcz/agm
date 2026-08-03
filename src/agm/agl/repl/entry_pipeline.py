"""Multi-module REPL program pipeline collaborator.

Implements the build_repl_graph → resolve_program → check_program → match
compilation → incremental link/exec pipeline for REPL entries that contain
import declarations or have cached library modules from prior entries. Driven
by ``ReplSession`` via the narrow ``EntryPipelineCtx`` Protocol. Must NOT import
``session`` (no cycle).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Protocol, cast

from agm.agl.diagnostics import Diagnostic
from agm.agl.repl.entry import EntryKind, EntryResult

if TYPE_CHECKING:
    from pathlib import Path

    from agm.agl.eval.ir_interpreter import IrInterpreter
    from agm.agl.ir.builtin_nominals import BuiltinNominals
    from agm.agl.ir.contracts import ContractPayload
    from agm.agl.ir.ids import NominalId, SymbolId
    from agm.agl.ir.program import NominalDescriptor
    from agm.agl.lower import LinkImage
    from agm.agl.matchcompile import MatchCompiledProgram
    from agm.agl.modules.ids import ModuleId
    from agm.agl.modules.loader import LoadedModule
    from agm.agl.modules.roots import RootSet
    from agm.agl.runtime.host_settings import HostSettingsPolicy
    from agm.agl.runtime.trace import TraceStore
    from agm.agl.runtime.types import HostEnvironment
    from agm.agl.scope.symbols import ConstructorRef, ScopeNode
    from agm.agl.semantics.types import Type
    from agm.agl.semantics.values import EnumValue, Frame, Value
    from agm.agl.syntax.advisories import SpacedQualifier
    from agm.agl.syntax.nodes import ImportDecl, Item, OpenDecl, Program, ScopeRegion
    from agm.agl.typecheck.env import CheckedModule, TypeEnvironment
    from agm.agl.typecheck.program import CheckedProgram


# ---------------------------------------------------------------------------
# Narrow context Protocol
# ---------------------------------------------------------------------------


class EntryPipelineCtx(Protocol):
    """The minimal ReplSession surface the program pipeline needs."""

    _loaded_lib_modules: dict[ModuleId, LoadedModule]
    _accumulated_imports: list[tuple[ImportDecl, ...]]
    _accumulated_opens: list[tuple[OpenDecl | ImportDecl | ScopeRegion, ...]]
    _link_image: LinkImage
    _ir_base_frame: Frame
    _session_scope: ScopeNode
    _session_scope_nodes: dict[tuple[str, ...], ScopeNode]
    _session_type_paths: dict[tuple[str, ...], str | None]
    _type_env: TypeEnvironment
    _ambient_constructor_candidates: dict[str, tuple[ConstructorRef, ...]]
    _ambient_type_names: frozenset[str]
    _trace_path: Path | None
    _default_strict_json: bool
    _default_loop_limit: int | None
    _default_call_depth_limit: int
    _default_stdlib: bool
    _shell_exec_timeout: float | None
    _persisted_host_settings: dict[str, Value]
    _persisted_timeout_setting: EnumValue
    _host_settings_policy: HostSettingsPolicy | None

    def _ensure_roots(self) -> RootSet: ...

    def _ambient_agents(self, host_env: HostEnvironment) -> frozenset[str]: ...

    def _fail(self, diagnostics: list[Diagnostic], warnings: list[Diagnostic]) -> EntryResult: ...

    def _build_check_only_result(
        self, program: Program, checked: CheckedModule, warnings: list[Diagnostic]
    ) -> EntryResult: ...

    def _pre_eval_param_check(
        self, program: Program, checked: CheckedModule, warnings: list[Diagnostic]
    ) -> tuple[EntryResult | None, dict[str, Value], str | None, dict[str, object]]: ...

    def _update_engine_settings(
        self,
        *,
        strict_json: bool,
        loop_limit: int | None,
        shell_exec_timeout: float | None,
    ) -> None: ...

    def _promote_ir_state(
        self,
        *,
        text: str,
        program: Program,
        checked: CheckedModule,
        next_start_id: int,
        entry_program_name: str | None,
        entry_active_config: dict[str, object],
        partial: bool,
        promoted_declaration_ids: frozenset[int],
    ) -> tuple[str, ...]: ...

    def _classify(self, program: Program) -> tuple[EntryKind, str | None]: ...

    def frame_value(self, symbol: SymbolId | None) -> Value | None: ...

    def _echo_data_ir(
        self, program: Program, checked: CheckedModule, captured: Value | None
    ) -> tuple[Value | None, Type | None]: ...

    def _quote_strings_for_entry(self, program: Program) -> bool: ...


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
        from agm.agl.modules.ids import ENTRY_ID
        from agm.agl.modules.loader import build_repl_graph
        from agm.agl.parser import AglSyntaxError
        from agm.agl.scope import AglScopeError
        from agm.agl.scope.program import resolve_program
        from agm.agl.typecheck import AglTypeError
        from agm.agl.typecheck.program import check_program

        roots = self._ctx._ensure_roots()

        try:
            entry_program, next_start_id, entry_imports, entry_opens = self._prepare_entry_program(
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
        except AglError as exc:
            return self._ctx._fail([exc.to_diagnostic()], tab_warnings)
        except Exception as exc:
            return self._ctx._fail([Diagnostic(message=str(exc), line=1)], tab_warnings)

        try:
            resolved_program = resolve_program(
                graph,
                ambient_agents=self._ctx._ambient_agents(host_env),
                entry_ambient_constructor_candidates=self._ctx._ambient_constructor_candidates,
                entry_ambient_type_names=self._ctx._ambient_type_names,
                entry_parent_scope=self._ctx._session_scope,
                entry_repl_session_scope=self._ctx._session_scope,
                entry_repl_session_scope_nodes=self._ctx._session_scope_nodes,
                entry_repl_session_type_paths=self._ctx._session_type_paths,
            )
        except AglScopeError as exc:
            return self._ctx._fail([exc.to_diagnostic()], tab_warnings)

        try:
            checked_program = check_program(
                resolved_program, host_env.capabilities, entry_seed_env=self._ctx._type_env
            )
        except AglTypeError as exc:
            return self._ctx._fail([exc.to_diagnostic()], tab_warnings)

        entry_cm = checked_program.modules[ENTRY_ID]

        # Collect warnings from all passes.
        warnings: list[Diagnostic] = [
            *tab_warnings,
            *resolved_program.warnings,
            *checked_program.warnings,
        ]

        from agm.agl.matchcompile import compile_program_matches, diagnostics_from_match_issues

        match_result = compile_program_matches(checked_program)
        if match_result.compiled is None:
            return self._ctx._fail(
                list(diagnostics_from_match_issues(match_result.issues)), warnings
            )
        compiled = match_result.compiled
        from agm.agl.matchcompile import MatchCompiledProgram

        assert isinstance(compiled, MatchCompiledProgram)

        checked = self._checked_program_from_module(entry_cm)
        if check_only:
            return self._ctx._build_check_only_result(orig_program, checked, warnings)

        pre_eval_result, param_values, entry_program_name, entry_active_config = (
            self._ctx._pre_eval_param_check(orig_program, checked, warnings)
        )
        if pre_eval_result is not None:
            return pre_eval_result

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
            entry_imports=entry_imports,
            entry_opens=entry_opens,
            param_values=param_values,
            entry_program_name=entry_program_name,
            entry_active_config=entry_active_config,
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
        from agm.agl.scope.program import resolve_program
        from agm.agl.typecheck.program import check_program

        roots = self._ctx._ensure_roots()
        entry_program, next_start_id, _entry_imports, _entry_opens = self._prepare_entry_program(
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
        )
        resolved_program = resolve_program(
            graph,
            ambient_agents=self._ctx._ambient_agents(host_env),
            entry_ambient_constructor_candidates=self._ctx._ambient_constructor_candidates,
            entry_ambient_type_names=self._ctx._ambient_type_names,
            entry_parent_scope=self._ctx._session_scope,
            entry_repl_session_scope=self._ctx._session_scope,
            entry_repl_session_scope_nodes=self._ctx._session_scope_nodes,
            entry_repl_session_type_paths=self._ctx._session_type_paths,
        )
        return check_program(
            resolved_program, host_env.capabilities, entry_seed_env=self._ctx._type_env
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
    ) -> tuple[
        Program, int, tuple[ImportDecl, ...], tuple[OpenDecl | ImportDecl | ScopeRegion, ...]
    ]:
        """Expand current wildcards, then inject retained imports and scope opens.

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
        import/export decls, retained opens/regions, this entry's remaining
        items]. This entry's own root header decls are hoisted ahead of the
        retained opens/regions -- rather than left in their original,
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
        entry_opens = self._retained_open_items(program.body.items)
        expanded, next_start_id, expanded_imports = self._expand_entry_wildcards(
            program, next_start_id, roots
        )
        # The expanded root imports feed the preamble's newest-generation
        # decision, so this entry's wildcards are globbed once, not again.
        import_preamble, open_preamble, next_start_id = self._retained_preamble(
            expanded_imports, entry_opens, roots, next_start_id
        )
        entry_headers, entry_rest = self._partition_entry_root_headers(expanded.body.items)
        preamble: list[Item] = [*import_preamble, *entry_headers, *open_preamble]
        rest_program = (
            expanded if not entry_headers else self._replace_body_items(expanded, entry_rest)
        )
        return (
            self._prepend_items(rest_program, preamble),
            next_start_id,
            entry_imports,
            entry_opens,
        )

    @staticmethod
    def _partition_entry_root_headers(items: tuple[Item, ...]) -> tuple[list[Item], list[Item]]:
        """Split *items* into this entry's own root header decls and everything else.

        A root ``import``/``export`` (empty ``scope_path``) is the entry's
        own header contribution, distinct from the *retained* root imports
        already folded into ``import_preamble`` -- this partition finds the
        current entry's, so ``_prepare_entry_program`` can hoist them ahead
        of retained opens/regions.
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
        entry_imports: tuple[ImportDecl, ...],
        entry_opens: tuple[OpenDecl | ImportDecl | ScopeRegion, ...],
        param_values: dict[str, Value],
        entry_program_name: str | None,
        entry_active_config: dict[str, object],
        contract_payloads: Mapping[int, "ContractPayload"],
    ) -> EntryResult:
        """Lower and execute one program entry in the persistent IR image."""
        from agm.agl.eval.ir_interpreter import IrInterpreter
        from agm.agl.lower import lower_repl_program
        from agm.agl.pipeline import _wire_extern_registry, exception_value_to_run_error
        from agm.agl.runtime.params import _materialize_ir_contracts
        from agm.agl.runtime.request import AgentCancelled
        from agm.agl.runtime.trace import TraceStore
        from agm.agl.semantics.exceptions import AglRaise

        # Companion paths for every module the checked program can reach: prior
        # entries' cached library modules plus this entry's newly linked ones.
        # ``_wire_extern_registry`` imports/resolves only what is not already
        # cached on ``host_env.extern_registry`` (mutated in place), so a
        # companion imports exactly once per session even across entries.
        companion_paths: dict[ModuleId, Path | None] = {
            mid: lm.companion_path for mid, lm in self._ctx._loaded_lib_modules.items()
        }
        companion_paths.update({mid: lm.companion_path for mid, lm in new_modules.items()})
        extern_diagnostics = _wire_extern_registry(
            checked=checked_program,
            capabilities=host_env.capabilities,
            registry=host_env.extern_registry,
            companion_paths=companion_paths,
        )
        if extern_diagnostics:
            return self._ctx._fail(extern_diagnostics, warnings)

        nominal_snapshot = self._ctx._link_image.snapshot_nominals()
        builtin_nominal_snapshot = self._ctx._link_image.snapshot_builtin_nominals()
        lowered = lower_repl_program(
            compiled,
            image=self._ctx._link_image,
            source_text=text,
            contract_payloads=contract_payloads,
        )
        ir_params = {
            param.symbol: param_values[param.public_name]
            for param in lowered.program.params
            if param.public_name in param_values
        }
        host_contracts, _ = _materialize_ir_contracts(lowered.program, host_env.codecs)
        trace = TraceStore(path=self._ctx._trace_path)
        trace.run_start()
        if self._ctx._host_settings_policy is not None:
            from agm.agl.runtime.host_settings import HostSettingsReconfigurer

            reconfigurer: HostSettingsReconfigurer | None = HostSettingsReconfigurer(
                registry=host_env.registry,
                trace=trace,
                policy=self._ctx._host_settings_policy,
            )
        else:
            reconfigurer = None
        interp = IrInterpreter(
            lowered.program,
            registry=host_env.registry,
            strict_json=self._ctx._default_strict_json,
            loop_limit=self._ctx._default_loop_limit,
            max_call_depth=self._ctx._default_call_depth_limit,
            shell_exec_timeout=self._ctx._shell_exec_timeout,
            trace=trace,
            param_values=ir_params,
            host_contracts=host_contracts,
            base_frame=self._ctx._ir_base_frame,
            extern_registry=host_env.extern_registry,
            host_reconfigurer=reconfigurer,
            builtin_host_settings={
                **self._ctx._persisted_host_settings,
                "timeout": self._ctx._persisted_timeout_setting,
            },
        )

        def completed_declaration_ids() -> frozenset[int]:
            # The entry frame is always populated by the closure pre-pass
            # before params run (see ``IrInterpreter.run``), so a declaration
            # completed before a failing param default stays promoted. The
            # promotion plan itself is conservative: it excludes params whose
            # symbols were not installed and applies the declaration-dependency
            # fixpoint.
            return lowered.promotion_plan.completed_declaration_ids(
                len(interp.module_initializer_values.get(lowered.program.entry_module, ())),
                interp.entry_param_symbols_installed,
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
            self._persist_interpreter_settings(interp, trace)
            promoted = completed_declaration_ids()
            installed = self._ctx._promote_ir_state(
                text=text,
                program=orig_program,
                checked=checked,
                next_start_id=new_next_id,
                entry_program_name=entry_program_name,
                entry_active_config=entry_active_config,
                partial=True,
                promoted_declaration_ids=promoted,
            )
            self._restore_unpromoted_entry_nominals(
                orig_program,
                promoted,
                nominal_snapshot,
                builtin_nominal_snapshot,
            )
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
                installed=installed,
            )
        except (AgentCancelled, KeyboardInterrupt) as exc:
            cancellation_message = (
                "Agent call cancelled — entry aborted."
                if isinstance(exc, AgentCancelled)
                else "Entry interrupted — entry aborted."
            )
            trace.run_end(ok=False)
            self._persist_interpreter_settings(interp, trace)
            promoted = completed_declaration_ids()
            installed = self._ctx._promote_ir_state(
                text=text,
                program=orig_program,
                checked=checked,
                next_start_id=new_next_id,
                entry_program_name=entry_program_name,
                entry_active_config=entry_active_config,
                partial=True,
                promoted_declaration_ids=promoted,
            )
            self._restore_unpromoted_entry_nominals(
                orig_program,
                promoted,
                nominal_snapshot,
                builtin_nominal_snapshot,
            )
            kind, name = self._ctx._classify(orig_program)
            return EntryResult(
                kind=kind,
                name=name,
                value=None,
                value_type=None,
                diagnostics=[Diagnostic(message=cancellation_message, line=1)],
                warnings=warnings,
                error=None,
                ok=False,
                trace_path=self._ctx._trace_path,
                installed=installed,
            )
        trace.run_end(ok=True)
        # Setting writes are ordinary non-transactional mutations: persist all
        # effects that completed, on success or before a later runtime failure.
        self._persist_interpreter_settings(interp, trace)
        self._ctx._promote_ir_state(
            text=text,
            program=orig_program,
            checked=checked,
            next_start_id=new_next_id,
            entry_program_name=entry_program_name,
            entry_active_config=entry_active_config,
            partial=False,
            promoted_declaration_ids=lowered.promotion_plan.completed_declaration_ids(
                len(lowered.program.modules[lowered.program.entry_module].initializers),
                interp.entry_param_symbols_installed,
            ),
        )
        self._ctx._loaded_lib_modules.update(new_modules)
        self._ctx._link_image.mark_linked(
            mid for mid in checked_program.modules if not mid.is_entry
        )
        self._retain_import_context(entry_imports, entry_opens)
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
        entry_opens: tuple[OpenDecl | ImportDecl | ScopeRegion, ...],
        roots: RootSet,
        next_start_id: int,
    ) -> tuple[list[ImportDecl], list[OpenDecl | ImportDecl | ScopeRegion], int]:
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
        Scope ``open`` declarations are not imports and remain cumulative.
        """
        generations: list[
            tuple[list[ImportDecl], tuple[OpenDecl | ImportDecl | ScopeRegion, ...]]
        ] = []
        latest_generation: dict[tuple[tuple[str, ...], tuple[str, ...]], int] = {}
        for retained_root_decls, scoped_items in zip(
            self._ctx._accumulated_imports, self._ctx._accumulated_opens, strict=True
        ):
            expanded_root_decls, next_start_id = self._expand_decls(
                retained_root_decls, roots, next_start_id
            )
            expanded_scoped, next_start_id = self._expand_retained_scoped_imports(
                scoped_items, roots, next_start_id
            )
            index = len(generations)
            generations.append((expanded_root_decls, expanded_scoped))
            for decl in (*expanded_root_decls, *self._scoped_import_decls(expanded_scoped)):
                latest_generation[self._generation_key(decl)] = index

        # *entry_imports* arrives already expanded; only the scoped ones still
        # need their module identities resolved. Wildcard expansion has one
        # definition: the node ids minted here are discarded with the
        # rebuilt declarations, since only each declaration's region path
        # and module identity are wanted.
        current_decls, _ = self._expand_decls(
            (*entry_imports, *self._scoped_import_decls(entry_opens)), roots, 0
        )
        current_index = len(generations)
        latest_generation.update(
            (self._generation_key(decl), current_index) for decl in current_decls
        )

        retained_root: list[ImportDecl] = []
        retained_scoped: list[OpenDecl | ImportDecl | ScopeRegion] = []
        for index, (root_decls, scoped_items) in enumerate(generations):
            retained_root.extend(
                decl
                for decl in root_decls
                if latest_generation[self._generation_key(decl)] == index
            )
            retained_scoped.extend(
                self._filter_retained_scoped_imports(
                    scoped_items,
                    index,
                    latest_generation,
                )
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
            for module in expand_wildcard(tuple(decl.module_path), roots, span=decl.span):
                expanded.append(
                    replace(
                        decl,
                        module_path=module.segments,
                        wildcard=False,
                        node_id=next_start_id,
                    )
                )
                next_start_id += 1
        return expanded, next_start_id

    @staticmethod
    def _retained_open_items(
        items: tuple[Item, ...],
    ) -> tuple[OpenDecl | ImportDecl | ScopeRegion, ...]:
        """Extract scope opens and region-scoped imports, retaining their enclosing regions.

        A region-scoped ``import``'s bare contribution is only meaningful
        nested exactly where it was written -- unlike a root ``import``,
        already retained by the flat, module-wide accumulation channel -- so
        only a scoped one (``item.scope_path`` non-empty, which holds
        precisely when it is a region's own item) is captured here.
        """
        from dataclasses import replace

        from agm.agl.syntax.nodes import ImportDecl, OpenDecl, ScopeRegion

        retained: list[OpenDecl | ImportDecl | ScopeRegion] = []
        for item in items:
            if isinstance(item, OpenDecl):
                retained.append(item)
            elif isinstance(item, ImportDecl) and item.scope_path:
                retained.append(item)
            elif isinstance(item, ScopeRegion):
                nested = EntryPipeline._retained_open_items(item.items)
                if nested:
                    retained.append(replace(item, items=nested))
        return tuple(retained)

    @staticmethod
    def _scoped_import_decls(
        items: tuple[OpenDecl | ImportDecl | ScopeRegion, ...],
    ) -> tuple[ImportDecl, ...]:
        """Flatten the region-scoped imports retained inside *items*."""
        from agm.agl.syntax.nodes import ImportDecl, static_items

        return tuple(
            item
            for item in static_items(cast("tuple[Item, ...]", items))
            if isinstance(item, ImportDecl)
        )

    @staticmethod
    def _rewrite_scoped_imports(
        items: tuple[OpenDecl | ImportDecl | ScopeRegion, ...],
        rewrite: Callable[[ImportDecl], Iterable[ImportDecl]],
    ) -> tuple[OpenDecl | ImportDecl | ScopeRegion, ...]:
        """Rebuild a retained scope tree, replacing each import through *rewrite*.

        The single structure-preserving walk over a retained
        ``open``/``import``/region tree: a scope ``open`` is carried through
        untouched, a region is rebuilt around its rewritten members, and a
        region left with no members is dropped -- one place that decides how
        a retained region survives, whatever the caller does to its imports.
        """
        from dataclasses import replace

        from agm.agl.syntax.nodes import ImportDecl, ScopeRegion

        rewritten: list[OpenDecl | ImportDecl | ScopeRegion] = []
        for item in items:
            if isinstance(item, ImportDecl):
                rewritten.extend(rewrite(item))
            elif isinstance(item, ScopeRegion):
                nested = EntryPipeline._rewrite_scoped_imports(
                    cast("tuple[OpenDecl | ImportDecl | ScopeRegion, ...]", item.items),
                    rewrite,
                )
                if nested:
                    rewritten.append(replace(item, items=nested))
            else:
                rewritten.append(item)
        return tuple(rewritten)

    @staticmethod
    def _expand_retained_scoped_imports(
        items: tuple[OpenDecl | ImportDecl | ScopeRegion, ...],
        roots: RootSet,
        next_start_id: int,
    ) -> tuple[tuple[OpenDecl | ImportDecl | ScopeRegion, ...], int]:
        """Expand scoped wildcard imports while preserving their region wrappers."""
        node_id = next_start_id

        def expand(decl: ImportDecl) -> tuple[ImportDecl, ...]:
            nonlocal node_id
            expanded, node_id = EntryPipeline._expand_decls((decl,), roots, node_id)
            return tuple(expanded)

        return EntryPipeline._rewrite_scoped_imports(items, expand), node_id

    @staticmethod
    def _filter_retained_scoped_imports(
        items: tuple[OpenDecl | ImportDecl | ScopeRegion, ...],
        generation: int,
        latest_generation: Mapping[tuple[tuple[str, ...], tuple[str, ...]], int],
    ) -> tuple[OpenDecl | ImportDecl | ScopeRegion, ...]:
        """Filter scoped imports by region path and module, keeping cumulative scope opens."""

        def keep_newest(decl: ImportDecl) -> tuple[ImportDecl, ...]:
            key = EntryPipeline._generation_key(decl)
            return (decl,) if latest_generation[key] == generation else ()

        return EntryPipeline._rewrite_scoped_imports(items, keep_newest)

    def _retain_import_context(
        self,
        entry_imports: tuple[ImportDecl, ...],
        entry_opens: tuple[OpenDecl | ImportDecl | ScopeRegion, ...],
    ) -> None:
        """Retain one successful entry's aligned root/scoped import generation.

        An entry that declares neither contributes no generation: an empty one
        can never own a module's newest generation nor retain anything, so
        recording it would only lengthen every later entry's replay.
        """
        if not entry_imports and not entry_opens:
            return
        self._ctx._accumulated_imports.append(entry_imports)
        self._ctx._accumulated_opens.append(entry_opens)

    def _persist_interpreter_settings(self, interp: "IrInterpreter", trace: "TraceStore") -> None:
        """Persist completed setting writes and the live trace destination."""
        self._ctx._update_engine_settings(
            strict_json=interp.strict_json,
            loop_limit=interp.loop_limit,
            shell_exec_timeout=interp.shell_exec_timeout,
        )
        self._ctx._persisted_host_settings = interp.builtin_host_settings
        self._ctx._persisted_timeout_setting = interp.timeout_setting
        # A store disabled by a failed write nulls its own path for the rest of
        # the entry; that is a transient I/O condition, not a destination the
        # session should adopt.  Keeping the session path lets the next entry
        # retry at the original destination.  A store that settled into no-log
        # mode deliberately (``std/config::log := false``) does persist ``None``.
        if not trace.disabled:
            self._ctx._trace_path = trace.path

    def _restore_unpromoted_entry_nominals(
        self,
        program: Program,
        promoted_declaration_ids: frozenset[int],
        nominal_snapshot: Mapping["NominalId", "NominalDescriptor"],
        builtin_nominal_snapshot: "BuiltinNominals",
    ) -> None:
        """Rollback nominal metadata whose declaration did not complete."""
        from agm.agl.ir.ids import NominalId
        from agm.agl.modules.ids import ENTRY_ID
        from agm.agl.syntax.nodes import EnumDef, ExceptionDef, RecordDef, ScopeRegion

        def type_declarations(
            items: tuple[object, ...],
        ) -> Iterator[RecordDef | EnumDef | ExceptionDef]:
            for item in items:
                if isinstance(item, ScopeRegion):
                    yield from type_declarations(item.items)
                elif isinstance(item, (RecordDef, EnumDef, ExceptionDef)):
                    yield item

        unpromoted = tuple(
            item
            for item in type_declarations(program.body.items)
            if item.node_id not in promoted_declaration_ids
        )
        nominal_ids = tuple(
            NominalId(ENTRY_ID, item.name, tuple(segment.name for segment in item.scope_path))
            for item in unpromoted
        )
        self._ctx._link_image.restore_nominals(nominal_snapshot, nominal_ids)
        self._ctx._link_image.restore_builtin_nominals(
            builtin_nominal_snapshot,
            (item.name for item in unpromoted if item.is_builtin),
        )
