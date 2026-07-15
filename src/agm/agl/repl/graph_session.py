"""Multi-module REPL graph-mode pipeline collaborator.

Implements the build_repl_graph → resolve_graph → check_graph → match
compilation → incremental link/exec pipeline for REPL entries that contain
import declarations or have cached library modules from prior entries. Driven
by ``ReplSession`` via the narrow ``GraphSessionCtx`` Protocol. Must NOT import
``session`` (no cycle).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

from agm.agl.diagnostics import Diagnostic
from agm.agl.repl.entry import EntryKind, EntryResult

if TYPE_CHECKING:
    from pathlib import Path

    from agm.agl.ir.contracts import ContractPayload
    from agm.agl.ir.ids import Location, NominalId
    from agm.agl.ir.program import NominalDescriptor
    from agm.agl.lower import LinkImage
    from agm.agl.matchcompile import MatchCompiledModuleGraph
    from agm.agl.modules.ids import ModuleId
    from agm.agl.modules.loader import LoadedModule
    from agm.agl.modules.roots import RootSet
    from agm.agl.runtime.types import HostEnvironment
    from agm.agl.scope.symbols import ConstructorRef, ScopeNode
    from agm.agl.semantics.types import Type
    from agm.agl.semantics.values import Frame, Value
    from agm.agl.syntax.nodes import ImportDecl, Program
    from agm.agl.syntax.spans import SourceSpan
    from agm.agl.typecheck.env import CheckedProgram, TypeEnvironment
    from agm.agl.typecheck.graph import CheckedModule, CheckedModuleGraph


# ---------------------------------------------------------------------------
# Narrow context Protocol
# ---------------------------------------------------------------------------


class GraphSessionCtx(Protocol):
    """The minimal ReplSession surface the graph-mode pipeline needs."""

    _loaded_lib_modules: dict[ModuleId, LoadedModule]
    _accumulated_imports: list[ImportDecl]
    _link_image: LinkImage
    _ir_base_frame: Frame
    _session_scope: ScopeNode
    _type_env: TypeEnvironment
    _ambient_constructor_candidates: dict[str, tuple[ConstructorRef, ...]]
    _ambient_type_names: frozenset[str]
    _trace_path: Path | None
    _default_strict_json: bool
    _default_call_depth_limit: int
    _shell_exec_timeout: float | None

    def _ensure_roots(self) -> RootSet: ...

    def _ambient_agents(self, host_env: HostEnvironment) -> frozenset[str]: ...

    def _fail(self, diagnostics: list[Diagnostic], warnings: list[Diagnostic]) -> EntryResult: ...

    def _build_check_only_result(
        self, program: Program, checked: CheckedProgram, warnings: list[Diagnostic]
    ) -> EntryResult: ...

    def _pre_eval_param_check(
        self, program: Program, checked: CheckedProgram, warnings: list[Diagnostic]
    ) -> tuple[EntryResult | None, dict[str, Value], str | None, dict[str, object]]: ...

    def _build_config_base(
        self, effective_config: dict[str, object]
    ) -> dict[str, Value]: ...

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
        checked: CheckedProgram,
        next_start_id: int,
        entry_program_name: str | None,
        entry_active_config: dict[str, object],
        partial: bool,
        failure_span: SourceSpan | Location | None,
    ) -> tuple[str, ...]: ...

    def _classify(self, program: Program) -> tuple[EntryKind, str | None]: ...

    def _echo_data_ir(
        self, program: Program, checked: CheckedProgram, captured: Value | None
    ) -> tuple[Value | None, Type | None]: ...

    def _quote_strings_for_entry(self, program: Program) -> bool: ...


# ---------------------------------------------------------------------------
# Collaborator class
# ---------------------------------------------------------------------------


class GraphSession:
    """Graph-mode pipeline collaborator for ``ReplSession``.

    Instantiated once per ``ReplSession`` (``self._graph_session``).  Holds
    no state of its own — all session state is borrowed via ``GraphSessionCtx``.
    """

    def __init__(self, ctx: GraphSessionCtx) -> None:
        self._ctx = ctx

    def eval_entry_graph_mode(
        self,
        *,
        text: str,
        orig_program: Program,
        pipeline_program: Program,
        host_env: HostEnvironment,
        tab_warnings: list[Diagnostic],
        next_start_id: int,
        check_only: bool,
    ) -> EntryResult:
        """Graph-mode pipeline for REPL entries that have imports or cached lib modules.

        Builds the module graph from the already-parsed *pipeline_program*, runs
        the full scope/typecheck/match-compilation passes with the session
        context, then returns a check-only result or lowers and evaluates.
        """
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
        from agm.agl.scope.graph import resolve_graph
        from agm.agl.typecheck import AglTypeError
        from agm.agl.typecheck.graph import check_graph

        roots = self._ctx._ensure_roots()

        # Inject accumulated import declarations from prior entries at the head
        # of the pipeline program so that open imports persist across entries.
        entry_program = self._inject_accumulated_imports(pipeline_program)

        try:
            graph, new_next_id, new_modules = build_repl_graph(
                entry_program,
                next_start_id,
                path=None,
                cached=self._ctx._loaded_lib_modules,
                roots=roots,
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
        except Exception as exc:
            return self._ctx._fail([Diagnostic(message=str(exc), line=1)], tab_warnings)

        try:
            rgraph = resolve_graph(
                graph,
                ambient_agents=self._ctx._ambient_agents(host_env),
                entry_ambient_constructor_candidates=self._ctx._ambient_constructor_candidates,
                entry_ambient_type_names=self._ctx._ambient_type_names,
                entry_parent_scope=self._ctx._session_scope,
                entry_repl_session_scope=self._ctx._session_scope,
            )
        except AglScopeError as exc:
            return self._ctx._fail([exc.to_diagnostic()], tab_warnings)

        try:
            cgraph = check_graph(
                rgraph, host_env.capabilities, entry_seed_env=self._ctx._type_env
            )
        except AglTypeError as exc:
            return self._ctx._fail([exc.to_diagnostic()], tab_warnings)

        entry_cm = cgraph.modules[ENTRY_ID]

        # Collect warnings from all passes.
        warnings: list[Diagnostic] = [
            *tab_warnings,
            *rgraph.warnings,
            *cgraph.warnings,
        ]

        from agm.agl.matchcompile import compile_graph_matches, diagnostics_from_match_issues

        match_result = compile_graph_matches(cgraph)
        if match_result.compiled is None:
            return self._ctx._fail(
                list(diagnostics_from_match_issues(match_result.issues)), warnings
            )
        compiled_graph = match_result.compiled
        from agm.agl.matchcompile import MatchCompiledModuleGraph

        assert isinstance(compiled_graph, MatchCompiledModuleGraph)

        checked = self._checked_program_from_module(entry_cm)
        if check_only:
            return self._ctx._build_check_only_result(orig_program, checked, warnings)

        pre_eval_result, param_values, entry_program_name, entry_active_config = (
            self._ctx._pre_eval_param_check(orig_program, checked, warnings)
        )
        if pre_eval_result is not None:
            return pre_eval_result

        from agm.agl.pipeline import _materialize_graph_custom_contract_payloads

        contract_payloads, contract_errors = _materialize_graph_custom_contract_payloads(
            cgraph,
            host_env.codecs,
        )
        if contract_errors:
            return self._ctx._fail(contract_errors, warnings)

        return self._evaluate_ir_graph_mode(
            text=text,
            orig_program=orig_program,
            checked=checked,
            entry_cm=entry_cm,
            cgraph=cgraph,
            compiled_graph=compiled_graph,
            host_env=host_env,
            warnings=warnings,
            new_next_id=new_next_id,
            new_modules=new_modules,
            param_values=param_values,
            entry_program_name=entry_program_name,
            entry_active_config=entry_active_config,
            contract_payloads=contract_payloads,
        )

    @staticmethod
    def _checked_program_from_module(entry: CheckedModule) -> CheckedProgram:
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
            argument_bindings=entry.argument_bindings,
            partial_calls=entry.partial_calls,
        )

    def _inject_accumulated_imports(self, program: Program) -> Program:
        """Return a new program with accumulated session imports prepended.

        Prior graph-mode entries may have imported modules via open import.
        To make those imports persist across entries, we prepend the stored
        ``ImportDecl`` nodes to the current entry's program items.  Nodes
        with already-present module_paths are de-duplicated (if the current
        entry re-imports the same module, the current entry's decl wins).
        """
        from agm.agl.syntax.nodes import Block, ImportDecl, Program

        if not self._ctx._accumulated_imports:
            return program

        # Collect (module_path, wildcard) pairs already imported in the current entry.
        current_import_paths: set[tuple[tuple[str, ...], bool]] = set()
        for item in program.body.items:
            if isinstance(item, ImportDecl):
                current_import_paths.add((tuple(item.module_path), item.wildcard))

        # Build the injected preamble: accumulated imports NOT already in the entry.
        preamble = [
            decl
            for decl in self._ctx._accumulated_imports
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

    def _evaluate_ir_graph_mode(
        self,
        *,
        text: str,
        orig_program: Program,
        checked: CheckedProgram,
        entry_cm: CheckedModule,
        cgraph: CheckedModuleGraph,
        compiled_graph: MatchCompiledModuleGraph,
        host_env: HostEnvironment,
        warnings: list[Diagnostic],
        new_next_id: int,
        new_modules: dict[ModuleId, LoadedModule],
        param_values: dict[str, Value],
        entry_program_name: str | None,
        entry_active_config: dict[str, object],
        contract_payloads: Mapping[int, "ContractPayload"],
    ) -> EntryResult:
        """Lower and execute one graph-mode entry in the persistent IR image."""
        from agm.agl.eval.ir_interpreter import IrInterpreter
        from agm.agl.lower import lower_repl_graph
        from agm.agl.pipeline import _wire_extern_registry, exception_value_to_run_error
        from agm.agl.runtime.params import _materialize_ir_contracts
        from agm.agl.runtime.request import AgentCancelled
        from agm.agl.runtime.trace import TraceStore
        from agm.agl.semantics.exceptions import AglRaise
        from agm.agl.syntax.nodes import ImportDecl

        # Companion paths for every module the checked graph can reach: prior
        # entries' cached library modules plus this entry's newly linked ones.
        # ``_wire_extern_registry`` imports/resolves only what is not already
        # cached on ``host_env.extern_registry`` (mutated in place), so a
        # companion imports exactly once per session even across entries.
        companion_paths: dict[ModuleId, Path | None] = {
            mid: lm.companion_path for mid, lm in self._ctx._loaded_lib_modules.items()
        }
        companion_paths.update({mid: lm.companion_path for mid, lm in new_modules.items()})
        extern_diagnostics = _wire_extern_registry(
            checked_graph=cgraph,
            capabilities=host_env.capabilities,
            registry=host_env.extern_registry,
            companion_paths=companion_paths,
        )
        if extern_diagnostics:
            return self._ctx._fail(extern_diagnostics, warnings)

        nominal_snapshot = self._ctx._link_image.snapshot_nominals()
        lowered = lower_repl_graph(
            compiled_graph,
            image=self._ctx._link_image,
            source_text=text,
            validate=True,
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
        config_base = self._ctx._build_config_base(entry_active_config)
        interp = IrInterpreter(
            lowered.program,
            registry=host_env.registry,
            strict_json=self._ctx._default_strict_json,
            max_call_depth=self._ctx._default_call_depth_limit,
            shell_exec_timeout=self._ctx._shell_exec_timeout,
            trace=trace,
            param_values=ir_params,
            host_contracts=host_contracts,
            base_frame=self._ctx._ir_base_frame,
            config_cli={},
            config_base=config_base,
            extern_registry=host_env.extern_registry,
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
            installed = self._ctx._promote_ir_state(
                text=text,
                program=orig_program,
                checked=checked,
                next_start_id=new_next_id,
                entry_program_name=entry_program_name,
                entry_active_config=entry_active_config,
                partial=True,
                failure_span=exc.span,
            )
            self._restore_unpromoted_entry_nominals(orig_program, exc.span, nominal_snapshot)
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
            cancel_span = exc.span if isinstance(exc, AgentCancelled) else None
            trace.run_end(ok=False)
            installed = self._ctx._promote_ir_state(
                text=text,
                program=orig_program,
                checked=checked,
                next_start_id=new_next_id,
                entry_program_name=entry_program_name,
                entry_active_config=entry_active_config,
                partial=True,
                failure_span=cancel_span,
            )
            self._restore_unpromoted_entry_nominals(orig_program, cancel_span, nominal_snapshot)
            kind, name = self._ctx._classify(orig_program)
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
                trace_path=self._ctx._trace_path,
                installed=installed,
            )
        trace.run_end(ok=True)
        # Promote engine settings BEFORE static state, mirroring session.py.
        self._ctx._update_engine_settings(
            strict_json=interp.strict_json,
            loop_limit=interp.loop_limit,
            shell_exec_timeout=interp.shell_exec_timeout,
        )
        self._ctx._promote_ir_state(
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
        self._ctx._loaded_lib_modules.update(new_modules)
        self._ctx._link_image.mark_linked(
            mid for mid in cgraph.modules if not mid.is_entry
        )
        import_indexes = {
            (tuple(item.module_path), item.wildcard): index
            for index, item in enumerate(self._ctx._accumulated_imports)
        }
        for item in entry_imports:
            key = (tuple(item.module_path), item.wildcard)
            index = import_indexes.get(key)
            if index is None:
                import_indexes[key] = len(self._ctx._accumulated_imports)
                self._ctx._accumulated_imports.append(item)
            else:
                self._ctx._accumulated_imports[index] = item
        marker = lowered.trailing_expression
        captured = (
            interp.module_initializer_values[lowered.program.entry_module][marker]
            if marker is not None
            else None
        )
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

    def _restore_unpromoted_entry_nominals(
        self,
        program: Program,
        failure_span: SourceSpan | Location | None,
        nominal_snapshot: Mapping["NominalId", "NominalDescriptor"],
    ) -> None:
        """Rollback entry nominal descriptors for type declarations after a failure."""
        from agm.agl.ir.ids import NominalId
        from agm.agl.modules.ids import ENTRY_ID
        from agm.agl.syntax.nodes import EnumDef, ExceptionDef, RecordDef

        nominal_ids = tuple(
            NominalId(ENTRY_ID, item.name)
            for item in program.body.items
            if isinstance(item, (RecordDef, EnumDef, ExceptionDef))
            and not (
                failure_span is not None and item.span.end_offset <= failure_span.start_offset
            )
        )
        self._ctx._link_image.restore_nominals(nominal_snapshot, nominal_ids)
