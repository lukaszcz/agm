"""Multi-module REPL program pipeline collaborator.

Implements the build_repl_graph → resolve_program → check_program → match
compilation → incremental link/exec pipeline for REPL entries that contain
import declarations or have cached library modules from prior entries. Driven
by ``ReplSession`` via the narrow ``EntryPipelineCtx`` Protocol. Must NOT import
``session`` (no cycle).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

from agm.agl.diagnostics import Diagnostic
from agm.agl.repl.entry import EntryKind, EntryResult

if TYPE_CHECKING:
    from pathlib import Path

    from agm.agl.eval.ir_interpreter import IrInterpreter
    from agm.agl.ir.contracts import ContractPayload
    from agm.agl.ir.ids import Location, NominalId
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
    from agm.agl.syntax.nodes import ImportDecl, Item, Program
    from agm.agl.syntax.spans import SourceSpan
    from agm.agl.typecheck.env import CheckedModule, TypeEnvironment
    from agm.agl.typecheck.program import CheckedProgram


# ---------------------------------------------------------------------------
# Narrow context Protocol
# ---------------------------------------------------------------------------


class EntryPipelineCtx(Protocol):
    """The minimal ReplSession surface the program pipeline needs."""

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
        failure_span: SourceSpan | Location | None,
    ) -> tuple[str, ...]: ...

    def _classify(self, program: Program) -> tuple[EntryKind, str | None]: ...

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
            entry_program, next_start_id, entry_imports = self._prepare_entry_program(
                pipeline_program, next_start_id, roots
            )
            graph, new_next_id, new_modules = build_repl_graph(
                entry_program,
                next_start_id,
                path=None,
                cached=self._ctx._loaded_lib_modules,
                roots=roots,
                default_stdlib=self._ctx._default_stdlib,
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
            param_values=param_values,
            entry_program_name=entry_program_name,
            entry_active_config=entry_active_config,
            contract_payloads=contract_payloads,
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
            partial_calls=entry.partial_calls,
        )

    def _prepare_entry_program(
        self,
        program: Program,
        next_start_id: int,
        roots: RootSet,
    ) -> tuple[Program, int, tuple[ImportDecl, ...]]:
        """Expand current wildcards, then inject imports retained by module identity.

        REPL replacement is finer grained than batch import merging: each
        wildcard expands to exact target modules before retained declarations
        are compared. A new entry replaces only the modules it names, while
        declarations for one module in that entry still union normally.
        """
        expanded, next_start_id, entry_imports = self._expand_entry_wildcards(
            program, next_start_id, roots
        )
        return (
            self._inject_accumulated_imports(expanded, entry_imports),
            next_start_id,
            entry_imports,
        )

    @staticmethod
    def _expand_entry_wildcards(
        program: Program,
        next_start_id: int,
        roots: RootSet,
    ) -> tuple[Program, int, tuple[ImportDecl, ...]]:
        """Expand wildcard imports into distinct exact-module declarations."""
        from dataclasses import replace

        from agm.agl.modules.resolver import expand_wildcard
        from agm.agl.syntax.nodes import Block, ImportDecl, Program

        items: list[Item] = []
        imports: list[ImportDecl] = []
        expanded_wildcard = False
        for item in program.body.items:
            if not isinstance(item, ImportDecl):
                items.append(item)
                continue
            if not item.wildcard:
                items.append(item)
                imports.append(item)
                continue
            expanded_wildcard = True
            for module in expand_wildcard(tuple(item.module_path), roots, span=item.span):
                expanded = replace(
                    item,
                    module_path=module.segments,
                    wildcard=False,
                    node_id=next_start_id,
                )
                next_start_id += 1
                items.append(expanded)
                imports.append(expanded)

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

    def _inject_accumulated_imports(
        self,
        program: Program,
        entry_imports: tuple[ImportDecl, ...],
    ) -> Program:
        """Prepend retained imports except those replaced in this entry."""
        from agm.agl.syntax.nodes import Block, Program

        if not self._ctx._accumulated_imports:
            return program

        preamble = self._retained_imports_after_replacement(entry_imports)
        if not preamble:
            return program
        return Program(
            body=Block(
                items=(*preamble, *program.body.items),
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
            self._persist_interpreter_settings(interp, trace)
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
                diagnostics=[Diagnostic(message="Agent call cancelled — entry aborted.", line=1)],
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
            failure_span=None,
        )
        self._ctx._loaded_lib_modules.update(new_modules)
        self._ctx._link_image.mark_linked(
            mid for mid in checked_program.modules if not mid.is_entry
        )
        self._replace_accumulated_imports(entry_imports)
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

    def _retained_imports_after_replacement(
        self, entry_imports: tuple[ImportDecl, ...]
    ) -> list[ImportDecl]:
        """Return accumulated declarations not replaced by an entry module identity."""
        from agm.agl.modules.ids import ModuleId

        replacement_modules = {ModuleId(segments=tuple(decl.module_path)) for decl in entry_imports}
        return [
            decl
            for decl in self._ctx._accumulated_imports
            if ModuleId(segments=tuple(decl.module_path)) not in replacement_modules
        ]

    def _replace_accumulated_imports(self, entry_imports: tuple[ImportDecl, ...]) -> None:
        """Replace retained declarations for every module touched by this entry."""
        self._ctx._accumulated_imports[:] = [
            *self._retained_imports_after_replacement(entry_imports),
            *entry_imports,
        ]

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
            and not (failure_span is not None and item.span.end_offset <= failure_span.start_offset)
        )
        self._ctx._link_image.restore_nominals(nominal_snapshot, nominal_ids)
