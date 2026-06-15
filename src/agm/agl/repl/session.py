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
    from agm.agl.scope.symbols import ResolvedProgram, ScopeNode
    from agm.agl.syntax.nodes import Program
    from agm.agl.typecheck.env import CheckedProgram, TypeEnvironment
    from agm.agl.typecheck.types import Type


EntryKind = Literal["expression", "binding", "declaration", "statement"]

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
        Path of the JSONL trace written for this entry (always ``None`` in M1b —
        tracing arrives in a later milestone).
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
    ) -> None:
        from agm.agl.eval.scope import Scope
        from agm.agl.runtime.runtime import WorkflowRuntime
        from agm.agl.scope.symbols import ScopeNode
        from agm.agl.typecheck.env import TypeEnvironment

        self._default_loop_limit = default_loop_limit
        self._default_strict_json = default_strict_json
        self._shell_exec_timeout = shell_exec_timeout

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
        # Source log of successfully-promoted entries (for dump_source / :save).
        self._source_log: list[str] = []

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
        from agm.agl.lexer import lex_tab_warnings
        from agm.agl.parser import AglSyntaxError, parse_program_seeded
        from agm.agl.scope import AglScopeError, resolve
        from agm.agl.typecheck import AglTypeError, check

        host_env = self._runtime.host_environment()

        # TAB advisories are computed from the raw source up front so they surface
        # on EVERY return path (mirroring ``WorkflowRuntime.run``), including a
        # failed parse where there is no other warning channel.
        tab_warnings = lex_tab_warnings(text)

        # [1] Parse (seeded so node ids stay globally unique across entries).
        try:
            program, next_start_id = parse_program_seeded(
                text, start_id=self._next_node_id
            )
        except AglSyntaxError as exc:
            return self._fail([exc.to_diagnostic()], list(tab_warnings))

        # [2] Resolve against the session scope (refs fall through; new decls
        # shadow).  resolve does NOT mutate the parent scope.
        try:
            resolved = resolve(program, parent_scope=self._session_scope)
        except AglScopeError as exc:
            return self._fail([exc.to_diagnostic()], list(tab_warnings))

        # [3] Type-check seeded with the session type env (check COPIES the seed
        # into a fresh env, so self._type_env is not mutated here).
        try:
            checked = check(resolved, host_env.capabilities, seed_env=self._type_env)
        except AglTypeError as exc:
            return self._fail([exc.to_diagnostic()], list(tab_warnings))

        # Surface TAB advisories ahead of the type checker's warnings on every
        # remaining path (typecheck-clean, eval success, or runtime raise).
        warnings: list[Diagnostic] = [*tab_warnings, *checked.warnings]

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
            if ref.kind is not BinderKind.input_binding:
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
        from agm.agl.syntax.nodes import ExprStmt, LetDecl, VarDecl

        kind, name = self._classify(program)
        value_type: Type | None = None
        # A parsed program always has at least one statement (the parser rejects
        # empty/comment-only input as a syntax error before reaching here).
        last = program.body[-1]
        if isinstance(last, ExprStmt):
            value_type = checked.node_types.get(last.expr.node_id)
        elif isinstance(last, (LetDecl, VarDecl)):
            value_type = checked.type_env.get_binding_type(last.node_id)
        return EntryResult(
            kind=kind,
            name=name,
            value=None,
            value_type=value_type,
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
        from agm.agl.repl.echo_interpreter import EchoInterpreter
        from agm.agl.runtime.contract import OutputContract
        from agm.agl.runtime.runtime import exception_value_to_run_error
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
        # ``Scope.set_value``), so a shallow value snapshot of the session frame
        # is a complete, correct rollback point — Value objects are immutable, so
        # storing the reference suffices.  On a runtime raise we restore each
        # binding's ``.value`` from this snapshot.
        value_snapshot: dict[str, Value] = {
            name: binding.value for name, binding in self._value_scope.bindings.items()
        }

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
            trace=None,  # M1b: tracing is a no-op.
        )
        # Echo the value of a trailing bare expression (captured during exec).
        if program.body and isinstance(program.body[-1], ExprStmt):
            interp.echo_node_id = program.body[-1].node_id

        try:
            interp.execute(child_scope)
        except AglRaise as exc:
            error = exception_value_to_run_error(exc.exc, span=exc.span)
            # Atomic-on-error: discard the child scope (new bindings) AND roll back
            # any in-place ``set`` mutations to prior session bindings.  The key
            # set cannot change during eval, so restoring values is sufficient.
            assert self._value_scope.bindings.keys() == value_snapshot.keys()
            for bname, binding in self._value_scope.bindings.items():
                binding.value = value_snapshot[bname]
            kind, name = self._classify(program)
            return EntryResult(
                kind=kind,
                name=name,
                value=None,
                value_type=None,
                diagnostics=[],
                warnings=warnings,
                error=error,
                ok=False,
            )

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
        )

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
        from agm.agl.syntax.nodes import InputDecl

        # Symbols: merge the entry root scope's bindings (overwrite/shadow).
        entry_root = checked.resolved.root_scope
        for bname, ref in entry_root.bindings.items():
            self._session_scope.bindings[bname] = ref

        # Types + binding types: union the entry's checked env into the session.
        self._type_env.seed_from(checked.type_env)

        # Runtime values: copy the child scope's top frame into the session scope.
        for vname, binding in child_scope.bindings.items():
            self._value_scope.bindings[vname] = binding

        # Declared inputs: register any InputDecl (value None until :set).
        for stmt in program.body:
            if isinstance(stmt, InputDecl):
                input_type = checked.type_env.get_binding_type(stmt.node_id)
                assert input_type is not None
                # A re-declared input keeps no stale value (shadows fresh).
                self._declared_inputs[stmt.name] = (input_type, None)

        self._source_log.append(text)
        self._next_node_id = next_start_id

    def _classify(self, program: "Program") -> tuple[EntryKind, str | None]:
        """Classify the entry by its last statement; return (kind, name)."""
        from agm.agl.syntax.nodes import (
            EnumDef,
            ExprStmt,
            InputDecl,
            LetDecl,
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
        if isinstance(last, (RecordDef, EnumDef, TypeAlias, InputDecl)):
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
        if isinstance(last, ExprStmt):
            return captured, checked.node_types.get(last.expr.node_id)
        if isinstance(last, (LetDecl, VarDecl)):
            binding = self._value_scope.lookup(last.name)
            value = binding.value if binding is not None else None
            value_type = checked.type_env.get_binding_type(last.node_id)
            return value, value_type
        return None, None

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
        resolved = resolve(program, parent_scope=self._session_scope)
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

        Registered named agents plus ``"prompt"`` when a default agent is
        configured.
        """
        host_env = self._runtime.host_environment()
        names = sorted(host_env.registry.agent_names)
        if self._has_default_agent:
            names.append("prompt")
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
        from agm.agl.runtime.runtime import convert_input

        entry = self._declared_inputs.get(name)
        if entry is None:
            raise AglError(
                f"{name!r} is not a declared input; declare it with "
                f"'input {name}: <type>' first."
            )
        declared_type, _ = entry
        try:
            value = convert_input(name, raw, declared_type)
        except ValueError as exc:
            raise AglError(str(exc)) from exc

        ref = self._session_scope.bindings.get(name)
        assert ref is not None  # a declared input is always a promoted binding
        self._value_scope.bindings.pop(name, None)
        self._value_scope.define(
            name, value, mutable=False, decl_span=ref.decl_span
        )
        self._declared_inputs[name] = (declared_type, value)

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
        self._source_log = []

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
