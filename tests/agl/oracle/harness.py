"""Differential oracle harness for M2/M3/M6a.

Runs both the legacy AST interpreter and the new IR pipeline
(lower_program → IrInterpreter) on the same AgL source and asserts
they produce identical results.

Design notes (extensibility for M7)
------------------------------------
- Snapshots are compared by name equality: both evaluators must bind exactly
  the same set of names (source-declared let/var names).  A binding present
  on one side but absent on the other is caught immediately with a diff message
  showing legacy-only / ir-only names.
- Error comparison is supported via ``assert_oracle_raises`` which checks
  that both pipelines raise ``AglRaise`` with structurally equivalent exceptions.
- stdout capture: M6a adds stdout capture (via ``contextlib.redirect_stdout``)
  and includes the captured output in the oracle agreement assertion.  Programs
  that produce no output yield empty strings and still pass.
- param_values: M6a harness helpers accept ``param_values: dict[str, Value]``
  (keyed by param public name) and route them to both evaluators.
"""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path

from agm.agl.capabilities import HostCapabilities
from agm.agl.eval.exceptions import AglRaise
from agm.agl.eval.interpreter import Interpreter, execute_graph
from agm.agl.eval.ir_interpreter import IrInterpreter
from agm.agl.eval.scope import Scope
from agm.agl.eval.values import (
    Closure,
    DictValue,
    EnumValue,
    ExceptionValue,
    IrClosureValue,
    ListValue,
    RecordValue,
    TextValue,
    Value,
)
from agm.agl.ir.ids import SymbolId
from agm.agl.ir.program import ExecutableProgram
from agm.agl.lower import lower_program
from agm.agl.lower.graph import lower_graph
from agm.agl.modules.ids import ModuleId
from agm.agl.modules.loader import load_graph
from agm.agl.modules.roots import RootSet
from agm.agl.parser import parse_program
from agm.agl.runtime.agents import AgentRegistry
from agm.agl.runtime.codec import OutputCodec, TextCodec
from agm.agl.runtime.contract import materialize_contract
from agm.agl.scope import resolve
from agm.agl.scope.graph import resolve_graph
from agm.agl.typecheck import check
from agm.agl.typecheck.graph import check_graph

_CLOSURE_SENTINEL: TextValue = TextValue("<closure>")

# ---------------------------------------------------------------------------
# Shared capabilities helper for the M2 node subset
# ---------------------------------------------------------------------------


def m2_caps() -> HostCapabilities:
    """HostCapabilities covering the M2 node subset (no agents, no shell)."""
    return HostCapabilities(
        agent_names=frozenset(),
        has_default_agent=False,
        supports_shell_exec=False,
        codec_kinds={
            "text": frozenset({"text"}),
            "json": frozenset(
                {"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}
            ),
        },
    )


def _m2_codecs() -> dict[str, OutputCodec]:
    """Codec map for the M2 node subset."""
    return {"text": TextCodec()}


# ---------------------------------------------------------------------------
# Pipeline helpers — legacy path
# ---------------------------------------------------------------------------


def _run_legacy(
    source: str,
    param_values: dict[str, Value] | None = None,
) -> tuple[dict[str, Value], str]:
    """Run *source* through the legacy AST interpreter.

    Returns ``(root-scope snapshot, captured stdout)``.
    Raises ``AglRaise`` for AgL-level errors.
    """
    program = parse_program(source)
    resolved = resolve(program)
    caps = m2_caps()
    checked = check(resolved, caps)
    codecs = _m2_codecs()
    contracts = {
        node_id: materialize_contract(spec, codecs)
        for node_id, spec in checked.contract_specs.items()
    }
    registry = AgentRegistry(named={}, default_agent=None)
    root_scope = Scope(parent=None)
    interp = Interpreter(
        checked=checked,
        registry=registry,
        contracts=contracts,
        type_env=checked.type_env,
        loop_limit=100,
        strict_json=False,
        source=source,
        param_values=param_values,
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        interp.execute(root_scope)
    return root_scope.snapshot(), buf.getvalue()


# ---------------------------------------------------------------------------
# Pipeline helpers — IR path
# ---------------------------------------------------------------------------


def _build_ir_param_values(
    executable: ExecutableProgram,
    param_values: dict[str, Value],
) -> dict[SymbolId, Value]:
    """Map public-name-keyed param values to SymbolId-keyed for the IR evaluator."""
    name_to_sym: dict[str, SymbolId] = {
        p.public_name: p.symbol for p in executable.params
    }
    result: dict[SymbolId, Value] = {}
    for name, val in param_values.items():
        sym = name_to_sym.get(name)
        assert sym is not None, (
            f"param value for {name!r} does not match any declared param"
            f" (declared: {sorted(name_to_sym)})"
        )
        result[sym] = val
    return result


def _run_ir(
    source: str,
    param_values: dict[str, Value] | None = None,
) -> tuple[dict[str, Value], str]:
    """Run *source* through the new IR pipeline.

    Returns ``({public_name: Value}, captured stdout)``.
    Raises ``AglRaise`` for AgL-level errors.
    """
    program = parse_program(source)
    resolved = resolve(program)
    caps = m2_caps()
    checked = check(resolved, caps)
    executable = lower_program(
        checked,
        source_text=source,
        source_label="<oracle>",
        validate=True,
    )
    ir_param_values: dict[SymbolId, Value] | None = None
    if param_values:
        ir_param_values = _build_ir_param_values(executable, param_values)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = IrInterpreter(executable, param_values=ir_param_values).run()
    return result, buf.getvalue()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assert_oracle_agrees(
    source: str,
    param_values: dict[str, Value] | None = None,
) -> tuple[dict[str, Value], dict[str, Value]]:
    """Assert that the legacy and IR evaluators agree on *source*.

    Runs both pipelines, first asserts that both snapshots have exactly the
    same set of binding names (legacy-only / ir-only differences are reported),
    then compares values for every name.  Also asserts that both evaluators
    produce identical stdout output (M6a: print parity).

    Parameters
    ----------
    source:
        The AgL source program to evaluate.
    param_values:
        Optional ``{public_name: Value}`` dict for entry-module ``param``
        declarations.  Passed to both evaluators (the legacy interpreter
        receives it by name; the IR evaluator receives it keyed by SymbolId).

    Returns
    -------
    ``(legacy_snapshot, ir_snapshot)`` so callers can make additional
    structural assertions (e.g. checking IR node shape).

    Raises
    ------
    ``AssertionError``
        When the name sets differ, when the two snapshots disagree on any
        compared binding value, or when the captured stdout differs.
    """
    legacy, legacy_stdout = _run_legacy(source, param_values)
    ir, ir_stdout = _run_ir(source, param_values)

    legacy_names = set(legacy.keys())
    ir_names = set(ir.keys())

    if legacy_names != ir_names:
        legacy_only = sorted(legacy_names - ir_names)
        ir_only = sorted(ir_names - legacy_names)
        parts: list[str] = []
        if legacy_only:
            parts.append(f"  legacy-only: {legacy_only}")
        if ir_only:
            parts.append(f"  ir-only:     {ir_only}")
        raise AssertionError("Oracle name-set mismatch:\n" + "\n".join(parts))

    mismatches: list[str] = []
    for name in sorted(legacy_names):
        legacy_val = _normalize_value(legacy[name])
        ir_val = _normalize_value(ir[name])
        if legacy_val != ir_val:
            mismatches.append(f"  {name!r}: legacy={legacy_val!r}, ir={ir_val!r}")

    assert not mismatches, "Oracle value disagreement:\n" + "\n".join(mismatches)

    assert legacy_stdout == ir_stdout, (
        f"Oracle stdout disagreement:\n"
        f"  legacy: {legacy_stdout!r}\n"
        f"  ir:     {ir_stdout!r}"
    )

    # Note: trace event sequences and external call sequences are compared in M7.

    return legacy, ir


def _normalize_value(val: Value) -> Value:
    """Recursively zero every ``trace_id`` inside any ``ExceptionValue`` in *val*.

    Walks container types (``ListValue``, ``DictValue``) and nominal types
    (``RecordValue``, ``EnumValue``, ``ExceptionValue``) to find and strip
    ``trace_id`` fields at any nesting depth.  Non-exception, non-container
    values are returned unchanged.
    """
    if isinstance(val, (Closure, IrClosureValue)):
        return _CLOSURE_SENTINEL
    if isinstance(val, ExceptionValue):
        return _normalize_exception(val)
    if isinstance(val, ListValue):
        normalized = tuple(_normalize_value(item) for item in val.elements)
        if normalized == val.elements:
            return val
        return ListValue(normalized)
    if isinstance(val, DictValue):
        new_entries = {k: _normalize_value(v) for k, v in val.entries.items()}
        if new_entries == val.entries:
            return val
        return DictValue(new_entries)
    if isinstance(val, RecordValue):
        new_fields = {k: _normalize_value(v) for k, v in val.fields.items()}
        if new_fields == val.fields:
            return val
        return RecordValue(nominal=val.nominal, display_name=val.display_name, fields=new_fields)
    if isinstance(val, EnumValue):
        new_fields = {k: _normalize_value(v) for k, v in val.fields.items()}
        if new_fields == val.fields:
            return val
        return EnumValue(
            nominal=val.nominal,
            display_name=val.display_name,
            variant=val.variant,
            fields=new_fields,
        )
    return val


def _normalize_exception(exc: ExceptionValue) -> ExceptionValue:
    """Strip nondeterministic trace values from an ``ExceptionValue`` at all depths.

    Legacy fills EVERY declared field absent from the constructor args with the
    SAME single auto-allocated trace-id value.  Normalisation rule: determine
    the canonical auto-trace value (the value currently under the ``trace_id``
    field), then zero EVERY field whose value equals that trace value — not only
    the field literally named ``trace_id``.  This ensures that any other field
    that received the same auto-injected trace value also compares equal across
    both pipelines.  Non-trace fields and nested containers are recursively
    normalised.
    """
    # Identify the canonical auto-trace value (may be absent for hand-built nodes).
    auto_trace: Value | None = exc.fields.get("trace_id")

    new_fields: dict[str, Value] = {}
    for key, field_val in exc.fields.items():
        # Zero the field if it holds the auto-trace value (covers both the
        # literal "trace_id" field and any other field auto-filled with the
        # same single trace id per construction).
        if auto_trace is not None and field_val == auto_trace:
            new_fields[key] = TextValue("")
        else:
            new_fields[key] = _normalize_value(field_val)
    return ExceptionValue(
        nominal=exc.nominal,
        display_name=exc.display_name,
        fields=new_fields,
    )


def assert_oracle_raises(
    source: str,
    param_values: dict[str, Value] | None = None,
) -> tuple[ExceptionValue, ExceptionValue]:
    """Assert that both pipelines raise AglRaise with equivalent exceptions.

    Runs both pipelines, catches AglRaise from both sides, normalizes
    trace_id fields, and asserts the exceptions are equivalent.

    Returns ``(legacy_exc, ir_exc)`` for additional structural assertions.

    Raises
    ------
    ``AssertionError``
        When either pipeline does not raise, or when the normalized exceptions differ.
    """
    legacy_exc: ExceptionValue | None = None
    ir_exc: ExceptionValue | None = None

    try:
        _run_legacy(source, param_values)
    except AglRaise as e:
        legacy_exc = e.exc

    try:
        _run_ir(source, param_values)
    except AglRaise as e:
        ir_exc = e.exc

    assert legacy_exc is not None, "Legacy pipeline did not raise AglRaise"
    assert ir_exc is not None, "IR pipeline did not raise AglRaise"

    norm_legacy = _normalize_exception(legacy_exc)
    norm_ir = _normalize_exception(ir_exc)

    assert norm_legacy == norm_ir, (
        f"Oracle exception disagreement:\n"
        f"  legacy: {norm_legacy!r}\n"
        f"  ir:     {norm_ir!r}"
    )

    return legacy_exc, ir_exc


def _write_module_file(root: Path, dotted: str, source: str) -> None:
    """Write a module source file at the expected path under *root*."""
    mid = ModuleId.from_dotted(dotted)
    p = root / mid.relpath().replace("/", os.sep)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(source)


def _roots_set(*paths: Path) -> RootSet:
    return RootSet(roots=frozenset(paths))


def assert_graph_oracle_agrees(
    entry_source: str,
    modules: dict[str, str],
    tmp_path: Path,
) -> tuple[dict[str, Value], dict[str, Value]]:
    """Assert that legacy and IR evaluators agree on a multi-module program.

    *modules* is a ``{dotted_name: source}`` mapping for library modules.
    The entry source is passed directly (no file needed).

    Returns ``(legacy_snapshot, ir_snapshot)``.
    """
    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)
    for dotted, source in modules.items():
        _write_module_file(root, dotted, source)

    mg = load_graph(entry_source, entry_path=None, roots=_roots_set(root))
    rg = resolve_graph(mg)
    caps = m2_caps()
    cg = check_graph(rg, caps)

    # Legacy path
    codecs = _m2_codecs()
    contracts = {}
    for _mid, cm in cg.modules.items():
        for node_id, spec in cm.contract_specs.items():
            contracts[node_id] = materialize_contract(spec, codecs)
    registry = AgentRegistry(named={}, default_agent=None)
    legacy_buf = io.StringIO()
    with contextlib.redirect_stdout(legacy_buf):
        legacy = execute_graph(cg, registry, contracts, loop_limit=100, strict_json=False)

    # IR path
    executable = lower_graph(cg, validate=True)
    ir_buf = io.StringIO()
    with contextlib.redirect_stdout(ir_buf):
        ir = IrInterpreter(executable).run()

    # Compare
    legacy_names = set(legacy.keys())
    ir_names = set(ir.keys())
    if legacy_names != ir_names:
        legacy_only = sorted(legacy_names - ir_names)
        ir_only = sorted(ir_names - legacy_names)
        parts: list[str] = []
        if legacy_only:
            parts.append(f"  legacy-only: {legacy_only}")
        if ir_only:
            parts.append(f"  ir-only:     {ir_only}")
        raise AssertionError("Oracle name-set mismatch:\n" + "\n".join(parts))

    mismatches: list[str] = []
    for name in sorted(legacy_names):
        legacy_val = _normalize_value(legacy[name])
        ir_val = _normalize_value(ir[name])
        if legacy_val != ir_val:
            mismatches.append(f"  {name!r}: legacy={legacy_val!r}, ir={ir_val!r}")
    assert not mismatches, "Oracle value disagreement:\n" + "\n".join(mismatches)

    assert legacy_buf.getvalue() == ir_buf.getvalue(), (
        f"Oracle graph stdout disagreement:\n"
        f"  legacy: {legacy_buf.getvalue()!r}\n"
        f"  ir:     {ir_buf.getvalue()!r}"
    )

    return legacy, ir


def assert_graph_oracle_raises(
    entry_source: str,
    modules: dict[str, str],
    tmp_path: Path,
) -> tuple[ExceptionValue, ExceptionValue]:
    """Assert that both pipelines raise AglRaise on a multi-module program.

    Returns ``(legacy_exc, ir_exc)`` for additional structural assertions.
    """
    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)
    for dotted, source in modules.items():
        _write_module_file(root, dotted, source)

    mg = load_graph(entry_source, entry_path=None, roots=_roots_set(root))
    rg = resolve_graph(mg)
    caps = m2_caps()
    cg = check_graph(rg, caps)

    codecs = _m2_codecs()
    contracts = {}
    for _mid, cm in cg.modules.items():
        for node_id, spec in cm.contract_specs.items():
            contracts[node_id] = materialize_contract(spec, codecs)
    registry = AgentRegistry(named={}, default_agent=None)

    legacy_exc: ExceptionValue | None = None
    ir_exc: ExceptionValue | None = None

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            execute_graph(cg, registry, contracts, loop_limit=100, strict_json=False)
    except AglRaise as e:
        legacy_exc = e.exc

    try:
        executable = lower_graph(cg, validate=True)
        with contextlib.redirect_stdout(io.StringIO()):
            IrInterpreter(executable).run()
    except AglRaise as e:
        ir_exc = e.exc

    assert legacy_exc is not None, "Legacy pipeline did not raise AglRaise"
    assert ir_exc is not None, "IR pipeline did not raise AglRaise"

    norm_legacy = _normalize_exception(legacy_exc)
    norm_ir = _normalize_exception(ir_exc)
    assert norm_legacy == norm_ir, (
        f"Oracle exception disagreement:\n"
        f"  legacy: {norm_legacy!r}\n"
        f"  ir:     {norm_ir!r}"
    )
    return legacy_exc, ir_exc
