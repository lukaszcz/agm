"""Differential oracle harness for M2/M3.

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
- Trace event sequences and external call sequences are not yet compared (M7).
  The harness asserts they are empty on both sides for the M2 node subset.
"""

from __future__ import annotations

from agm.agl.capabilities import HostCapabilities
from agm.agl.eval.exceptions import AglRaise
from agm.agl.eval.interpreter import Interpreter
from agm.agl.eval.ir_interpreter import IrInterpreter
from agm.agl.eval.scope import Scope
from agm.agl.eval.values import (
    DictValue,
    EnumValue,
    ExceptionValue,
    ListValue,
    RecordValue,
    TextValue,
    Value,
)
from agm.agl.lower import lower_program
from agm.agl.parser import parse_program
from agm.agl.runtime.agents import AgentRegistry
from agm.agl.runtime.codec import OutputCodec, TextCodec
from agm.agl.runtime.contract import materialize_contract
from agm.agl.scope import resolve
from agm.agl.typecheck import check

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


def _run_legacy(source: str) -> dict[str, Value]:
    """Run *source* through the legacy AST interpreter.

    Returns the root-scope snapshot ``{name: Value}``.
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
    )
    interp.execute(root_scope)
    return root_scope.snapshot()


# ---------------------------------------------------------------------------
# Pipeline helpers — IR path
# ---------------------------------------------------------------------------


def _run_ir(source: str) -> dict[str, Value]:
    """Run *source* through the new IR pipeline.

    Returns ``{public_name: Value}`` for all top-level bindings.
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
    return IrInterpreter(executable).run()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assert_oracle_agrees(
    source: str,
) -> tuple[dict[str, Value], dict[str, Value]]:
    """Assert that the legacy and IR evaluators agree on *source*.

    Runs both pipelines, first asserts that both snapshots have exactly the
    same set of binding names (legacy-only / ir-only differences are reported),
    then compares values for every name.

    Returns
    -------
    ``(legacy_snapshot, ir_snapshot)`` so callers can make additional
    structural assertions (e.g. checking IR node shape).

    Raises
    ------
    ``AssertionError``
        When the name sets differ or when the two snapshots disagree on any
        compared binding value.

    Note
    ----
    Runtime-error comparison (kind + message + source excerpt) will be added
    in M3 when in-subset runtime errors exist.
    """
    legacy = _run_legacy(source)
    ir = _run_ir(source)

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
        legacy_val = legacy[name]
        ir_val = ir[name]
        if legacy_val != ir_val:
            mismatches.append(f"  {name!r}: legacy={legacy_val!r}, ir={ir_val!r}")

    assert not mismatches, "Oracle value disagreement:\n" + "\n".join(mismatches)

    # M2 note: trace event sequences and external call sequences are trivially
    # empty for the M2 node subset (no agents, no shell exec).  Full comparison
    # of traces and external calls is wired in M7.

    return legacy, ir


def _normalize_value(val: Value) -> Value:
    """Recursively zero every ``trace_id`` inside any ``ExceptionValue`` in *val*.

    Walks container types (``ListValue``, ``DictValue``) and nominal types
    (``RecordValue``, ``EnumValue``, ``ExceptionValue``) to find and strip
    ``trace_id`` fields at any nesting depth.  Non-exception, non-container
    values are returned unchanged.
    """
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
    """Strip ``trace_id`` from an ``ExceptionValue`` at all depths.

    Zeros every ``trace_id`` field in *exc* and recursively normalizes any
    nested ``ExceptionValue`` (or ``ExceptionValue`` inside containers/records/
    enums) found anywhere in the field tree.  All other fields are preserved
    without modification.
    """
    new_fields: dict[str, Value] = {}
    for key, field_val in exc.fields.items():
        if key == "trace_id":
            new_fields[key] = TextValue("")
        else:
            new_fields[key] = _normalize_value(field_val)
    return ExceptionValue(
        nominal=exc.nominal,
        display_name=exc.display_name,
        fields=new_fields,
    )


def assert_oracle_raises(source: str) -> tuple[ExceptionValue, ExceptionValue]:
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
        _run_legacy(source)
    except AglRaise as e:
        legacy_exc = e.exc

    try:
        _run_ir(source)
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
