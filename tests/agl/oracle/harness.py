"""Differential oracle harness for M2.

Runs both the legacy AST interpreter and the new IR pipeline
(lower_program → IrInterpreter) on the same AgL source and asserts
they produce identical results.

Design notes (extensibility for M7)
------------------------------------
- Snapshots are compared by name equality: both evaluators must bind exactly
  the same set of names (source-declared let/var names).  A binding present
  on one side but absent on the other is caught immediately with a diff message
  showing legacy-only / ir-only names.
- Error comparison will be added in M3 when in-subset runtime errors exist;
  it will check kind + message + source excerpt.
- Trace event sequences and external call sequences are not yet compared (M7).
  The harness asserts they are empty on both sides for the M2 node subset.
"""

from __future__ import annotations

from agm.agl.capabilities import HostCapabilities
from agm.agl.eval.interpreter import Interpreter
from agm.agl.eval.ir_interpreter import IrInterpreter
from agm.agl.eval.scope import Scope
from agm.agl.eval.values import Value
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
