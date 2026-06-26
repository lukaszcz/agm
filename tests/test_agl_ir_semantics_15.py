"""IR semantic tests for M6c: exec() builtin (shell execution).

Differential ir_semantic: each test asserts that the ir_reference AST interpreter and
the new IR pipeline produce identical values and stdout for the same AgL
program when given the same scripted shell results.
"""

from __future__ import annotations

import pytest

from agm.core.process import ProcessCaptureResult
from tests.agl.ir_harness import (
    evaluate_ir_raises_with_shell,
    evaluate_ir_with_shell,
    m6c_caps,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok(stdout: str, *, returncode: int = 0, stderr: str = "") -> ProcessCaptureResult:
    """Successful ProcessCaptureResult."""
    return ProcessCaptureResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        elapsed=0.01,
        timed_out=False,
        spawn_error=None,
        spawn_errno=None,
    )


def _timed_out(
    *,
    returncode: int = -1,
    stdout: str = "",
    stderr: str = "",
) -> ProcessCaptureResult:
    """Timed-out ProcessCaptureResult."""
    return ProcessCaptureResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        elapsed=0.5,
        timed_out=True,
        spawn_error=None,
        spawn_errno=None,
    )


def _spawn_failed(msg: str = "No such file or directory") -> ProcessCaptureResult:
    """Spawn-failed ProcessCaptureResult."""
    return ProcessCaptureResult(
        returncode=None,
        stdout="",
        stderr="",
        elapsed=0.0,
        timed_out=False,
        spawn_error=msg,
        spawn_errno=2,
    )


def _fail(returncode: int, stdout: str = "", stderr: str = "") -> ProcessCaptureResult:
    """Failed (non-zero exit) ProcessCaptureResult."""
    return ProcessCaptureResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        elapsed=0.01,
        timed_out=False,
        spawn_error=None,
        spawn_errno=None,
    )


# ---------------------------------------------------------------------------
# T1: Simple text exec
# ---------------------------------------------------------------------------


def test_t1_text_exec() -> None:
    """exec() with text output: strips trailing newline."""
    source = 'let result: text = exec("echo hello")\nresult'
    commands = {"echo hello": _ok("hello\n")}
    ir_reference, ir = evaluate_ir_with_shell(source, commands)
    from agm.agl.semantics.values import TextValue

    assert ir["result"] == TextValue("hello")


# ---------------------------------------------------------------------------
# T2: Typed/JSON exec (decode to int)
# ---------------------------------------------------------------------------


def test_t2_typed_exec_json() -> None:
    """exec() with int annotation: output is parsed as JSON int."""
    source = "let n: int = exec(\"echo 42\")\nn"
    commands = {"echo 42": _ok("42\n")}
    ir_reference, ir = evaluate_ir_with_shell(source, commands)
    from agm.agl.semantics.values import IntValue

    assert ir["n"] == IntValue(42)


# ---------------------------------------------------------------------------
# T3: Structured exec (ExecResult record — non-zero exit doesn't raise)
# ---------------------------------------------------------------------------


def test_t3_structured_exec() -> None:
    """exec() returning ExecResult: non-zero exit is data, not an error."""
    source = "let r: ExecResult = exec(\"exit 1\")\nr"
    commands = {"exit 1": _fail(1, stdout="", stderr="error msg")}
    ir_reference, ir = evaluate_ir_with_shell(source, commands)
    from agm.agl.semantics.values import IntValue, RecordValue

    assert isinstance(ir["r"], RecordValue)
    assert ir["r"].display_name == "ExecResult"
    assert ir["r"].fields["exit_code"] == IntValue(1)


# ---------------------------------------------------------------------------
# T4: Non-zero exit with text contract raises ExecError
# ---------------------------------------------------------------------------


def test_t4_nonzero_exit_text() -> None:
    """exec() with text output and non-zero exit raises ExecError."""
    source = 'let result: text = exec("false")\nresult'
    commands = {"false": _fail(1)}
    ir_reference_exc, ir_exc = evaluate_ir_raises_with_shell(source, commands)
    assert ir_exc.display_name == "ExecError"


# ---------------------------------------------------------------------------
# T5: Timeout raises ExecError with timed_out=True
# ---------------------------------------------------------------------------


def test_t5_timeout() -> None:
    """exec() that times out raises ExecError with timed_out=True."""
    source = 'let result: text = exec("sleep 999")\nresult'
    commands = {"sleep 999": _timed_out()}
    ir_reference_exc, ir_exc = evaluate_ir_raises_with_shell(source, commands)
    from agm.agl.semantics.values import BoolValue

    assert ir_exc.display_name == "ExecError"
    assert ir_exc.fields["timed_out"] == BoolValue(True)


# ---------------------------------------------------------------------------
# T6: Spawn error raises ExecError with timed_out=False
# ---------------------------------------------------------------------------


def test_t6_spawn_error() -> None:
    """exec() that fails to spawn raises ExecError."""
    source = 'let result: text = exec("nonexistent_cmd")\nresult'
    commands = {"nonexistent_cmd": _spawn_failed("No such file or directory")}
    ir_reference_exc, ir_exc = evaluate_ir_raises_with_shell(source, commands)
    from agm.agl.semantics.values import BoolValue

    assert ir_exc.display_name == "ExecError"
    assert ir_exc.fields["timed_out"] == BoolValue(False)


# ---------------------------------------------------------------------------
# T7: Retry policy — fail first attempt, succeed on retry
# ---------------------------------------------------------------------------


def test_t7_retry_success() -> None:
    """exec() with Retry(n:1): first invocation returns bad JSON, retry succeeds."""
    call_count = [0]

    def fake_shell(
        args: list[str],
        *,
        idle_timeout: float | None = None,
        isolate_process_group: bool = False,
    ) -> ProcessCaptureResult:
        call_count[0] += 1
        if call_count[0] == 1:
            return _ok("not_a_number\n")
        return _ok("99\n")

    source = "let n: int = exec(\"cmd\", on_parse_error: Retry(n: 1))\nn"
    from tests.agl.ir_harness import _run_ir_exec

    caps = m6c_caps()

    call_count[0] = 0
    ir_reference_snap, _ = _run_ir_exec(source, fake_shell, caps)
    call_count[0] = 0
    ir_snap, _ = _run_ir_exec(source, fake_shell, caps)

    from agm.agl.semantics.values import IntValue

    assert ir_reference_snap["n"] == IntValue(99)
    assert ir_snap["n"] == IntValue(99)


# ---------------------------------------------------------------------------
# T8: Retry exhaustion — all retries fail → AgentParseError
# ---------------------------------------------------------------------------


def test_t8_retry_exhaustion() -> None:
    """exec() with Retry(n:2): all 3 attempts return bad JSON → AgentParseError.

    Routes through evaluate_ir_raises_with_shell so the full exception value
    (including message fields) is compared between ir_reference and IR.
    """
    source = "let n: int = exec(\"cmd\", on_parse_error: Retry(n: 2))\nn"
    commands = {"cmd": _ok("not_a_number\n")}
    ir_reference_exc, ir_exc = evaluate_ir_raises_with_shell(source, commands)
    assert ir_exc.display_name == "AgentParseError"


# ---------------------------------------------------------------------------
# T9: exec inside a user function
# ---------------------------------------------------------------------------


def test_t9_exec_inside_function() -> None:
    """exec() inside a function body lowers and evaluates correctly."""
    source = (
        'def get_output() -> text = exec("echo from_fn")\n'
        "let result: text = get_output()\n"
        "()"
    )
    commands = {"echo from_fn": _ok("from_fn\n")}
    ir_reference, ir = evaluate_ir_with_shell(source, commands)
    from agm.agl.semantics.values import TextValue

    assert ir["result"] == TextValue("from_fn")


# ---------------------------------------------------------------------------
# T10: Golden lowering — IrExec node + dry_run_inventory
# ---------------------------------------------------------------------------


def test_t10_golden_lowering() -> None:
    """Lowering exec() produces an IrExec node and populates dry_run_inventory."""
    from agm.agl.ir.nodes import IrBind, IrExec
    from agm.agl.lower import lower_program
    from agm.agl.parser import parse_program
    from agm.agl.scope import resolve
    from agm.agl.typecheck import check

    source = 'let result = exec("echo hi")\nresult'
    caps = m6c_caps()
    program = parse_program(source)
    resolved = resolve(program)
    checked = check(resolved, caps)
    executable = lower_program(
        checked,
        source_text=source,
        source_label="<test>",
        validate=True,
    )

    # Check that the entry module initializers contain an IrExec node
    entry_mod = executable.modules[executable.entry_module]
    exec_nodes = [
        init.value
        for init in entry_mod.initializers
        if isinstance(init, IrBind) and isinstance(init.value, IrExec)
    ]
    assert len(exec_nodes) == 1, f"Expected 1 IrExec node, found {len(exec_nodes)}"

    # Check dry_run_inventory
    assert len(executable.dry_run_inventory) >= 1
    entry = executable.dry_run_inventory[0]
    assert entry.callee == "exec"
    assert entry.codec_name == "text"
    # text codec → no JSON schema
    assert entry.has_schema is False


# ---------------------------------------------------------------------------
# T11: Defensive fallback — parse_agent_output returns empty failure (line 1334)
# ---------------------------------------------------------------------------


def test_t11_exec_empty_parse_failure_raises_agent_parse_error() -> None:
    """IrInterpreter defensive fallback: if parse_agent_output returns ok=False
    with neither errors nor error_msg, AgentParseError still raises (line 1334)."""
    import unittest.mock

    from agm.agl.eval.ir_interpreter import IrInterpreter
    from agm.agl.ir.contracts import ContractRequest
    from agm.agl.ir.ids import ContractId, SourceId
    from agm.agl.ir.nodes import IrConstText, IrExec
    from agm.agl.ir.program import (
        ExecutableModule,
        ExecutableProgram,
        SourceFile,
    )
    from agm.agl.modules.ids import ENTRY_ID
    from agm.agl.semantics.exceptions import AglRaise
    from agm.core.process import ProcessCaptureResult

    source_id = SourceId(0)

    from agm.agl.ir.ids import Location
    loc = Location(
        source_id=source_id,
        start_offset=0,
        end_offset=1,
        start_line=1,
        start_col=0,
    )
    cid = ContractId(value=0)
    contract = ContractRequest(
        codec_name="json",
        strict_json=False,
        json_schema='{"type":"integer"}',
        decode=None,
        target_type_label="int",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    node = IrExec(
        location=loc,
        command=IrConstText(loc, "cmd"),
        contract_id=cid,
        max_attempts=1,
    )
    prog = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=(node,))},
        symbols={},
        nominals={},
        sources={source_id: SourceFile(display_name="<test>", normalized_text="x")},
        functions={},
        contracts={cid: contract},
    )

    ok_result = ProcessCaptureResult(
        returncode=0,
        stdout="invalid\n",
        stderr="",
        elapsed=0.01,
        timed_out=False,
        spawn_error=None,
        spawn_errno=None,
    )
    from agm.agl.runtime.codec import ParseResult
    empty_failure = ParseResult(ok=False, value=None, error_msg="", errors=())

    with unittest.mock.patch(
        "agm.core.process.run_capture_result", return_value=ok_result
    ):
        with unittest.mock.patch(
            "agm.agl.eval.ir_interpreter._parse_contract_output", return_value=empty_failure
        ):
            with pytest.raises(AglRaise) as exc_info:
                IrInterpreter(prog).run()
    assert exc_info.value.exc.display_name == "AgentParseError"


# ---------------------------------------------------------------------------
# T12: Retry-then-error — first attempt parse-fails, retry exits non-zero
# ---------------------------------------------------------------------------


def test_t12_retry_then_nonzero_exit() -> None:
    """exec() with Retry(n:1): first attempt returns bad JSON, retry exits non-zero.

    Verifies the retry-error path through _run_exec_shell in the IR matches
    the ir_reference interpreter via the differential ir_semantic.
    """
    call_count = [0]

    def fake_shell(
        args: list[str],
        *,
        idle_timeout: float | None = None,
        isolate_process_group: bool = False,
    ) -> ProcessCaptureResult:
        call_count[0] += 1
        if call_count[0] == 1:
            return _ok("not_a_number\n")
        return _fail(1, stdout="", stderr="retry failed")

    source = "let n: int = exec(\"cmd\", on_parse_error: Retry(n: 1))\nn"
    from agm.agl.semantics.exceptions import AglRaise
    from tests.agl.ir_harness import _normalize_exception, _run_ir_exec

    caps = m6c_caps()

    ir_reference_exc: object = None
    try:
        call_count[0] = 0
        _run_ir_exec(source, fake_shell, caps)
    except AglRaise as e:
        ir_reference_exc = e.exc

    ir_exc: object = None
    try:
        call_count[0] = 0
        _run_ir_exec(source, fake_shell, caps)
    except AglRaise as e:
        ir_exc = e.exc

    from agm.agl.semantics.values import ExceptionValue

    assert isinstance(ir_reference_exc, ExceptionValue), "IR reference did not raise AglRaise"
    assert isinstance(ir_exc, ExceptionValue), "IR did not raise AglRaise"
    assert ir_reference_exc.display_name == "ExecError"
    assert ir_exc.display_name == "ExecError"
    assert _normalize_exception(ir_reference_exc) == _normalize_exception(ir_exc)
