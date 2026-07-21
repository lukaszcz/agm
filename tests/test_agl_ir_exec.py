"""IR evaluation tests for the exec() builtin (shell execution).

Each test evaluates an AgL program through the IR pipeline with scripted shell results
and asserts the produced values, stdout, and raised exceptions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from agm.agl.eval.ir_interpreter import IrInterpreter
    from agm.agl.ir.ids import ContractId
    from agm.agl.ir.nodes import IrExec

from agm.core.process import ProcessCaptureResult
from tests.agl.ir_harness import (
    _compiled_checked,
    evaluate_ir_raises_with_shell,
    evaluate_ir_with_shell,
    shell_caps,
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


def _unit_exec_effect() -> tuple[IrInterpreter, IrExec, ContractId]:
    """Build the evaluator seam for an output-discarding exec contract."""
    from agm.agl.eval.ir_interpreter import IrInterpreter
    from agm.agl.ir.contracts import ContractRequest
    from agm.agl.ir.ids import ContractId, Location, SourceId
    from agm.agl.ir.nodes import IrConstText, IrExec
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.modules.ids import ENTRY_ID

    source_id = SourceId(0)
    location = Location(source_id, 0, 1, 1, 0)
    contract_id = ContractId(0)
    node = IrExec(location, IrConstText(location, "cmd"), contract_id, max_attempts=1)
    program = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=())},
        symbols={},
        nominals={},
        sources={source_id: SourceFile(display_name="<test>", normalized_text="x")},
        functions={},
        contracts={
            contract_id: ContractRequest(
                codec_name="none",
                strict_json=None,
                json_schema=None,
                decode=None,
                target_type_label="unit",
                structured_exec=False,
                format_instructions="",
                is_unit=True,
            )
        },
    )
    interpreter = IrInterpreter(program)
    return interpreter, node, contract_id


# ---------------------------------------------------------------------------
# Simple text exec
# ---------------------------------------------------------------------------


def test_t1_text_exec() -> None:
    """exec() with text output: strips trailing newline."""
    source = 'let result: text = exec("echo hello")\nresult'
    commands = {"echo hello": _ok("hello\n")}
    ir = evaluate_ir_with_shell(source, commands)
    from agm.agl.semantics.values import TextValue

    assert ir["result"] == TextValue("hello")


# ---------------------------------------------------------------------------
# Typed/JSON exec (decode to int)
# ---------------------------------------------------------------------------


def test_t2_typed_exec_json() -> None:
    """exec() with int annotation: output is parsed as JSON int."""
    source = 'let n: int = exec("echo 42")\nn'
    commands = {"echo 42": _ok("42\n")}
    ir = evaluate_ir_with_shell(source, commands)
    from agm.agl.semantics.values import IntValue

    assert ir["n"] == IntValue(42)


# ---------------------------------------------------------------------------
# Structured exec (ExecResult record — non-zero exit doesn't raise)
# ---------------------------------------------------------------------------


def test_t3_structured_exec() -> None:
    """exec() returning ExecResult: non-zero exit is data, not an error."""
    source = 'let r: ExecResult = exec("exit 1")\nr'
    commands = {"exit 1": _fail(1, stdout="", stderr="error msg")}
    ir = evaluate_ir_with_shell(source, commands)
    from agm.agl.semantics.values import IntValue, RecordValue

    assert isinstance(ir["r"], RecordValue)
    assert ir["r"].display_name == "ExecResult"
    assert ir["r"].fields["exit_code"] == IntValue(1)


# ---------------------------------------------------------------------------
# Non-zero exit with text contract raises ExecError
# ---------------------------------------------------------------------------


def test_t4_nonzero_exit_text() -> None:
    """exec() with text output and non-zero exit raises ExecError."""
    source = 'let result: text = exec("false")\nresult'
    commands = {"false": _fail(1)}
    ir_exc = evaluate_ir_raises_with_shell(source, commands)
    assert ir_exc.display_name == "ExecError"


def test_t4a_unit_exec_discards_successful_output() -> None:
    """A unit exec succeeds without parsing or retaining stdout."""
    import unittest.mock

    from agm.agl.eval import VOID_VALUE

    interpreter, node, contract_id = _unit_exec_effect()
    with unittest.mock.patch("agm.core.process.run_capture_result", return_value=_ok("ignored")):
        result = interpreter._effects.eval_ir_exec(node, node.command, contract_id, 1)
    assert result is VOID_VALUE


def test_t4a_full_pipeline_unit_exec_discards_successful_output() -> None:
    """A checked unit exec reaches the evaluator as an outputless contract."""
    commands = {"emit": _ok("ignored output\n")}

    result = evaluate_ir_with_shell('exec("emit")\n()', commands)

    assert result == {}


def test_t4b_full_pipeline_unit_exec_still_raises_on_nonzero_exit() -> None:
    """A checked unit exec still maps a shell failure to ExecError."""
    ir_exc = evaluate_ir_raises_with_shell('exec("fail")\n()', {"fail": _fail(2)})

    assert ir_exc.display_name == "ExecError"


def test_t4c_unit_exec_still_raises_on_nonzero_exit() -> None:
    """Discarding successful output does not suppress ExecError."""
    import unittest.mock

    from agm.agl.semantics.exceptions import AglRaise

    interpreter, node, contract_id = _unit_exec_effect()
    with unittest.mock.patch("agm.core.process.run_capture_result", return_value=_fail(2)):
        with pytest.raises(AglRaise) as exc_info:
            interpreter._effects.eval_ir_exec(node, node.command, contract_id, 1)
    assert exc_info.value.exc.display_name == "ExecError"


# ---------------------------------------------------------------------------
# Timeout raises ExecError with timed_out=True
# ---------------------------------------------------------------------------


def test_t5_timeout() -> None:
    """Parsed exec that times out raises ExecError with timed_out=True."""
    source = 'let result: text = exec("sleep 999")\nresult'
    commands = {"sleep 999": _timed_out()}
    ir_exc = evaluate_ir_raises_with_shell(source, commands)
    from agm.agl.semantics.values import BoolValue

    assert ir_exc.display_name == "ExecError"
    assert ir_exc.fields["timed_out"] == BoolValue(True)


def test_t5a_structured_exec_timeout_raises_exec_error() -> None:
    """Structured exec raises ExecError, rather than returning a timed-out record."""
    source = 'let result: ExecResult = exec("sleep 999")\nresult'
    ir_exc = evaluate_ir_raises_with_shell(source, {"sleep 999": _timed_out()})
    from agm.agl.semantics.values import BoolValue

    assert ir_exc.display_name == "ExecError"
    assert ir_exc.fields["timed_out"] == BoolValue(True)


# ---------------------------------------------------------------------------
# Spawn error raises ExecError with timed_out=False
# ---------------------------------------------------------------------------


def test_t6_spawn_error() -> None:
    """exec() that fails to spawn raises ExecError."""
    source = 'let result: text = exec("nonexistent_cmd")\nresult'
    commands = {"nonexistent_cmd": _spawn_failed("No such file or directory")}
    ir_exc = evaluate_ir_raises_with_shell(source, commands)
    from agm.agl.semantics.values import BoolValue

    assert ir_exc.display_name == "ExecError"
    assert ir_exc.fields["timed_out"] == BoolValue(False)


# ---------------------------------------------------------------------------
# Retry policy — fail first attempt, succeed on retry
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

    source = 'let n: int = exec("cmd", on_parse_error = Retry(n = 1))\nn'
    from tests.agl.ir_harness import _run_ir_exec

    caps = shell_caps()

    call_count[0] = 0
    ir_snap, _ = _run_ir_exec(source, fake_shell, caps)

    from agm.agl.semantics.values import IntValue

    assert ir_snap["n"] == IntValue(99)


# ---------------------------------------------------------------------------
# Retry exhaustion — all retries fail → AgentParseError
# ---------------------------------------------------------------------------


def test_t8_retry_exhaustion() -> None:
    """exec() with Retry(n:2): all 3 attempts return bad JSON → AgentParseError.

    Routes through evaluate_ir_raises_with_shell; asserts the IR pipeline raises
    AgentParseError.
    """
    source = 'let n: int = exec("cmd", on_parse_error = Retry(n = 2))\nn'
    commands = {"cmd": _ok("not_a_number\n")}
    ir_exc = evaluate_ir_raises_with_shell(source, commands)
    assert ir_exc.display_name == "AgentParseError"


# ---------------------------------------------------------------------------
# exec inside a user function
# ---------------------------------------------------------------------------


def test_t9_exec_inside_function() -> None:
    """exec() inside a function body lowers and evaluates correctly."""
    source = 'def get_output() -> text = exec("echo from_fn")\nlet result: text = get_output()\n()'
    commands = {"echo from_fn": _ok("from_fn\n")}
    ir = evaluate_ir_with_shell(source, commands)
    from agm.agl.semantics.values import TextValue

    assert ir["result"] == TextValue("from_fn")


# ---------------------------------------------------------------------------
# Golden lowering — IrExec node + dry_run_inventory
# ---------------------------------------------------------------------------


def test_t10_golden_lowering() -> None:
    """Lowering exec() produces an IrExec node and populates dry_run_inventory."""
    from agm.agl.ir.nodes import IrBind, IrExec
    from agm.agl.lower import lower_module
    from agm.agl.parser import parse_program
    from agm.agl.scope import resolve_module
    from agm.agl.typecheck import check_module

    source = 'let result = exec("echo hi")\nresult'
    caps = shell_caps()
    program = parse_program(source)
    resolved = resolve_module(program)
    checked = check_module(resolved, caps)
    executable = lower_module(
        _compiled_checked(checked),
        source_text=source,
        source_label="<test>",
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
# Defensive fallback — parse_agent_output returns empty failure
# ---------------------------------------------------------------------------


def test_t11_exec_empty_parse_failure_raises_agent_parse_error() -> None:
    """IrInterpreter defensive fallback: if parse_agent_output returns ok=False
    with neither errors nor error_msg, AgentParseError still raises."""
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

    with unittest.mock.patch("agm.core.process.run_capture_result", return_value=ok_result):
        with unittest.mock.patch(
            "agm.agl.eval.ir_interpreter._parse_contract_output", return_value=empty_failure
        ):
            with pytest.raises(AglRaise) as exc_info:
                IrInterpreter(prog).run()
    assert exc_info.value.exc.display_name == "AgentParseError"


# ---------------------------------------------------------------------------
# Retry-then-error — first attempt parse-fails, retry exits non-zero
# ---------------------------------------------------------------------------


def test_t12_retry_then_nonzero_exit() -> None:
    """exec() with Retry(n:1): first attempt returns bad JSON, retry exits non-zero.

    Verifies that the retry-error path through _run_exec_shell raises ExecError.
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

    source = 'let n: int = exec("cmd", on_parse_error = Retry(n = 1))\nn'
    from agm.agl.semantics.exceptions import AglRaise
    from agm.agl.semantics.values import ExceptionValue
    from tests.agl.ir_harness import _run_ir_exec

    caps = shell_caps()

    call_count[0] = 0
    with pytest.raises(AglRaise) as exc_info:
        _run_ir_exec(source, fake_shell, caps)
    assert isinstance(exc_info.value.exc, ExceptionValue)
    assert exc_info.value.exc.display_name == "ExecError"
