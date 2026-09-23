"""IR evaluation tests for the exec() builtin (shell execution).

Each test evaluates an AgL program through the IR pipeline with scripted shell results
and asserts the produced values, stdout, and raised exceptions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.semantics.exceptions import AglRaise
from agm.core.process import CapturedOutput, ProcessCaptureResult
from tests._agl_helpers import (
    agl_roots,
    let_root_capture,
    session_sandbox_context,
    unavailable_sandbox_context,
    write_sandbox_home,
)
from tests.agl.ir_harness import (
    ScriptedShellCall,
    completed_bindings,
    evaluate_ir_raises_with_shell,
    evaluate_ir_with_agents,
    evaluate_ir_with_shell,
    inline_main_items,
    lower_inline_ir,
    run_inline_ir_with_shell,
    shell_caps,
    uncaught_error,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok(stdout: str, *, returncode: int = 0, stderr: str = "") -> ProcessCaptureResult:
    """Successful ProcessCaptureResult."""
    return ProcessCaptureResult(
        returncode=returncode,
        stdout=CapturedOutput(data=stdout.encode(), truncated=False),
        stderr=CapturedOutput(data=stderr.encode(), truncated=False),
        elapsed=0.01,
        timed_out=False,
        spawn_error=None,
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
        stdout=CapturedOutput(data=stdout.encode(), truncated=True),
        stderr=CapturedOutput(data=stderr.encode(), truncated=True),
        elapsed=0.5,
        timed_out=True,
        spawn_error=None,
    )


def _spawn_failed(msg: str = "No such file or directory") -> ProcessCaptureResult:
    """Spawn-failed ProcessCaptureResult."""
    return ProcessCaptureResult(
        returncode=None,
        stdout=CapturedOutput(data=b"", truncated=False),
        stderr=CapturedOutput(data=b"", truncated=False),
        elapsed=0.0,
        timed_out=False,
        spawn_error=msg,
    )


def _fail(returncode: int, stdout: str = "", stderr: str = "") -> ProcessCaptureResult:
    """Failed (non-zero exit) ProcessCaptureResult."""
    return ProcessCaptureResult(
        returncode=returncode,
        stdout=CapturedOutput(data=stdout.encode(), truncated=False),
        stderr=CapturedOutput(data=stderr.encode(), truncated=False),
        elapsed=0.01,
        timed_out=False,
        spawn_error=None,
    )


# ---------------------------------------------------------------------------
# Simple text exec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    (
        pytest.param('let result: text = exec("echo hello")\nresult', id="call_form"),
        pytest.param("let result: text = exec $ echo hello\nresult", id="verbatim_literal"),
    ),
)
def test_text_exec_strips_trailing_newline(source: str) -> None:
    """exec() with text output strips a trailing newline, in call and verbatim-literal form."""
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
    assert ir["r"].fields["exit-code"] == IntValue(1)
    program = lower_inline_ir(source, caps=shell_caps())
    assert ir["r"].nominal == program.builtin_nominals.nominal("ExecResult")


# ---------------------------------------------------------------------------
# Non-zero exit with text contract raises ExecError
# ---------------------------------------------------------------------------


def test_t4_nonzero_exit_text() -> None:
    """exec() with text output and non-zero exit raises ExecError."""
    source = 'let result: text = exec("false")\nresult'
    commands = {"false": _fail(1)}
    ir_exc = evaluate_ir_raises_with_shell(source, commands)
    assert ir_exc.type_name == "ExecError"


def test_t4a_full_pipeline_unit_exec_discards_successful_output() -> None:
    """A checked unit exec reaches the evaluator as an outputless contract."""
    commands = {"emit": _ok("ignored output\n")}

    result = evaluate_ir_with_shell('exec("emit")\n()', commands)

    assert result == {}


def test_t4b_full_pipeline_unit_exec_still_raises_on_nonzero_exit() -> None:
    """A checked unit exec still maps a shell failure to ExecError."""
    source = 'exec("fail")\n()'
    ir_exc = evaluate_ir_raises_with_shell(source, {"fail": _fail(2)})
    assert ir_exc.type_name == "ExecError"


# ---------------------------------------------------------------------------
# Timeout raises ExecError with timed_out=True
# ---------------------------------------------------------------------------


def test_t5_timeout() -> None:
    """Parsed exec that times out raises ExecError with timed_out=True."""
    source = 'let result: text = exec("sleep 999")\nresult'
    commands = {"sleep 999": _timed_out()}
    ir_exc = evaluate_ir_raises_with_shell(source, commands)
    assert ir_exc.type_name == "ExecError"
    assert ir_exc.fields["timed-out"] is True


def test_t5a_structured_exec_timeout_raises_exec_error() -> None:
    """Structured exec raises ExecError, rather than returning a timed-out record."""
    source = 'let result: ExecResult = exec("sleep 999")\nresult'
    ir_exc = evaluate_ir_raises_with_shell(source, {"sleep 999": _timed_out()})
    assert ir_exc.type_name == "ExecError"
    assert ir_exc.fields["timed-out"] is True


# ---------------------------------------------------------------------------
# Spawn error raises ExecError with timed_out=False
# ---------------------------------------------------------------------------


def test_t6_spawn_error() -> None:
    """exec() that fails to spawn raises ExecError."""
    source = 'let result: text = exec("nonexistent_cmd")\nresult'
    commands = {"nonexistent_cmd": _spawn_failed("No such file or directory")}
    ir_exc = evaluate_ir_raises_with_shell(source, commands)
    assert ir_exc.type_name == "ExecError"
    assert ir_exc.fields["timed-out"] is False


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
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        isolate_process_group: bool = False,
        interrupt_cleanup_cmd: list[str] | None = None,
    ) -> ProcessCaptureResult:
        del idle_timeout, cwd, env, isolate_process_group, interrupt_cleanup_cmd
        call_count[0] += 1
        if call_count[0] == 1:
            return _ok("not_a_number\n")
        return _ok("99\n")

    source = 'let n: int = exec("cmd", on-parse-error = Retry(n = 1))\nn'
    call_count[0] = 0
    ir_snap = completed_bindings(run_inline_ir_with_shell(source, fake_shell))

    from agm.agl.semantics.values import IntValue

    assert ir_snap["n"] == IntValue(99)


def test_t7_retry_forwards_spawn_settings_on_every_attempt() -> None:
    """Retries reuse the evaluated child environment, working directory, and timeout."""
    calls: list[tuple[dict[str, str] | None, Path | None, float | None]] = []

    def fake_shell(
        args: list[str],
        *,
        idle_timeout: float | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        isolate_process_group: bool = False,
        interrupt_cleanup_cmd: list[str] | None = None,
    ) -> ProcessCaptureResult:
        del args, isolate_process_group, interrupt_cleanup_cmd
        calls.append((env, cwd, idle_timeout))
        return _ok("not-an-int\n" if len(calls) == 1 else "9\n")

    source = (
        "import std/env::Environ\n"
        'let child = Environ(vars = {"ONLY": "child"})\n'
        'let n: int = exec("cmd", env = child, '
        'cwd = Option[text]::Some(value = "/work"), '
        'timeout = Option[text]::Some(value = "2s"), '
        "on-parse-error = Retry(n = 1))\n"
        "n"
    )
    snapshot = completed_bindings(run_inline_ir_with_shell(source, fake_shell))

    from agm.agl.semantics.values import IntValue

    assert snapshot["n"] == IntValue(9)
    assert calls == [
        ({"ONLY": "child"}, Path("/work"), 2.0),
        ({"ONLY": "child"}, Path("/work"), 2.0),
    ]


# ---------------------------------------------------------------------------
# Retry exhaustion — all retries fail → AgentParseError
# ---------------------------------------------------------------------------


def test_t8_retry_exhaustion() -> None:
    """exec() with Retry(n:2): all 3 attempts return bad JSON → ExecError.

    Routes through evaluate_ir_raises_with_shell and keeps shell failures in
    the ExecError family.
    """
    source = 'let n: int = exec("cmd", on-parse-error = Retry(n = 2))\nn'
    commands = {"cmd": _ok("not_a_number\n")}
    ir_exc = evaluate_ir_raises_with_shell(source, commands)
    assert ir_exc.type_name == "ExecError"


@pytest.mark.parametrize("retries,expected_runs", [(0, 1), (1, 2), (2, 3)])
def test_retry_reruns_the_shell_exactly_once_per_attempt(retries: int, expected_runs: int) -> None:
    """Retry(n) spends exactly n + 1 attempts, each one re-running the command."""
    runs: list[str] = []

    def fake_shell(
        args: list[str],
        *,
        idle_timeout: float | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        isolate_process_group: bool = False,
        interrupt_cleanup_cmd: list[str] | None = None,
    ) -> ProcessCaptureResult:
        del idle_timeout, cwd, env, isolate_process_group, interrupt_cleanup_cmd
        runs.append(" ".join(args))
        return _ok("not_a_number\n")

    source = f'let n: int = exec("cmd", on-parse-error = Retry(n = {retries}))\nn'
    exc = uncaught_error(run_inline_ir_with_shell(source, fake_shell))

    assert exc.type_name == "ExecError"
    assert len(runs) == expected_runs


# ---------------------------------------------------------------------------
# exec inside a user function
# ---------------------------------------------------------------------------


def test_t9_exec_inside_function() -> None:
    """exec() inside a function body lowers and evaluates correctly."""
    source = 'def get-output() -> text = exec("echo from_fn")\nlet result: text = get-output()\n()'
    commands = {"echo from_fn": _ok("from_fn\n")}
    ir = evaluate_ir_with_shell(source, commands)
    from agm.agl.semantics.values import TextValue

    assert ir["result"] == TextValue("from_fn")


# ---------------------------------------------------------------------------
# Golden lowering — IrExec node + dry_run_inventory
# ---------------------------------------------------------------------------


def test_t10_golden_lowering() -> None:
    """Lowering exec() produces an IrExec node and populates dry_run_inventory."""
    from agm.agl.ir.nodes import IrBind, IrExec, IrSequence

    source = 'let result = exec("echo hi")\nresult'
    executable = lower_inline_ir(source, caps=shell_caps())

    # Check that the inline command's synthetic main contains an IrExec node.
    exec_nodes = [
        let_root_capture(init).value
        for init in inline_main_items(executable)
        if isinstance(init, (IrSequence, IrBind))
        and isinstance(let_root_capture(init).value, IrExec)
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
    """IrInterpreter defensive fallback produces ExecError for empty parse failures."""
    import unittest.mock

    from agm.agl.eval.ir_interpreter import IrInterpreter
    from agm.agl.ir.contracts import ContractRequest
    from agm.agl.ir.ids import ContractId, NominalId, SourceId
    from agm.agl.ir.nodes import IrConstText, IrExec, IrMakeDict, IrMakeRecord
    from agm.agl.ir.program import (
        ExecutableModule,
        ExecutableProgram,
        SourceFile,
    )
    from agm.agl.ir.reserved_nominals import require_reserved_enum_member_id
    from agm.agl.modules.ids import ENTRY_ID
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
        env=IrMakeRecord(
            loc,
            nominal=NominalId(-1),
            fields=(("vars", IrMakeDict(loc, ())),),
        ),
        cwd=IrMakeRecord(
            loc,
            NominalId(require_reserved_enum_member_id("Option", "None")),
            (),
        ),
        timeout=IrMakeRecord(
            loc,
            NominalId(require_reserved_enum_member_id("Option", "None")),
            (),
        ),
        sandbox=IrMakeRecord(
            loc,
            NominalId(require_reserved_enum_member_id("Option", "None")),
            (),
        ),
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
        stdout=CapturedOutput(data=b"invalid\n", truncated=False),
        stderr=CapturedOutput(data=b"", truncated=False),
        elapsed=0.01,
        timed_out=False,
        spawn_error=None,
    )
    from agm.agl.runtime.codec import ParseResult

    empty_failure = ParseResult(ok=False, value=None, error_msg="", errors=())

    with unittest.mock.patch("agm.core.process.run_capture_result", return_value=ok_result):
        with unittest.mock.patch(
            "agm.agl.eval.ir_interpreter._parse_contract_output", return_value=empty_failure
        ):
            with pytest.raises(AglRaise) as exc_info:
                IrInterpreter(prog).run()
    assert exc_info.value.exc.nominal == prog.builtin_nominals.nominal("ExecError")


# ---------------------------------------------------------------------------
# Retry-then-error — first attempt parse-fails, retry exits non-zero
# ---------------------------------------------------------------------------


def test_exec_with_an_extended_environment_reaches_the_process_boundary() -> None:
    """An explicit extended Environ is passed intact to the shell child."""
    calls: list[dict[str, str] | None] = []

    def fake_shell(
        args: list[str],
        *,
        idle_timeout: float | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        isolate_process_group: bool = False,
        interrupt_cleanup_cmd: list[str] | None = None,
    ) -> ProcessCaptureResult:
        del args, idle_timeout, cwd, isolate_process_group, interrupt_cleanup_cmd
        calls.append(env)
        return _ok("ok\\n")

    source = (
        "import std/env::*\n"
        'let child = environ.extended({"base": "override", "extra": "value"})\n'
        'let output: text = exec("child", env = child)\n'
        "output"
    )
    from unittest.mock import patch

    from agm.agl import PipelineDriver
    from tests._agl_helpers import run_inline_command

    with patch("agm.core.process.run_capture_result", side_effect=fake_shell):
        result = run_inline_command(
            PipelineDriver(get_sandbox_context=None),
            source,
            roots=agl_roots(),
            process_environment={"base": "original", "preserved": "kept"},
        )

    assert result.ok, result.diagnostics
    assert calls == [{"base": "override", "preserved": "kept", "extra": "value"}]


def test_exec_dollar_literal_uses_the_live_std_config_timeout_default() -> None:
    """An omitted timeout on exec $ reads std/config at the call point."""
    calls: list[float | None] = []

    def fake_shell(
        args: list[str],
        *,
        idle_timeout: float | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        isolate_process_group: bool = False,
        interrupt_cleanup_cmd: list[str] | None = None,
    ) -> ProcessCaptureResult:
        del args, cwd, env, isolate_process_group, interrupt_cleanup_cmd
        calls.append(idle_timeout)
        return _ok("ok\\n")

    source = (
        "import std/config\n"
        'std/config::timeout := Option[text]::Some(value = "2s")\n'
        "let output: text = exec $ configured\n"
        "output"
    )
    completed_bindings(run_inline_ir_with_shell(source, fake_shell))

    assert calls == [2.0]


def test_t13_exec_spawn_parameters_and_defaults() -> None:
    """Exec evaluates its env/cwd/timeout operands; omitted operands use ambient bindings."""
    calls: list[tuple[dict[str, str] | None, object | None, float | None]] = []

    def fake_shell(
        args: list[str],
        *,
        idle_timeout: float | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        isolate_process_group: bool = False,
        interrupt_cleanup_cmd: list[str] | None = None,
    ) -> ProcessCaptureResult:
        del args, isolate_process_group, interrupt_cleanup_cmd
        calls.append((env, cwd, idle_timeout))
        return _ok("ok\\n")

    source = (
        "import std/env::Environ\n"
        'let child = Environ(vars = {"ONLY": "child"})\n'
        'let explicit: text = exec("explicit", env = child, '
        'cwd = Option[text]::Some(value = "/work"), '
        'timeout = Option[text]::Some(value = "2s"))\n'
        "let ambient: text = exec $ ambient\n"
        "let configured: text = exec $ configured\n"
        "()"
    )
    completed_bindings(
        run_inline_ir_with_shell(
            source,
            fake_shell,
            process_environment={"AMBIENT": "present"},
            shell_exec_timeout=3.5,
        )
    )

    assert calls == [
        ({"ONLY": "child"}, Path("/work"), 2.0),
        ({"AMBIENT": "present"}, None, 3.5),
        ({"AMBIENT": "present"}, None, 3.5),
    ]


def test_t12_retry_then_nonzero_exit() -> None:
    """exec() with Retry(n:1): first attempt returns bad JSON, retry exits non-zero.

    Verifies that the retry-error path through _run_exec_shell raises ExecError.
    """
    call_count = [0]

    def fake_shell(
        args: list[str],
        *,
        idle_timeout: float | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        isolate_process_group: bool = False,
        interrupt_cleanup_cmd: list[str] | None = None,
    ) -> ProcessCaptureResult:
        del idle_timeout, cwd, env, isolate_process_group, interrupt_cleanup_cmd
        call_count[0] += 1
        if call_count[0] == 1:
            return _ok("not_a_number\n")
        return _fail(1, stdout="", stderr="retry failed")

    source = 'let n: int = exec("cmd", on-parse-error = Retry(n = 1))\nn'
    call_count[0] = 0
    exc = uncaught_error(run_inline_ir_with_shell(source, fake_shell))
    assert exc.type_name == "ExecError"


def test_t14_invalid_exec_timeout_raises_type_error_before_shell_execution() -> None:
    """An explicit invalid timeout is rejected while evaluating exec operands."""
    source = 'let _: text = exec("must-not-run", timeout = Option[text]::Some(value = "bad"))\n()'

    ir_exc = evaluate_ir_raises_with_shell(source, {})
    assert ir_exc.type_name == "TypeError"


# ---------------------------------------------------------------------------
# Strict decoding of invalid UTF-8 process output
# ---------------------------------------------------------------------------


def test_structured_exec_with_undecodable_stdout_raises_exec_error() -> None:
    """A structured exec() with undecodable stdout raises ExecError naming the offset."""
    source = 'let r: ExecResult = exec("cmd")\nr'
    commands = {
        "cmd": ProcessCaptureResult(
            returncode=0,
            stdout=CapturedOutput(data=b"ok \xff bad", truncated=False),
            stderr=CapturedOutput(data=b"", truncated=False),
            elapsed=0.01,
            timed_out=False,
            spawn_error=None,
        )
    }
    ir_exc = evaluate_ir_raises_with_shell(source, commands)
    assert ir_exc.type_name == "ExecError"
    assert "stdout is not valid UTF-8 at byte 3" in ir_exc.fields["message"]


def test_nonzero_exit_with_undecodable_stderr_gives_empty_field_and_names_the_offset() -> None:
    """A non-zero exit with undecodable stderr keeps the field empty but names the offset."""
    source = 'let result: text = exec("cmd")\nresult'
    commands = {
        "cmd": ProcessCaptureResult(
            returncode=1,
            stdout=CapturedOutput(data=b"", truncated=False),
            stderr=CapturedOutput(data=b"err \xff", truncated=False),
            elapsed=0.01,
            timed_out=False,
            spawn_error=None,
        )
    }
    ir_exc = evaluate_ir_raises_with_shell(source, commands)
    assert ir_exc.type_name == "ExecError"
    assert ir_exc.fields["stderr"] == ""
    assert "stderr is not valid UTF-8 at byte 4" in ir_exc.fields["message"]


def test_parsed_form_undecodable_stdout_raises_without_retrying() -> None:
    """A decode failure on parsed-form output raises immediately, never via retry."""
    runs: list[str] = []

    def fake_shell(
        args: list[str],
        *,
        idle_timeout: float | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        isolate_process_group: bool = False,
        interrupt_cleanup_cmd: list[str] | None = None,
    ) -> ProcessCaptureResult:
        del idle_timeout, cwd, env, isolate_process_group, interrupt_cleanup_cmd
        runs.append(args[2])
        return ProcessCaptureResult(
            returncode=0,
            stdout=CapturedOutput(data=b"\xff", truncated=False),
            stderr=CapturedOutput(data=b"", truncated=False),
            elapsed=0.01,
            timed_out=False,
            spawn_error=None,
        )

    source = 'let n: int = exec("cmd", on-parse-error = Retry(n = 2))\nn'
    exc = uncaught_error(run_inline_ir_with_shell(source, fake_shell))
    assert exc.type_name == "ExecError"
    assert len(runs) == 1


def test_timeout_drops_only_an_incomplete_trailing_utf8_sequence() -> None:
    """An idle-timeout kill mid-character drops that incomplete tail, not the rest."""
    source = 'let result: text = exec("sleep 999")\nresult'
    commands = {
        "sleep 999": ProcessCaptureResult(
            returncode=-1,
            stdout=CapturedOutput(data="café".encode()[:-1], truncated=True),
            stderr=CapturedOutput(data=b"", truncated=True),
            elapsed=0.5,
            timed_out=True,
            spawn_error=None,
        )
    }
    ir_exc = evaluate_ir_raises_with_shell(source, commands)
    assert ir_exc.type_name == "ExecError"
    assert ir_exc.fields["stdout"] == "caf"


def test_timeout_with_invalid_bytes_elsewhere_still_raises_a_decode_note() -> None:
    """A truncated stream still raises on invalid bytes that are not just an incomplete tail."""
    source = 'let result: text = exec("sleep 999")\nresult'
    commands = {
        "sleep 999": ProcessCaptureResult(
            returncode=-1,
            stdout=CapturedOutput(data=b"ok \xff", truncated=True),
            stderr=CapturedOutput(data=b"", truncated=True),
            elapsed=0.5,
            timed_out=True,
            spawn_error=None,
        )
    }
    ir_exc = evaluate_ir_raises_with_shell(source, commands)
    assert ir_exc.type_name == "ExecError"
    assert "stdout is not valid UTF-8 at byte 3" in ir_exc.fields["message"]


# A direct exec()/ask() call sitting in an unresolved generic parameter slot
# ---------------------------------------------------------------------------
#
# Such a call's own contextual target is a bare, unresolved solver variable
# until the checker applies the builtin default at region close. These
# confirm the whole pipeline — not just the checker — runs the finalized
# default target end to end.


def test_exec_in_a_generic_argument_slot_defaults_to_exec_result() -> None:
    """``id(exec "…")`` infers ``ExecResult`` and executes it."""
    source = 'def id[A](x: A) -> A = x\nlet result = id(exec "echo hi")\nresult'
    commands = {"echo hi": _ok("hi\n")}
    ir = evaluate_ir_with_shell(source, commands)
    from agm.agl.semantics.values import IntValue, RecordValue

    assert isinstance(ir["result"], RecordValue)
    assert ir["result"].fields["exit-code"] == IntValue(0)


def test_ask_in_a_generic_argument_slot_defaults_to_text() -> None:
    """``id(ask "…")`` infers ``text`` and executes it."""
    source = 'def id[A](x: A) -> A = x\nlet result = id(ask "hi")\nresult'
    ir = evaluate_ir_with_agents(source, {}, default_responses=["hello"])
    from agm.agl.semantics.values import TextValue

    assert ir["result"] == TextValue("hello")


# ---------------------------------------------------------------------------
# Sandboxed exec (``sandbox = Some(Sandbox(...))``)
# ---------------------------------------------------------------------------


def test_sandboxed_exec_wraps_argv_and_selects_profile_from_first_shell_word(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``sandbox = Some(Sandbox())`` wraps the shell child under the real
    systemd-run/srt chain, selecting settings by the command's first shell
    word -- never ``sh``, the wrapper's own executable."""
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    write_sandbox_home(home, extra_settings_files=("echo",))
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: "/usr/bin/tool")

    source = 'let result: text = exec("echo hi", sandbox = Some(Sandbox()))\nresult'
    commands = {"echo hi": _ok("hi\n")}
    argv_log: list[list[str]] = []
    ir = evaluate_ir_with_shell(
        source,
        commands,
        argv_log=argv_log,
        get_sandbox_context=session_sandbox_context(home),
    )
    from agm.agl.semantics.values import TextValue

    assert ir["result"] == TextValue("hi")
    assert len(argv_log) == 1
    argv = argv_log[0]
    assert argv[:4] == ["systemd-run", "--user", "--scope", "-q"]
    srt_index = argv.index("srt")
    assert argv[srt_index + 1] == "--settings"
    # The profile file chosen is the command's first shell word ("echo"),
    # never "sh" -- the shell wrapper exec always runs the command under.
    assert argv[srt_index + 2] == str(home / ".agm" / "sandbox" / "echo.json")
    assert argv[srt_index + 3] == "--"


def test_sandboxed_exec_honours_run_config_memory_for_the_selected_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``[run.<profile>].memory`` resolves the profile's own limit, proving the
    resolved profile's *limits* actually apply -- not just that its name
    reaches settings resolution (the previous test)."""
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    write_sandbox_home(home, run_toml='[run.make]\nmemory = "1G"\n', extra_settings_files=("make",))
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: "/usr/bin/tool")

    source = 'let result: text = exec("make test", sandbox = Some(Sandbox()))\nresult'
    commands = {"make test": _ok("ok\n")}
    argv_log: list[list[str]] = []
    ir = evaluate_ir_with_shell(
        source,
        commands,
        argv_log=argv_log,
        get_sandbox_context=session_sandbox_context(home),
    )
    from agm.agl.semantics.values import TextValue

    assert ir["result"] == TextValue("ok")
    argv = argv_log[0]
    assert "MemoryMax=1G" in argv
    srt_index = argv.index("srt")
    assert argv[srt_index + 2] == str(home / ".agm" / "sandbox" / "make.json")


def test_unsandboxed_exec_never_requests_a_sandbox_context() -> None:
    """``sandbox = None`` (the default) never builds a ``SandboxContext``: the
    unsandboxed path stays exactly as cheap as before sandboxing existed."""
    source = 'let result: text = exec("echo hi")\nresult'
    commands = {"echo hi": _ok("hi\n")}
    ir = evaluate_ir_with_shell(source, commands, get_sandbox_context=unavailable_sandbox_context)
    from agm.agl.semantics.values import TextValue

    assert ir["result"] == TextValue("hi")


def test_sandboxed_exec_preparation_failure_raises_exec_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A sandbox preparation failure raises ``ExecError`` exactly like a spawn
    failure: exit-code -1, empty stdout, the failure message as stderr, never
    timed out -- no shell process is ever started."""
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: None)

    source = 'exec("echo hi", sandbox = Some(Sandbox()))\n()'
    ir_exc = evaluate_ir_raises_with_shell(
        source, {}, get_sandbox_context=session_sandbox_context(home)
    )
    assert ir_exc.type_name == "ExecError"
    assert ir_exc.fields["exit-code"] == -1
    assert ir_exc.fields["stdout"] == ""
    assert ir_exc.fields["stderr"]
    assert ir_exc.fields["timed-out"] is False


def test_sandboxed_exec_without_a_host_sandbox_context_raises_exec_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A host that passes no ``get_sandbox_context`` (no sandbox capability)
    raises ``ExecError`` for a sandboxed ``exec``, in the same spawn-failure
    shape as any other preparation failure -- no shell process ever starts."""
    monkeypatch.chdir(tmp_path)

    source = 'exec("echo hi", sandbox = Some(Sandbox()))\n()'
    ir_exc = evaluate_ir_raises_with_shell(source, {})
    assert ir_exc.type_name == "ExecError"
    assert ir_exc.fields["exit-code"] == -1
    assert ir_exc.fields["stdout"] == ""
    assert "sandbox context" in ir_exc.fields["stderr"]
    assert ir_exc.fields["timed-out"] is False


def test_sandboxed_exec_retry_reprepares_the_sandbox_on_every_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A retried sandboxed exec prepares (and cleans up) a fresh sandboxed
    command per attempt, rather than reusing the first attempt's prepared argv."""
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    write_sandbox_home(home)
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: "/usr/bin/tool")

    source = 'let n: int = exec("cmd", sandbox = Some(Sandbox()), on-parse-error = Retry(n = 1))\nn'
    call_count = [0]
    argv_log: list[list[str]] = []

    def fake_shell(
        args: list[str],
        *,
        idle_timeout: float | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        isolate_process_group: bool = False,
        interrupt_cleanup_cmd: list[str] | None = None,
    ) -> ProcessCaptureResult:
        del idle_timeout, cwd, env, isolate_process_group, interrupt_cleanup_cmd
        argv_log.append(args)
        assert args[-2] == "-c"
        call_count[0] += 1
        if call_count[0] == 1:
            return _ok("not_a_number\n")
        return _ok("99\n")

    ir = completed_bindings(
        run_inline_ir_with_shell(
            source, fake_shell, get_sandbox_context=session_sandbox_context(home)
        )
    )
    from agm.agl.semantics.values import IntValue

    assert ir["n"] == IntValue(99)
    assert len(argv_log) == 2
    unit_names = []
    for argv in argv_log:
        assert argv[:4] == ["systemd-run", "--user", "--scope", "-q"]
        unit_names.append(argv[argv.index("--unit") + 1])
    # A fresh scope name per attempt proves the sandbox is re-prepared, not
    # reused -- a "prepare once, reuse" implementation would pass every
    # assertion above with the same name both times.
    assert unit_names[0] != unit_names[1]


def test_sandboxed_exec_structured_form_also_honours_sandbox(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The structured (``ExecResult``) form wraps the shell child exactly like
    the parsed form -- ``sandbox`` is not a parsed-form-only operand."""
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    write_sandbox_home(home, extra_settings_files=("echo",))
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: "/usr/bin/tool")

    source = 'let r: ExecResult = exec("echo hi", sandbox = Some(Sandbox()))\nr'
    commands = {"echo hi": _ok("hi\n")}
    argv_log: list[list[str]] = []
    ir = evaluate_ir_with_shell(
        source,
        commands,
        argv_log=argv_log,
        get_sandbox_context=session_sandbox_context(home),
    )
    from agm.agl.semantics.values import IntValue, RecordValue

    assert isinstance(ir["r"], RecordValue)
    assert ir["r"].fields["exit-code"] == IntValue(0)
    assert len(argv_log) == 1
    argv = argv_log[0]
    assert argv[:4] == ["systemd-run", "--user", "--scope", "-q"]
    srt_index = argv.index("srt")
    assert argv[srt_index + 2] == str(home / ".agm" / "sandbox" / "echo.json")


def test_sandboxed_exec_forwards_prepared_env_cwd_and_interrupt_cleanup_cmd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The prepared command's ``env``/``cwd``/``interrupt_cleanup_cmd`` -- not
    the caller's raw operands -- are what actually reach the process
    boundary: the sandbox backend's own env addition, the request's cwd
    verbatim, and a scope-stop teardown command matching the wrapped argv's
    own ``--unit`` name."""
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    write_sandbox_home(home, extra_settings_files=("echo",))
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: "/usr/bin/tool")
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    source = (
        'let result: text = exec("echo hi", '
        f'cwd = Option[text]::Some(value = "{work_dir}"), '
        "sandbox = Some(Sandbox()))\n"
        "result"
    )
    commands = {"echo hi": _ok("hi\n")}
    call_log: list[ScriptedShellCall] = []
    ir = evaluate_ir_with_shell(
        source,
        commands,
        call_log=call_log,
        get_sandbox_context=session_sandbox_context(home),
    )
    from agm.agl.semantics.values import TextValue

    assert ir["result"] == TextValue("hi")
    assert len(call_log) == 1
    call = call_log[0]
    assert call.cwd == work_dir
    assert call.env is not None
    assert call.env["NODE_USE_ENV_PROXY"] == "1"
    unit_name = call.args[call.args.index("--unit") + 1]
    assert call.interrupt_cleanup_cmd == ["systemctl", "--user", "--no-block", "stop", unit_name]


def test_sandboxed_exec_cwd_never_selects_sandbox_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``exec``'s ``cwd`` operand sets only the working directory the command
    runs in. A ``.sandbox/<name>.json`` at that directory is never a settings
    candidate -- only the host ``SandboxContext``'s own directory (here,
    ``home``) is -- so the directory a call operates on can never supply the
    settings that confine it."""
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    write_sandbox_home(home, extra_settings_files=("echo",))
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: "/usr/bin/tool")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    call_site_sandbox = work_dir / ".sandbox"
    call_site_sandbox.mkdir()
    (call_site_sandbox / "echo.json").write_text("{}", encoding="utf-8")

    source = (
        'let result: text = exec("echo hi", '
        f'cwd = Option[text]::Some(value = "{work_dir}"), '
        "sandbox = Some(Sandbox()))\n"
        "result"
    )
    commands = {"echo hi": _ok("hi\n")}
    argv_log: list[list[str]] = []
    call_log: list[ScriptedShellCall] = []
    ir = evaluate_ir_with_shell(
        source,
        commands,
        argv_log=argv_log,
        call_log=call_log,
        get_sandbox_context=session_sandbox_context(home),
    )
    from agm.agl.semantics.values import TextValue

    assert ir["result"] == TextValue("hi")
    argv = argv_log[0]
    srt_index = argv.index("srt")
    # Settings resolve from the host context's own directory (a single
    # candidate found there, never merged with `work_dir/.sandbox/echo.json`)...
    assert argv[srt_index + 2] == str(home / ".agm" / "sandbox" / "echo.json")
    # ...while the command still runs in the per-call `cwd`.
    assert call_log[0].cwd == work_dir


def _spy_prepared_close(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Record every ``PreparedSandboxCommand.close()`` call across this test."""
    from agm.sandbox.request import PreparedSandboxCommand

    calls: list[bool] = []
    original_close = PreparedSandboxCommand.close

    def spy(self: PreparedSandboxCommand) -> None:
        calls.append(True)
        original_close(self)

    monkeypatch.setattr(PreparedSandboxCommand, "close", spy)
    return calls


def test_sandboxed_exec_closes_the_prepared_command_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    write_sandbox_home(home)
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: "/usr/bin/tool")
    closed = _spy_prepared_close(monkeypatch)

    source = 'let result: text = exec("echo hi", sandbox = Some(Sandbox()))\nresult'
    evaluate_ir_with_shell(
        source, {"echo hi": _ok("hi\n")}, get_sandbox_context=session_sandbox_context(home)
    )

    assert closed == [True]


def test_sandboxed_exec_closes_the_prepared_command_on_non_zero_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    write_sandbox_home(home)
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: "/usr/bin/tool")
    closed = _spy_prepared_close(monkeypatch)

    source = 'let result: text = exec("false", sandbox = Some(Sandbox()))\nresult'
    evaluate_ir_raises_with_shell(
        source, {"false": _fail(1)}, get_sandbox_context=session_sandbox_context(home)
    )

    assert closed == [True]


def test_sandboxed_exec_closes_the_prepared_command_on_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    write_sandbox_home(home)
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: "/usr/bin/tool")
    closed = _spy_prepared_close(monkeypatch)

    source = 'let result: text = exec("sleep 99", sandbox = Some(Sandbox()))\nresult'
    evaluate_ir_raises_with_shell(
        source, {"sleep 99": _timed_out()}, get_sandbox_context=session_sandbox_context(home)
    )

    assert closed == [True]


def test_sandboxed_exec_closes_the_prepared_command_on_spawn_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    write_sandbox_home(home)
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: "/usr/bin/tool")
    closed = _spy_prepared_close(monkeypatch)

    source = 'let result: text = exec("cmd", sandbox = Some(Sandbox()))\nresult'
    evaluate_ir_raises_with_shell(
        source, {"cmd": _spawn_failed()}, get_sandbox_context=session_sandbox_context(home)
    )

    assert closed == [True]
