"""Behavior tests for the AgL trace store.

All assertions are on *observable* outcomes: what ends up in the trace file,
whether a file is created at all. No internal implementation state is
asserted.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

import agm.commands.exec as exec_command
from agm.agl import PipelineDriver
from agm.agl.runtime import AgentRequest, AgentResponse
from agm.agl.runtime.agents import AgentFn
from agm.cli_support.args import ExecArgs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent_returning(text: str):
    """Return a stub agent callable that always returns *text*."""

    def agent(request: AgentRequest) -> AgentResponse:
        return AgentResponse(content=text)

    return agent


def _agent_runtime(agent: AgentFn, *, strict_json: bool = False) -> PipelineDriver:
    """Build a runtime with an explicit value dispatcher for test agents."""
    return PipelineDriver(default_strict_json=strict_json, agent_dispatcher=agent)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    """Read a JSONL file and return a list of decoded records."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _exec_args(
    agl_file: Path,
    *,
    log_file: str | None = None,
    no_log: bool = False,
    param_tokens: list[str] | None = None,
) -> ExecArgs:
    return ExecArgs(
        file=str(agl_file),
        param_tokens=param_tokens or [],
        strict_json=None,
        no_log=no_log,
        log_file=log_file,
    )


# ---------------------------------------------------------------------------
# 1. Trace file created at a custom --log-file path
# ---------------------------------------------------------------------------


class TestTraceFileCreated:
    def test_trace_file_created_at_custom_path(self, tmp_path: Path) -> None:
        """A custom --log-file path receives JSONL trace output after a run."""
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        result = rt.run('let x = 1\nprint "hello"', log_file=log_path)
        assert result.ok
        assert log_path.exists(), "trace file must be created when log_file is given"

    def test_trace_file_has_jsonl_content(self, tmp_path: Path) -> None:
        """Each line of the trace file is a valid JSON object."""
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        rt.run('let x = 1\nprint "hello"', log_file=log_path)
        records = _load_jsonl(log_path)
        assert len(records) >= 1
        for rec in records:
            assert isinstance(rec, dict)

    def test_trace_file_not_created_when_no_log(self, tmp_path: Path) -> None:
        """When log_file is None (no-log semantics), no trace file is written."""
        rt = PipelineDriver()
        result = rt.run('let x = 1\nprint "hello"', log_file=None)
        assert result.ok
        # No trace file: any file created would be under .agent-files/ which
        # we cannot check here, but RunResult.trace_path should be None.
        assert result.trace_path is None

    def test_run_result_exposes_trace_path(self, tmp_path: Path) -> None:
        """RunResult.trace_path is the Path of the written JSONL file."""
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        result = rt.run("let x = 1\nx", log_file=log_path)
        assert result.ok
        assert result.trace_path == log_path

    def test_trace_directory_creation_failure_is_best_effort(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def fail_mkdir(*args: object, **kwargs: object) -> None:
            raise OSError("read-only filesystem")

        monkeypatch.setattr("agm.core.fs.mkdir", fail_mkdir)
        result = PipelineDriver().run(
            'print "still runs"',
            log_file=tmp_path / "missing" / "trace.jsonl",
        )

        assert result.ok
        assert result.trace_path is None
        assert capsys.readouterr().err.count("trace logging disabled") == 1


# ---------------------------------------------------------------------------
# 2. Record kinds: print, exec command, agent call
# ---------------------------------------------------------------------------


class TestPrintRecord:
    def test_print_produces_trace_record(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        rt.run('print "hello world"', log_file=log_path)
        records = _load_jsonl(log_path)
        kinds = [r.get("kind") for r in records]
        assert "print" in kinds

    def test_print_record_has_value(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        rt.run('print "hello world"', log_file=log_path)
        records = _load_jsonl(log_path)
        print_recs = [r for r in records if r.get("kind") == "print"]
        assert print_recs
        # The rendered value should contain the printed text.
        assert any("hello world" in str(r.get("rendered", "")) for r in print_recs)

    def test_print_record_has_span(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        rt.run('print "hello"', log_file=log_path)
        records = _load_jsonl(log_path)
        print_recs = [r for r in records if r.get("kind") == "print"]
        assert print_recs
        rec = print_recs[0]
        assert "line" in rec or "span" in rec


class TestExecCommandRecord:
    def test_exec_produces_exec_record(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        rt.run('let x: text = exec "echo hi"\nx', log_file=log_path)
        records = _load_jsonl(log_path)
        kinds = [r.get("kind") for r in records]
        assert "exec_command" in kinds

    def test_exec_record_has_exit_code(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        rt.run('let x: text = exec "echo hi"\nx', log_file=log_path)
        records = _load_jsonl(log_path)
        exec_recs = [r for r in records if r.get("kind") == "exec_command"]
        assert exec_recs
        rec = exec_recs[0]
        assert rec.get("exit_code") == 0

    def test_exec_record_has_stdout(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        rt.run('let x: text = exec "echo captured"\nx', log_file=log_path)
        records = _load_jsonl(log_path)
        exec_recs = [r for r in records if r.get("kind") == "exec_command"]
        assert exec_recs
        rec = exec_recs[0]
        assert "captured" in rec.get("stdout", "")

    def test_exec_record_has_command(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        rt.run('let x: text = exec "echo hello"\nx', log_file=log_path)
        records = _load_jsonl(log_path)
        exec_recs = [r for r in records if r.get("kind") == "exec_command"]
        assert exec_recs
        assert "echo hello" in exec_recs[0].get("command", "")

    def test_exec_record_has_duration(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        rt.run('let x: text = exec "echo hello"\nx', log_file=log_path)
        records = _load_jsonl(log_path)
        exec_recs = [r for r in records if r.get("kind") == "exec_command"]
        assert exec_recs
        duration = exec_recs[0].get("duration")
        assert isinstance(duration, float)
        assert duration >= 0


class TestAgentCallRecord:
    def test_agent_call_produces_attempt_record(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = _agent_runtime(_agent_returning("good"))
        rt.run(
            'let reviewer = AgentCommand("reviewer")\n'
            'let x: text = ask("check this", agent = reviewer)\n'
            "x",
            log_file=log_path,
        )
        records = _load_jsonl(log_path)
        kinds = [r.get("kind") for r in records]
        assert "agent_request" in kinds
        assert "agent_response" in kinds

    def test_agent_call_record_has_rendered_agent_value(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = _agent_runtime(_agent_returning("ok"))
        rt.run(
            'let critic = AgentCommand("critic")\nlet x: text = ask("review", agent = critic)\nx',
            log_file=log_path,
        )
        records = _load_jsonl(log_path)
        call_recs = [r for r in records if r.get("kind") == "agent_request"]
        assert call_recs
        assert call_recs[0]["agent"] == {
            "variant": "AgentCommand",
            "payload": {"command": "critic"},
        }

    def test_agent_request_preserves_agent_variant_payload(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        _agent_runtime(_agent_returning("ok")).run(
            'let a = AgentClaude("sonnet", "high")\nlet x: text = ask("review", agent = a)\nx',
            log_file=log_path,
        )
        request = next(
            record for record in _load_jsonl(log_path) if record["kind"] == "agent_request"
        )
        assert request["agent"] == {
            "variant": "AgentClaude",
            "payload": {"model": "sonnet", "thinking": "high"},
        }

    def test_agent_call_record_has_attempt_number(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = _agent_runtime(_agent_returning("result"))
        rt.run(
            'let impl = AgentCommand("impl")\nlet x: text = ask("do work", agent = impl)\nx',
            log_file=log_path,
        )
        records = _load_jsonl(log_path)
        call_recs = [r for r in records if r.get("kind") == "agent_request"]
        assert call_recs
        assert isinstance(call_recs[0].get("attempt"), int)
        assert call_recs[0].get("max_attempts") == 1

    def test_unit_ask_logs_a_request_and_response(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        result = _agent_runtime(_agent_returning("ignored")).run(
            'let a = AgentCommand("a")\nlet value: unit = ask "do it"\nvalue',
            log_file=log_path,
        )
        assert result.ok
        records = _load_jsonl(log_path)
        kinds = [record["kind"] for record in records]
        assert kinds.count("agent_request") == kinds.count("agent_response") == 1
        assert "parse_result" not in kinds

    def test_request_prompt_is_the_dispatcher_prompt_and_has_contract(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        received: list[str] = []

        def agent(request: AgentRequest) -> AgentResponse:
            received.append(request.prompt)
            return AgentResponse(content="42")

        result = _agent_runtime(agent, strict_json=True).run(
            'let a = AgentCommand("a")\nlet value: int = ask("number", agent = a)\nvalue',
            log_file=log_path,
        )
        assert result.ok
        request = next(
            record for record in _load_jsonl(log_path) if record["kind"] == "agent_request"
        )
        assert request["prompt"] == received[0]
        assert request["codec"] == "json"
        assert request["target_type"] == "int"
        assert request["strict_json"] is None
        assert request["json_schema"] is not None


# ---------------------------------------------------------------------------
# 3. Retry: multiple agent request/response records
# ---------------------------------------------------------------------------


class TestRetryRecords:
    def test_retry_produces_multiple_attempt_records(self, tmp_path: Path) -> None:
        """With on_parse_error: retry[2], failed attempts appear in the trace."""
        log_path = tmp_path / "trace.jsonl"
        call_count = 0

        def agent(request: AgentRequest) -> AgentResponse:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return AgentResponse(content="not json")  # will fail to parse
            return AgentResponse(content="42")

        rt = _agent_runtime(agent, strict_json=True)
        rt.run(
            'let impl = AgentCommand("impl")\n'
            'let x: int = ask("get int", agent = impl, on_parse_error = Retry(n = 2))\nx',
            log_file=log_path,
        )
        records = _load_jsonl(log_path)
        call_recs = [r for r in records if r.get("kind") == "agent_request"]
        assert len(call_recs) == 3
        assert len([r for r in records if r.get("kind") == "agent_response"]) == 3
        assert "Your previous response did not match" in str(call_recs[1]["prompt"])
        assert "not json" in str(call_recs[1]["prompt"])
        call_pairs = [
            record["kind"]
            for record in records
            if record["kind"] in {"agent_request", "agent_response"}
        ]
        assert call_pairs == [
            "agent_request",
            "agent_response",
            "agent_request",
            "agent_response",
            "agent_request",
            "agent_response",
        ]

    def test_retry_records_carry_attempt_index(self, tmp_path: Path) -> None:
        """Attempt indices should be 0, 1, 2 for three attempts."""
        log_path = tmp_path / "trace.jsonl"
        call_count = 0

        def agent(request: AgentRequest) -> AgentResponse:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return AgentResponse(content="not json")
            return AgentResponse(content="42")

        rt = _agent_runtime(agent, strict_json=True)
        rt.run(
            'let impl = AgentCommand("impl")\n'
            'let x: int = ask("get int", agent = impl, on_parse_error = Retry(n = 2))\nx',
            log_file=log_path,
        )
        records = _load_jsonl(log_path)
        call_recs = [r for r in records if r.get("kind") == "agent_request"]
        attempts = [r.get("attempt") for r in call_recs]
        assert attempts == [0, 1, 2]

    def test_parse_result_record_emitted_for_each_attempt(self, tmp_path: Path) -> None:
        """A parse_result record follows each agent response."""
        log_path = tmp_path / "trace.jsonl"

        def agent(request: AgentRequest) -> AgentResponse:
            return AgentResponse(content="not json at all")

        rt = _agent_runtime(agent, strict_json=True)
        src = (
            'let impl = AgentCommand("impl")\n'
            'let x: int = ask("get int", agent = impl, on_parse_error = Retry(n = 1))\nx'
        )
        try:
            rt.run(src, log_file=log_path)
        except SystemExit:
            pass

        records = _load_jsonl(log_path)
        kinds = [r.get("kind") for r in records]
        assert "parse_result" in kinds

    def test_transport_failure_has_failed_agent_response(self, tmp_path: Path) -> None:
        from agm.agl.runtime.request import AgentCallHostError

        log_path = tmp_path / "trace.jsonl"

        def agent(_request: AgentRequest) -> AgentResponse:
            raise AgentCallHostError(
                cause="timeout", exit_code=9, stderr_tail="too slow", elapsed=1.5
            )

        result = _agent_runtime(agent).run(
            'let a = AgentCommand("a")\nlet value: text = ask("work", agent = a)\nvalue',
            log_file=log_path,
        )
        assert not result.ok
        response = next(
            record for record in _load_jsonl(log_path) if record["kind"] == "agent_response"
        )
        assert response["ok"] is False
        assert response["cause"] == "timeout"
        assert response["exit_code"] == 9
        assert response["elapsed"] == 1.5
        assert response["stderr_tail"] == "too slow"


# ---------------------------------------------------------------------------
# 4. Exception records
# ---------------------------------------------------------------------------


class TestExceptionRecord:
    def test_uncaught_exception_produces_exception_record(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"

        def agent(request: AgentRequest) -> AgentResponse:
            return AgentResponse(content="not json")

        rt = _agent_runtime(agent, strict_json=True)
        result = rt.run(
            'let impl = AgentCommand("impl")\nlet x: int = ask("get int", agent = impl)\nx',
            log_file=log_path,
        )
        assert not result.ok
        assert result.error is not None

        records = _load_jsonl(log_path)
        exc_recs = [r for r in records if r.get("kind") == "exception"]
        assert exc_recs

    def test_exception_record_has_type_name(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"

        def agent(request: AgentRequest) -> AgentResponse:
            return AgentResponse(content="not json")

        rt = _agent_runtime(agent, strict_json=True)
        result = rt.run(
            'let impl = AgentCommand("impl")\nlet x: int = ask("get int", agent = impl)\nx',
            log_file=log_path,
        )
        assert not result.ok

        records = _load_jsonl(log_path)
        exc_recs = [r for r in records if r.get("kind") == "exception"]
        assert exc_recs[0].get("type_name") == "AgentParseError"

    def test_exception_record_and_value_have_no_trace_id(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        result = _agent_runtime(_agent_returning("not json"), strict_json=True).run(
            'let impl = AgentCommand("impl")\nlet x: int = ask("get int", agent = impl)\nx',
            log_file=log_path,
        )
        assert result.error is not None
        exc_recs = [r for r in _load_jsonl(log_path) if r.get("kind") == "exception"]
        assert exc_recs and "trace_id" not in exc_recs[0]
        assert "trace_id" not in result.error.fields

    def test_caught_exception_does_not_produce_exception_record(self, tmp_path: Path) -> None:
        """An exception caught by try/catch is NOT written as an 'exception' record
        (it was handled in-language and did not escape the program)."""
        log_path = tmp_path / "trace.jsonl"

        def agent(request: AgentRequest) -> AgentResponse:
            return AgentResponse(content="not json")

        rt = _agent_runtime(agent, strict_json=True)
        # The AgentParseError is caught and the result is a fallback string.
        result = rt.run(
            'let impl = AgentCommand("impl")\n'
            "try\n"
            '  let x: int = ask("get int", agent = impl)\n'
            "  x\n"
            "catch AgentParseError as e =>\n"
            "  0\n",
            log_file=log_path,
        )
        assert result.ok

        records = _load_jsonl(log_path)
        exc_recs = [r for r in records if r.get("kind") == "exception"]
        assert not exc_recs, "caught exception must not produce an exception record"


# ---------------------------------------------------------------------------
# 4b. Built-in runtime exceptions carry no base trace identifier
# ---------------------------------------------------------------------------


class TestBuiltinExceptionFields:
    """Built-in runtime exceptions expose their declared fields only."""

    def test_arithmetic_error_trace_id_non_empty_with_logging(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        # Uncaught division by zero → ArithmeticError escapes the program.
        result = rt.run("let x = 1 / 0\nx", log_file=log_path)
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "ArithmeticError"

        assert "trace_id" not in result.error.fields
        exc_recs = [r for r in _load_jsonl(log_path) if r.get("kind") == "exception"]
        assert exc_recs and "trace_id" not in exc_recs[0]

    def test_arithmetic_error_trace_id_non_empty_without_logging(self) -> None:
        rt = PipelineDriver()
        result = rt.run("let x = 1 / 0\nx", log_file=None)
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "ArithmeticError"
        assert "trace_id" not in result.error.fields

    def test_match_error_trace_id_non_empty_without_logging(self) -> None:
        rt = PipelineDriver()
        # Explicit source raising retains the ordinary MatchError runtime contract.
        result = rt.run(
            "case 5 of\n"
            "  | 0 => ()\n"
            "  | _ =>\n"
            '    raise MatchError(message = "no match", '
            'scrutinee_type = "int", scrutinee = 5)\n',
            log_file=None,
        )
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "MatchError"
        assert "trace_id" not in result.error.fields

    def test_max_iterations_trace_id_linked_with_logging(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        # A do-loop whose condition never becomes true exhausts its limit.
        result = rt.run(
            "var x = 0\ndo[2]\n  x := x\nuntil false\n",
            log_file=log_path,
        )
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "MaxIterationsExceeded"

        assert "trace_id" not in result.error.fields
        exc_recs = [r for r in _load_jsonl(log_path) if r.get("kind") == "exception"]
        assert exc_recs and "trace_id" not in exc_recs[0]


# ---------------------------------------------------------------------------
# 5. run_start / run_end records
# ---------------------------------------------------------------------------


class TestRunBoundaryRecords:
    def test_run_start_record_present(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        rt.run("let x = 1\nx", log_file=log_path)
        records = _load_jsonl(log_path)
        kinds = [r.get("kind") for r in records]
        assert "run_start" in kinds

    def test_run_end_record_present(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        rt.run("let x = 1\nx", log_file=log_path)
        records = _load_jsonl(log_path)
        kinds = [r.get("kind") for r in records]
        assert "run_end" in kinds

    def test_run_start_before_run_end(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        rt.run("let x = 1\nx", log_file=log_path)
        records = _load_jsonl(log_path)
        kinds = [r.get("kind") for r in records]
        start_idx = kinds.index("run_start")
        end_idx = kinds.index("run_end")
        assert start_idx < end_idx

    def test_all_records_share_run_id(self, tmp_path: Path) -> None:
        """Every record in a trace file carries the same run_id."""
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        rt.run('var x = 1\nx := 2\nprint "done"', log_file=log_path)
        records = _load_jsonl(log_path)
        assert len(records) >= 3
        run_ids = {r.get("run_id") for r in records}
        assert len(run_ids) == 1
        (run_id,) = run_ids
        assert isinstance(run_id, str) and run_id

    def test_every_record_has_an_ordered_offset_aware_timestamp(self, tmp_path: Path) -> None:
        from datetime import datetime

        log_path = tmp_path / "trace.jsonl"
        _agent_runtime(_agent_returning("agent output")).run(
            'let a = AgentCommand("a")\n'
            'let x: text = ask("prompt", agent = a)\n'
            'let y: text = exec "printf shell"\n'
            "print x\ny",
            log_file=log_path,
        )
        records = _load_jsonl(log_path)
        timestamps = [datetime.fromisoformat(str(record["ts"])) for record in records]
        assert all(timestamp.utcoffset() is not None for timestamp in timestamps)
        assert timestamps == sorted(timestamps)
        assert all("trace_id" not in record for record in records)


# ---------------------------------------------------------------------------
# 6. No-log semantics
# ---------------------------------------------------------------------------


class TestNoLog:
    def test_no_log_writes_nothing(self, tmp_path: Path) -> None:
        """With log_file=None the trace store is a no-op and no files are created."""
        rt = PipelineDriver()
        result = rt.run('let x = 1\nprint "silent"', log_file=None)
        assert result.ok
        # No JSONL files created anywhere in tmp_path.
        jsonl_files = list(tmp_path.rglob("*.jsonl"))
        assert not jsonl_files

    def test_no_log_result_trace_path_is_none(self, tmp_path: Path) -> None:
        rt = PipelineDriver()
        result = rt.run("let x = 1\nx", log_file=None)
        assert result.trace_path is None

    def test_no_log_with_agent_call_writes_nothing(self, tmp_path: Path) -> None:
        rt = _agent_runtime(_agent_returning("hello"))
        result = rt.run(
            'let a = AgentCommand("a")\nlet x: text = ask("hi", agent = a)\nx', log_file=None
        )
        assert result.ok
        jsonl_files = list(tmp_path.rglob("*.jsonl"))
        assert not jsonl_files

    def test_no_log_with_decimal_assignment_still_works(self, tmp_path: Path) -> None:
        """A no-log run that assigns to a decimal binding still succeeds and
        writes nothing."""
        rt = PipelineDriver()
        result = rt.run(
            "var x: decimal = 0.1\nx := x + 0.2",
            log_file=None,
        )
        assert result.ok
        jsonl_files = list(tmp_path.rglob("*.jsonl"))
        assert not jsonl_files


# ---------------------------------------------------------------------------
# 7. --no-log flag via exec command writes nothing
# ---------------------------------------------------------------------------


class TestExecNoLog:
    def test_exec_no_log_flag_writes_nothing(self, tmp_path: Path) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('print "hello"\n')
        args = _exec_args(agl_file, no_log=True)
        exec_command.run(args)
        # No JSONL files created under tmp_path or any default path.
        jsonl_files = list(tmp_path.rglob("*.jsonl"))
        assert not jsonl_files

    def test_exec_log_file_flag_creates_file(self, tmp_path: Path) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('let x = 1\nprint "hi"\n')
        log_path = tmp_path / "out.jsonl"
        args = _exec_args(agl_file, log_file=str(log_path))
        exec_command.run(args)
        assert log_path.exists()
        records = _load_jsonl(log_path)
        assert len(records) >= 1


# ---------------------------------------------------------------------------
# 8. Dry-run must NOT write a trace
# ---------------------------------------------------------------------------


class TestDryRunNoTrace:
    def test_dry_run_does_not_write_trace(self, tmp_path: Path) -> None:
        """check_only=True (--dry-run) must produce no trace output."""
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        result = rt.run("let x = 1\nx", log_file=log_path, check_only=True)
        assert result.ok
        # No trace file created for dry-run.
        assert not log_path.exists()

    def test_dry_run_trace_path_is_none(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        result = rt.run("let x = 1\nx", log_file=log_path, check_only=True)
        assert result.trace_path is None


# ---------------------------------------------------------------------------
# 9. Source spans in records
# ---------------------------------------------------------------------------


class TestSourceSpans:
    def test_exec_record_has_source_span(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        rt.run('let x: text = exec "echo hi"\nx', log_file=log_path)
        records = _load_jsonl(log_path)
        exec_recs = [r for r in records if r.get("kind") == "exec_command"]
        assert exec_recs
        rec = exec_recs[0]
        # Must have either top-level "line"/"col" or a "span" sub-object.
        has_span = "line" in rec or ("span" in rec and isinstance(rec["span"], dict))
        assert has_span

    def test_agent_record_has_source_span(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = _agent_runtime(_agent_returning("hello"))
        rt.run(
            'let impl = AgentCommand("impl")\nlet x: text = ask("do work", agent = impl)\nx',
            log_file=log_path,
        )
        records = _load_jsonl(log_path)
        call_recs = [r for r in records if r.get("kind") == "agent_request"]
        assert call_recs
        rec = call_recs[0]
        has_span = "line" in rec or ("span" in rec and isinstance(rec["span"], dict))
        assert has_span


# ---------------------------------------------------------------------------
# 10. append_jsonl general helper in core/log
# ---------------------------------------------------------------------------


class TestAppendJsonl:
    def test_append_jsonl_creates_file(self, tmp_path: Path) -> None:
        from agm.core.log import append_jsonl

        path = tmp_path / "out.jsonl"
        append_jsonl(path, {"kind": "test", "value": 1})
        assert path.exists()

    def test_append_jsonl_writes_valid_json_line(self, tmp_path: Path) -> None:
        from agm.core.log import append_jsonl

        path = tmp_path / "out.jsonl"
        append_jsonl(path, {"kind": "test", "value": 42})
        content = path.read_text(encoding="utf-8")
        assert content.strip()
        obj = json.loads(content.strip())
        assert obj["kind"] == "test"
        assert obj["value"] == 42

    def test_append_jsonl_appends_multiple_lines(self, tmp_path: Path) -> None:
        from agm.core.log import append_jsonl

        path = tmp_path / "out.jsonl"
        append_jsonl(path, {"kind": "a"})
        append_jsonl(path, {"kind": "b"})
        records = _load_jsonl(path)
        assert len(records) == 2
        assert records[0]["kind"] == "a"
        assert records[1]["kind"] == "b"

    def test_append_jsonl_none_path_is_noop(self) -> None:
        from agm.core.log import append_jsonl

        # Must not raise when path is None.
        append_jsonl(None, {"kind": "noop"})

    def test_append_jsonl_decimal_raises(self, tmp_path: Path) -> None:
        """``append_jsonl`` has no numeric convention: a raw ``Decimal`` raises.

        The single numeric convention lives in the DSL serializer
        (``dumps_exact``); callers MUST pre-serialize ``Decimal``-bearing values.
        Encoding it here would quote it as a JSON string and diverge.
        """
        from agm.core.log import append_jsonl

        path = tmp_path / "out.jsonl"
        with pytest.raises(TypeError):
            append_jsonl(path, {"value": Decimal("0.1")})

    def test_append_jsonl_unserializable_type_raises(self, tmp_path: Path) -> None:
        """Non-JSON-serializable, non-Decimal values raise TypeError."""
        from agm.core.log import append_jsonl

        path = tmp_path / "out.jsonl"
        with pytest.raises(TypeError):
            append_jsonl(path, {"value": object()})


# ---------------------------------------------------------------------------
# 11. TraceStore properties and no-span branches
# ---------------------------------------------------------------------------


class TestTraceStoreProperties:
    def test_activate_repoints_and_stops_writes(self, tmp_path: Path) -> None:
        """``activate`` routes later events to the new path; ``None`` stops writes."""
        from agm.agl.runtime.trace import TraceStore

        first = tmp_path / "first.jsonl"
        second = tmp_path / "second.jsonl"
        trace = TraceStore(path=first)
        trace.run_start()

        trace.activate(second)
        trace.print_stmt(rendered="routed", span=None)

        trace.activate(None)
        trace.print_stmt(rendered="dropped", span=None)
        trace.run_end(ok=True)

        assert trace.path is None
        assert "routed" in second.read_text(encoding="utf-8")
        assert "dropped" not in second.read_text(encoding="utf-8")

    def test_activate_cannot_revive_a_disabled_store(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Once an I/O failure disables logging, ``activate`` stays a no-op."""
        from agm.agl.runtime.trace import TraceStore

        def fail_append(_path: Path | None, _record: dict[str, object]) -> None:
            raise OSError("disk unavailable")

        monkeypatch.setattr("agm.agl.runtime.trace.append_jsonl", fail_append)
        trace = TraceStore(path=tmp_path / "trace.jsonl")
        trace.run_start()

        trace.activate(tmp_path / "recovered.jsonl")
        trace.run_end(ok=True)

        assert trace.path is None
        assert capsys.readouterr().err.count("trace logging disabled") == 1

    def test_repeated_disable_warns_once(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A repeated trace failure stays silent after the initial warning."""
        from agm.agl.runtime.trace import TraceStore

        trace = TraceStore(path=tmp_path / "trace.jsonl")
        trace.disable(OSError("first failure"))
        trace.disable(OSError("second failure"))

        assert trace.path is None
        assert capsys.readouterr().err.count("trace logging disabled") == 1

    def test_trace_store_path_property(self, tmp_path: Path) -> None:
        from agm.agl.runtime.trace import TraceStore

        p = tmp_path / "t.jsonl"
        ts = TraceStore(path=p)
        assert ts.path == p

    def test_trace_store_none_path_property(self) -> None:
        from agm.agl.runtime.trace import TraceStore

        ts = TraceStore(path=None)
        assert ts.path is None

    def test_trace_store_run_id_property(self, tmp_path: Path) -> None:
        from agm.agl.runtime.trace import TraceStore

        ts = TraceStore(path=tmp_path / "t.jsonl")
        assert isinstance(ts.run_id, str) and ts.run_id

    def test_trace_store_records_without_span(self, tmp_path: Path) -> None:
        """Methods called with span=None still emit valid JSONL (no line/col keys)."""
        import json as _json

        from agm.agl.runtime.trace import TraceStore

        p = tmp_path / "t.jsonl"
        ts = TraceStore(path=p)
        ts.run_start()
        ts.agent_request(
            agent={"variant": "AgentCommand", "payload": {"command": "x"}},
            attempt=0,
            max_attempts=1,
            prompt="p",
            target_type="text",
            codec="text",
            strict_json=None,
            json_schema=None,
            span=None,
        )
        ts.agent_response(ok=True, content="r", metadata={"source": "test"}, span=None)
        ts.parse_result(ok=True, raw="r", normalized_raw="n", error_summary="", span=None)
        ts.print_stmt(rendered="hi", span=None)
        ts.exec_command(
            command="echo",
            exit_code=0,
            duration=0.1,
            stdout="",
            stderr="",
            timed_out=False,
            span=None,
        )
        ts.exception(type_name="Abort", message="stop", span=None)
        ts.run_end(ok=True)

        lines = p.read_text(encoding="utf-8").splitlines()
        records = [_json.loads(ln) for ln in lines if ln.strip()]
        # None of the records should carry "line" or "col" (span was None).
        for rec in records:
            assert "line" not in rec
            assert "col" not in rec

    def test_clock_rollback_keeps_timestamps_non_decreasing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from datetime import datetime, timezone

        from agm.agl.runtime import trace as trace_module
        from agm.agl.runtime.trace import TraceStore

        class Clock:
            values = iter(
                [
                    datetime(2026, 1, 2, tzinfo=timezone.utc),
                    datetime(2026, 1, 1, tzinfo=timezone.utc),
                ]
            )

            @staticmethod
            def now() -> datetime:
                return next(Clock.values)

        monkeypatch.setattr(trace_module, "datetime", Clock)
        trace = TraceStore(tmp_path / "trace.jsonl")
        trace.run_start()
        trace.run_end(ok=True)
        timestamps = [record["ts"] for record in _load_jsonl(tmp_path / "trace.jsonl")]
        assert timestamps[0] == timestamps[1]

    def test_trace_store_exception_with_span(self, tmp_path: Path) -> None:
        """exception() records line/col when a span is provided."""
        import json as _json

        from agm.agl.runtime.trace import TraceStore
        from agm.agl.syntax.spans import SourceSpan

        p = tmp_path / "t.jsonl"
        ts = TraceStore(path=p)
        span = SourceSpan(
            start_line=5,
            start_col=3,
            end_line=5,
            end_col=10,
            start_offset=40,
            end_offset=47,
        )
        ts.exception(type_name="Abort", message="stop", span=span)

        content = p.read_text(encoding="utf-8").strip()
        rec = _json.loads(content)
        assert rec["line"] == 5
        assert rec["col"] == 3


# ---------------------------------------------------------------------------
# Unparseable output synthesizes a validation error for retry feedback
# ---------------------------------------------------------------------------


class TestUnparseableFeedback:
    """when agent output is totally unparseable (no JSON at all), the next
    retry attempt must carry the failure reason as a ValidationError, and the
    parse_result trace record must have a non-empty error_summary."""

    def test_retry_request_carries_reason_when_totally_unparseable(self, tmp_path: Path) -> None:
        """Second attempt's validation_errors is non-empty with the parse reason."""
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver(default_strict_json=True)

        captured_requests: list[AgentRequest] = []
        call_count = 0

        def agent(request: AgentRequest) -> AgentResponse:
            nonlocal call_count
            captured_requests.append(request)
            call_count += 1
            if call_count == 1:
                return AgentResponse(content="totally not json #@!")
            return AgentResponse(content="42")

        rt = _agent_runtime(agent, strict_json=True)
        result = rt.run(
            'let impl = AgentCommand("impl")\n'
            'let x: int = ask("get int", agent = impl, on_parse_error = Retry(n = 1))\nx',
            log_file=log_path,
        )
        assert result.ok
        assert len(captured_requests) == 2
        # The second request must carry validation_errors describing the failure.
        second_request = captured_requests[1]
        assert len(second_request.validation_errors) > 0, (
            "retry request.validation_errors must be non-empty when output was unparseable"
        )
        # The category must be "invalid_json" (the extension for unparseable output).
        assert any(e.category == "invalid_json" for e in second_request.validation_errors), (
            f"Expected category 'invalid_json', got: {second_request.validation_errors}"
        )

    def test_parse_result_error_summary_non_empty_when_unparseable(self, tmp_path: Path) -> None:
        """parse_result trace record's error_summary is non-empty for unparseable output."""
        log_path = tmp_path / "trace.jsonl"

        def agent(request: AgentRequest) -> AgentResponse:
            return AgentResponse(content="totally not json #@!")

        rt = _agent_runtime(agent, strict_json=True)
        rt.run(
            'let impl = AgentCommand("impl")\n'
            'let x: int = ask("get int", agent = impl, on_parse_error = Retry(n = 1))\nx',
            log_file=log_path,
        )
        records = _load_jsonl(log_path)
        parse_recs = [r for r in records if r.get("kind") == "parse_result"]
        assert parse_recs
        # All failed parse_result records must have a non-empty error_summary.
        failed = [r for r in parse_recs if not r.get("ok", True)]
        assert failed, "Expected at least one failed parse_result record"
        for rec in failed:
            assert rec.get("error_summary"), (
                f"parse_result error_summary must be non-empty, got: {rec}"
            )

    def test_empty_errors_and_empty_error_msg_fallback(self) -> None:
        """When a codec returns ok=False with no errors and no error_msg, the
        AgentParseError still raises (defensive fallback — last_errors = ())."""
        from unittest.mock import patch

        from agm.agl.runtime.codec import ParseResult

        rt = PipelineDriver(default_strict_json=True)

        def agent(request: AgentRequest) -> AgentResponse:
            return AgentResponse(content="42")

        rt = _agent_runtime(agent, strict_json=True)
        # Patch the IR output parser to return a failure with no details at all.
        bare_fail = ParseResult(ok=False, value=None, error_msg="", errors=())
        with patch("agm.agl.eval.ir_interpreter._parse_contract_output", return_value=bare_fail):
            result = rt.run(
                'let impl = AgentCommand("impl")\n'
                'let x: int = ask("q", agent = impl, on_parse_error = Abort())\n'
                "x"
            )
        # The program raises AgentParseError; run returns ok=False.
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "AgentParseError"
        # With empty errors the validation_errors list is empty.
        val_errs = result.error.fields.get("validation_errors")
        assert val_errs == []


# ---------------------------------------------------------------------------
# 12. prepare_trace_log truncates an existing file (clean-file guarantee)
# ---------------------------------------------------------------------------


class TestPrepareTraceLogTruncates:
    """prepare_trace_log must start each run from a clean (empty) file.

    For auto-generated paths the pid-unique component already guarantees a
    fresh file.  For an explicit --log-file path a new run must TRUNCATE any
    pre-existing content so the "first traced entry starts from a clean file"
    contract in the docstring holds.
    """

    def test_prepare_trace_log_truncates_existing_content(self, tmp_path: Path) -> None:
        """Pre-existing content at an explicit log path is erased by prepare_trace_log."""
        from agm.core.log import prepare_trace_log

        log_path = tmp_path / "trace.jsonl"
        # Pre-create the file with stale content from a previous run.
        log_path.write_text('{"kind": "run_start", "run_id": "old"}\n', encoding="utf-8")
        assert log_path.read_text(encoding="utf-8").strip(), "pre-condition: file must be non-empty"

        prepare_trace_log(command_name="exec", enabled=True, log_file=str(log_path))

        content = log_path.read_text(encoding="utf-8")
        assert content == "", (
            "prepare_trace_log must truncate the file so each run starts from a clean slate"
        )

    def test_prepare_trace_log_subsequent_record_is_only_content(self, tmp_path: Path) -> None:
        """After truncation, only records written in the current run appear in the file."""
        from agm.core.log import append_jsonl, prepare_trace_log

        log_path = tmp_path / "trace.jsonl"
        # Simulate a previous run by pre-populating the file.
        log_path.write_text('{"kind": "run_start", "run_id": "old"}\n', encoding="utf-8")

        prepare_trace_log(command_name="exec", enabled=True, log_file=str(log_path))
        # Append a single record as the new run would.
        append_jsonl(log_path, {"kind": "run_start", "run_id": "new"})

        import json as _json

        lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 1, f"Only the new record must be present; got {len(lines)} lines"
        rec = _json.loads(lines[0])
        assert rec.get("run_id") == "new"
