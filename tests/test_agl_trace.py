"""Behavior tests for the AgL trace store.

All assertions are on *observable* outcomes: what ends up in the trace file,
whether a file is created at all. No internal implementation state is
asserted.
"""

from __future__ import annotations

import json
import unittest.mock
from decimal import Decimal
from pathlib import Path
from typing import Protocol, cast

import pytest
import semver

import agm.commands.exec as exec_command
from agm.agl import PipelineDriver
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import RunResult
from agm.agl.runtime import AgentRequest, AgentResponse
from agm.agl.runtime.agents import AgentFn
from agm.agl.runtime.externs import ExternRegistry
from agm.cli_support.args import ExecArgs
from agm.packages.manifest import PackageManifest
from agm.packages.model import PackageInfo
from tests._agl_helpers import REPO_STDLIB_ROOT, agl_roots, run_inline_command, write_file_program
from tests._http_helpers import install as install_fake_http
from tests._process_helpers import FakeShell
from tests.agl.ir_harness import write_companion_file, write_module_file

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
    trace_file: str | None = None,
    no_trace: bool = False,
    argument_tokens: list[str] | None = None,
) -> ExecArgs:
    return ExecArgs(
        file=str(agl_file),
        argument_tokens=argument_tokens or [],
        strict_json=None,
        no_trace=no_trace,
        trace_file=trace_file,
    )


# ---------------------------------------------------------------------------
# 1. Trace file created at a custom --trace-file path
# ---------------------------------------------------------------------------


def _run_inline(runtime: PipelineDriver, source: str, **kwargs: object) -> RunResult:
    """Run this module's command-style source through the inline entry transform."""
    return run_inline_command(runtime, source, **kwargs)


class TestTraceFileCreated:
    def test_trace_file_created_at_custom_path(self, tmp_path: Path) -> None:
        """A custom --trace-file path receives JSONL trace output after a run."""
        trace_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        result = _run_inline(rt, 'let x = 1\nprint "hello"', trace_file=trace_path)
        assert result.ok
        assert trace_path.exists(), "trace file must be created when trace_file is given"

    def test_trace_file_has_jsonl_content(self, tmp_path: Path) -> None:
        """Each line of the trace file is a valid JSON object."""
        trace_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        _run_inline(rt, 'let x = 1\nprint "hello"', trace_file=trace_path)
        records = _load_jsonl(trace_path)
        assert len(records) >= 1
        for rec in records:
            assert isinstance(rec, dict)

    def test_trace_file_not_created_when_no_trace(self, tmp_path: Path) -> None:
        """When trace_file is None (no-trace semantics), no trace file is written."""
        rt = PipelineDriver()
        result = _run_inline(rt, 'let x = 1\nprint "hello"', trace_file=None)
        assert result.ok
        # No trace file: any file created would be under .agent-files/ which
        # we cannot check here, but RunResult.trace_path should be None.
        assert result.trace_path is None

    def test_run_result_exposes_trace_path(self, tmp_path: Path) -> None:
        """RunResult.trace_path is the Path of the written JSONL file."""
        trace_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        result = _run_inline(rt, "let x = 1\nx", trace_file=trace_path)
        assert result.ok
        assert result.trace_path == trace_path

    def test_trace_directory_creation_failure_is_best_effort(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def fail_mkdir(*args: object, **kwargs: object) -> None:
            raise OSError("read-only filesystem")

        monkeypatch.setattr("agm.core.fs.mkdir", fail_mkdir)
        result = _run_inline(
            PipelineDriver(),
            'print "still runs"',
            trace_file=tmp_path / "missing" / "trace.jsonl",
        )

        assert result.ok
        assert result.trace_path is None
        assert capsys.readouterr().err.count("trace logging disabled") == 1


# ---------------------------------------------------------------------------
# 2. Record kinds: print, exec command, agent call
# ---------------------------------------------------------------------------


class TestPrintRecord:
    def test_print_produces_trace_record(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        _run_inline(rt, 'print "hello world"', trace_file=trace_path)
        records = _load_jsonl(trace_path)
        kinds = [r.get("kind") for r in records]
        assert "print" in kinds

    def test_print_record_has_value(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        _run_inline(rt, 'print "hello world"', trace_file=trace_path)
        records = _load_jsonl(trace_path)
        print_recs = [r for r in records if r.get("kind") == "print"]
        assert print_recs
        # The rendered value should contain the printed text.
        assert any("hello world" in str(r.get("rendered", "")) for r in print_recs)

    def test_print_record_has_span(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        _run_inline(rt, 'print "hello"', trace_file=trace_path)
        records = _load_jsonl(trace_path)
        print_recs = [r for r in records if r.get("kind") == "print"]
        assert print_recs
        rec = print_recs[0]
        assert "line" in rec or "span" in rec


class TestExecCommandRecord:
    def _exec_record(
        self, tmp_path: Path, source: str, shell: FakeShell | None = None
    ) -> dict[str, object]:
        """Run *source* and return its single ``exec_command`` trace record."""
        trace_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        with unittest.mock.patch(
            "agm.core.process.run_capture_result", side_effect=shell or FakeShell(stdout="captured")
        ):
            _run_inline(rt, source, trace_file=trace_path)
        records = [r for r in _load_jsonl(trace_path) if r.get("kind") == "exec_command"]
        assert len(records) == 1
        return records[0]

    def test_a_successful_exec_is_recorded_in_full(self, tmp_path: Path) -> None:
        rec = self._exec_record(tmp_path, 'let x: text = exec "echo hello"\nx')

        assert "echo hello" in cast(str, rec["command"])
        assert rec["exit_code"] == 0
        assert "captured" in cast(str, rec["stdout"])
        assert rec["timed_out"] is False
        duration = rec["duration"]
        assert isinstance(duration, float)
        assert duration >= 0

    def test_a_failing_exec_records_its_exit_code(self, tmp_path: Path) -> None:
        shell = FakeShell(responses=[{"command": "false", "returncode": 3, "stderr": "nope"}])
        rec = self._exec_record(tmp_path, 'let r = exec "false"\nr.exit-code', shell)

        assert rec["exit_code"] == 3
        assert "nope" in cast(str, rec["stderr"])
        assert rec["timed_out"] is False

    def test_a_timed_out_exec_is_recorded_as_timed_out(self, tmp_path: Path) -> None:
        shell = FakeShell(
            responses=[
                {"command": "sleep 99", "returncode": 7, "stdout": "partial", "timed_out": True}
            ]
        )
        rec = self._exec_record(tmp_path, 'exec "sleep 99"', shell)

        assert rec["timed_out"] is True
        assert rec["exit_code"] == 7
        assert "partial" in cast(str, rec["stdout"])

    def test_a_timed_out_exec_without_an_exit_code_records_minus_one(self, tmp_path: Path) -> None:
        shell = FakeShell(
            responses=[{"command": "sleep 99", "returncode": None, "timed_out": True}]
        )
        rec = self._exec_record(tmp_path, 'exec "sleep 99"', shell)

        assert rec["timed_out"] is True
        assert rec["exit_code"] == -1

    def test_a_shell_that_cannot_spawn_is_recorded_as_a_failure(self, tmp_path: Path) -> None:
        shell = FakeShell(
            responses=[{"command": "whatever", "returncode": None, "spawn_error": "no shell"}]
        )
        rec = self._exec_record(tmp_path, 'exec "whatever"', shell)

        assert rec["exit_code"] == -1
        assert "no shell" in cast(str, rec["stderr"])
        assert rec["timed_out"] is False


class TestAgentCallRecord:
    def test_agent_call_produces_attempt_record(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"
        rt = _agent_runtime(_agent_returning("good"))
        _run_inline(
            rt,
            'let reviewer = AgentCommand("reviewer")\nlet x: text = reviewer.ask("check this")\nx',
            trace_file=trace_path,
        )
        records = _load_jsonl(trace_path)
        kinds = [r.get("kind") for r in records]
        assert "agent_request" in kinds
        assert "agent_response" in kinds

    def test_agent_call_record_has_rendered_agent_value(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"
        rt = _agent_runtime(_agent_returning("ok"))
        _run_inline(
            rt,
            'let critic = AgentCommand("critic")\nlet x: text = critic.ask("review")\nx',
            trace_file=trace_path,
        )
        records = _load_jsonl(trace_path)
        call_recs = [r for r in records if r.get("kind") == "agent_request"]
        assert call_recs
        assert call_recs[0]["agent"] == {
            "variant": "AgentCommand",
            "payload": {"command": "critic"},
        }

    def test_agent_request_preserves_agent_variant_payload(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"
        run_inline_command(
            _agent_runtime(_agent_returning("ok")),
            'let a = AgentClaude("sonnet", "high")\nlet x: text = a.ask("review")\nx',
            trace_file=trace_path,
        )
        request = next(
            record for record in _load_jsonl(trace_path) if record["kind"] == "agent_request"
        )
        assert request["agent"] == {
            "variant": "AgentClaude",
            "payload": {"model": "sonnet", "thinking": "high"},
        }

    def test_agent_call_record_has_attempt_number(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"
        rt = _agent_runtime(_agent_returning("result"))
        _run_inline(
            rt,
            'let impl = AgentCommand("impl")\nlet x: text = impl.ask("do work")\nx',
            trace_file=trace_path,
        )
        records = _load_jsonl(trace_path)
        call_recs = [r for r in records if r.get("kind") == "agent_request"]
        assert call_recs
        assert isinstance(call_recs[0].get("attempt"), int)
        assert call_recs[0].get("max_attempts") == 1

    def test_unit_ask_logs_a_request_and_response(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"
        result = run_inline_command(
            _agent_runtime(_agent_returning("ignored")),
            'let a = AgentCommand("a")\nlet value: unit = ask "do it"\nvalue',
            trace_file=trace_path,
        )
        assert result.ok
        records = _load_jsonl(trace_path)
        kinds = [record["kind"] for record in records]
        assert kinds.count("agent_request") == kinds.count("agent_response") == 1
        assert "parse_result" not in kinds

    def test_request_prompt_is_the_dispatcher_prompt_and_has_contract(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"
        received: list[str] = []

        def agent(request: AgentRequest) -> AgentResponse:
            received.append(request.prompt)
            return AgentResponse(content="42")

        result = run_inline_command(
            _agent_runtime(agent, strict_json=True),
            'let a = AgentCommand("a")\nlet value: int = a.ask("number")\nvalue',
            trace_file=trace_path,
        )
        assert result.ok
        request = next(
            record for record in _load_jsonl(trace_path) if record["kind"] == "agent_request"
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
        trace_path = tmp_path / "trace.jsonl"
        call_count = 0

        def agent(request: AgentRequest) -> AgentResponse:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return AgentResponse(content="not json")  # will fail to parse
            return AgentResponse(content="42")

        rt = _agent_runtime(agent, strict_json=True)
        _run_inline(
            rt,
            'let impl = AgentCommand("impl")\n'
            'let x: int = impl.ask("get int", on-parse-error = Retry(n = 2))\nx',
            trace_file=trace_path,
        )
        records = _load_jsonl(trace_path)
        call_recs = [r for r in records if r.get("kind") == "agent_request"]
        assert len(call_recs) == 3
        assert len([r for r in records if r.get("kind") == "agent_response"]) == 3
        assert "Validation errors:" in str(call_recs[1]["prompt"])
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
        trace_path = tmp_path / "trace.jsonl"
        call_count = 0

        def agent(request: AgentRequest) -> AgentResponse:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return AgentResponse(content="not json")
            return AgentResponse(content="42")

        rt = _agent_runtime(agent, strict_json=True)
        _run_inline(
            rt,
            'let impl = AgentCommand("impl")\n'
            'let x: int = impl.ask("get int", on-parse-error = Retry(n = 2))\nx',
            trace_file=trace_path,
        )
        records = _load_jsonl(trace_path)
        call_recs = [r for r in records if r.get("kind") == "agent_request"]
        attempts = [r.get("attempt") for r in call_recs]
        assert attempts == [0, 1, 2]

    def test_parse_result_record_emitted_for_each_attempt(self, tmp_path: Path) -> None:
        """A parse_result record follows each agent response."""
        trace_path = tmp_path / "trace.jsonl"

        def agent(request: AgentRequest) -> AgentResponse:
            return AgentResponse(content="not json at all")

        rt = _agent_runtime(agent, strict_json=True)
        src = (
            'let impl = AgentCommand("impl")\n'
            'let x: int = impl.ask("get int", on-parse-error = Retry(n = 1))\nx'
        )
        try:
            _run_inline(rt, src, trace_file=trace_path)
        except SystemExit:
            pass

        records = _load_jsonl(trace_path)
        kinds = [r.get("kind") for r in records]
        assert "parse_result" in kinds

    def test_transport_failure_has_failed_agent_response(self, tmp_path: Path) -> None:
        from agm.agl.runtime.request import AgentCallHostError

        trace_path = tmp_path / "trace.jsonl"

        def agent(_request: AgentRequest) -> AgentResponse:
            raise AgentCallHostError(
                cause="timeout", exit_code=9, stderr_tail="too slow", elapsed=1.5
            )

        result = run_inline_command(
            _agent_runtime(agent),
            'let a = AgentCommand("a")\nlet value: text = a.ask("work")\nvalue',
            trace_file=trace_path,
        )
        assert not result.ok
        response = next(
            record for record in _load_jsonl(trace_path) if record["kind"] == "agent_response"
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
        trace_path = tmp_path / "trace.jsonl"

        def agent(request: AgentRequest) -> AgentResponse:
            return AgentResponse(content="not json")

        rt = _agent_runtime(agent, strict_json=True)
        result = _run_inline(
            rt,
            'let impl = AgentCommand("impl")\nlet x: int = impl.ask("get int")\nx',
            trace_file=trace_path,
        )
        assert not result.ok
        assert result.error is not None

        records = _load_jsonl(trace_path)
        exc_recs = [r for r in records if r.get("kind") == "exception"]
        assert exc_recs

    def test_exception_record_has_type_name(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"

        def agent(request: AgentRequest) -> AgentResponse:
            return AgentResponse(content="not json")

        rt = _agent_runtime(agent, strict_json=True)
        result = _run_inline(
            rt,
            'let impl = AgentCommand("impl")\nlet x: int = impl.ask("get int")\nx',
            trace_file=trace_path,
        )
        assert not result.ok

        records = _load_jsonl(trace_path)
        exc_recs = [r for r in records if r.get("kind") == "exception"]
        assert exc_recs[0].get("type_name") == "AgentParseError"

    def test_exception_record_and_value_have_no_trace_id(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"
        result = run_inline_command(
            _agent_runtime(_agent_returning("not json"), strict_json=True),
            'let impl = AgentCommand("impl")\nlet x: int = impl.ask("get int")\nx',
            trace_file=trace_path,
        )
        assert result.error is not None
        exc_recs = [r for r in _load_jsonl(trace_path) if r.get("kind") == "exception"]
        assert exc_recs and "trace_id" not in exc_recs[0]
        assert "trace_id" not in result.error.fields

    def test_caught_exception_does_not_produce_exception_record(self, tmp_path: Path) -> None:
        """An exception caught by try/catch is NOT written as an 'exception' record
        (it was handled in-language and did not escape the program)."""
        trace_path = tmp_path / "trace.jsonl"

        def agent(request: AgentRequest) -> AgentResponse:
            return AgentResponse(content="not json")

        rt = _agent_runtime(agent, strict_json=True)
        # The AgentParseError is caught and the result is a fallback string.
        result = _run_inline(
            rt,
            'let impl = AgentCommand("impl")\n'
            "try\n"
            '  let x: int = impl.ask("get int")\n'
            "  x\n"
            "catch AgentParseError as e =>\n"
            "  0\n",
            trace_file=trace_path,
        )
        assert result.ok

        records = _load_jsonl(trace_path)
        exc_recs = [r for r in records if r.get("kind") == "exception"]
        assert not exc_recs, "caught exception must not produce an exception record"


# ---------------------------------------------------------------------------
# 4b. Built-in runtime exceptions carry no base trace identifier
# ---------------------------------------------------------------------------


class TestBuiltinExceptionFields:
    """Built-in runtime exceptions expose their declared fields only."""

    def test_arithmetic_error_trace_id_non_empty_with_logging(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        # Uncaught division by zero → ArithmeticError escapes the program.
        result = _run_inline(rt, "let x = 1 / 0\nx", trace_file=trace_path)
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "ArithmeticError"

        assert "trace_id" not in result.error.fields
        exc_recs = [r for r in _load_jsonl(trace_path) if r.get("kind") == "exception"]
        assert exc_recs and "trace_id" not in exc_recs[0]

    def test_arithmetic_error_trace_id_non_empty_without_logging(self) -> None:
        rt = PipelineDriver()
        result = _run_inline(rt, "let x = 1 / 0\nx", trace_file=None)
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "ArithmeticError"
        assert "trace_id" not in result.error.fields

    def test_match_error_trace_id_non_empty_without_logging(self) -> None:
        rt = PipelineDriver()
        # Explicit source raising retains the ordinary MatchError runtime contract.
        result = _run_inline(
            rt,
            "case 5 of\n"
            "  | 0 => ()\n"
            "  | _ =>\n"
            '    raise MatchError(message = "no match", '
            'scrutinee-type = "int", scrutinee = 5)\n',
            trace_file=None,
        )
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "MatchError"
        assert "trace_id" not in result.error.fields

    def test_max_iterations_trace_id_linked_with_logging(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        # A do-loop whose condition never becomes true exhausts its limit.
        result = _run_inline(
            rt,
            "var x = 0\ndo[2]\n  x := x\nuntil false\n",
            trace_file=trace_path,
        )
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "MaxIterationsExceeded"

        assert "trace_id" not in result.error.fields
        exc_recs = [r for r in _load_jsonl(trace_path) if r.get("kind") == "exception"]
        assert exc_recs and "trace_id" not in exc_recs[0]


# ---------------------------------------------------------------------------
# 5. run_start / run_end records
# ---------------------------------------------------------------------------


class TestRunBoundaryRecords:
    def test_run_start_record_present(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        _run_inline(rt, "let x = 1\nx", trace_file=trace_path)
        records = _load_jsonl(trace_path)
        kinds = [r.get("kind") for r in records]
        assert "run_start" in kinds

    def test_run_end_record_present(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        _run_inline(rt, "let x = 1\nx", trace_file=trace_path)
        records = _load_jsonl(trace_path)
        kinds = [r.get("kind") for r in records]
        assert "run_end" in kinds

    def test_run_start_before_run_end(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        _run_inline(rt, "let x = 1\nx", trace_file=trace_path)
        records = _load_jsonl(trace_path)
        kinds = [r.get("kind") for r in records]
        start_idx = kinds.index("run_start")
        end_idx = kinds.index("run_end")
        assert start_idx < end_idx

    def test_all_records_share_run_id(self, tmp_path: Path) -> None:
        """Every record in a trace file carries the same run_id."""
        trace_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        _run_inline(rt, 'var x = 1\nx := 2\nprint "done"', trace_file=trace_path)
        records = _load_jsonl(trace_path)
        assert len(records) >= 3
        run_ids = {r.get("run_id") for r in records}
        assert len(run_ids) == 1
        (run_id,) = run_ids
        assert isinstance(run_id, str) and run_id

    def test_every_record_has_an_ordered_offset_aware_timestamp(self, tmp_path: Path) -> None:
        from datetime import datetime

        trace_path = tmp_path / "trace.jsonl"
        run_inline_command(
            _agent_runtime(_agent_returning("agent output")),
            'let a = AgentCommand("a")\n'
            'let x: text = a.ask("prompt")\n'
            'let y: text = exec "printf shell"\n'
            "print x\ny",
            trace_file=trace_path,
        )
        records = _load_jsonl(trace_path)
        timestamps = [datetime.fromisoformat(str(record["ts"])) for record in records]
        assert all(timestamp.utcoffset() is not None for timestamp in timestamps)
        assert timestamps == sorted(timestamps)
        assert all("trace_id" not in record for record in records)


# ---------------------------------------------------------------------------
# 6. No-trace semantics
# ---------------------------------------------------------------------------


class TestNoLog:
    def test_no_trace_writes_nothing(self, tmp_path: Path) -> None:
        """With trace_file=None the trace store is a no-op and no files are created."""
        rt = PipelineDriver()
        result = _run_inline(rt, 'let x = 1\nprint "silent"', trace_file=None)
        assert result.ok
        # No JSONL files created anywhere in tmp_path.
        jsonl_files = list(tmp_path.rglob("*.jsonl"))
        assert not jsonl_files

    def test_no_trace_result_trace_path_is_none(self, tmp_path: Path) -> None:
        rt = PipelineDriver()
        result = _run_inline(rt, "let x = 1\nx", trace_file=None)
        assert result.trace_path is None

    def test_no_trace_with_agent_call_writes_nothing(self, tmp_path: Path) -> None:
        rt = _agent_runtime(_agent_returning("hello"))
        result = _run_inline(
            rt, 'let a = AgentCommand("a")\nlet x: text = a.ask("hi")\nx', trace_file=None
        )
        assert result.ok
        jsonl_files = list(tmp_path.rglob("*.jsonl"))
        assert not jsonl_files

    def test_no_trace_with_decimal_assignment_still_works(self, tmp_path: Path) -> None:
        """A no-trace run that assigns to a decimal binding still succeeds and
        writes nothing."""
        rt = PipelineDriver()
        result = _run_inline(
            rt,
            "var x: decimal = 0.1\nx := x + 0.2",
            trace_file=None,
        )
        assert result.ok
        jsonl_files = list(tmp_path.rglob("*.jsonl"))
        assert not jsonl_files


# ---------------------------------------------------------------------------
# 7. --no-trace flag via exec command writes nothing
# ---------------------------------------------------------------------------


class TestExecNoLog:
    def test_exec_no_trace_flag_writes_nothing(self, tmp_path: Path) -> None:
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, 'print "hello"\n')
        args = _exec_args(agl_file, no_trace=True)
        exec_command.run(args)
        # No JSONL files created under tmp_path or any default path.
        jsonl_files = list(tmp_path.rglob("*.jsonl"))
        assert not jsonl_files

    def test_exec_trace_file_flag_creates_file(self, tmp_path: Path) -> None:
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, 'let x = 1\nprint "hi"\n')
        trace_path = tmp_path / "out.jsonl"
        args = _exec_args(agl_file, trace_file=str(trace_path))
        exec_command.run(args)
        assert trace_path.exists()
        records = _load_jsonl(trace_path)
        assert len(records) >= 1


# ---------------------------------------------------------------------------
# 8. Dry-run must NOT write a trace
# ---------------------------------------------------------------------------


class TestDryRunNoTrace:
    def test_dry_run_does_not_write_trace(self, tmp_path: Path) -> None:
        """check_only=True (--dry-run) must produce no trace output."""
        trace_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        result = _run_inline(rt, "let x = 1\nx", trace_file=trace_path, check_only=True)
        assert result.ok
        # No trace file created for dry-run.
        assert not trace_path.exists()

    def test_dry_run_trace_path_is_none(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        result = _run_inline(rt, "let x = 1\nx", trace_file=trace_path, check_only=True)
        assert result.trace_path is None


# ---------------------------------------------------------------------------
# 9. Source spans in records
# ---------------------------------------------------------------------------


class TestSourceSpans:
    def test_exec_record_has_source_span(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        _run_inline(rt, 'let x: text = exec "echo hi"\nx', trace_file=trace_path)
        records = _load_jsonl(trace_path)
        exec_recs = [r for r in records if r.get("kind") == "exec_command"]
        assert exec_recs
        rec = exec_recs[0]
        # Must have either top-level "line"/"col" or a "span" sub-object.
        has_span = "line" in rec or ("span" in rec and isinstance(rec["span"], dict))
        assert has_span

    def test_agent_record_has_source_span(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"
        rt = _agent_runtime(_agent_returning("hello"))
        _run_inline(
            rt,
            'let impl = AgentCommand("impl")\nlet x: text = impl.ask("do work")\nx',
            trace_file=trace_path,
        )
        records = _load_jsonl(trace_path)
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

    def test_trace_store_run_id_in_records(self, tmp_path: Path) -> None:
        from agm.agl.runtime.trace import TraceStore

        p = tmp_path / "t.jsonl"
        ts = TraceStore(path=p)
        ts.run_start()
        records = _load_jsonl(p)
        run_id = records[0]["run_id"]
        assert isinstance(run_id, str) and run_id

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

    def test_companion_record_omits_site_when_none(self, tmp_path: Path) -> None:
        """``site=None`` (no attributable call site at all) omits the key."""
        import json as _json

        from agm.agl.runtime.trace import TraceStore

        p = tmp_path / "t.jsonl"
        ts = TraceStore(path=p)
        ts.companion_record("lib/logger", "probe", {}, span=None, site=None)

        rec = _json.loads(p.read_text(encoding="utf-8").strip())
        assert rec["origin"] == "lib/logger"
        assert "site" not in rec
        assert "line" not in rec


# ---------------------------------------------------------------------------
# Unparseable output synthesizes a validation error for retry feedback
# ---------------------------------------------------------------------------


class TestUnparseableFeedback:
    """when agent output is totally unparseable (no JSON at all), the next
    retry attempt must carry the failure reason as a ValidationError, and the
    parse_result trace record must have a non-empty error_summary."""

    def test_retry_request_carries_reason_when_totally_unparseable(self, tmp_path: Path) -> None:
        """Second attempt's validation_errors is non-empty with the parse reason."""
        trace_path = tmp_path / "trace.jsonl"
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
        result = _run_inline(
            rt,
            'let impl = AgentCommand("impl")\n'
            'let x: int = impl.ask("get int", on-parse-error = Retry(n = 1))\nx',
            trace_file=trace_path,
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
        trace_path = tmp_path / "trace.jsonl"

        def agent(request: AgentRequest) -> AgentResponse:
            return AgentResponse(content="totally not json #@!")

        rt = _agent_runtime(agent, strict_json=True)
        _run_inline(
            rt,
            'let impl = AgentCommand("impl")\n'
            'let x: int = impl.ask("get int", on-parse-error = Retry(n = 1))\nx',
            trace_file=trace_path,
        )
        records = _load_jsonl(trace_path)
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
            result = _run_inline(
                rt,
                'let impl = AgentCommand("impl")\n'
                'let x: int = impl.ask("q", on-parse-error = Abort())\n'
                "x",
            )
        # The program raises AgentParseError; run returns ok=False.
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "AgentParseError"
        # With empty errors the validation_errors list is empty.
        val_errs = result.error.fields.get("validation-errors")
        assert val_errs == []


# ---------------------------------------------------------------------------
# 12. prepare_trace_log truncates an existing file (clean-file guarantee)
# ---------------------------------------------------------------------------


class TestPrepareTraceLogTruncates:
    """prepare_trace_log must start each run from a clean (empty) file.

    For auto-generated paths the pid-unique component already guarantees a
    fresh file.  For an explicit --trace-file path a new run must TRUNCATE any
    pre-existing content so the "first traced entry starts from a clean file"
    contract in the docstring holds.
    """

    def test_prepare_trace_log_truncates_existing_content(self, tmp_path: Path) -> None:
        """Pre-existing content at an explicit trace path is erased by prepare_trace_log."""
        from agm.core.log import prepare_trace_log

        trace_path = tmp_path / "trace.jsonl"
        # Pre-create the file with stale content from a previous run.
        trace_path.write_text('{"kind": "run_start", "run_id": "old"}\n', encoding="utf-8")
        assert trace_path.read_text(encoding="utf-8").strip(), (
            "pre-condition: file must be non-empty"
        )

        prepare_trace_log(command_name="exec", enabled=True, trace_file=str(trace_path))

        content = trace_path.read_text(encoding="utf-8")
        assert content == "", (
            "prepare_trace_log must truncate the file so each run starts from a clean slate"
        )

    def test_prepare_trace_log_subsequent_record_is_only_content(self, tmp_path: Path) -> None:
        """After truncation, only records written in the current run appear in the file."""
        from agm.core.log import append_jsonl, prepare_trace_log

        trace_path = tmp_path / "trace.jsonl"
        # Simulate a previous run by pre-populating the file.
        trace_path.write_text('{"kind": "run_start", "run_id": "old"}\n', encoding="utf-8")

        prepare_trace_log(command_name="exec", enabled=True, trace_file=str(trace_path))
        # Append a single record as the new run would.
        append_jsonl(trace_path, {"kind": "run_start", "run_id": "new"})

        import json as _json

        lines = [ln for ln in trace_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 1, f"Only the new record must be present; got {len(lines)} lines"
        rec = _json.loads(lines[0])
        assert rec.get("run_id") == "new"


# ---------------------------------------------------------------------------
# 13. Companion trace hook (``runtime.trace(kind, payload)``)
# ---------------------------------------------------------------------------


def _write_extern_entry(tmp_path: Path, source: str, companion_source: str) -> Path:
    """Write an inline extern-declaring entry and its Python companion as real sibling files."""
    entry_path = tmp_path / "entry.agl"
    entry_path.write_text(source)
    (tmp_path / "entry.py").write_text(companion_source)
    return entry_path


class _EmitCompanion(Protocol):
    """A loaded companion's typed surface for the detached-call test below."""

    def emit(self) -> None: ...


class TestCompanionTraceHook:
    """``runtime.trace(kind, payload)`` lets a companion emit its own trace records."""

    def test_companion_trace_produces_a_record_with_origin_and_span(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"
        source = "extern def emit() -> unit\nemit()\n()\n"
        companion = (
            "from agl import runtime\n\ndef emit():\n    runtime.trace('probe', {'value': 42})\n"
        )
        entry_path = _write_extern_entry(tmp_path, source, companion)
        result = run_inline_command(
            PipelineDriver(), source, entry_path=entry_path, trace_file=trace_path
        )
        assert result.ok

        records = _load_jsonl(trace_path)
        probe_recs = [r for r in records if r.get("kind") == "probe"]
        assert probe_recs
        rec = probe_recs[0]
        assert rec.get("origin") == "<entry>"
        assert rec.get("site") == "<entry>"
        assert rec.get("value") == 42
        # `emit()` is the call on line 2 of `source`.
        assert rec.get("line") == 2
        assert rec.get("col") == 1

    def test_companion_trace_origin_is_the_calling_module_path(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"
        root = tmp_path / "root"
        write_module_file(root, "lib/logger", "extern def emit() -> unit")
        write_companion_file(
            root,
            "lib/logger",
            "from agl import runtime\n\ndef emit():\n    runtime.trace('probe', {'value': 1})\n",
        )
        result = run_inline_command(
            PipelineDriver(),
            "import lib/logger\nlib/logger::emit()",
            roots=agl_roots(root),
            default_stdlib=False,
            trace_file=trace_path,
        )
        assert result.ok

        records = _load_jsonl(trace_path)
        probe_recs = [r for r in records if r.get("kind") == "probe"]
        assert probe_recs
        assert probe_recs[0]["origin"] == "lib/logger"
        # `origin` is the extern's own declaring module; `site` is the
        # caller's, which here differs from it.
        assert probe_recs[0]["site"] == "<entry>"

    def test_companion_trace_writes_nothing_when_logging_is_off(self, tmp_path: Path) -> None:
        source = "extern def emit() -> unit\nemit()\n()\n"
        companion = (
            "from agl import runtime\n\ndef emit():\n    runtime.trace('probe', {'value': 1})\n"
        )
        entry_path = _write_extern_entry(tmp_path, source, companion)
        result = run_inline_command(
            PipelineDriver(), source, entry_path=entry_path, trace_file=None
        )
        assert result.ok
        assert not list(tmp_path.rglob("*.jsonl"))

    def test_companion_trace_cyclic_payload_degrades_instead_of_raising(
        self, tmp_path: Path
    ) -> None:
        trace_path = tmp_path / "trace.jsonl"
        source = "extern def emit() -> unit\nemit()\n()\n"
        companion = (
            "from agl import runtime\n\n"
            "def emit():\n"
            "    payload = {'a': 1}\n"
            "    payload['self'] = payload\n"
            "    runtime.trace('probe', payload)\n"
        )
        entry_path = _write_extern_entry(tmp_path, source, companion)
        result = run_inline_command(
            PipelineDriver(), source, entry_path=entry_path, trace_file=trace_path
        )
        assert result.ok

        records = _load_jsonl(trace_path)
        probe_recs = [r for r in records if r.get("kind") == "probe"]
        assert probe_recs
        assert probe_recs[0]["self"] == "<cyclic value>"

    def test_companion_trace_cyclic_list_payload_degrades_instead_of_raising(
        self, tmp_path: Path
    ) -> None:
        trace_path = tmp_path / "trace.jsonl"
        source = "extern def emit() -> unit\nemit()\n()\n"
        companion = (
            "from agl import runtime\n\n"
            "def emit():\n"
            "    items = [1, 2]\n"
            "    items.append(items)\n"
            "    runtime.trace('probe', {'items': items})\n"
        )
        entry_path = _write_extern_entry(tmp_path, source, companion)
        result = run_inline_command(
            PipelineDriver(), source, entry_path=entry_path, trace_file=trace_path
        )
        assert result.ok

        records = _load_jsonl(trace_path)
        probe_recs = [r for r in records if r.get("kind") == "probe"]
        assert probe_recs
        assert probe_recs[0]["items"] == [1, 2, "<cyclic value>"]

    def test_companion_trace_non_json_value_degrades_instead_of_raising(
        self, tmp_path: Path
    ) -> None:
        trace_path = tmp_path / "trace.jsonl"
        source = "extern def emit() -> unit\nemit()\n()\n"
        companion = (
            "from agl import runtime\n\n"
            "def emit():\n"
            "    runtime.trace('probe', {'bad': object()})\n"
        )
        entry_path = _write_extern_entry(tmp_path, source, companion)
        result = run_inline_command(
            PipelineDriver(), source, entry_path=entry_path, trace_file=trace_path
        )
        assert result.ok

        records = _load_jsonl(trace_path)
        probe_recs = [r for r in records if r.get("kind") == "probe"]
        assert probe_recs
        assert probe_recs[0]["bad"] == "<object has no JSON representation>"

    def test_companion_trace_invalid_scalars_degrade_instead_of_raising(
        self, tmp_path: Path
    ) -> None:
        trace_path = tmp_path / "trace.jsonl"
        source = "extern def emit() -> unit\nemit()\n()\n"
        companion = (
            "from agl import runtime\n"
            "import math\n\n"
            "def emit():\n"
            "    runtime.trace('probe', {'text': chr(0xD800), 'number': math.nan})\n"
        )
        entry_path = _write_extern_entry(tmp_path, source, companion)
        result = run_inline_command(
            PipelineDriver(), source, entry_path=entry_path, trace_file=trace_path
        )

        assert result.ok
        records = _load_jsonl(trace_path)
        probe_recs = [record for record in records if record.get("kind") == "probe"]
        assert probe_recs[0]["text"] == "<str has no JSON representation>"
        assert probe_recs[0]["number"] == "<float has no JSON representation>"

    def test_companion_trace_direct_call_outside_evaluation_is_a_silent_noop(
        self, tmp_path: Path
    ) -> None:
        """A companion calling ``runtime.trace`` with no active interpreter is a no-op."""
        companion_path = write_companion_file(
            tmp_path,
            "entry",
            "from agl import runtime\n\ndef emit():\n    runtime.trace('probe', {'value': 1})\n",
        )
        companion = cast(_EmitCompanion, ExternRegistry().load_companion(ENTRY_ID, companion_path))
        companion.emit()  # must not raise

    def test_companion_trace_reserved_key_raises_a_value_error(self, tmp_path: Path) -> None:
        """A payload key colliding with the envelope is a companion programmer error."""
        trace_path = tmp_path / "trace.jsonl"
        source = "extern def emit() -> unit\nemit()\n()\n"
        companion = (
            "from agl import runtime\n\ndef emit():\n    runtime.trace('probe', {'kind': 'x'})\n"
        )
        entry_path = _write_extern_entry(tmp_path, source, companion)
        result = run_inline_command(
            PipelineDriver(), source, entry_path=entry_path, trace_file=trace_path
        )
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "ExternError"

    def test_companion_trace_decimal_value_renders_as_exact_text(self, tmp_path: Path) -> None:
        """A ``Decimal`` payload value uses the DSL's own exact-number convention."""
        trace_path = tmp_path / "trace.jsonl"
        source = "extern def emit() -> unit\nemit()\n()\n"
        companion = (
            "from decimal import Decimal\n"
            "from agl import runtime\n\n"
            "def emit():\n"
            "    runtime.trace('probe', {'amount': Decimal('1.50')})\n"
        )
        entry_path = _write_extern_entry(tmp_path, source, companion)
        result = run_inline_command(
            PipelineDriver(), source, entry_path=entry_path, trace_file=trace_path
        )
        assert result.ok

        records = _load_jsonl(trace_path)
        probe_recs = [r for r in records if r.get("kind") == "probe"]
        assert probe_recs
        assert probe_recs[0]["amount"] == "1.50"

    def test_companion_trace_agl_json_value_unwraps_to_its_raw_json(self, tmp_path: Path) -> None:
        """An ``AglJson`` payload value (JSON already crossed the boundary) unwraps."""
        trace_path = tmp_path / "trace.jsonl"
        source = "extern def emit() -> unit\nemit()\n()\n"
        companion = (
            "from agl import json, runtime\n\n"
            "def emit():\n"
            "    runtime.trace('probe', {'payload': json({'nested': [1, 2]})})\n"
        )
        entry_path = _write_extern_entry(tmp_path, source, companion)
        result = run_inline_command(
            PipelineDriver(), source, entry_path=entry_path, trace_file=trace_path
        )
        assert result.ok

        records = _load_jsonl(trace_path)
        probe_recs = [r for r in records if r.get("kind") == "probe"]
        assert probe_recs
        assert probe_recs[0]["payload"] == {"nested": [1, 2]}

    def test_companion_trace_span_through_a_function_value_call(self, tmp_path: Path) -> None:
        """The span is the first-class function value's call site, not its declaration."""
        trace_path = tmp_path / "trace.jsonl"
        source = "extern def emit() -> unit\nlet f = emit\nf()\n()\n"
        companion = "from agl import runtime\n\ndef emit():\n    runtime.trace('probe', {})\n"
        entry_path = _write_extern_entry(tmp_path, source, companion)
        result = run_inline_command(
            PipelineDriver(), source, entry_path=entry_path, trace_file=trace_path
        )
        assert result.ok

        records = _load_jsonl(trace_path)
        probe_recs = [r for r in records if r.get("kind") == "probe"]
        assert probe_recs
        # `f()` is the call on line 3 of `source`; `emit`'s own declaration
        # (line 1) must not be reported.
        assert probe_recs[0]["line"] == 3
        assert probe_recs[0]["col"] == 1

    def test_companion_trace_span_is_restored_after_a_crossed_extern_callback(
        self, tmp_path: Path
    ) -> None:
        """A crossed callback with no AgL call site of its own inherits the outer span,
        and the outer call's own span is restored once the callback returns."""
        trace_path = tmp_path / "trace.jsonl"
        source = (
            "extern def relay(f: () -> unit) -> unit\n"
            "extern def probe() -> unit\n"
            "relay(probe)\n"
            "()\n"
        )
        companion = (
            "from agl import runtime\n\n"
            "def relay(f):\n"
            "    runtime.trace('a_before', {})\n"
            "    f()\n"
            "    runtime.trace('a_after', {})\n\n"
            "def probe():\n"
            "    runtime.trace('b_probe', {})\n"
        )
        entry_path = _write_extern_entry(tmp_path, source, companion)
        result = run_inline_command(
            PipelineDriver(), source, entry_path=entry_path, trace_file=trace_path
        )
        assert result.ok

        records = _load_jsonl(trace_path)
        kinds = {"a_before", "b_probe", "a_after"}
        by_kind = {r["kind"]: r for r in records if r["kind"] in kinds}
        assert set(by_kind) == kinds
        # `relay(probe)` (line 3) is the only real AgL call site here: the
        # crossed callback (`probe`, invoked directly by `relay`'s own Python
        # body) has none of its own, so it inherits that outer span, and the
        # outer call's span is restored once the crossed callback returns.
        for kind in kinds:
            assert by_kind[kind]["line"] == 3
            assert by_kind[kind]["col"] == 1
            assert by_kind[kind]["site"] == "<entry>"

    def test_companion_trace_span_through_a_package_functions_default_argument(
        self, tmp_path: Path
    ) -> None:
        """A default-argument expression evaluates in its own function's
        module and call-site context, not the caller's: an extern call inside
        a package function's default is attributed to the caller's own call
        to that function, not misattributed as an internal library call."""
        trace_path = tmp_path / "trace.jsonl"
        root = tmp_path / "root"
        write_module_file(
            root,
            "lib/logger",
            "extern def emit() -> int\ndef wrap(x: int = emit()) -> int = x\n",
        )
        write_companion_file(
            root,
            "lib/logger",
            "from agl import runtime\n\n"
            "def emit():\n    runtime.trace('probe', {})\n    return 0\n",
        )
        source = "import lib/logger\nlet _ = lib/logger::wrap()\n()\n"
        result = run_inline_command(
            PipelineDriver(),
            source,
            roots=agl_roots(root),
            default_stdlib=False,
            trace_file=trace_path,
        )
        assert result.ok

        records = _load_jsonl(trace_path)
        probe_recs = [r for r in records if r.get("kind") == "probe"]
        assert probe_recs
        # `lib/logger::wrap()` (line 2) is the entry's own call; `wrap`'s
        # default-argument expression, though it runs inside `lib/logger`,
        # must not be attributed to some other internal library site.
        assert probe_recs[0]["line"] == 2
        assert probe_recs[0]["col"] == source.splitlines()[1].index("lib/logger::wrap()") + 1
        assert probe_recs[0]["origin"] == "lib/logger"
        assert probe_recs[0]["site"] == "<entry>"

    def test_companion_trace_span_skips_an_internal_call_within_the_same_package(
        self, tmp_path: Path
    ) -> None:
        """A package's own AgL wrapper calling its extern is an implementation
        detail: the span is the caller's call to the wrapper, not the
        wrapper's own internal call to the extern."""
        trace_path = tmp_path / "trace.jsonl"
        root = tmp_path / "root"
        write_module_file(
            root, "lib/logger", "extern def emit() -> unit\ndef wrap() -> unit = emit()\n"
        )
        write_companion_file(
            root,
            "lib/logger",
            "from agl import runtime\n\ndef emit():\n    runtime.trace('probe', {})\n",
        )
        result = run_inline_command(
            PipelineDriver(),
            "import lib/logger\nlib/logger::wrap()",
            roots=agl_roots(root),
            default_stdlib=False,
            trace_file=trace_path,
        )
        assert result.ok

        records = _load_jsonl(trace_path)
        probe_recs = [r for r in records if r.get("kind") == "probe"]
        assert probe_recs
        # `lib/logger::wrap()` (line 2 of the entry) is the call outside
        # `lib/logger`'s own package; `wrap`'s internal call to `emit()`,
        # inside `lib/logger` itself, must not be reported.
        assert probe_recs[0]["line"] == 2
        assert probe_recs[0]["col"] == 1
        assert probe_recs[0]["origin"] == "lib/logger"
        assert probe_recs[0]["site"] == "<entry>"

    def test_companion_trace_span_walks_through_several_layers_of_wrapping(
        self, tmp_path: Path
    ) -> None:
        """The nearest call site outside the package is found however many of
        the package's own functions the call passes through first."""
        trace_path = tmp_path / "trace.jsonl"
        root = tmp_path / "root"
        write_module_file(
            root,
            "lib/logger",
            "extern def emit() -> unit\n"
            "def inner() -> unit = emit()\n"
            "def outer() -> unit = inner()\n",
        )
        write_companion_file(
            root,
            "lib/logger",
            "from agl import runtime\n\ndef emit():\n    runtime.trace('probe', {})\n",
        )
        result = run_inline_command(
            PipelineDriver(),
            "import lib/logger\nlib/logger::outer()",
            roots=agl_roots(root),
            default_stdlib=False,
            trace_file=trace_path,
        )
        assert result.ok

        records = _load_jsonl(trace_path)
        probe_recs = [r for r in records if r.get("kind") == "probe"]
        assert probe_recs
        assert probe_recs[0]["line"] == 2
        assert probe_recs[0]["col"] == 1
        assert probe_recs[0]["site"] == "<entry>"

    def test_companion_trace_span_falls_back_to_the_immediate_call_when_the_whole_chain_is_internal(
        self, tmp_path: Path
    ) -> None:
        """When even the program entry itself belongs to the extern's own
        package, there is no outside call site to attribute to: the span
        falls back to the immediate internal call, same as a single-file
        entry script calling its own extern directly."""
        trace_path = tmp_path / "trace.jsonl"
        root = tmp_path / "root"
        main_path = root / "src" / "main.agl"
        main_path.parent.mkdir(parents=True)
        main_path.write_text(
            "extern def emit() -> unit\n"
            "def inner() -> unit = emit()\n"
            "def outer() -> unit = inner()\n"
            "outer()\n"
        )
        main_path.with_suffix(".py").write_text(
            "from agl import runtime\n\ndef emit():\n    runtime.trace('probe', {})\n"
        )
        package = PackageInfo(root, PackageManifest("pkg", semver.Version.parse("1.0.0")))
        roots = RootSet(
            roots=frozenset(), packages=(package,), stdlib_roots=frozenset({REPO_STDLIB_ROOT})
        )
        result = run_inline_command(
            PipelineDriver(),
            main_path.read_text(),
            roots=roots,
            entry_path=main_path,
            default_stdlib=False,
            trace_file=trace_path,
        )
        assert result.ok

        records = _load_jsonl(trace_path)
        probe_recs = [r for r in records if r.get("kind") == "probe"]
        assert probe_recs
        assert probe_recs[0]["line"] == 2
        assert probe_recs[0]["col"] == 23
        # No call site outside the extern's own mount exists here: `site`
        # falls back to that same mount, matching `origin`.
        assert probe_recs[0]["origin"] == "pkg/main"
        assert probe_recs[0]["site"] == "pkg/main"

    def test_companion_trace_span_through_a_user_lambda_passed_to_a_package_function(
        self, tmp_path: Path
    ) -> None:
        """A lambda the user passes into a package's higher-order function is
        attributed to its own call site in the user's module, not to the
        package's internal call that invokes it."""
        trace_path = tmp_path / "trace.jsonl"
        root = tmp_path / "root"
        write_module_file(
            root,
            "lib/logger",
            "extern def emit() -> unit\ndef apply(f: int -> unit) -> unit = f(1)\n",
        )
        write_companion_file(
            root,
            "lib/logger",
            "from agl import runtime\n\ndef emit():\n    runtime.trace('probe', {})\n",
        )
        source = "import lib/logger\nlib/logger::apply(fn(z: int) => lib/logger::emit())\n"
        result = run_inline_command(
            PipelineDriver(),
            source,
            roots=agl_roots(root),
            default_stdlib=False,
            trace_file=trace_path,
        )
        assert result.ok

        records = _load_jsonl(trace_path)
        probe_recs = [r for r in records if r.get("kind") == "probe"]
        assert probe_recs
        # `lib/logger::emit()` inside the lambda body (line 2) is the call the
        # user wrote; `apply`'s own internal `f(1)` call, inside `lib/logger`
        # itself, must not be reported.
        assert probe_recs[0]["line"] == 2
        assert probe_recs[0]["col"] == source.splitlines()[1].index("lib/logger::emit()") + 1
        assert probe_recs[0]["origin"] == "lib/logger"
        assert probe_recs[0]["site"] == "<entry>"

    def test_companion_trace_span_through_a_companion_invoked_crossed_closure(
        self, tmp_path: Path
    ) -> None:
        """A companion invoking a crossed AgL closure that itself calls a
        tracing extern is attributed to the closure's own call site."""
        trace_path = tmp_path / "trace.jsonl"
        root = tmp_path / "root"
        write_module_file(root, "lib/logger", "extern def relay(f: int -> unit) -> unit\n")
        write_companion_file(
            root, "lib/logger", "from agl import runtime\n\ndef relay(f):\n    f(1)\n"
        )
        entry_path = root / "entry" / "main.agl"
        entry_path.parent.mkdir(parents=True)
        source = (
            "import lib/logger\n"
            "extern def probe(z: int) -> unit\n"
            "lib/logger::relay(fn(z: int) => probe(z))\n"
        )
        entry_path.write_text(source)
        entry_path.with_suffix(".py").write_text(
            "from agl import runtime\n\ndef probe(z):\n    runtime.trace('probe', {'z': z})\n"
        )
        result = run_inline_command(
            PipelineDriver(),
            source,
            roots=agl_roots(root),
            entry_path=entry_path,
            default_stdlib=False,
            trace_file=trace_path,
        )
        assert result.ok

        records = _load_jsonl(trace_path)
        probe_recs = [r for r in records if r.get("kind") == "probe"]
        assert probe_recs
        # `probe(z)`, the extern call written inside the crossed closure
        # `lib/logger::relay` invokes (line 3), not `relay`'s own call to it.
        assert probe_recs[0]["z"] == 1
        assert probe_recs[0]["line"] == 3
        assert probe_recs[0]["col"] == source.splitlines()[2].index("probe(z)") + 1
        assert probe_recs[0]["origin"] == "<entry>"
        assert probe_recs[0]["site"] == "<entry>"


# ---------------------------------------------------------------------------
# 14. std/http trace records: span attribution and redaction end-to-end
# ---------------------------------------------------------------------------


class TestHttpTraceRecords:
    """``std/http``'s ``runtime.trace`` records, run through the real pipeline."""

    def test_http_trace_span_is_the_user_call_site_through_client_wrappers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``Client::get`` reaches the ``request`` extern through several of
        ``std/http``'s own wrapper layers (``Client::get`` -> ``Client::request``
        -> ``ClientInternals::send`` -> ``request`` -> ``RequestInternals::send``);
        the trace span must be the user's own call, not any of those internal
        calls inside ``std/http`` itself."""
        trace_path = tmp_path / "trace.jsonl"
        adapter = install_fake_http(
            monkeypatch,
            [{"status": 200, "headers": {"set-cookie": "session=topsecret"}, "body": "ok"}],
        )
        source = """import std/http
program def main() -> unit =
  let api = http::client(auth = http::Auth::Bearer("secret-token"))
  let _ = api.get("https://x/y")
  ()
"""
        result = run_inline_command(
            PipelineDriver(), source, entry_path=tmp_path / "entry.agl", trace_file=trace_path
        )
        assert result.ok, result.error
        adapter.assert_complete()

        records = _load_jsonl(trace_path)
        request_recs = [r for r in records if r.get("kind") == "http_request"]
        response_recs = [r for r in records if r.get("kind") == "http_response"]
        assert request_recs and response_recs

        # `api.get("https://x/y")` is on line 4 of the source: the user's own
        # call site, not `Client::get`'s, `request`'s, or any other wrapper's
        # internal call inside `std/http`.
        for rec in (*request_recs, *response_recs):
            assert rec["origin"] == "std/http"
            assert rec["site"] == "<entry>"
            assert rec["line"] == 4
            assert rec["col"] == 11

        assert request_recs[0]["headers"]["authorization"] == "<redacted>"
        assert response_recs[0]["headers"]["set-cookie"] == "<redacted>"

    def test_http_failure_trace_span_is_the_user_call_site(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A transport failure's ``http_failure`` record gets the same
        call-site attribution as a completed exchange."""
        trace_path = tmp_path / "trace.jsonl"
        adapter = install_fake_http(monkeypatch, [{"fail": "connection"}])
        source = """import std/http
program def main() -> unit =
  let _ = http::get("https://x/y")
  ()
"""
        result = run_inline_command(
            PipelineDriver(), source, entry_path=tmp_path / "entry.agl", trace_file=trace_path
        )
        assert not result.ok
        adapter.assert_complete()

        records = _load_jsonl(trace_path)
        failure_recs = [r for r in records if r.get("kind") == "http_failure"]
        assert failure_recs
        assert failure_recs[0]["origin"] == "std/http"
        assert failure_recs[0]["site"] == "<entry>"
        assert failure_recs[0]["line"] == 3
        assert failure_recs[0]["col"] == 11
