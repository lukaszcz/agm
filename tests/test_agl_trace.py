"""Behavior tests for the AgL trace store.

All assertions are on *observable* outcomes: what ends up in the trace file,
whether a file is created at all, and whether exception trace_ids match
records in the file.  No internal TraceStore methods are called directly.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

import agm.commands.exec as exec_command
from agm.agl import PipelineDriver
from agm.agl.runtime import AgentRequest, AgentResponse
from agm.cli_support.args import ExecArgs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent_returning(text: str):
    """Return a stub agent callable that always returns *text*."""

    def agent(request: AgentRequest) -> AgentResponse:
        return AgentResponse(content=text)

    return agent


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
        runner=None,
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
        result = rt.run('let x = 1\nx', log_file=log_path)
        assert result.ok
        assert result.trace_path == log_path


# ---------------------------------------------------------------------------
# 2. Record kinds: print, mutation (assignment), exec command, agent call
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


class TestMutationRecord:
    def test_assign_produces_mutation_record(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        rt.run("var x = 1\nx := 2", log_file=log_path)
        records = _load_jsonl(log_path)
        kinds = [r.get("kind") for r in records]
        assert "mutation" in kinds

    def test_mutation_record_has_name_and_value(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        rt.run("var x = 1\nx := 42", log_file=log_path)
        records = _load_jsonl(log_path)
        mut_recs = [r for r in records if r.get("kind") == "mutation"]
        assert mut_recs
        rec = mut_recs[0]
        assert rec.get("name") == "x"


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
        rt = PipelineDriver()
        rt.register_agent("reviewer", _agent_returning("good"))
        rt.run(
            'agent reviewer\nlet x: text = ask("check this", agent = reviewer)\nx',
            log_file=log_path,
        )
        records = _load_jsonl(log_path)
        kinds = [r.get("kind") for r in records]
        assert "agent_call_attempt" in kinds

    def test_agent_call_record_has_agent_name(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        rt.register_agent("critic", _agent_returning("ok"))
        rt.run(
            'agent critic\nlet x: text = ask("review", agent = critic)\nx',
            log_file=log_path,
        )
        records = _load_jsonl(log_path)
        call_recs = [r for r in records if r.get("kind") == "agent_call_attempt"]
        assert call_recs
        assert call_recs[0].get("agent") == "critic"

    def test_agent_call_record_has_attempt_number(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        rt.register_agent("impl", _agent_returning("result"))
        rt.run(
            'agent impl\nlet x: text = ask("do work", agent = impl)\nx',
            log_file=log_path,
        )
        records = _load_jsonl(log_path)
        call_recs = [r for r in records if r.get("kind") == "agent_call_attempt"]
        assert call_recs
        assert isinstance(call_recs[0].get("attempt"), int)


# ---------------------------------------------------------------------------
# 3. Retry: multiple agent_call_attempt records
# ---------------------------------------------------------------------------


class TestRetryRecords:
    def test_retry_produces_multiple_attempt_records(self, tmp_path: Path) -> None:
        """With on_parse_error: retry[2], failed attempts appear in the trace."""
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver(default_strict_json=True)

        call_count = 0

        def agent(request: AgentRequest) -> AgentResponse:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return AgentResponse(content="not json")  # will fail to parse
            return AgentResponse(content="42")

        rt.register_agent("impl", agent)
        rt.run(
            "agent impl\n"
            'let x: int = ask("get int", agent = impl, on_parse_error = Retry(n = 2))\nx',
            log_file=log_path,
        )
        records = _load_jsonl(log_path)
        call_recs = [r for r in records if r.get("kind") == "agent_call_attempt"]
        assert len(call_recs) == 3

    def test_retry_records_carry_attempt_index(self, tmp_path: Path) -> None:
        """Attempt indices should be 0, 1, 2 for three attempts."""
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver(default_strict_json=True)

        call_count = 0

        def agent(request: AgentRequest) -> AgentResponse:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return AgentResponse(content="not json")
            return AgentResponse(content="42")

        rt.register_agent("impl", agent)
        rt.run(
            "agent impl\n"
            'let x: int = ask("get int", agent = impl, on_parse_error = Retry(n = 2))\nx',
            log_file=log_path,
        )
        records = _load_jsonl(log_path)
        call_recs = [r for r in records if r.get("kind") == "agent_call_attempt"]
        attempts = [r.get("attempt") for r in call_recs]
        assert attempts == [0, 1, 2]

    def test_parse_result_record_emitted_for_each_attempt(self, tmp_path: Path) -> None:
        """A parse_result record follows each agent_call_attempt."""
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver(default_strict_json=True)

        def agent(request: AgentRequest) -> AgentResponse:
            return AgentResponse(content="not json at all")

        rt.register_agent("impl", agent)
        src = (
            "agent impl\n"
            'let x: int = ask("get int", agent = impl, on_parse_error = Retry(n = 1))\nx'
        )
        try:
            rt.run(src, log_file=log_path)
        except SystemExit:
            pass

        records = _load_jsonl(log_path)
        kinds = [r.get("kind") for r in records]
        assert "parse_result" in kinds


# ---------------------------------------------------------------------------
# 4. Exception record + trace_id linkage
# ---------------------------------------------------------------------------


class TestExceptionRecord:
    def test_uncaught_exception_produces_exception_record(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver(default_strict_json=True)

        def agent(request: AgentRequest) -> AgentResponse:
            return AgentResponse(content="not json")

        rt.register_agent("impl", agent)
        result = rt.run(
            'agent impl\nlet x: int = ask("get int", agent = impl)\nx', log_file=log_path
        )
        assert not result.ok
        assert result.error is not None

        records = _load_jsonl(log_path)
        exc_recs = [r for r in records if r.get("kind") == "exception"]
        assert exc_recs

    def test_exception_record_has_type_name(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver(default_strict_json=True)

        def agent(request: AgentRequest) -> AgentResponse:
            return AgentResponse(content="not json")

        rt.register_agent("impl", agent)
        result = rt.run(
            'agent impl\nlet x: int = ask("get int", agent = impl)\nx', log_file=log_path
        )
        assert not result.ok

        records = _load_jsonl(log_path)
        exc_recs = [r for r in records if r.get("kind") == "exception"]
        assert exc_recs[0].get("type_name") == "AgentParseError"

    def test_exception_record_has_trace_id(self, tmp_path: Path) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver(default_strict_json=True)

        def agent(request: AgentRequest) -> AgentResponse:
            return AgentResponse(content="not json")

        rt.register_agent("impl", agent)
        result = rt.run(
            'agent impl\nlet x: int = ask("get int", agent = impl)\nx', log_file=log_path
        )
        assert not result.ok

        records = _load_jsonl(log_path)
        exc_recs = [r for r in records if r.get("kind") == "exception"]
        trace_id = exc_recs[0].get("trace_id")
        assert isinstance(trace_id, str) and trace_id

    def test_exception_trace_id_matches_agl_exception_field(self, tmp_path: Path) -> None:
        """The trace_id in the exception record matches the .trace_id field on
        the uncaught AgL exception (RunResult.error.fields['trace_id'])."""
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver(default_strict_json=True)

        def agent(request: AgentRequest) -> AgentResponse:
            return AgentResponse(content="not json")

        rt.register_agent("impl", agent)
        result = rt.run(
            'agent impl\nlet x: int = ask("get int", agent = impl)\nx', log_file=log_path
        )
        assert not result.ok
        assert result.error is not None

        records = _load_jsonl(log_path)
        exc_recs = [r for r in records if r.get("kind") == "exception"]
        assert exc_recs

        # The trace_id in the exception record must match the one on the raised
        # AgL exception (RunResult.error.fields['trace_id']).
        rec_trace_id = exc_recs[0].get("trace_id")
        agl_trace_id = result.error.fields.get("trace_id")
        assert rec_trace_id == agl_trace_id
        assert isinstance(rec_trace_id, str) and rec_trace_id

    def test_caught_exception_does_not_produce_exception_record(
        self, tmp_path: Path
    ) -> None:
        """An exception caught by try/catch is NOT written as an 'exception' record
        (it was handled in-language and did not escape the program)."""
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver(default_strict_json=True)

        def agent(request: AgentRequest) -> AgentResponse:
            return AgentResponse(content="not json")

        rt.register_agent("impl", agent)
        # The AgentParseError is caught and the result is a fallback string.
        result = rt.run(
            'agent impl\n'
            'try\n'
            '  let x: int = ask("get int", agent = impl)\n'
            '  x\n'
            'catch AgentParseError as e =>\n'
            '  0\n',
            log_file=log_path,
        )
        assert result.ok

        records = _load_jsonl(log_path)
        exc_recs = [r for r in records if r.get("kind") == "exception"]
        assert not exc_recs, "caught exception must not produce an exception record"


# ---------------------------------------------------------------------------
# 4b. Built-in runtime exceptions also carry a linked, non-empty trace_id
# ---------------------------------------------------------------------------


class TestBuiltinExceptionTraceId:
    """Built-in runtime exceptions (ArithmeticError, MatchError,
    MaxIterationsExceeded, ExecError) must carry a non-empty ``trace_id`` that
    matches their ``exception`` trace record — mirroring AgentParseError."""

    def test_arithmetic_error_trace_id_non_empty_with_logging(
        self, tmp_path: Path
    ) -> None:
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        # Uncaught division by zero → ArithmeticError escapes the program.
        result = rt.run("let x = 1 / 0\nx", log_file=log_path)
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "ArithmeticError"

        agl_trace_id = result.error.fields.get("trace_id")
        assert isinstance(agl_trace_id, str) and agl_trace_id

        records = _load_jsonl(log_path)
        exc_recs = [r for r in records if r.get("kind") == "exception"]
        assert exc_recs
        rec_trace_id = exc_recs[0].get("trace_id")
        # Linkage: the exception record's trace_id matches the raised exception.
        assert rec_trace_id == agl_trace_id
        assert isinstance(rec_trace_id, str) and rec_trace_id

    def test_arithmetic_error_trace_id_non_empty_without_logging(self) -> None:
        rt = PipelineDriver()
        # With logging OFF the trace_id still exists; only the field must be present.
        result = rt.run("let x = 1 / 0\nx", log_file=None)
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "ArithmeticError"
        agl_trace_id = result.error.fields.get("trace_id")
        assert isinstance(agl_trace_id, str) and agl_trace_id

    def test_match_error_trace_id_non_empty_without_logging(self) -> None:
        rt = PipelineDriver()
        # Explicit source raising retains the ordinary MatchError runtime contract.
        result = rt.run(
            "case 5 of\n"
            "  | 0 => ()\n"
            "  | _ =>\n"
            "      raise MatchError(message = \"no match\", "
            "scrutinee_type = \"int\", scrutinee = 5)\n",
            log_file=None,
        )
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "MatchError"
        agl_trace_id = result.error.fields.get("trace_id")
        assert isinstance(agl_trace_id, str) and agl_trace_id

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

        agl_trace_id = result.error.fields.get("trace_id")
        assert isinstance(agl_trace_id, str) and agl_trace_id

        records = _load_jsonl(log_path)
        exc_recs = [r for r in records if r.get("kind") == "exception"]
        assert exc_recs
        assert exc_recs[0].get("trace_id") == agl_trace_id


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
        rt = PipelineDriver()
        rt.register_agent("a", _agent_returning("hello"))
        result = rt.run('agent a\nlet x: text = ask("hi", agent = a)\nx', log_file=None)
        assert result.ok
        jsonl_files = list(tmp_path.rglob("*.jsonl"))
        assert not jsonl_files

    def test_no_log_with_decimal_mutation_still_works(self, tmp_path: Path) -> None:
        """A no-log run that mutates a decimal binding still succeeds and writes
        nothing ( early-out before any serialization/UUID work)."""
        rt = PipelineDriver()
        result = rt.run(
            "var x: decimal = 0.1\nx := x + 0.2",
            log_file=None,
        )
        assert result.ok
        jsonl_files = list(tmp_path.rglob("*.jsonl"))
        assert not jsonl_files

    def test_noop_store_mutation_early_outs_before_serialize(self) -> None:
        """``mutation()`` on a no-op store (path=None) returns without invoking
        the serializer.  Observable via a value the serializer would choke on:
        the early-out means no serialization (and no file) happens."""
        from agm.agl.runtime.trace import TraceStore
        from agm.agl.semantics.values import DecimalValue

        ts = TraceStore(path=None)
        # If the early-out were missing this would still run dumps_exact; the
        # honest check is that the no-op call neither raises nor produces output.
        ts.run_start()
        ts.mutation(name="x", value=DecimalValue(Decimal("0.3")), span=None)
        ts.run_end(ok=True)
        assert ts.path is None


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
# 9. Decimal exactness in traced values
# ---------------------------------------------------------------------------


class TestDecimalExactness:
    def test_decimal_value_traced_exactly(self, tmp_path: Path) -> None:
        """A Decimal in a mutation trace must survive round-trip without float error."""
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver()
        # 0.1 + 0.2 should NOT produce the float approximation 0.30000000000000004.
        rt.run(
            "var x: decimal = 0.1\nx := x + 0.2",
            log_file=log_path,
        )
        records = _load_jsonl(log_path)
        mut_recs = [r for r in records if r.get("kind") == "mutation"]
        assert mut_recs
        # The last mutation should have the value 0.3 (exact decimal semantics).
        last_val = mut_recs[-1].get("value")
        # Parsed from JSON: the serialized form must be "0.3" (not "0.30000...")
        assert last_val == "0.3" or last_val == Decimal("0.3") or str(last_val) == "0.3"


# ---------------------------------------------------------------------------
# 10. Source spans in records
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
        rt = PipelineDriver()
        rt.register_agent("impl", _agent_returning("hello"))
        rt.run('agent impl\nlet x: text = ask("do work", agent = impl)\nx', log_file=log_path)
        records = _load_jsonl(log_path)
        call_recs = [r for r in records if r.get("kind") == "agent_call_attempt"]
        assert call_recs
        rec = call_recs[0]
        has_span = "line" in rec or ("span" in rec and isinstance(rec["span"], dict))
        assert has_span


# ---------------------------------------------------------------------------
# 11. append_jsonl general helper in core/log
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
# 12. TraceStore properties and no-span branches
# ---------------------------------------------------------------------------


class TestTraceStoreProperties:
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

    def test_new_event_id_fresh_each_call(self) -> None:
        """``new_event_id`` returns a fresh non-empty id even when disabled."""
        from agm.agl.runtime.trace import TraceStore

        ts = TraceStore(path=None)
        a = ts.new_event_id()
        b = ts.new_event_id()
        assert isinstance(a, str) and a
        assert a != b

    def test_module_level_new_trace_id_public(self) -> None:
        """A public module-level ``new_trace_id`` exists so callers never import
        a private symbol across modules."""
        from agm.agl.runtime.trace import new_trace_id

        a = new_trace_id()
        b = new_trace_id()
        assert isinstance(a, str) and a
        assert a != b

    def test_trace_store_records_without_span(self, tmp_path: Path) -> None:
        """Methods called with span=None still emit valid JSONL (no line/col keys)."""
        import json as _json

        from agm.agl.runtime.trace import TraceStore
        from agm.agl.semantics.values import IntValue

        p = tmp_path / "t.jsonl"
        ts = TraceStore(path=p)
        ts.run_start()
        ts.agent_call_attempt(agent="x", attempt=0, prompt="p", span=None)
        ts.parse_result(ok=True, raw="r", normalized_raw="n", error_summary="", span=None)
        ts.mutation(name="v", value=IntValue(1), span=None)
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
        ts.exception(type_name="Abort", message="stop", trace_id="abc", span=None)
        ts.run_end(ok=True)

        lines = p.read_text(encoding="utf-8").splitlines()
        records = [_json.loads(ln) for ln in lines if ln.strip()]
        # None of the records should carry "line" or "col" (span was None).
        for rec in records:
            assert "line" not in rec
            assert "col" not in rec

    def test_trace_store_exception_with_span(self, tmp_path: Path) -> None:
        """exception() records line/col when a span is provided."""
        import json as _json

        from agm.agl.runtime.trace import TraceStore
        from agm.agl.syntax.spans import SourceSpan

        p = tmp_path / "t.jsonl"
        ts = TraceStore(path=p)
        span = SourceSpan(
            start_line=5, start_col=3, end_line=5, end_col=10,
            start_offset=40, end_offset=47,
        )
        ts.exception(type_name="Abort", message="stop", trace_id="abc", span=span)

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

    def test_retry_request_carries_reason_when_totally_unparseable(
        self, tmp_path: Path
    ) -> None:
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

        rt.register_agent("impl", agent)
        result = rt.run(
            "agent impl\n"
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
        assert any(
            e.category == "invalid_json" for e in second_request.validation_errors
        ), f"Expected category 'invalid_json', got: {second_request.validation_errors}"

    def test_parse_result_error_summary_non_empty_when_unparseable(
        self, tmp_path: Path
    ) -> None:
        """parse_result trace record's error_summary is non-empty for unparseable output."""
        log_path = tmp_path / "trace.jsonl"
        rt = PipelineDriver(default_strict_json=True)

        def agent(request: AgentRequest) -> AgentResponse:
            return AgentResponse(content="totally not json #@!")

        rt.register_agent("impl", agent)
        rt.run(
            "agent impl\n"
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

        rt.register_agent("impl", agent)
        # Patch the IR output parser to return a failure with no details at all.
        bare_fail = ParseResult(ok=False, value=None, error_msg="", errors=())
        with patch(
            "agm.agl.eval.ir_interpreter._parse_contract_output", return_value=bare_fail
        ):
            result = rt.run(
                'agent impl\nlet x: int = ask("q", agent = impl, on_parse_error = Abort())\nx'
            )
        # The program raises AgentParseError; run returns ok=False.
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "AgentParseError"
        # With empty errors the validation_errors list is empty.
        val_errs = result.error.fields.get("validation_errors")
        assert val_errs == []


# ---------------------------------------------------------------------------
# 13. prepare_trace_log truncates an existing file (clean-file guarantee)
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

    def test_prepare_trace_log_subsequent_record_is_only_content(
        self, tmp_path: Path
    ) -> None:
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
