"""AgL trace store — records every significant runtime event as JSONL.

Records every agent-call attempt, parse result, retry, mutation, ``print``,
``exec`` command (with exit code and outputs), and exception, with a
``trace_id`` + source span.  Persisted under ``.agent-files/`` via
``core/log`` helpers.  Honors ``--no-log``/``--log-file`` (plan §9.6,
§10.1).

Design §12.6 record format followed:
- Every record carries: ``run_id``, ``kind``, ``trace_id``, ``line``,
  ``col`` (from the source span when available).
- ``agent_call_attempt``: agent name, attempt index, prompt text.
- ``parse_result``: ok flag, normalized output, error summary.
- ``mutation``: binding name and serialized new value.
- ``print``: rendered console output.
- ``exec_command``: command text, exit_code, stdout, stderr, timed_out.
- ``exception``: exception type_name + trace_id (for linkage to
  ``ExceptionValue.fields['trace_id']``).
- ``run_start`` / ``run_end``: boundary markers; ``run_end`` carries
  ``ok`` (whether the run succeeded).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from agm.agl.runtime.serialize import value_to_json_obj
from agm.core.log import append_jsonl

if TYPE_CHECKING:
    from agm.agl.eval.values import Value
    from agm.agl.syntax.spans import SourceSpan


def _new_id() -> str:
    """Generate a fresh unique identifier (UUID4 hex string)."""
    return uuid.uuid4().hex


class TraceStore:
    """Writes structured JSONL trace records for one AgL run.

    The per-run ``run_id`` ties all records together.  Each record also gets
    its own ``trace_id`` (a fresh UUID per event) — this is the value placed
    in ``ExceptionValue.fields['trace_id']`` so exceptions can be linked back
    to the matching record in the trace file.

    When *path* is ``None`` every method is a no-op (no-log mode).
    """

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._run_id: str = _new_id()

    @property
    def path(self) -> Path | None:
        """The trace file path, or ``None`` when logging is disabled."""
        return self._path

    @property
    def run_id(self) -> str:
        """Per-run identifier shared by all records in this trace."""
        return self._run_id

    def _emit(self, kind: str, trace_id: str, extra: dict[str, object]) -> None:
        """Append one JSONL record to the trace file."""
        record: dict[str, object] = {
            "run_id": self._run_id,
            "kind": kind,
            "trace_id": trace_id,
        }
        record.update(extra)
        append_jsonl(self._path, record)

    def run_start(self) -> None:
        """Record the start of a run (boundary marker)."""
        self._emit("run_start", _new_id(), {})

    def run_end(self, *, ok: bool) -> None:
        """Record the end of a run with the overall outcome."""
        self._emit("run_end", _new_id(), {"ok": ok})

    def agent_call_attempt(
        self,
        *,
        agent: str,
        attempt: int,
        prompt: str,
        span: "SourceSpan | None" = None,
    ) -> str:
        """Record one agent-call attempt; return the event's ``trace_id``."""
        trace_id = _new_id()
        extra: dict[str, object] = {
            "agent": agent,
            "attempt": attempt,
            "prompt": prompt,
        }
        if span is not None:
            extra["line"] = span.start_line
            extra["col"] = span.start_col
        self._emit("agent_call_attempt", trace_id, extra)
        return trace_id

    def parse_result(
        self,
        *,
        ok: bool,
        raw: str,
        normalized_raw: str,
        error_summary: str,
        span: "SourceSpan | None" = None,
    ) -> None:
        """Record the outcome of a codec parse attempt."""
        extra: dict[str, object] = {
            "ok": ok,
            "raw": raw,
            "normalized_raw": normalized_raw,
            "error_summary": error_summary,
        }
        if span is not None:
            extra["line"] = span.start_line
            extra["col"] = span.start_col
        self._emit("parse_result", _new_id(), extra)

    def mutation(
        self,
        *,
        name: str,
        value: "Value",
        span: "SourceSpan | None" = None,
    ) -> None:
        """Record a ``set`` mutation of a mutable binding."""
        from agm.agl.runtime.serialize import dumps_exact

        serialized = dumps_exact(value_to_json_obj(value), indent=None)
        extra: dict[str, object] = {
            "name": name,
            "value": serialized,
        }
        if span is not None:
            extra["line"] = span.start_line
            extra["col"] = span.start_col
        self._emit("mutation", _new_id(), extra)

    def print_stmt(
        self,
        *,
        rendered: str,
        span: "SourceSpan | None" = None,
    ) -> None:
        """Record a ``print`` statement output."""
        extra: dict[str, object] = {"rendered": rendered}
        if span is not None:
            extra["line"] = span.start_line
            extra["col"] = span.start_col
        self._emit("print", _new_id(), extra)

    def exec_command(
        self,
        *,
        command: str,
        exit_code: int,
        stdout: str,
        stderr: str,
        timed_out: bool,
        span: "SourceSpan | None" = None,
    ) -> None:
        """Record a completed ``exec`` shell command (success or failure)."""
        extra: dict[str, object] = {
            "command": command,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": timed_out,
        }
        if span is not None:
            extra["line"] = span.start_line
            extra["col"] = span.start_col
        self._emit("exec_command", _new_id(), extra)

    def exception(
        self,
        *,
        type_name: str,
        message: str,
        trace_id: str,
        span: "SourceSpan | None" = None,
    ) -> None:
        """Record an uncaught AgL exception that escapes the program.

        *trace_id* is the same value placed in
        ``ExceptionValue.fields['trace_id']`` — this linkage lets callers
        cross-reference the exception record in the trace file with the raised
        exception (design §12.6 / §8.1).
        """
        extra: dict[str, object] = {
            "type_name": type_name,
            "message": message,
        }
        if span is not None:
            extra["line"] = span.start_line
            extra["col"] = span.start_col
        self._emit("exception", trace_id, extra)


def noop_trace() -> TraceStore:
    """Return a no-op ``TraceStore`` (path=None → all writes are silent)."""
    return TraceStore(path=None)
