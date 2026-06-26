"""AgL trace store — records every significant runtime event as JSONL.

Records every agent-call attempt, parse result, retry, mutation, ``print``,
``exec`` command (with exit code, duration, and outputs), and exception, with a
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
- ``exec_command``: command text, exit_code, duration, stdout, stderr, timed_out.
- ``exception``: exception type_name + trace_id (for linkage to
  ``ExceptionValue.fields['trace_id']``).
- ``run_start`` / ``run_end``: boundary markers; ``run_end`` carries
  ``ok`` (whether the run succeeded).
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from agm.agl.runtime.serialize import value_to_json_obj
from agm.core.log import append_jsonl

if TYPE_CHECKING:
    from agm.agl.ir.ids import Location
    from agm.agl.semantics.values import Value
    from agm.agl.syntax.spans import SourceSpan


def new_trace_id() -> str:
    """Generate a fresh unique trace/event identifier (UUID4 hex string).

    Public module-level helper so other components (e.g. the interpreter) can
    mint ids without importing a private symbol across modules.
    """
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
        self._run_id: str = new_trace_id()
        # Set once a trace write fails: logging is disabled for the rest of the
        # run and a single warning is emitted (F2b).  Program semantics are
        # unaffected — a failed trace write must never abort the run.
        self._disabled: bool = False

    def new_event_id(self) -> str:
        """Return a fresh event-level trace id.

        Cheap and always valid, even when logging is disabled (``path`` is
        ``None``).  Callers (e.g. the interpreter at a built-in raise site) mint
        an id here and place it in both an ``ExceptionValue.fields['trace_id']``
        and the eventual ``exception`` record so the two can be cross-referenced
        (design §8.1 / §12.6).  When logging is disabled the id still exists; a
        ``trace_id`` that references no record is acceptable (§8.1 only requires
        the field be present).
        """
        return new_trace_id()

    @property
    def path(self) -> Path | None:
        """The trace file path, or ``None`` when logging is disabled."""
        return self._path

    @property
    def run_id(self) -> str:
        """Per-run identifier shared by all records in this trace."""
        return self._run_id

    def _emit(self, kind: str, trace_id: str, extra: dict[str, object]) -> None:
        """Append one JSONL record to the trace file (best-effort).

        A trace write must never corrupt program semantics: if the file becomes
        unwritable mid-run (e.g. permissions change), the ``OSError`` is caught,
        a single ``warning: trace logging disabled: <reason>`` line is emitted to
        stderr, and the store is disabled for the rest of the run (F2b).
        """
        if self._disabled:
            return
        record: dict[str, object] = {
            "run_id": self._run_id,
            "kind": kind,
            "trace_id": trace_id,
        }
        record.update(extra)
        try:
            append_jsonl(self._path, record)
        except OSError as exc:
            self._disabled = True
            print(f"warning: trace logging disabled: {exc}", file=sys.stderr)

    def run_start(self) -> None:
        """Record the start of a run (boundary marker)."""
        if self._path is None:
            return
        self._emit("run_start", new_trace_id(), {})

    def run_end(self, *, ok: bool) -> None:
        """Record the end of a run with the overall outcome."""
        if self._path is None:
            return
        self._emit("run_end", new_trace_id(), {"ok": ok})

    def agent_call_attempt(
        self,
        *,
        agent: str,
        attempt: int,
        prompt: str,
        span: "SourceSpan | Location | None" = None,
    ) -> str:
        """Record one agent-call attempt; return the event's ``trace_id``.

        Returns a fresh id even when logging is disabled (no record written) so
        callers can always thread a valid ``trace_id`` through.
        """
        trace_id = new_trace_id()
        if self._path is None:
            return trace_id
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
        span: "SourceSpan | Location | None" = None,
    ) -> None:
        """Record the outcome of a codec parse attempt."""
        if self._path is None:
            return
        extra: dict[str, object] = {
            "ok": ok,
            "raw": raw,
            "normalized_raw": normalized_raw,
            "error_summary": error_summary,
        }
        if span is not None:
            extra["line"] = span.start_line
            extra["col"] = span.start_col
        self._emit("parse_result", new_trace_id(), extra)

    def mutation(
        self,
        *,
        name: str,
        value: "Value",
        span: "SourceSpan | Location | None" = None,
    ) -> None:
        """Record a ``:=`` mutation of a mutable binding."""
        if self._path is None:
            return
        from agm.agl.runtime.serialize import dumps_exact

        serialized = dumps_exact(value_to_json_obj(value), indent=None)
        extra: dict[str, object] = {
            "name": name,
            "value": serialized,
        }
        if span is not None:
            extra["line"] = span.start_line
            extra["col"] = span.start_col
        self._emit("mutation", new_trace_id(), extra)

    def print_stmt(
        self,
        *,
        rendered: str,
        span: "SourceSpan | Location | None" = None,
    ) -> None:
        """Record a ``print`` statement output."""
        if self._path is None:
            return
        extra: dict[str, object] = {"rendered": rendered}
        if span is not None:
            extra["line"] = span.start_line
            extra["col"] = span.start_col
        self._emit("print", new_trace_id(), extra)

    def exec_command(
        self,
        *,
        command: str,
        exit_code: int,
        duration: float,
        stdout: str,
        stderr: str,
        timed_out: bool,
        span: "SourceSpan | Location | None" = None,
    ) -> str:
        """Record a completed ``exec`` shell command; return the event ``trace_id``.

        Returns a fresh id even when logging is disabled (no record written) so
        callers can always thread a valid ``trace_id`` through — mirroring
        ``agent_call_attempt`` so a typed-exec ``AgentParseError`` can link to
        the ``exec_command`` record (F3).
        """
        trace_id = new_trace_id()
        if self._path is None:
            return trace_id
        extra: dict[str, object] = {
            "command": command,
            "exit_code": exit_code,
            "duration": duration,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": timed_out,
        }
        if span is not None:
            extra["line"] = span.start_line
            extra["col"] = span.start_col
        self._emit("exec_command", trace_id, extra)
        return trace_id

    def exception(
        self,
        *,
        type_name: str,
        message: str,
        trace_id: str,
        span: "SourceSpan | Location | None" = None,
    ) -> None:
        """Record an uncaught AgL exception that escapes the program.

        *trace_id* is the same value placed in
        ``ExceptionValue.fields['trace_id']`` — this linkage lets callers
        cross-reference the exception record in the trace file with the raised
        exception (design §12.6 / §8.1).
        """
        if self._path is None:
            return
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
