"""Best-effort JSONL tracing for observable AgL runtime effects.

Trace records contain an ISO-8601 offset-aware timestamp, the run identifier,
and a kind-specific payload.  Only run boundaries, stdout, agent requests and
responses, and shell execution are traced; ordinary expression evaluation is
intentionally absent.
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from agm.core.log import append_jsonl

if TYPE_CHECKING:
    from agm.agl.ir.ids import Location
    from agm.agl.syntax.spans import SourceSpan


class TraceStore:
    """Write structured records for one AgL run without affecting its semantics."""

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._run_id = uuid.uuid4().hex
        self._disabled = False
        self._last_timestamp = ""

    def disable(self, reason: OSError) -> None:
        """Disable logging after a filesystem failure, warning exactly once."""
        if not self._disabled:
            print(f"warning: trace logging disabled: {reason}", file=sys.stderr)
        self._disabled = True
        self._path = None

    def activate(self, path: Path | None) -> None:
        """Repoint the store according to the current live log settings."""
        if not self._disabled:
            self._path = path

    @property
    def path(self) -> Path | None:
        """The trace destination, if tracing is currently enabled."""
        return self._path

    @property
    def disabled(self) -> bool:
        """Whether an I/O failure, rather than settings, disabled this store."""
        return self._disabled

    @property
    def run_id(self) -> str:
        """The identifier shared by records written during this run."""
        return self._run_id

    def _emit(self, kind: str, extra: dict[str, object]) -> None:
        """Append a record, disabling this best-effort service on I/O failure."""
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        # Wall clocks may move backwards; preserve trace-file ordering as an
        # ordering guarantee even when NTP adjusts the local clock.
        if timestamp < self._last_timestamp:
            timestamp = self._last_timestamp
        self._last_timestamp = timestamp
        record: dict[str, object] = {"ts": timestamp, "run_id": self._run_id, "kind": kind}
        record.update(extra)
        try:
            append_jsonl(self._path, record)
        except OSError as exc:
            self.disable(exc)

    @staticmethod
    def _with_span(
        extra: dict[str, object], span: "SourceSpan | Location | None"
    ) -> dict[str, object]:
        if span is not None:
            extra["line"] = span.start_line
            extra["col"] = span.start_col
        return extra

    def run_start(self) -> None:
        if self._path is not None:
            self._emit("run_start", {})

    def run_end(self, *, ok: bool) -> None:
        if self._path is not None:
            self._emit("run_end", {"ok": ok})

    def agent_request(
        self,
        *,
        agent: dict[str, object],
        attempt: int,
        max_attempts: int,
        prompt: str,
        target_type: str,
        codec: str,
        strict_json: bool | None,
        json_schema: object | None,
        span: "SourceSpan | Location | None" = None,
    ) -> None:
        if self._path is not None:
            self._emit(
                "agent_request",
                self._with_span(
                    {
                        "agent": agent,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "prompt": prompt,
                        "target_type": target_type,
                        "codec": codec,
                        "strict_json": strict_json,
                        "json_schema": json_schema,
                    },
                    span,
                ),
            )

    def agent_response(
        self,
        *,
        ok: bool,
        content: str | None = None,
        metadata: dict[str, object] | None = None,
        cause: str | None = None,
        cancelled: bool = False,
        reason: str | None = None,
        call_info: dict[str, object] | None = None,
        span: "SourceSpan | Location | None" = None,
    ) -> None:
        if self._path is None:
            return
        extra: dict[str, object] = {"ok": ok}
        if content is not None:
            extra["content"] = content
        if metadata:
            extra["metadata"] = metadata
        if cause is not None:
            extra["cause"] = cause
        if cancelled:
            extra["cancelled"] = True
        if reason is not None:
            extra["reason"] = reason
        if call_info:
            extra.update(call_info)
        self._emit("agent_response", self._with_span(extra, span))

    def parse_result(
        self,
        *,
        ok: bool,
        raw: str,
        normalized_raw: str,
        error_summary: str,
        span: "SourceSpan | Location | None" = None,
    ) -> None:
        if self._path is not None:
            self._emit(
                "parse_result",
                self._with_span(
                    {
                        "ok": ok,
                        "raw": raw,
                        "normalized_raw": normalized_raw,
                        "error_summary": error_summary,
                    },
                    span,
                ),
            )

    def print_stmt(self, *, rendered: str, span: "SourceSpan | Location | None" = None) -> None:
        if self._path is not None:
            self._emit("print", self._with_span({"rendered": rendered}, span))

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
    ) -> None:
        if self._path is not None:
            self._emit(
                "exec_command",
                self._with_span(
                    {
                        "command": command,
                        "exit_code": exit_code,
                        "duration": duration,
                        "stdout": stdout,
                        "stderr": stderr,
                        "timed_out": timed_out,
                    },
                    span,
                ),
            )

    def exception(
        self,
        *,
        type_name: str,
        message: str,
        span: "SourceSpan | Location | None" = None,
    ) -> None:
        """Record an uncaught AgL exception that escapes the program."""
        if self._path is not None:
            self._emit(
                "exception",
                self._with_span({"type_name": type_name, "message": message}, span),
            )


def noop_trace() -> TraceStore:
    """Return a disabled-by-settings trace store."""
    return TraceStore(path=None)
