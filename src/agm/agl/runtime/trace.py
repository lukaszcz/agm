"""Best-effort JSONL tracing for observable AgL runtime effects.

Trace records contain an ISO-8601 offset-aware timestamp, the run identifier,
and a kind-specific payload.  Run boundaries, stdout, agent requests and
responses, shell execution, escaping exceptions, and a companion's own
``runtime.trace`` records are traced; ordinary expression evaluation is
intentionally absent.
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from agm.agl.runtime.boundary import AglJson
from agm.agl.runtime.serialize import dumps_exact
from agm.agl.semantics.cycles import CYCLIC_VALUE_MARKER, non_data_marker
from agm.core.log import append_jsonl

if TYPE_CHECKING:
    from agm.agl.ir.ids import Location
    from agm.agl.syntax.spans import SourceSpan

_JSON_SCALARS = (str, int, float, bool)

#: Envelope keys :meth:`TraceStore.companion_record` writes itself; a
#: companion's own payload may never contribute one, so its shape can never
#: collide with the envelope it is embedded in.
RESERVED_ENVELOPE_KEYS = frozenset({"ts", "run_id", "kind", "origin", "line", "col"})


def _sanitize(value: object, active: frozenset[int]) -> object:
    """Return *value* with any reference cycle or non-JSON type replaced by a marker.

    Unlike every other trace payload -- always assembled from known-safe
    primitives -- a companion's ``runtime.trace`` payload is arbitrary data it
    built itself, so this walk degrades what it cannot represent instead of
    failing the call. Mirrors the value-walk cycle guard in
    ``semantics/cycles.py``, but over plain Python containers rather than AgL
    values. ``Decimal`` renders through the DSL's own exact-number convention
    (:func:`agm.agl.runtime.serialize.dumps_exact`); an ``AglJson`` value
    (JSON already crossed the boundary) unwraps to its raw JSON.
    """
    if value is None or isinstance(value, _JSON_SCALARS):
        return value
    if isinstance(value, Decimal):
        return dumps_exact(value, indent=None)
    if isinstance(value, AglJson):
        return _sanitize(value.value, active)
    if isinstance(value, Mapping):
        if id(value) in active:
            return CYCLIC_VALUE_MARKER
        return _sanitize_mapping(value, active | {id(value)})
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        if id(value) in active:
            return CYCLIC_VALUE_MARKER
        nested = active | {id(value)}
        return [_sanitize(item, nested) for item in value]
    return non_data_marker(type(value).__name__)


def _sanitize_mapping(payload: "Mapping[str, object]", active: frozenset[int]) -> dict[str, object]:
    """Sanitize *payload*'s values into a plain ``dict[str, object]`` (see :func:`_sanitize`).

    *active* must already include ``id(payload)`` for *payload* itself to be
    caught on re-entry through one of its own values.
    """
    return {str(key): _sanitize(item, active) for key, item in payload.items()}


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

    def _emit(self, kind: str, extra: dict[str, object]) -> None:
        """Append a record, disabling this best-effort service on I/O failure.

        The envelope (``ts``/``run_id``/``kind``) is laid over *extra* last,
        not merged into a dict *extra* could contribute to, so it always wins
        by construction — never merely by *extra* happening not to collide.
        """
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        # Wall clocks may move backwards; preserve trace-file ordering as an
        # ordering guarantee even when NTP adjusts the local clock.
        if timestamp < self._last_timestamp:
            timestamp = self._last_timestamp
        self._last_timestamp = timestamp
        record: dict[str, object] = dict(extra)
        record["ts"] = timestamp
        record["run_id"] = self._run_id
        record["kind"] = kind
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

    def companion_record(
        self,
        origin: str,
        kind: str,
        payload: "Mapping[str, object]",
        span: "SourceSpan | Location | None" = None,
    ) -> None:
        """Emit one record for a companion's own ``runtime.trace(kind, payload)`` call.

        *origin* is the calling companion's AgL module path. *payload* is
        degraded field by field (see :func:`_sanitize`) rather than raising,
        since it is the companion's own data, unlike every other record kind.
        Unlike every sibling method, this has no ``self._path is not None``
        guard of its own: the sole caller (``_CompanionRuntime.trace``) already
        checks :attr:`path` first, so it can skip computing *origin* — a
        display-path rendering — when this store is not writing; :meth:`_emit`
        still no-ops on a disabled store regardless.

        :raises ValueError: *payload* uses one of :data:`RESERVED_ENVELOPE_KEYS`
            — a companion programmer error, never silently overwritten or
            allowed to overwrite the envelope.
        """
        reserved = RESERVED_ENVELOPE_KEYS & payload.keys()
        if reserved:
            raise ValueError(f"runtime.trace payload uses reserved key(s): {sorted(reserved)}")
        extra = _sanitize_mapping(payload, frozenset({id(payload)}))
        extra["origin"] = origin
        self._emit(kind, self._with_span(extra, span))


def noop_trace() -> TraceStore:
    """Return a disabled-by-settings trace store."""
    return TraceStore(path=None)
