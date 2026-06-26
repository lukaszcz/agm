"""AgL exception model and Python control-flow carriers.

``ExceptionValue`` (defined in ``agm.agl.semantics.values``) is the AgL-level
exception object — it is a first-class value.  Python's control flow for
propagating AgL exceptions uses ``AglRaise``, a subclass of ``Exception`` that
wraps the ``ExceptionValue`` being thrown.

``AglRaise`` is intentionally separate from ``agm.agl.diagnostics.AglError``
(which represents *static* pipeline errors).  At runtime, only ``AglRaise``
is raised; it propagates up the Python call stack and is caught by:
  - a ``try``/``catch`` statement evaluator (matching by type name),
  - or the top-level ``WorkflowRuntime.run()`` dispatcher (converts to
    ``RunResult.error``).

``make_builtin_exception`` is the single shared factory for built-in exception
values. The IR interpreter supplies the trace id.
"""

from __future__ import annotations

from agm.agl.ir.ids import Location, NominalId
from agm.agl.modules.ids import PRELUDE_ID
from agm.agl.semantics.values import ExceptionValue, TextValue, Value


def make_builtin_exception(
    type_name: str, message: str, *, trace_id: str = "", **extra: Value
) -> ExceptionValue:
    """Create an ``ExceptionValue`` for a built-in exception type.

    Built-in exceptions use ``NominalId(PRELUDE_ID, type_name)``.
    ``trace_id`` is minted by the *caller's* evaluator (per-evaluator identity).
    Extra keyword arguments become additional fields beyond ``message`` and
    ``trace_id``.
    """
    fields: dict[str, Value] = {
        "message": TextValue(message),
        "trace_id": TextValue(trace_id),
    }
    fields.update(extra)
    return ExceptionValue(
        nominal=NominalId(PRELUDE_ID, type_name),
        display_name=type_name,
        fields=fields,
    )


class AglRaise(Exception):
    """Python carrier for a propagating AgL exception value.

    Raised by ``raise`` statements and by built-in operations that can fail
    (parse failures, loop exhaustion, pattern-match failures, etc.).

    ``exc`` is the ``ExceptionValue`` being propagated.
    ``span`` is the source span of the statement that raised this exception
    (when known — design §12.6: source location is part of runtime error
    reporting).  ``None`` when the raise site does not have span information
    available (e.g. binary-op arithmetic errors).
    """

    def __init__(
        self, exc: ExceptionValue, *, span: Location | None = None
    ) -> None:
        super().__init__(exc.display_name)
        self.exc = exc
        self.span = span
