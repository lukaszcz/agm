"""AgL exception model and Python control-flow carriers.

``ExceptionValue`` (defined in ``agm.agl.semantics.values``) is the AgL-level
exception object — it is a first-class value.  Python's control flow for
propagating AgL exceptions uses ``AglRaise``, a subclass of ``Exception`` that
wraps the ``ExceptionValue`` being thrown.

``AglRaise`` is intentionally separate from ``agm.agl.diagnostics.AglError``
(which represents *static* pipeline errors).  At runtime, only ``AglRaise``
is raised; it propagates up the Python call stack and is caught by:
  - a ``try``/``catch`` statement evaluator (matching by type name),
  - or the top-level ``PipelineDriver.run()`` dispatcher (converts to
    ``RunResult.error``).

``make_builtin_exception`` is the single shared factory for built-in exception
values. The IR interpreter supplies the trace id.
"""

from __future__ import annotations

from agm.agl.ir.builtin_nominals import BuiltinNominals
from agm.agl.ir.ids import Location
from agm.agl.semantics.values import ExceptionValue, TextValue, Value


def make_builtin_exception(
    type_name: str, message: str, *, nominals: BuiltinNominals, trace_id: str = "", **extra: Value
) -> ExceptionValue:
    """Create an ``ExceptionValue`` for a built-in exception type.

    The exception's nominal is ``nominals.nominal(type_name)`` — the caller's
    built-in nominal table, so the value carries the identity that table
    resolves for *type_name* rather than a hardcoded one. ``display_name``
    is derived from that same nominal (``NominalId.display_name``), so a
    scoped declaration reports its own spelling instead of the bare name.
    ``trace_id`` is minted by the *caller's* evaluator (per-evaluator identity).
    Extra keyword arguments become additional fields beyond ``message`` and
    ``trace_id``.
    """
    fields: dict[str, Value] = {
        "message": TextValue(message),
        "trace_id": TextValue(trace_id),
    }
    fields.update(extra)
    nominal = nominals.nominal(type_name)
    return ExceptionValue(
        nominal=nominal,
        display_name=nominal.display_name,
        fields=fields,
    )


class AglRaise(Exception):
    """Python carrier for a propagating AgL exception value.

    Raised by ``raise`` statements and by built-in operations that can fail
    (parse failures, loop exhaustion, arithmetic failures, etc.).

    ``exc`` is the ``ExceptionValue`` being propagated.
    ``span`` is the source span of the statement that raised this exception
    (when known — design : source location is part of runtime error
    reporting).  ``None`` when the raise site does not have span information
    available (e.g. binary-op arithmetic errors).
    """

    def __init__(self, exc: ExceptionValue, *, span: Location | None = None) -> None:
        super().__init__(exc.display_name)
        self.exc = exc
        self.span = span
