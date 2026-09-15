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
values.
"""

from __future__ import annotations

from collections.abc import Mapping

from agm.agl.ir.builtin_nominals import BuiltinNominals
from agm.agl.ir.ids import Location
from agm.agl.semantics.values import ExceptionValue, TextValue, Value


def exception_message(exc: ExceptionValue) -> str:
    """Return *exc*'s ``message`` field text, or ``""`` when it is absent or not text.

    Shared by ``AglRaise`` and ``agm.agl.runtime.boundary.AglException`` so a
    Python-side ``str()`` of a propagating AgL exception is the same in both
    directions of the FFI boundary.
    """
    message = exc.fields.get("message")
    return message.value if isinstance(message, TextValue) else ""


def make_builtin_exception(
    type_name: str,
    message: str,
    *,
    nominals: BuiltinNominals,
    fields: Mapping[str, Value] | None = None,
) -> ExceptionValue:
    """Create an ``ExceptionValue`` for a built-in exception type.

    The exception's identity comes from ``nominals.resolve(type_name).nominal``
    — the caller's built-in nominal table — so the value carries the identity
    that table resolves for *type_name* rather than a hardcoded one; its
    display spelling is looked up from the program's descriptor table when
    rendered, never stored on the value. *fields* maps the declared AgL field
    names beyond ``message`` to their values.
    """
    all_fields: dict[str, Value] = {"message": TextValue(message)}
    all_fields.update(fields or {})
    declared = nominals.resolve(type_name)
    return ExceptionValue(nominal=declared.nominal, fields=all_fields)


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
        super().__init__(exception_message(exc))
        self.exc = exc
        self.span = span
