"""AgL exception model and Python control-flow carriers.

``ExceptionValue`` (defined in ``values.py``) is the AgL-level exception
object — it is a first-class value.  Python's control flow for propagating
AgL exceptions uses ``AglRaise``, a subclass of ``Exception`` that wraps the
``ExceptionValue`` being thrown.

``AglRaise`` is intentionally separate from ``agm.agl.diagnostics.AglError``
(which represents *static* pipeline errors).  At runtime, only ``AglRaise``
is raised; it propagates up the Python call stack and is caught by:
  - a ``try``/``catch`` statement evaluator (matching by type name),
  - or the top-level ``WorkflowRuntime.run()`` dispatcher (converts to
    ``RunResult.error``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agm.agl.eval.values import ExceptionValue

if TYPE_CHECKING:
    from agm.agl.syntax.spans import SourceSpan


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

    def __init__(self, exc: ExceptionValue, *, span: "SourceSpan | None" = None) -> None:
        super().__init__(exc.display_name)
        self.exc = exc
        self.span: SourceSpan | None = span
