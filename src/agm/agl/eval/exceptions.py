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

from agm.agl.eval.values import ExceptionValue


class AglRaise(Exception):
    """Python carrier for a propagating AgL exception value.

    Raised by ``raise`` statements and by built-in operations that can fail
    (parse failures, loop exhaustion, pattern-match failures, etc.).

    ``exc`` is the ``ExceptionValue`` being propagated.
    """

    def __init__(self, exc: ExceptionValue) -> None:
        super().__init__(exc.type_name)
        self.exc = exc
