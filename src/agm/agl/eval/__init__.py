"""AgL evaluator (Component 6).

Public API
----------
- :class:`Interpreter` — tree-walking evaluator.
- :class:`Scope` — runtime scope frame.
- :class:`Binding` — a single scope binding.
- :class:`AglRaise` — Python carrier for a propagating AgL exception.
- Value types: ``TextValue``, ``IntValue``, ``DecimalValue``, ``BoolValue``,
  ``JsonValue``, ``ListValue``, ``DictValue``, ``RecordValue``, ``EnumValue``,
  ``ExceptionValue``, ``Value``.

Note (v2 rewrite in progress)
------------------------------
The ``Interpreter`` import is deferred because ``agm.agl.eval.interpreter``
references AST nodes that were removed/renamed by the S1a AST contract; eager
import would crash at module load (taking down every ``agm.agl.*`` consumer,
including the lexer and AST test suites) until the interpreter is rewritten.
To keep ``__all__`` honest during this window, ``"Interpreter"`` is added to
``__all__`` only under ``TYPE_CHECKING``.

TODO(S4): rewrite interpreter.py for the v2 AST, restore the eager
``from agm.agl.eval.interpreter import Interpreter`` import, and move
``"Interpreter"`` back into the unconditional ``__all__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agm.agl.eval.exceptions import AglRaise
from agm.agl.eval.scope import Binding, Scope
from agm.agl.eval.values import (
    BoolValue,
    DecimalValue,
    DictValue,
    EnumValue,
    ExceptionValue,
    IntValue,
    JsonValue,
    ListValue,
    RecordValue,
    TextValue,
    Value,
)

__all__ = [
    "AglRaise",
    "Binding",
    "BoolValue",
    "DecimalValue",
    "DictValue",
    "EnumValue",
    "ExceptionValue",
    "IntValue",
    "JsonValue",
    "ListValue",
    "RecordValue",
    "Scope",
    "TextValue",
    "Value",
]

if TYPE_CHECKING:
    from agm.agl.eval.interpreter import Interpreter

    __all__ += ["Interpreter"]
