"""AgL type-checking pass (Component 5).

Public API
----------
- :func:`check` — minimal-but-principled M1 type pass:
  ``ResolvedProgram × HostCapabilities → CheckedProgram``.
- :class:`CheckedProgram` — frozen dataclass with ``node_types``,
  ``contract_specs``, and ``warnings``.
- :class:`OutputContractSpec` — per-call codec + target-type record.
- :class:`AglTypeError` — fatal type error (span-aware ``AglError`` subclass).

Note (v2 rewrite in progress)
------------------------------
The ``check`` import is deferred because ``checker.py`` references AST nodes
that were removed/renamed by the S1a AST contract; eager import would crash at
module load until the checker is rewritten.  To keep ``__all__`` honest during
this window, ``"check"`` is added to ``__all__`` only under ``TYPE_CHECKING``.

TODO(S3): rewrite checker.py for the v2 AST, restore the eager
``from agm.agl.typecheck.checker import check`` import, and move ``"check"``
back into the unconditional ``__all__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agm.agl.typecheck.env import (
    AglTypeError,
    CheckedProgram,
    OutputContractSpec,
)
from agm.agl.typecheck.types import (
    BoolType,
    DecimalType,
    DictType,
    EnumType,
    ExceptionType,
    IntType,
    JsonType,
    ListType,
    RecordType,
    TextType,
    Type,
)

__all__ = [
    "AglTypeError",
    "BoolType",
    "CheckedProgram",
    "DecimalType",
    "DictType",
    "EnumType",
    "ExceptionType",
    "IntType",
    "JsonType",
    "ListType",
    "OutputContractSpec",
    "RecordType",
    "TextType",
    "Type",
]

if TYPE_CHECKING:
    from agm.agl.typecheck.checker import check

    __all__ += ["check"]
