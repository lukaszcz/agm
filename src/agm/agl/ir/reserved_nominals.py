"""Fixed identities for host-known nominal types with no source declaration.

A record/enum/exception is ordinarily identified by the declaration that
introduced it (its AST node). Some nominal types the host recognizes have no
such declaration in hand at the point a value needs one — a session run with
``--no-stdlib`` parses no standard library at all, and the host mints certain
values (a raised built-in exception, a structured ``exec`` ``ExecResult``, an
``Option`` value, ...) directly. This module fixes a stable identity for
every such name so a handle naming one is never left without an identity to
carry.

It also fixes :data:`NO_DECL_ID`, the value a handle carries when no
declaration identity is attached to it at all. The three ranges are disjoint
by construction: an AST node id is non-negative, ``NO_DECL_ID`` is ``-1``,
and every reserved identity is ``-2`` or below.

This module holds only that data and its lookup, so ``semantics/`` (the typed
nominal-handle model) and the passes above it can depend on it without a
layering violation — it sits in ``ir/``, the typeless data root that
``semantics`` already depends on wholesale and that ``typecheck`` reaches by
explicit per-module allowance, and is itself a pure data leaf importing
nothing from ``agm`` (see ``tests/test_agl_dependencies.py``), matching
``ir/builtin_nominals.py``.

The reserved names are exactly: every built-in exception name
(``semantics.types.BUILTIN_EXCEPTION_NAMES``), every built-in prelude type
name (``semantics.types.BUILTIN_PRELUDE_TYPE_NAMES``), plus ``"Option"`` —
``std/option::Option`` is an ordinary enum declaration, but the host mints
``Option`` values directly (``runtime/option.py``), so it needs a fixed
identity the same way a built-in does. This module cannot import
``semantics`` (see above), so the list is spelled out literally here; a
dependency test asserts the two catalogs never drift apart.
"""

from __future__ import annotations

import types
from collections.abc import Mapping

__all__ = [
    "NO_DECL_ID",
    "RESERVED_NOMINAL_NAMES",
    "RESERVED_NOMINAL_IDS",
    "RESERVED_ENUM_MEMBER_IDS",
    "require_reserved_enum_member_id",
    "require_reserved_nominal_id",
    "reserved_nominal_id",
]

#: The identity a nominal handle carries when no declaration identity is
#: attached to it. Distinct from every AST node id (those are non-negative,
#: and ``0`` is a real id — the first node of a program's first declaration)
#: and from every reserved identity (see :data:`RESERVED_NOMINAL_IDS`).
NO_DECL_ID: int = -1

#: Every bare name the host recognizes without needing a source declaration:
#: built-in exceptions, built-in prelude types, and ``Option``. Order fixes
#: each name's negative id (see ``RESERVED_NOMINAL_IDS``) and is otherwise
#: insignificant.
RESERVED_NOMINAL_NAMES: tuple[str, ...] = (
    # Built-in exceptions (mirrors semantics.types.BUILTIN_EXCEPTION_NAMES).
    "Exception",
    "AgentCallError",
    "AgentParseError",
    "ExecError",
    "ExternError",
    "MaxIterationsExceeded",
    "MatchError",
    "IndexError",
    "KeyError",
    "TypeError",
    "ArithmeticError",
    "UndefinedVariableError",
    "ImmutableBindingError",
    "Abort",
    "RecursionError",
    "CastError",
    "JsonParseError",
    "RangeError",
    "CyclicValueError",
    # Built-in prelude types (mirrors semantics.types.BUILTIN_PRELUDE_TYPE_NAMES).
    "ExecResult",
    "ParsePolicy",
    "Agent",
    "OutputContract",
    "OutputContractOption",
    "AgentRequest",
    # Host-minted std/option::Option (see runtime/option.py).
    "Option",
    # Appended prelude types preserve every established reserved identity above.
    "SessionTransport",
    "Session",
    "SessionStats",
    "SessionError",
)

#: Bare reserved name -> its stable, distinct identity. Derived from
#: :data:`RESERVED_NOMINAL_NAMES` (never hand-maintained) so ids stay stable
#: across edits that only reorder or append names elsewhere. Every id is
#: ``-2`` or below, so it can collide with neither an AST node id (always
#: non-negative) nor :data:`NO_DECL_ID`.
RESERVED_NOMINAL_IDS: Mapping[str, int] = types.MappingProxyType(
    {name: NO_DECL_ID - (index + 1) for index, name in enumerate(RESERVED_NOMINAL_NAMES)}
)

#: Stable fallback identities for members of host-known enums. This range is
#: disjoint from parser node ids and the top-level reserved identities.
RESERVED_ENUM_MEMBER_IDS: Mapping[tuple[str, str], int] = types.MappingProxyType(
    {
        ("ParsePolicy", "Abort"): -1000,
        ("ParsePolicy", "Retry"): -1001,
        ("Agent", "AgentCommand"): -1010,
        ("Agent", "AgentClaude"): -1011,
        ("Agent", "AgentCodex"): -1012,
        ("Agent", "AgentPi"): -1013,
        ("OutputContractOption", "None"): -1020,
        ("OutputContractOption", "Some"): -1021,
        ("Option", "None"): -1030,
        ("Option", "Some"): -1031,
        ("SessionTransport", "Cli"): -1040,
        ("SessionTransport", "Rpc"): -1041,
    }
)


def reserved_nominal_id(name: str) -> int | None:
    """Return *name*'s reserved identity, or ``None`` if *name* is not reserved."""
    return RESERVED_NOMINAL_IDS.get(name)


def require_reserved_nominal_id(name: str) -> int:
    """Return *name*'s reserved identity, for a caller that knows it has one.

    For the host's own built-in constants and canonical shapes, whose names
    are in :data:`RESERVED_NOMINAL_NAMES` by construction.
    """
    reserved_id = RESERVED_NOMINAL_IDS.get(name)
    assert reserved_id is not None, f"compiler bug: {name!r} is not a reserved nominal name"
    return reserved_id


def require_reserved_enum_member_id(enum_name: str, member_name: str) -> int:
    """Return the stable fallback identity for a host-known enum member."""
    member_id = RESERVED_ENUM_MEMBER_IDS.get((enum_name, member_name))
    assert member_id is not None, (
        f"compiler bug: {enum_name!r}::{member_name!r} is not a reserved enum member"
    )
    return member_id
