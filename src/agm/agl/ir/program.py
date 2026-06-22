"""Top-level program descriptor for the AgL typeless execution IR.

``ExecutableProgram`` is the root data structure emitted by the lowering/
linking phase and consumed by the new evaluator.  It carries no checker
``Type``, ``TypeEnvironment``, ``FunctionSignature``, ``CastSpec``, type
expression, or ``node_types``/``binding_types`` table.

All descriptors are immutable frozen dataclasses.  The dict fields on
``ExecutableProgram`` are populated by the linker before any evaluator sees the
program; they are treated as immutable after construction even though the Python
``dict`` type is technically mutable.  Do not mutate these tables at runtime.

M1 fields only.  Fields deferred to later milestones are noted in comments:
  M3  — richer field/variant metadata on ``NominalDescriptor``
  M4  — ``functions`` and ``params`` tables on ``ExecutableProgram``
  M5  — ``exports`` and ``agents`` on ``ExecutableModule``
  M6  — call-site and host-prep metadata on ``ExecutableProgram``
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from agm.agl.ir.ids import FunctionId, NominalId, SourceId, SymbolId
from agm.agl.ir.nodes import IrExpr
from agm.agl.modules.ids import ModuleId

__all__ = [
    "ExecutableModule",
    "ExecutableProgram",
    "NominalDescriptor",
    "NominalKind",
    "SourceFile",
    "SymbolDescriptor",
    "VariantDescriptor",
]


# ---------------------------------------------------------------------------
# Nominal kind
# ---------------------------------------------------------------------------


class NominalKind(enum.Enum):
    """Discriminates between nominal type families."""

    RECORD = "record"
    ENUM = "enum"
    EXCEPTION = "exception"


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SymbolDescriptor:
    """Descriptor for a named binding (let/var/param).

    ``symbol_id`` — the linker-allocated identity handle.
    ``mutable``   — ``True`` for ``var`` bindings, ``False`` for ``let``/params.
    ``public_name`` — the user-facing name (for error messages / debug);
                      ``None`` for synthesised lowering-internal symbols.
    ``owner``     — the module or function that declares this symbol.
    """

    symbol_id: SymbolId
    mutable: bool
    public_name: str | None
    owner: ModuleId | FunctionId


@dataclass(frozen=True, slots=True)
class VariantDescriptor:
    """Descriptor for one enum variant (M3d).

    ``name``   — the variant name.
    ``fields`` — declared field names in declaration order (names only; no
                 checker ``Type`` objects — the IR is typeless).
    """

    name: str
    fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NominalDescriptor:
    """Descriptor for a named nominal type (record, enum, or exception).

    ``nominal``      — the ``NominalId`` key for this descriptor.
    ``display_name`` — user-facing type name.
    ``kind``         — RECORD, ENUM, or EXCEPTION.
    ``fields``       — declared field names in declaration order (names only;
                       used for RECORD and EXCEPTION; ``()`` for ENUM which
                       stores fields per-variant in ``variants``).
    ``variants``     — for ENUM: ordered tuple of ``VariantDescriptor`` objects
                       (one per variant, in declaration order).  ``()`` for
                       RECORD and EXCEPTION.

    Safe defaults for ``fields`` and ``variants`` are ``()`` so existing M1/M2
    construction sites remain valid without keyword arguments for these fields.
    """

    nominal: NominalId
    display_name: str
    kind: NominalKind
    fields: tuple[str, ...] = ()
    variants: tuple[VariantDescriptor, ...] = ()


# ---------------------------------------------------------------------------
# Source file
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceFile:
    """A normalised source file record.

    ``display_name``    — human-readable file name for error messages.
    ``normalized_text`` — the normalised UTF-8 source text (LF line endings).
    """

    display_name: str
    normalized_text: str


# ---------------------------------------------------------------------------
# Module descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExecutableModule:
    """A single linked module in the executable program.

    ``module_id``    — the unique logical module identity.
    ``initializers`` — a sequence of IR expressions that constitute the
                       module-level initialiser (executed once, in order, when
                       the module is first loaded).

    Deferred to later milestones:
      M5 — ``exports: dict[str, SymbolId]``   (exported public names)
      M6 — ``agents: tuple[...]``              (declared agent bindings)
    """

    module_id: ModuleId
    initializers: tuple[IrExpr, ...]


# ---------------------------------------------------------------------------
# Program root
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExecutableProgram:
    """Root descriptor of a fully linked, typeless executable program.

    Emitted by the lowering/linking phase and consumed by the evaluator.
    All tables are populated by the linker before any evaluator accesses the
    program; treat them as immutable after construction (even though the
    underlying Python ``dict``s are mutable — the dataclass reference itself
    is frozen).

    Fields present in M1:
      ``entry_module`` — the module id of the program entry point.
      ``modules``      — map from ``ModuleId`` to ``ExecutableModule``.
      ``symbols``      — map from ``SymbolId`` to ``SymbolDescriptor``.
      ``nominals``     — map from ``NominalId`` to ``NominalDescriptor``.
      ``sources``      — map from ``SourceId`` to ``SourceFile``.

    Deferred to later milestones:
      M4 — ``functions: dict[FunctionId, FunctionDescriptor]``
      M4 — ``params: dict[ParamId, ParamDescriptor]``
      M6 — ``call_sites: dict[CallSiteId, CallSiteDescriptor]``
    """

    entry_module: ModuleId
    modules: dict[ModuleId, ExecutableModule]
    symbols: dict[SymbolId, SymbolDescriptor]
    nominals: dict[NominalId, NominalDescriptor]
    sources: dict[SourceId, SourceFile]
