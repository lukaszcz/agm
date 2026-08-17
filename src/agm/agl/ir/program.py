"""Top-level program descriptor for the AgL typeless execution IR.

``ExecutableProgram`` is the root data structure emitted by the lowering/
linking phase and consumed by the new evaluator.  It carries no checker
``Type``, ``TypeEnvironment``, ``FunctionSignature``, ``CastSpec``, type
expression, or ``node_types``/``binding_types`` table.

All descriptors are immutable frozen dataclasses.  The dict fields on
``ExecutableProgram`` are populated by the linker before any evaluator sees the
program; they are treated as immutable after construction even though the Python
``dict`` type is technically mutable.  Do not mutate these tables at runtime.

The descriptor keeps the runtime program tables used by lowering, linking,
and evaluation.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from agm.agl.ir.builtin_nominals import NO_BUILTIN_DECLARATIONS, BuiltinNominals
from agm.agl.ir.builtin_vars import BuiltinVarKey
from agm.agl.ir.contracts import ContractRequest, ParamDecoder
from agm.agl.ir.ids import ContractId, FunctionId, Location, NominalId, SourceId, SymbolId
from agm.agl.ir.nodes import IrExpr, IrFunctionParam
from agm.agl.modules.ids import ENTRY_ID, ModuleId, spell_scope_path

__all__ = [
    "ContractId",
    "ContractRequest",
    "DryRunEntry",
    "ExecutableModule",
    "ExecutableProgram",
    "ExternFunctionBody",
    "FunctionDescriptor",
    "FunctionImpl",
    "IrFunctionBody",
    "IrParam",
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
    ``synthetic`` — whether lowering allocated this symbol without a source binding.
    """

    symbol_id: SymbolId
    mutable: bool
    public_name: str | None
    owner: ModuleId | FunctionId
    synthetic: bool = False


@dataclass(frozen=True, slots=True)
class VariantDescriptor:
    """Descriptor for one enum variant.

    ``name``   — the variant name.
    ``fields`` — declared field names in declaration order (names only; no
                 checker ``Type`` objects — the IR is typeless).
    """

    name: str
    fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NominalDescriptor:
    """Descriptor for a named nominal type (record, enum, or exception).

    ``nominal``      — the ``NominalId`` key for this descriptor: an opaque
                       handle carrying no spelling or module of its own (see
                       ``ir.ids.NominalId``).
    ``module_id``    — the module that declares this nominal.
    ``scope_path``   — the declaration's scope path within its module.
    ``declared_name``— the bare name the declaration was written under (no
                       scope prefix).
    ``display_name`` — the scoped source spelling, DERIVED from ``scope_path``
                       and ``declared_name`` rather than stored, so a
                       descriptor can never contradict its own path; used for
                       diagnostics and rendering.
    ``kind``         — RECORD, ENUM, or EXCEPTION.
    ``fields``       — declared field names in declaration order (names only;
                       used for RECORD and EXCEPTION; ``()`` for ENUM which
                       stores fields per-variant in ``variants``).
    ``variants``     — for ENUM: ordered tuple of ``VariantDescriptor`` objects
                       (one per variant, in declaration order).  ``()`` for
                       RECORD and EXCEPTION.
    ``bears_name_path`` — whether this identity is the one its
                       ``(module_id, scope_path, declared_name)`` path
                       currently resolves to, per the type table's name
                       index. A superseded declaration, and a declaration
                       from an unpromoted REPL entry, both remain in
                       ``ExecutableProgram.nominals`` (it is derived from
                       every declaration the type table retains, not just
                       the live ones) with this ``False`` — metadata about
                       the identity's current standing, not part of its
                       shape, so it is excluded from equality/hashing
                       (``compare=False``) the same way ``TypeDef.is_builtin``
                       is: two snapshots of the same identity taken before
                       and after a later redeclaration must still compare
                       equal.

    Safe defaults for ``fields`` and ``variants`` are ``()`` so construction sites
    can omit them when the descriptor does not need nominal details.
    """

    nominal: NominalId
    module_id: ModuleId
    scope_path: tuple[str, ...]
    declared_name: str
    kind: NominalKind
    fields: tuple[str, ...] = ()
    variants: tuple[VariantDescriptor, ...] = ()
    bears_name_path: bool = field(default=True, compare=False)

    @property
    def display_name(self) -> str:
        """The scoped source spelling this declaration was written under."""
        return spell_scope_path((*self.scope_path, self.declared_name))


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


@dataclass(frozen=True, slots=True)
class IrFunctionBody:
    """Ordinary function implementation: an AgL expression evaluated per call."""

    body: IrExpr


@dataclass(frozen=True, slots=True)
class ExternFunctionBody:
    """``extern def`` implementation: crosses into a companion Python module.

    ``name``     — the extern's final declared member name; ``runtime.externs.ExternRegistry``
                   resolves it in the owning module's companion, then the boundary walkers
                   pass encoded arguments positionally.
    The boundary dispatches on runtime values, so an extern retains no type
    schema after lowering.
    """

    name: str


FunctionImpl = IrFunctionBody | ExternFunctionBody


@dataclass(frozen=True, slots=True)
class FunctionDescriptor:
    """Descriptor for any callable — ordinary function or extern def."""

    function_id: FunctionId
    function_symbol: SymbolId
    module_id: ModuleId
    params: "tuple[IrFunctionParam, ...]"
    impl: FunctionImpl
    param_labels: tuple[str, ...] = ()
    result_label: str = "?"
    is_synthetic_main: bool = False

    @property
    def is_extern(self) -> bool:
        return isinstance(self.impl, ExternFunctionBody)


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

    """

    module_id: ModuleId
    initializers: tuple[IrExpr, ...]


# ---------------------------------------------------------------------------
# Module parameter descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IrParam:
    """Descriptor for a module ``param`` declaration.

    ``symbol``      — the linker-allocated ``SymbolId`` for this param binding.
    ``module``      — the declaring module's logical identity.
    ``public_name`` — the scope-path user-facing param name.
    ``required``    — ``True`` when the param has no default (host must supply
                      a value; reaching ``run()`` without one is a host bug).
    ``default``     — an ``IrExpr`` to evaluate when the host supplies no value
                      (``None`` when ``required`` is ``True``).
    ``location``    — source location of the ``param`` declaration.

    ``IrParam`` is metadata — it is NOT a member of ``IrExpr``.  The IR
    evaluator reads ``program.params`` in ``run()`` and installs each param's
    value into the base frame BEFORE running any module initializer.
    """

    symbol: SymbolId
    public_name: str
    required: bool
    default: "IrExpr | None"
    location: Location
    external_decoder: ParamDecoder | None = None
    module: ModuleId = ENTRY_ID

    @property
    def qualified_public_name(self) -> str:
        """Return this param's module-qualified external spelling."""
        return f"{self.module.display()}::{self.public_name}"


# ---------------------------------------------------------------------------
# Dry-run inventory
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DryRunEntry:
    """Inventory entry for a single call site in a linked module.

    module            — module containing the call site.
    callee            — human-readable callee label (agent name, "exec", etc.).
    codec_name        — codec used ("text", "json").
    target_type_label — repr(target_type) from the contract spec, or "text".
    has_schema        — True when the contract carries a JSON Schema.
    parse_policy      — parse policy string from the call site record.
    line              — 1-based source line of the call.
    col               — 0-based source column of the call.
    """

    module: ModuleId
    callee: str
    codec_name: str
    target_type_label: str
    has_schema: bool
    parse_policy: str
    line: int
    col: int


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

    ``entry_module`` — the module id of the program entry point.
      ``modules``      — map from ``ModuleId`` to ``ExecutableModule``.
      ``symbols``      — map from ``SymbolId`` to ``SymbolDescriptor``.
      ``nominals``     — map from ``NominalId`` to ``NominalDescriptor``.
      ``sources``      — map from ``SourceId`` to ``SourceFile``.
      ``functions``    — ordinary functions and externs, keyed by ``FunctionId``.
      ``program_symbols`` — source declaration node id -> symbol for each linked
        ``program def``. Its values are unique.
      ``program_functions`` — public ``program def`` symbol -> zero-argument,
        body-backed function id, used by the host invocation entry point. Its
        keys are exactly the ``program_symbols`` values, making the two tables
        a bidirectional index.
      ``synthetic_main_symbol`` — the synthesized inline-source ``main`` entry,
        when this program was wrapped. Its ``program_functions`` entry resolves
        to the sole function descriptor marked ``is_synthetic_main``; file programs
        leave this ``None``. When a
        host explicitly invokes it, its final frame supplements reported bindings.
      ``builtin_nominals`` — bare built-in type name -> the ``NominalId`` a
        host mints for it (see ``agm.agl.ir.builtin_nominals``), built during
        lowering from the program's ``builtin`` declarations. Defaults to
        ``NO_BUILTIN_DECLARATIONS`` (every name resolves to the shipped
        standard library's own identity), which keeps the many direct
        ``ExecutableProgram`` constructions in ``tests/`` working without
        threading this table through every one of them.
      ``builtin_var_declarations`` — all module-qualified ``builtin var``
        declaration identities in the linked modules. This allows structural
        validation to confirm that every non-engine structured host-backed key
        names an actual declaration, including its scope path.
      ``builtin_setting_defaults`` — builtin-var key -> a checked, constant
        IR expression declared by ``builtin var``. The evaluator uses it only
        when the host did not seed that binding. Legacy string keys remain
        supported as root ``std/config`` engine-setting keys.

    """

    entry_module: ModuleId
    modules: dict[ModuleId, ExecutableModule]
    symbols: dict[SymbolId, SymbolDescriptor]
    nominals: dict[NominalId, NominalDescriptor]
    sources: dict[SourceId, SourceFile]
    functions: dict[FunctionId, FunctionDescriptor] = field(default_factory=dict)
    program_symbols: dict[int, SymbolId] = field(default_factory=dict)
    program_functions: dict[SymbolId, FunctionId] = field(default_factory=dict)
    synthetic_main_symbol: SymbolId | None = None
    params: tuple[IrParam, ...] = ()
    contracts: dict["ContractId", "ContractRequest"] = field(default_factory=dict)
    dry_run_inventory: "tuple[DryRunEntry, ...]" = ()
    builtin_nominals: BuiltinNominals = NO_BUILTIN_DECLARATIONS
    builtin_var_declarations: frozenset[BuiltinVarKey] = frozenset()
    builtin_setting_defaults: dict[BuiltinVarKey | str, IrExpr] = field(default_factory=dict)
