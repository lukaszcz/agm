"""Independently lowered module bodies and their linkable tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agm.agl import artifact_serialization
from agm.agl.artifact_cache import retain_lowered_module, retained_lowered_module
from agm.agl.ir.builtin_vars import BuiltinVarKey
from agm.agl.ir.contracts import ContractRequest
from agm.agl.ir.ids import ContractId, FunctionId, SymbolId
from agm.agl.ir.nodes import IrExpr
from agm.agl.ir.program import ExecutableModule, FunctionDescriptor, SymbolDescriptor
from agm.agl.lower.lowerer import _LinkState
from agm.agl.syntax.resources import ResourceError, resolve_resource


@dataclass(frozen=True, slots=True)
class LoweredModule:
    """A module's IR, with stable handles for local and imported declarations."""

    module: ExecutableModule
    symbols: dict[SymbolId, SymbolDescriptor]
    functions: dict[FunctionId, FunctionDescriptor]
    contracts: dict[ContractId, ContractRequest]
    declarations: dict[int, SymbolId]
    function_symbols: dict[int, SymbolId]
    function_ids: dict[int, FunctionId]
    defaults: dict[BuiltinVarKey | str, IrExpr]
    resources: tuple[tuple[Path | None, str | None, Path], ...]

    def link_into(self, link: _LinkState) -> None:
        """Publish this module's declarations and bodies to the program linker."""
        link.symbols.update(self.symbols)
        link.functions.update(self.functions)
        link.contracts.update(self.contracts)
        link.decl_to_sym.update(self.declarations)
        link.fn_node_to_sym.update(self.function_symbols)
        link.fn_node_to_id.update(self.function_ids)


def capture(
    module: ExecutableModule,
    link: _LinkState,
    defaults: dict[BuiltinVarKey | str, IrExpr],
    resources: tuple[tuple[Path | None, str | None, Path], ...],
    contract_start: int,
    contract_end: int,
) -> LoweredModule:
    """Extract the tables owned by one freshly lowered module."""
    functions = {fid: fn for fid, fn in link.functions.items() if fn.module_id == module.module_id}
    symbols = {
        sid: sym
        for sid, sym in link.symbols.items()
        if sym.owner == module.module_id or sym.owner in functions
    }
    return LoweredModule(
        module,
        symbols,
        functions,
        {
            cid: value
            for cid, value in link.contracts.items()
            if contract_start <= cid.value < contract_end
        },
        {nid: sid for nid, sid in link.decl_to_sym.items() if sid in symbols},
        {nid: sid for nid, sid in link.fn_node_to_sym.items() if sid in symbols},
        {nid: fid for nid, fid in link.fn_node_to_id.items() if fid in functions},
        defaults,
        resources,
    )


def load(key: bytes) -> LoweredModule | None:
    """Read reusable IR, revalidating filesystem-dependent resource paths."""
    module = retained_lowered_module(key)
    if module is None:
        restored = artifact_serialization.load(key, "ir")
        if not isinstance(restored, LoweredModule):
            return None
        module = restored
        retain_lowered_module(key, module)
    try:
        for root, relative, expected in module.resources:
            if resolve_resource(root, relative) != expected:
                return None
    except ResourceError:
        return None
    return module


def save(key: bytes, module: LoweredModule) -> None:
    """Publish a successfully lowered module for later CLI processes."""
    retain_lowered_module(key, module)
    artifact_serialization.save(key, "ir", module)
