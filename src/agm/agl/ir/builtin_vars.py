"""Identifiers for host-backed ``builtin var`` bindings."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias

from agm.agl.ir.static_keys import StaticBindingKey, static_binding_key
from agm.agl.modules.ids import STD_CONFIG_ID, ModuleId
from agm.config.engine_keys import ENGINE_KEY_NAMES

BuiltinVarKey: TypeAlias = StaticBindingKey
"""A host-backed binding identity: its owning module, scope path, and name."""


def builtin_var_key(module_id: ModuleId, scope_path: Iterable[str], name: str) -> BuiltinVarKey:
    """Build a structured host-backed binding identity."""
    return static_binding_key(module_id, scope_path, name)


def is_engine_builtin_var_key(key: BuiltinVarKey) -> bool:
    """Whether *key* names a root ``std/config`` engine-setting binding."""
    module_id, scope_path, name = key
    return module_id == STD_CONFIG_ID and not scope_path and name in ENGINE_KEY_NAMES
