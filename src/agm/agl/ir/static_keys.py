"""Structured identities for host-addressable static bindings."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias

from agm.agl.modules.ids import ModuleId

StaticBindingKey: TypeAlias = tuple[ModuleId, tuple[str, ...], str]
"""A static binding identity: its owning module, scope path, and name."""


def static_binding_key(
    module_id: ModuleId, scope_path: Iterable[str], name: str
) -> StaticBindingKey:
    """Build a structured static binding identity."""
    return module_id, tuple(scope_path), name
