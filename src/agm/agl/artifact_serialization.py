"""Data-only persistence for compiler artifacts, with current-source anchors."""

from __future__ import annotations

import io
import pickle
import sys
from dataclasses import is_dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import cast

from immutables import Map

from agm.agl.modules.disk_cache import artifact_entry, read_payload, write_payload

# These mutable compiler tables contain data, never host callbacks. All other
# permitted globals must be dataclasses or enums in the compiler's data layers.
_TABLE_CLASSES = frozenset(
    {
        ("agm.agl.semantics.type_table", "TypeTable"),
        ("agm.agl.semantics.persistent", "PersistentDict"),
        ("agm.agl.typecheck.env", "TypeEnvironment"),
        ("agm.agl.typecheck.program", "_DeclKeyDict"),
    }
)
_DATA_MODULES = (
    "agm.agl.syntax.",
    "agm.agl.scope.",
    "agm.agl.semantics.",
    "agm.agl.typecheck.",
    "agm.agl.matchcompile.",
    "agm.agl.ir.",
)
_DATA_LEAVES = frozenset(
    {
        "agm.agl.modules.ids",
        "agm.agl.modules.loader",
        "agm.agl.capabilities",
        "agm.agl.diagnostics",
        "agm.agl.attributes",
        "agm.agl.zones",
        "agm.agl.lower.module",
    }
)
_SCALARS: dict[tuple[str, str], type[object]] = {
    (cls.__module__, cls.__name__): cls for cls in (type(Path()), Decimal, dict, Map)
}


class _Writer(pickle.Pickler):
    def __init__(self, stream: io.BytesIO, anchors: tuple[object, ...]) -> None:
        super().__init__(stream, protocol=5)
        self.anchors = {id(value): index for index, value in enumerate(anchors)}

    def persistent_id(self, obj: object) -> int | None:
        return self.anchors.get(id(obj))

    def reducer_override(self, obj: object) -> object:
        if isinstance(obj, MappingProxyType):
            return dict, (dict(cast(MappingProxyType[object, object], obj)),)
        return cast(object, NotImplemented)


class _Reader(pickle.Unpickler):
    def __init__(self, stream: io.BytesIO, anchors: tuple[object, ...]) -> None:
        super().__init__(stream)
        self.anchors = anchors

    def persistent_load(self, pid: object) -> object:
        if not isinstance(pid, int) or not 0 <= pid < len(self.anchors):
            raise pickle.UnpicklingError("invalid source anchor")
        return self.anchors[pid]

    def find_class(self, module: str, name: str) -> object:
        scalar = _SCALARS.get((module, name))
        if scalar is not None:
            return scalar
        # Never import a module named by disk contents. Compilation has already
        # imported the data layers needed by the stage being restored.
        value = cast(object, getattr(sys.modules.get(module), name, None))
        if isinstance(value, cast(type[object], type)) and (
            (module, name) in _TABLE_CLASSES
            or (
                (module.startswith(_DATA_MODULES) or module in _DATA_LEAVES)
                and (is_dataclass(value) or issubclass(cast(type[object], value), Enum))
            )
        ):
            return value
        raise pickle.UnpicklingError(f"not compiler data: {module}.{name}")


def load(key: bytes, kind: str, anchors: tuple[object, ...] = ()) -> object:
    """Restore a compatible artifact using this compilation's source objects."""
    try:
        payload = read_payload(*artifact_entry(key, kind))
        if payload is None:
            return None
        return cast(object, _Reader(io.BytesIO(payload), anchors).load())
    except (OSError, EOFError, pickle.UnpicklingError, ValueError, TypeError, AttributeError):
        return None


def save(key: bytes, kind: str, value: object, anchors: tuple[object, ...] = ()) -> None:
    """Persist compiler data; unsupported or unavailable storage is a miss."""
    try:
        stream = io.BytesIO()
        _Writer(stream, anchors).dump(value)
        write_payload(*artifact_entry(key, kind), stream.getvalue())
    except (OSError, pickle.PicklingError, TypeError, AttributeError):
        pass
