"""Parsed-module persistence over the shared AgL artifact storage envelope.

Validates a parsed module by source content, node-id base, and default-stdlib
flag, beyond the storage envelope's compiler-digest identity
(``agm.agl.artifact_storage``). A syntax-only unpickler restricts restored
objects to the AST's dataclasses and enums.
"""

from __future__ import annotations

import io
import pickle
from dataclasses import is_dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, cast

from agm.agl.artifact_storage import artifact_entry, read_payload, write_payload

if TYPE_CHECKING:
    from agm.agl.modules.ids import ModuleId
    from agm.agl.modules.loader import LoadedModule


_SYNTAX_CLASSES: dict[tuple[str, str], type[object]] | None = None


def _syntax_classes() -> dict[tuple[str, str], type[object]]:
    """Allow only the data classes and scalar constructors a parsed module uses."""
    global _SYNTAX_CLASSES
    if _SYNTAX_CLASSES is not None:
        return _SYNTAX_CLASSES
    from agm.agl.modules.ids import ModuleId
    from agm.agl.modules.loader import LoadedModule
    from agm.agl.syntax import advisories, nodes, spans, types

    class_type = cast(type[object], type)
    classes: list[type[object]] = [LoadedModule, ModuleId, type(Path()), Decimal]
    for module in (advisories, nodes, spans, types):
        for value in cast(dict[str, object], vars(module)).values():
            if isinstance(value, class_type):
                cls = cast(type[object], value)
                if is_dataclass(cls) or issubclass(cls, Enum):
                    classes.append(cls)
    _SYNTAX_CLASSES = {(cls.__module__, cls.__name__): cls for cls in classes}
    return _SYNTAX_CLASSES


class _SyntaxUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> type[object]:
        try:
            return _syntax_classes()[module, name]
        except KeyError as exc:
            raise pickle.UnpicklingError("not a syntax data class") from exc


def _entry(
    module_id: ModuleId, path: Path, source: str, start_id: int, default_stdlib: bool
) -> tuple[Path, bytes]:
    key = repr((str(path), module_id.segments, start_id, default_stdlib)).encode()
    return artifact_entry(key, "cache", validator=source.encode())


def load(
    module_id: ModuleId, path: Path, source: str, start_id: int, default_stdlib: bool
) -> tuple[LoadedModule, int] | None:
    """Read a compatible artifact; an absent or damaged cache is a miss."""
    from agm.agl.modules.loader import LoadedModule

    try:
        entry, identity = _entry(module_id, path, source, start_id, default_stdlib)
        payload = read_payload(entry, identity)
        if payload is None:
            return None
        value = cast(object, _SyntaxUnpickler(io.BytesIO(payload)).load())
        if not isinstance(value, tuple) or len(value) != 2:
            return None
        module, next_id = cast(tuple[object, object], value)
        if not isinstance(module, LoadedModule) or not isinstance(next_id, int):
            return None
        if module.companion_path is not None and not module.companion_path.is_file():
            return None
        return module, next_id
    except (OSError, EOFError, pickle.UnpicklingError, ValueError, TypeError, AttributeError):
        return None


def save(module: LoadedModule, next_id: int, *, start_id: int, default_stdlib: bool) -> None:
    """Atomically replace an entry; cache I/O must never prevent compilation."""
    try:
        path = cast(Path, module.path)
        entry, identity = _entry(
            module.module_id, path, module.source_text, start_id, default_stdlib
        )
        write_payload(entry, identity, pickle.dumps((module, next_id), protocol=5))
    except OSError:
        pass
