"""Atomic, versioned compiler-artifact storage and parsed-module persistence.

The storage envelope validates compiler/dependency versions and payload integrity.
The parsed-module API accepts syntax data only; later stages use the data-only
serializer in ``artifact_serialization`` over the same disposable storage.
"""

from __future__ import annotations

import hashlib
import io
import os
import pickle
import sys
from contextlib import suppress
from dataclasses import is_dataclass
from decimal import Decimal
from enum import Enum
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from agm.agl.modules.ids import ModuleId
    from agm.agl.modules.loader import LoadedModule


_COMPILER_DIGEST: bytes | None = None
_SYNTAX_CLASSES: dict[tuple[str, str], type[object]] | None = None


def _compiler_digest() -> bytes:
    """Invalidate artifacts when AGM code, grammar, Python, or the parser dependency changes."""
    global _COMPILER_DIGEST
    if _COMPILER_DIGEST is not None:
        return _COMPILER_DIGEST
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256((sys.version + version("lark") + version("immutables")).encode())
    for path in sorted(root.rglob("*")):
        if path.suffix in {".py", ".lark"}:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    _COMPILER_DIGEST = digest.digest()
    return _COMPILER_DIGEST


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
    entry, _ = artifact_entry(key, "cache")
    identity = hashlib.sha256(_compiler_digest() + key + source.encode()).digest()
    return entry, identity


def artifact_entry(key: bytes, kind: str) -> tuple[Path, bytes]:
    """Locate a versioned compiler artifact in the user's disposable cache."""
    cache_home = os.environ.get("XDG_CACHE_HOME")
    root = Path(cache_home) if cache_home else Path.home() / ".cache"
    slot = hashlib.sha256(key).hexdigest()
    identity = hashlib.sha256(_compiler_digest() + key).digest()
    return root / "agm" / "agl" / (slot + "." + kind), identity


def read_payload(entry: Path, identity: bytes) -> bytes | None:
    """Read a complete, compatible payload, or report a cache miss."""
    try:
        data = entry.read_bytes()
        payload = data[64:]
        if data[:32] != identity or data[32:64] != hashlib.sha256(payload).digest():
            return None
        return payload
    except OSError:
        return None


def write_payload(entry: Path, identity: bytes, payload: bytes) -> None:
    """Publish a complete payload atomically; unavailable storage is harmless."""
    temporary: Path | None = None
    try:
        entry.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=entry.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(identity + hashlib.sha256(payload).digest() + payload)
        temporary.replace(entry)
    except OSError:
        pass
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


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
