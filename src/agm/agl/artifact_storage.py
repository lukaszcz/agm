"""Atomic, versioned compiler-artifact storage envelope.

Locates a slot for a caller-chosen key under the disposable AgL cache, keys
identity by compiler digest plus that key and an optional validator, and
reads/writes payloads atomically. Shared by the AgL disk caches; imports
nothing under ``agm``.
"""

from __future__ import annotations

import hashlib
import os
import sys
from contextlib import suppress
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile

_COMPILER_DIGEST: bytes | None = None


def _compiler_digest() -> bytes:
    """Invalidate artifacts when AGM code, grammar, Python, or the parser dependency changes."""
    global _COMPILER_DIGEST
    if _COMPILER_DIGEST is not None:
        return _COMPILER_DIGEST
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256((sys.version + version("lark") + version("immutables")).encode())
    for path in sorted(root.rglob("*")):
        if path.suffix in {".py", ".lark"}:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    _COMPILER_DIGEST = digest.digest()
    return _COMPILER_DIGEST


def artifact_entry(key: bytes, kind: str, *, validator: bytes = b"") -> tuple[Path, bytes]:
    """Locate a versioned compiler artifact in the user's disposable cache.

    *validator* extends the identity without moving the slot, for a caller
    whose reuse condition is not fully captured by *key* alone.
    """
    cache_home = os.environ.get("XDG_CACHE_HOME")
    root = Path(cache_home) if cache_home else Path.home() / ".cache"
    slot = hashlib.sha256(key).hexdigest()
    identity = hashlib.sha256(_compiler_digest() + key + validator).digest()
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
