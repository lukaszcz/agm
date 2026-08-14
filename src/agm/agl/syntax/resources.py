"""Static validation and link-time resolution of resource builtin paths."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from agm.agl.syntax.nodes import Call, StringLit
from agm.agl.syntax.spans import SourceSpan
from agm.core.path import is_portable_relative_path


class ResourceError(ValueError):
    """Raised when a resource call cannot be safely resolved."""

    def __init__(self, message: str, *, span: SourceSpan | None = None) -> None:
        super().__init__(message)
        self.span = span


def resource_path(call: Call, *, is_directory: bool) -> str | None:
    """Validate *call* syntax and return its relative path, if it has one."""
    if is_directory:
        if call.type_args or call.args or call.named_args:
            raise ResourceError("resource-dir() accepts no arguments")
        return None
    if (
        call.type_args
        or len(call.args) != 1
        or call.named_args
        or not isinstance(call.args[0], StringLit)
    ):
        raise ResourceError("resource() requires exactly one text literal argument")
    path = call.args[0].value
    _validate_relative_path(path)
    return path


def resolve_resource(anchor: Path | None, relative_path: str | None) -> Path:
    """Resolve a checked resource under *anchor*, requiring an existing target."""
    if anchor is None:
        raise ResourceError("resource calls require a file-backed declaring module")
    root = anchor.resolve()
    target = root if relative_path is None else (root / PurePosixPath(relative_path)).resolve()
    if not target.is_relative_to(root):
        raise ResourceError(f"resource target escapes its anchor directory: {relative_path!r}")
    if not target.exists():
        raise ResourceError(f"resource target does not exist: {target}")
    return target


def _validate_relative_path(path: str) -> None:
    if not is_portable_relative_path(path):
        raise ResourceError("resource() path must be a relative forward-slash path without '..'")
