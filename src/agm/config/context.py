"""Current process context for config loading, and its per-process read memo."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from agm.core.env import resolve_env
from agm.project.layout import discover_current_project_dir


@dataclass(frozen=True)
class ConfigContext:
    """Resolved context used to load AGM configuration."""

    home: Path
    proj_dir: Path | None
    cwd: Path


def current_config_context(
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> ConfigContext:
    """Return config-loading paths for the current command invocation."""

    resolved_env = resolve_env(env)
    resolved_cwd = Path.cwd().resolve() if cwd is None else cwd.resolve()
    home = Path(resolved_env.get("HOME", "~"))

    try:
        proj_dir = discover_current_project_dir(resolved_cwd, env=resolved_env)
    except SystemExit:
        proj_dir = None

    return ConfigContext(home=home, proj_dir=proj_dir, cwd=resolved_cwd)


_memo: dict[Hashable, tuple[Hashable, object]] = {}


def context_cached[T](key: Hashable, stamp: Hashable, compute: Callable[[], T]) -> T:
    """Return the memoized value for *key*, recomputing it whenever *stamp* differs.

    Activation state and configuration are read from several independent
    entry points within one invocation, and parsing them costs far more than
    keying them. *key* identifies what is being read and *stamp* everything
    that read derives from — for a file-backed one a :func:`content_stamp` of
    its sources — so a rewrite supersedes the entry whoever made it, and a
    long-lived process holds one entry per thing it reads rather than one per
    state that thing has been in.
    """

    entry = _memo.get(key)
    if entry is not None and entry[0] == stamp:
        return cast(T, entry[1])
    value = compute()
    _memo[key] = (stamp, value)
    return value


def invalidate_context_cache() -> None:
    """Drop every memoized configuration and activation read.

    Called by whatever writes activation state, so a process that changes a
    selection reads back its own write rather than the selection it replaced.
    """

    _memo.clear()


def content_stamp(paths: Iterable[Path]) -> tuple[bytes, ...]:
    """Return one content digest per path, empty for a path that cannot be read.

    Hashing a config or index file costs a fraction of parsing it, so a memo
    key built from this needs no cooperation from whoever writes the file.
    """

    return tuple(_content_digest(path) for path in paths)


def _content_digest(path: Path) -> bytes:
    try:
        return hashlib.blake2b(path.read_bytes(), digest_size=16).digest()
    except OSError:
        return b""
