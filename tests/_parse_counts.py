"""Counting the modules a flow actually hands to the AgL parser.

Every AgL source is meant to reach the parser once per process: the module
loader, package command discovery, and package discipline validation all serve
their parses from one cache. Tests that guard that share the counter here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from agm.agl.modules import loader
from agm.agl.modules.ids import ModuleId
from agm.agl.modules.parsed_module_cache import clear_parsed_module_cache

__all__ = ["parse_counts"]


@contextmanager
def parse_counts(monkeypatch: pytest.MonkeyPatch) -> Iterator[Counter[str]]:
    """Count parses of each module path while the block runs.

    The parsed-module cache is cleared on entry so the count is of this block's
    own work rather than of whatever an earlier test left warm.
    """
    counts: Counter[str] = Counter()
    parse = loader._parse_imported_module

    def counting(
        module_id: ModuleId,
        path: Path,
        start_id: int,
        source_text: str,
        *,
        default_stdlib: bool,
    ) -> tuple[loader.LoadedModule, int]:
        counts[str(path)] += 1
        return parse(module_id, path, start_id, source_text, default_stdlib=default_stdlib)

    monkeypatch.setattr(loader, "_parse_imported_module", counting)
    clear_parsed_module_cache()
    yield counts
