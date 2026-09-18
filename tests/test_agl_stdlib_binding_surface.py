"""Guard test: the standard library's exported module-level binding surface.

A module-level simple ``let``/``var`` of any std module is exported (see
``agm.agl.syntax.nodes.exported_binding_name``), so adding one silently
changes what an importer can read or write. This asserts the exact reviewed
set, and that none of it leaks into ``std/prelude``'s bare surface.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from agm.agl.parser import parse_program
from agm.agl.scope.symbols import AglScopeError
from agm.agl.syntax.nodes import Item, LetDecl, ScopeRegion, VarDecl, exported_binding_name
from agm.packages.layout import MODULE_TREE_DIRNAME
from tests._agl_helpers import REPO_STDLIB_ROOT
from tests.agl.module_graph import _DEFAULT_CAPABILITIES, resolve_and_check_inline_entry

STDLIB_MODULES_DIR = REPO_STDLIB_ROOT / MODULE_TREE_DIRNAME

# Every std module's own exported module-level let/var names, keyed by full
# ``::``-qualified scope path, reviewed and pinned: a new one here is a
# deliberate public-surface change.
EXPECTED_STDLIB_BINDING_SURFACE = {
    "math": frozenset({"pi", "e"}),
    "log": frozenset({"level"}),
}


def _qualified_exported_bindings(
    items: tuple[Item, ...], enclosing_path: tuple[str, ...]
) -> Iterator[str]:
    """Yield ``::``-qualified exported let/var names, descending into scope regions."""
    for item in items:
        if isinstance(item, ScopeRegion):
            yield from _qualified_exported_bindings(
                item.items, enclosing_path + (item.segment.name,)
            )
        elif isinstance(item, (LetDecl, VarDecl)):
            name = exported_binding_name(item)
            if name is not None:
                path = enclosing_path + tuple(seg.name for seg in item.scope_path) + (name,)
                yield "::".join(path)


def _exported_bindings(module_name: str) -> frozenset[str]:
    """Exported let/var names of one std module, via the real syntax layer."""
    source = (STDLIB_MODULES_DIR / f"{module_name}.agl").read_text(encoding="utf-8")
    program = parse_program(source, filename=f"{module_name}.agl")
    return frozenset(_qualified_exported_bindings(program.body.items, ()))


def test_stdlib_module_exported_bindings_match_the_reviewed_surface() -> None:
    module_names = sorted(p.stem for p in STDLIB_MODULES_DIR.glob("*.agl"))
    actual = {name: bindings for name in module_names if (bindings := _exported_bindings(name))}
    assert actual == EXPECTED_STDLIB_BINDING_SURFACE


@pytest.mark.parametrize(
    "name", sorted({name for names in EXPECTED_STDLIB_BINDING_SURFACE.values() for name in names})
)
def test_prelude_does_not_re_export_any_stdlib_module_level_binding(name: str) -> None:
    """None of the std library's own exported bindings reach a bare name
    through ``std/prelude``, confirmed against the real resolver rather than
    by reading ``prelude.agl``'s export lines."""
    with pytest.raises(AglScopeError):
        resolve_and_check_inline_entry(name, _DEFAULT_CAPABILITIES)
