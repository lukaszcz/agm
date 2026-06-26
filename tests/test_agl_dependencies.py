"""Architectural dependency contracts for the AgL execution pipeline."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

AGL_ROOT = Path(__file__).parents[1] / "src" / "agm" / "agl"


def _agl_imports(package: str) -> list[tuple[Path, str]]:
    imports: list[tuple[Path, str]] = []
    for path in sorted((AGL_ROOT / package).glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                if node.module.startswith("agm.agl."):
                    imports.append((path, node.module))
            elif isinstance(node, ast.Import):
                imports.extend(
                    (path, alias.name)
                    for alias in node.names
                    if alias.name.startswith("agm.agl.")
                )
    return imports


def _is_allowed(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == prefix or module.startswith(f"{prefix}.") for prefix in prefixes)


@pytest.mark.parametrize(
    ("package", "allowed"),
    [
        ("ir", ("agm.agl.ir", "agm.agl.modules.ids")),
        (
            "lower",
            (
                "agm.agl._text",
                "agm.agl.ir",
                "agm.agl.lower",
                "agm.agl.modules.ids",
                "agm.agl.scope",
                "agm.agl.syntax",
                "agm.agl.type_schema",
                "agm.agl.typecheck",
            ),
        ),
        (
            "eval",
            (
                "agm.agl.eval",
                "agm.agl.ir",
                "agm.agl.modules.ids",
                "agm.agl.runtime",
                "agm.agl.semantics",
            ),
        ),
    ],
)
def test_execution_package_dependency_contract(
    package: str, allowed: tuple[str, ...]
) -> None:
    violations = [
        f"{path.relative_to(AGL_ROOT)} imports {module}"
        for path, module in _agl_imports(package)
        if not _is_allowed(module, allowed)
    ]

    assert violations == []
