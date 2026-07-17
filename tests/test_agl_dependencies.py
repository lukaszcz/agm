"""Architectural dependency contracts for the AgL execution pipeline."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

AGL_ROOT = Path(__file__).parents[1] / "src" / "agm" / "agl"
SRC_ROOT = AGL_ROOT.parents[1]


def _agl_imports(package: str) -> list[tuple[Path, str]]:
    imports: list[tuple[Path, str]] = []
    for path in sorted((AGL_ROOT / package).rglob("*.py")):
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


def _agm_imports(package: str) -> list[tuple[Path, str]]:
    """Return every absolute import under the shared ``agm`` namespace."""
    imports: list[tuple[Path, str]] = []
    for path in sorted((AGL_ROOT / package).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    module_parts = path.relative_to(SRC_ROOT).with_suffix("").parts
                    package_parts = (
                        module_parts if path.name == "__init__.py" else module_parts[:-1]
                    )
                    keep = len(package_parts) - (node.level - 1)
                    imported_parts = (() if node.module is None else tuple(node.module.split(".")))
                    module = ".".join((*package_parts[:keep], *imported_parts))
                else:
                    module = node.module
                if module == "agm" or (module is not None and module.startswith("agm.")):
                    imports.append((path, module))
            elif isinstance(node, ast.Import):
                imports.extend(
                    (path, alias.name)
                    for alias in node.names
                    if alias.name == "agm" or alias.name.startswith("agm.")
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
                "agm.agl.ir",
                "agm.agl.lower",
                "agm.agl.matchcompile",
                "agm.agl.modules.ids",
                "agm.agl.scope",
                "agm.agl.semantics",
                "agm.agl.syntax",
                "agm.agl.type_schema",
                "agm.agl.typecheck",
            ),
        ),
        (
            "matchcompile",
            (
                "agm.agl.diagnostics",
                "agm.agl.matchcompile",
                "agm.agl.modules.ids",
                "agm.agl.scope",
                "agm.agl.semantics",
                "agm.agl.syntax",
                "agm.agl.typecheck",
            ),
        ),
        (
            "scope",
            (
                "agm.agl.diagnostics",
                "agm.agl.modules",
                "agm.agl.scope",
                "agm.agl.semantics",
                "agm.agl.syntax",
            ),
        ),
        (
            "semantics",
            ("agm.agl.ir", "agm.agl.modules.ids", "agm.agl.semantics"),
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
        (
            "runtime",
            (
                "agm.agl.capabilities",
                "agm.agl.diagnostics",
                "agm.agl.ir",
                "agm.agl.modules.ids",
                "agm.agl.runtime",
                "agm.agl.semantics",
                "agm.agl.syntax.spans",
                "agm.agl.typecheck.env",
                "agm.agl.type_schema",
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


def test_ir_all_agm_dependencies_are_explicit() -> None:
    """Keep the IR on its own data plus the canonical config-key data leaf."""
    allowed = (
        "agm.agl.ir",
        "agm.agl.modules.ids",
        "agm.config.engine_keys",
    )
    violations = [
        f"{path.relative_to(AGL_ROOT)} imports {module}"
        for path, module in _agm_imports("ir")
        if not _is_allowed(module, allowed)
    ]

    assert violations == []


def test_lower_consumes_only_matchcompile_public_contract() -> None:
    violations = [
        f"{path.relative_to(AGL_ROOT)} imports {module}"
        for path, module in _agl_imports("lower")
        if module.startswith("agm.agl.matchcompile.")
    ]

    assert violations == []


@pytest.mark.parametrize("package", ("lower", "ir", "eval", "runtime"))
def test_execution_packages_do_not_import_the_inference_engine(package: str) -> None:
    violations = [
        f"{path.relative_to(AGL_ROOT)} imports {module}"
        for path, module in _agl_imports(package)
        if module == "agm.agl.typecheck.inference"
        or module.startswith("agm.agl.typecheck.inference.")
    ]

    assert violations == []
