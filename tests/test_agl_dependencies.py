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
                    (path, alias.name) for alias in node.names if alias.name.startswith("agm.agl.")
                )
    return imports


def _imported_modules(path: Path, node: ast.Import | ast.ImportFrom) -> tuple[str, ...]:
    """Return the absolute module names one import node in ``path`` names.

    A relative import resolves against the importing module's own package, so
    ``from . import zones`` cannot hide a dependency from a contract that reads
    module names.
    """
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if not node.level:
        return () if node.module is None else (node.module,)
    module_parts = path.relative_to(SRC_ROOT).with_suffix("").parts
    package_parts = module_parts if path.name == "__init__.py" else module_parts[:-1]
    keep = len(package_parts) - (node.level - 1)
    imported_parts = () if node.module is None else tuple(node.module.split("."))
    return (".".join((*package_parts[:keep], *imported_parts)),)


def _is_agm(module: str) -> bool:
    return module == "agm" or module.startswith("agm.")


def _agm_imports(package: str) -> list[tuple[Path, str]]:
    """Return every import under the shared ``agm`` namespace, relatives included."""
    imports: list[tuple[Path, str]] = []
    for path in sorted((AGL_ROOT / package).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.extend(
                    (path, module) for module in _imported_modules(path, node) if _is_agm(module)
                )
    return imports


def _is_allowed(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == prefix or module.startswith(f"{prefix}.") for prefix in prefixes)


@pytest.mark.parametrize(
    ("package", "allowed"),
    [
        ("ir", ("agm.agl.ir", "agm.agl.modules.ids", "agm.agl.zones")),
        (
            "lower",
            (
                "agm.agl.ir",
                "agm.agl.lower",
                "agm.agl.matchcompile",
                "agm.agl.modules.ids",
                "agm.agl.scope",
                "agm.agl.self_validation",
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
                "agm.agl.self_validation",
                "agm.agl.semantics",
                "agm.agl.syntax",
                "agm.agl.typecheck",
            ),
        ),
        (
            "scope",
            (
                "agm.agl.attributes",
                "agm.agl.diagnostics",
                "agm.agl.artifact_cache",
                "agm.agl.modules",
                "agm.agl.scope",
                "agm.agl.semantics",
                "agm.agl.syntax",
                "agm.agl.zones",
            ),
        ),
        (
            "typecheck",
            (
                "agm.agl.capabilities",
                "agm.agl.diagnostics",
                "agm.agl.ir.ids",
                "agm.agl.ir.reserved_nominals",
                "agm.agl.artifact_cache",
                "agm.agl.modules.ids",
                "agm.agl.scope",
                "agm.agl.self_validation",
                "agm.agl.semantics",
                "agm.agl.syntax",
                "agm.agl.typecheck",
                "agm.agl.zones",
            ),
        ),
        (
            "semantics",
            (
                "agm.agl.ir",
                "agm.agl.modules.ids",
                "agm.agl.self_validation",
                "agm.agl.semantics",
                "agm.agl.zones",
            ),
        ),
        (
            "syntax",
            (
                "agm.agl.modules.ids",
                "agm.agl.syntax",
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
        (
            "runtime",
            (
                "agm.agl.capabilities",
                "agm.agl.diagnostics",
                "agm.agl.ir",
                "agm.agl.modules.ids",
                "agm.agl.runtime",
                "agm.agl.self_validation",
                "agm.agl.semantics",
                "agm.agl.syntax.spans",
                "agm.agl.typecheck.env",
                "agm.agl.type_schema",
                "agm.agl.zones",
            ),
        ),
    ],
)
def test_execution_package_dependency_contract(package: str, allowed: tuple[str, ...]) -> None:
    """Keep each pass on the layers below it.

    ``agm.agl.artifact_cache`` is admitted into the scope and type-check passes
    because it is a leaf: it imports nothing at run time, so consulting it
    couples a pass to no other layer.
    """
    violations = [
        f"{path.relative_to(AGL_ROOT)} imports {module}"
        for path, module in _agl_imports(package)
        if not _is_allowed(module, allowed)
    ]

    assert violations == []


def _agm_imports_of_file(path: Path) -> list[str]:
    """Return every ``agm`` module one file imports, relatives resolved."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        module
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for module in _imported_modules(path, node)
        if _is_agm(module)
    ]


@pytest.mark.parametrize(
    ("leaf", "allowed"),
    [
        ("zones.py", ()),
        ("attributes.py", ("agm.agl.zones",)),
    ],
)
def test_vocabulary_leaves_sit_below_every_pass(leaf: str, allowed: tuple[str, ...]) -> None:
    """Keep the shared vocabulary modules importable from every layer.

    ``zones`` is the bottom leaf and imports nothing under ``agm``;
    ``attributes`` names the zones its ``@arg-*`` entries select and so may
    import that one module, and nothing else.
    """
    violations = [
        f"{leaf} imports {module}"
        for module in _agm_imports_of_file(AGL_ROOT / leaf)
        if module not in allowed
    ]

    assert violations == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("from . import zones", "agm.agl"),
        ("from .syntax import Attribute", "agm.agl.syntax"),
        ("from ..config import engine_keys", "agm.config"),
    ],
)
def test_a_relative_import_in_a_leaf_would_be_caught(source: str, expected: str) -> None:
    """A leaf cannot hide a dependency behind a relative import."""
    node = ast.parse(f"{source}\n").body[0]
    assert isinstance(node, ast.ImportFrom)
    resolved = _imported_modules(AGL_ROOT / "attributes.py", node)

    assert resolved == (expected,)
    assert _is_agm(expected)


def test_ir_all_agm_dependencies_are_explicit() -> None:
    """Keep the IR on its own data plus the canonical config-key data leaf."""
    allowed = (
        "agm.agl.ir",
        "agm.agl.modules.ids",
        "agm.agl.zones",
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


#: Only prelude injection may key on the prelude module's identity: the host
#: recognizes a standard built-in declaration by the ``builtin`` keyword in any
#: standard-library module, never by that module being ``std/prelude``.
#: ``modules/ids.py`` defines the id and ``modules/loader.py`` injects the
#: synthetic import (and decides which module supersedes or forgoes it).
_PRELUDE_ID_ALLOWLIST: frozenset[str] = frozenset(
    {
        "modules/ids.py",
        "modules/loader.py",
    }
)


def test_prelude_module_identity_is_used_only_for_prelude_injection() -> None:
    """Nothing outside prelude injection may ask which module is the prelude."""
    violations = sorted(
        str(path.relative_to(AGL_ROOT))
        for path in AGL_ROOT.rglob("*.py")
        if "STD_PRELUDE_ID" in path.read_text(encoding="utf-8")
        and str(path.relative_to(AGL_ROOT)) not in _PRELUDE_ID_ALLOWLIST
    )

    assert violations == []
