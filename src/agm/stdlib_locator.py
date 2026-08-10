"""Locate the standard library shipped with AGM.

Source checkouts keep ``stdlib/`` at the repository root, while built wheels
install the same tree inside the ``agm`` package.  Runtime resolution and the
managed-package installer share this locator so both contexts select the same
shipped artifact.
"""

from __future__ import annotations

from pathlib import Path


def shipped_stdlib_root() -> Path:
    """Return AGM's bundled or source-checkout standard-library root."""
    package_root = Path(__file__).resolve().parent
    bundled_root = package_root / "stdlib"
    if bundled_root.is_dir():
        return bundled_root
    return package_root.parents[1] / "stdlib"
