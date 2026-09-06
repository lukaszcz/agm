"""Pure package-store layout constants.

This is a leaf module with no imports from ``agm.config`` or the rest of the
package domain, so configuration loading can depend on it without creating an
import cycle back into ``agm.packages``. Every path here is computed from an
already-resolved AGM home; resolving that home (``AGM_HOME``, an installation
prefix, or the project-relative default) stays the responsibility of
``agm.config.general.agm_home_dir``.
"""

from __future__ import annotations

from pathlib import Path

STORE_DIRNAME = "packages"
"""The package-store directory name directly beneath an AGM home."""

ACTIVATION_INDEX_FILENAME = "index.toml"
"""The activation-index file name directly beneath the package store."""

MODULE_TREE_DIRNAME = "src"
"""The directory holding a package's AgL module tree, directly beneath its root.

Fixed for every package, so a module id's leading segment names the package
while the rest of the id is a path beneath this directory.
"""


def store_root_path(agm_home: Path) -> Path:
    """Return the package-store root beneath an already-resolved AGM home."""

    return agm_home / STORE_DIRNAME


def activation_index_path(agm_home: Path) -> Path:
    """Return the activation-index path beneath an already-resolved AGM home."""

    return store_root_path(agm_home) / ACTIVATION_INDEX_FILENAME
