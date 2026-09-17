"""Initialize a package directory for ``agm pkg init``."""

from __future__ import annotations

import sys
from pathlib import Path

from agm.cli_support.args import PkgInitArgs
from agm.core import dry_run, fs
from agm.core.toml import dumps_toml, empty_toml_doc, set_toml_table_value
from agm.packages.discipline import DisciplineError, validate_unreserved_package_name
from agm.packages.distribution import MANIFEST_NAME
from agm.packages.layout import MODULE_TREE_DIRNAME
from agm.packages.manifest import ManifestError, load_manifest_text

STARTER_MODULE_NAME = "main.agl"
STARTER_MODULE_SOURCE = "program def main() -> unit = ()\n"


def run(args: PkgInitArgs) -> None:
    """Write a manifest and starter module into a new or existing directory."""

    root = Path.cwd() if args.directory is None else Path(args.directory)
    try:
        content = render_manifest(
            name=root.resolve().name if args.name is None else args.name, version=args.version
        )
        manifest = load_manifest_text(content)
        validate_unreserved_package_name(manifest.name)
        manifest_path = root / MANIFEST_NAME
        if fs.exists(manifest_path):
            raise ManifestError(f"{manifest_path} already exists")
        module_path = root / MODULE_TREE_DIRNAME / STARTER_MODULE_NAME
        created = (manifest_path,) if fs.exists(module_path) else (manifest_path, module_path)
        if dry_run.enabled():
            for path in created:
                dry_run.print_operation("init-package", str(path))
            return
        fs.mkdir(module_path.parent, parents=True, exist_ok=True)
        fs.write_text(manifest_path, content)
        if module_path in created:
            fs.write_text(module_path, STARTER_MODULE_SOURCE)
    except (DisciplineError, ManifestError, OSError) as exc:
        print(f"pkg init: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    for path in created:
        print(path)


def render_manifest(*, name: str, version: str) -> str:
    """Render the manifest text of a freshly initialized package."""

    document = empty_toml_doc()
    set_toml_table_value(document, "package", "name", name)
    set_toml_table_value(document, "package", "version", version)
    return dumps_toml(document)
