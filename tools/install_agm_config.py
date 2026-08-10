"""Install AGM user configuration files."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from agm.config.general import agm_home_dir
from agm.packages.install import install_directory
from agm.packages.manifest import load_manifest
from agm.packages.record import verify_record, write_record
from agm.packages.store import canonical_package_store_path


class _InstallArgs(Protocol):
    force: bool
    prefix: str | None


@dataclass(frozen=True)
class InstallUserConfigResult:
    installed: list[Path]
    skipped: list[Path]
    pruned: list[Path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python tools/install_agm_config.py")
    parser.add_argument(
        "prefix",
        nargs="?",
        help="Install AGM config files under PREFIX/.agm instead of $HOME/.agm.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing config files at the destination.",
    )
    return parser


def _install_file(*, source: Path, destination: Path, force: bool) -> bool:
    if destination.exists() and not force:
        return False
    shutil.copy2(source, destination)
    return True


def _path_depth(path: Path) -> int:
    """Return *path*'s depth (its number of components), for deepest-first sorting."""
    return len(path.parts)


def _prepare_managed_destination(destination_dir: Path) -> None:
    """Create *destination_dir*, refusing trees that contain symlinks."""
    if destination_dir.is_symlink():
        raise RuntimeError(f"Refusing to refresh symlinked directory: {destination_dir}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    symlink = next(
        (path for path in destination_dir.rglob("*") if path.is_symlink()),
        None,
    )
    if symlink is not None:
        raise RuntimeError(f"Refusing to refresh directory containing symlink: {symlink}")


def _install_tree_files(
    *,
    source_dir: Path,
    destination_dir: Path,
    force: bool,
    installed: list[Path],
    skipped: list[Path],
    pruned: list[Path] | None = None,
) -> None:
    """Copy every file under *source_dir* into *destination_dir*.

    When *pruned* is given, *destination_dir* is additionally made an exact
    mirror of *source_dir*: any file under *destination_dir* with no
    corresponding file under *source_dir* is removed (appended to *pruned*),
    followed by any directory left empty as a result. Both the copy and the
    prune walk are confined strictly to *destination_dir* — nothing outside
    it is ever touched. Used for managed-artifact trees (the stdlib) that
    AGM owns outright, as opposed to user-editable config the caller may
    have customized.
    """
    source_relatives: set[Path] = set()
    for source in sorted(source_dir.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(source_dir)
        source_relatives.add(relative)
        destination = destination_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if _install_file(source=source, destination=destination, force=force):
            installed.append(destination)
        else:
            skipped.append(destination)

    if pruned is None:
        return

    for destination in sorted(destination_dir.rglob("*")):
        if not destination.is_file():
            continue
        stale = destination.relative_to(destination_dir) not in source_relatives
        if stale:
            destination.unlink()
            pruned.append(destination)

    # Remove directories left empty by pruning, deepest first.
    stale_dirs = [d for d in destination_dir.rglob("*") if d.is_dir()]
    stale_dirs.sort(key=_path_depth, reverse=True)
    for directory in stale_dirs:
        try:
            directory.rmdir()
        except OSError:
            pass  # not empty (still holds files that were never pruned)


def install_user_config(
    *,
    repo_root: Path,
    install_root: Path,
    force: bool = False,
) -> InstallUserConfigResult:
    agm_config_dir = agm_home_dir(home=install_root)
    sandbox_dir = agm_config_dir / "sandbox"
    prompts_dir = agm_config_dir / "prompts"
    micro_syntax_dir = install_root / ".config" / "micro" / "syntax"
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    installed: list[Path] = []
    skipped: list[Path] = []
    pruned: list[Path] = []

    config_source = repo_root / "config" / "config.toml"
    config_destination = agm_config_dir / "config.toml"
    if _install_file(source=config_source, destination=config_destination, force=force):
        installed.append(config_destination)
    else:
        skipped.append(config_destination)

    for sandbox_source in sorted((repo_root / "config" / "sandbox").iterdir()):
        if not sandbox_source.is_file():
            continue
        sandbox_destination = sandbox_dir / sandbox_source.name
        if _install_file(source=sandbox_source, destination=sandbox_destination, force=force):
            installed.append(sandbox_destination)
        else:
            skipped.append(sandbox_destination)

    for prompt_source in sorted((repo_root / "prompts").iterdir()):
        if not prompt_source.is_file():
            continue
        prompt_destination = prompts_dir / prompt_source.name
        if _install_file(source=prompt_source, destination=prompt_destination, force=force):
            installed.append(prompt_destination)
        else:
            skipped.append(prompt_destination)

    # The installed stdlib is one managed, immutable store package. The
    # package domain owns activation and its index; this installer owns its
    # force-refresh, pruning, and destination symlink refusal.
    stdlib_source = repo_root / "stdlib"
    manifest = load_manifest(stdlib_source / "package.toml")
    store_destination = canonical_package_store_path(
        manifest.name, manifest.version, home=install_root
    )
    _prepare_managed_destination(store_destination)
    _install_tree_files(
        source_dir=stdlib_source,
        destination_dir=store_destination,
        force=True,
        installed=installed,
        skipped=skipped,
        pruned=pruned,
    )
    write_record(store_destination)
    verify_record(store_destination)
    install_directory(stdlib_source, home=install_root)

    micro_source_dir = repo_root / "config" / "micro"
    if micro_source_dir.exists():
        _install_tree_files(
            source_dir=micro_source_dir,
            destination_dir=micro_syntax_dir,
            force=force,
            installed=installed,
            skipped=skipped,
        )

    return InstallUserConfigResult(installed=installed, skipped=skipped, pruned=pruned)


def main(argv: list[str] | None = None) -> int:
    args = cast(_InstallArgs, build_parser().parse_args(argv))
    result = install_user_config(
        repo_root=Path(__file__).resolve().parents[1],
        install_root=Path.home() if args.prefix is None else Path(args.prefix),
        force=args.force,
    )
    for path in result.installed:
        print(f"Installed {path}")
    for path in result.skipped:
        print(f"Skipped {path}")
    for path in result.pruned:
        print(f"Removed {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
