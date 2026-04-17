"""Install AGM user configuration files."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast


class _InstallArgs(Protocol):
    force: bool


@dataclass(frozen=True)
class InstallUserConfigResult:
    installed: list[Path]
    skipped: list[Path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python tools/install_agm_config.py")
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing files under ~/.agm.",
    )
    return parser


def _install_file(*, source: Path, destination: Path, force: bool) -> bool:
    if destination.exists() and not force:
        return False
    shutil.copy2(source, destination)
    return True


def install_user_config(
    *,
    repo_root: Path,
    home: Path,
    force: bool = False,
) -> InstallUserConfigResult:
    agm_config_dir = home / ".agm"
    sandbox_dir = agm_config_dir / "sandbox"
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    installed: list[Path] = []
    skipped: list[Path] = []

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

    return InstallUserConfigResult(installed=installed, skipped=skipped)


def main(argv: list[str] | None = None) -> int:
    args = cast(_InstallArgs, build_parser().parse_args(argv))
    result = install_user_config(
        repo_root=Path(__file__).resolve().parents[1],
        home=Path.home(),
        force=args.force,
    )
    for path in result.installed:
        print(f"Installed {path}")
    for path in result.skipped:
        print(f"Skipped {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
