"""Install AGM user configuration files."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from agm.config.general import agm_home_dir
from agm.packages.install import refresh_managed_stdlib


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


def _install_tree_files(
    *,
    source_dir: Path,
    destination_dir: Path,
    force: bool,
    installed: list[Path],
    skipped: list[Path],
) -> None:
    """Copy every file under *source_dir* into *destination_dir*."""
    for source in sorted(source_dir.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(source_dir)
        destination = destination_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if _install_file(source=source, destination=destination, force=force):
            installed.append(destination)
        else:
            skipped.append(destination)


def install_user_config(
    *,
    repo_root: Path,
    install_root: Path,
    force: bool = False,
    env: Mapping[str, str] | None = None,
) -> InstallUserConfigResult:
    agm_config_dir = agm_home_dir(home=install_root, env=env)
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

    # The package domain owns the managed stdlib's locked staging,
    # publication, integrity record, activation, and stale-file replacement.
    stdlib = refresh_managed_stdlib(repo_root / "packages" / "stdlib", home=install_root, env=env)
    installed.append(stdlib.root)

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
        env={"AGM_HOME": str(Path(args.prefix) / ".agm")} if args.prefix is not None else None,
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
