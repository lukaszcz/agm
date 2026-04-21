"""Install AGM user configuration files."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast


class _InstallArgs(Protocol):
    force: bool
    prefix: str | None


@dataclass(frozen=True)
class InstallUserConfigResult:
    installed: list[Path]
    skipped: list[Path]


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


def install_user_config(
    *,
    repo_root: Path,
    install_root: Path,
    force: bool = False,
) -> InstallUserConfigResult:
    agm_config_dir = install_root / ".agm"
    sandbox_dir = agm_config_dir / "sandbox"
    prompts_dir = agm_config_dir / "prompts"
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)

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

    for prompt_source in sorted((repo_root / "config" / "prompts").iterdir()):
        if not prompt_source.is_file():
            continue
        prompt_destination = prompts_dir / prompt_source.name
        if _install_file(source=prompt_source, destination=prompt_destination, force=force):
            installed.append(prompt_destination)
        else:
            skipped.append(prompt_destination)

    return InstallUserConfigResult(installed=installed, skipped=skipped)


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
