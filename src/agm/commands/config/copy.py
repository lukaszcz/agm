"""agm config copy."""

from __future__ import annotations

from pathlib import Path

from agm.commands.args import ConfigCopyArgs
from agm.project.layout import copy_config


def run(args: ConfigCopyArgs) -> None:
    copy_config(
        project_dir=Path(args.project_dir) if args.project_dir is not None else None,
        target=Path(args.dirname),
    )
