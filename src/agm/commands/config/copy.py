"""agm config copy."""

from __future__ import annotations

from pathlib import Path

from agm.cli_support.args import ConfigCopyArgs
from agm.project.layout import copy_config


def run(args: ConfigCopyArgs) -> None:
    copy_config(target=Path(args.dirname))
