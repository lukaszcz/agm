"""agm config copy."""

from __future__ import annotations

import argparse
from pathlib import Path

from agm.utils.project import copy_config


def run(args: argparse.Namespace) -> None:
    copy_config(
        project_dir=Path(args.project_dir) if args.project_dir is not None else None,
        target=Path(args.dirname),
    )
