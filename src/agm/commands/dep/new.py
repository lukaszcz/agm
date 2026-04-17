"""agm dep new."""

from __future__ import annotations

import sys

from agm.commands.args import DepNewArgs
from agm.commands.dep.common import default_branch_from_remote, derive_dep_name
from agm.core.process import run_foreground
from agm.project.layout import current_project_dir


def run(args: DepNewArgs) -> None:
    project_dir = current_project_dir()
    dep = derive_dep_name(args.repo_url)
    dep_dir = project_dir / "deps" / dep
    if dep_dir.exists():
        print(f"error: deps/{dep} already exists", file=sys.stderr)
        raise SystemExit(1)

    resolved_branch = args.branch or default_branch_from_remote(args.repo_url)
    target_dir = dep_dir / resolved_branch
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    returncode = run_foreground(
        ["git", "clone", "--branch", resolved_branch, args.repo_url, str(target_dir)],
    )
    if returncode != 0:
        try:
            dep_dir.rmdir()
        except OSError:
            pass
        raise SystemExit(returncode)
