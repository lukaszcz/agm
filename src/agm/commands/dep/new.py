"""agm dep new."""

from __future__ import annotations

import sys

from agm.commands.args import DepNewArgs
from agm.commands.dep.common import default_branch_from_remote, derive_dep_name
from agm.core import dry_run
from agm.core.fs import exists, mkdir, rmdir
from agm.core.process import require_success
from agm.project.layout import current_project_dir, project_deps_dir


def run(args: DepNewArgs) -> None:
    project_dir = current_project_dir()
    dep = derive_dep_name(args.repo_url)
    dep_dir = project_deps_dir(project_dir) / dep
    if exists(dep_dir):
        print(f"error: deps/{dep} already exists", file=sys.stderr)
        raise SystemExit(1)

    resolved_branch = args.branch or default_branch_from_remote(args.repo_url)
    target_dir = dep_dir / resolved_branch
    mkdir(target_dir.parent, parents=True, exist_ok=True)
    try:
        require_success(
            ["git", "clone", "--branch", resolved_branch, args.repo_url, str(target_dir)],
        )
    except SystemExit:
        if dry_run.enabled():
            raise
        try:
            rmdir(dep_dir)
        except OSError:
            pass
        raise
