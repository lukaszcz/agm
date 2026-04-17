"""agm dep switch."""

from __future__ import annotations

import sys

import agm.vcs.git as git_helpers
from agm.commands.args import DepSwitchArgs
from agm.commands.dep.common import default_branch_from_repo, main_dep_repo
from agm.project.layout import current_project_dir, project_deps_dir


def run(args: DepSwitchArgs) -> None:
    project_dir = current_project_dir()
    dep_dir = project_deps_dir(project_dir) / args.dep
    if not dep_dir.is_dir():
        print(f"error: deps/{args.dep} does not exist", file=sys.stderr)
        raise SystemExit(1)

    repo_path = main_dep_repo(dep_dir)
    target_dir = dep_dir / args.branch
    if target_dir.exists():
        print(f"error: deps/{args.dep}/{args.branch} already exists", file=sys.stderr)
        raise SystemExit(1)

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    git_helpers.fetch(repo_path)
    if args.create_branch:
        default_branch = default_branch_from_repo(repo_path)
        git_helpers.worktree_add(
            repo_path,
            target_dir,
            args.branch,
            create=True,
            start_point=default_branch,
        )
        return
    git_helpers.worktree_add(repo_path, target_dir, args.branch)
