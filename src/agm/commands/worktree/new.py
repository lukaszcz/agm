"""agm worktree new."""

from __future__ import annotations

from agm.cli_support.args import WorktreeNewArgs
from agm.project.config_git import commit_config_dir_changes
from agm.project.layout import discover_current_project_dir, project_config_dir
from agm.project.worktree import ensure_worktree


def run(args: WorktreeNewArgs) -> None:
    worktree_path = ensure_worktree(
        new_branch=args.branch,
        worktrees_dir=args.worktrees_dir,
        branch=None,
        existing_ok=False,
        reuse_existing_branch=True,
    )
    project_dir = discover_current_project_dir(worktree_path)
    if project_dir is not None:
        commit_config_dir_changes(
            project_dir,
            f"chore: add config for {args.branch}",
            add_paths=[project_config_dir(project_dir) / args.branch],
        )
