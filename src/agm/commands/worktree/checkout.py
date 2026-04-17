"""agm worktree checkout."""

from __future__ import annotations

from agm.commands.args import WorktreeCheckoutArgs
from agm.parser import exit_with_usage_error
from agm.utils.worktree import ensure_worktree


def run(args: WorktreeCheckoutArgs) -> None:
    branch_name = args.new_branch if args.new_branch is not None else args.branch
    if branch_name is None:
        command_name = args.command or "worktree"
        subcommand_name = args.wt_command or "checkout"
        exit_with_usage_error(
            [command_name, subcommand_name],
            "error: branch name is required unless -b is provided",
        )
    ensure_worktree(
        new_branch=args.new_branch,
        worktrees_dir=args.worktrees_dir,
        branch=args.branch,
        existing_ok=args.new_branch is None,
    )
