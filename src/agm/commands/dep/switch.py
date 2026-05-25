"""agm dep switch."""

from __future__ import annotations

import sys
from pathlib import Path

import agm.vcs.git as git_helpers
from agm.commands.args import DepSwitchArgs
from agm.commands.dep.common import main_dep_repo
from agm.core.fs import exists, is_dir, mkdir
from agm.project.config_git import commit_config_dir_changes
from agm.project.dependency_env import (
    config_toml_file,
    current_config_branch,
    update_dependency_config,
)
from agm.project.layout import project_deps_dir, require_current_project_dir


def _checkout_name(dep_dir: Path, path: Path) -> str | None:
    resolved_dep_dir = dep_dir.resolve(strict=False)
    resolved_path = path.resolve(strict=False)
    if resolved_path == resolved_dep_dir or resolved_dep_dir not in resolved_path.parents:
        return None
    return resolved_path.relative_to(resolved_dep_dir).as_posix()


def _existing_checkout_name(
    *,
    dep_dir: Path,
    repo_path: Path,
    target: str,
) -> str | None:
    target_path = (dep_dir / target).resolve(strict=False)
    branch_match: str | None = None
    for worktree in git_helpers.worktree_list(repo_path):
        checkout_name = _checkout_name(dep_dir, worktree.path)
        if checkout_name is None:
            continue
        if worktree.path.resolve(strict=False) == target_path:
            return checkout_name
        if worktree.branch == target:
            branch_match = checkout_name
    return branch_match


def run(args: DepSwitchArgs) -> None:
    project_dir = require_current_project_dir()
    dep_dir = project_deps_dir(project_dir) / args.dep
    if not is_dir(dep_dir):
        print(f"error: deps/{args.dep} does not exist", file=sys.stderr)
        raise SystemExit(1)

    repo_path = main_dep_repo(dep_dir)
    checkout_name = _existing_checkout_name(
        dep_dir=dep_dir,
        repo_path=repo_path,
        target=args.branch,
    )
    if checkout_name is not None:
        config_branch = current_config_branch(project_dir)
        update_dependency_config(
            project_dir=project_dir,
            dep_name=args.dep,
            dep_branch=checkout_name,
            config_branch=config_branch,
        )
        commit_config_dir_changes(
            project_dir, f"chore: switch dependency {args.dep}",
            add_paths=[config_toml_file(project_dir, config_branch)],
        )
        return

    target_dir = dep_dir / args.branch
    if exists(target_dir):
        print(f"error: deps/{args.dep}/{args.branch} already exists", file=sys.stderr)
        raise SystemExit(1)

    mkdir(target_dir.parent, parents=True, exist_ok=True)
    git_helpers.fetch(repo_path)
    if args.create_branch:
        default_branch = git_helpers.default_branch_from_repo(repo_path)
        git_helpers.worktree_add(
            repo_path,
            target_dir,
            args.branch,
            create=True,
            start_point=default_branch,
        )
    else:
        git_helpers.worktree_add(repo_path, target_dir, args.branch)
    config_branch = current_config_branch(project_dir)
    update_dependency_config(
        project_dir=project_dir,
        dep_name=args.dep,
        dep_branch=args.branch,
        config_branch=config_branch,
    )
    commit_config_dir_changes(
        project_dir, f"chore: switch dependency {args.dep}",
        add_paths=[config_toml_file(project_dir, config_branch)],
    )
