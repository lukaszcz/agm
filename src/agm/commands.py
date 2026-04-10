"""Command implementations backed by native Python modules."""

from __future__ import annotations

from typing import NoReturn

from agm import dep as dep_module
from agm import fetch as fetch_module
from agm import init as init_module
from agm import sandbox as sandbox_module
from agm import session as session_module
from agm import tmux_layout as tmux_layout_module
from agm import tmux_session as tmux_session_module
from agm import worktree as worktree_module


def _exit_success() -> NoReturn:
    raise SystemExit(0)


def cmd_branch_sync() -> NoReturn:
    worktree_module.branch_sync()
    _exit_success()


def cmd_config_copy(
    *,
    project_dir: str | None,
    dirname: str,
) -> NoReturn:
    from pathlib import Path

    from agm.project import copy_config

    copy_config(
        project_dir=Path(project_dir) if project_dir is not None else None,
        target=Path(dirname),
    )
    _exit_success()


def cmd_worktree_checkout(
    *,
    new_branch: str | None,
    worktrees_dir: str | None,
    branch: str | None,
) -> NoReturn:
    worktree_module.worktree_checkout(
        new_branch=new_branch,
        worktrees_dir=worktrees_dir,
        branch=branch,
    )
    _exit_success()


def cmd_worktree_new(
    *,
    branch: str,
    worktrees_dir: str | None,
) -> NoReturn:
    worktree_module.worktree_checkout(
        new_branch=branch,
        worktrees_dir=worktrees_dir,
        branch=None,
    )
    _exit_success()


def cmd_worktree_remove(
    *,
    force: bool,
    branch: str,
) -> NoReturn:
    worktree_module.worktree_remove(force=force, branch=branch)
    _exit_success()


def cmd_dep(
    *,
    subcmd: str,
    branch: str | None = None,
    repo_url: str | None = None,
    dep: str | None = None,
    create_branch: bool = False,
) -> NoReturn:
    if subcmd == "new":
        assert repo_url is not None
        dep_module.dep_new(branch=branch, repo_url=repo_url)
        _exit_success()
    if subcmd == "switch":
        assert dep is not None
        assert branch is not None
        dep_module.dep_switch(dep=dep, branch=branch, create_branch=create_branch)
        _exit_success()
    raise SystemExit(1)


def cmd_fetch() -> NoReturn:
    fetch_module.fetch_all()
    _exit_success()


def cmd_init(
    *,
    branch: str | None,
    positional: list[str],
) -> NoReturn:
    init_module.init_project(branch=branch, positional=positional)
    _exit_success()


def cmd_open(
    *,
    pane_count: str | None,
    branch: str | None,
) -> NoReturn:
    session_module.open_session(pane_count=pane_count, branch=branch)
    _exit_success()


def cmd_new(
    *,
    pane_count: str | None,
    parent: str | None,
    branch: str,
) -> NoReturn:
    session_module.new_session(pane_count=pane_count, parent=parent, branch=branch)
    _exit_success()


def cmd_checkout(
    *,
    pane_count: str | None,
    parent: str | None,
    branch: str,
) -> NoReturn:
    session_module.checkout_session(pane_count=pane_count, parent=parent, branch=branch)
    _exit_success()


def cmd_run(
    *,
    no_patch: bool,
    settings_file: str | None,
    run_command: list[str],
) -> NoReturn:
    sandbox_module.run_in_sandbox(
        no_patch=no_patch,
        settings_file=settings_file,
        run_command=run_command,
    )
    raise AssertionError("unreachable")


def cmd_tmux_new(
    *,
    detach: bool,
    pane_count: str | None,
    session_name: str | None,
) -> NoReturn:
    tmux_session_module.create_tmux_session(
        detach=detach,
        pane_count=pane_count,
        session_name=session_name,
    )
    _exit_success()


def cmd_tmux_layout(
    *,
    pane_count: str,
    window_id: str,
    width: str,
    height: str,
) -> NoReturn:
    tmux_layout_module.apply_layout(
        pane_count=int(pane_count),
        window_id=window_id,
        width=int(width),
        height=int(height),
    )
    _exit_success()
