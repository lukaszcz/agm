"""Command implementations that dispatch to the underlying shell scripts."""

from __future__ import annotations

import os
import sys
from typing import NoReturn

from plumbum import CommandNotFound, local


def _resolve(script: str) -> str:
    """Look up *script* on ``PATH`` via plumbum and return its absolute path."""
    try:
        cmd = local[script]
    except CommandNotFound:
        print(f"error: {script} not found on PATH", file=sys.stderr)
        sys.exit(127)
    return cmd.executable


def _run(script: str, args: list[str]) -> NoReturn:
    """Replace the current process with *script* executed with *args*.

    Uses ``os.execvp`` so the script inherits the full environment and stdio.
    """
    executable = _resolve(script)
    os.execvp(executable, [script, *args])


# ---------------------------------------------------------------------------
# agm br sync  →  brsync.sh
# ---------------------------------------------------------------------------

def cmd_branch_sync() -> NoReturn:
    _run("brsync.sh", [])


# ---------------------------------------------------------------------------
# agm config {cp|copy}  →  cpconfig.sh [-d project-dir] dirname
# ---------------------------------------------------------------------------

def cmd_config_copy(
    *,
    project_dir: str | None,
    dirname: str,
) -> NoReturn:
    args: list[str] = []
    if project_dir is not None:
        args.extend(["-d", project_dir])
    args.append(dirname)
    _run("cpconfig.sh", args)


# ---------------------------------------------------------------------------
# agm wt {co|checkout}  →  mkwt.sh [-b branch] [-d dir] [branch]
# ---------------------------------------------------------------------------

def cmd_worktree_checkout(
    *,
    new_branch: str | None,
    worktrees_dir: str | None,
    branch: str | None,
) -> NoReturn:
    args: list[str] = []
    if new_branch is not None:
        args.extend(["-b", new_branch])
    if worktrees_dir is not None:
        args.extend(["-d", worktrees_dir])
    if branch is not None:
        args.append(branch)
    _run("mkwt.sh", args)


# ---------------------------------------------------------------------------
# agm wt new  →  mkwt.sh -b <branch> [-d dir]
# ---------------------------------------------------------------------------

def cmd_worktree_new(
    *,
    branch: str,
    worktrees_dir: str | None,
) -> NoReturn:
    args: list[str] = ["-b", branch]
    if worktrees_dir is not None:
        args.extend(["-d", worktrees_dir])
    _run("mkwt.sh", args)


# ---------------------------------------------------------------------------
# agm wt {rm|remove}  →  rmwt.sh [-f] <branch>
# ---------------------------------------------------------------------------

def cmd_worktree_remove(
    *,
    force: bool,
    branch: str,
) -> NoReturn:
    args: list[str] = []
    if force:
        args.append("-f")
    args.append(branch)
    _run("rmwt.sh", args)


# ---------------------------------------------------------------------------
# agm dep new  →  pm-dep.sh new [-b branch] repo-url
# agm dep switch  →  pm-dep.sh switch dep [-b] branch
# ---------------------------------------------------------------------------

def cmd_dep(
    *,
    subcmd: str,
    branch: str | None = None,
    repo_url: str | None = None,
    dep: str | None = None,
    create_branch: bool = False,
) -> NoReturn:
    args: list[str] = [subcmd]
    if subcmd == "new":
        if branch is not None:
            args.extend(["-b", branch])
        assert repo_url is not None
        args.append(repo_url)
    elif subcmd == "switch":
        assert dep is not None
        args.append(dep)
        if create_branch:
            args.append("-b")
        assert branch is not None
        args.append(branch)
    _run("pm-dep.sh", args)


# ---------------------------------------------------------------------------
# agm fetch  →  pm-fetch.sh
# ---------------------------------------------------------------------------

def cmd_fetch() -> NoReturn:
    _run("pm-fetch.sh", [])


# ---------------------------------------------------------------------------
# agm init  →  pm-init.sh [-b branch] [project-name] [repo-url]
# ---------------------------------------------------------------------------

def cmd_init(
    *,
    branch: str | None,
    positional: list[str],
) -> NoReturn:
    args: list[str] = []
    if branch is not None:
        args.extend(["-b", branch])
    args.extend(positional)
    _run("pm-init.sh", args)


# ---------------------------------------------------------------------------
# agm open  →  pm.sh open [-n pane_count] [branch]
# ---------------------------------------------------------------------------

def cmd_open(
    *,
    pane_count: str | None,
    branch: str | None,
) -> NoReturn:
    args: list[str] = ["open"]
    if pane_count is not None:
        args.extend(["-n", pane_count])
    if branch is not None:
        args.append(branch)
    _run("pm.sh", args)


# ---------------------------------------------------------------------------
# agm new  →  pm.sh new [-n pane_count] [-p parent] branch
# ---------------------------------------------------------------------------

def cmd_new(
    *,
    pane_count: str | None,
    parent: str | None,
    branch: str,
) -> NoReturn:
    args: list[str] = ["new"]
    if pane_count is not None:
        args.extend(["-n", pane_count])
    if parent is not None:
        args.extend(["-p", parent])
    args.append(branch)
    _run("pm.sh", args)


# ---------------------------------------------------------------------------
# agm {co|checkout}  →  pm.sh co [-n pane_count] [-p parent] branch
# ---------------------------------------------------------------------------

def cmd_checkout(
    *,
    pane_count: str | None,
    parent: str | None,
    branch: str,
) -> NoReturn:
    args: list[str] = ["co"]
    if pane_count is not None:
        args.extend(["-n", pane_count])
    if parent is not None:
        args.extend(["-p", parent])
    args.append(branch)
    _run("pm.sh", args)


# ---------------------------------------------------------------------------
# agm run  →  sandbox.sh [--no-patch] [-f settings.json] <command> [args...]
# ---------------------------------------------------------------------------

def cmd_run(
    *,
    no_patch: bool,
    settings_file: str | None,
    run_command: list[str],
) -> NoReturn:
    args: list[str] = []
    if no_patch:
        args.append("--no-patch")
    if settings_file is not None:
        args.extend(["-f", settings_file])
    args.extend(run_command)
    _run("sandbox.sh", args)


# ---------------------------------------------------------------------------
# agm tmux new  →  tmux.sh [-d] [-n pane_count] [session_name]
# ---------------------------------------------------------------------------

def cmd_tmux_new(
    *,
    detach: bool,
    pane_count: str | None,
    session_name: str | None,
) -> NoReturn:
    args: list[str] = []
    if detach:
        args.append("-d")
    if pane_count is not None:
        args.extend(["-n", pane_count])
    if session_name is not None:
        args.append(session_name)
    _run("tmux.sh", args)


# ---------------------------------------------------------------------------
# agm tmux layout  →  tmux-apply-layout.sh pane_count window_id width height
# ---------------------------------------------------------------------------

def cmd_tmux_layout(
    *,
    pane_count: str,
    window_id: str,
    width: str,
    height: str,
) -> NoReturn:
    _run("tmux-apply-layout.sh", [pane_count, window_id, width, height])
