"""AGM – unified CLI that dispatches subcommands to underlying shell scripts."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import NoReturn

from agm.commands import (
    cmd_branch_sync,
    cmd_checkout,
    cmd_config_copy,
    cmd_dep,
    cmd_fetch,
    cmd_init,
    cmd_new,
    cmd_open,
    cmd_run,
    cmd_tmux_layout,
    cmd_tmux_new,
    cmd_worktree_checkout,
    cmd_worktree_new,
    cmd_worktree_remove,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agm",
        description="Manage worktrees, project dependencies, configuration and tmux sessions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- agm {br|branch} sync ---
    br_parser = subparsers.add_parser("br", help="Branch operations")
    br_alias = subparsers.add_parser("branch", help="Branch operations")
    for p in (br_parser, br_alias):
        br_sub = p.add_subparsers(dest="br_command", required=True)
        br_sub.add_parser("sync", help="Sync remote branches locally")

    # --- agm config {cp|copy} ---
    config_parser = subparsers.add_parser("config", help="Configuration operations")
    config_sub = config_parser.add_subparsers(dest="config_command", required=True)
    for name in ("cp", "copy"):
        cp_parser = config_sub.add_parser(name, help="Copy configuration files")
        cp_parser.add_argument("-d", dest="project_dir", metavar="project-dir", default=None,
                               help="Project directory")
        cp_parser.add_argument("dirname", help="Target directory")

    # --- agm {wt|worktree} {co|checkout} ---
    # --- agm {wt|worktree} new ---
    # --- agm {wt|worktree} {rm|remove} ---
    for wt_name in ("wt", "worktree"):
        wt_parser = subparsers.add_parser(wt_name, help="Worktree operations")
        wt_sub = wt_parser.add_subparsers(dest="wt_command", required=True)

        for co_name in ("co", "checkout"):
            wt_co = wt_sub.add_parser(co_name, help="Check out a worktree")
            wt_co.add_argument("-b", dest="new_branch", metavar="branch-name", default=None,
                               help="Create a new branch")
            wt_co.add_argument("-d", dest="worktrees_dir", metavar="dir", default=None,
                               help="Worktrees directory")
            wt_co.add_argument("branch", nargs="?", default=None,
                               help="Branch name to check out")

        wt_new = wt_sub.add_parser("new", help="Create a new worktree branch")
        wt_new.add_argument("-d", dest="worktrees_dir", metavar="dir", default=None,
                            help="Worktrees directory")
        wt_new.add_argument("branch", help="New branch name")

        for rm_name in ("rm", "remove"):
            wt_rm = wt_sub.add_parser(rm_name, help="Remove a worktree")
            wt_rm.add_argument("-f", dest="force", action="store_true", default=False,
                               help="Force removal")
            wt_rm.add_argument("branch", help="Branch name to remove")

    # --- agm dep ---
    dep_parser = subparsers.add_parser("dep", help="Manage dependencies")
    dep_sub = dep_parser.add_subparsers(dest="dep_command", required=True)

    dep_new = dep_sub.add_parser("new", help="Clone a new dependency")
    dep_new.add_argument("-b", dest="branch", metavar="branch", default=None,
                         help="Branch to clone")
    dep_new.add_argument("repo_url", metavar="repo-url", help="Repository URL")

    dep_switch = dep_sub.add_parser("switch", help="Switch dependency branch")
    dep_switch.add_argument("-b", dest="create_branch", action="store_true", default=False,
                            help="Create a new branch")
    dep_switch.add_argument("dep", help="Dependency name")
    dep_switch.add_argument("branch", help="Branch name")

    # --- agm fetch ---
    subparsers.add_parser("fetch", help="Fetch repo and dependencies")

    # --- agm init ---
    init_parser = subparsers.add_parser("init", help="Initialize a new project")
    init_parser.add_argument("-b", dest="branch", metavar="branch", default=None,
                             help="Branch to clone")
    init_parser.add_argument("positional", nargs="+", metavar="arg",
                             help="[project-name] [repo-url] or repo-url")

    # --- agm open ---
    open_parser = subparsers.add_parser("open", help="Open a project session")
    open_parser.add_argument("-n", dest="pane_count", metavar="pane_count", default=None,
                             help="Number of tmux panes")
    open_parser.add_argument("branch", nargs="?", default=None, help="Branch name")

    # --- agm new ---
    new_parser = subparsers.add_parser("new", help="Create a new branch and open session")
    new_parser.add_argument("-n", dest="pane_count", metavar="pane_count", default=None,
                            help="Number of tmux panes")
    new_parser.add_argument("-p", dest="parent", metavar="parent", default=None,
                            help="Parent branch")
    new_parser.add_argument("branch", help="Branch name")

    # --- agm {co|checkout} ---
    for co_name in ("co", "checkout"):
        co_parser = subparsers.add_parser(co_name, help="Check out a branch and open session")
        co_parser.add_argument("-n", dest="pane_count", metavar="pane_count", default=None,
                               help="Number of tmux panes")
        co_parser.add_argument("-p", dest="parent", metavar="parent", default=None,
                               help="Parent branch")
        co_parser.add_argument("branch", help="Branch name")

    # --- agm run ---
    run_parser = subparsers.add_parser("run", help="Run command in sandbox")
    run_parser.add_argument("--no-patch", dest="no_patch", action="store_true", default=False,
                            help="Disable PROJ_DIR patching")
    run_parser.add_argument("-f", dest="settings_file", metavar="settings.json", default=None,
                            help="Settings file")
    run_parser.add_argument("run_command", nargs=argparse.REMAINDER, metavar="command",
                            help="Command to run in sandbox")

    # --- agm tmux new ---
    # --- agm tmux layout ---
    tmux_parser = subparsers.add_parser("tmux", help="Tmux session operations")
    tmux_sub = tmux_parser.add_subparsers(dest="tmux_command", required=True)

    tmux_new = tmux_sub.add_parser("new", help="Create a new tmux session")
    tmux_new.add_argument("-d", "--detach", dest="detach", action="store_true", default=False,
                          help="Create session detached")
    tmux_new.add_argument("-n", dest="pane_count", metavar="pane_count", default=None,
                          help="Number of panes")
    tmux_new.add_argument("session_name", nargs="?", default=None, help="Session name")

    tmux_layout = tmux_sub.add_parser("layout", help="Apply tmux layout")
    tmux_layout.add_argument("pane_count", help="Number of panes")
    tmux_layout.add_argument("window_id", help="Tmux window ID")
    tmux_layout.add_argument("width", help="Window width")
    tmux_layout.add_argument("height", help="Window height")

    return parser


def dispatch(args: argparse.Namespace) -> NoReturn:
    """Route parsed arguments to the appropriate command handler."""
    cmd: str = args.command

    if cmd in ("br", "branch"):
        # only subcommand is "sync"
        cmd_branch_sync()

    elif cmd == "config":
        # only subcommand is "cp" / "copy"
        cmd_config_copy(project_dir=args.project_dir, dirname=args.dirname)

    elif cmd in ("wt", "worktree"):
        wt_cmd: str = args.wt_command
        if wt_cmd in ("co", "checkout"):
            if args.new_branch is not None:
                # -b was given → same as "wt new" but via mkwt.sh -b
                cmd_worktree_checkout(
                    new_branch=args.new_branch,
                    worktrees_dir=args.worktrees_dir,
                    branch=args.branch,
                )
            else:
                cmd_worktree_checkout(
                    new_branch=None,
                    worktrees_dir=args.worktrees_dir,
                    branch=args.branch,
                )
        elif wt_cmd == "new":
            # alias for mkwt.sh -b <branch>
            cmd_worktree_new(
                branch=args.branch,
                worktrees_dir=args.worktrees_dir,
            )
        elif wt_cmd in ("rm", "remove"):
            cmd_worktree_remove(force=args.force, branch=args.branch)

    elif cmd == "dep":
        dep_cmd: str = args.dep_command
        if dep_cmd == "new":
            cmd_dep(subcmd="new", branch=args.branch, repo_url=args.repo_url)
        elif dep_cmd == "switch":
            cmd_dep(
                subcmd="switch",
                dep=args.dep,
                create_branch=args.create_branch,
                branch=args.branch,
            )

    elif cmd == "fetch":
        cmd_fetch()

    elif cmd == "init":
        cmd_init(branch=args.branch, positional=args.positional)

    elif cmd == "open":
        cmd_open(pane_count=args.pane_count, branch=args.branch)

    elif cmd == "new":
        cmd_new(pane_count=args.pane_count, parent=args.parent, branch=args.branch)

    elif cmd in ("co", "checkout"):
        cmd_checkout(pane_count=args.pane_count, parent=args.parent, branch=args.branch)

    elif cmd == "run":
        cmd_run(
            no_patch=args.no_patch,
            settings_file=args.settings_file,
            run_command=args.run_command,
        )

    elif cmd == "tmux":
        tmux_cmd: str = args.tmux_command
        if tmux_cmd == "new":
            cmd_tmux_new(
                detach=args.detach,
                pane_count=args.pane_count,
                session_name=args.session_name,
            )
        elif tmux_cmd == "layout":
            cmd_tmux_layout(
                pane_count=args.pane_count,
                window_id=args.window_id,
                width=args.width,
                height=args.height,
            )

    # Should never reach here because argparse enforces required subcommands,
    # but satisfy the type checker.
    sys.exit(0)


def main(argv: Sequence[str] | None = None) -> NoReturn:
    parser = build_parser()
    args = parser.parse_args(argv)
    dispatch(args)


if __name__ == "__main__":
    main()
