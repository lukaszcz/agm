"""AGM command-line interface."""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections.abc import Sequence
from typing import NoReturn, Protocol, cast

import agm.commands.branch.sync as branch_sync_command
import agm.commands.config.copy as config_copy_command
import agm.commands.dep.new as dep_new_command
import agm.commands.dep.remove as dep_remove_command
import agm.commands.dep.switch as dep_switch_command
import agm.commands.fetch as fetch_command
import agm.commands.init as init_command
import agm.commands.open as open_command
import agm.commands.run as run_command
import agm.commands.tmux.layout as tmux_layout_command
import agm.commands.tmux.new as tmux_new_command
import agm.commands.worktree.checkout as worktree_checkout_command
import agm.commands.worktree.new as worktree_new_command
import agm.commands.worktree.remove as worktree_remove_command
import agm.commands.worktree.setup as worktree_setup_command
from agm.commands.args import (
    ConfigCopyArgs,
    DepNewArgs,
    DepRemoveArgs,
    DepSwitchArgs,
    InitArgs,
    OpenArgs,
    RunArgs,
    TmuxLayoutArgs,
    TmuxNewArgs,
    WorktreeCheckoutArgs,
    WorktreeNewArgs,
    WorktreeRemoveArgs,
    WorktreeSetupArgs,
)


class _Writeable(Protocol):
    def write(self, data: str) -> object: ...


class _DispatchArgs(Protocol):
    command: str | None
    help_command: str | None
    br_command: str | None
    config_command: str | None
    wt_command: str | None
    dep_command: str | None
    tmux_command: str | None


class _HelpTextArgumentParser(argparse.ArgumentParser):
    def __init__(
        self,
        prog: str | None = None,
        usage: str | None = None,
        description: str | None = None,
        epilog: str | None = None,
        parents: Sequence[argparse.ArgumentParser] = (),
        formatter_class: type[argparse.HelpFormatter] = argparse.HelpFormatter,
        prefix_chars: str = "-",
        fromfile_prefix_chars: str | None = None,
        argument_default: object | None = None,
        conflict_handler: str = "error",
        add_help: bool = True,
        allow_abbrev: bool = True,
        exit_on_error: bool = True,
        *,
        help_text: str | None = None,
    ) -> None:
        self._help_text = help_text
        super().__init__(
            prog=prog,
            usage=usage,
            description=description,
            epilog=epilog,
            parents=parents,
            formatter_class=formatter_class,
            prefix_chars=prefix_chars,
            fromfile_prefix_chars=fromfile_prefix_chars,
            argument_default=argument_default,
            conflict_handler=conflict_handler,
            add_help=add_help,
            allow_abbrev=allow_abbrev,
            exit_on_error=exit_on_error,
        )

    def format_help(self) -> str:
        if self._help_text is not None:
            return self._help_text
        return super().format_help()

    def print_help(self, file: _Writeable | None = None) -> None:
        if file is None:
            file = sys.stdout
        print(self.format_help(), end="", file=file)

_HELP_TEXTS: dict[str, str] = {
    "open": textwrap.dedent("""\
        agm open [-n PANES] [-p PARENT] TARGET

        Open a tmux session for a project checkout.

        Behavior:
          repo           Open the main repo session.
          default branch Open the main repo session when TARGET matches the
                         branch currently checked out in repo/.
          existing wt    Open the tmux session for worktrees/BRANCH.
          existing br    Check out BRANCH into a worktree, then open it.
          missing br     Create BRANCH from PARENT/current branch, then open it.

        Examples:
          agm open repo
          agm open main
          agm open feat/login
          agm open -n 4 feat/login
          agm open -p main feat/search
    """),
    "init": textwrap.dedent("""\
        agm init [-b BRANCH] [PROJECT_NAME] REPO_URL

        Initialize a new project by cloning a repository. If PROJECT_NAME is
        omitted it is derived from the repo URL.
    """),
    "fetch": textwrap.dedent("""\
        agm fetch

        Fetch the main repository and all checked-out dependencies.
    """),
    "branch": textwrap.dedent("""\
        agm branch sync
        agm br     sync

        Branch management commands.
    """),
    "config": textwrap.dedent("""\
        agm config copy [-d PROJECT_DIR] DIRNAME
        agm config cp   [-d PROJECT_DIR] DIRNAME

        Copy project configuration files into a target directory.
    """),
    "worktree": textwrap.dedent("""\
        agm worktree checkout [-b BRANCH] [-d DIR] [BRANCH]
        agm worktree new      [-d DIR] BRANCH
        agm worktree setup
        agm worktree remove   [-f] BRANCH
        agm wt co | wt new | wt setup | wt rm

        Low-level git worktree management.
    """),
    "dep": textwrap.dedent("""\
        agm dep new    [-b BRANCH] REPO_URL
        agm dep rm     [--all] DEP | DEP/BRANCH
        agm dep switch [-b] DEP BRANCH

        Manage project dependency checkouts under deps/.
    """),
    "run": textwrap.dedent("""\
        agm run [--no-patch] [-f SETTINGS] COMMAND [ARGS...]

        Run a command inside an Anthropic Sandbox Runtime container.
    """),
    "tmux": textwrap.dedent("""\
        agm tmux new    [-d] [-n PANES] [SESSION]
        agm tmux layout PANES WINDOW_ID WIDTH HEIGHT

        Tmux session and layout management.
    """),
    "help": textwrap.dedent("""\
        agm help [COMMAND]

        Show help information for top-level commands.
    """),
}

_HELP_ALIASES: dict[str, str] = {
    "br": "branch",
    "wt": "worktree",
    "cp": "config",
    "copy": "config",
}

_COMMAND_OVERVIEW: list[tuple[str, str]] = [
    ("open", "Open a project session, creating or checking out a branch as needed"),
    ("init", "Initialize a new project by cloning a repository"),
    ("fetch", "Fetch latest changes for the repo and all dependencies"),
    ("branch (br)", "Branch management (sync remote tracking branches)"),
    ("config", "Copy project configuration files"),
    ("worktree (wt)", "Low-level git worktree management"),
    ("dep", "Manage project dependency checkouts"),
    ("run", "Run a command inside an Anthropic Sandbox Runtime"),
    ("tmux", "Tmux session and layout management"),
    ("help", "Show help for a command"),
]


def _overview_text() -> str:
    lines = [
        "agm - Agent Management Framework",
        "",
        "Usage: agm <command> [options] [args]",
        "",
        "Commands:",
    ]
    width = max(len(name) for name, _ in _COMMAND_OVERVIEW)
    for name, desc in _COMMAND_OVERVIEW:
        lines.append(f"  {name:<{width + 2}} {desc}")
    lines.extend(
        [
            "",
            "Run 'agm help <command>' for detailed help on a specific command.",
            "Run 'agm <command> --help' for option summary.",
        ]
    )
    return "\n".join(lines) + "\n"


def _help_text_for(command: str) -> str | None:
    canonical = _HELP_ALIASES.get(command, command)
    return _HELP_TEXTS.get(canonical)


def _print_overview() -> None:
    print(_overview_text(), end="")


def _print_command_help(command: str) -> None:
    text = _help_text_for(command)
    if text is None:
        print(f"agm: unknown command '{command}'", file=sys.stderr)
        print("\nRun 'agm help' to see available commands.", file=sys.stderr)
        raise SystemExit(1)
    print(text, end="")


def build_parser() -> argparse.ArgumentParser:
    parser = _HelpTextArgumentParser(
        prog="agm",
        description="Agent Management Framework",
        epilog="Run 'agm help <command>' for detailed help on a specific command.",
        help_text=_overview_text(),
    )
    subparsers = parser.add_subparsers(dest="command", parser_class=_HelpTextArgumentParser)

    help_parser = subparsers.add_parser(
        "help",
        help="Show help for a command",
        help_text=_HELP_TEXTS["help"],
    )
    help_parser.add_argument("help_command", nargs="?", default=None, metavar="command")

    open_parser = subparsers.add_parser(
        "open",
        help="Open a tmux session for a project checkout",
        help_text=_HELP_TEXTS["open"],
    )
    open_parser.add_argument("-n", dest="pane_count", metavar="pane_count", default=None)
    open_parser.add_argument("-p", dest="parent", metavar="parent", default=None)
    open_parser.add_argument("branch", metavar="target")

    br_parser = subparsers.add_parser(
        "br",
        help="Branch operations (alias for 'branch')",
        help_text=_help_text_for("br"),
    )
    branch_parser = subparsers.add_parser(
        "branch",
        help="Branch operations",
        help_text=_help_text_for("branch"),
    )
    for current in (br_parser, branch_parser):
        current_sub = current.add_subparsers(
            dest="br_command",
            parser_class=_HelpTextArgumentParser,
        )
        current_sub.add_parser("sync", help="Sync remote tracking branches")

    config_parser = subparsers.add_parser(
        "config",
        help="Copy project configuration files",
        help_text=_help_text_for("config"),
    )
    config_sub = config_parser.add_subparsers(
        dest="config_command",
        parser_class=_HelpTextArgumentParser,
    )
    for name in ("cp", "copy"):
        current = config_sub.add_parser(name, help="Copy configuration files")
        current.add_argument("-d", dest="project_dir", metavar="project-dir", default=None)
        current.add_argument("dirname")

    for wt_name in ("wt", "worktree"):
        wt_parser = subparsers.add_parser(
            wt_name,
            help="Git worktree management",
            help_text=_help_text_for(wt_name),
        )
        wt_sub = wt_parser.add_subparsers(dest="wt_command", parser_class=_HelpTextArgumentParser)
        for co_name in ("co", "checkout"):
            current = wt_sub.add_parser(co_name, help="Check out a branch into a worktree")
            current.add_argument("-b", dest="new_branch", metavar="branch-name", default=None)
            current.add_argument("-d", dest="worktrees_dir", metavar="dir", default=None)
            current.add_argument("branch", nargs="?", default=None)
        wt_new = wt_sub.add_parser("new", help="Create a new branch and its worktree")
        wt_new.add_argument("-d", dest="worktrees_dir", metavar="dir", default=None)
        wt_new.add_argument("branch")
        wt_sub.add_parser("setup", help="Run setup scripts for the current checkout")
        for rm_name in ("rm", "remove"):
            current = wt_sub.add_parser(rm_name, help="Remove a worktree")
            current.add_argument("-f", dest="force", action="store_true", default=False)
            current.add_argument("branch")

    dep_parser = subparsers.add_parser(
        "dep",
        help="Manage project dependency checkouts",
        help_text=_help_text_for("dep"),
    )
    dep_sub = dep_parser.add_subparsers(dest="dep_command", parser_class=_HelpTextArgumentParser)
    dep_new = dep_sub.add_parser("new", help="Clone a new dependency")
    dep_new.add_argument("-b", dest="branch", metavar="branch", default=None)
    dep_new.add_argument("repo_url", metavar="repo-url")
    dep_switch = dep_sub.add_parser("switch", help="Switch a dependency branch")
    dep_switch.add_argument("-b", dest="create_branch", action="store_true", default=False)
    dep_switch.add_argument("dep")
    dep_switch.add_argument("branch")
    dep_rm = dep_sub.add_parser("rm", help="Remove a dependency worktree or repo")
    dep_rm.add_argument("--all", dest="all", action="store_true", default=False)
    dep_rm.add_argument("target")

    subparsers.add_parser(
        "fetch",
        help="Fetch the repo and dependencies",
        help_text=_help_text_for("fetch"),
    )

    init_parser = subparsers.add_parser(
        "init",
        help="Initialize a new project",
        help_text=_help_text_for("init"),
    )
    init_parser.add_argument("-b", dest="branch", metavar="branch", default=None)
    init_parser.add_argument("positional", nargs="+", metavar="arg")

    run_parser = subparsers.add_parser(
        "run",
        help="Run a command inside an Anthropic Sandbox Runtime",
        help_text=_help_text_for("run"),
    )
    run_parser.add_argument("--no-patch", dest="no_patch", action="store_true", default=False)
    run_parser.add_argument("-f", dest="settings_file", metavar="settings.json", default=None)
    run_parser.add_argument("run_command", nargs=argparse.REMAINDER, metavar="command")

    tmux_parser = subparsers.add_parser(
        "tmux",
        help="Tmux session and layout management",
        help_text=_help_text_for("tmux"),
    )
    tmux_sub = tmux_parser.add_subparsers(dest="tmux_command", parser_class=_HelpTextArgumentParser)
    tmux_new = tmux_sub.add_parser("new", help="Create a new tmux session")
    tmux_new.add_argument("-d", "--detach", dest="detach", action="store_true", default=False)
    tmux_new.add_argument("-n", dest="pane_count", metavar="pane_count", default=None)
    tmux_new.add_argument("session_name", nargs="?", default=None)
    tmux_layout = tmux_sub.add_parser("layout", help="Apply a tiled pane layout")
    tmux_layout.add_argument("pane_count")
    tmux_layout.add_argument("window_id")
    tmux_layout.add_argument("width")
    tmux_layout.add_argument("height")
    return parser


def dispatch(args: _DispatchArgs) -> NoReturn:
    cmd = args.command
    if cmd is None:
        _print_overview()
        raise SystemExit(0)
    if cmd == "help":
        if args.help_command is None:
            _print_overview()
        else:
            _print_command_help(args.help_command)
        raise SystemExit(0)
    if cmd in {"br", "branch"}:
        if args.br_command is None:
            _print_command_help(cmd)
            raise SystemExit(0)
        branch_sync_command.run(args)
        raise SystemExit(0)
    if cmd == "open":
        open_command.run(cast(OpenArgs, args))
        raise SystemExit(0)
    if cmd == "config":
        if args.config_command is None:
            _print_command_help(cmd)
            raise SystemExit(0)
        config_copy_command.run(cast(ConfigCopyArgs, args))
        raise SystemExit(0)
    if cmd in {"wt", "worktree"}:
        if args.wt_command is None:
            _print_command_help(cmd)
            raise SystemExit(0)
        if args.wt_command in {"co", "checkout"}:
            worktree_checkout_command.run(cast(WorktreeCheckoutArgs, args))
        elif args.wt_command == "new":
            worktree_new_command.run(cast(WorktreeNewArgs, args))
        elif args.wt_command == "setup":
            worktree_setup_command.run(cast(WorktreeSetupArgs, args))
        else:
            worktree_remove_command.run(cast(WorktreeRemoveArgs, args))
        raise SystemExit(0)
    if cmd == "dep":
        if args.dep_command is None:
            _print_command_help(cmd)
            raise SystemExit(0)
        if args.dep_command == "new":
            dep_new_command.run(cast(DepNewArgs, args))
        elif args.dep_command == "rm":
            dep_remove_command.run(cast(DepRemoveArgs, args))
        else:
            dep_switch_command.run(cast(DepSwitchArgs, args))
        raise SystemExit(0)
    if cmd == "fetch":
        fetch_command.run(args)
        raise SystemExit(0)
    if cmd == "init":
        init_command.run(cast(InitArgs, args))
        raise SystemExit(0)
    if cmd == "run":
        run_command.run(cast(RunArgs, args))
        raise AssertionError("unreachable")
    if cmd == "tmux":
        if args.tmux_command is None:
            _print_command_help(cmd)
            raise SystemExit(0)
        if args.tmux_command == "new":
            tmux_new_command.run(cast(TmuxNewArgs, args))
        else:
            tmux_layout_command.run(cast(TmuxLayoutArgs, args))
        raise SystemExit(0)
    raise SystemExit(0)


def main(argv: Sequence[str] | None = None) -> NoReturn:
    parser = build_parser()
    args = parser.parse_args(argv)
    dispatch(cast(_DispatchArgs, args))


if __name__ == "__main__":
    main()
