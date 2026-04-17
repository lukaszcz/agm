"""Argument parser and help text utilities for AGM."""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections.abc import Sequence
from typing import TYPE_CHECKING, NoReturn, Protocol, cast

if TYPE_CHECKING:
    from argparse import _SubParsersAction as _SubParsersActionType

    _SubParsersActionAlias = _SubParsersActionType[argparse.ArgumentParser]
else:
    _SubParsersActionAlias = argparse._SubParsersAction


class _Writeable(Protocol):
    def write(self, data: str) -> object: ...


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

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n\n{self.format_help()}")


_HELP_TEXTS: dict[str, str] = {
    "open": textwrap.dedent("""\
        agm open [-d] [-n PANES] [-p PARENT] TARGET

        Open a tmux session for a project checkout.

        Options:
          -d          Create the tmux session without attaching to it.
          -n PANES    Create the session with PANES panes.
          -p PARENT   Base a newly created branch worktree on PARENT instead of
                      the repo/ checkout's current branch.

        Behavior:
          repo           Open the main repo session.
          default branch Open the main repo session when TARGET matches the
                         branch currently checked out in repo/.
          existing wt    Open the tmux session for worktrees/BRANCH.
          existing br    Check out BRANCH into a worktree, then open it.
          missing br     Create BRANCH from PARENT/current branch, then open it.

        Examples:
          agm open repo
          agm open -d repo
          agm open main
          agm open feat/login
          agm open -n 4 feat/login
          agm open -p main feat/search
    """),
    "init": textwrap.dedent("""\
        agm init [-b BRANCH] PROJECT_NAME
        agm init [-b BRANCH] [PROJECT_NAME] REPO_URL

        Initialize a new project directory. When REPO_URL is provided, agm also
        clones it into repo/. If PROJECT_NAME is omitted in that form, it is
        derived from the repo URL.

        Options:
          -b BRANCH  Clone this branch when REPO_URL is provided.
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

        Copy project configuration files into an existing target directory.

        Options:
          -d PROJECT_DIR  Read shared config files from this project instead of
                          auto-detecting the current project.
    """),
    "worktree": textwrap.dedent("""\
        agm worktree new      [-d DIR] BRANCH
        agm worktree setup
        agm worktree remove   [-f] BRANCH
        agm wt new | wt setup | wt rm

        Low-level git worktree management.

        Options:
          agm worktree new -d DIR
              Create the worktree under DIR instead of the default worktrees/
              or .worktrees/ directory.
          agm worktree remove -f
              Force removal even when git reports uncommitted or locked state.
    """),
    "dep": textwrap.dedent("""\
        agm dep new    [-b BRANCH] REPO_URL
        agm dep rm     [--all] DEP | DEP/BRANCH | DEP/repo | DEP/MAIN_BRANCH
        agm dep switch [-b] DEP BRANCH

        Manage project dependency checkouts under deps/.

        Options:
          agm dep new -b BRANCH
              Clone BRANCH instead of the dependency's default branch.
          agm dep rm --all
              Remove the entire dependency directory, including the main repo
              checkout and any linked worktrees.
          agm dep switch -b
              Create BRANCH from the dependency's default branch before adding
              the new worktree.

        Targets:
          DEP/BRANCH      Remove a dependency worktree for BRANCH.
          DEP/repo        Remove the main dependency checkout.
          DEP/MAIN_BRANCH Remove the main dependency checkout by branch name.
    """),
    "run": textwrap.dedent("""\
        agm run [--no-patch] [-f SETTINGS] COMMAND [ARGS...]

        Run a command inside an Anthropic Sandbox Runtime container.

        Command config:
          $HOME/.agm/config.toml, $PROJ_DIR/config/config.toml, and
          ./.agm/config.toml are loaded in that order when present.
          [run.<command>] alias = "<other-command>" makes
          "agm run <command>" execute <other-command> instead.

        Options:
          -f SETTINGS  Use this settings file directly instead of discovering
                       and combining the default sandbox settings files.
          --no-patch   Do not append $PROJ_DIR/notes and $PROJ_DIR/deps to
                       filesystem.allowWrite after loading the selected
                       settings.

        Settings resolution:
          default      For each directory below, load <command>.json when it
                       exists there; otherwise try the aliased command's
                       settings file, then fall back to default.json.
                       Then merge the existing files in this order:
                         1. $HOME/.agm/sandbox/<command>.json
                            fallback: $HOME/.agm/sandbox/default.json
                         2. $PROJ_DIR/config/sandbox/<command>.json
                            fallback: $PROJ_DIR/config/sandbox/default.json
                         3. ./.sandbox/<command>.json
                            fallback: ./.sandbox/default.json
                       Later files override earlier ones. network and
                       filesystem are merged by key; ignoreViolations replaces
                       the earlier value; enabled and
                       enableWeakerNestedSandbox are overridden when set.
          -f SETTINGS  Skip default discovery and use SETTINGS as-is.

        Automatic patching:
          Unless --no-patch is set, agm adds $PROJ_DIR/notes and
          $PROJ_DIR/deps to filesystem.allowWrite when PROJ_DIR is set.
    """),
    "tmux": textwrap.dedent("""\
        agm tmux new    [-d] [-n PANES] [SESSION]
        agm tmux layout PANES WINDOW_ID WIDTH HEIGHT

        Tmux session and layout management.

        Options:
          agm tmux new -d
              Create the session without attaching to it.
          agm tmux new -n PANES
              Create the session with PANES panes.
    """),
    "help": textwrap.dedent("""\
        agm help [COMMAND...]

        Show help information for commands and subcommands.
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


def help_text_for(command: str) -> str | None:
    canonical = _HELP_ALIASES.get(command, command)
    return _HELP_TEXTS.get(canonical)


def print_overview() -> None:
    print(_overview_text(), end="")


def print_command_help(command: str) -> None:
    text = help_text_for(command)
    if text is None:
        print(f"agm: unknown command '{command}'", file=sys.stderr)
        print("\nRun 'agm help' to see available commands.", file=sys.stderr)
        raise SystemExit(1)
    print(text, end="")


def _subparsers_action(
    parser: argparse.ArgumentParser,
) -> _SubParsersActionAlias | None:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return cast(_SubParsersActionAlias, action)
    return None


def _resolve_parser(command_path: Sequence[str]) -> _HelpTextArgumentParser:
    parser = cast(_HelpTextArgumentParser, build_parser())
    current = parser
    for command in command_path:
        subparsers = _subparsers_action(current)
        if subparsers is None or command not in subparsers.choices:
            raise ValueError(f"unknown command path: {' '.join(command_path)}")
        current = cast(_HelpTextArgumentParser, subparsers.choices[command])
    return current


def print_help_for_command_path(
    command_path: Sequence[str],
    file: _Writeable | None = None,
) -> None:
    _resolve_parser(command_path).print_help(file)


def exit_with_usage_error(command_path: Sequence[str], message: str, *, exit_code: int = 1) -> None:
    print(message, file=sys.stderr)
    print(file=sys.stderr)
    print_help_for_command_path(command_path, file=sys.stderr)
    raise SystemExit(exit_code)


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
    empty_help_command: list[str] = []
    help_parser.add_argument(
        "help_command",
        nargs="*",
        default=empty_help_command,
        metavar="command",
    )

    open_parser = subparsers.add_parser(
        "open",
        help="Open a tmux session for a project checkout",
        help_text=_HELP_TEXTS["open"],
    )
    open_parser.add_argument(
        "-d",
        "--detached",
        dest="detached",
        action="store_true",
        default=False,
        help="create the tmux session without attaching to it",
    )
    open_parser.add_argument(
        "-n",
        dest="pane_count",
        metavar="pane_count",
        default=None,
        help="create the session with this many panes",
    )
    open_parser.add_argument(
        "-p",
        dest="parent",
        metavar="parent",
        default=None,
        help="base a newly created branch worktree on this checkout instead of repo/",
    )
    open_parser.add_argument(
        "branch",
        metavar="target",
        help="repo, an existing branch, or a branch name to create and open",
    )

    br_parser = subparsers.add_parser(
        "br",
        help="Branch operations (alias for 'branch')",
        help_text=help_text_for("br"),
    )
    branch_parser = subparsers.add_parser(
        "branch",
        help="Branch operations",
        help_text=help_text_for("branch"),
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
        help_text=help_text_for("config"),
    )
    config_sub = config_parser.add_subparsers(
        dest="config_command",
        parser_class=_HelpTextArgumentParser,
    )
    for name in ("cp", "copy"):
        current = config_sub.add_parser(
            name,
            help="Copy configuration files",
            description="Copy known project config files into an existing target directory.",
        )
        current.add_argument(
            "-d",
            dest="project_dir",
            metavar="project-dir",
            default=None,
            help="read shared config from this project instead of auto-detecting it",
        )
        current.add_argument(
            "dirname",
            help="existing target directory that will receive copied config files",
        )

    for wt_name in ("wt", "worktree"):
        wt_parser = subparsers.add_parser(
            wt_name,
            help="Git worktree management",
            help_text=help_text_for(wt_name),
        )
        wt_sub = wt_parser.add_subparsers(dest="wt_command", parser_class=_HelpTextArgumentParser)
        wt_new = wt_sub.add_parser(
            "new",
            help="Create a new branch worktree or check out an existing branch",
            description="Create a branch worktree under the default worktrees directory or DIR.",
        )
        wt_new.add_argument(
            "-d",
            dest="worktrees_dir",
            metavar="dir",
            default=None,
            help="create the worktree under DIR instead of the default worktrees location",
        )
        wt_new.add_argument("branch", help="branch name to create or check out")
        wt_sub.add_parser(
            "setup",
            help="Run setup scripts for the current checkout",
            description="Run configured setup scripts for the current repo or worktree checkout.",
        )
        for rm_name in ("rm", "remove"):
            current = wt_sub.add_parser(
                rm_name,
                help="Remove a worktree",
                description="Remove a worktree and delete its local branch.",
            )
            current.add_argument(
                "-f",
                dest="force",
                action="store_true",
                default=False,
                help="force removal even when git reports uncommitted or locked state",
            )
            current.add_argument("branch", help="branch whose worktree should be removed")

    dep_parser = subparsers.add_parser(
        "dep",
        help="Manage project dependency checkouts",
        help_text=help_text_for("dep"),
    )
    dep_sub = dep_parser.add_subparsers(dest="dep_command", parser_class=_HelpTextArgumentParser)
    dep_new = dep_sub.add_parser(
        "new",
        help="Clone a new dependency",
        description="Clone a dependency into deps/ using its default branch or BRANCH.",
    )
    dep_new.add_argument(
        "-b",
        dest="branch",
        metavar="branch",
        default=None,
        help="clone BRANCH instead of the dependency's default branch",
    )
    dep_new.add_argument("repo_url", metavar="repo-url", help="git URL for the dependency")
    dep_switch = dep_sub.add_parser(
        "switch",
        help="Switch a dependency branch",
        description="Add a dependency worktree for BRANCH under deps/DEP/.",
    )
    dep_switch.add_argument(
        "-b",
        dest="create_branch",
        action="store_true",
        default=False,
        help="create BRANCH from the dependency's default branch before adding the worktree",
    )
    dep_switch.add_argument("dep", help="dependency name under deps/")
    dep_switch.add_argument("branch", help="branch to check out or create")
    dep_rm = dep_sub.add_parser(
        "rm",
        help="Remove a dependency worktree or repo",
        description=(
            "Remove a dependency worktree by DEP/BRANCH, or remove the main checkout "
            "with DEP/repo, DEP/MAIN_BRANCH, or --all DEP."
        ),
    )
    dep_rm.add_argument(
        "--all",
        dest="all",
        action="store_true",
        default=False,
        help="remove the entire dependency directory; target must be DEP",
    )
    dep_rm.add_argument(
        "target",
        help="dependency target: DEP/BRANCH, DEP/repo, DEP/MAIN_BRANCH, or DEP with --all",
    )

    subparsers.add_parser(
        "fetch",
        help="Fetch the repo and dependencies",
        help_text=help_text_for("fetch"),
    )

    init_parser = subparsers.add_parser(
        "init",
        help="Initialize a new project",
        help_text=help_text_for("init"),
    )
    init_parser.add_argument(
        "-b",
        dest="branch",
        metavar="branch",
        default=None,
        help="clone BRANCH when a repository URL is provided",
    )
    init_parser.add_argument(
        "positional",
        nargs="+",
        metavar="arg",
        help="PROJECT_NAME, or PROJECT_NAME plus REPO_URL, or REPO_URL alone",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run a command inside an Anthropic Sandbox Runtime",
        help_text=help_text_for("run"),
    )
    run_parser.add_argument(
        "--no-patch",
        dest="no_patch",
        action="store_true",
        default=False,
        help="skip appending $PROJ_DIR/notes and $PROJ_DIR/deps to allowWrite",
    )
    run_parser.add_argument(
        "-f",
        dest="settings_file",
        metavar="settings.json",
        default=None,
        help="use this sandbox settings file directly instead of default discovery",
    )
    run_parser.add_argument(
        "run_command",
        nargs=argparse.REMAINDER,
        metavar="command",
        help="command and arguments to execute inside the sandbox",
    )

    tmux_parser = subparsers.add_parser(
        "tmux",
        help="Tmux session and layout management",
        help_text=help_text_for("tmux"),
    )
    tmux_sub = tmux_parser.add_subparsers(dest="tmux_command", parser_class=_HelpTextArgumentParser)
    tmux_new = tmux_sub.add_parser(
        "new",
        help="Create a new tmux session",
        description="Create a tmux session, optionally detached and with a chosen pane count.",
    )
    tmux_new.add_argument(
        "-d",
        "--detach",
        dest="detach",
        action="store_true",
        default=False,
        help="create the session without attaching to it",
    )
    tmux_new.add_argument(
        "-n",
        dest="pane_count",
        metavar="pane_count",
        default=None,
        help="create the session with this many panes",
    )
    tmux_new.add_argument(
        "session_name",
        nargs="?",
        default=None,
        help="optional tmux session name",
    )
    tmux_layout = tmux_sub.add_parser(
        "layout",
        help="Apply a tiled pane layout",
        description="Apply AGM's tiled pane layout to an existing tmux window.",
    )
    tmux_layout.add_argument("pane_count", help="number of panes to arrange")
    tmux_layout.add_argument("window_id", help="tmux window id, for example @1")
    tmux_layout.add_argument("width", help="window width in cells")
    tmux_layout.add_argument("height", help="window height in cells")
    return parser
