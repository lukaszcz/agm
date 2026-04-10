"""AGM – unified CLI that dispatches subcommands to underlying shell scripts."""

from __future__ import annotations

import argparse
import sys
import textwrap
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

# ---------------------------------------------------------------------------
# Detailed help text for each command, used by ``agm help <command>``
# ---------------------------------------------------------------------------

_HELP_TEXTS: dict[str, str] = {
    "open": textwrap.dedent("""\
        agm open [-n PANES] [BRANCH]

        Open a tmux session for a project branch. If BRANCH is omitted, the
        current branch is used. The session starts in the worktree directory
        that corresponds to the given branch.

        Options:
          -n PANES   Number of tmux panes to create (default: from config)
          BRANCH     Branch name to open (default: current branch)

        Examples:
          agm open                  # open session for current branch
          agm open feat/login       # open session for feat/login
          agm open -n 4 feat/login  # open with 4 panes
    """),
    "new": textwrap.dedent("""\
        agm new [-n PANES] [-p PARENT] BRANCH

        Create a new branch worktree and immediately open a tmux session for
        it. The new branch is created from PARENT (or the default branch if
        -p is not given).

        Options:
          -n PANES   Number of tmux panes to create (default: from config)
          -p PARENT  Parent branch to fork from (default: default branch)
          BRANCH     Name for the new branch (required)

        Examples:
          agm new feat/search
          agm new -p develop feat/search
          agm new -n 3 -p main feat/search
    """),
    "checkout": textwrap.dedent("""\
        agm checkout [-n PANES] [-p PARENT] BRANCH
        agm co       [-n PANES] [-p PARENT] BRANCH

        Check out an existing branch into a worktree (creating the worktree if
        needed) and open a tmux session for it. If -p is given and the branch
        does not yet exist locally, it is created from PARENT.

        Options:
          -n PANES   Number of tmux panes to create (default: from config)
          -p PARENT  Parent branch for new branch creation
          BRANCH     Branch name to check out (required)

        Examples:
          agm co feat/login
          agm checkout -n 4 feat/login
          agm co -p main feat/new-thing
    """),
    "init": textwrap.dedent("""\
        agm init [-b BRANCH] [PROJECT_NAME] REPO_URL

        Initialize a new project by cloning a repository. If PROJECT_NAME is
        omitted it is derived from the repo URL. An optional -b flag selects
        the branch to check out after cloning.

        Options:
          -b BRANCH       Branch to clone (default: repo default)
          PROJECT_NAME    Directory name for the project (optional)
          REPO_URL        Git repository URL (required)

        Examples:
          agm init https://github.com/org/repo.git
          agm init myproject https://github.com/org/repo.git
          agm init -b develop myproject https://github.com/org/repo.git
    """),
    "fetch": textwrap.dedent("""\
        agm fetch

        Fetch the latest changes for the main repository and for every
        dependency that has a checked-out worktree under deps/. This runs
        ``git fetch`` in each relevant directory.

        This command takes no arguments.

        Examples:
          agm fetch
    """),
    "branch": textwrap.dedent("""\
        agm branch sync
        agm br     sync

        Branch management commands.

        Subcommands:
          sync    Fetch and prune origin, then create local tracking branches
                  for every remote branch that is not yet merged into
                  origin/main.

        Examples:
          agm br sync
          agm branch sync
    """),
    "config": textwrap.dedent("""\
        agm config copy [-d PROJECT_DIR] DIRNAME
        agm config cp   [-d PROJECT_DIR] DIRNAME

        Copy project configuration files into a target directory.

        Subcommands:
          copy (cp)   Copy configuration files to DIRNAME.

        Options:
          -d PROJECT_DIR   Project directory to read config from
          DIRNAME          Target directory for the copied files (required)

        Examples:
          agm config cp mydir
          agm config copy -d /path/to/project target
    """),
    "worktree": textwrap.dedent("""\
        agm worktree checkout [-b BRANCH] [-d DIR] [BRANCH]
        agm worktree new      [-d DIR] BRANCH
        agm worktree remove   [-f] BRANCH
        agm wt co | wt new | wt rm   (short aliases)

        Low-level git worktree management.

        Subcommands:
          checkout (co)    Check out a branch into a worktree directory.
                           Use -b to create a new branch at the same time.
          new              Create a new branch and its worktree (shorthand
                           for ``wt co -b BRANCH``).
          remove (rm)      Remove a worktree and delete the local branch.
                           Use -f to force removal of a dirty worktree.

        Options (checkout):
          -b BRANCH   Create a new branch instead of checking out existing one
          -d DIR      Directory to store worktrees (default: .worktrees)
          BRANCH      Branch name (optional with -b)

        Options (new):
          -d DIR      Directory to store worktrees (default: .worktrees)
          BRANCH      New branch name (required)

        Options (remove):
          -f          Force removal even if worktree is dirty
          BRANCH      Branch name to remove (required)

        Examples:
          agm wt co feat/login
          agm wt new feat/search
          agm wt rm -f old-branch
          agm worktree checkout -b feat/new -d /custom/dir
    """),
    "dep": textwrap.dedent("""\
        agm dep new    [-b BRANCH] REPO_URL
        agm dep switch [-b] DEP BRANCH

        Manage project dependency checkouts under deps/.

        Subcommands:
          new       Clone a new dependency repository. The clone is placed
                    under deps/<repo-name>/<branch>/.
          switch    Switch an existing dependency to a different branch by
                    adding a worktree. Use -b to create a new branch from
                    the dependency's default branch.

        Options (new):
          -b BRANCH    Branch to clone (default: repo default)
          REPO_URL     Git repository URL (required)

        Options (switch):
          -b           Create a new branch instead of switching to existing
          DEP          Dependency name (directory under deps/)
          BRANCH       Branch name (required)

        Examples:
          agm dep new https://github.com/org/lib.git
          agm dep new -b v2 https://github.com/org/lib.git
          agm dep switch mylib feat/update
          agm dep switch -b mylib feat/new-thing
    """),
    "run": textwrap.dedent("""\
        agm run [--no-patch] [-f SETTINGS] COMMAND [ARGS...]

        Run a command inside an Anthropic Sandbox Runtime container. Settings
        are loaded from ~/.sandbox/default.json and/or ./.sandbox/default.json
        (local overrides global with section-aware merging).

        If the PROJ_DIR environment variable is set, the settings file is
        temporarily patched to add $PROJ_DIR/notes and $PROJ_DIR/issues to
        the filesystem.allowWrite list. Use --no-patch to disable this.

        Options:
          --no-patch          Disable automatic PROJ_DIR patching
          -f SETTINGS         Use an explicit settings file instead of defaults
          COMMAND [ARGS...]   The command to run inside the sandbox

        Examples:
          agm run npm test
          agm run -f .sandbox/ci.json make build
          agm run --no-patch python3 script.py
    """),
    "tmux": textwrap.dedent("""\
        agm tmux new    [-d] [-n PANES] [SESSION]
        agm tmux layout PANES WINDOW_ID WIDTH HEIGHT

        Tmux session and layout management.

        Subcommands:
          new       Create a new tmux session with a tiled pane layout.
                    Use -d to create it detached.
          layout    Apply a tiled pane layout to an existing tmux window.
                    This is mainly used internally.

        Options (new):
          -d, --detach   Create the session without attaching to it
          -n PANES       Number of panes to create
          SESSION        Session name (default: derived from directory)

        Options (layout):
          PANES       Number of panes (required)
          WINDOW_ID   Tmux window identifier, e.g. @1 (required)
          WIDTH       Window width in columns (required)
          HEIGHT      Window height in rows (required)

        Examples:
          agm tmux new
          agm tmux new -d -n 4 my-session
          agm tmux layout 4 @1 200 50
    """),
    "help": textwrap.dedent("""\
        agm help [COMMAND]

        Show help information. Without arguments, lists all available commands
        with short descriptions. With a COMMAND argument, shows detailed help
        for that specific command.

        Examples:
          agm help            # list all commands
          agm help open       # detailed help for 'open'
          agm help worktree   # detailed help for 'worktree'
    """),
}

# Aliases that map to canonical help entries
_HELP_ALIASES: dict[str, str] = {
    "br": "branch",
    "wt": "worktree",
    "co": "checkout",
    "cp": "config",
    "copy": "config",
}

# Short one-line descriptions for the overview listing
_COMMAND_OVERVIEW: list[tuple[str, str]] = [
    ("open", "Open a tmux session for a project branch"),
    ("new", "Create a new branch worktree and open a tmux session"),
    ("checkout (co)", "Check out a branch into a worktree and open a tmux session"),
    ("init", "Initialize a new project by cloning a repository"),
    ("fetch", "Fetch latest changes for the repo and all dependencies"),
    ("branch (br)", "Branch management (sync remote tracking branches)"),
    ("config", "Copy project configuration files"),
    ("worktree (wt)", "Low-level git worktree management (checkout, new, remove)"),
    ("dep", "Manage project dependency checkouts (new, switch)"),
    ("run", "Run a command inside an Anthropic Sandbox Runtime"),
    ("tmux", "Tmux session and layout management (new, layout)"),
    ("help", "Show help for a command"),
]


def _print_overview() -> None:
    """Print the short command overview."""
    print("agm - Agent Management Framework\n")
    print("Usage: agm <command> [options] [args]\n")
    print("Commands:")
    width = max(len(name) for name, _ in _COMMAND_OVERVIEW)
    for name, desc in _COMMAND_OVERVIEW:
        print(f"  {name:<{width + 2}} {desc}")
    print()
    print("Run 'agm help <command>' for detailed help on a specific command.")
    print("Run 'agm <command> --help' for option summary.")


def _print_command_help(command: str) -> None:
    """Print detailed help for a single command."""
    canonical = _HELP_ALIASES.get(command, command)
    text = _HELP_TEXTS.get(canonical)
    if text is None:
        print(f"agm: unknown command '{command}'", file=sys.stderr)
        print(f"\nRun 'agm help' to see available commands.", file=sys.stderr)
        sys.exit(1)
    print(text, end="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agm",
        description="Agent Management Framework — manage worktrees, project dependencies, configuration and tmux sessions.",
        epilog="Run 'agm help <command>' for detailed help on a specific command.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- agm help [command] ---
    help_parser = subparsers.add_parser(
        "help",
        help="Show help for a command",
        description="Show help information. Without arguments, lists all commands. With a command name, shows detailed help.",
    )
    help_parser.add_argument("help_command", nargs="?", default=None, metavar="command",
                             help="Command to show help for")

    # --- agm {br|branch} sync ---
    br_parser = subparsers.add_parser(
        "br",
        help="Branch operations (alias for 'branch')",
        description="Branch management commands. Use 'agm br sync' to synchronize remote tracking branches.",
    )
    br_alias = subparsers.add_parser(
        "branch",
        help="Branch operations",
        description="Branch management commands. Use 'agm branch sync' to synchronize remote tracking branches.",
    )
    for p in (br_parser, br_alias):
        br_sub = p.add_subparsers(dest="br_command", required=True)
        br_sub.add_parser(
            "sync",
            help="Fetch/prune origin and create local tracking branches for unmerged remote branches",
            description="Fetch and prune origin, then create local tracking branches for every remote branch not yet merged into origin/main.",
        )

    # --- agm config {cp|copy} ---
    config_parser = subparsers.add_parser(
        "config",
        help="Copy project configuration files",
        description="Configuration file management. Copy project config files into a target directory.",
    )
    config_sub = config_parser.add_subparsers(dest="config_command", required=True)
    for name in ("cp", "copy"):
        cp_help = "Copy configuration files to a target directory"
        cp_parser = config_sub.add_parser(
            name,
            help=cp_help + (" (alias for 'copy')" if name == "cp" else ""),
            description=cp_help + ". Reads config from the project directory and writes it to DIRNAME.",
        )
        cp_parser.add_argument("-d", dest="project_dir", metavar="project-dir", default=None,
                               help="Project directory to read configuration from")
        cp_parser.add_argument("dirname", help="Target directory for the copied configuration files")

    # --- agm {wt|worktree} {co|checkout} ---
    # --- agm {wt|worktree} new ---
    # --- agm {wt|worktree} {rm|remove} ---
    for wt_name in ("wt", "worktree"):
        is_alias = wt_name == "wt"
        wt_parser = subparsers.add_parser(
            wt_name,
            help="Git worktree management" + (" (alias for 'worktree')" if is_alias else " (checkout, new, remove)"),
            description="Low-level git worktree management: create, check out, and remove worktrees.",
        )
        wt_sub = wt_parser.add_subparsers(dest="wt_command", required=True)

        for co_name in ("co", "checkout"):
            co_is_alias = co_name == "co"
            wt_co = wt_sub.add_parser(
                co_name,
                help="Check out a branch into a worktree" + (" (alias for 'checkout')" if co_is_alias else ""),
                description="Check out an existing branch into a worktree directory. Use -b to create a new branch at the same time.",
            )
            wt_co.add_argument("-b", dest="new_branch", metavar="branch-name", default=None,
                               help="Create a new branch instead of checking out an existing one")
            wt_co.add_argument("-d", dest="worktrees_dir", metavar="dir", default=None,
                               help="Directory to store worktrees (default: .worktrees)")
            wt_co.add_argument("branch", nargs="?", default=None,
                               help="Branch name to check out")

        wt_new = wt_sub.add_parser(
            "new",
            help="Create a new branch and its worktree",
            description="Create a new branch and its worktree directory. Equivalent to 'agm wt co -b BRANCH'.",
        )
        wt_new.add_argument("-d", dest="worktrees_dir", metavar="dir", default=None,
                            help="Directory to store worktrees (default: .worktrees)")
        wt_new.add_argument("branch", help="Name for the new branch")

        for rm_name in ("rm", "remove"):
            rm_is_alias = rm_name == "rm"
            wt_rm = wt_sub.add_parser(
                rm_name,
                help="Remove a worktree and delete the local branch" + (" (alias for 'remove')" if rm_is_alias else ""),
                description="Remove a worktree directory and delete the corresponding local branch. Use -f to force removal of a dirty worktree.",
            )
            wt_rm.add_argument("-f", dest="force", action="store_true", default=False,
                               help="Force removal even if worktree has uncommitted changes")
            wt_rm.add_argument("branch", help="Branch name whose worktree to remove")

    # --- agm dep ---
    dep_parser = subparsers.add_parser(
        "dep",
        help="Manage project dependency checkouts (new, switch)",
        description="Manage dependency checkouts under deps/. Clone new dependencies or switch existing ones to different branches.",
    )
    dep_sub = dep_parser.add_subparsers(dest="dep_command", required=True)

    dep_new = dep_sub.add_parser(
        "new",
        help="Clone a new dependency repository into deps/",
        description="Clone a new dependency repository. The clone is placed under deps/<repo-name>/<branch>/.",
    )
    dep_new.add_argument("-b", dest="branch", metavar="branch", default=None,
                         help="Branch to clone (default: repository default branch)")
    dep_new.add_argument("repo_url", metavar="repo-url", help="Git repository URL to clone")

    dep_switch = dep_sub.add_parser(
        "switch",
        help="Switch a dependency to a different branch via worktree",
        description="Switch an existing dependency to a different branch by adding a worktree. Use -b to create a new branch from the default.",
    )
    dep_switch.add_argument("-b", dest="create_branch", action="store_true", default=False,
                            help="Create a new branch instead of switching to an existing one")
    dep_switch.add_argument("dep", help="Dependency name (directory name under deps/)")
    dep_switch.add_argument("branch", help="Branch name to switch to or create")

    # --- agm fetch ---
    subparsers.add_parser(
        "fetch",
        help="Fetch latest changes for the repo and all dependencies",
        description="Fetch the latest changes for the main repository and for every dependency that has a checked-out worktree under deps/.",
    )

    # --- agm init ---
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize a new project by cloning a repository",
        description="Initialize a new project by cloning a repository. If project name is omitted, it is derived from the repo URL.",
    )
    init_parser.add_argument("-b", dest="branch", metavar="branch", default=None,
                             help="Branch to check out after cloning (default: repo default)")
    init_parser.add_argument("positional", nargs="+", metavar="arg",
                             help="[project-name] repo-url")

    # --- agm open ---
    open_parser = subparsers.add_parser(
        "open",
        help="Open a tmux session for a project branch",
        description="Open a tmux session for a project branch. If no branch is given, the current branch is used.",
    )
    open_parser.add_argument("-n", dest="pane_count", metavar="pane_count", default=None,
                             help="Number of tmux panes to create")
    open_parser.add_argument("branch", nargs="?", default=None, help="Branch name to open (default: current branch)")

    # --- agm new ---
    new_parser = subparsers.add_parser(
        "new",
        help="Create a new branch worktree and open a tmux session",
        description="Create a new branch worktree and immediately open a tmux session for it. The branch is created from PARENT or the default branch.",
    )
    new_parser.add_argument("-n", dest="pane_count", metavar="pane_count", default=None,
                            help="Number of tmux panes to create")
    new_parser.add_argument("-p", dest="parent", metavar="parent", default=None,
                            help="Parent branch to fork from (default: default branch)")
    new_parser.add_argument("branch", help="Name for the new branch")

    # --- agm {co|checkout} ---
    for co_name in ("co", "checkout"):
        is_alias = co_name == "co"
        co_parser = subparsers.add_parser(
            co_name,
            help="Check out a branch and open a tmux session" + (" (alias for 'checkout')" if is_alias else ""),
            description="Check out an existing branch into a worktree and open a tmux session. Creates the worktree if it does not exist.",
        )
        co_parser.add_argument("-n", dest="pane_count", metavar="pane_count", default=None,
                               help="Number of tmux panes to create")
        co_parser.add_argument("-p", dest="parent", metavar="parent", default=None,
                               help="Parent branch if the branch needs to be created")
        co_parser.add_argument("branch", help="Branch name to check out")

    # --- agm run ---
    run_parser = subparsers.add_parser(
        "run",
        help="Run a command inside an Anthropic Sandbox Runtime",
        description="Run a command inside an Anthropic Sandbox Runtime container. Settings are loaded from ~/.sandbox/default.json and/or ./.sandbox/default.json.",
    )
    run_parser.add_argument("--no-patch", dest="no_patch", action="store_true", default=False,
                            help="Disable automatic PROJ_DIR-based settings patching")
    run_parser.add_argument("-f", dest="settings_file", metavar="settings.json", default=None,
                            help="Use an explicit settings file instead of defaults")
    run_parser.add_argument("run_command", nargs=argparse.REMAINDER, metavar="command",
                            help="Command (and arguments) to run in the sandbox")

    # --- agm tmux new ---
    # --- agm tmux layout ---
    tmux_parser = subparsers.add_parser(
        "tmux",
        help="Tmux session and layout management (new, layout)",
        description="Tmux session and layout management. Create new sessions or apply pane layouts.",
    )
    tmux_sub = tmux_parser.add_subparsers(dest="tmux_command", required=True)

    tmux_new = tmux_sub.add_parser(
        "new",
        help="Create a new tmux session with a tiled pane layout",
        description="Create a new tmux session with a tiled pane layout. Use -d to create the session without attaching.",
    )
    tmux_new.add_argument("-d", "--detach", dest="detach", action="store_true", default=False,
                          help="Create the session without attaching to it")
    tmux_new.add_argument("-n", dest="pane_count", metavar="pane_count", default=None,
                          help="Number of panes to create")
    tmux_new.add_argument("session_name", nargs="?", default=None, help="Session name (default: derived from directory)")

    tmux_layout = tmux_sub.add_parser(
        "layout",
        help="Apply a tiled pane layout to an existing tmux window",
        description="Apply a tiled pane layout to an existing tmux window. Mainly used internally.",
    )
    tmux_layout.add_argument("pane_count", help="Number of panes")
    tmux_layout.add_argument("window_id", help="Tmux window identifier (e.g. @1)")
    tmux_layout.add_argument("width", help="Window width in columns")
    tmux_layout.add_argument("height", help="Window height in rows")

    return parser


def dispatch(args: argparse.Namespace) -> NoReturn:
    """Route parsed arguments to the appropriate command handler."""
    cmd: str = args.command

    if cmd == "help":
        if args.help_command is None:
            _print_overview()
        else:
            _print_command_help(args.help_command)
        sys.exit(0)

    elif cmd in ("br", "branch"):
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
