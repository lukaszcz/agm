"""Help text and usage utilities for AGM's Typer CLI."""

from __future__ import annotations

import sys
import textwrap
from collections.abc import Sequence
from typing import Protocol


class _Writeable(Protocol):
    def write(self, data: str) -> object: ...


_HELP_TEXTS: dict[str, str] = {
    "open": textwrap.dedent("""\
        agm open [-d|--detach] [-n|--num-panes PANES] [-p|--parent PARENT] TARGET

        Open a tmux session for a project worktree, creating or checking out a branch as needed.

        Options:
          -d, --detach            Create the tmux session without attaching to it.
          -n, --num-panes PANES   Create the session with PANES panes.
          -p, --parent PARENT     Base a newly created branch worktree on PARENT instead of
                                  the main checkout's current branch.

        Behavior:
          repo           Open the main checkout session.
          default branch Open the main checkout session when TARGET matches the
                         branch currently checked out in the main checkout.
          existing wt    Open the tmux session for an existing branch worktree.
          existing branch Check out BRANCH into a worktree, then open it.
          missing branch  Create BRANCH from PARENT/current branch, then open it.

        Examples:
          agm open repo
          agm open -d repo
          agm open main
          agm open feat/login
          agm open --num-panes 4 feat/login
          agm open --parent main feat/search
    """),
    "close": textwrap.dedent("""\
        agm close BRANCH

        Close a project session for a branch worktree.

        Remove the branch worktree via agm wt rm, then kill
        the corresponding tmux session.
    """),
    "init": textwrap.dedent("""\
        agm init [--embedded | --workspace] [-b|--branch BRANCH] PROJECT_NAME
        agm init [--embedded | --workspace] [-b|--branch BRANCH] [PROJECT_NAME] REPO_URL

        Initialize a new project directory. When REPO_URL is provided, agm also
        clones it into repo/ by default, or into the project root with
        --embedded. If PROJECT_NAME is omitted in that form, it is derived from
        the repo URL. Without an explicit layout flag, agm chooses the embedded
        layout when the target project directory is already a git repo;
        otherwise it chooses the workspace layout.

        Options:
          --embedded   Force the embedded layout with AGM data under .agm/.
          --workspace  Force the workspace layout with repo/, deps/, notes/,
                       worktrees/, and config/ under the project root.
          -b, --branch BRANCH
                       Clone this branch when REPO_URL is provided.
    """),
    "fetch": textwrap.dedent("""\
        agm fetch

        Fetch the main repository and all checked-out dependencies, then create
        missing local tracking branches for origin branches not merged into
        origin/main in each repo.
    """),
    "loop": textwrap.dedent("""\
        agm loop [CMD] [-c|--command COMMAND] [--tasks-dir DIR] [--no-log|--log-file PATH]

        Repeatedly run a prompt command against ``loop.md`` until the command
        returns only ``COMPLETE`` after whitespace is removed.

        Command config:
          [loop] command = "claude -p" in config.toml sets the default command
          prefix. [loop] tasks_dir = ".agent-files/tasks" sets the tasks
          directory checked for ``PROGRESS.md``. ``agm loop CMD`` selects
          ``[loop.CMD]`` overrides; those values override ``[loop]``. If
          ``[loop.CMD].command`` is unset, AGM uses ``CMD`` as the command
          prefix. ``agm loop --command "..."`` and ``agm loop --tasks-dir ...``
          override those values.

        Behavior:
          Appends ``@<resolved-loop-prompt>`` as the final argument to the
          selected command.
          Creates a ``loop-YYYYMMDD-HHMMSS.log`` file in the current directory
          by default, or writes to ``--log-file PATH``. ``--no-log`` disables
          file logging entirely. The command prints each step header and stops
          when the response is ``COMPLETE``.
    """),
    "config": textwrap.dedent("""\
        agm config copy DIRNAME
        agm config cp   DIRNAME

        Copy project configuration files into an existing target directory.
    """),
    "worktree": textwrap.dedent("""\
        agm worktree new      [-d|--dir DIR] BRANCH
        agm worktree setup
        agm worktree remove   [-f|--force] BRANCH
        agm wt new | wt setup | wt rm

        Low-level git worktree management.

        Options:
          agm worktree new --dir DIR
              Create the worktree under DIR instead of the default project
              worktrees directory.
          agm worktree remove --force
              Force removal even when git reports uncommitted or locked state.
    """),
    "dep": textwrap.dedent("""\
        agm dep new    [-b|--branch BRANCH] REPO_URL
        agm dep rm     [--all] DEP | DEP/BRANCH | DEP/repo | DEP/MAIN_BRANCH
        agm dep switch [-b|--branch] DEP BRANCH

        Manage project dependency checkouts under the project's dependency directory.

        Options:
          agm dep new --branch BRANCH
              Clone BRANCH instead of the dependency's default branch.
          agm dep rm --all
              Remove the entire dependency directory, including the main repo
              checkout and any linked worktrees.
          agm dep switch --branch
              Create BRANCH from the dependency's default branch before adding
              the new worktree.

        Targets:
          DEP/BRANCH      Remove a dependency worktree for BRANCH.
          DEP/repo        Remove the main dependency checkout.
          DEP/MAIN_BRANCH Remove the main dependency checkout by branch name.
    """),
    "run": textwrap.dedent("""\
        agm run [--no-sandbox] [--no-patch] [--memory LIMIT] [-f|--file SETTINGS] COMMAND [ARGS...]

        Run a command inside an Anthropic Sandbox Runtime container.

        Command config:
          <install-prefix>/.agm/config.toml is loaded when present,
          otherwise $HOME/.agm/config.toml is used, followed by the
          project config.toml and ./.agm/config.toml.
          [run.<command>] alias = "<other-command>" makes
          "agm run <command>" execute <other-command> instead.

        Options:
          --no-sandbox
                       Run COMMAND directly without wrapping it in srt.
                       This skips sandbox settings discovery and patching.
          -f, --file SETTINGS
                       Use this settings file directly instead of discovering
                       and combining the default sandbox settings files.
          --memory LIMIT
                       Wrap srt in systemd-run --user --scope and set
                       MemoryMax=LIMIT. The default is 20G. Values <= 0
                       disable memory limiting.
          --no-patch   Do not append the project notes and deps directories to
                       filesystem.allowWrite after loading the selected
                       settings.

        Settings resolution:
          default      For each directory below, load <command>.json when it
                       exists there; otherwise try the aliased command's
                       settings file, then fall back to default.json.
                       Then merge the existing files in this order:
                         1. $HOME/.agm/sandbox/<command>.json
                            fallback: $HOME/.agm/sandbox/default.json
                         2. the project sandbox config directory
                         3. ./.sandbox/<command>.json
                            fallback: ./.sandbox/default.json
                       Later files override earlier ones. network and
                       filesystem are merged by key; ignoreViolations replaces
                       the earlier value; enabled and
                       enableWeakerNestedSandbox are overridden when set.
          -f, --file SETTINGS
                       Skip default discovery and use SETTINGS as-is.

        Automatic patching:
          Unless --no-patch is set, agm adds the project notes and deps
          directories to filesystem.allowWrite when PROJ_DIR is set.
    """),
    "tmux": textwrap.dedent("""\
        agm tmux open   [-d|--detach] [-n|--num-panes PANES] [SESSION]
        agm tmux close  SESSION
        agm tmux layout PANES [-w|--window WINDOW_ID]

        Tmux session and layout management.

        Options:
          agm tmux open --detach
              Create the session without attaching to it.
          agm tmux open --num-panes PANES
              Create the session with PANES panes.
    """),
    "help": textwrap.dedent("""\
        agm help [COMMAND...]

        Show help information for commands and subcommands.

        Global options:
          --install-completion  Install shell completion for the current shell.
          --show-completion     Print the shell completion script.
    """),
}

_HELP_ALIASES: dict[str, str] = {
    "wt": "worktree",
    "cp": "config",
    "copy": "config",
}

_COMMAND_OVERVIEW: list[tuple[str, str]] = [
    ("open", "Open a project session"),
    ("close", "Close a project session"),
    ("init", "Initialize a new project"),
    ("dep", "Manage project dependency checkouts"),
    (
        "fetch",
        "Fetch upstream changes for the repo and all dependencies",
    ),
    ("loop", "Run the loop prompt until completion"),
    ("run", "Run a command in a sandbox"),
    ("config", "Manage project configuration files"),
    ("worktree", "Git worktree management"),
    ("tmux", "Tmux session and layout management"),
    ("help", "Show help for a command"),
]

_PATH_HELP_TEXTS: dict[tuple[str, ...], str] = {
    ("config", "cp"): textwrap.dedent("""\
        agm config cp DIRNAME

        Copy known project config files into an existing target directory.
    """),
    ("config", "copy"): textwrap.dedent("""\
        agm config copy DIRNAME

        Copy known project config files into an existing target directory.
    """),
    ("wt", "new"): textwrap.dedent("""\
        agm wt new [-d|--dir DIR] BRANCH

        Create a new branch worktree or check out an existing branch.
    """),
    ("wt", "setup"): textwrap.dedent("""\
        agm wt setup

        Run configured setup scripts for the current repo or worktree checkout.
    """),
    ("wt", "rm"): textwrap.dedent("""\
        agm wt rm [-f|--force] BRANCH

        Remove a worktree and delete its local branch.
    """),
    ("worktree", "new"): textwrap.dedent("""\
        agm worktree new [-d|--dir DIR] BRANCH

        Create a new branch worktree or check out an existing branch.
    """),
    ("worktree", "setup"): textwrap.dedent("""\
        agm worktree setup

        Run configured setup scripts for the current repo or worktree checkout.
    """),
    ("worktree", "remove"): textwrap.dedent("""\
        agm worktree remove [-f|--force] BRANCH

        Remove a worktree and delete its local branch.
    """),
    ("dep", "new"): textwrap.dedent("""\
        agm dep new [-b|--branch BRANCH] REPO_URL

        Clone a dependency into deps/ using its default branch or BRANCH.
    """),
    ("dep", "switch"): textwrap.dedent("""\
        agm dep switch [-b|--branch] DEP BRANCH

        Add a dependency worktree for BRANCH under deps/DEP/.
    """),
    ("dep", "rm"): textwrap.dedent("""\
        agm dep rm [--all] TARGET

        Remove a dependency worktree by DEP/BRANCH, or remove the main checkout
        with DEP/repo, DEP/MAIN_BRANCH, or --all DEP.
    """),
    ("tmux", "open"): textwrap.dedent("""\
        agm tmux open [-d|--detach] [-n|--num-panes PANES] [SESSION]

        Create a tmux session, optionally detached and with a chosen pane count.
    """),
    ("tmux", "close"): textwrap.dedent("""\
        agm tmux close SESSION

        Kill an existing tmux session by name.
    """),
    ("tmux", "layout"): textwrap.dedent("""\
        agm tmux layout PANES [-w|--window WINDOW_ID]

        Apply AGM's tiled pane layout to the current tmux window.
    """),
}


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
            "Global options:",
            "  --install-completion  Install shell completion for the current shell.",
            "  --show-completion     Print the shell completion script.",
            "",
            "Run 'agm help <command>' for detailed help on a specific command.",
        ]
    )
    return "\n".join(lines) + "\n"


def help_text_for(command: str) -> str | None:
    canonical = _HELP_ALIASES.get(command, command)
    return _HELP_TEXTS.get(canonical)


def print_overview(file: _Writeable | None = None) -> None:
    output = sys.stdout if file is None else file
    print(_overview_text(), end="", file=output)


def print_command_help(command: str, file: _Writeable | None = None) -> None:
    text = help_text_for(command)
    if text is None:
        print(f"agm: unknown command '{command}'", file=sys.stderr)
        print("\nRun 'agm help' to see available commands.", file=sys.stderr)
        raise SystemExit(1)
    output = sys.stdout if file is None else file
    print(text, end="", file=output)


def _canonical_command_path(command_path: Sequence[str]) -> tuple[str, ...]:
    if len(command_path) == 1:
        return (_HELP_ALIASES.get(command_path[0], command_path[0]),)
    return tuple(command_path)


def _help_text_for_path(command_path: Sequence[str]) -> str:
    normalized = _canonical_command_path(command_path)
    if len(normalized) == 1:
        text = help_text_for(normalized[0])
        if text is None:
            raise ValueError(f"unknown command path: {' '.join(command_path)}")
        return text
    text = _PATH_HELP_TEXTS.get(tuple(command_path))
    if text is None:
        raise ValueError(f"unknown command path: {' '.join(command_path)}")
    return text


def print_help_for_command_path(
    command_path: Sequence[str],
    file: _Writeable | None = None,
) -> None:
    output = sys.stdout if file is None else file
    print(_help_text_for_path(command_path), end="", file=output)


def exit_with_usage_error(command_path: Sequence[str], message: str, *, exit_code: int = 1) -> None:
    help_text = _help_text_for_path(command_path)
    usage_line, _, _ = help_text.partition("\n")
    print(message, file=sys.stderr)
    print(file=sys.stderr)
    print(f"usage: {usage_line}", file=sys.stderr)
    print(file=sys.stderr)
    print(help_text, end="", file=sys.stderr)
    raise SystemExit(exit_code)
