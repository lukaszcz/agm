"""Help text and usage utilities for AGM's Typer CLI."""

from __future__ import annotations

import sys
import textwrap
from collections.abc import Sequence
from typing import NoReturn, Protocol

from agm.cli_support.execution_options import execution_options_section
from agm.command_catalog import COMMAND_OVERVIEW
from agm.core.env import help_width
from agm.util.text import first_paragraph, format_description_column


class _Writeable(Protocol):
    def write(self, data: str) -> object: ...


# Option rows shared by the AgL commands' help pages.
_MODULE_PATH_ROW = (
    "  -I, --module-path DIR  Add a module search root (repeatable; relative to the\n"
    "                         working directory).\n"
)
_NO_STDLIB_ROW = "  --no-stdlib            Disable the automatic `import std/prelude::*`.\n"

_HELP_TEXTS: dict[str, str] = {
    "open": textwrap.dedent("""\
        agm open [-d] [-n PANES] [-p PARENT] [--no-fetch] [--dry-run] TARGET

        Open a workspace's tmux session, creating or checking out TARGET's branch
        as needed. Also: agm workspace open, agm wsp open.

        Options:
          -d, --detach           Do not attach to the session.
          -n, --num-panes PANES  Create the session with PANES panes.
          -p, --parent PARENT    Base a new branch on PARENT instead of the main
                                 workspace's branch.
          --no-fetch             Resolve remote branches from local refs.

        TARGET:
          repo              The main workspace.
          main's branch     The main workspace (its checked-out branch).
          branch workspace  That workspace; an error with --parent.
          existing branch   Checked out into a new worktree; a remote-only branch
                            tracks its remote (error if several remotes carry
                            it); --parent is ignored with a warning.
          new branch        Created from PARENT or the main workspace's branch.

        A workspace whose session is already running is an error; attach to it.

        Examples:
          agm open repo
          agm open -d -n 4 feat/login
          agm open -p main feat/search
    """),
    "close": textwrap.dedent("""\
        agm close [-f] [-D] [--keep-branch] [--keep-workspace] [--dry-run] BRANCH

        Close a branch workspace: remove its worktree and workspace config,
        safe-delete its branch (git branch -d), and kill its tmux session.
        Also: agm workspace close, agm wsp close.

        Options:
          -f, --force       Remove the worktree despite uncommitted or untracked
                            changes; implies -D.
          -D                Force-delete the branch (git branch -D). The worktree is
                            removed only if the deletion would succeed.
          --keep-branch     Keep the local branch.
          --keep-workspace  Keep the worktree and branch; only kill the session.
    """),
    "init": textwrap.dedent("""\
        agm init [--embedded|--split] [--no-git-init] [--no-repo-git]
                 [--no-config-git] [--no-notes-git] [--dry-run] [PROJECT_NAME]
        agm init ... [-b BRANCH] [PROJECT_NAME] REPO_URL
        agm init --clone ... [-b BRANCH] REPO_URL

        Initialize a project in the current directory, or in PROJECT_NAME/. With
        REPO_URL, clone it into repo/ (split) or the project root (embedded).

        The layout defaults to embedded when the project directory is already a
        git repo, else split. An embedded repo without commits gets an initial
        commit of the generated .gitignore.

        Options:
          --embedded           Embedded layout: AGM data under .agm/.
          --split              Split layout: repo/, deps/, notes/, worktrees/,
                               config/.
          --clone              Name the project directory after REPO_URL.
          -b, --branch BRANCH  Clone BRANCH.
          --no-git-init        Create no git repos in repo/, config/, notes/.
          --no-repo-git        Create no git repo in repo/.
          --no-config-git      Create no git repo in config/.
          --no-notes-git       Create no git repo in notes/.
    """),
    "workspace": textwrap.dedent("""\
        agm workspace COMMAND
        agm wsp COMMAND

        Manage workspaces: the main repo and the worktrees under the project's
        worktrees directory, with their config, dependency environment, setup
        scripts, and tmux sessions.

        Commands:
          open         Open a workspace's tmux session (also: agm open)
          close        Close a branch workspace (also: agm close)
          setup        Run the current workspace's setup scripts
          list         List open workspaces
          shell-regen  Regenerate a session's shell wrapper files
    """),
    "sync": textwrap.dedent("""\
        agm sync COMMAND

        Synchronize the main repo, dependency repos, and their worktrees.

        Commands:
          fetch  Fetch all repos and create missing local tracking branches
          pull   Fetch, then merge in every worktree
    """),
    "config": textwrap.dedent("""\
        agm config COMMAND

        Manage project configuration.

        Commands:
          copy    Copy the project's dot config files into a directory (alias: cp)
          env     Print shell statements that set the workspace environment
          update  Create missing project and workspace config.toml files

        Config files, later ones overriding earlier:
          <install-prefix>/.agm/config.toml
          the AGM home's config.toml ($AGM_HOME, else an installed
            <install-prefix>/.agm, else ~/.agm)
          the project config directory's config.toml
          ./.agm/config.toml

        Path-valued settings, including [modules] lib_root and roots, expand ~
        and %{VAR} from the environment; an unresolved or malformed hole stays
        literal.
    """),
    "worktree": textwrap.dedent("""\
        agm worktree COMMAND
        agm wt COMMAND

        Low-level Git worktree management.

        Commands:
          new     Create a worktree for a branch
          remove  Remove a worktree and its branch (alias: rm)
    """),
    "dep": textwrap.dedent("""\
        agm dep COMMAND

        Manage dependency repos and their worktrees under deps/. Config.toml
        [deps] tables record each workspace's dependency checkouts.

        Commands:
          list    List dependency checkouts
          new     Clone a dependency
          switch  Select or create a dependency checkout
          rm      Remove a dependency checkout (alias: remove)
    """),
    "pkg": textwrap.dedent("""\
        agm pkg COMMAND

        Manage AgL packages in AGM's versioned store.

        Commands:
          init       Create a package
          check      Validate a package
          create     Build a package archive
          install    Install and activate a package
          switch     Activate an installed version
          uninstall  Remove an installed version
          list       List installed packages
          info       Describe an active package
          sync       Install the active packages' Python requirements

        package.toml [commands] tables form multi-word command paths; an entry
        names a program (documented by its @doc) or, without one, a group with
        its own doc. [aliases] maps alternate paths to commands or groups, and
        an alias path also names config tables: rev = "devel review" makes [rev]
        equivalent to [devel.review]. Manifest reference: docs/commands/pkg.md.
    """),
    "run": textwrap.dedent("""\
        agm run [--no-sandbox] [-f SETTINGS] [--no-patch] [--pty|--no-pty]
                [--memory LIMIT|--no-memory-limit] [--swap LIMIT|--no-swap-limit]
                [--dry-run] COMMAND [ARGS...]

        Run COMMAND in an Anthropic Sandbox Runtime (srt) sandbox.
        Sandboxed Codex launches include --no-daemon to keep execution inside srt.

        Options:
          --no-sandbox         Run COMMAND directly, without srt or settings.
          -f, --file SETTINGS  Use the SETTINGS file as-is.
          --no-patch           Do not add the project's notes/, deps/, and repo .git
                               to filesystem.allowWrite (added when PROJ_DIR is set).
          --pty, --no-pty      Relay a pseudo-terminal when stdin and stdout are
                               terminals (default: on).
          --memory LIMIT       Run in a delegated systemd-run --user scope with
                               MemoryMax=LIMIT (default: 32G), exporting
                               SANDBOX_CGROUP.
          --swap LIMIT         Set MemorySwapMax=LIMIT (sandbox default: 0).
          --no-memory-limit    Do not set MemoryMax.
          --no-swap-limit      Do not set MemorySwapMax.

        LIMIT: 0, infinity or unlimited, a percentage, or whitespace-separated
        <number>[KMGTPE][B] groups (case-sensitive).

        config.toml: [run] and [run.COMMAND] set memory, swap, and pty;
        [run.COMMAND] alias = "OTHER" runs OTHER instead.

        Settings: each directory below contributes COMMAND.json, else the
        alias's file, else default.json; later files override earlier:
          1. $AGM_HOME/sandbox/ (for the AGM home default, see agm help config)
          2. the project config directory's sandbox/
          3. ./.sandbox/
        network and filesystem merge by key; lists append, deduplicated. A later
        deniedDomains entry removes an earlier allowedDomains one, and a later
        denyRead/denyWrite an earlier allowRead/allowWrite. ignoreViolations is
        replaced; enabled and enableWeakerNestedSandbox override when set.
    """),
    "tmux": textwrap.dedent("""\
        agm tmux COMMAND

        Manage tmux sessions and layouts.

        Commands:
          open    Create a tmux session
          close   Kill a tmux session
          layout  Tile a window's panes
    """),
    "check": textwrap.dedent("""\
        agm check [OPTIONS] FILE...

        Statically check each AgL FILE, from parsing through lowering, without
        running anything. A FILE needs no `program def`. Every FILE is checked;
        diagnostics go to stderr as `path:line:col[-[endline:]endcol]: error:
        message` (or `warning:`). For package structure, see agm pkg check.

        Options:
        """)
    + _MODULE_PATH_ROW
    + _NO_STDLIB_ROW
    + textwrap.dedent("""\

        Exit codes:
          0  No errors (warnings do not count).
          1  An error diagnostic, an unreadable FILE, or invalid module roots.
    """),
    "help": textwrap.dedent("""\
        agm help [COMMAND...]

        Show the command overview, or COMMAND's help (as agm COMMAND --help).
    """),
}

# Help for commands that take the shared execution options: the text before and after
# their ``Execution options`` section.
_EXECUTION_HELP_TEXTS: dict[str, tuple[str, str]] = {
    "exec": (
        textwrap.dedent("""\
            agm exec [OPTIONS] (FILE | PACKAGE/MODULE::PROGRAM | -c SOURCE)
                     [ARG]... [--NAME VALUE]...

            Run an AgL program from FILE, an installed PACKAGE/MODULE::PROGRAM, or
            inline SOURCE.

            Options:
              -c, --code SOURCE      AgL source; wrapped in `program def main`
                                     unless it declares a program.
              -p, --program PATH     Select a `program def` by declaration path.
            """)
        + _MODULE_PATH_ROW
        + _NO_STDLIB_ROW,
        textwrap.dedent("""\
            Execution options override config files; a `std/config::KEY := VALUE`
            write overrides both from its program point.

            Program arguments:
              The selected program's value parameters fill ARG slots (positional)
              and take --NAME options (named). bool: --NAME/--no-NAME; Option[T]:
              --NAME VALUE/--no-NAME, Optional[T] also --NAME default; text:
              verbatim; Agent: host Agent syntax; others: JSON or AgL value syntax.
              An omitted argument falls back to its @opt-env variable (empty is
              unset), the program's config table (named only), then its default.
              `@param` bindings in the program's import closure add bare and
              dotted flags (--A.logging.verbose), falling back to @opt-env, the
              program's table, the declaring module's table, then the initializer.
              A bare flag matching several bindings errors when used.

            With a single program or -p, `agm exec FILE -h` shows the program's
            help. With several programs and no -p, this help lists them.

            Exit codes:
              0  Success.
              1  Failure before execution: unreadable source, static error, or
                 invalid config or arguments.
              2  Uncaught AgL exception.
        """),
    ),
    "repl": (
        textwrap.dedent("""\
            agm repl [OPTIONS]

            Evaluate AgL entries in one persistent session: bindings, types, and
            declarations accumulate, and each entry runs exactly once.

            Options:
              --quiet      Do not echo entry results.
              --plain      Use the plain line front end even on a terminal.
              --no-stdlib  Disable the automatic `import std/prelude::*`; kept by
                           :reset.
            """),
        textwrap.dedent("""\
            Execution options override [exec] config; a `std/config::KEY := VALUE`
            entry holds for the session until :reset.

            The interactive console needs a terminal on stdin and stdout and TERM
            other than dumb; otherwise the plain front end runs, without styling.

            :help lists meta-commands; :quit or Ctrl-D exits; Ctrl-C cancels the
            current entry.

            Exit codes:
              0  Normal exit.
              1  Setup failure before the prompt: invalid [exec] config or
                 execution option, or an unwritable --trace-file.
        """),
    ),
}

_HELP_ALIASES: dict[str, str] = {
    "wt": "worktree",
    "wsp": "workspace",
}

# Subcommand paths documented by another path's help.
_PATH_ALIASES: dict[tuple[str, ...], tuple[str, ...]] = {
    ("workspace", "open"): ("open",),
    ("workspace", "close"): ("close",),
    ("config", "cp"): ("config", "copy"),
    ("worktree", "rm"): ("worktree", "remove"),
    ("dep", "remove"): ("dep", "rm"),
}

_PATH_HELP_TEXTS: dict[tuple[str, ...], str] = {
    ("workspace", "setup"): textwrap.dedent("""\
        agm workspace setup [--dry-run]

        Run the current workspace's configured setup scripts.
    """),
    ("workspace", "list"): textwrap.dedent("""\
        agm workspace list [-v]

        List open workspaces.

        Options:
          -v, --verbose  Show workspace directories.
    """),
    ("workspace", "shell-regen"): textwrap.dedent("""\
        agm workspace shell-regen [--dry-run] SHELL_DIR

        Regenerate the session shell wrapper and rc files in SHELL_DIR. Run by
        the wrapper when its cache directory was deleted.
    """),
    ("sync", "fetch"): textwrap.dedent("""\
        agm sync fetch [--dry-run]

        Fetch the main repo and all checked-out dependencies, then create missing
        local tracking branches.
    """),
    ("sync", "pull"): textwrap.dedent("""\
        agm sync pull [--dry-run]

        Run agm sync fetch, then git merge in the main workspace, every branch
        workspace, and every dependency worktree.
    """),
    ("config", "copy"): textwrap.dedent("""\
        agm config copy [--dry-run] DIRNAME

        Copy the project's dot config files into the existing directory DIRNAME.
        Alias: cp.
    """),
    ("config", "env"): textwrap.dedent("""\
        agm config env

        Print shell statements that set the current workspace's environment from
        the project and workspace config.toml [deps] tables, .env, .env.local,
        and env.sh. Apply with: eval "$(agm config env)"
    """),
    ("config", "update"): textwrap.dedent("""\
        agm config update [--dry-run]

        Create missing project and workspace config.toml files in the project
        config directory.
    """),
    ("worktree", "new"): textwrap.dedent("""\
        agm worktree new [-d DIR] [--no-fetch] [--dry-run] BRANCH

        Create a worktree for BRANCH, creating the branch or checking out an
        existing one.

        Options:
          -d, --dir DIR  Create it under DIR instead of the project's worktrees
                         directory.
          --no-fetch     Resolve remote branches from local refs.
    """),
    ("worktree", "remove"): textwrap.dedent("""\
        agm worktree remove [-f] [--dry-run] BRANCH

        Remove BRANCH's worktree and delete its local branch. Alias: rm.

        Options:
          -f, --force  Remove despite uncommitted changes or a lock.
    """),
    ("dep", "list"): textwrap.dedent("""\
        agm dep list [-v] [--all]

        List the current workspace's dependency checkouts as DEP/BRANCH.

        Options:
          -v, --verbose  Show checkout paths.
          --all          List every dependency checkout on disk.
    """),
    ("dep", "new"): textwrap.dedent("""\
        agm dep new [-b BRANCH] [--dry-run] REPO_URL

        Clone a dependency into deps/.

        Options:
          -b, --branch BRANCH  Clone BRANCH instead of the default branch.
    """),
    ("dep", "switch"): textwrap.dedent("""\
        agm dep switch [-b] [--no-fetch] [--dry-run] DEP BRANCH

        Use DEP's checkout named BRANCH (directory name or checked-out branch),
        adding a worktree at deps/DEP/BRANCH for an existing branch if none
        exists, and record it in config.toml [deps].

        Options:
          -b, --branch  Create BRANCH from DEP's default branch first.
          --no-fetch    Resolve remote branches from local refs.
    """),
    ("dep", "rm"): textwrap.dedent("""\
        agm dep rm [--dry-run] DEP/NAME_OR_BRANCH | DEP/repo | DEP/MAIN_CHECKOUT
        agm dep rm --all [--dry-run] DEP

        Remove a dependency worktree (by directory name under deps/DEP/ or
        checked-out branch), the main checkout (DEP/repo or its directory
        name), or with --all the whole deps/DEP/ directory. Alias: remove.

        A target is a relative path without . or .. components, resolving below
        deps/ (a worktree below its dependency). With --all, a detached or
        out-of-dependency worktree stops removal before anything is removed.
    """),
    ("pkg", "init"): textwrap.dedent("""\
        agm pkg init [--name NAME] [--version VERSION] [--dry-run] [DIR]

        Create a package in DIR (default: the current directory, created if
        missing): a package.toml manifest and a starter src/main.agl.

        Options:
          --name NAME        Package name (default: the directory name).
          --version VERSION  Initial version (default: 0.1.0).
    """),
    ("pkg", "check"): textwrap.dedent("""\
        agm pkg check [DIR]

        Validate the package in DIR (default: the current directory): manifest,
        module-tree discipline, registered program references, and [python]
        requirements, which are reported, not installed (see agm pkg sync).
    """),
    ("pkg", "create"): textwrap.dedent("""\
        agm pkg create [-o FILE] [--dry-run] [DIR]

        Validate the package in DIR (default: the current directory) and write
        a deterministic <name>-<version>.agmpkg archive beside it, or to FILE.
    """),
    ("pkg", "install"): textwrap.dedent("""\
        agm pkg install [--editable] [--reinstall] [--shadow] [--dry-run] SRC

        Install a package directory or .agmpkg archive into the store and
        activate it. URL dependencies are fetched and verified against their
        SHA-256. If any [python] requirement of the active packages is
        unsatisfied, all of them are installed together into AGM's interpreter
        (uv, else pip); a failure fails the install.

        Options:
          --editable   Activate the SRC directory in place.
          --reinstall  Replace the installation of the same name and version;
                       not with --editable.
          --shadow     Replace conflicting registered commands instead of
                       failing.
    """),
    ("pkg", "uninstall"): textwrap.dedent("""\
        agm pkg uninstall [--dry-run] NAME[@VERSION]

        Verify and remove a stored version (default: the active one), clearing
        its activation. An editable package is only deactivated.
    """),
    ("pkg", "switch"): textwrap.dedent("""\
        agm pkg switch [--dry-run] NAME@VERSION

        Activate a version already in the store.
    """),
    ("pkg", "list"): textwrap.dedent("""\
        agm pkg list

        List installed versions and active editable packages.
    """),
    ("pkg", "info"): textwrap.dedent("""\
        agm pkg info NAME

        Show an active package's metadata, registered commands, direct
        dependencies, and [python] requirements, each installed VERSION,
        installed VERSION (unsatisfied), missing, or not applicable (marker
        false).
    """),
    ("pkg", "sync"): textwrap.dedent("""\
        agm pkg sync [--dry-run]

        If any [python] requirement of the active packages is unsatisfied,
        install all of them together into AGM's interpreter (uv, else pip) and
        list the unsatisfied ones. Nothing is uninstalled.
    """),
    ("tmux", "open"): textwrap.dedent("""\
        agm tmux open [-d] [-n PANES] [--dry-run] [SESSION]

        Create a tmux session, named SESSION or by tmux.

        Options:
          -d, --detach           Do not attach to the session.
          -n, --num-panes PANES  Create the session with PANES panes.
    """),
    ("tmux", "close"): textwrap.dedent("""\
        agm tmux close [--dry-run] SESSION

        Kill the tmux session SESSION.
    """),
    ("tmux", "layout"): textwrap.dedent("""\
        agm tmux layout [-w WINDOW_ID] [--dry-run] PANES

        Apply AGM's tiled layout for PANES panes to the current tmux window.

        Options:
          -w, --window WINDOW_ID  Target window WINDOW_ID instead.
    """),
}


def _registered_command_overview() -> tuple[tuple[str, str], ...]:
    """Read registered command names and summaries without compiling any program.

    A command's summary is the opening paragraph of its program's ``@doc``,
    cached in the activation index when the package was installed.
    """
    try:
        from agm.cli_dispatch import load_command_index
        from agm.config.context import current_config_context

        context = current_config_context()
        index = load_command_index(home=context.home, proj_dir=context.proj_dir, cwd=context.cwd)
    except (OSError, SystemExit, ValueError):
        return ()
    return tuple(
        (path_name, first_paragraph(registration.doc) if registration.doc else "")
        for path_name, registration in sorted(index.commands.items())
    )


def _overview_text() -> str:
    lines = [
        "agm - Agent Management Framework",
        "",
        "Usage: agm COMMAND [ARGS]...",
        "",
        "Commands:",
    ]
    width = help_width()
    lines.extend(format_description_column(COMMAND_OVERVIEW, width=width))
    registered_commands = _registered_command_overview()
    if registered_commands:
        lines.extend(("", "Registered commands:"))
        lines.extend(format_description_column(registered_commands, width=width))
    lines.extend(
        [
            "",
            "Options:",
            "  --version             Show the AGM and AgL standard-library versions.",
            "  --install-completion  Install shell completion for the current shell.",
            "  --show-completion     Print the shell completion script.",
            "",
            "Command options:",
            "  -h, --help  Show the command's help.",
            "  --dry-run   Print planned actions instead of performing them, where the",
            "              command's usage lists it.",
            "",
            "Run 'agm help COMMAND' for a command's help.",
        ]
    )
    return "\n".join(lines) + "\n"


def help_text_for(command: str) -> str | None:
    canonical = _HELP_ALIASES.get(command, command)
    if canonical in _EXECUTION_HELP_TEXTS:
        before, after = _EXECUTION_HELP_TEXTS[canonical]
        return f"{before}\n{execution_options_section(width=help_width())}\n{after}"
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
    """Resolve command-group and subcommand aliases to the path documenting them."""
    head, *rest = command_path
    path = (_HELP_ALIASES.get(head, head), *rest)
    return _PATH_ALIASES.get(path, path)


def _help_text_for_path(command_path: Sequence[str]) -> str:
    """Return *command_path*'s help, its usage spelled as typed when it is an alias."""
    normalized = _canonical_command_path(command_path)
    if len(normalized) == 1:
        text = help_text_for(normalized[0])
    else:
        text = _PATH_HELP_TEXTS.get(normalized)
    if text is None:
        raise ValueError(f"unknown command path: {' '.join(command_path)}")
    usage, separator, body = text.partition("\n\n")
    typed = f"agm {' '.join(command_path)}"
    if typed not in usage:
        usage = usage.replace(f"agm {' '.join(normalized)}", typed)
    return usage + separator + body


def print_help_for_command_path(
    command_path: Sequence[str],
    file: _Writeable | None = None,
) -> None:
    output = sys.stdout if file is None else file
    print(_help_text_for_path(command_path), end="", file=output)


def exit_with_usage_error(
    command_path: Sequence[str], message: str, *, exit_code: int = 1
) -> NoReturn:
    """Report *message* with the command's usage lines and exit with *exit_code*."""
    usage, _, _ = _help_text_for_path(command_path).partition("\n\n")
    prefix = "usage: "
    print(message, file=sys.stderr)
    print(prefix + usage.replace("\n", "\n" + " " * len(prefix)), file=sys.stderr)
    print(f"Run 'agm help {' '.join(command_path)}' for details.", file=sys.stderr)
    raise SystemExit(exit_code)
