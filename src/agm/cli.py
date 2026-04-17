"""AGM command-line interface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn, Protocol, cast

import agm.commands.branch.sync as branch_sync_command
import agm.commands.close as close_command
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
import agm.commands.worktree.new as worktree_new_command
import agm.commands.worktree.remove as worktree_remove_command
import agm.commands.worktree.setup as worktree_setup_command
from agm import parser as parser_helpers
from agm.commands.args import (
    CloseArgs,
    ConfigCopyArgs,
    DepNewArgs,
    DepRemoveArgs,
    DepSwitchArgs,
    InitArgs,
    OpenArgs,
    RunArgs,
    TmuxLayoutArgs,
    TmuxNewArgs,
    WorktreeNewArgs,
    WorktreeRemoveArgs,
    WorktreeSetupArgs,
)

build_parser = parser_helpers.build_parser
print_command_help = parser_helpers.print_command_help
print_help_for_command_path = parser_helpers.print_help_for_command_path
print_overview = parser_helpers.print_overview
_HELP_TEXTS = parser_helpers._HELP_TEXTS
_HELP_ALIASES = parser_helpers._HELP_ALIASES
_COMMAND_OVERVIEW = parser_helpers._COMMAND_OVERVIEW


class _DispatchArgs(Protocol):
    command: str | None
    help_command: list[str]
    br_command: str | None
    config_command: str | None
    wt_command: str | None
    dep_command: str | None
    tmux_command: str | None


def _has_run_command(args: RunArgs) -> bool:
    return bool(run_command.normalize_run_command(list(args.run_command)))


def dispatch(args: _DispatchArgs) -> NoReturn:
    cmd = args.command
    if cmd is None:
        print_overview()
        raise SystemExit(0)
    if cmd == "help":
        if not args.help_command:
            print_overview()
        else:
            try:
                print_help_for_command_path(args.help_command)
            except ValueError:
                print_command_help(" ".join(args.help_command))
        raise SystemExit(0)
    if cmd in {"br", "branch"}:
        if args.br_command is None:
            print_command_help(cmd)
            raise SystemExit(0)
        branch_sync_command.run(args)
        raise SystemExit(0)
    if cmd == "open":
        open_command.run(cast(OpenArgs, args))
        raise SystemExit(0)
    if cmd == "close":
        close_command.run(cast(CloseArgs, args))
        raise SystemExit(0)
    if cmd == "config":
        if args.config_command is None:
            print_command_help(cmd)
            raise SystemExit(0)
        config_copy_command.run(cast(ConfigCopyArgs, args))
        raise SystemExit(0)
    if cmd in {"wt", "worktree"}:
        if args.wt_command is None:
            print_command_help(cmd)
            raise SystemExit(0)
        if args.wt_command == "new":
            worktree_new_command.run(cast(WorktreeNewArgs, args))
        elif args.wt_command == "setup":
            worktree_setup_command.run(cast(WorktreeSetupArgs, args))
        else:
            worktree_remove_command.run(cast(WorktreeRemoveArgs, args))
        raise SystemExit(0)
    if cmd == "dep":
        if args.dep_command is None:
            print_command_help(cmd)
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
        run_args = cast(RunArgs, args)
        if not _has_run_command(run_args):
            print_command_help(cmd)
            raise SystemExit(0)
        run_command.run(run_args)
        raise AssertionError("unreachable")
    if cmd == "tmux":
        if args.tmux_command is None:
            print_command_help(cmd)
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
