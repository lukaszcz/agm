"""CLI completion helpers for Typer parameters."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Sequence
from copy import copy
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, ParamSpec, Protocol, TypeVar, cast

import click
from click.shell_completion import CompletionItem
from typer.core import TyperCommand, TyperOption

import agm.vcs.git as git_helpers
from agm.config.context import current_config_context
from agm.config.general import load_merged_config, load_run_config
from agm.project.dependency_checkout import main_dep_repo

if TYPE_CHECKING:
    from agm.agl.runtime.types import ProgramDeclInfo

from agm.project.layout import (
    current_workspace_or_project_root,
    default_worktrees_dir,
    discover_current_project_dir,
    project_deps_dir,
    project_repo_dir,
)

_P = ParamSpec("_P")
_CandidateT = TypeVar("_CandidateT")


class _ContextWithMetadata(Protocol):
    """The Click context fields AGM uses beyond Typer's narrow stub."""

    meta: dict[str, object]


def _completes_quietly(
    complete: Callable[_P, list[_CandidateT]],
) -> Callable[_P, list[_CandidateT]]:
    """Return *complete* with every failure answered as no candidates.

    A completer runs inside the user's shell while they are typing, so a
    project that cannot be read, a command that exits, or any other failure
    has to leave the prompt alone rather than print over it.
    """

    @wraps(complete)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> list[_CandidateT]:
        try:
            return complete(*args, **kwargs)
        except (Exception, SystemExit):
            return []

    return guarded


_COMMON_PANE_COUNTS = ["1", "2", "3", "4", "6", "8", "12", "16"]

_HELP_TREE: dict[tuple[str, ...], list[str]] = {
    (): [
        "open",
        "close",
        "init",
        "workspace",
        "wsp",
        "sync",
        "config",
        "wt",
        "worktree",
        "dep",
        "pkg",
        "review",
        "revise",
        "refine",
        "exec",
        "repl",
        "check",
        "run",
        "loop",
        "tmux",
        "help",
    ],
    ("loop",): ["select", "run", "step"],
    ("config",): ["cp", "copy", "env", "update"],
    ("workspace",): ["open", "close", "setup", "list"],
    ("wsp",): ["open", "close", "setup", "list"],
    ("sync",): ["fetch", "pull"],
    ("wt",): ["new", "rm", "remove"],
    ("worktree",): ["new", "rm", "remove"],
    ("dep",): ["list", "new", "switch", "rm", "remove"],
    ("pkg",): ["init", "check", "create", "install", "uninstall", "list", "info"],
    ("tmux",): ["open", "close", "layout"],
}


def _match(candidates: set[str] | list[str], incomplete: str) -> list[str]:
    return sorted(candidate for candidate in candidates if candidate.startswith(incomplete))


def _resolve_project_repo_dir() -> Path | None:
    try:
        project_dir = discover_current_project_dir()
    except SystemExit:
        return None
    if project_dir is None:
        return None
    return project_repo_dir(project_dir)


def _resolve_project_deps_dir() -> Path | None:
    try:
        project_dir = discover_current_project_dir()
    except SystemExit:
        return None
    if project_dir is None:
        return None
    return project_deps_dir(project_dir)


def _branch_candidates(repo_dir: Path) -> set[str]:
    candidates: set[str] = set()

    try:
        candidates.add(git_helpers.current_branch(repo_dir))
    except SystemExit:
        pass

    try:
        for worktree in git_helpers.worktree_list(repo_dir):
            if worktree.branch is not None:
                candidates.add(worktree.branch)
    except SystemExit:
        pass

    for ref in ("refs/heads", "refs/remotes/origin"):
        returncode, stdout, _ = git_helpers.fetch_output(
            ["git", "-C", str(repo_dir), "for-each-ref", "--format=%(refname:short)", ref]
        )
        if returncode != 0:
            continue
        for line in stdout.splitlines():
            if ref == "refs/remotes/origin":
                if line == "origin/HEAD":
                    continue
                if line.startswith("origin/"):
                    candidates.add(line.removeprefix("origin/"))
            elif line:
                candidates.add(line)
    return candidates


def _worktree_branch_candidates(repo_dir: Path) -> set[str]:
    try:
        repo_branch = git_helpers.current_branch(repo_dir)
    except SystemExit:
        return set()

    project_dir = current_workspace_or_project_root(repo_dir)
    worktrees_dir = default_worktrees_dir(project_dir)
    branches: set[str] = set()
    try:
        for worktree in git_helpers.worktree_list(repo_dir):
            branch = worktree.branch
            if branch is None:
                try:
                    relative_path = worktree.path.relative_to(worktrees_dir)
                except ValueError:
                    continue
                branch = relative_path.as_posix()
            if branch and branch != repo_branch:
                branches.add(branch)
    except SystemExit:
        return set()
    return branches


def _resolve_dep_repo(dep_name: str) -> Path | None:
    deps_dir = _resolve_project_deps_dir()
    if deps_dir is None:
        return None
    dep_dir = deps_dir / dep_name
    if not dep_dir.is_dir():
        return None
    try:
        return main_dep_repo(dep_dir)
    except SystemExit:
        return None


def _path_candidates(incomplete: str) -> list[str]:
    current = Path.cwd()
    base_dir = current
    prefix = incomplete
    if incomplete:
        incomplete_path = Path(incomplete)
        if incomplete_path.parent != Path("."):
            base_dir = (current / incomplete_path.parent).resolve(strict=False)
            prefix = incomplete_path.name

    if not base_dir.is_dir():
        return []

    candidates: set[str] = set()
    for path in base_dir.iterdir():
        if not path.name.startswith(prefix):
            continue
        try:
            relative = path.relative_to(current)
            display = str(relative)
        except ValueError:
            display = str(path)
        if path.is_dir():
            display = f"{display}/"
        candidates.add(display)
    return sorted(candidates)


@_completes_quietly
def complete_help_path(ctx: click.Context, incomplete: str) -> list[str]:
    params = cast(dict[str, object], ctx.params)
    help_command = tuple(_string_list(params.get("help_command")))
    static = _match(_HELP_TREE.get(help_command, []), incomplete)
    registered = complete_registered_commands(help_command, incomplete)
    return sorted(set(static) | set(registered))


def registered_command_completion(
    command_path: Sequence[str], incomplete: str
) -> tuple[list[str], bool]:
    """Return next path segments and whether the path resolves to a registered command."""
    try:
        from agm.cli_dispatch import load_command_index, resolve_registered_command

        context = current_config_context()
        index = load_command_index(home=context.home, proj_dir=context.proj_dir, cwd=context.cwd)
        prefix = tuple(command_path)
        candidates = {
            words[len(prefix)]
            for path_name in index.commands
            if (words := tuple(path_name.split()))[: len(prefix)] == prefix
            and len(words) > len(prefix)
        }
        return _match(candidates, incomplete), resolve_registered_command(
            command_path, index.commands
        ) is not None
    except (Exception, SystemExit):
        return [], False


@_completes_quietly
def registered_command_param_completion(
    command_path: Sequence[str], incomplete: str
) -> list[CompletionItem]:
    """Complete parameters for an already resolved registered command.

    Combines ``--dry-run`` with the referenced program's own value-parameter
    option flags (and their ``--no-`` forms), the same mechanism
    ``registered_command_help`` renders.
    """
    from agm.cli_dispatch import load_command_index, resolve_registered_command
    from agm.cli_support.program_options import program_command_for
    from agm.commands.exec_program import registered_program_declaration

    context = current_config_context()
    index = load_command_index(home=context.home, proj_dir=context.proj_dir, cwd=context.cwd)
    resolution = resolve_registered_command(command_path, index.commands)
    if resolution is None:
        return []
    if resolution.registration.program is None:
        return [CompletionItem("--help")] if "--help".startswith(incomplete) else []
    declaration = registered_program_declaration(
        resolution.registration.program, resolution.registration.package, context=context
    )
    program_command = program_command_for(declaration)
    flags = (
        "--dry-run",
        *(() if program_command is None else program_command.option_spellings()),
    )
    return [CompletionItem(flag) for flag in flags if flag.startswith(incomplete)]


def complete_registered_commands(command_path: Sequence[str], incomplete: str) -> list[str]:
    """Complete the next active registered-command path segment."""
    return registered_command_completion(command_path, incomplete)[0]


@_completes_quietly
def complete_open_target(incomplete: str) -> list[str]:
    repo_dir = _resolve_project_repo_dir()
    if repo_dir is None:
        return []
    candidates = _branch_candidates(repo_dir)
    candidates.add("repo")
    return _match(candidates, incomplete)


@_completes_quietly
def complete_close_branch(incomplete: str) -> list[str]:
    repo_dir = _resolve_project_repo_dir()
    if repo_dir is None:
        return []
    return _match(_worktree_branch_candidates(repo_dir), incomplete)


@_completes_quietly
def complete_worktree_branch(incomplete: str) -> list[str]:
    repo_dir = _resolve_project_repo_dir()
    if repo_dir is None:
        return []
    return _match(_branch_candidates(repo_dir), incomplete)


@_completes_quietly
def complete_dep_name(incomplete: str) -> list[str]:
    deps_dir = _resolve_project_deps_dir()
    if deps_dir is None or not deps_dir.is_dir():
        return []
    names = {path.name for path in deps_dir.iterdir() if path.is_dir()}
    return _match(names, incomplete)


@_completes_quietly
def complete_dep_branch(ctx: click.Context, incomplete: str) -> list[str]:
    params = cast(dict[str, object], ctx.params)
    raw_dep = params.get("dep")
    dep_name = raw_dep if isinstance(raw_dep, str) else ""
    if not dep_name:
        return []
    repo_dir = _resolve_dep_repo(dep_name)
    if repo_dir is None:
        return []
    return _match(_branch_candidates(repo_dir), incomplete)


@_completes_quietly
def complete_dep_target(incomplete: str) -> list[str]:
    deps_dir = _resolve_project_deps_dir()
    if deps_dir is None or not deps_dir.is_dir():
        return []

    candidates: set[str] = set()
    for dep_dir in (path for path in deps_dir.iterdir() if path.is_dir()):
        dep_name = dep_dir.name
        candidates.add(dep_name)
        repo_dir = _resolve_dep_repo(dep_name)
        if repo_dir is None:
            continue
        candidates.add(f"{dep_name}/repo")
        for branch in _worktree_branch_candidates(repo_dir):
            candidates.add(f"{dep_name}/{branch}")
    return _match(candidates, incomplete)


@_completes_quietly
def complete_run_command(ctx: click.Context, incomplete: str) -> list[str]:
    params = cast(dict[str, object], ctx.params)
    raw_cmd = params.get("run_command_args")
    cmd_args = cast(list[str], raw_cmd) if isinstance(raw_cmd, list) else []
    if cmd_args:
        return _path_candidates(incomplete)

    candidates: set[str] = set()
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        path = Path(directory)
        if not path.is_dir():
            continue
        for candidate in path.iterdir():
            if (
                candidate.is_file()
                and os.access(candidate, os.X_OK)
                and candidate.name.startswith(incomplete)
            ):
                candidates.add(candidate.name)

    try:
        context = current_config_context()
        run_config = load_run_config(
            home=context.home,
            proj_dir=context.proj_dir,
            cwd=context.cwd,
        )
    except (OSError, SystemExit):
        run_config = None
    if run_config is not None:
        candidates.update(
            command_name
            for command_name in run_config.aliases
            if command_name.startswith(incomplete)
        )
    return sorted(candidates)


def _tmux_candidates(list_command: str, name_format: str, incomplete: str) -> list[str]:
    """Return the names one ``tmux list-*`` query reports, or none when it fails."""
    result = subprocess.run(
        ["tmux", list_command, "-F", name_format],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return _match(set(result.stdout.splitlines()), incomplete)


@_completes_quietly
def complete_tmux_session(incomplete: str) -> list[str]:
    return _tmux_candidates("list-sessions", "#{session_name}", incomplete)


@_completes_quietly
def complete_tmux_window(incomplete: str) -> list[str]:
    return _tmux_candidates("list-windows", "#{window_id}", incomplete)


@_completes_quietly
def complete_pane_count(incomplete: str) -> list[str]:
    return _match(_COMMON_PANE_COUNTS, incomplete)


@_completes_quietly
def complete_path_argument(ctx: click.Context, args: list[str], incomplete: str) -> list[str]:
    del ctx, args
    return _path_candidates(incomplete)


@_completes_quietly
def complete_dir_argument(ctx: click.Context, args: list[str], incomplete: str) -> list[str]:
    """Complete directory paths (directories only, for options like ``-I/--module-path``)."""
    del ctx, args
    return [c for c in _path_candidates(incomplete) if c.endswith("/")]


@_completes_quietly
def complete_package_source(ctx: click.Context, args: list[str], incomplete: str) -> list[str]:
    """Complete package directories and portable package archives."""
    del ctx, args
    return [
        candidate
        for candidate in _path_candidates(incomplete)
        if candidate.endswith("/") or candidate.endswith(".agmpkg")
    ]


def _configured_command_names(
    section: str, *, home: Path, proj_dir: Path | None, cwd: Path
) -> set[str]:
    merged = load_merged_config(home=home, proj_dir=proj_dir, cwd=cwd)
    table = merged.get(section)
    if not isinstance(table, dict):
        return set()
    return {key for key, value in table.items() if isinstance(key, str) and isinstance(value, dict)}


@_completes_quietly
def complete_agl_file(ctx: click.Context, args: list[str], incomplete: str) -> list[str]:
    """Complete ``.agl`` file paths for the ``agm exec FILE`` argument."""
    del ctx, args
    candidates = _path_candidates(incomplete)
    return [c for c in candidates if c.endswith(".agl") or c.endswith("/")]


def _program_argument_completion_items(
    program: "ProgramDeclInfo", incomplete: str
) -> list[CompletionItem]:
    """Return ``CompletionItem`` objects for *program*'s own value-parameter flags.

    *program* is the declaration the caller's shared discovery already
    selected, exactly as ``agm exec --help`` selects it
    (:meth:`~agm.cli_support.program_discovery.ExecProgramDiscovery.selection`),
    so completion never disagrees with help about which program's flags apply.
    Degrades silently to ``[]``.
    """
    from agm.cli_support.program_options import program_command_for

    program_command = program_command_for(program)
    if program_command is None:
        return []
    return [
        CompletionItem(flag)
        for flag in program_command.option_spellings()
        if flag.startswith(incomplete)
    ]


def _string_list(value: object) -> list[str]:
    """Return *value* as a list of strings, or empty when it is not one.

    Completion reads parameters out of a resiliently parsed context, where a
    value may be missing or of any shape, so every list-valued parameter is
    narrowed the same way.
    """
    if isinstance(value, (list, tuple)):
        values = cast(list[object] | tuple[object, ...], value)
        if all(isinstance(item, str) for item in values):
            return [cast(str, item) for item in values]
    return []


class ExecCommand(TyperCommand):
    """Typer Command subclass for ``agm exec`` that augments shell completion.

    When the incomplete token starts with ``--``, the standard completion (built-in
    exec options) is extended with ``--<param>`` / ``--no-<param>`` items discovered
    from the FILE or ``-c``/``--command`` source already parsed into ``ctx.params``.
    Degrades to base completion on any error (unreadable file, parse failure, etc.).
    """

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Parse *args*, preserving program values and the host marker in the tail.

        Click removes the marker as it parses, so it is doubled first: see
        ``program_options.retain_end_of_options``, whose counterpart
        ``split_exec_tail`` consumes the survivor only when it is what names
        the FILE, and otherwise forwards it to the program.
        """
        from agm.cli_support.program_discovery import ExecProgramDiscovery
        from agm.cli_support.program_options import (
            option_value_map,
            program_command_for,
            protect_host_option_values,
            protect_potential_program_values,
            retain_end_of_options,
            split_exec_tail,
        )

        preview = copy(self)
        preview.params = [param for param in self.params if param.name != "_dry_run"]
        preview_ctx = click.Context(
            cast(click.Command, preview),
            info_name=ctx.info_name,
            parent=ctx.parent,
            allow_extra_args=True,
            ignore_unknown_options=True,
            help_option_names=[],
            resilient_parsing=True,
        )
        host_options = option_value_map(
            tuple(param for param in self.params if isinstance(param, TyperOption))
        )
        preview_args = protect_potential_program_values(args, host_options)
        TyperCommand.parse_args(
            preview,
            preview_ctx,
            retain_end_of_options(preview_args, host_options),
        )
        preview_params = cast(dict[str, object], preview_ctx.params)
        raw_command = preview_params.get("command")
        raw_program = preview_params.get("program")
        discovery = ExecProgramDiscovery(
            command=raw_command if isinstance(raw_command, str) else None,
            requested_program=raw_program if isinstance(raw_program, str) else None,
            module_paths=_string_list(preview_params.get("module_paths")),
            no_stdlib=bool(preview_params.get("no_stdlib")),
        )
        selected = split_exec_tail(
            _string_list(preview_params.get("tail")),
            program_command_for_file=(
                None if isinstance(raw_command, str) else discovery.command_for_file
            ),
        )
        program_command = (
            program_command_for(discovery.selection(selected.file).selected)
            if preview_args != args
            else None
        )
        protected, replacements = protect_host_option_values(args, program_command, host_options)
        remaining = super().parse_args(ctx, retain_end_of_options(protected, host_options))
        ctx.args[:] = [replacements.get(token, token) for token in ctx.args]
        parsed_params = cast(dict[str, object], ctx.params)
        tail = _string_list(parsed_params.get("tail"))
        if tail:
            parsed_params["tail"] = tuple(replacements.get(token, token) for token in tail)
        cast(_ContextWithMetadata, ctx).meta["exec_program_discovery"] = discovery
        return [replacements.get(token, token) for token in remaining]

    def shell_complete(self, ctx: click.Context, incomplete: str) -> list[CompletionItem]:
        base = super().shell_complete(ctx, incomplete)
        if not incomplete.startswith("-"):
            return base
        from agm.cli_support.program_discovery import ExecProgramDiscovery
        from agm.cli_support.program_options import split_exec_tail

        params = cast(dict[str, object], ctx.params)
        raw_command = params.get("command")
        raw_program = params.get("program")
        requested_program = raw_program if isinstance(raw_program, str) else None
        # One discovery for this completion: the FILE derivation below may
        # probe several candidate tokens with it, and the selection it settles
        # on is then read back without a second static pipeline pass.
        discovery = ExecProgramDiscovery(
            command=raw_command if isinstance(raw_command, str) else None,
            requested_program=requested_program,
            module_paths=_string_list(params.get("module_paths")),
            no_stdlib=bool(params.get("no_stdlib")),
        )
        # The FILE selector comes from the same derivation execution uses, so
        # completion offers a program's own flags for exactly the invocations
        # that would run it.
        file = split_exec_tail(
            _string_list(params.get("tail")),
            program_command_for_file=(
                None if isinstance(raw_command, str) else discovery.command_for_file
            ),
        ).file
        try:
            selection = discovery.selection(file)
            if selection.selected is None:
                return base
            extra = _program_argument_completion_items(selection.selected, incomplete)
            items_by_value: dict[str, CompletionItem] = {}
            for item in (*base, *extra):
                items_by_value[cast(str, item.value)] = item
            return list(items_by_value.values())
        except (Exception, SystemExit):
            # Completion is advisory: any failure past the shared discovery
            # (option-map projection, item building) degrades to the built-in
            # exec options rather than breaking the user's shell.
            return base


@_completes_quietly
def complete_revise_command_or_review_file(
    ctx: click.Context, args: list[str], incomplete: str
) -> list[str]:
    del ctx, args
    try:
        context = current_config_context()
        command_matches = _match(
            _configured_command_names(
                "revise",
                home=context.home,
                proj_dir=context.proj_dir,
                cwd=context.cwd,
            ),
            incomplete,
        )
    except (OSError, SystemExit):
        command_matches = []
    if command_matches:
        return command_matches
    return _path_candidates(incomplete)
