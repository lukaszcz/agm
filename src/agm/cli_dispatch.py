"""Lazy fallback dispatch for commands registered by active packages."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import click
from click.shell_completion import CompletionItem
from typer.core import TyperCommand, TyperGroup, TyperOption

from agm.config.context import current_config_context
from agm.core import dry_run

if TYPE_CHECKING:
    from agm.agl.runtime.types import ParamDeclInfo, ProgramDeclInfo
    from agm.packages.activation import ActivationIndex, CommandRegistration


@dataclass(frozen=True, slots=True)
class RegisteredCommandResolution:
    """An indexed command registration and the arguments left after its path."""

    registration: CommandRegistration
    trailing_args: tuple[str, ...]


def load_command_index(*, home: Path, proj_dir: Path | None, cwd: Path) -> ActivationIndex:
    """Load the command index for the project-selected packages."""

    from agm.packages import activation

    return activation.effective_command_index(home=home, proj_dir=proj_dir, cwd=cwd)


def set_dry_run(ctx: object, param: object, value: bool) -> None:
    """Set invocation-wide dry-run state from any command level."""
    del param
    root = cast(click.Context, ctx)
    while root.parent is not None:
        root = root.parent
    root_meta = cast(dict[str, bool], root.meta)
    enabled = value or bool(root_meta.get("dry_run"))
    root_meta["dry_run"] = enabled
    dry_run.set_enabled(enabled)


def _is_usage_error(error: Exception) -> bool:
    """Recognize Click errors across Typer's bundled-Click compatibility boundary.

    Typer vendors its own copy of Click (``typer._click``) from 0.25 onward, so
    the ``UsageError`` raised by command resolution is a *different class* from
    ``click.UsageError`` whenever AGM is installed against such a Typer.  An
    ``isinstance`` check alone silently stops recognizing unknown commands
    there, which disables the registered-package-command fallback below.
    """

    return isinstance(error, click.UsageError) or error.__class__.__name__ == "UsageError"


def _path_length(item: tuple[str, CommandRegistration]) -> int:
    """Return the number of words in an indexed command path."""

    return len(item[0].split())


def registered_command_help(
    path_name: str,
    registration: CommandRegistration,
    *,
    params: tuple[ParamDeclInfo, ...] | None = None,
    program: "ProgramDeclInfo | None" = None,
) -> str:
    """Render help for one package-registered command.

    Pass *params* when the caller has already discovered the legacy
    ``param`` inventory, and *program* when the caller has already
    discovered the referenced ``program def``'s own declaration, so
    rendering help does not compile either one again. Leaving either at its
    ``None`` default triggers a fresh discovery here, exactly as if none had
    been found — harmless when the caller's own discovery legitimately came
    back empty too; ``registered_program_declaration`` already degrades its
    own failures to ``None``, so no further guard is needed here.

    The usage line names *program*'s own positional slots and options, not
    the raw declaration path, so it reads like the command the reader
    actually invokes rather than the ``program def`` behind it.
    """
    if program is None:
        from agm.commands.exec_program import registered_program_declaration

        program = registered_program_declaration(registration.program, registration.package)
    option_map = None
    if program is not None:
        from agm.cli_support.program_options import program_option_map_or_none

        option_map = program_option_map_or_none(program.parameters)

    try:
        if params is not None:
            from agm.cli_support.exec_params import param_option_flags

            flags = param_option_flags(params)
        else:
            from agm.commands.exec_program import registered_program_param_flags

            flags = registered_program_param_flags(registration.program, registration.package)
    except (Exception, SystemExit):
        flags = ()

    usage = (
        option_map.usage_line(f"agm {path_name}") if option_map is not None else f"agm {path_name}"
    )
    if flags:
        usage += " [--PARAM VALUE]..."
    usage += " [--dry-run]"

    description = registration.description or "Run the registered AgL program."
    lines = [
        usage,
        "",
        description,
        "",
        "Options:",
        "  --dry-run  Statically check the program without executing it.",
    ]
    if flags:
        lines.extend(("", "Program parameters:", *(f"  {flag}" for flag in flags)))
    if option_map is not None:
        described = option_map.option_lines()
        if described:
            lines.extend(("", "Program arguments:", *(f"  {line}" for line in described)))
    return "\n".join(lines) + "\n"


def print_registered_command_help(command_path: Sequence[str]) -> bool:
    """Print registered-command help when *command_path* names one exactly."""
    try:
        context = current_config_context()
        index = load_command_index(home=context.home, proj_dir=context.proj_dir, cwd=context.cwd)
    except (OSError, ValueError, SystemExit):
        return False
    path_name = " ".join(command_path)
    registration = index.commands.get(path_name)
    if registration is None:
        return False
    print(registered_command_help(path_name, registration), end="")
    return True


def resolve_registered_command(
    args: Sequence[str], commands: Mapping[str, CommandRegistration]
) -> RegisteredCommandResolution | None:
    """Resolve the longest registered command path at the start of *args*."""

    for path_name, registration in sorted(commands.items(), key=_path_length, reverse=True):
        words = tuple(path_name.split())
        if tuple(args[: len(words)]) == words:
            return RegisteredCommandResolution(registration, tuple(args[len(words) :]))
    return None


class RegisteredProgramCommand(TyperCommand):
    """Click command that gives all remaining arguments to one AgL program."""

    def __init__(self, path_name: str, registration: CommandRegistration) -> None:
        context_settings: dict[str, bool | list[str]] = {
            "allow_extra_args": True,
            "ignore_unknown_options": True,
            "help_option_names": [],
        }
        super().__init__(name="registered-program", context_settings=context_settings)
        self.params.append(
            TyperOption(
                param_decls=["--dry-run"],
                default=False,
                is_flag=True,
                is_eager=True,
                expose_value=False,
                callback=set_dry_run,
                help="Statically check the program without executing it.",
            )
        )
        self._path_name = path_name
        self._registration = registration

    def invoke(self, ctx: click.Context) -> None:
        params: tuple[ParamDeclInfo, ...] | None = None
        program: "ProgramDeclInfo | None" = None
        help_requested = "--help" in ctx.args
        if not help_requested and "-h" in ctx.args:
            # This is the first point at which an unknown command has been proven
            # to be registered, so AgL remains unloaded for all builtin commands.
            from agm.cli_support.exec_params import param_value_taking_flags
            from agm.cli_support.program_options import (
                program_option_map_or_none,
                short_help_requested,
            )
            from agm.commands.exec_program import (
                registered_program_declaration,
                registered_program_params,
            )

            params = registered_program_params(
                self._registration.program, self._registration.package
            )
            program = registered_program_declaration(
                self._registration.program, self._registration.package
            )
            option_map = None if program is None else program_option_map_or_none(program.parameters)
            value_flags = param_value_taking_flags(params) | (
                frozenset[str]() if option_map is None else option_map.value_taking_flags()
            )
            help_requested = short_help_requested(ctx.args, value_flags=value_flags)
        if help_requested:
            print(
                registered_command_help(
                    self._path_name, self._registration, params=params, program=program
                ),
                end="",
            )
            return
        from agm.commands.exec_program import RegisteredParamUsageError, run_registered

        try:
            run_registered(
                self._registration.program,
                list(ctx.args),
                package=self._registration.package,
                command_path=self._path_name,
            )
        except RegisteredParamUsageError as exc:
            print(f"error: {exc.message}", file=sys.stderr)
            print(file=sys.stderr)
            print(
                registered_command_help(
                    self._path_name, self._registration, params=exc.params, program=exc.program
                ),
                end="",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc


class RegisteredCommandGroup(TyperGroup):
    """Root group that falls back to package registrations after builtins miss."""

    def shell_complete(self, ctx: click.Context, incomplete: str) -> list[CompletionItem]:
        """Extend root completion with the next registered-command path segment."""
        from agm.completion import registered_command_completion

        command_path = [*ctx._protected_args, *ctx.args]
        registered_segments, is_registered = registered_command_completion(command_path, incomplete)
        if command_path and is_registered:
            if incomplete.startswith("--"):
                from agm.completion import registered_command_param_completion

                return registered_command_param_completion(command_path, incomplete)
            return [CompletionItem(segment) for segment in registered_segments]
        if command_path and registered_segments:
            return [CompletionItem(segment) for segment in registered_segments]

        items_by_value: dict[str, CompletionItem] = {
            cast(str, item.value): item for item in super().shell_complete(ctx, incomplete)
        }
        for segment in registered_segments:
            items_by_value[segment] = CompletionItem(segment)
        return list(items_by_value.values())

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, TyperCommand | None, list[str]]:
        try:
            name, command, trailing_args = super().resolve_command(ctx, args)
            return name, command, trailing_args
        except Exception as exc:
            if not _is_usage_error(exc):
                raise
            context = current_config_context()
            try:
                index = load_command_index(
                    home=context.home, proj_dir=context.proj_dir, cwd=context.cwd
                )
            except ValueError as index_error:
                ctx.fail(f"cannot load package command index: {index_error}")
            resolution = resolve_registered_command(args, index.commands)
            if resolution is None:
                raise
            path_name = " ".join(args[: len(args) - len(resolution.trailing_args)])
            command = RegisteredProgramCommand(path_name, resolution.registration)
            return (
                path_name,
                command,
                list(resolution.trailing_args),
            )
