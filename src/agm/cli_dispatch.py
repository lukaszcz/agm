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
    from agm.agl.runtime.types import ProgramDeclInfo
    from agm.cli_support.program_discovery import ProgramDiscoveryArtifacts
    from agm.cli_support.program_options import ProgramCommand
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


DRY_RUN_HELP = "Statically check the program without executing it."


def _dry_run_option() -> TyperOption:
    """Return the eager ``--dry-run`` flag a dispatched command parses."""
    return TyperOption(
        param_decls=["--dry-run"],
        default=False,
        is_flag=True,
        is_eager=True,
        expose_value=False,
        callback=set_dry_run,
        help=DRY_RUN_HELP,
    )


def _dry_run_help_option() -> click.Option:
    """Return the ``--dry-run`` entry a registered command's help lists.

    The parsed flag is eager and carries a callback that help rendering must
    not run, so the listed entry is a plain option — derived from the parsed
    declaration, spellings and help alike, so the two cannot drift apart.
    """
    parsed = _dry_run_option()
    return click.Option([*parsed.opts, *parsed.secondary_opts], is_flag=True, help=parsed.help)


def registered_command_help(
    path_name: str,
    registration: CommandRegistration,
    *,
    program: "ProgramDeclInfo | None",
    command: "ProgramCommand | None",
) -> str:
    """Render help for one package-registered command.

    *program* is the caller's own discovery result for the referenced
    ``program def`` — a declaration, or ``None`` when discovery was attempted
    and found nothing (an unresolvable reference, a package mismatch, or a
    source that no longer compiles) — and *command* is that program's own
    built command, or ``None`` for the same reasons plus a colliding flag
    projection. Both are required rather than defaulted so that a ``None``
    here can never be mistaken for "not looked up yet" and trigger a second
    full compile, or a second build, of what the caller already holds.

    The help is the referenced program's own command help, spelled for the
    command the reader invokes rather than the ``program def`` behind it: its
    usage line names ``agm <path>`` and the program's positional slots, its
    options are the program's own plus ``--dry-run``, and its description is
    the manifest's, which a package author writes for this command, in
    preference to the program's own ``@doc``.
    """
    from agm.cli_support.program_options import render_program_help

    description = registration.description or (None if program is None else program.doc)
    return render_program_help(
        command,
        program_name=f"agm {path_name}",
        description=description or "Run the registered AgL program.",
        extra_options=(_dry_run_help_option(),),
    )


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
    from agm.cli_support.program_options import program_command_for
    from agm.commands.exec_program import registered_program_declaration

    program = registered_program_declaration(registration.program, registration.package)
    print(
        registered_command_help(
            path_name, registration, program=program, command=program_command_for(program)
        ),
        end="",
    )
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
        self.params.append(_dry_run_option())
        self._path_name = path_name
        self._registration = registration

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Parse *args*, keeping the reader's end-of-options marker in the tail.

        Click removes the marker as it parses this command's own options, so
        it is doubled first (``program_options.retain_end_of_options``): Click
        removes the copy it would have removed anyway and the survivor stays
        in ``ctx.args`` for the program's own parser. A registered command
        names no FILE of its own, so every marker it is given belongs to the
        program — one ``--`` is the program's end-of-options marker, exactly
        as it is for ``agm exec``.
        """
        from agm.cli_support.program_options import (
            option_value_map,
            program_command_for,
            protect_host_option_values,
            protect_potential_program_values,
            retain_end_of_options,
        )

        host_options = option_value_map(
            tuple(param for param in self.params if isinstance(param, TyperOption))
        )
        preview = protect_potential_program_values(args, host_options)
        program, pipeline_cache = (
            self._discover_program_with_artifacts() if preview != args else (None, None)
        )
        program_command = program_command_for(program)
        protected, replacements = protect_host_option_values(args, program_command, host_options)
        remaining = super().parse_args(ctx, retain_end_of_options(protected, host_options))
        ctx.args[:] = [replacements.get(token, token) for token in ctx.args]
        if program is not None:
            cast(dict[str, object], ctx.meta)["registered_program"] = program
        if pipeline_cache is not None:
            cast(dict[str, object], ctx.meta)["registered_pipeline_cache"] = pipeline_cache
        return [replacements.get(token, token) for token in remaining]

    def _discover_program(self) -> "ProgramDeclInfo | None":
        """Discover the referenced ``program def``'s declaration, or ``None``."""
        from agm.commands.exec_program import registered_program_declaration

        return registered_program_declaration(
            self._registration.program, self._registration.package
        )

    def _discover_program_with_artifacts(
        self,
    ) -> "tuple[ProgramDeclInfo | None, ProgramDiscoveryArtifacts | None]":
        """Discover the declaration and retain its reusable static artifacts."""
        from agm.commands.exec_program import registered_program_declaration

        artifacts: list[ProgramDiscoveryArtifacts] = []
        program = registered_program_declaration(
            self._registration.program,
            self._registration.package,
            artifact_sink=artifacts,
        )
        return program, artifacts[0] if artifacts else None

    def invoke(self, ctx: click.Context) -> None:
        from agm.cli_support.program_options import (
            ProgramHelpRequested,
            contains_help_flag,
            program_command_for,
            program_help_requested,
        )

        metadata = cast(dict[str, object], ctx.meta)
        if contains_help_flag(ctx.args):
            # This is the first point at which an unknown command has been proven
            # to be registered, so AgL remains unloaded for all builtin commands.
            cached_program = metadata.pop("registered_program", None)
            program = (
                cast("ProgramDeclInfo", cached_program)
                if cached_program is not None
                else self._discover_program()
            )
            command = program_command_for(program)
            if program_help_requested(ctx.args, command):
                print(
                    self._help(program=program, command=command),
                    end="",
                )
                return
        from agm.commands.exec_program import RegisteredProgramUsageError, run_registered

        try:
            cached_pipeline = metadata.pop("registered_pipeline_cache", None)
            if cached_pipeline is None:
                run_registered(
                    self._registration.program,
                    list(ctx.args),
                    package=self._registration.package,
                    command_path=self._path_name,
                )
            else:
                run_registered(
                    self._registration.program,
                    list(ctx.args),
                    package=self._registration.package,
                    command_path=self._path_name,
                    pipeline_cache=cast("ProgramDiscoveryArtifacts", cached_pipeline),
                )
        except ProgramHelpRequested as exc:
            # A help request Click recognized only while parsing the program's
            # own command — a flag bundled into a short group of its own. The
            # request carries the very command it was parsing, so the help is
            # rendered from that rather than from a second build.
            print(self._help(program=exc.program, command=exc.command), end="")
        except RegisteredProgramUsageError as exc:
            print(f"error: {exc.message}", file=sys.stderr)
            print(file=sys.stderr)
            print(
                self._help(program=exc.program, command=program_command_for(exc.program)),
                end="",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc

    def _help(self, *, program: "ProgramDeclInfo | None", command: "ProgramCommand | None") -> str:
        """Render this registered command's help for an already-discovered program."""
        return registered_command_help(
            self._path_name, self._registration, program=program, command=command
        )


class RegisteredCommandGroup(TyperGroup):
    """Root group that falls back to package registrations after builtins miss."""

    def shell_complete(self, ctx: click.Context, incomplete: str) -> list[CompletionItem]:
        """Extend root completion with the next registered-command path segment."""
        from agm.completion import registered_command_completion

        # Click splits a group's tail into ``_protected_args`` (the first word,
        # the one a subcommand name would come from) and ``args`` (the rest),
        # and exposes no undeprecated accessor for the pair: ``protected_args``
        # warns and is slated for removal, while ``args`` alone drops the very
        # word that starts a registered command path. Click's own
        # ``shell_completion`` reads the same attribute for the same reason.
        command_path = [*ctx._protected_args, *ctx.args]
        registered_segments, is_registered = registered_command_completion(command_path, incomplete)
        if command_path and is_registered:
            if incomplete.startswith("-"):
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
