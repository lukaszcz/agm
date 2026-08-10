"""Lazy fallback dispatch for commands registered by active packages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import click
from click.shell_completion import CompletionItem
from typer.core import TyperCommand, TyperGroup

from agm.config.context import current_config_context


class CommandRegistrationLike(Protocol):
    """The command-index fields needed by the CLI fallback."""

    @property
    def package(self) -> str: ...

    @property
    def program(self) -> str: ...

    @property
    def description(self) -> str | None: ...


class CommandIndexLike(Protocol):
    """The active package index view needed by the CLI fallback."""

    @property
    def commands(self) -> Mapping[str, CommandRegistrationLike]: ...


@dataclass(frozen=True, slots=True)
class RegisteredCommandResolution:
    """An indexed command registration and the arguments left after its path."""

    registration: CommandRegistrationLike
    trailing_args: tuple[str, ...]


def load_activation_index(
    *, home: Path, proj_dir: Path | None = None, cwd: Path | None = None
) -> CommandIndexLike:
    """Load the command index for the global or project-selected packages."""

    from agm.packages import activation

    if cwd is not None:
        return activation.effective_command_index(home=home, proj_dir=proj_dir, cwd=cwd)
    return activation.load_activation_index(home=home)


def _is_usage_error(error: Exception) -> bool:
    """Recognize Click errors across Typer's bundled-Click compatibility boundary."""

    return isinstance(error, click.UsageError) or error.__class__.__name__ == "UsageError"


def _path_length(item: tuple[str, CommandRegistrationLike]) -> int:
    """Return the number of words in an indexed command path."""

    return len(item[0].split())


def resolve_registered_command(
    args: Sequence[str], commands: Mapping[str, CommandRegistrationLike]
) -> RegisteredCommandResolution | None:
    """Resolve the longest registered command path at the start of *args*."""

    for path_name, registration in sorted(commands.items(), key=_path_length, reverse=True):
        words = tuple(path_name.split())
        if tuple(args[: len(words)]) == words:
            return RegisteredCommandResolution(registration, tuple(args[len(words) :]))
    return None


class RegisteredProgramCommand(TyperCommand):
    """Click command that gives all remaining arguments to one AgL program."""

    def __init__(self, path_name: str, registration: CommandRegistrationLike) -> None:
        context_settings: dict[str, bool | list[str]] = {
            "allow_extra_args": True,
            "ignore_unknown_options": True,
            "help_option_names": [],
        }
        super().__init__(name="registered-program", context_settings=context_settings)
        self._path_name = path_name
        self._registration = registration

    def invoke(self, ctx: click.Context) -> None:
        # This is the first point at which an unknown command has been proven
        # to be registered, so AgL remains unloaded for all builtin commands.
        from agm.commands.exec_program import run_registered

        run_registered(
            self._registration.program,
            list(ctx.args),
            package=self._registration.package,
            command_path=self._path_name,
        )


class RegisteredCommandGroup(TyperGroup):
    """Root group that falls back to package registrations after builtins miss."""

    def shell_complete(self, ctx: click.Context, incomplete: str) -> list[CompletionItem]:
        """Extend root completion with the next registered-command path segment."""
        from agm.completion import registered_command_completion

        command_path = [*ctx._protected_args, *ctx.args]
        registered_segments, is_registered = registered_command_completion(command_path, incomplete)
        if command_path and (registered_segments or is_registered):
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
                index = load_activation_index(
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
