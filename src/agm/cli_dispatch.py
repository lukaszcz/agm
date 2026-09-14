"""Lazy fallback dispatch for commands registered by active packages."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, TypedDict, cast

import click
from click.shell_completion import CompletionItem
from typer.core import TyperCommand, TyperGroup, TyperOption

from agm.cli_support.args import ExecArgs
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
    """Load the command index for the project-selected packages.

    Dispatch is a read of activation state: an editable package's source
    failing discovery falls back to its manifest's own commands rather than
    breaking every ``agm`` command.
    """

    from agm.packages import activation

    return activation.effective_command_index(
        home=home, proj_dir=proj_dir, cwd=cwd, fallback_to_manifest_commands=True
    )


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


def registered_run_options(ctx: click.Context) -> tuple[TyperOption, ...]:
    """Return ``agm exec``'s run-time options, which a registered command parses too.

    They are ``agm exec``'s own option objects, taken from the ``agm`` group at
    the root of *ctx*, so both commands spell, parse, complete, and document
    them identically.
    """
    from agm.cli_support.program_options import REGISTERED_RUN_FLAGS

    exec_command = cast(click.Group, ctx.find_root().command).commands["exec"]
    return tuple(
        param
        for param in exec_command.params
        if isinstance(param, TyperOption)
        and {*param.opts, *param.secondary_opts} <= REGISTERED_RUN_FLAGS
    )


class _RunOptionValues(TypedDict):
    """The ``ExecArgs`` fields :func:`registered_run_options` fill, keyed by option name."""

    strict_json: bool | None
    max_call_depth: int | None
    default_agent: str | None
    log_file: str | None
    no_log: bool
    log: bool
    timeout: str | None
    no_timeout: bool
    no_log_file: bool


def _help_paragraphs(*parts: str | None) -> str:
    """Join distinct authored help blocks in presentation order."""
    return "\n\n".join({part: None for part in parts if part})


def registered_command_help(
    path_name: str,
    registration: CommandRegistration,
    *,
    program: "ProgramDeclInfo | None",
    command: "ProgramCommand | None",
    run_options: Sequence[click.Parameter],
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
    options are the program's own plus *run_options* and ``--dry-run``. The
    manifest description introduces the source ``@doc`` and any additional
    manifest help.
    """
    from agm.cli_support.program_options import render_program_help

    description = _help_paragraphs(
        registration.description, None if program is None else program.doc, registration.help
    )
    return render_program_help(
        command,
        program_name=f"agm {path_name}",
        description=description or "Run the registered AgL program.",
        extra_options=(*run_options, _dry_run_help_option()),
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
    group_help = registered_group_help(path_name, index.commands)
    if group_help is not None:
        print(group_help)
        return True
    if registration is None or registration.program is None:
        return False
    from agm.cli_support.program_options import REGISTERED_RESERVED_FLAGS, program_command_for
    from agm.commands.exec_program import registered_program_declaration

    program = registered_program_declaration(registration.program, registration.package)
    print(
        registered_command_help(
            path_name,
            registration,
            program=program,
            command=program_command_for(program, REGISTERED_RESERVED_FLAGS),
            run_options=registered_run_options(click.get_current_context()),
        ),
        end="",
    )
    return True


def registered_group_help(
    path_name: str, commands: Mapping[str, CommandRegistration]
) -> str | None:
    """Render authored guidance and a generated listing for an explicit or implicit group."""
    registration = commands.get(path_name)
    if registration is not None and registration.program is not None:
        return None
    descendants = {
        path[len(path_name) + 1 :]: command
        for path, command in sorted(commands.items())
        if path.startswith(path_name + " ")
    }
    if not descendants:
        return None

    program_docs: dict[tuple[str, str], str | None] = {}

    def program_doc(program: str, package: str) -> str | None:
        from agm.commands.exec_program import registered_program_declaration

        key = (program, package)
        if key not in program_docs:
            declaration = registered_program_declaration(program, package)
            program_docs[key] = None if declaration is None else declaration.doc
        return program_docs[key]

    listed_commands: dict[str, click.Command] = {}
    for path, command in descendants.items():
        summary = command.description or command.help
        if not summary:
            if command.program:
                summary = (
                    program_doc(command.program, command.package) or f"Run the {path} workflow."
                )
            else:
                summary = "Browse subcommands."
        listed_commands[path] = click.Command(path, help=summary)

    guidance = _help_paragraphs(
        registration.description if registration else None,
        registration.help if registration else None,
    )
    group = click.Group(
        help=guidance or f"Commands available under {path_name}.",
        commands=listed_commands,
    )
    return group.get_help(click.Context(group, info_name=f"agm {path_name}"))


class RegisteredHelpCommand(TyperCommand):
    """A group surface whose default action displays its generated help."""

    def __init__(self, help_text: str) -> None:
        super().__init__(
            name="registered-group",
            context_settings={
                "allow_extra_args": True,
                "ignore_unknown_options": True,
                "help_option_names": [],
            },
        )
        self._help_text = help_text

    def invoke(self, ctx: click.Context) -> None:
        if any(arg not in ("--help", "-h") for arg in ctx.args):
            ctx.fail("unknown subcommand or option")
        print(self._help_text)


def resolve_registered_command(
    args: Sequence[str], commands: Mapping[str, CommandRegistration]
) -> RegisteredCommandResolution | None:
    """Resolve the longest registered command path at the start of *args*."""

    from agm.packages.activation import CommandRegistration

    paths = dict(commands)
    for path, registration in commands.items():
        words = tuple(path.split())
        for length in range(1, len(words)):
            paths.setdefault(
                " ".join(words[:length]), CommandRegistration(registration.package, None)
            )
    for path_name, registration in sorted(paths.items(), key=_path_length, reverse=True):
        words = tuple(path_name.split())
        if tuple(args[: len(words)]) == words:
            return RegisteredCommandResolution(registration, tuple(args[len(words) :]))
    return None


class RegisteredProgramCommand(TyperCommand):
    """Click command that runs one AgL program with *run_options* and its remaining arguments.

    *run_options* are :func:`registered_run_options`.
    """

    def __init__(
        self,
        path_name: str,
        registration: CommandRegistration,
        run_options: Sequence[TyperOption],
    ) -> None:
        context_settings: dict[str, bool | list[str]] = {
            "allow_extra_args": True,
            "ignore_unknown_options": True,
            "help_option_names": [],
        }
        super().__init__(name="registered-program", context_settings=context_settings)
        self.params.extend(run_options)
        self.params.append(_dry_run_option())
        self._run_options = tuple(run_options)
        self._path_name = path_name
        self._registration = registration
        assert registration.program is not None
        self._program = registration.program

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
            REGISTERED_RESERVED_FLAGS,
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
        program_command = program_command_for(program, REGISTERED_RESERVED_FLAGS)
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

        return registered_program_declaration(self._program, self._registration.package)

    def _discover_program_with_artifacts(
        self,
    ) -> "tuple[ProgramDeclInfo | None, ProgramDiscoveryArtifacts | None]":
        """Discover the declaration and retain its reusable static artifacts."""
        from agm.commands.exec_program import registered_program_declaration

        artifacts: list[ProgramDiscoveryArtifacts] = []
        program = registered_program_declaration(
            self._program,
            self._registration.package,
            artifact_sink=artifacts,
        )
        return program, artifacts[0] if artifacts else None

    def invoke(self, ctx: click.Context) -> None:
        from agm.cli_support.program_options import (
            REGISTERED_RESERVED_FLAGS,
            ProgramHelpRequested,
            contains_help_flag,
            program_command_for,
            program_help_requested,
        )
        from agm.cli_support.run_options import exec_option_conflict

        metadata = cast(dict[str, object], ctx.meta)
        cached_program = cast("ProgramDeclInfo | None", metadata.pop("registered_program", None))

        def program() -> "ProgramDeclInfo | None":
            return cached_program if cached_program is not None else self._discover_program()

        if contains_help_flag(ctx.args):
            # This is the first point at which an unknown command has been proven
            # to be registered, so AgL remains unloaded for all builtin commands.
            declaration = program()
            command = program_command_for(declaration, REGISTERED_RESERVED_FLAGS)
            if program_help_requested(ctx.args, command):
                print(self._help(program=declaration, command=command), end="")
                return
        exec_args = ExecArgs(
            file=self._program,
            argument_tokens=list(ctx.args),
            **cast(_RunOptionValues, ctx.params),
        )
        conflict = exec_option_conflict(exec_args)
        if conflict is not None:
            self._usage_error(conflict, program())
        from agm.commands.exec_program import RegisteredProgramUsageError, run_registered

        try:
            run_registered(
                self._program,
                exec_args.argument_tokens,
                args=exec_args,
                package=self._registration.package,
                command_path=self._path_name,
                pipeline_cache=cast(
                    "ProgramDiscoveryArtifacts | None",
                    metadata.pop("registered_pipeline_cache", None),
                ),
            )
        except ProgramHelpRequested as exc:
            # A help request Click recognized only while parsing the program's
            # own command — a flag bundled into a short group of its own. The
            # request carries the very command it was parsing, so the help is
            # rendered from that rather than from a second build.
            print(self._help(program=exc.program, command=exc.command), end="")
        except RegisteredProgramUsageError as exc:
            self._usage_error(exc.message, exc.program)

    def _usage_error(self, message: str, program: "ProgramDeclInfo | None") -> NoReturn:
        """Report *message* with this command's help on stderr, and exit 1."""
        from agm.cli_support.program_options import REGISTERED_RESERVED_FLAGS, program_command_for

        print(f"error: {message}", file=sys.stderr)
        print(file=sys.stderr)
        print(
            self._help(
                program=program,
                command=program_command_for(program, REGISTERED_RESERVED_FLAGS),
            ),
            end="",
            file=sys.stderr,
        )
        raise SystemExit(1)

    def _help(self, *, program: "ProgramDeclInfo | None", command: "ProgramCommand | None") -> str:
        """Render this registered command's help for an already-discovered program."""
        return registered_command_help(
            self._path_name,
            self._registration,
            program=program,
            command=command,
            run_options=self._run_options,
        )


def _command_path(ctx: click.Context) -> list[str]:
    """Return the words a group context left unparsed: a registered command path and its tail.

    Click splits a group's tail into ``_protected_args`` (the first word, the
    one a subcommand name would come from) and ``args`` (the rest), and
    exposes no undeprecated accessor for the pair: ``protected_args`` warns
    and is slated for removal, while ``args`` alone drops the very word that
    starts a registered command path. Click's own ``shell_completion`` reads
    the same attribute for the same reason.
    """
    return [*ctx._protected_args, *ctx.args]


class RegisteredCommandGroup(TyperGroup):
    """Root group that falls back to package registrations after builtins miss."""

    def get_params(self, ctx: click.Context) -> list[click.Parameter]:
        """Return the group's parameters, plus completion-only registered program value options.

        Completion leaves a registered command path unresolved on this
        group's context, so the options its program's flag values complete
        through are added here, and only while completing.
        """
        params = super().get_params(ctx)
        command_path = _command_path(ctx)
        if not ctx.resilient_parsing or not command_path:
            return params
        from agm.completion import registered_command_value_options

        return [*params, *registered_command_value_options(command_path, ctx)]

    def shell_complete(self, ctx: click.Context, incomplete: str) -> list[CompletionItem]:
        """Extend root completion with the next registered-command path segment.

        Past a registered program's path, a non-option token completes that
        program's next positional slot.
        """
        from agm.completion import registered_command_completion

        command_path = _command_path(ctx)
        registered_segments, is_registered = registered_command_completion(command_path, incomplete)
        if command_path and is_registered:
            if incomplete.startswith("-"):
                from agm.completion import registered_command_param_completion

                return registered_command_param_completion(command_path, incomplete, ctx)
            from agm.completion import registered_command_positional_completion

            return [
                *(CompletionItem(segment) for segment in registered_segments),
                *registered_command_positional_completion(command_path, incomplete, ctx),
            ]
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
            group_help = registered_group_help(path_name, index.commands)
            command = (
                RegisteredHelpCommand(group_help)
                if group_help is not None
                else RegisteredProgramCommand(
                    path_name, resolution.registration, registered_run_options(ctx)
                )
            )
            return (
                path_name,
                command,
                list(resolution.trailing_args),
            )
