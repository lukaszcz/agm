from __future__ import annotations

from collections.abc import Callable
from typing import ParamSpec, TypeVar

CommandParam = ParamSpec("CommandParam")
CommandReturn = TypeVar("CommandReturn")
OptionValue = TypeVar("OptionValue")


class Context:
    resilient_parsing: bool
    invoked_subcommand: str | None
    info_name: str | None
    parent: Context | None
    args: list[str]


class Exit(Exception):
    def __init__(self, code: int | None = ...) -> None: ...


class Typer:
    def __init__(
        self,
        *,
        add_completion: bool = ...,
        context_settings: dict[str, bool | list[str]] | None = ...,
        invoke_without_command: bool = ...,
    ) -> None: ...
    def callback(
        self,
        *,
        invoke_without_command: bool = ...,
    ) -> Callable[
        [Callable[CommandParam, CommandReturn]],
        Callable[CommandParam, CommandReturn],
    ]: ...
    def command(
        self,
        name: str | None = ...,
        *,
        context_settings: dict[str, bool | list[str]] | None = ...,
    ) -> Callable[
        [Callable[CommandParam, CommandReturn]],
        Callable[CommandParam, CommandReturn],
    ]: ...
    def add_typer(self, typer_instance: Typer, *, name: str) -> None: ...
    def __call__(self) -> None: ...


def Option(
    default: OptionValue,
    *param_decls: str,
    callback: Callable[[Context, object, OptionValue], object] | None = ...,
    expose_value: bool = ...,
    is_eager: bool = ...,
    help: str | None = ...,
) -> OptionValue: ...
