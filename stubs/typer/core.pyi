"""Minimal Typer core stubs used by AGM's custom command classes."""

from __future__ import annotations

from collections.abc import Callable

import click
from click.shell_completion import CompletionItem


class TyperOption(click.Option):
    def __init__(
        self,
        *,
        param_decls: list[str],
        default: object = ...,
        callback: Callable[[object, object, bool], None] | None = ...,
        expose_value: bool = ...,
        is_eager: bool = ...,
        is_flag: bool | None = ...,
        help: str | None = ...,
    ) -> None: ...


class TyperGroup:
    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, TyperCommand | None, list[str]]: ...
    def shell_complete(self, ctx: click.Context, incomplete: str) -> list[CompletionItem]: ...


class TyperCommand:
    """Typed façade for Typer's Click command base."""

    params: list[click.Parameter]

    def __init__(
        self,
        *,
        name: str | None = ...,
        context_settings: dict[str, bool | list[str]] | None = ...,
    ) -> None: ...
    def invoke(self, ctx: click.Context) -> None: ...
    def shell_complete(self, ctx: click.Context, incomplete: str) -> list[CompletionItem]: ...
