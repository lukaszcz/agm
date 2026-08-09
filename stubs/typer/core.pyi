"""Minimal Typer core stubs used by AGM's custom command classes."""

from __future__ import annotations

import click
from click.shell_completion import CompletionItem


class TyperGroup:
    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, TyperCommand | None, list[str]]: ...
    def shell_complete(self, ctx: click.Context, incomplete: str) -> list[CompletionItem]: ...


class TyperCommand:
    """Typed façade for Typer's Click command base."""

    def __init__(
        self,
        *,
        name: str | None = ...,
        context_settings: dict[str, bool | list[str]] | None = ...,
    ) -> None: ...
    def invoke(self, ctx: click.Context) -> None: ...
    def shell_complete(self, ctx: click.Context, incomplete: str) -> list[CompletionItem]: ...
