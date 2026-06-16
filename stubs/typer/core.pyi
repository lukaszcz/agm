"""Minimal stub for typer.core covering the TyperCommand class."""

from __future__ import annotations

import click
from click.shell_completion import CompletionItem


class TyperCommand:
    """Typer's command class, wrapping click.Command.

    Modelled as a standalone class in this stub (not subclassing click.Command
    directly) to avoid ``Any``-propagation via click's ``Callable[..., Any]``
    callback parameter, which would trigger mypy's ``[misc]`` rule on every
    subclass passed as ``type[TyperCommand]``.
    """

    def shell_complete(self, ctx: click.Context, incomplete: str) -> list[CompletionItem]: ...
