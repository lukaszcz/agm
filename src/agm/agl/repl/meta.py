"""Meta-command (``:`` prefix) dispatch for the AgL REPL.

A meta-command is any console line whose first non-blank character is ``:`` —
``:help``, ``:quit``, etc.  The leading colon never collides with AgL syntax
(no AgL statement begins with ``:``), so the loop can route on that single
character.

This is the **minimal** M2 surface.  Only the commands needed to make the REPL
runnable end-to-end are implemented now: ``:help``, ``:quit`` / ``:exit``, and a
clean error for an unknown ``:command``.  The dispatcher is a registry/table so
M3 can register the richer set (``:reset``, ``:type``, ``:bindings`` / ``:env``,
``:agents``, ``:inputs``, ``:set``, ``:agent``, ``:load``, ``:save``) by adding
one entry to ``_COMMANDS`` — no change to the loop or to ``dispatch_meta``.

**Extension point for M3:** add a ``MetaCommand`` to the ``_COMMANDS`` tuple (or
register one at runtime via :func:`register_meta_command`).  Each handler takes
``(arg, ctx)`` and returns a :class:`MetaOutcome`.  ``META_COMMANDS`` (the name
list the completer reads) is derived from ``_COMMANDS`` automatically, so a new
command is offered in tab-completion for free.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agm.agl.repl.session import ReplSession


@dataclass(slots=True)
class MetaContext:
    """Mutable console state threaded through meta-command handlers.

    ``session``  — the live :class:`ReplSession` (handlers that mutate or query
                   session state use this; M2 handlers do not need it yet).
    ``echo``     — whether successful entries are echoed (``:set echo`` in M3).
    ``quit``     — set ``True`` by a handler to ask the loop to exit.
    """

    session: "ReplSession"
    echo: bool = True
    quit: bool = False


@dataclass(frozen=True, slots=True)
class MetaOutcome:
    """Structured result of dispatching one meta-command.

    ``text``  — text the loop should print (``None`` → print nothing).
    ``quit``  — whether the loop should exit after this command.
    """

    text: str | None = None
    quit: bool = False


# A handler receives the argument string (everything after the command word,
# stripped) and the mutable context, and returns an outcome.
MetaHandler = Callable[[str, MetaContext], MetaOutcome]


@dataclass(frozen=True, slots=True)
class MetaCommand:
    """One registered meta-command: its names, one-line usage, and handler."""

    names: tuple[str, ...]
    usage: str
    summary: str
    handler: MetaHandler


def _handle_help(arg: str, ctx: MetaContext) -> MetaOutcome:
    """``:help`` — list available meta-commands with brief usage."""
    del arg, ctx
    lines = ["Available commands:"]
    for command in _COMMANDS:
        lines.append(f"  {command.usage:<18} {command.summary}")
    return MetaOutcome(text="\n".join(lines))


def _handle_quit(arg: str, ctx: MetaContext) -> MetaOutcome:
    """``:quit`` / ``:exit`` — leave the REPL."""
    del arg
    ctx.quit = True
    return MetaOutcome(text=None, quit=True)


# Registry: the authoritative table of built-in meta-commands.  M3 appends to
# this list (or calls ``register_meta_command``); both the dispatcher and the
# completer's ``META_COMMANDS`` read from it, so there is a single source of
# truth for the command names.
_COMMANDS: list[MetaCommand] = [
    MetaCommand(
        names=("help",),
        usage=":help",
        summary="List available meta-commands.",
        handler=_handle_help,
    ),
    MetaCommand(
        names=("quit", "exit"),
        usage=":quit / :exit",
        summary="Exit the REPL (or press Ctrl-D).",
        handler=_handle_quit,
    ),
]


def register_meta_command(command: MetaCommand) -> None:
    """Register an additional meta-command (the M3 extension entry point)."""
    _COMMANDS.append(command)


def _command_index() -> dict[str, MetaCommand]:
    """Build a name → command lookup from the current registry."""
    index: dict[str, MetaCommand] = {}
    for command in _COMMANDS:
        for name in command.names:
            index[name] = command
    return index


def meta_command_names() -> tuple[str, ...]:
    """Return all registered meta-command names with the leading ``:``.

    This is the single source of truth shared with the console completer so
    tab-completion always matches the live registry.
    """
    names: list[str] = []
    for command in _COMMANDS:
        names.extend(f":{name}" for name in command.names)
    return tuple(names)


def dispatch_meta(line: str, ctx: MetaContext) -> MetaOutcome:
    """Parse a leading-``:`` *line* and route it to its handler.

    *line* is the raw console entry including the leading ``:``.  The first
    whitespace-delimited word (minus the colon) selects the command; the rest is
    passed to the handler as its argument string.  An unknown command yields a
    clean error outcome rather than raising.
    """
    body = line.strip()
    assert body.startswith(":")  # the loop only calls us for ``:`` lines
    without_colon = body[1:]
    name, _, rest = without_colon.partition(" ")
    arg = rest.strip()

    command = _command_index().get(name)
    if command is None:
        return MetaOutcome(
            text=f"Unknown command ':{name}'. Type :help for the command list."
        )
    return command.handler(arg, ctx)
