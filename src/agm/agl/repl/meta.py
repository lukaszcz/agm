"""Meta-command (``:`` prefix) dispatch for the AgL REPL.

A meta-command is any console line whose first non-blank character is ``:`` —
``:help``, ``:quit``, etc.  The leading colon never collides with AgL syntax
(no AgL statement begins with ``:``), so the loop can route on that single
character.

The meta-command set is implemented here: ``:help``, ``:quit`` /
``:exit``, ``:reset``, ``:type``, ``:bindings`` / ``:env``, ``:agents``,
``:params``, ``:set``, ``:agent``, ``:load``, ``:save``, plus a clean error for
an unknown ``:command``.  The dispatcher is a registry/table (``_COMMANDS``), so
the command set is a single source of truth shared by the dispatcher and the
completer.

**Runtime extension seam:** :func:`register_meta_command` registers an additional
``MetaCommand`` at runtime (a host can extend the surface without editing this
module).  Each handler takes ``(arg, ctx)`` and returns a :class:`MetaOutcome`.
The completer's name list is derived from ``_COMMANDS`` automatically, so a newly
registered command is offered in tab-completion for free.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from agm.agl.repl.agentmode import AgentMode
from agm.agl.repl.themes import THEME_NAMES

if TYPE_CHECKING:
    from agm.agl.repl.session import ReplSession


@dataclass(slots=True)
class MetaContext:
    """Mutable console state threaded through meta-command handlers.

    ``session``     — the live :class:`ReplSession` (handlers query/mutate it).
    ``echo``        — whether successful entries are echoed; the loop reads this
                      live, so ``:set echo on|off`` toggles it by mutation.
    ``agent_mode``  — the shared, mutable agent-call mode holder (``:agent`` reads
                      and mutates it; M4's confirming wrapper will read it). It has
                      no observable effect on evaluation until M4 wires the wrapper.
    ``quit``        — set ``True`` by a handler to ask the loop to exit.
    ``theme``       — current highlight theme name; mutated by ``:theme`` so the
                      loop can update the ``PromptSession`` style live.
    """

    session: "ReplSession"
    echo: bool = True
    agent_mode: AgentMode = field(default_factory=AgentMode)
    quit: bool = False
    theme: str = "auto"


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
    width = max(len(command.usage) for command in _COMMANDS)
    for command in _COMMANDS:
        lines.append(f"  {command.usage:<{width}} {command.summary}")
    return MetaOutcome(text="\n".join(lines))


def _handle_quit(arg: str, ctx: MetaContext) -> MetaOutcome:
    """``:quit`` / ``:exit`` — leave the REPL."""
    del arg
    ctx.quit = True
    return MetaOutcome(text=None, quit=True)


def _handle_reset(arg: str, ctx: MetaContext) -> MetaOutcome:
    """``:reset`` — clear the entire session env (bindings, types, decls, params)."""
    del arg
    ctx.session.reset()
    return MetaOutcome(text="Session reset.")


def _handle_type(arg: str, ctx: MetaContext) -> MetaOutcome:
    """``:type EXPR`` — type-check EXPR against the session and print its type.

    No evaluation, no promotion.  An empty EXPR prints a usage hint; any
    pipeline failure (syntax / scope / type / non-expression) is caught and
    returned as a clean error string so it never escapes ``dispatch_meta``.
    """
    from agm.agl.diagnostics import AglError

    if not arg:
        return MetaOutcome(text="usage: :type EXPR")
    try:
        type_str = ctx.session.type_of(arg)
    except AglError as exc:
        return MetaOutcome(text=str(exc))
    return MetaOutcome(text=type_str)


def _handle_bindings(arg: str, ctx: MetaContext) -> MetaOutcome:
    """``:bindings`` / ``:env`` — list current bindings as ``name : Type = value``."""
    del arg
    from agm.agl.repl.render import format_typed_value

    bindings = ctx.session.bindings()
    if not bindings:
        return MetaOutcome(text="No bindings.")
    lines = [format_typed_value(name, typ, value) for name, typ, value in bindings]
    return MetaOutcome(text="\n".join(lines))


def _handle_agents(arg: str, ctx: MetaContext) -> MetaOutcome:
    """``:agents`` — list available agent names and report the current mode."""
    del arg
    names = ctx.session.agents()
    lines: list[str] = []
    if names:
        lines.append("Available agents:")
        lines.extend(f"  {name}" for name in names)
    else:
        lines.append("No agents available (only the default 'ask' agent, if configured).")
    lines.append(f"Agent-call mode: {ctx.agent_mode.mode}")
    return MetaOutcome(text="\n".join(lines))


def _handle_params(arg: str, ctx: MetaContext) -> MetaOutcome:
    """``:params`` — list declared params with their resolved values (``name : Type = value``)."""
    del arg
    from agm.agl.repl.render import format_typed_value

    params = ctx.session.declared_params()
    if not params:
        return MetaOutcome(text="No params declared.")
    lines = [format_typed_value(name, typ, value) for name, typ, value in params]
    return MetaOutcome(text="\n".join(lines))


def _handle_set(arg: str, ctx: MetaContext) -> MetaOutcome:
    """``:set echo on|off`` — toggle result echoing.

    Only ``echo on|off`` is supported.  Input-setting via ``:set name=value``
    is no longer available (M6): params are resolved eagerly from config or
    defaults when declared.
    """
    echo_outcome = _try_set_echo(arg, ctx)
    if echo_outcome is not None:
        return echo_outcome
    return MetaOutcome(text="usage: :set echo on|off")


def _try_set_echo(arg: str, ctx: MetaContext) -> MetaOutcome | None:
    """Handle the ``:set echo on|off`` special case, or ``None`` if not that form."""
    parts = arg.split()
    if len(parts) != 2 or parts[0] != "echo":
        return None
    state = parts[1]
    if state == "on":
        ctx.echo = True
        return MetaOutcome(text="Echo on.")
    if state == "off":
        ctx.echo = False
        return MetaOutcome(text="Echo off.")
    return MetaOutcome(text="usage: :set echo on|off")


def _handle_agent(arg: str, ctx: MetaContext) -> MetaOutcome:
    """``:agent confirm|auto`` — set the agent-call mode; no arg reports it.

    The mode is recorded in the shared :class:`AgentMode` holder so M4's
    confirming wrapper can read it; it has no observable effect until then.
    """
    if not arg:
        return MetaOutcome(text=f"Agent-call mode: {ctx.agent_mode.mode}")
    if arg == "confirm":
        ctx.agent_mode.mode = "confirm"
        return MetaOutcome(text="Agent-call mode: confirm")
    if arg == "auto":
        ctx.agent_mode.mode = "auto"
        return MetaOutcome(text="Agent-call mode: auto")
    return MetaOutcome(text="usage: :agent confirm|auto")


def _handle_load(arg: str, ctx: MetaContext) -> MetaOutcome:
    """``:load FILE`` — run a file's statements into the session, one per entry.

    The file is evaluated incrementally (one top-level statement per entry, as if
    typed at the prompt), so each statement's echo / errors / warnings surface via
    ``render_entry_result`` exactly as an interactive entry would; the rendered
    texts are joined with newlines.  The load halts at the first failing statement.
    A file-not-found or read error is caught and returned as a clean error; an
    empty / comment-only file (no statements) produces a terse note.
    """
    from agm.agl.repl.render import render_entry_result

    if not arg:
        return MetaOutcome(text="usage: :load FILE")
    try:
        results = ctx.session.load_file(Path(arg))
    except OSError as exc:
        return MetaOutcome(text=f"Error: cannot read {arg}: {exc.strerror or exc}")
    if not results:
        return MetaOutcome(text=f"Loaded {arg} (no statements to run).")
    rendered = [
        text
        for r in results
        if (text := render_entry_result(r, echo=ctx.echo, check_only=False)) is not None
    ]
    return MetaOutcome(text="\n".join(rendered) if rendered else None)


def _handle_save(arg: str, ctx: MetaContext) -> MetaOutcome:
    """``:save FILE`` — write the accumulated session source to FILE.

    A write / OS error is caught and returned as a clean error.
    """
    from agm.core.fs import write_text

    if not arg:
        return MetaOutcome(text="usage: :save FILE")
    try:
        write_text(Path(arg), ctx.session.dump_source())
    except OSError as exc:
        return MetaOutcome(text=f"Error: cannot write {arg}: {exc.strerror or exc}")
    return MetaOutcome(text=f"Saved session source to {arg}")


def _handle_theme(arg: str, ctx: MetaContext) -> MetaOutcome:
    """``:theme [dark|light|auto]`` — show or switch the syntax-highlighting theme.

    With no argument, prints the current theme name.  With an argument, switches
    to the named theme; the console loop observes the change and updates the
    ``PromptSession`` style live, then persists the choice via its
    ``on_theme_save`` callback.
    """
    arg = arg.strip()
    if not arg:
        return MetaOutcome(text=f"Theme: {ctx.theme}")
    if arg not in THEME_NAMES:
        names = ", ".join(THEME_NAMES)
        return MetaOutcome(text=f"Unknown theme {arg!r}. Available: {names}.")
    ctx.theme = arg
    return MetaOutcome(text=f"Theme set to {arg!r}.")


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
    MetaCommand(
        names=("reset",),
        usage=":reset",
        summary="Clear the entire session (bindings, types, decls, params).",
        handler=_handle_reset,
    ),
    MetaCommand(
        names=("type",),
        usage=":type EXPR",
        summary="Type-check EXPR against the session; print its type (no eval).",
        handler=_handle_type,
    ),
    MetaCommand(
        names=("bindings", "env"),
        usage=":bindings / :env",
        summary="List current bindings with types and values.",
        handler=_handle_bindings,
    ),
    MetaCommand(
        names=("agents",),
        usage=":agents",
        summary="List available agents and the current agent-call mode.",
        handler=_handle_agents,
    ),
    MetaCommand(
        names=("params",),
        usage=":params",
        summary="List declared params with their resolved values.",
        handler=_handle_params,
    ),
    MetaCommand(
        names=("set",),
        usage=":set echo on|off",
        summary="Toggle result echoing.",
        handler=_handle_set,
    ),
    MetaCommand(
        names=("agent",),
        usage=":agent confirm|auto",
        summary="Switch the agent-call mode (or report it with no arg).",
        handler=_handle_agent,
    ),
    MetaCommand(
        names=("load",),
        usage=":load FILE",
        summary="Parse and run an .agl file's statements into the session.",
        handler=_handle_load,
    ),
    MetaCommand(
        names=("save",),
        usage=":save FILE",
        summary="Write the accumulated session source to a file.",
        handler=_handle_save,
    ),
    MetaCommand(
        names=("theme",),
        usage=":theme [dark|light|auto]",
        summary="Show or switch the syntax-highlighting theme (saved to config).",
        handler=_handle_theme,
    ),
]


# Module-level caches for the command index and name tuple.  Both are rebuilt
# once at import time (when ``_COMMANDS`` is fully populated) and invalidated
# by ``register_meta_command`` whenever a new command is added at runtime.
_command_index_cache: dict[str, MetaCommand] | None = None
_command_names_cache: tuple[str, ...] | None = None


def _rebuild_caches() -> None:
    """Rebuild both module-level caches from the current ``_COMMANDS`` list."""
    global _command_index_cache, _command_names_cache
    index: dict[str, MetaCommand] = {}
    names: list[str] = []
    for command in _COMMANDS:
        for name in command.names:
            index[name] = command
            names.append(f":{name}")
    _command_index_cache = index
    _command_names_cache = tuple(names)


def register_meta_command(command: MetaCommand) -> None:
    """Register an additional meta-command (the M3 extension entry point)."""
    _COMMANDS.append(command)
    _rebuild_caches()


def _command_index() -> dict[str, MetaCommand]:
    """Return the cached name → command lookup, building it on first call."""
    if _command_index_cache is None:
        _rebuild_caches()
    assert _command_index_cache is not None
    return _command_index_cache


def meta_command_names() -> tuple[str, ...]:
    """Return all registered meta-command names with the leading ``:``.

    This is the single source of truth shared with the console completer so
    tab-completion always matches the live registry.
    """
    if _command_names_cache is None:
        _rebuild_caches()
    assert _command_names_cache is not None
    return _command_names_cache


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
    parts = without_colon.split(None, 1)
    name = parts[0] if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""

    command = _command_index().get(name)
    if command is None:
        return MetaOutcome(
            text=f"Unknown command ':{name}'. Type :help for the command list."
        )
    return command.handler(arg, ctx)
