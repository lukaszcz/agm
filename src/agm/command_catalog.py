"""Canonical catalog of AGM top-level CLI commands.

Single source of truth for the top-level command names and their one-line
overview descriptions, consumed by the CLI help layer (:mod:`agm.parser`).

This is a pure data leaf: it imports nothing from ``agm`` and pulls in no CLI
machinery.
"""

from __future__ import annotations

# Ordered (name, one-line description) for every top-level AGM command, as shown
# by ``agm help``.
COMMAND_OVERVIEW: tuple[tuple[str, str], ...] = (
    ("open", "Open a workspace"),
    ("close", "Close a workspace"),
    ("workspace", "Manage AGM workspaces"),
    ("init", "Initialize a project"),
    ("sync", "Fetch and merge project repositories"),
    ("dep", "Manage project dependency checkouts"),
    ("pkg", "Manage AgL packages"),
    ("loop", "Run the loop prompt until completion"),
    ("review", "Run the review prompt"),
    ("revise", "Run the revision prompt"),
    ("refine", "Run review/revise refinement"),
    ("exec", "Execute an AgL workflow program"),
    ("repl", "Start an interactive AgL REPL"),
    ("check", "Statically check AgL files"),
    ("run", "Run a command in a sandbox"),
    ("config", "Manage project configuration files"),
    ("worktree", "Git worktree management"),
    ("tmux", "Tmux session and layout management"),
    ("help", "Show help for a command"),
)

# All top-level command names, in catalog order.
COMMAND_NAMES: tuple[str, ...] = tuple(name for name, _ in COMMAND_OVERVIEW)

# Built-in command aliases.  These are not command-overview entries, but they
# reserve the same package-command and config-routing namespace as commands.
COMMAND_ALIASES: tuple[str, ...] = ("wsp", "wt")

# Every name reserved by AGM's built-in command surface.
RESERVED_COMMAND_NAMES: frozenset[str] = frozenset(COMMAND_NAMES) | frozenset(COMMAND_ALIASES)


def invalid_command_path(command_path: str) -> str | None:
    """Describe why *command_path* cannot name a registered command, or ``None``.

    Package commands extend the same command tree as AGM's own commands, so a
    registered path must be plain space-separated words that neither start at
    a reserved name nor look like an option.
    """

    words = command_path.split()
    if not words or " ".join(words) != command_path:
        return "must be space-separated words"
    if words[0] in RESERVED_COMMAND_NAMES:
        return "begins with a reserved AGM command"
    if any(word.startswith("-") for word in words):
        return "contains an option"
    return None
