"""Canonical catalog of AGM top-level CLI commands.

Single source of truth for the top-level command names and their one-line
overview descriptions, consumed by the CLI help layer (:mod:`agm.parser`), and
for the top-level names AGM reserves in its command tree and config namespace.

This is a pure data leaf: it imports nothing from ``agm`` and pulls in no CLI
machinery.
"""

from __future__ import annotations

# Ordered (name, one-line description) for every top-level AGM command, as shown
# by ``agm help``.
COMMAND_OVERVIEW: tuple[tuple[str, str], ...] = (
    ("open", "Open a workspace's tmux session"),
    ("close", "Close a branch workspace"),
    ("workspace", "Manage workspaces"),
    ("init", "Initialize a project"),
    ("sync", "Fetch and merge project repositories"),
    ("dep", "Manage dependency checkouts"),
    ("pkg", "Manage AgL packages"),
    ("loop", "Run agent prompts until COMPLETE"),
    ("review", "Run the review prompt"),
    ("revise", "Run the revision prompt"),
    ("refine", "Alternate review and revise"),
    ("exec", "Run an AgL program"),
    ("repl", "Start the AgL REPL"),
    ("check", "Statically check AgL files"),
    ("run", "Run a command in a sandbox"),
    ("config", "Manage project configuration"),
    ("worktree", "Manage Git worktrees"),
    ("tmux", "Manage tmux sessions and layouts"),
    ("help", "Show help"),
)

# All top-level command names, in catalog order.
COMMAND_NAMES: tuple[str, ...] = tuple(name for name, _ in COMMAND_OVERVIEW)

# Built-in command aliases.  These are not command-overview entries, but they
# reserve the same package-command and config-routing namespace as commands.
COMMAND_ALIASES: tuple[str, ...] = ("wsp", "wt")

# Every name reserved by AGM's built-in command surface.
RESERVED_COMMAND_NAMES: frozenset[str] = frozenset(COMMAND_NAMES) | frozenset(COMMAND_ALIASES)

# Top-level config sections that describe AGM's configuration schema rather than
# AgL modules, and key their own nested tables by user-chosen names -- an agent
# CLI, a dependency, a module root, a package pin. ``params`` keeps the removed
# legacy ``[params.*]`` namespace from reading as an AgL module route.
SCHEMA_CONFIG_SECTION_NAMES: frozenset[str] = frozenset(
    {"agent", "deps", "modules", "packages", "params"}
)

# Every top-level config section name AGM owns. A validated package's name and
# command paths never take one, so neither collides with AGM's own config.
RESERVED_CONFIG_SECTION_NAMES: frozenset[str] = RESERVED_COMMAND_NAMES | SCHEMA_CONFIG_SECTION_NAMES


def invalid_command_path(
    command_path: str, *, reserved: frozenset[str] = RESERVED_COMMAND_NAMES
) -> str | None:
    """Describe why *command_path* cannot name a registered command, or ``None``.

    Package commands extend the same command tree as AGM's own commands, so a
    registered path must be plain space-separated words that neither start at
    a *reserved* name nor look like an option. The default reserves only what
    would clash at dispatch, which is all an already-installed registration is
    held to; package validation passes :data:`RESERVED_CONFIG_SECTION_NAMES`.
    """

    words = command_path.split()
    if not words or " ".join(words) != command_path:
        return "must be space-separated words"
    if words[0] in reserved:
        return "begins with a name reserved by AGM"
    if any(word.startswith("-") for word in words):
        return "contains an option"
    return None
