"""Canonical catalog of AGM top-level CLI commands.

Single source of truth for the top-level command names and their one-line
overview descriptions, shared by the CLI help layer (:mod:`agm.parser`) and the
AgL reserved-program-name guard (:mod:`agm.agl.semantics.engine_keys`) so the
two never drift.

This is a pure data leaf: it imports nothing from ``agm`` and pulls in no CLI
machinery, so the AgL semantics layer can depend on it without breaching the
AgL firewall (which forbids reaching the parser/CLI, not plain shared data).
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
    ("loop", "Run the loop prompt until completion"),
    ("review", "Run the review prompt"),
    ("revise", "Run the revision prompt"),
    ("refine", "Run review/revise refinement"),
    ("exec", "Execute an AgL workflow program"),
    ("repl", "Start an interactive AgL REPL"),
    ("run", "Run a command in a sandbox"),
    ("config", "Manage project configuration files"),
    ("worktree", "Git worktree management"),
    ("tmux", "Tmux session and layout management"),
    ("help", "Show help for a command"),
)

# All top-level command names, in catalog order.
COMMAND_NAMES: tuple[str, ...] = tuple(name for name, _ in COMMAND_OVERVIEW)
