"""Tests for the AGM command-tree layout.

End-to-end command coverage is enforced separately and dynamically by
``tests/_command_coverage.py``, which records the commands actually executed
through the ``agm`` binary rather than scanning test sources for call literals.
"""

from __future__ import annotations

from pathlib import Path


def test_command_subpackages_match_cli_command_groups() -> None:
    """commands/ sub-packages must mirror CLI command groups (repository guideline)."""
    commands_dir = Path(__file__).resolve().parents[1] / "src" / "agm" / "commands"
    subpackages = {
        path.name
        for path in commands_dir.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    }

    assert subpackages == {"config", "dep", "loop", "pkg", "sync", "tmux", "workspace", "worktree"}
