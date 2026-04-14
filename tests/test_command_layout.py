from __future__ import annotations

from pathlib import Path


def test_command_subpackages_match_cli_command_groups() -> None:
    commands_dir = Path(__file__).resolve().parents[1] / "src" / "agm" / "commands"
    subpackages = {
        path.name
        for path in commands_dir.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    }

    assert subpackages == {"branch", "config", "dep", "tmux", "worktree"}
