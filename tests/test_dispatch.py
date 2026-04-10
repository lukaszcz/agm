"""Tests that verify each subcommand dispatches to the correct script
with the correct arguments.  ``os.execvp`` is mocked so nothing actually
runs."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from agm.cli import main, _HELP_TEXTS, _HELP_ALIASES


def run_agm(mock_execvp: MagicMock, argv: list[str]) -> tuple[str, list[str]]:
    """Run ``main(argv)`` and return ``(executable, argv_list)`` passed to execvp."""
    with pytest.raises(SystemExit):
        main(argv)
    mock_execvp.assert_called_once()
    call_args = mock_execvp.call_args
    executable: str = str(call_args[0][0])
    argv_list: list[str] = call_args[0][1]
    return executable, argv_list


# ── help ─────────────────────────────────────────────────────────────────────

class TestHelp:
    def test_help_overview(self, mock_execvp: Any, capsys: Any) -> None:
        """'agm help' (no args) prints the overview and exits 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["help"])
        assert exc_info.value.code == 0
        mock_execvp.assert_not_called()
        out = capsys.readouterr().out
        assert "agm - Agent Management Framework" in out
        assert "Commands:" in out
        assert "open" in out

    def test_help_specific_command(self, mock_execvp: Any, capsys: Any) -> None:
        """'agm help open' prints detailed help for open."""
        with pytest.raises(SystemExit) as exc_info:
            main(["help", "open"])
        assert exc_info.value.code == 0
        mock_execvp.assert_not_called()
        out = capsys.readouterr().out
        assert "agm open" in out
        assert "BRANCH" in out

    def test_help_alias(self, mock_execvp: Any, capsys: Any) -> None:
        """'agm help br' shows the same help as 'agm help branch'."""
        with pytest.raises(SystemExit) as exc_info:
            main(["help", "br"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "agm branch sync" in out

    def test_help_unknown_command(self, mock_execvp: Any, capsys: Any) -> None:
        """'agm help bogus' prints an error and exits 1."""
        with pytest.raises(SystemExit) as exc_info:
            main(["help", "bogus"])
        assert exc_info.value.code == 1
        mock_execvp.assert_not_called()
        err = capsys.readouterr().err
        assert "unknown command" in err

    def test_all_commands_have_help(self) -> None:
        """Every canonical command must have a _HELP_TEXTS entry."""
        expected = {"open", "new", "checkout", "init", "fetch", "branch",
                    "config", "worktree", "dep", "run", "tmux", "help"}
        assert set(_HELP_TEXTS.keys()) == expected

    def test_all_aliases_resolve(self) -> None:
        """Every alias in _HELP_ALIASES must point to a valid _HELP_TEXTS key."""
        for alias, canonical in _HELP_ALIASES.items():
            assert canonical in _HELP_TEXTS, f"alias '{alias}' -> '{canonical}' not in _HELP_TEXTS"


# ── branch sync ──────────────────────────────────────────────────────────────

class TestBranchSync:
    def test_br_sync(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["br", "sync"])
        assert "brsync.sh" in exe
        assert argv == ["brsync.sh"]

    def test_branch_sync(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["branch", "sync"])
        assert "brsync.sh" in exe


# ── config copy ──────────────────────────────────────────────────────────────

class TestConfigCopy:
    def test_config_cp(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["config", "cp", "mydir"])
        assert "cpconfig.sh" in exe
        assert argv == ["cpconfig.sh", "mydir"]

    def test_config_copy_with_d(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["config", "copy", "-d", "/proj", "target"])
        assert "cpconfig.sh" in exe
        assert argv == ["cpconfig.sh", "-d", "/proj", "target"]


# ── worktree checkout ────────────────────────────────────────────────────────

class TestWorktreeCheckout:
    def test_wt_co(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["wt", "co", "feat/x"])
        assert "mkwt.sh" in exe
        assert argv == ["mkwt.sh", "feat/x"]

    def test_wt_co_with_b(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["wt", "co", "-b", "new-br"])
        assert "mkwt.sh" in exe
        assert argv == ["mkwt.sh", "-b", "new-br"]

    def test_worktree_checkout_with_d(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["worktree", "checkout", "-d", "/wt", "br"])
        assert "mkwt.sh" in exe
        assert argv == ["mkwt.sh", "-d", "/wt", "br"]


# ── worktree new ─────────────────────────────────────────────────────────────

class TestWorktreeNew:
    def test_wt_new(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["wt", "new", "feat/y"])
        assert "mkwt.sh" in exe
        assert argv == ["mkwt.sh", "-b", "feat/y"]

    def test_wt_new_with_d(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["wt", "new", "-d", "/custom", "feat/z"])
        assert "mkwt.sh" in exe
        assert argv == ["mkwt.sh", "-b", "feat/z", "-d", "/custom"]


# ── worktree remove ──────────────────────────────────────────────────────────

class TestWorktreeRemove:
    def test_wt_rm(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["wt", "rm", "old-branch"])
        assert "rmwt.sh" in exe
        assert argv == ["rmwt.sh", "old-branch"]

    def test_wt_rm_force(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["worktree", "remove", "-f", "old-branch"])
        assert "rmwt.sh" in exe
        assert argv == ["rmwt.sh", "-f", "old-branch"]


# ── dep ──────────────────────────────────────────────────────────────────────

class TestDep:
    def test_dep_new(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["dep", "new", "https://github.com/org/repo.git"])
        assert "pm-dep.sh" in exe
        assert argv == ["pm-dep.sh", "new", "https://github.com/org/repo.git"]

    def test_dep_new_with_branch(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["dep", "new", "-b", "dev", "https://github.com/org/repo.git"])
        assert argv == ["pm-dep.sh", "new", "-b", "dev", "https://github.com/org/repo.git"]

    def test_dep_switch(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["dep", "switch", "mylib", "feat/x"])
        assert "pm-dep.sh" in exe
        assert argv == ["pm-dep.sh", "switch", "mylib", "feat/x"]

    def test_dep_switch_create(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["dep", "switch", "-b", "mylib", "feat/x"])
        assert argv == ["pm-dep.sh", "switch", "mylib", "-b", "feat/x"]


# ── fetch ────────────────────────────────────────────────────────────────────

class TestFetch:
    def test_fetch(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["fetch"])
        assert "pm-fetch.sh" in exe
        assert argv == ["pm-fetch.sh"]


# ── init ─────────────────────────────────────────────────────────────────────

class TestInit:
    def test_init_url(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["init", "https://github.com/org/repo.git"])
        assert "pm-init.sh" in exe
        assert argv == ["pm-init.sh", "https://github.com/org/repo.git"]

    def test_init_name_and_url(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["init", "myproj", "https://github.com/org/repo.git"])
        assert argv == ["pm-init.sh", "myproj", "https://github.com/org/repo.git"]

    def test_init_with_branch(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["init", "-b", "dev", "myproj"])
        assert argv == ["pm-init.sh", "-b", "dev", "myproj"]


# ── open ─────────────────────────────────────────────────────────────────────

class TestOpen:
    def test_open_bare(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["open"])
        assert "pm.sh" in exe
        assert argv == ["pm.sh", "open"]

    def test_open_with_branch(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["open", "feat/x"])
        assert argv == ["pm.sh", "open", "feat/x"]

    def test_open_with_pane_count(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["open", "-n", "6"])
        assert argv == ["pm.sh", "open", "-n", "6"]

    def test_open_with_all(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["open", "-n", "3", "main"])
        assert argv == ["pm.sh", "open", "-n", "3", "main"]


# ── new (pm.sh new) ─────────────────────────────────────────────────────────

class TestNew:
    def test_new(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["new", "feat/y"])
        assert "pm.sh" in exe
        assert argv == ["pm.sh", "new", "feat/y"]

    def test_new_with_parent(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["new", "-p", "main", "feat/y"])
        assert argv == ["pm.sh", "new", "-p", "main", "feat/y"]

    def test_new_with_all(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["new", "-n", "2", "-p", "main", "feat/y"])
        assert argv == ["pm.sh", "new", "-n", "2", "-p", "main", "feat/y"]


# ── co / checkout ────────────────────────────────────────────────────────────

class TestCheckout:
    def test_co(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["co", "feat/z"])
        assert "pm.sh" in exe
        assert argv == ["pm.sh", "co", "feat/z"]

    def test_checkout(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["checkout", "feat/z"])
        assert argv == ["pm.sh", "co", "feat/z"]

    def test_co_with_all(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["co", "-n", "4", "-p", "dev", "feat/z"])
        assert argv == ["pm.sh", "co", "-n", "4", "-p", "dev", "feat/z"]


# ── run ──────────────────────────────────────────────────────────────────────

class TestRun:
    def test_run_simple(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["run", "npm", "test"])
        assert "sandbox.sh" in exe
        assert argv == ["sandbox.sh", "npm", "test"]

    def test_run_with_f(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["run", "-f", "ci.json", "make"])
        assert argv == ["sandbox.sh", "-f", "ci.json", "make"]

    def test_run_no_patch(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["run", "--no-patch", "echo", "hi"])
        assert argv == ["sandbox.sh", "--no-patch", "echo", "hi"]


# ── tmux new ─────────────────────────────────────────────────────────────────

class TestTmuxNew:
    def test_tmux_new_bare(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["tmux", "new"])
        assert "tmux.sh" in exe
        assert argv == ["tmux.sh"]

    def test_tmux_new_with_all(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["tmux", "new", "-d", "-n", "8", "mysession"])
        assert argv == ["tmux.sh", "-d", "-n", "8", "mysession"]

    def test_tmux_new_detach_long(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["tmux", "new", "--detach"])
        assert argv == ["tmux.sh", "-d"]


# ── tmux layout ──────────────────────────────────────────────────────────────

class TestTmuxLayout:
    def test_tmux_layout(self, mock_execvp: Any) -> None:
        exe, argv = run_agm(mock_execvp, ["tmux", "layout", "4", "@1", "200", "50"])
        assert "tmux-apply-layout.sh" in exe
        assert argv == ["tmux-apply-layout.sh", "4", "@1", "200", "50"]
