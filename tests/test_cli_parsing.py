"""Tests for argument parsing – ensures the CLI accepts exactly the right
options and rejects invalid ones."""

from __future__ import annotations

import argparse

import pytest

from agm.cli import build_parser


@pytest.fixture()
def parser() -> argparse.ArgumentParser:
    return build_parser()


# ── helpers ──────────────────────────────────────────────────────────────────

def parse(parser: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace:
    return parser.parse_args(argv)


def assert_rejects(parser: argparse.ArgumentParser, argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


# ── br / branch sync ────────────────────────────────────────────────────────

class TestBranchSync:
    def test_br_sync(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["br", "sync"])
        assert ns.command == "br"
        assert ns.br_command == "sync"

    def test_branch_sync(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["branch", "sync"])
        assert ns.command == "branch"
        assert ns.br_command == "sync"

    def test_br_without_subcommand(self, parser: argparse.ArgumentParser) -> None:
        assert_rejects(parser, ["br"])


# ── config cp / copy ────────────────────────────────────────────────────────

class TestConfigCopy:
    def test_config_cp(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["config", "cp", "mydir"])
        assert ns.command == "config"
        assert ns.config_command == "cp"
        assert ns.dirname == "mydir"
        assert ns.project_dir is None

    def test_config_copy_with_d(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["config", "copy", "-d", "/some/dir", "target"])
        assert ns.config_command == "copy"
        assert ns.project_dir == "/some/dir"
        assert ns.dirname == "target"

    def test_config_cp_missing_dirname(self, parser: argparse.ArgumentParser) -> None:
        assert_rejects(parser, ["config", "cp"])


# ── wt / worktree co / checkout ─────────────────────────────────────────────

class TestWorktreeCheckout:
    def test_wt_co_branch(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["wt", "co", "feat/x"])
        assert ns.command == "wt"
        assert ns.wt_command == "co"
        assert ns.branch == "feat/x"
        assert ns.new_branch is None

    def test_worktree_checkout_with_b(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["worktree", "checkout", "-b", "new-br"])
        assert ns.new_branch == "new-br"
        assert ns.branch is None

    def test_wt_co_with_d(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["wt", "co", "-d", "/wt", "br"])
        assert ns.worktrees_dir == "/wt"
        assert ns.branch == "br"

    def test_wt_co_with_b_and_d(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["wt", "co", "-b", "new-br", "-d", "/wt"])
        assert ns.new_branch == "new-br"
        assert ns.worktrees_dir == "/wt"


# ── wt / worktree new ───────────────────────────────────────────────────────

class TestWorktreeNew:
    def test_wt_new(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["wt", "new", "feat/y"])
        assert ns.wt_command == "new"
        assert ns.branch == "feat/y"

    def test_wt_new_with_d(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["wt", "new", "-d", "/custom", "feat/z"])
        assert ns.worktrees_dir == "/custom"
        assert ns.branch == "feat/z"

    def test_wt_new_missing_branch(self, parser: argparse.ArgumentParser) -> None:
        assert_rejects(parser, ["wt", "new"])


# ── wt / worktree rm / remove ───────────────────────────────────────────────

class TestWorktreeRemove:
    def test_wt_rm(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["wt", "rm", "old-branch"])
        assert ns.wt_command == "rm"
        assert ns.branch == "old-branch"
        assert ns.force is False

    def test_worktree_remove_force(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["worktree", "remove", "-f", "old-branch"])
        assert ns.force is True
        assert ns.branch == "old-branch"

    def test_wt_rm_missing_branch(self, parser: argparse.ArgumentParser) -> None:
        assert_rejects(parser, ["wt", "rm"])


# ── dep ──────────────────────────────────────────────────────────────────────

class TestDep:
    def test_dep_new(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["dep", "new", "https://github.com/org/repo.git"])
        assert ns.dep_command == "new"
        assert ns.repo_url == "https://github.com/org/repo.git"
        assert ns.branch is None

    def test_dep_new_with_branch(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["dep", "new", "-b", "main", "https://github.com/org/repo.git"])
        assert ns.branch == "main"

    def test_dep_switch(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["dep", "switch", "mylib", "feat/x"])
        assert ns.dep_command == "switch"
        assert ns.dep == "mylib"
        assert ns.branch == "feat/x"
        assert ns.create_branch is False

    def test_dep_switch_create(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["dep", "switch", "-b", "mylib", "feat/x"])
        assert ns.create_branch is True

    def test_dep_rm(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["dep", "rm", "mylib/feat/x"])
        assert ns.dep_command == "rm"
        assert ns.target == "mylib/feat/x"
        assert ns.all is False

    def test_dep_rm_all(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["dep", "rm", "--all", "mylib"])
        assert ns.dep_command == "rm"
        assert ns.target == "mylib"
        assert ns.all is True

    def test_dep_rm_missing_target(self, parser: argparse.ArgumentParser) -> None:
        assert_rejects(parser, ["dep", "rm"])

    def test_dep_missing_subcommand(self, parser: argparse.ArgumentParser) -> None:
        assert_rejects(parser, ["dep"])


# ── fetch ────────────────────────────────────────────────────────────────────

class TestFetch:
    def test_fetch(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["fetch"])
        assert ns.command == "fetch"

    def test_fetch_rejects_args(self, parser: argparse.ArgumentParser) -> None:
        assert_rejects(parser, ["fetch", "extra"])


# ── init ─────────────────────────────────────────────────────────────────────

class TestInit:
    def test_init_project_and_url(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["init", "myproj", "https://github.com/org/repo.git"])
        assert ns.positional == ["myproj", "https://github.com/org/repo.git"]
        assert ns.branch is None

    def test_init_url_only(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["init", "https://github.com/org/repo.git"])
        assert ns.positional == ["https://github.com/org/repo.git"]

    def test_init_with_branch(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["init", "-b", "dev", "myproj"])
        assert ns.branch == "dev"
        assert ns.positional == ["myproj"]

    def test_init_missing_args(self, parser: argparse.ArgumentParser) -> None:
        assert_rejects(parser, ["init"])


# ── open ─────────────────────────────────────────────────────────────────────

class TestOpen:
    def test_open_missing_target(self, parser: argparse.ArgumentParser) -> None:
        assert_rejects(parser, ["open"])

    def test_open_repo(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["open", "repo"])
        assert ns.command == "open"
        assert ns.pane_count is None
        assert ns.parent is None
        assert ns.branch == "repo"

    def test_open_with_branch(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["open", "feat/x"])
        assert ns.branch == "feat/x"

    def test_open_with_pane_count(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["open", "-n", "6", "repo"])
        assert ns.pane_count == "6"

    def test_open_with_parent_and_branch(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["open", "-p", "main", "feat/y"])
        assert ns.branch == "feat/y"
        assert ns.parent == "main"

    def test_open_with_all(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["open", "-n", "2", "-p", "main", "feat/y"])
        assert ns.pane_count == "2"
        assert ns.parent == "main"
        assert ns.branch == "feat/y"


# ── run ──────────────────────────────────────────────────────────────────────

class TestRun:
    def test_run_simple(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["run", "npm", "test"])
        assert ns.run_command == ["npm", "test"]
        assert ns.no_patch is False
        assert ns.settings_file is None

    def test_run_with_f(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["run", "-f", "ci.json", "make"])
        assert ns.settings_file == "ci.json"
        assert ns.run_command == ["make"]

    def test_run_no_patch(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["run", "--no-patch", "echo", "hi"])
        assert ns.no_patch is True
        assert ns.run_command == ["echo", "hi"]

    def test_run_no_command(self, parser: argparse.ArgumentParser) -> None:
        # REMAINDER allows empty list – sandbox.sh will handle the error
        ns = parse(parser, ["run"])
        assert ns.run_command == []


# ── tmux new ─────────────────────────────────────────────────────────────────

class TestTmuxNew:
    def test_tmux_new_bare(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["tmux", "new"])
        assert ns.tmux_command == "new"
        assert ns.detach is False
        assert ns.pane_count is None
        assert ns.session_name is None

    def test_tmux_new_with_all(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["tmux", "new", "-d", "-n", "8", "mysession"])
        assert ns.detach is True
        assert ns.pane_count == "8"
        assert ns.session_name == "mysession"

    def test_tmux_new_detach_long(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["tmux", "new", "--detach"])
        assert ns.detach is True


# ── tmux layout ──────────────────────────────────────────────────────────────

class TestTmuxLayout:
    def test_tmux_layout(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["tmux", "layout", "4", "@1", "200", "50"])
        assert ns.tmux_command == "layout"
        assert ns.pane_count == "4"
        assert ns.window_id == "@1"
        assert ns.width == "200"
        assert ns.height == "50"

    def test_tmux_layout_missing_args(self, parser: argparse.ArgumentParser) -> None:
        assert_rejects(parser, ["tmux", "layout", "4", "@1"])


# ── help ─────────────────────────────────────────────────────────────────────

class TestHelp:
    def test_help_bare(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["help"])
        assert ns.command == "help"
        assert ns.help_command is None

    def test_help_with_command(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["help", "open"])
        assert ns.command == "help"
        assert ns.help_command == "open"

    def test_help_with_alias(self, parser: argparse.ArgumentParser) -> None:
        ns = parse(parser, ["help", "br"])
        assert ns.help_command == "br"

    def test_help_with_unknown(self, parser: argparse.ArgumentParser) -> None:
        # argparse accepts any string; validation is in dispatch
        ns = parse(parser, ["help", "bogus"])
        assert ns.help_command == "bogus"


# ── top-level ────────────────────────────────────────────────────────────────

class TestTopLevel:
    def test_no_command(self, parser: argparse.ArgumentParser) -> None:
        assert_rejects(parser, [])

    def test_unknown_command(self, parser: argparse.ArgumentParser) -> None:
        assert_rejects(parser, ["bogus"])


# ── help text coverage ──────────────────────────────────────────────────────

class TestHelpTextCoverage:
    """Verify that _HELP_TEXTS and _HELP_ALIASES are complete and consistent."""

    def test_every_canonical_command_has_help_text(self) -> None:
        from agm.cli import _HELP_TEXTS
        canonical_commands = {
            "open", "init", "fetch", "branch", "config", "worktree", "dep", "run", "tmux", "help",
        }
        for cmd in canonical_commands:
            assert cmd in _HELP_TEXTS, f"missing help text for '{cmd}'"

    def test_every_overview_command_has_help_text(self) -> None:
        from agm.cli import _COMMAND_OVERVIEW, _HELP_TEXTS
        for name, _ in _COMMAND_OVERVIEW:
            # Overview names may include aliases like "checkout (co)".
            canonical = name.split(" (")[0]
            assert canonical in _HELP_TEXTS, (
                f"overview lists '{canonical}' but _HELP_TEXTS has no entry"
            )

    def test_aliases_point_to_valid_commands(self) -> None:
        from agm.cli import _HELP_ALIASES, _HELP_TEXTS
        for alias, target in _HELP_ALIASES.items():
            assert target in _HELP_TEXTS, (
                f"alias '{alias}' → '{target}' but '{target}' not in _HELP_TEXTS"
            )

    def test_no_empty_help_texts(self) -> None:
        from agm.cli import _HELP_TEXTS
        for cmd, text in _HELP_TEXTS.items():
            assert text.strip(), f"help text for '{cmd}' is empty"

    def test_help_texts_contain_command_name(self) -> None:
        from agm.cli import _HELP_TEXTS
        for cmd, text in _HELP_TEXTS.items():
            assert f"agm {cmd}" in text, (
                f"help text for '{cmd}' doesn't mention 'agm {cmd}'"
            )
