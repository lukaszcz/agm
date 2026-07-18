"""Tests for agm.project.config_git."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

import agm.core.dry_run as dry_run
import agm.project.config_git as config_git
import agm.vcs.git as git_helpers
from agm.project.config_git import _add_paths, commit_config_dir_changes

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is required")


class TestAddPaths:
    """Tests for _add_paths."""

    def test_stages_changes_with_git_add(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        git_root = tmp_path / "config"
        git_root.mkdir()

        captured_cmds: list[list[str]] = []

        def fake_run_capture(
            cmd: list[str],
            *,
            env: dict[str, str] | None = None,
            **_kwargs: object,
        ) -> tuple[int, str, str]:
            captured_cmds.append(cmd)
            return 0, "", ""

        monkeypatch.setattr(config_git, "run_capture", fake_run_capture)

        _add_paths(git_root, [git_root / "feature"], env={})

        assert len(captured_cmds) == 1
        cmd = captured_cmds[0]
        assert "add" in cmd
        assert "-A" in cmd

    def test_ignores_pathspec_did_not_match_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        git_root = tmp_path / "config"
        git_root.mkdir()

        def fake_run_capture(
            cmd: list[str],
            *,
            env: dict[str, str] | None = None,
            **_kwargs: object,
        ) -> tuple[int, str, str]:
            if "add" in cmd:
                return 1, "", "pathspec 'feature' did not match any files"
            return 0, "", ""

        monkeypatch.setattr(config_git, "run_capture", fake_run_capture)

        # Should not raise
        _add_paths(git_root, [git_root / "feature"], env={})

    def test_dry_run_does_not_invoke_git_add(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        git_root = tmp_path / "config"
        git_root.mkdir()

        captured_cmds: list[list[str]] = []

        def fake_run_capture(
            cmd: list[str],
            *,
            env: dict[str, str] | None = None,
            **_kwargs: object,
        ) -> tuple[int, str, str]:
            captured_cmds.append(cmd)
            return 0, "", ""

        monkeypatch.setattr(config_git, "run_capture", fake_run_capture)
        monkeypatch.setattr(dry_run, "enabled", lambda: True)

        _add_paths(git_root, [git_root / "feature"], env={})

        # Dry-run must not mutate the git index.
        assert captured_cmds == []

    def test_reraises_unexpected_git_add_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        git_root = tmp_path / "config"
        git_root.mkdir()

        run_capture_calls: list[list[str]] = []

        def fake_run_capture(
            cmd: list[str],
            *,
            env: dict[str, str] | None = None,
            **_kwargs: object,
        ) -> tuple[int, str, str]:
            run_capture_calls.append(cmd)
            if "add" in cmd:
                return 1, "", "some unexpected error"
            return 0, "", ""

        monkeypatch.setattr(config_git, "run_capture", fake_run_capture)

        with pytest.raises(SystemExit) as exc_info:
            _add_paths(git_root, [git_root / "feature"], env={})

        # The failure is surfaced without re-running the failing git add.
        assert exc_info.value.code == 1
        assert len(run_capture_calls) == 1


class TestCommitConfigDirChanges:
    """Tests for commit_config_dir_changes."""

    def test_uses_default_env_when_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Kept as a mock: a real-repo version would require git identity in
        # os.environ (the process env), which is not portable across CI setups.
        # The assertion verifies that the env=None default is forwarded to
        # require_success rather than silently replaced with {} (which would
        # strip HOME and break git identity resolution in practice).
        project_dir = tmp_path / "proj"
        config_dir = project_dir / "config"
        config_dir.mkdir(parents=True)
        workspace_config = config_dir / "feature"
        workspace_config.mkdir()

        def fake_exact_repo_root(path: Path, *, env: dict[str, str] | None = None) -> Path:
            return config_dir

        def fake_add_paths(
            root: Path, paths: list[Path], *, env: dict[str, str] | None = None
        ) -> None:
            pass

        def fake_has_staged_changes(
            repo_dir: Path,
            paths: Sequence[Path],
            *,
            env: dict[str, str] | None = None,
        ) -> bool:
            return True

        monkeypatch.setattr(git_helpers, "exact_repo_root", fake_exact_repo_root)
        monkeypatch.setattr(config_git, "_add_paths", fake_add_paths)
        monkeypatch.setattr(git_helpers, "has_staged_changes", fake_has_staged_changes)

        commands_run: list[tuple[list[str], dict[str, str] | None]] = []

        def fake_require_success(
            cmd: list[str], *, env: dict[str, str] | None = None, **_kwargs: object
        ) -> None:
            commands_run.append((cmd, env))

        monkeypatch.setattr(config_git, "require_success", fake_require_success)

        commit_config_dir_changes(
            project_dir,
            "chore: test",
            add_paths=[workspace_config],
        )

        assert commands_run[0][1] is None


@needs_git
class TestCommitConfigDirChangesRealRepo:
    """Integration tests exercising the real git commit path.

    Unlike the mocked tests above, these run real ``git`` against a real
    config repository whose author identity lives only in a global config
    reachable via the ``HOME`` of the passed ``env``.  This mirrors the
    common user setup and guards against passing an environment that
    lacks ``HOME``/``PATH`` (where ``git commit`` would fail with
    "Author identity unknown").
    """

    def _make_repo(self, tmp_path: Path) -> tuple[Path, dict[str, str]]:
        home = tmp_path / "home"
        home.mkdir()
        (home / ".gitconfig").write_text(
            "[user]\n\tname = Test User\n\temail = test@example.com\n",
            encoding="utf-8",
        )
        env = {**os.environ, "HOME": str(home)}
        env.pop("GIT_CONFIG_GLOBAL", None)

        project_dir = tmp_path / "proj"
        config_dir = project_dir / "config"
        config_dir.mkdir(parents=True)
        subprocess.run(["git", "-C", str(config_dir), "init", "-q"], check=True, env=env)
        # No local identity is configured: the commit must rely on the
        # global config reachable through HOME.
        return project_dir, env

    def test_commits_new_workspace_config_using_env_identity(self, tmp_path: Path) -> None:
        project_dir, env = self._make_repo(tmp_path)
        config_dir = project_dir / "config"
        workspace_config = config_dir / "feature"
        workspace_config.mkdir()
        (workspace_config / "config.toml").write_text("[deps]\n", encoding="utf-8")

        commit_config_dir_changes(
            project_dir,
            "chore: add config for feature",
            add_paths=[workspace_config],
            env=env,
        )

        log = subprocess.run(
            ["git", "-C", str(config_dir), "log", "--oneline"],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        assert "chore: add config for feature" in log.stdout

    def test_dry_run_leaves_index_and_history_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir, env = self._make_repo(tmp_path)
        config_dir = project_dir / "config"
        workspace_config = config_dir / "feature"
        workspace_config.mkdir()
        (workspace_config / "config.toml").write_text("[deps]\n", encoding="utf-8")

        monkeypatch.setattr(dry_run, "enabled", lambda: True)
        commit_config_dir_changes(
            project_dir,
            "chore: add config for feature",
            add_paths=[workspace_config],
            env=env,
        )

        # Nothing must be staged and no commit must exist after a dry run.
        staged = subprocess.run(
            ["git", "-C", str(config_dir), "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        assert staged.stdout.strip() == ""
        log = subprocess.run(
            ["git", "-C", str(config_dir), "log", "--oneline"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert log.stdout.strip() == ""

    def test_add_paths_commit_excludes_unrelated_staged_changes(self, tmp_path: Path) -> None:
        project_dir, env = self._make_repo(tmp_path)
        config_dir = project_dir / "config"

        # An unrelated file is staged before the scoped commit runs.
        unrelated = config_dir / "unrelated.txt"
        unrelated.write_text("hand-edited\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(config_dir), "add", "unrelated.txt"], check=True, env=env)

        workspace_config = config_dir / "feature"
        workspace_config.mkdir()
        (workspace_config / "config.toml").write_text("[deps]\n", encoding="utf-8")

        commit_config_dir_changes(
            project_dir,
            "chore: add config for feature",
            add_paths=[workspace_config],
            env=env,
        )

        committed = subprocess.run(
            ["git", "-C", str(config_dir), "show", "--name-only", "--format=", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        # Only the scoped path is committed; the unrelated staged file is left alone.
        assert "feature/config.toml" in committed.stdout
        assert "unrelated.txt" not in committed.stdout
        still_staged = subprocess.run(
            ["git", "-C", str(config_dir), "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        assert "unrelated.txt" in still_staged.stdout

    def test_fails_when_env_lacks_identity_and_home(self, tmp_path: Path) -> None:
        project_dir, env = self._make_repo(tmp_path)
        config_dir = project_dir / "config"
        workspace_config = config_dir / "feature"
        workspace_config.mkdir()
        (workspace_config / "config.toml").write_text("[deps]\n", encoding="utf-8")

        # An environment without HOME (the previous open.py regression
        # passed env={}) cannot resolve the global git identity and must
        # fail loudly rather than create a broken or identity-less commit.
        # GIT_CONFIG_NOSYSTEM keeps a system-wide identity from masking it.
        broken_env = {"PATH": env["PATH"], "GIT_CONFIG_NOSYSTEM": "1"}
        with pytest.raises(SystemExit) as exc_info:
            commit_config_dir_changes(
                project_dir,
                "chore: add config for feature",
                add_paths=[workspace_config],
                env=broken_env,
            )
        assert exc_info.value.code != 0

        log = subprocess.run(
            ["git", "-C", str(config_dir), "log", "--oneline"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert log.stdout.strip() == ""

    def test_config_dir_does_not_exist_does_nothing(self, tmp_path: Path) -> None:
        """Config dir being absent is silently ignored — no exception, no commit."""
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        # config subdir intentionally absent; exact_repo_root returns None before
        # calling git (containing_root checks path.exists()), so env={} is safe.
        commit_config_dir_changes(project_dir, "chore: test", env={})

    def test_config_dir_not_git_repo_does_nothing(self, tmp_path: Path) -> None:
        """Config dir that exists but is not a git repo is silently ignored."""
        _, env = self._make_repo(tmp_path)
        # Create a separate project whose config dir has no git init.
        project_dir = tmp_path / "proj2"
        config_dir = project_dir / "config"
        config_dir.mkdir(parents=True)

        commit_config_dir_changes(project_dir, "chore: test", env=env)

        # No git repo should have been created as a side effect.
        assert not (config_dir / ".git").exists()

    def test_no_add_paths_commits_tracked_changes_and_config_toml(self, tmp_path: Path) -> None:
        """Without add_paths, tracked modifications and new config.toml are committed."""
        project_dir, env = self._make_repo(tmp_path)
        config_dir = project_dir / "config"

        # Track a file, then modify it.
        tracked_file = config_dir / "settings.toml"
        tracked_file.write_text("[base]\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(config_dir), "add", "settings.toml"], check=True, env=env)
        subprocess.run(
            ["git", "-C", str(config_dir), "commit", "-m", "initial"], check=True, env=env
        )
        tracked_file.write_text("[base]\nupdated = true\n", encoding="utf-8")

        # Add a new untracked config.toml that should also be picked up.
        config_toml = config_dir / "config.toml"
        config_toml.write_text("[project]\n", encoding="utf-8")

        commit_config_dir_changes(project_dir, "chore: update config", env=env)

        committed = subprocess.run(
            ["git", "-C", str(config_dir), "show", "--name-only", "--format=", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        assert "settings.toml" in committed.stdout
        assert "config.toml" in committed.stdout
        log = subprocess.run(
            ["git", "-C", str(config_dir), "log", "--format=%s", "-1"],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        assert "chore: update config" in log.stdout

    def test_no_add_paths_commits_tracked_changes_without_config_toml(self, tmp_path: Path) -> None:
        """Without add_paths and no config.toml present, only tracked changes commit."""
        project_dir, env = self._make_repo(tmp_path)
        config_dir = project_dir / "config"

        # Track a file, then modify it; no config.toml exists.
        tracked_file = config_dir / "settings.toml"
        tracked_file.write_text("[base]\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(config_dir), "add", "settings.toml"], check=True, env=env)
        subprocess.run(
            ["git", "-C", str(config_dir), "commit", "-m", "initial"], check=True, env=env
        )
        tracked_file.write_text("[base]\nupdated = true\n", encoding="utf-8")

        commit_config_dir_changes(project_dir, "chore: update config", env=env)

        committed = subprocess.run(
            ["git", "-C", str(config_dir), "show", "--name-only", "--format=", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        assert "settings.toml" in committed.stdout
        assert "config.toml" not in committed.stdout

    def test_no_staged_changes_does_not_create_commit(self, tmp_path: Path) -> None:
        """When nothing is staged after _add_paths, no commit is created."""
        project_dir, env = self._make_repo(tmp_path)
        config_dir = project_dir / "config"
        # An empty workspace dir has no files to stage.
        workspace_config = config_dir / "feature"
        workspace_config.mkdir()

        commit_config_dir_changes(
            project_dir,
            "chore: test",
            add_paths=[workspace_config],
            env=env,
        )

        log = subprocess.run(
            ["git", "-C", str(config_dir), "log", "--oneline"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert log.stdout.strip() == ""

    def test_empty_add_paths_does_not_create_commit(self, tmp_path: Path) -> None:
        """Passing add_paths=[] causes an early return with no git operations."""
        project_dir, env = self._make_repo(tmp_path)
        config_dir = project_dir / "config"

        # Stage a file so a commit would be possible if add_paths were not empty.
        staged_file = config_dir / "something.txt"
        staged_file.write_text("content\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(config_dir), "add", "something.txt"], check=True, env=env)

        commit_config_dir_changes(project_dir, "chore: test", add_paths=[], env=env)

        # No commit should have been created.
        log = subprocess.run(
            ["git", "-C", str(config_dir), "log", "--oneline"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert log.stdout.strip() == ""
        # The staged file must remain staged (function did nothing).
        still_staged = subprocess.run(
            ["git", "-C", str(config_dir), "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        assert "something.txt" in still_staged.stdout
