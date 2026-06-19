"""Tests for agm.project.workspace_shell."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from agm.project.workspace_shell import (
    SHELL_SUBDIR,
    WRAPPER_NAME,
    _sanitize_session_key,
    ensure_workspace_shell,
    regenerate_workspace_shell,
    remove_workspace_shell,
    workspace_shell_dir,
    workspace_shell_root,
)


class TestWorkspaceShellRoot:
    def test_honors_xdg_cache_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        assert workspace_shell_root() == tmp_path / "cache" / "agm" / SHELL_SUBDIR

    def test_falls_back_to_home_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        assert workspace_shell_root() == tmp_path / "home" / ".cache" / "agm" / SHELL_SUBDIR


class TestSanitizeSessionKey:
    def test_replaces_slashes(self) -> None:
        key = _sanitize_session_key("proj/feat/test")
        assert "/" not in key
        assert key.startswith("proj__feat__test-")

    def test_empty_session_name_uses_default(self) -> None:
        key = _sanitize_session_key("")
        assert key.startswith("session-")

    def test_strips_leading_trailing_dots(self) -> None:
        key = _sanitize_session_key("...weird...")
        assert not key.startswith(".")
        assert not key.endswith(".")

    def test_distinct_names_get_distinct_keys(self) -> None:
        assert _sanitize_session_key("proj/feat") != _sanitize_session_key("proj/other")


class TestEnsureWorkspaceShell:
    def _setup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        return workspace_shell_dir("s")

    def test_real_shell_falls_back_when_shell_points_at_wrapper(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        shell_dir = self._setup(tmp_path, monkeypatch)
        wrapper_aware_env = {"SHELL": str(shell_dir / WRAPPER_NAME)}
        wrapper = ensure_workspace_shell("s", env=wrapper_aware_env)
        text = wrapper.read_text(encoding="utf-8")
        # The wrapper must exec /bin/sh, not itself.
        assert "AGM_REAL_SHELL=/bin/sh" in text

    def test_real_shell_uses_env_shell(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._setup(tmp_path, monkeypatch)
        wrapper = ensure_workspace_shell("s", env={"SHELL": "/bin/bash"})
        text = wrapper.read_text(encoding="utf-8")
        assert "AGM_REAL_SHELL=/bin/bash" in text

    def test_real_shell_defaults_when_she_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._setup(tmp_path, monkeypatch)
        wrapper = ensure_workspace_shell("s", env={})
        text = wrapper.read_text(encoding="utf-8")
        assert "AGM_REAL_SHELL=/bin/sh" in text

    def test_wrapper_self_heal_heredoc_writes_rc_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._setup(tmp_path, monkeypatch)
        wrapper = ensure_workspace_shell("s", env={"SHELL": "/bin/bash"})
        text = wrapper.read_text(encoding="utf-8")
        # The self-heal block embeds each rc body via a heredoc.
        assert "AGM_EOF" in text
        assert "cat > " in text
        assert ". \"$HOME/.bashrc\"" in text


class TestRegenerateWorkspaceShell:
    def test_rewrites_rc_files_into_existing_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        shell_dir = workspace_shell_dir("s")
        shell_dir.mkdir(parents=True)
        # Remove the rc subdirs so regeneration has to recreate them.
        regenerate_workspace_shell(shell_dir)
        assert (shell_dir / "zsh" / ".zshrc").is_file()
        assert (shell_dir / "bash" / "bashrc").is_file()
        assert (shell_dir / "sh" / "shrc").is_file()
        assert (shell_dir / WRAPPER_NAME).is_file()

    def test_uses_agm_real_shell_when_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        monkeypatch.setenv("AGM_REAL_SHELL", "/bin/zsh")
        shell_dir = workspace_shell_dir("s")
        shell_dir.mkdir(parents=True)
        regenerate_workspace_shell(shell_dir)
        text = (shell_dir / WRAPPER_NAME).read_text(encoding="utf-8")
        assert "AGM_REAL_SHELL=/bin/zsh" in text

    def test_errors_when_dir_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        shell_dir = workspace_shell_dir("missing")
        with pytest.raises(SystemExit):
            regenerate_workspace_shell(shell_dir)


class TestRemoveWorkspaceShell:
    def test_removes_existing_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        ensure_workspace_shell("s", env={"SHELL": "/bin/bash"})
        shell_dir = workspace_shell_dir("s")
        assert shell_dir.exists()
        remove_workspace_shell("s")
        assert not shell_dir.exists()

    def test_missing_dir_is_noop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        # Should not raise when nothing exists.
        remove_workspace_shell("never-opened")


class TestShellRegenCommand:
    def test_run_invokes_regenerate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        shell_dir = workspace_shell_dir("s")
        shell_dir.mkdir(parents=True)
        from agm.commands.workspace import shell_regen as cmd

        cmd.run(str(shell_dir))
        assert (shell_dir / WRAPPER_NAME).is_file()


class TestSelfHealE2E:
    def test_wrapper_recreates_rc_files_after_partial_deletion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("bash is required")

        cache = tmp_path / "cache"
        home = tmp_path / "home"
        bin_dir = tmp_path / "bin"
        monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
        home.mkdir()
        bin_dir.mkdir()
        (home / ".bashrc").write_text('export USERRC_RAN=1\n', encoding="utf-8")
        agm = bin_dir / "agm"
        agm.write_text(
            "\n".join(
                [
                    "#!/bin/sh",
                    'if [ "$1" = config ] && [ "$2" = env ]; then',
                    '  printf "export HOLDIR=%s/hold\\n" "$PWD"',
                    "  exit 0",
                    "fi",
                    "exit 64",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        agm.chmod(0o755)

        shell_dir = workspace_shell_dir("s")
        wrapper = ensure_workspace_shell("s", env={"SHELL": bash})

        # Delete only the rc subdirectories, leaving the wrapper file intact.
        shutil.rmtree(shell_dir / "zsh")
        shutil.rmtree(shell_dir / "bash")
        shutil.rmtree(shell_dir / "sh")
        assert not (shell_dir / "bash" / "bashrc").exists()

        result = subprocess.run(
            [str(wrapper)],
            input='printf "healed:%s\\n" "$HOLDIR"\\nexit\\n',
            cwd=tmp_path,
            env={
                **os.environ,
                "HOME": str(home),
                "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                "SHELL": bash,
            },
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        assert result.returncode == 0
        assert f"healed:{tmp_path}/hold" in result.stdout
        # The wrapper regenerated the rc files in place.
        assert (shell_dir / "bash" / "bashrc").exists()
        assert (shell_dir / "zsh" / ".zshrc").exists()
        assert (shell_dir / "sh" / "shrc").exists()
