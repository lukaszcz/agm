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

    def test_real_shell_prefers_agm_real_shell_when_shell_is_wrapper(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Nested open: inside a workspace shell $SHELL points at our wrapper, but
        # the real shell is preserved in $AGM_REAL_SHELL.  We must use it rather
        # than degrading to /bin/sh.
        shell_dir = self._setup(tmp_path, monkeypatch)
        env = {
            "SHELL": str(shell_dir / WRAPPER_NAME),
            "AGM_REAL_SHELL": "/usr/bin/zsh",
        }
        wrapper = ensure_workspace_shell("s", env=env)
        text = wrapper.read_text(encoding="utf-8")
        assert "AGM_REAL_SHELL=/usr/bin/zsh" in text

    def test_real_shell_prefers_agm_real_shell_over_plain_shell(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._setup(tmp_path, monkeypatch)
        env = {"SHELL": "/bin/bash", "AGM_REAL_SHELL": "/usr/bin/zsh"}
        wrapper = ensure_workspace_shell("s", env=env)
        text = wrapper.read_text(encoding="utf-8")
        assert "AGM_REAL_SHELL=/usr/bin/zsh" in text

    def test_real_shell_ignores_agm_real_shell_pointing_at_wrapper(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A corrupt AGM_REAL_SHELL pointing back at the wrapper must not be used;
        # fall through to $SHELL instead of looping back into ourselves.
        shell_dir = self._setup(tmp_path, monkeypatch)
        env = {
            "SHELL": "/bin/bash",
            "AGM_REAL_SHELL": str(shell_dir / WRAPPER_NAME),
        }
        wrapper = ensure_workspace_shell("s", env=env)
        text = wrapper.read_text(encoding="utf-8")
        assert "AGM_REAL_SHELL=/bin/bash" in text

    def test_wrapper_captures_user_zdotdir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The zsh branch must save the user's real ZDOTDIR before overwriting it,
        # mirroring the sh AGM_USER_ENV capture, so the user's own zsh config dir
        # is replayed (and preserved across nested opens).
        self._setup(tmp_path, monkeypatch)
        wrapper = ensure_workspace_shell("s", env={"SHELL": "/bin/zsh"})
        text = wrapper.read_text(encoding="utf-8")
        assert 'export AGM_USER_ZDOTDIR="${ZDOTDIR:-$HOME}"' in text

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


class TestShPathDoesNotRecurse:
    def test_shrc_does_not_source_itself(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sh rc must not source ``$ENV`` (which points back at itself).

        Regression test: the wrapper exports ``ENV`` pointing at the agm ``shrc``
        before exec'ing ``sh -i``.  If ``shrc`` then sources ``$ENV`` it sources
        itself recursively, exhausting file descriptors ("Too many open files").
        """

        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        shell_dir = workspace_shell_dir("s")
        ensure_workspace_shell("s", env={"SHELL": "/bin/sh"})
        shrc = (shell_dir / "sh" / "shrc").read_text(encoding="utf-8")
        assert '. "$ENV"' not in shrc

    def test_sh_wrapper_terminates_without_fd_exhaustion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sh = shutil.which("sh")
        if sh is None:
            pytest.skip("sh is required")

        cache = tmp_path / "cache"
        home = tmp_path / "home"
        bin_dir = tmp_path / "bin"
        monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
        home.mkdir()
        bin_dir.mkdir()
        # The user's original $ENV startup file — must be sourced exactly once.
        user_env = home / ".userenv"
        user_env.write_text('export USERENV_RAN=1\n', encoding="utf-8")
        agm = bin_dir / "agm"
        agm.write_text(
            "\n".join(
                [
                    "#!/bin/sh",
                    'if [ "$1" = config ] && [ "$2" = env ]; then',
                    "  exit 0",
                    "fi",
                    "exit 64",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        agm.chmod(0o755)

        wrapper = ensure_workspace_shell("s", env={"SHELL": sh})

        result = subprocess.run(
            [str(wrapper)],
            input='printf "ran:%s\\n" "${USERENV_RAN:-0}"\nexit\n',
            cwd=tmp_path,
            env={
                **os.environ,
                "HOME": str(home),
                "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                "SHELL": sh,
                "ENV": str(user_env),
            },
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        assert "Too many open files" not in result.stderr
        assert result.returncode == 0
        # The user's original $ENV file was replayed exactly once.
        assert "ran:1" in result.stdout


def _fake_agm(bin_dir: Path) -> None:
    """Write a fake ``agm`` whose ``config env`` is a no-op."""

    agm = bin_dir / "agm"
    agm.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                'if [ "$1" = config ] && [ "$2" = env ]; then',
                "  exit 0",
                "fi",
                "exit 64",
                "",
            ]
        ),
        encoding="utf-8",
    )
    agm.chmod(0o755)


class TestNestedOpen:
    """A workspace shell opened from within another must load normal config.

    Inside a workspace shell ``$SHELL`` points at our wrapper and the real shell
    is preserved as ``$AGM_REAL_SHELL``.  A second ``agm open`` must resolve the
    real shell from ``$AGM_REAL_SHELL`` (not degrade to ``/bin/sh``) so the
    user's ``~/.bashrc``/``~/.zshrc`` is sourced as usual.
    """

    def test_inner_wrapper_resolves_real_shell_from_agm_real_shell(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("bash is required")

        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        outer = ensure_workspace_shell("outer", env={"SHELL": bash})
        # Emulate the environment a workspace bash shell exports: SHELL rewritten
        # to the wrapper, the real shell preserved in AGM_REAL_SHELL.
        inner_env = {"SHELL": str(outer), "AGM_REAL_SHELL": bash}
        inner = ensure_workspace_shell("inner", env=inner_env)
        text = inner.read_text(encoding="utf-8")
        assert f"AGM_REAL_SHELL={bash}" in text

    def test_nested_open_sources_user_bashrc(
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
        _fake_agm(bin_dir)

        outer = ensure_workspace_shell("outer", env={"SHELL": bash})
        # The inner open runs with the workspace shell's leaked environment.
        inner_env = {"SHELL": str(outer), "AGM_REAL_SHELL": bash}
        inner = ensure_workspace_shell("inner", env=inner_env)

        result = subprocess.run(
            [str(inner)],
            input='printf "ran:%s\\n" "${USERRC_RAN:-0}"\nexit\n',
            cwd=tmp_path,
            env={
                **os.environ,
                "HOME": str(home),
                "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                "SHELL": str(outer),
                "AGM_REAL_SHELL": bash,
            },
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        assert result.returncode == 0
        # The nested shell is a real bash that sourced ~/.bashrc.
        assert "ran:1" in result.stdout

    def test_nested_open_sources_user_env_for_sh(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sh = shutil.which("sh")
        if sh is None:
            pytest.skip("sh is required")

        cache = tmp_path / "cache"
        home = tmp_path / "home"
        bin_dir = tmp_path / "bin"
        monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
        home.mkdir()
        bin_dir.mkdir()
        # The user's original $ENV startup file — the sh "rc".
        user_env = home / ".userenv"
        user_env.write_text('export USERENV_RAN=1\n', encoding="utf-8")
        _fake_agm(bin_dir)

        outer = ensure_workspace_shell("outer", env={"SHELL": sh})
        outer_shrc = workspace_shell_dir("outer") / "sh" / "shrc"
        # Emulate the environment a workspace sh shell exports after the outer
        # open: SHELL -> wrapper, real shell preserved, ENV pointing at the outer
        # shrc, and the user's original $ENV saved as AGM_USER_ENV.
        inner_env = {
            "SHELL": str(outer),
            "AGM_REAL_SHELL": sh,
            "ENV": str(outer_shrc),
            "AGM_USER_ENV": str(user_env),
        }
        inner = ensure_workspace_shell("inner", env=inner_env)

        result = subprocess.run(
            [str(inner)],
            input='printf "ran:%s\\n" "${USERENV_RAN:-0}"\nexit\n',
            cwd=tmp_path,
            env={
                **os.environ,
                "HOME": str(home),
                "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                "SHELL": str(outer),
                "AGM_REAL_SHELL": sh,
                "ENV": str(outer_shrc),
                "AGM_USER_ENV": str(user_env),
            },
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        assert "Too many open files" not in result.stderr
        assert result.returncode == 0
        # The nested sh still replays the user's original $ENV exactly.
        assert "ran:1" in result.stdout


class TestZshCustomZdotdir:
    """Wrapper correctly sources user shell config across shells.

    bash and sh cases run unconditionally (these shells are always available).
    The zsh case covers the ZDOTDIR save/replay branch and is skipped only when
    zsh is genuinely absent.
    """

    @pytest.mark.parametrize(
        ("shell_name", "expected"),
        [
            ("bash", "ran:1"),
            ("sh", "ran:1"),
            pytest.param(
                "zsh",
                "custom:1 home:0",
                marks=pytest.mark.skipif(
                    shutil.which("zsh") is None, reason="zsh not available"
                ),
            ),
        ],
    )
    def test_wrapper_sources_user_rc(
        self,
        shell_name: str,
        expected: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Wrapper sources the correct user rc for each shell.

        bash: sources ``~/.bashrc``; sh: sources the user ``$ENV`` file;
        zsh: sources the custom ZDOTDIR's ``.zshrc``, NOT ``$HOME/.zshrc``.
        """
        shell = shutil.which(shell_name)
        assert shell is not None, f"{shell_name} must be available on this machine"

        home = tmp_path / "home"
        bin_dir = tmp_path / "bin"
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        home.mkdir()
        bin_dir.mkdir()
        _fake_agm(bin_dir)

        extra_run_env: dict[str, str] = {}

        if shell_name == "bash":
            (home / ".bashrc").write_text('export CUSTOMRC_RAN=1\n', encoding="utf-8")
            input_cmd = 'printf "ran:%s\\n" "${CUSTOMRC_RAN:-0}"\nexit\n'
        elif shell_name == "sh":
            user_env_file = home / ".userenv"
            user_env_file.write_text('export CUSTOMRC_RAN=1\n', encoding="utf-8")
            extra_run_env["ENV"] = str(user_env_file)
            input_cmd = 'printf "ran:%s\\n" "${CUSTOMRC_RAN:-0}"\nexit\n'
        else:  # zsh — ZDOTDIR save/replay branch
            zdot = home / "zdot"
            zdot.mkdir()
            # Custom ZDOTDIR's rc; $HOME/.zshrc must NOT be sourced.
            (zdot / ".zshrc").write_text('export CUSTOMRC_RAN=1\n', encoding="utf-8")
            (home / ".zshrc").write_text('export HOMERC_RAN=1\n', encoding="utf-8")
            extra_run_env["ZDOTDIR"] = str(zdot)
            input_cmd = (
                'printf "custom:%s home:%s\\n" "${CUSTOMRC_RAN:-0}" "${HOMERC_RAN:-0}"\nexit\n'
            )

        wrapper = ensure_workspace_shell("s", env={"SHELL": shell})

        result = subprocess.run(
            [str(wrapper)],
            input=input_cmd,
            cwd=tmp_path,
            env={
                **os.environ,
                "HOME": str(home),
                "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                "SHELL": shell,
                **extra_run_env,
            },
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        assert result.returncode == 0
        assert expected in result.stdout


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
