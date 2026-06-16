"""Tests for ExecConfig and load_exec_config (M0)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.config.general import ExecConfig, load_exec_config


class TestExecConfig:
    def test_default_values(self) -> None:
        cfg = ExecConfig(
            runner=None,
            strict_json=False,
            default_loop_limit=5,
            timeout=None,
            agents={},
            log=False,
            log_file=None,
        )
        assert cfg.runner is None
        assert cfg.strict_json is False
        assert cfg.default_loop_limit == 5
        assert cfg.timeout is None
        assert cfg.agents == {}
        assert cfg.log is False
        assert cfg.log_file is None

    def test_frozen(self) -> None:
        cfg = ExecConfig(
            runner=None,
            strict_json=False,
            default_loop_limit=5,
            timeout=None,
            agents={},
            log=False,
            log_file=None,
        )
        with pytest.raises((AttributeError, TypeError)):
            cfg.runner = "something"


class TestLoadExecConfig:
    def test_load_defaults_when_no_config(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        cfg = load_exec_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.runner is None
        assert cfg.strict_json is False
        assert cfg.default_loop_limit == 5
        assert cfg.timeout is None
        assert cfg.agents == {}
        assert cfg.log is False
        assert cfg.log_file is None

    def test_load_exec_config_from_toml(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / ".agm").mkdir()
        (home / ".agm" / "config.toml").write_text(
            "\n".join(
                [
                    "[exec]",
                    'runner = "claude -p"',
                    "strict_json = true",
                    "default_loop_limit = 10",
                    'timeout = "30m"',
                    "",
                    "[exec.agents]",
                    'reviewer = "claude -p"',
                    'impl = "codex exec"',
                ]
            )
        )

        cfg = load_exec_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.runner == "claude -p"
        assert cfg.strict_json is True
        assert cfg.default_loop_limit == 10
        assert cfg.timeout == pytest.approx(1800.0)
        assert cfg.agents == {"reviewer": "claude -p", "impl": "codex exec"}

    def test_project_config_overrides_home(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / ".agm").mkdir()
        (home / ".agm" / "config.toml").write_text(
            "\n".join(
                [
                    "[exec]",
                    'runner = "home-runner"',
                    "default_loop_limit = 3",
                ]
            )
        )

        proj_dir = tmp_path / "proj"
        (proj_dir / "config").mkdir(parents=True)
        (proj_dir / "config" / "config.toml").write_text(
            "\n".join(
                [
                    "[exec]",
                    'runner = "proj-runner"',
                ]
            )
        )

        cfg = load_exec_config(home=home, proj_dir=proj_dir, cwd=tmp_path)
        assert cfg.runner == "proj-runner"
        # default_loop_limit comes from home config since project doesn't override
        assert cfg.default_loop_limit == 3

    def test_command_name_selects_sub_table(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / ".agm").mkdir()
        (home / ".agm" / "config.toml").write_text(
            "\n".join(
                [
                    "[exec]",
                    'runner = "default-runner"',
                    "",
                    "[exec.myflow]",
                    'runner = "flow-runner"',
                ]
            )
        )

        cfg = load_exec_config(
            home=home, proj_dir=None, cwd=tmp_path, command_name="myflow"
        )
        assert cfg.runner == "flow-runner"

    def test_command_name_none_uses_base_table(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / ".agm").mkdir()
        (home / ".agm" / "config.toml").write_text(
            "\n".join(
                [
                    "[exec]",
                    'runner = "default-runner"',
                    "",
                    "[exec.myflow]",
                    'runner = "flow-runner"',
                ]
            )
        )

        cfg = load_exec_config(home=home, proj_dir=None, cwd=tmp_path, command_name=None)
        assert cfg.runner == "default-runner"

    def test_numeric_timeout(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / ".agm").mkdir()
        (home / ".agm" / "config.toml").write_text(
            "\n".join(
                [
                    "[exec]",
                    "timeout = 60",
                ]
            )
        )
        cfg = load_exec_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.timeout == pytest.approx(60.0)

    def test_agents_command_name_does_not_merge_agents_as_scalar(self, tmp_path: Path) -> None:
        """``command_name="agents"`` must not treat ``[exec.agents]`` as a per-command override.

        The reserved ``[exec.agents]`` map must stay intact and must not be merged
        into the base table as scalar config (which would, e.g., clobber ``runner``).
        """
        home = tmp_path / "home"
        home.mkdir()
        (home / ".agm").mkdir()
        (home / ".agm" / "config.toml").write_text(
            "\n".join(
                [
                    "[exec]",
                    'runner = "default-runner"',
                    "",
                    "[exec.agents]",
                    'reviewer = "claude -p"',
                    'impl = "codex exec"',
                ]
            )
        )

        cfg = load_exec_config(
            home=home, proj_dir=None, cwd=tmp_path, command_name="agents"
        )
        # The base [exec] scalars are unchanged.
        assert cfg.runner == "default-runner"
        # The agents map is preserved intact, not merged in as scalar config.
        assert cfg.agents == {"reviewer": "claude -p", "impl": "codex exec"}

    def test_empty_agent_value_skipped(self, tmp_path: Path) -> None:
        """Agent entries with empty/blank values are ignored."""
        home = tmp_path / "home"
        home.mkdir()
        (home / ".agm").mkdir()
        (home / ".agm" / "config.toml").write_text(
            "\n".join(
                [
                    "[exec.agents]",
                    'good = "claude -p"',
                    'bad = ""',
                ]
            )
        )
        cfg = load_exec_config(home=home, proj_dir=None, cwd=tmp_path)
        assert "good" in cfg.agents
        assert "bad" not in cfg.agents

    def test_log_true_loaded_from_config(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / ".agm").mkdir()
        (home / ".agm" / "config.toml").write_text("[exec]\nlog = true\n")
        cfg = load_exec_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.log is True

    def test_log_false_by_default(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        cfg = load_exec_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.log is False

    def test_log_file_loaded_from_config(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / ".agm").mkdir()
        log_path = tmp_path / "trace.jsonl"
        (home / ".agm" / "config.toml").write_text(
            f"[exec]\nlog_file = {str(log_path)!r}\n"
        )
        cfg = load_exec_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.log_file == str(log_path)

    def test_log_file_none_by_default(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        cfg = load_exec_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.log_file is None
