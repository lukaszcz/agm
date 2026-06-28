"""Tests for ExecConfig and load_exec_config (M0)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.config.general import (
    ExecConfig,
    load_exec_config,
    load_params_config,
    params_config_from_merged,
)


class TestExecConfig:
    def test_default_values(self) -> None:
        cfg = ExecConfig(
            runner=None,
            strict_json=False,
            timeout=None,
            agents={},
            log=False,
            log_file=None,
        )
        assert cfg.runner is None
        assert cfg.strict_json is False
        assert cfg.timeout is None
        assert cfg.agents == {}
        assert cfg.log is False
        assert cfg.log_file is None

    def test_frozen(self) -> None:
        cfg = ExecConfig(
            runner=None,
            strict_json=False,
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
                    "strict_json = true",
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
        # strict_json comes from home config since project doesn't override it
        assert cfg.strict_json is True

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

    def test_max_call_depth_none_by_default(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        cfg = load_exec_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.max_call_depth is None

    def test_max_call_depth_loaded_from_config(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / ".agm").mkdir()
        (home / ".agm" / "config.toml").write_text("[exec]\nmax_call_depth = 128\n")
        cfg = load_exec_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.max_call_depth == 128

    def test_max_call_depth_non_positive_ignored(self, tmp_path: Path) -> None:
        """A non-positive config value falls back to None (canonical default applies)."""
        home = tmp_path / "home"
        home.mkdir()
        (home / ".agm").mkdir()
        (home / ".agm" / "config.toml").write_text("[exec]\nmax_call_depth = 0\n")
        cfg = load_exec_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.max_call_depth is None


class TestParamsConfig:
    def test_load_params_config_from_toml(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / ".agm").mkdir()
        (home / ".agm" / "config.toml").write_text(
            "\n".join(["[params.demo]", 'topic = "docs"', "count = 3"])
        )

        cfg = load_params_config(
            home=home,
            proj_dir=None,
            cwd=tmp_path,
            program_name="demo",
        )
        assert cfg == {"topic": "docs", "count": 3}

    def test_params_config_from_merged_non_table_is_empty(self) -> None:
        assert params_config_from_merged({"params": {"demo": "not-a-table"}}, "demo") == {}
