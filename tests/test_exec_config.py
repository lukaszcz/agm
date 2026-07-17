"""Tests for ExecConfig and load_exec_config."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.config.general import (
    ExecConfig,
    exec_config_from_merged,
    load_exec_config,
    load_merged_config,
    load_program_config,
    program_config_from_merged,
)


class TestExecConfig:
    def test_explicit_values(self) -> None:
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
        # No [exec] max-iters → the valve is OFF (None), not a baked-in default.
        assert cfg.default_loop_limit is None
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
                    "strict-json = true",
                    "max-iters = 10",
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
                    "max-iters = 3",
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
            f"[exec]\nlog-file = {str(log_path)!r}\n"
        )
        cfg = load_exec_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.log_file == str(log_path)

    def test_log_file_none_by_default(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        cfg = load_exec_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.log_file is None


class TestProgramConfig:
    def test_load_program_config_from_toml(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / ".agm").mkdir()
        (home / ".agm" / "config.toml").write_text(
            "\n".join(["[demo]", 'topic = "docs"', "count = 3"])
        )

        cfg = load_program_config(
            "demo",
            home=home,
            proj_dir=None,
            cwd=tmp_path,
        )
        assert cfg == {"topic": "docs", "count": 3}

    def test_program_config_from_merged_non_table_is_empty(self) -> None:
        assert program_config_from_merged({"demo": "not-a-table"}, "demo") == {}

    def test_program_config_from_merged_absent_is_empty(self) -> None:
        assert program_config_from_merged({}, "demo") == {}

    def test_program_config_from_merged_returns_all_keys(self) -> None:
        merged = {"demo": {"topic": "docs", "timeout": "60s", "count": 3}}
        cfg = program_config_from_merged(merged, "demo")
        assert cfg == {"topic": "docs", "timeout": "60s", "count": 3}


class TestExecConfigProgramTableOverride:
    def test_program_table_overrides_exec_engine_keys(self, tmp_path: Path) -> None:
        """[<program>] engine keys override the global [exec] values."""
        home = tmp_path / "home"
        home.mkdir()
        (home / ".agm").mkdir()
        (home / ".agm" / "config.toml").write_text(
            "\n".join(
                [
                    "[exec]",
                    'runner = "global-runner"',
                    "max-iters = 5",
                    "",
                    "[myprog]",
                    'runner = "prog-runner"',
                    "max-iters = 10",
                ]
            )
        )
        merged = load_merged_config(home=home, proj_dir=None, cwd=tmp_path)
        prog_table = program_config_from_merged(merged, "myprog")
        cfg = exec_config_from_merged(merged, program_table=prog_table)
        assert cfg.runner == "prog-runner"
        assert cfg.default_loop_limit == 10

    def test_program_table_partial_override(self, tmp_path: Path) -> None:
        """[<program>] overrides only the keys it specifies; others fall back to [exec]."""
        home = tmp_path / "home"
        home.mkdir()
        (home / ".agm").mkdir()
        (home / ".agm" / "config.toml").write_text(
            "\n".join(
                [
                    "[exec]",
                    'runner = "global-runner"',
                    "max-iters = 7",
                    "",
                    "[myprog]",
                    'runner = "prog-runner"',
                ]
            )
        )
        merged = load_merged_config(home=home, proj_dir=None, cwd=tmp_path)
        prog_table = program_config_from_merged(merged, "myprog")
        cfg = exec_config_from_merged(merged, program_table=prog_table)
        assert cfg.runner == "prog-runner"
        assert cfg.default_loop_limit == 7  # from [exec], not overridden

    def test_no_program_table_uses_exec_defaults(self, tmp_path: Path) -> None:
        """Without a program table, exec_config_from_merged uses [exec] alone."""
        home = tmp_path / "home"
        home.mkdir()
        (home / ".agm").mkdir()
        (home / ".agm" / "config.toml").write_text(
            "\n".join(["[exec]", 'runner = "exec-runner"'])
        )
        merged = load_merged_config(home=home, proj_dir=None, cwd=tmp_path)
        cfg = exec_config_from_merged(merged)
        assert cfg.runner == "exec-runner"
