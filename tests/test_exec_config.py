"""Tests for exec configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.config.general import (
    ExecConfig,
    exec_config_from_merged,
    load_exec_config,
    load_merged_config,
)


class TestExecConfig:
    def test_explicit_values(self) -> None:
        cfg = ExecConfig(
            strict_json=False,
            default_loop_limit=5,
            timeout=None,
            log=False,
            log_file=None,
            runner="claude",
        )
        assert cfg.strict_json is False
        assert cfg.default_loop_limit == 5
        assert cfg.timeout is None
        assert cfg.log is False
        assert cfg.log_file is None
        assert cfg.runner == "claude"

    def test_runner_defaults_to_none(self) -> None:
        cfg = ExecConfig(
            strict_json=False,
            default_loop_limit=5,
            timeout=None,
            log=False,
            log_file=None,
        )
        assert cfg.runner is None

    def test_frozen(self) -> None:
        cfg = ExecConfig(
            strict_json=False,
            default_loop_limit=5,
            timeout=None,
            log=False,
            log_file=None,
        )
        with pytest.raises((AttributeError, TypeError)):
            cfg.strict_json = True


class TestLoadExecConfig:
    def test_load_defaults_when_no_config(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        cfg = load_exec_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.strict_json is False
        assert cfg.default_loop_limit is None
        assert cfg.timeout is None
        assert cfg.log is False
        assert cfg.log_file is None
        assert cfg.runner is None

    def test_runner_loaded_from_toml(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        config = home / ".agm" / "config.toml"
        config.parent.mkdir(parents=True)
        config.write_text('[exec]\nrunner = "claude"\n')
        cfg = load_exec_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.runner == "claude"

    def test_blank_runner_is_none(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        config = home / ".agm" / "config.toml"
        config.parent.mkdir(parents=True)
        config.write_text('[exec]\nrunner = ""\n')
        cfg = load_exec_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.runner is None

    def test_runner_is_not_a_program_table_override(self, tmp_path: Path) -> None:
        """Unlike an engine key, ``runner`` never comes from a qualified program table."""
        home = tmp_path / "home"
        config = home / ".agm" / "config.toml"
        config.parent.mkdir(parents=True)
        config.write_text('[exec]\nrunner = "claude"\n')
        merged = load_merged_config(home=home, proj_dir=None, cwd=tmp_path)
        cfg = exec_config_from_merged(merged, program_table={"runner": "codex"})
        assert cfg.runner == "claude"

    def test_load_exec_config_from_toml(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        config = home / ".agm" / "config.toml"
        config.parent.mkdir(parents=True)
        config.write_text('[exec]\nstrict-json = true\nmax-iters = 10\ntimeout = "30m"\n')
        cfg = load_exec_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.strict_json is True
        assert cfg.default_loop_limit == 10
        assert cfg.timeout == pytest.approx(1800.0)

    def test_project_config_overrides_home(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text("[exec]\nmax-iters = 3\n")
        proj_dir = tmp_path / "proj"
        (proj_dir / "config").mkdir(parents=True)
        (proj_dir / "config" / "config.toml").write_text("[exec]\nmax-iters = 7\n")
        cfg = load_exec_config(home=home, proj_dir=proj_dir, cwd=tmp_path)
        assert cfg.default_loop_limit == 7

    def test_command_name_selects_sub_table(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        config = home / ".agm" / "config.toml"
        config.parent.mkdir(parents=True)
        config.write_text("[exec]\nmax-iters = 3\n\n[exec.myflow]\nmax-iters = 7\n")
        cfg = load_exec_config(home=home, proj_dir=None, cwd=tmp_path, command_name="myflow")
        assert cfg.default_loop_limit == 7

    def test_command_name_none_uses_base_table(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        config = home / ".agm" / "config.toml"
        config.parent.mkdir(parents=True)
        config.write_text("[exec]\nmax-iters = 3\n\n[exec.myflow]\nmax-iters = 7\n")
        cfg = load_exec_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.default_loop_limit == 3

    def test_numeric_timeout(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        config = home / ".agm" / "config.toml"
        config.parent.mkdir(parents=True)
        config.write_text("[exec]\ntimeout = 60\n")
        cfg = load_exec_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.timeout == pytest.approx(60.0)

    def test_log_settings_loaded_from_config(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        config = home / ".agm" / "config.toml"
        config.parent.mkdir(parents=True)
        log_path = tmp_path / "trace.jsonl"
        config.write_text(f"[exec]\nlog = true\nlog-file = {str(log_path)!r}\n")
        cfg = load_exec_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.log is True
        assert cfg.log_file == str(log_path)

    def test_escaped_log_file_interpolation_is_not_reapplied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NAME", "expanded")
        home = tmp_path / "home"
        config_dir = home / ".agm"
        literal_dir = config_dir / "%{NAME}"
        literal_dir.mkdir(parents=True)
        (literal_dir / "base.jsonl").touch()
        (literal_dir / "nested.jsonl").touch()
        (config_dir / "config.toml").write_text(
            "\n".join(
                [
                    "[exec]",
                    'log-file = "\\\\%{NAME}/base.jsonl"',
                    "",
                    "[exec.myflow]",
                    'log-file = "\\\\%{NAME}/nested.jsonl"',
                ]
            )
        )

        base = load_exec_config(home=home, proj_dir=None, cwd=tmp_path)
        nested = load_exec_config(home=home, proj_dir=None, cwd=tmp_path, command_name="myflow")

        assert base.log_file == str(literal_dir / "base.jsonl")
        assert nested.log_file == str(literal_dir / "nested.jsonl")


class TestExecConfigProgramTableOverride:
    def test_program_table_overrides_exec_engine_keys(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        config = home / ".agm" / "config.toml"
        config.parent.mkdir(parents=True)
        config.write_text("[exec]\nmax-iters = 5\n")
        merged = load_merged_config(home=home, proj_dir=None, cwd=tmp_path)
        cfg = exec_config_from_merged(merged, program_table={"max-iters": 10})
        assert cfg.default_loop_limit == 10

    def test_program_table_partial_override(self, tmp_path: Path) -> None:
        merged = {"exec": {"max-iters": 7, "strict-json": False}}
        cfg = exec_config_from_merged(merged, program_table={"strict-json": True})
        assert cfg.default_loop_limit == 7
        assert cfg.strict_json is True
