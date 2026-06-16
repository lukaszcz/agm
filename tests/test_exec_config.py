"""Tests for ExecConfig and load_exec_config (M0); load_params_config (M5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.config.general import ExecConfig, load_exec_config, load_params_config


class TestExecConfig:
    def test_default_values(self) -> None:
        cfg = ExecConfig(
            runner=None,
            strict_json=False,
            default_loop_limit=5,
            timeout=None,
            agents={},
        )
        assert cfg.runner is None
        assert cfg.strict_json is False
        assert cfg.default_loop_limit == 5
        assert cfg.timeout is None
        assert cfg.agents == {}

    def test_frozen(self) -> None:
        cfg = ExecConfig(
            runner=None,
            strict_json=False,
            default_loop_limit=5,
            timeout=None,
            agents={},
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


# ---------------------------------------------------------------------------
# load_params_config (M5)
# ---------------------------------------------------------------------------


class TestLoadParamsConfig:
    """Tests for ``load_params_config`` — loading ``[params.<program>]`` tables."""

    def _home(self, tmp_path: Path, *, toml: str = "") -> Path:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        if toml:
            (home / ".agm" / "config.toml").write_text(toml)
        return home

    def test_missing_params_section_returns_empty_dict(self, tmp_path: Path) -> None:
        home = self._home(tmp_path, toml="[exec]\nrunner = 'claude'\n")
        result = load_params_config("my_prog", home=home, proj_dir=None, cwd=tmp_path)
        assert result == {}

    def test_missing_program_table_returns_empty_dict(self, tmp_path: Path) -> None:
        home = self._home(tmp_path, toml="[params.other_prog]\nfoo = 'bar'\n")
        result = load_params_config("my_prog", home=home, proj_dir=None, cwd=tmp_path)
        assert result == {}

    def test_program_table_returned_as_toml_native_values(self, tmp_path: Path) -> None:
        home = self._home(
            tmp_path,
            toml=(
                "[params.my_prog]\n"
                'greeting = "hello"\n'
                "count = 3\n"
                "enabled = true\n"
                'tags = ["a", "b"]\n'
            ),
        )
        result = load_params_config("my_prog", home=home, proj_dir=None, cwd=tmp_path)
        assert result == {
            "greeting": "hello",
            "count": 3,
            "enabled": True,
            "tags": ["a", "b"],
        }

    def test_bool_native_not_stringified(self, tmp_path: Path) -> None:
        """bool values must stay native bool, not str(True)='True'."""
        home = self._home(tmp_path, toml="[params.prog]\nflag = true\n")
        result = load_params_config("prog", home=home, proj_dir=None, cwd=tmp_path)
        assert result["flag"] is True
        assert isinstance(result["flag"], bool)

    def test_decimal_as_string_preserved(self, tmp_path: Path) -> None:
        """Decimal param values must be quoted strings in config (TOML native float rejected)."""
        home = self._home(tmp_path, toml='[params.prog]\nratio = "3.14"\n')
        result = load_params_config("prog", home=home, proj_dir=None, cwd=tmp_path)
        assert result["ratio"] == "3.14"
        assert isinstance(result["ratio"], str)

    def test_no_config_file_returns_empty_dict(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        result = load_params_config("prog", home=home, proj_dir=None, cwd=tmp_path)
        assert result == {}

    def test_project_config_overrides_home(self, tmp_path: Path) -> None:
        """Later layers (project config) override earlier layers (home config)."""
        home = self._home(tmp_path, toml="[params.prog]\nname = 'from_home'\nextra = 'kept'\n")
        proj_dir = tmp_path / "proj"
        (proj_dir / "config").mkdir(parents=True)
        (proj_dir / "config" / "config.toml").write_text(
            "[params.prog]\nname = 'from_proj'\n"
        )
        result = load_params_config("prog", home=home, proj_dir=proj_dir, cwd=tmp_path)
        assert result["name"] == "from_proj"
        # Key only in home config is preserved (deep merge).
        assert result["extra"] == "kept"

    def test_cwd_config_overrides_project(self, tmp_path: Path) -> None:
        """cwd/.agm/config.toml is the highest-precedence layer."""
        home = self._home(tmp_path, toml="[params.prog]\nname = 'home'\n")
        cwd_agm = tmp_path / ".agm"
        cwd_agm.mkdir()
        (cwd_agm / "config.toml").write_text("[params.prog]\nname = 'cwd'\n")
        result = load_params_config("prog", home=home, proj_dir=None, cwd=tmp_path)
        assert result["name"] == "cwd"

    def test_different_program_names_do_not_interfere(self, tmp_path: Path) -> None:
        home = self._home(
            tmp_path,
            toml="[params.prog_a]\nfoo = 'a'\n\n[params.prog_b]\nfoo = 'b'\n",
        )
        assert load_params_config("prog_a", home=home, proj_dir=None, cwd=tmp_path) == {
            "foo": "a"
        }
        assert load_params_config("prog_b", home=home, proj_dir=None, cwd=tmp_path) == {
            "foo": "b"
        }

    def test_int_value_native(self, tmp_path: Path) -> None:
        home = self._home(tmp_path, toml="[params.prog]\ncount = 42\n")
        result = load_params_config("prog", home=home, proj_dir=None, cwd=tmp_path)
        assert result["count"] == 42
        assert isinstance(result["count"], int)

    def test_list_value_native(self, tmp_path: Path) -> None:
        home = self._home(tmp_path, toml='[params.prog]\ntags = ["x", "y"]\n')
        result = load_params_config("prog", home=home, proj_dir=None, cwd=tmp_path)
        assert result["tags"] == ["x", "y"]
