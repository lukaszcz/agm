"""Tests for general config and sandbox utility helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.config.general import load_loop_config, load_run_config
from agm.config.sandbox import sandbox_settings_candidates


def test_load_run_config_merges_global_and_local_sections(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / ".agm").mkdir()
    (home / ".agm" / "config.toml").write_text(
        "\n".join(
            [
                "[run]",
                'memory = "20G"',
                "",
                "[run.echo]",
                'alias = "printf"',
                'memory = "10G"',
                "",
                "[run.keep]",
                'alias = "cat"',
                "",
            ]
        )
    )

    project = tmp_path / "project"
    (project / "config").mkdir(parents=True)
    (project / "config" / "config.toml").write_text(
        "\n".join(
            [
                "[run.echo]",
                'alias = "cat"',
                'memory = "5G"',
                "",
                "[run.local]",
                'alias = "sed"',
                "",
            ]
        )
    )

    config = load_run_config(home=home, proj_dir=project, cwd=tmp_path / "work")

    assert config.alias_for("echo") == "cat"
    assert config.alias_for("keep") == "cat"
    assert config.alias_for("local") == "sed"
    assert config.alias_for("missing") is None
    assert config.memory_limit_for("echo") == "5G"
    assert config.memory_limit_for("keep") == "20G"
    assert config.memory_limit_for("local") == "20G"
    assert config.memory_limit_for("missing") == "20G"


def test_load_run_config_prefers_dot_agm_config_after_project_config(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()

    project = tmp_path / "project"
    (project / "config").mkdir(parents=True)
    (project / "config" / "config.toml").write_text('[run.echo]\nalias = "printf"\n')

    work = tmp_path / "work"
    (work / ".agm").mkdir(parents=True)
    (work / ".agm" / "config.toml").write_text('[run.echo]\nalias = "cat"\n')

    config = load_run_config(home=home, proj_dir=project, cwd=work)

    assert config.alias_for("echo") == "cat"


def test_load_run_config_uses_install_prefix_before_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "prefix"
    (prefix / ".agm").mkdir(parents=True)
    (prefix / ".agm" / "config.toml").write_text('[run.echo]\nalias = "printf"\n')

    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    (home / ".agm" / "config.toml").write_text('[run.echo]\nalias = "cat"\n')

    monkeypatch.setattr("agm.config.general.agm_installation_prefix", lambda: prefix)

    config = load_run_config(home=home, proj_dir=None, cwd=tmp_path / "work")

    assert config.alias_for("echo") == "printf"


def test_load_run_config_falls_back_to_home_when_install_prefix_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    (home / ".agm" / "config.toml").write_text('[run.echo]\nalias = "printf"\n')

    monkeypatch.setattr("agm.config.general.agm_installation_prefix", lambda: tmp_path / "prefix")

    config = load_run_config(home=home, proj_dir=None, cwd=tmp_path / "work")

    assert config.alias_for("echo") == "printf"


def test_load_loop_config_reads_tasks_dir(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    (home / ".agm" / "config.toml").write_text(
        '[loop]\ncommand = "claude -p"\ntasks_dir = "custom/tasks"\n'
    )

    config = load_loop_config(home=home, proj_dir=None, cwd=tmp_path / "work")

    assert config.command == "claude -p"
    assert config.tasks_dir == "custom/tasks"


def test_load_run_config_prefers_dot_agm_memory_after_project_config(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()

    project = tmp_path / "project"
    (project / "config").mkdir(parents=True)
    (project / "config" / "config.toml").write_text(
        '[run]\nmemory = "10G"\n[run.echo]\nmemory = "5G"\n'
    )

    work = tmp_path / "work"
    (work / ".agm").mkdir(parents=True)
    (work / ".agm" / "config.toml").write_text('[run.echo]\nmemory = "2G"\n')

    config = load_run_config(home=home, proj_dir=project, cwd=work)

    assert config.memory_limit_for("echo") == "2G"
    assert config.memory_limit_for("other") == "10G"


def test_sandbox_settings_candidates_fall_back_to_alias_command(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / ".agm" / "sandbox").mkdir(parents=True)
    (home / ".agm" / "sandbox" / "printf.json").write_text("{}")

    project = tmp_path / "project"
    (project / "config" / "sandbox").mkdir(parents=True)
    (project / "config" / "sandbox" / "default.json").write_text("{}")

    work = tmp_path / "work"
    (work / ".sandbox").mkdir(parents=True)
    (work / ".sandbox" / "default.json").write_text("{}")

    candidates = sandbox_settings_candidates(
        cwd=work,
        home=home,
        proj_dir=project,
        command_name="echo",
        alias_command_name="printf",
    )

    assert candidates == [
        home / ".agm" / "sandbox" / "printf.json",
        project / "config" / "sandbox" / "default.json",
        work / ".sandbox" / "default.json",
    ]
