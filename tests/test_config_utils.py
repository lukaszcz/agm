"""Tests for general config and sandbox utility helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.config.general import (
    ConfigCommandNotFound,
    _optional_bool,
    _unique_paths,
    load_loop_config,
    load_refine_config,
    load_review_config,
    load_revise_config,
    load_run_config,
)
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
                'swap = "1G"',
                "",
                "[run.echo]",
                'alias = "printf"',
                'memory = "10G"',
                'swap = "2G"',
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
                'swap = "512M"',
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
    assert config.swap_limit_for("echo") == "512M"
    assert config.swap_limit_for("keep") == "1G"
    assert config.swap_limit_for("local") == "1G"
    assert config.swap_limit_for("missing") == "1G"


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


def test_load_run_config_prefers_home_over_install_prefix(
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

    assert config.alias_for("echo") == "cat"


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
        '[loop]\nrunner = "claude -p"\nselector = "codex exec"\ntasks_dir = "custom/tasks"\n'
    )

    cwd = tmp_path / "work"
    cwd.mkdir()
    (cwd / "custom" / "tasks").mkdir(parents=True)

    config = load_loop_config(home=home, proj_dir=None, cwd=cwd)

    assert config.runner == "claude -p"
    assert config.selector == "codex exec"
    assert config.tasks_dir == str(cwd / "custom" / "tasks")


def test_load_loop_config_prefers_command_specific_overrides(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    (home / ".agm" / "custom" / "tasks").mkdir(parents=True)
    (home / ".agm" / "config.toml").write_text(
        '[loop]\nrunner = "claude -p"\nselector = "opencode prompt"\ntasks_dir = "custom/tasks"\n'
        '[loop.codex]\nrunner = "codex exec"\nselector = "claude --print"\n'
        'tasks_dir = "codex/tasks"\n'
    )

    cwd = tmp_path / "work"
    cwd.mkdir()
    (cwd / "codex" / "tasks").mkdir(parents=True)

    config = load_loop_config(home=home, proj_dir=None, cwd=cwd, command_name="codex")

    assert config.runner == "codex exec"
    assert config.selector == "claude --print"
    assert config.tasks_dir == str(cwd / "codex" / "tasks")


def test_load_loop_config_rejects_missing_required_named_section(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    (home / ".agm" / "config.toml").write_text('[loop]\nrunner = "claude -p"\n')
    cwd = tmp_path / "work"
    cwd.mkdir()

    with pytest.raises(ConfigCommandNotFound) as exc_info:
        load_loop_config(
            home=home,
            proj_dir=None,
            cwd=cwd,
            command_name="fronend",
            require_command=True,
        )

    assert exc_info.value.section_name == "loop"
    assert exc_info.value.command_name == "fronend"


def test_load_run_config_prefers_dot_agm_memory_after_project_config(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()

    project = tmp_path / "project"
    (project / "config").mkdir(parents=True)
    (project / "config" / "config.toml").write_text(
        '[run]\nmemory = "10G"\nswap = "2G"\n[run.echo]\nmemory = "5G"\nswap = "1G"\n'
    )

    work = tmp_path / "work"
    (work / ".agm").mkdir(parents=True)
    (work / ".agm" / "config.toml").write_text('[run.echo]\nmemory = "2G"\nswap = "256M"\n')

    config = load_run_config(home=home, proj_dir=project, cwd=work)

    assert config.memory_limit_for("echo") == "2G"
    assert config.memory_limit_for("other") == "10G"
    assert config.swap_limit_for("echo") == "256M"
    assert config.swap_limit_for("other") == "2G"


# --- Path resolution and env var expansion in config file paths ---


def test_load_loop_config_resolves_relative_prompt_file_against_config_dir(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    (home / ".agm" / "prompt.md").write_text("home prompt")
    (home / ".agm" / "config.toml").write_text(
        '[loop]\nprompt_file = "prompt.md"\n'
    )

    cwd = tmp_path / "work"
    cwd.mkdir()

    config = load_loop_config(home=home, proj_dir=None, cwd=cwd)

    assert config.prompt_file == str(home / ".agm" / "prompt.md")


def test_load_loop_config_resolves_relative_prompt_file_falls_back_to_cwd(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    # prompt.md does NOT exist in home/.agm/, so cwd fallback applies
    (home / ".agm" / "config.toml").write_text(
        '[loop]\nprompt_file = "prompt.md"\n'
    )

    cwd = tmp_path / "work"
    cwd.mkdir()
    (cwd / "prompt.md").write_text("cwd prompt")

    config = load_loop_config(home=home, proj_dir=None, cwd=cwd)

    assert config.prompt_file == str(cwd / "prompt.md")


def test_load_loop_config_resolves_relative_selector_prompt_file_against_config_dir(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    (home / ".agm" / "select.md").write_text("home selector")
    (home / ".agm" / "config.toml").write_text(
        '[loop]\nselector_prompt_file = "select.md"\n'
    )

    cwd = tmp_path / "work"
    cwd.mkdir()

    config = load_loop_config(home=home, proj_dir=None, cwd=cwd)

    assert config.selector_prompt_file == str(home / ".agm" / "select.md")


def test_load_loop_config_resolves_relative_tasks_dir_against_config_dir(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    (home / ".agm" / "custom" / "tasks").mkdir(parents=True)
    (home / ".agm" / "config.toml").write_text(
        '[loop]\ntasks_dir = "custom/tasks"\n'
    )

    cwd = tmp_path / "work"
    cwd.mkdir()

    config = load_loop_config(home=home, proj_dir=None, cwd=cwd)

    assert config.tasks_dir == str(home / ".agm" / "custom" / "tasks")


def test_load_loop_config_resolves_relative_tasks_dir_falls_back_to_cwd(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    # custom/tasks does NOT exist in home/.agm/, so cwd fallback applies
    (home / ".agm" / "config.toml").write_text(
        '[loop]\ntasks_dir = "custom/tasks"\n'
    )

    cwd = tmp_path / "work"
    cwd.mkdir()
    (cwd / "custom" / "tasks").mkdir(parents=True)

    config = load_loop_config(home=home, proj_dir=None, cwd=cwd)

    assert config.tasks_dir == str(cwd / "custom" / "tasks")


def test_load_loop_config_expands_env_vars_in_prompt_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    (custom_dir / "prompt.md").write_text("expanded prompt")
    monkeypatch.setenv("AGM_TEST_DIR", str(custom_dir))

    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    (home / ".agm" / "config.toml").write_text(
        '[loop]\nprompt_file = "$AGM_TEST_DIR/prompt.md"\n'
    )

    cwd = tmp_path / "work"
    cwd.mkdir()

    config = load_loop_config(home=home, proj_dir=None, cwd=cwd)

    assert config.prompt_file == str(custom_dir / "prompt.md")


def test_load_loop_config_expands_env_vars_in_tasks_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MY_TASKS", "/opt/tasks")

    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    (home / ".agm" / "config.toml").write_text(
        '[loop]\ntasks_dir = "$MY_TASKS"\n'
    )

    cwd = tmp_path / "work"
    cwd.mkdir()

    config = load_loop_config(home=home, proj_dir=None, cwd=cwd)

    assert config.tasks_dir == "/opt/tasks"


def test_load_loop_config_project_config_relative_paths_resolve_against_project_config_dir(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()

    project = tmp_path / "project"
    (project / ".agm" / "config").mkdir(parents=True)
    (project / ".agm" / "config" / "config.toml").write_text(
        '[loop]\nprompt_file = "proj-prompt.md"\n'
    )
    (project / ".agm" / "config" / "proj-prompt.md").write_text("project prompt")

    cwd = tmp_path / "work"
    cwd.mkdir()

    config = load_loop_config(home=home, proj_dir=project, cwd=cwd)

    assert config.prompt_file == str(project / ".agm" / "config" / "proj-prompt.md")


def test_load_loop_config_keeps_absolute_paths_unchanged(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    (home / ".agm" / "config.toml").write_text(
        '[loop]\nprompt_file = "/absolute/path/prompt.md"\n'
    )

    cwd = tmp_path / "work"
    cwd.mkdir()

    config = load_loop_config(home=home, proj_dir=None, cwd=cwd)

    assert config.prompt_file == "/absolute/path/prompt.md"


def test_load_review_revise_and_refine_config_read_new_sections(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    (home / ".agm" / "review.md").write_text("review", encoding="utf-8")
    (home / ".agm" / "revise.md").write_text("revise", encoding="utf-8")
    (home / ".agm" / "config.toml").write_text(
        "\n".join(
            [
                "[review]",
                'runner = "reviewer"',
                'scope = "branch"',
                'aspects = "correctness"',
                'extra_aspects = "tests"',
                'prompt_file = "review.md"',
                'review_file = "saved-review.md"',
                "",
                "[revise]",
                'runner = "reviser"',
                'prompt_file = "revise.md"',
                "",
                "[refine]",
                "max_steps = 7",
                'runner = "both"',
                'reviewer = "review-only"',
                'reviser = "revise-only"',
                "save_review = true",
                'review_prompt_file = "review.md"',
                'revise_prompt_file = "revise.md"',
            ]
        )
    )
    cwd = tmp_path / "work"
    cwd.mkdir()

    review = load_review_config(home=home, proj_dir=None, cwd=cwd)
    revise = load_revise_config(home=home, proj_dir=None, cwd=cwd)
    refine = load_refine_config(home=home, proj_dir=None, cwd=cwd)

    assert review.runner == "reviewer"
    assert review.extra_aspects == "tests"
    assert review.prompt_file == str(home / ".agm" / "review.md")
    assert review.review_file == str(cwd / "saved-review.md")
    assert revise.runner == "reviser"
    assert revise.prompt_file == str(home / ".agm" / "revise.md")
    assert refine.max_steps == 7
    assert refine.runner == "both"
    assert refine.reviewer == "review-only"
    assert refine.reviser == "revise-only"
    assert refine.save_review is True
    assert refine.review_prompt_file == str(home / ".agm" / "review.md")
    assert refine.revise_prompt_file == str(home / ".agm" / "revise.md")


def test_load_review_config_preserves_special_review_file_values(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    (home / ".agm" / "config.toml").write_text(
        '[review]\nreview_file = "none"\n[review.frontend]\nreview_file = "auto"\n',
        encoding="utf-8",
    )
    cwd = tmp_path / "work"
    cwd.mkdir()

    base = load_review_config(home=home, proj_dir=None, cwd=cwd)
    frontend = load_review_config(
        home=home,
        proj_dir=None,
        cwd=cwd,
        command_name="frontend",
    )

    assert base.review_file == "none"
    assert frontend.review_file == "auto"


def test_load_review_config_resolves_prompt_paths_named_like_review_file_sentinels(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    (home / ".agm" / "config.toml").write_text(
        "\n".join(
            [
                "[review]",
                'prompt_file = "auto"',
                'extra_prompt_file = "none"',
            ]
        ),
        encoding="utf-8",
    )
    cwd = tmp_path / "work"
    cwd.mkdir()

    config = load_review_config(home=home, proj_dir=None, cwd=cwd)

    assert config.prompt_file == str(cwd / "auto")
    assert config.extra_prompt_file == str(cwd / "none")


def test_load_review_revise_and_refine_config_read_named_sections(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    (home / ".agm" / "frontend-review.md").write_text("review", encoding="utf-8")
    (home / ".agm" / "frontend-revise.md").write_text("revise", encoding="utf-8")
    (home / ".agm" / "config.toml").write_text(
        "\n".join(
            [
                "[review]",
                'runner = "reviewer"',
                'extra_prompt = "base review"',
                "[review.frontend]",
                'runner = "frontend-reviewer"',
                'prompt_file = "frontend-review.md"',
                "",
                "[revise]",
                'runner = "reviser"',
                'extra_prompt = "base revise"',
                "[revise.frontend]",
                'runner = "frontend-reviser"',
                'prompt_file = "frontend-revise.md"',
                "",
                "[refine]",
                "max_steps = 7",
                'runner = "both"',
                "[refine.frontend]",
                "max_steps = 3",
                'reviewer = "frontend-review-only"',
            ]
        )
    )
    cwd = tmp_path / "work"
    cwd.mkdir()

    review = load_review_config(home=home, proj_dir=None, cwd=cwd, command_name="frontend")
    revise = load_revise_config(home=home, proj_dir=None, cwd=cwd, command_name="frontend")
    refine = load_refine_config(home=home, proj_dir=None, cwd=cwd, command_name="frontend")

    assert review.runner == "frontend-reviewer"
    assert review.extra_prompt == "base review"
    assert review.prompt_file == str(home / ".agm" / "frontend-review.md")
    assert revise.runner == "frontend-reviser"
    assert revise.extra_prompt == "base revise"
    assert revise.prompt_file == str(home / ".agm" / "frontend-revise.md")
    assert refine.max_steps == 3
    assert refine.runner == "both"
    assert refine.reviewer == "frontend-review-only"


def test_load_review_config_rejects_missing_named_section(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    (home / ".agm" / "config.toml").write_text('[review]\nrunner = "reviewer"\n')
    cwd = tmp_path / "work"
    cwd.mkdir()

    with pytest.raises(ConfigCommandNotFound) as exc_info:
        load_review_config(home=home, proj_dir=None, cwd=cwd, command_name="fronend")

    assert exc_info.value.section_name == "review"
    assert exc_info.value.command_name == "fronend"


def test_load_review_config_allows_optional_missing_named_section(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    (home / ".agm" / "config.toml").write_text('[review]\nrunner = "reviewer"\n')
    cwd = tmp_path / "work"
    cwd.mkdir()

    config = load_review_config(
        home=home,
        proj_dir=None,
        cwd=cwd,
        command_name="fronend",
        require_command=False,
    )

    assert config.runner == "reviewer"


def test_load_loop_config_expands_braced_env_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROJ_DIR", str(tmp_path / "myproject"))
    (tmp_path / "myproject").mkdir()
    (tmp_path / "myproject" / "config").mkdir()
    (tmp_path / "myproject" / "config" / "prompt.md").write_text("project prompt")

    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    (home / ".agm" / "config.toml").write_text(
        '[loop]\nprompt_file = "${PROJ_DIR}/config/prompt.md"\n'
    )

    cwd = tmp_path / "work"
    cwd.mkdir()

    config = load_loop_config(home=home, proj_dir=None, cwd=cwd)

    assert config.prompt_file == str(tmp_path / "myproject" / "config" / "prompt.md")


def test_load_loop_config_resolves_command_specific_relative_paths_against_config_dir(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    (home / ".agm" / "codex-prompt.md").write_text("codex prompt")
    (home / ".agm" / "config.toml").write_text(
        '[loop.codex]\nprompt_file = "codex-prompt.md"\n'
    )

    cwd = tmp_path / "work"
    cwd.mkdir()

    config = load_loop_config(home=home, proj_dir=None, cwd=cwd, command_name="codex")

    assert config.prompt_file == str(home / ".agm" / "codex-prompt.md")


def test_load_loop_config_expands_user_tilde_in_prompt_file(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    (home / ".agm" / "config.toml").write_text(
        '[loop]\nprompt_file = "~/prompts/prompt.md"\n'
    )

    cwd = tmp_path / "work"
    cwd.mkdir()

    config = load_loop_config(home=home, proj_dir=None, cwd=cwd)

    expected = str(Path.home() / "prompts" / "prompt.md")
    assert config.prompt_file == expected


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


class TestUniquePaths:
    def test_deduplicates_paths(self, tmp_path: Path) -> None:
        p1 = tmp_path / "a"
        p2 = tmp_path / "b"
        p3 = tmp_path / "a"  # duplicate
        result = _unique_paths([p1, p2, p3])
        assert result == [p1, p2]

    def test_preserves_order(self, tmp_path: Path) -> None:
        paths = [tmp_path / name for name in ["c", "a", "b", "a"]]
        result = _unique_paths(paths)
        assert result == [tmp_path / "c", tmp_path / "a", tmp_path / "b"]


class TestOptionalBool:
    def test_returns_bool_value(self) -> None:
        assert _optional_bool({"flag": True}, "flag") is True
        assert _optional_bool({"flag": False}, "flag") is False

    def test_defaults_to_false(self) -> None:
        assert _optional_bool({}, "flag") is False

    def test_defaults_to_false_for_malformed_value(self) -> None:
        assert _optional_bool({"flag": "yes"}, "flag") is False

    def test_custom_default_true(self) -> None:
        assert _optional_bool({}, "flag", default=True) is True

    def test_custom_default_false(self) -> None:
        assert _optional_bool({}, "flag", default=False) is False

    def test_custom_default_ignored_when_value_is_bool(self) -> None:
        assert _optional_bool({"flag": False}, "flag", default=True) is False
