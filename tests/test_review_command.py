"""Tests for review, revise, and refine commands."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pytest

import agm.commands.refine as refine_mod
import agm.commands.review as review_mod
import agm.commands.revise as revise_mod
from agm.commands.args import RefineArgs, ReviewArgs, ReviseArgs
from agm.commands.refine import _write_review_file, refine
from agm.commands.review import (
    DEFAULT_REVIEW_ASPECTS,
    prepare_review,
)
from agm.commands.revise import prepare_revise
from agm.core import dry_run


class _FixedDatetime:
    @classmethod
    def now(cls) -> datetime:
        return datetime(2026, 5, 13, 14, 25, 30)


def _setup_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    prompt_dir = home / ".agm" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "review.md").write_text(
        "review $REVIEW_SCOPE for $REVIEW_ASPECTS\n",
        encoding="utf-8",
    )
    (prompt_dir / "revise.md").write_text("revise @${REVIEW_FILE}\n", encoding="utf-8")
    return home


def _review_args(
    *,
    command_name: str | None = None,
    runner: str | None = "fake-reviewer",
    scope: str | None = None,
    aspects: str | None = None,
    extra_aspects: str | None = None,
    prompt: str | None = None,
    prompt_file: str | None = None,
    extra_prompt: str | None = None,
    extra_prompt_file: str | None = None,
    review_file: str | None = None,
    no_review_file: bool = False,
) -> ReviewArgs:
    return ReviewArgs(
        command_name=command_name,
        runner=runner,
        scope=scope,
        aspects=aspects,
        extra_aspects=extra_aspects,
        prompt=prompt,
        prompt_file=prompt_file,
        extra_prompt=extra_prompt,
        extra_prompt_file=extra_prompt_file,
        review_file=review_file,
        no_review_file=no_review_file,
    )


def _revise_args(
    review_file: str,
    *,
    command_name: str | None = None,
    runner: str | None = "fake-reviser",
    prompt: str | None = None,
    prompt_file: str | None = None,
    extra_prompt: str | None = None,
    extra_prompt_file: str | None = None,
) -> ReviseArgs:
    return ReviseArgs(
        command_name=command_name,
        review_file=review_file,
        runner=runner,
        prompt=prompt,
        prompt_file=prompt_file,
        extra_prompt=extra_prompt,
        extra_prompt_file=extra_prompt_file,
    )


def test_prepare_review_expands_scope_and_aspects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")

    temp_files: list[Path] = []
    prepared = prepare_review(
        _review_args(scope="feature branch", extra_aspects="performance"),
        temp_files=temp_files,
    )

    assert prepared.command == ["fake-reviewer"]
    assert prepared.effective_file.read_text(encoding="utf-8") == (
        "review feature branch for "
        f"{DEFAULT_REVIEW_ASPECTS}, performance\n"
    )


def test_prepare_review_uses_default_loop_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    (home / ".agm" / "config.toml").write_text('[loop]\nrunner = "loop-runner -p"\n')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("PROJ_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    monkeypatch.setattr("agm.config.general.agm_installation_prefix", lambda: None)

    prepared = prepare_review(_review_args(runner=None), temp_files=[])

    assert prepared.command == ["loop-runner", "-p"]


def test_prepare_review_uses_builtin_runner_when_loop_runner_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")

    prepared = prepare_review(_review_args(runner=None), temp_files=[])

    assert prepared.command == ["claude", "-p"]


def test_prepare_review_uses_inline_prompt_and_extra_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")

    prepared = prepare_review(
        _review_args(
            prompt="inline $REVIEW_SCOPE",
            extra_prompt="extra $REVIEW_ASPECTS",
        ),
        temp_files=[],
    )

    assert prepared.effective_file.read_text(encoding="utf-8") == (
        f"inline {review_mod.DEFAULT_REVIEW_SCOPE}\n"
        f"extra {DEFAULT_REVIEW_ASPECTS}"
    )


def test_prepare_review_uses_cli_prompt_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    prompt = tmp_path / "review-custom.md"
    prompt.write_text("custom $REVIEW_SCOPE", encoding="utf-8")
    extra = tmp_path / "extra.md"
    extra.write_text("extra", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")

    prepared = prepare_review(
        _review_args(prompt_file=str(prompt), extra_prompt_file=str(extra)),
        temp_files=[],
    )

    assert prepared.effective_file.read_text(encoding="utf-8") == (
        f"custom {review_mod.DEFAULT_REVIEW_SCOPE}\nextra"
    )


def test_prepare_review_uses_config_prompt_and_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    (home / ".agm" / "config.toml").write_text(
        '[review]\nprompt = "from config $REVIEW_SCOPE"\nextra_prompt = "extra"\n'
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")

    prepared = prepare_review(_review_args(prompt=None, extra_prompt=None), temp_files=[])

    assert prepared.effective_file.read_text(encoding="utf-8") == (
        f"from config {review_mod.DEFAULT_REVIEW_SCOPE}\nextra"
    )


def test_prepare_review_uses_named_config_prompt_and_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    (home / ".agm" / "config.toml").write_text(
        '[review]\nextra_prompt = "base extra"\n'
        '[review.frontend]\nprompt = "frontend $REVIEW_SCOPE"\n'
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")

    prepared = prepare_review(_review_args(command_name="frontend"), temp_files=[])

    assert prepared.effective_file.read_text(encoding="utf-8") == (
        f"frontend {review_mod.DEFAULT_REVIEW_SCOPE}\nbase extra"
    )


def test_prepare_review_exits_when_named_config_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = _setup_home(tmp_path)
    (home / ".agm" / "config.toml").write_text('[review]\nextra_prompt = "base extra"\n')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        prepare_review(_review_args(command_name="fronend"), temp_files=[])

    err = capsys.readouterr().err
    assert "review subcommand 'fronend' is not defined" in err
    assert "usage: agm review [COMMAND]" in err


def test_prepare_review_uses_config_prompt_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    prompt = home / ".agm" / "configured-review.md"
    prompt.write_text("configured", encoding="utf-8")
    extra = home / ".agm" / "configured-extra.md"
    extra.write_text("extra", encoding="utf-8")
    (home / ".agm" / "config.toml").write_text(
        '[review]\nprompt_file = "configured-review.md"\n'
        'extra_prompt_file = "configured-extra.md"\n'
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")

    prepared = prepare_review(_review_args(), temp_files=[])

    assert prepared.effective_file.read_text(encoding="utf-8") == "configured\nextra"


def test_prepare_review_uses_aspects_without_extra_aspects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")

    prepared = prepare_review(_review_args(aspects="security"), temp_files=[])

    assert "for security\n" in prepared.effective_file.read_text(encoding="utf-8")


def test_prepare_revise_sets_review_file_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    review_file = tmp_path / "review.md"
    review_file.write_text("issue\n", encoding="utf-8")

    temp_files: list[Path] = []
    prepared = prepare_revise(_revise_args("review.md"), temp_files=temp_files)

    assert prepared.command == ["fake-reviser"]
    assert prepared.env["REVIEW_FILE"] == str(review_file)
    assert prepared.effective_file.read_text(encoding="utf-8") == f"revise @{review_file}\n"


def test_prepare_revise_accepts_absolute_review_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    review_file = tmp_path / "absolute-review.md"

    prepared = prepare_revise(_revise_args(str(review_file)), temp_files=[])

    assert prepared.env["REVIEW_FILE"] == str(review_file)


def test_prepare_revise_uses_config_prompt_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    prompt = home / ".agm" / "configured-revise.md"
    prompt.write_text("configured $REVIEW_FILE", encoding="utf-8")
    extra = home / ".agm" / "configured-revise-extra.md"
    extra.write_text("extra", encoding="utf-8")
    (home / ".agm" / "config.toml").write_text(
        '[revise]\nprompt_file = "configured-revise.md"\n'
        'extra_prompt_file = "configured-revise-extra.md"\n'
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    review_file = tmp_path / "review.md"

    prepared = prepare_revise(_revise_args(str(review_file)), temp_files=[])

    assert prepared.effective_file.read_text(encoding="utf-8") == (
        f"configured {review_file}\nextra"
    )


def test_prepare_revise_uses_config_inline_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    (home / ".agm" / "config.toml").write_text(
        '[revise]\nprompt = "configured $REVIEW_FILE"\nextra_prompt = "extra"\n'
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    review_file = tmp_path / "review.md"

    prepared = prepare_revise(_revise_args(str(review_file)), temp_files=[])

    assert prepared.effective_file.read_text(encoding="utf-8") == (
        f"configured {review_file}\nextra"
    )


def test_prepare_revise_uses_named_config_inline_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    (home / ".agm" / "config.toml").write_text(
        '[revise]\nextra_prompt = "base extra"\n'
        '[revise.frontend]\nprompt = "frontend $REVIEW_FILE"\n'
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    review_file = tmp_path / "review.md"

    prepared = prepare_revise(
        _revise_args(str(review_file), command_name="frontend"),
        temp_files=[],
    )

    assert prepared.effective_file.read_text(encoding="utf-8") == (
        f"frontend {review_file}\nbase extra"
    )


def test_prepare_revise_rejects_lone_config_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = _setup_home(tmp_path)
    (home / ".agm" / "config.toml").write_text('[revise.frontend]\nprompt = "frontend"\n')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        prepare_revise(_revise_args("frontend"), temp_files=[])

    assert exc_info.value.code == 1
    assert "revise command 'frontend' was provided without REVIEW_FILE" in capsys.readouterr().err


def test_prepare_revise_exits_when_named_config_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = _setup_home(tmp_path)
    (home / ".agm" / "config.toml").write_text('[revise]\nextra_prompt = "base extra"\n')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    review_file = tmp_path / "review.md"

    with pytest.raises(SystemExit):
        prepare_revise(_revise_args(str(review_file), command_name="fronend"), temp_files=[])

    err = capsys.readouterr().err
    assert "revise subcommand 'fronend' is not defined" in err
    assert "usage: agm revise [COMMAND]" in err
    assert "REVIEW_FILE" in err


def test_prepare_review_exits_when_default_prompt_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    (home / ".agm" / "prompts").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")

    with pytest.raises(SystemExit):
        prepare_review(_review_args(), temp_files=[])


def test_prepare_revise_exits_when_default_prompt_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    (home / ".agm" / "prompts").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")

    with pytest.raises(SystemExit):
        prepare_revise(_revise_args("review.md"), temp_files=[])


def test_write_review_file_preserves_env_vars(tmp_path: Path) -> None:
    temp_files: list[Path] = []
    path = _write_review_file(
        "issue=$VALUE and ${TOKEN}\n",
        temp_files=temp_files,
    )

    assert path in temp_files
    assert path.read_text(encoding="utf-8") == "issue=$VALUE and ${TOKEN}\n"


def test_unlink_temp_file_ignores_missing_untracked_file(tmp_path: Path) -> None:
    refine_mod._unlink_temp_file(tmp_path / "missing.md", temp_files=[])


def test_review_once_runs_prompt_and_cleans_temp_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    cleaned: list[list[Path]] = []

    def fake_run_prompt_command(
        command: list[str],
        target: Path,
        *,
        env: dict[str, str],
        stdout_callback: Callable[[str], None] | None = None,
        stderr_callback: Callable[[str], None] | None = None,
    ) -> str:
        assert command == ["fake-reviewer"]
        assert target.is_file()
        assert env["REVIEW_SCOPE"] == review_mod.DEFAULT_REVIEW_SCOPE
        if callable(stdout_callback):
            stdout_callback("out")
        if callable(stderr_callback):
            stderr_callback("err")
        return "outerr"

    def fake_cleanup(temp_files: list[Path]) -> None:
        cleaned.append(list(temp_files))

    monkeypatch.setattr("agm.agent.runner.run_prompt_command", fake_run_prompt_command)
    monkeypatch.setattr("agm.commands.review.cleanup_temp_files", fake_cleanup)

    output = review_mod.review_once(_review_args(no_review_file=True))

    captured = capsys.readouterr()
    assert captured.out == "out"
    assert captured.err == "err"
    assert output == "outerr"
    assert len(cleaned) == 1


def test_review_once_reuses_config_for_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    original_review_config = review_mod._review_config
    load_count = 0

    def counting_review_config(
        command_name: str | None, *, require_command: bool
    ) -> review_mod.ReviewConfig:
        nonlocal load_count
        load_count += 1
        return original_review_config(command_name, require_command=require_command)

    def fake_run_prompt_command(
        command: list[str],
        target: Path,
        *,
        env: dict[str, str],
        stdout_callback: Callable[[str], None] | None = None,
        stderr_callback: Callable[[str], None] | None = None,
    ) -> str:
        del command, target, env, stdout_callback, stderr_callback
        return "review output\n"

    monkeypatch.setattr("agm.commands.review._review_config", counting_review_config)
    monkeypatch.setattr("agm.agent.runner.run_prompt_command", fake_run_prompt_command)

    review_mod.review_once(_review_args(no_review_file=True))

    assert load_count == 1


def test_review_once_saves_output_to_default_review_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    monkeypatch.setattr("agm.core.log.git_helpers.containing_root", lambda _path: None)
    monkeypatch.setattr("agm.commands.review.datetime", _FixedDatetime)

    def fake_run_prompt_command(
        command: list[str],
        target: Path,
        *,
        env: dict[str, str],
        stdout_callback: Callable[[str], None] | None = None,
        stderr_callback: Callable[[str], None] | None = None,
    ) -> str:
        del command, target, env, stdout_callback, stderr_callback
        return "review output\n"

    monkeypatch.setattr("agm.agent.runner.run_prompt_command", fake_run_prompt_command)

    output = review_mod.review_once(_review_args())

    review_file = tmp_path / ".agent-files" / "review-20260513-142530.md"
    assert output == "review output\n"
    assert review_file.read_text(encoding="utf-8") == "review output\n"
    assert capsys.readouterr().out == f"Saved review to {review_file}\n"


def test_review_once_honors_explicit_and_disabled_review_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")

    def fake_run_prompt_command(
        command: list[str],
        target: Path,
        *,
        env: dict[str, str],
        stdout_callback: Callable[[str], None] | None = None,
        stderr_callback: Callable[[str], None] | None = None,
    ) -> str:
        del command, target, env, stdout_callback, stderr_callback
        return "review output\n"

    monkeypatch.setattr("agm.agent.runner.run_prompt_command", fake_run_prompt_command)

    review_mod.review_once(_review_args(review_file="saved/review.md"))
    review_mod.review_once(_review_args(no_review_file=True))

    assert (tmp_path / "saved" / "review.md").read_text(encoding="utf-8") == (
        "review output\n"
    )
    assert not (tmp_path / ".agent-files").exists()


def test_review_once_honors_none_and_absolute_review_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")

    def fake_run_prompt_command(
        command: list[str],
        target: Path,
        *,
        env: dict[str, str],
        stdout_callback: Callable[[str], None] | None = None,
        stderr_callback: Callable[[str], None] | None = None,
    ) -> str:
        del command, target, env, stdout_callback, stderr_callback
        return "review output\n"

    monkeypatch.setattr("agm.agent.runner.run_prompt_command", fake_run_prompt_command)

    absolute_review_file = tmp_path / "absolute-review.md"
    review_mod.review_once(_review_args(review_file="none"))
    review_mod.review_once(_review_args(review_file=str(absolute_review_file)))

    assert absolute_review_file.read_text(encoding="utf-8") == "review output\n"
    assert not (tmp_path / ".agent-files").exists()


def test_revise_once_dry_run_prints_configuration_and_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    dry_run.set_enabled(True)
    try:
        output = revise_mod.revise_once(_revise_args("review.md"))
    finally:
        dry_run.set_enabled(False)

    captured = capsys.readouterr()
    assert output == ""
    assert "dry-run: revise configuration" in captured.out
    assert "dry-run: command [agent]:" in captured.out


def test_revise_stream_callbacks_write_non_empty_chunks(capsys: pytest.CaptureFixture[str]) -> None:
    revise_mod.write_stdout("out")
    revise_mod.write_stderr("err")
    revise_mod.write_stdout("")
    revise_mod.write_stderr("")

    captured = capsys.readouterr()
    assert captured.out == "out"
    assert captured.err == "err"


def test_prepare_revise_uses_default_loop_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    (home / ".agm" / "config.toml").write_text('[loop]\nrunner = "loop-runner -p"\n')
    review_file = tmp_path / "review.md"
    review_file.write_text("review\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("PROJ_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    monkeypatch.setattr("agm.config.general.agm_installation_prefix", lambda: None)

    prepared = prepare_revise(_revise_args("review.md", runner=None), temp_files=[])

    assert prepared.command == ["loop-runner", "-p"]


def test_review_once_dry_run_prints_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    dry_run.set_enabled(True)
    try:
        output = review_mod.review_once(_review_args())
    finally:
        dry_run.set_enabled(False)

    captured = capsys.readouterr()
    assert output == ""
    assert "dry-run: review configuration" in captured.out


def test_write_stream_helpers_ignore_empty_chunks(
    capsys: pytest.CaptureFixture[str],
) -> None:
    review_mod.write_stdout("")
    review_mod.write_stderr("")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_refine_repeats_revise_for_unknown_status_and_honors_max_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    reviews: list[ReviewArgs] = []
    revisions: list[ReviseArgs] = []

    def fake_review_once(
        args: ReviewArgs,
        *,
        stdout_callback: Callable[[str], None] = review_mod.write_stdout,
        stderr_callback: Callable[[str], None] = review_mod.write_stderr,
    ) -> str:
        del stdout_callback, stderr_callback
        reviews.append(args)
        return "review result\n"

    def fake_revise_once(
        args: ReviseArgs,
        *,
        stdout_callback: Callable[[str], None] = review_mod.write_stdout,
        stderr_callback: Callable[[str], None] = review_mod.write_stderr,
    ) -> str:
        del stdout_callback, stderr_callback
        revisions.append(args)
        return "try again\n"

    monkeypatch.setattr("agm.commands.refine.review_once", fake_review_once)
    monkeypatch.setattr("agm.commands.refine.revise_once", fake_revise_once)

    refine(
        RefineArgs(
            max_steps=3,
            runner=None,
            reviewer=None,
            reviser=None,
            scope=None,
            aspects=None,
            review_prompt=None,
            review_prompt_file=None,
            extra_review_prompt=None,
            extra_review_prompt_file=None,
            revise_prompt=None,
            revise_prompt_file=None,
            extra_revise_prompt=None,
            extra_revise_prompt_file=None,
        )
    )

    assert len(reviews) == 1
    assert len(revisions) == 3


def test_refine_runs_fresh_review_after_continue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    reviews: list[ReviewArgs] = []
    outputs = iter(["CONTINUE\n", "COMPLETE\n"])

    def fake_review_once(
        args: ReviewArgs,
        *,
        stdout_callback: Callable[[str], None] = review_mod.write_stdout,
        stderr_callback: Callable[[str], None] = review_mod.write_stderr,
    ) -> str:
        del stdout_callback, stderr_callback
        reviews.append(args)
        return "review result\n"

    def fake_revise_once(
        args: ReviseArgs,
        *,
        stdout_callback: Callable[[str], None] = review_mod.write_stdout,
        stderr_callback: Callable[[str], None] = review_mod.write_stderr,
    ) -> str:
        del args, stdout_callback, stderr_callback
        return next(outputs)

    monkeypatch.setattr("agm.commands.refine.review_once", fake_review_once)
    monkeypatch.setattr("agm.commands.refine.revise_once", fake_revise_once)

    refine(
        RefineArgs(
            max_steps=5,
            runner="both",
            reviewer="reviewer",
            reviser="reviser",
            scope="scope",
            aspects="aspects",
            review_prompt=None,
            review_prompt_file=None,
            extra_review_prompt=None,
            extra_review_prompt_file=None,
            revise_prompt=None,
            revise_prompt_file=None,
            extra_revise_prompt=None,
            extra_revise_prompt_file=None,
        )
    )

    assert len(reviews) == 2
    assert reviews[0].runner == "reviewer"
    assert reviews[0].scope == "scope"
    assert reviews[0].aspects == "aspects"
    assert all(review.no_review_file for review in reviews)
    assert all(review.review_file is None for review in reviews)


def test_refine_save_review_enables_auto_review_file_for_each_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    reviews: list[ReviewArgs] = []
    outputs = iter(["CONTINUE\n", "COMPLETE\n"])

    def fake_review_once(
        args: ReviewArgs,
        *,
        stdout_callback: Callable[[str], None] = review_mod.write_stdout,
        stderr_callback: Callable[[str], None] = review_mod.write_stderr,
    ) -> str:
        del stdout_callback, stderr_callback
        reviews.append(args)
        return "review result\n"

    def fake_revise_once(
        args: ReviseArgs,
        *,
        stdout_callback: Callable[[str], None] = review_mod.write_stdout,
        stderr_callback: Callable[[str], None] = review_mod.write_stderr,
    ) -> str:
        del args, stdout_callback, stderr_callback
        return next(outputs)

    monkeypatch.setattr("agm.commands.refine.review_once", fake_review_once)
    monkeypatch.setattr("agm.commands.refine.revise_once", fake_revise_once)

    refine(
        RefineArgs(
            max_steps=5,
            runner=None,
            reviewer=None,
            reviser=None,
            scope=None,
            aspects=None,
            review_prompt=None,
            review_prompt_file=None,
            extra_review_prompt=None,
            extra_review_prompt_file=None,
            revise_prompt=None,
            revise_prompt_file=None,
            extra_revise_prompt=None,
            extra_revise_prompt_file=None,
            save_review=True,
        )
    )

    assert len(reviews) == 2
    assert all(review.review_file == "auto" for review in reviews)
    assert not any(review.no_review_file for review in reviews)


def test_refine_leaves_missing_scope_and_aspects_for_review_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    (home / ".agm" / "config.toml").write_text("[refine.frontend]\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    reviews: list[ReviewArgs] = []

    def fake_review_once(
        args: ReviewArgs,
        *,
        stdout_callback: Callable[[str], None] = review_mod.write_stdout,
        stderr_callback: Callable[[str], None] = review_mod.write_stderr,
    ) -> str:
        del stdout_callback, stderr_callback
        reviews.append(args)
        return "review result\n"

    def fake_revise_once(
        args: ReviseArgs,
        *,
        stdout_callback: Callable[[str], None] = review_mod.write_stdout,
        stderr_callback: Callable[[str], None] = review_mod.write_stderr,
    ) -> str:
        del args, stdout_callback, stderr_callback
        return "COMPLETE\n"

    monkeypatch.setattr("agm.commands.refine.review_once", fake_review_once)
    monkeypatch.setattr("agm.commands.refine.revise_once", fake_revise_once)

    refine(
        RefineArgs(
            max_steps=None,
            runner=None,
            reviewer=None,
            reviser=None,
            scope=None,
            aspects=None,
            review_prompt=None,
            review_prompt_file=None,
            extra_review_prompt=None,
            extra_review_prompt_file=None,
            revise_prompt=None,
            revise_prompt_file=None,
            extra_revise_prompt=None,
            extra_revise_prompt_file=None,
            command_name="frontend",
        )
    )

    assert reviews[0].scope is None
    assert reviews[0].aspects is None


def test_refine_uses_named_config_and_forwards_command_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    (home / ".agm" / "config.toml").write_text(
        '[refine.frontend]\nrunner = "frontend-runner"\nscope = "frontend scope"\n'
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    reviews: list[ReviewArgs] = []
    revisions: list[ReviseArgs] = []

    def fake_review_once(
        args: ReviewArgs,
        *,
        stdout_callback: Callable[[str], None] = review_mod.write_stdout,
        stderr_callback: Callable[[str], None] = review_mod.write_stderr,
    ) -> str:
        del stdout_callback, stderr_callback
        reviews.append(args)
        return "review result\n"

    def fake_revise_once(
        args: ReviseArgs,
        *,
        stdout_callback: Callable[[str], None] = review_mod.write_stdout,
        stderr_callback: Callable[[str], None] = review_mod.write_stderr,
    ) -> str:
        del stdout_callback, stderr_callback
        revisions.append(args)
        return "COMPLETE\n"

    monkeypatch.setattr("agm.commands.refine.review_once", fake_review_once)
    monkeypatch.setattr("agm.commands.refine.revise_once", fake_revise_once)

    refine(
        RefineArgs(
            max_steps=None,
            runner=None,
            reviewer=None,
            reviser=None,
            scope=None,
            aspects=None,
            review_prompt=None,
            review_prompt_file=None,
            extra_review_prompt=None,
            extra_review_prompt_file=None,
            revise_prompt=None,
            revise_prompt_file=None,
            extra_revise_prompt=None,
            extra_revise_prompt_file=None,
            command_name="frontend",
        )
    )

    assert reviews[0].runner == "frontend-runner"
    assert reviews[0].scope == "frontend scope"
    assert reviews[0].command_name == "frontend"
    assert revisions[0].runner == "frontend-runner"
    assert revisions[0].command_name == "frontend"


def test_refine_writes_review_and_revise_output_to_log_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    log_file = tmp_path / "refine.log"

    def fake_review_once(
        args: ReviewArgs,
        *,
        stdout_callback: Callable[[str], None] = review_mod.write_stdout,
        stderr_callback: Callable[[str], None] = review_mod.write_stderr,
    ) -> str:
        del args
        stdout_callback("review stdout\n")
        stderr_callback("review stderr\n")
        return "review result\n"

    def fake_revise_once(
        args: ReviseArgs,
        *,
        stdout_callback: Callable[[str], None] = review_mod.write_stdout,
        stderr_callback: Callable[[str], None] = review_mod.write_stderr,
    ) -> str:
        del args
        stdout_callback("revise stdout\n")
        stderr_callback("revise stderr\n")
        return "COMPLETE\n"

    monkeypatch.setattr("agm.commands.refine.review_once", fake_review_once)
    monkeypatch.setattr("agm.commands.refine.revise_once", fake_revise_once)

    refine(
        RefineArgs(
            max_steps=1,
            runner=None,
            reviewer=None,
            reviser=None,
            scope=None,
            aspects=None,
            review_prompt=None,
            review_prompt_file=None,
            extra_review_prompt=None,
            extra_review_prompt_file=None,
            revise_prompt=None,
            revise_prompt_file=None,
            extra_revise_prompt=None,
            extra_revise_prompt_file=None,
            log_file=str(log_file),
        )
    )

    log_content = log_file.read_text(encoding="utf-8")
    assert "Step 1" in log_content
    assert log_content.endswith("review stdout\nreview stderr\nrevise stdout\nrevise stderr\n")


def test_refine_step_header_is_printed_and_logged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    log_file = tmp_path / "refine.log"

    def fake_review_once(
        args: ReviewArgs,
        *,
        stdout_callback: Callable[[str], None] = review_mod.write_stdout,
        stderr_callback: Callable[[str], None] = review_mod.write_stderr,
    ) -> str:
        del args, stderr_callback
        stdout_callback("review stdout\n")
        return "review result\n"

    def fake_revise_once(
        args: ReviseArgs,
        *,
        stdout_callback: Callable[[str], None] = review_mod.write_stdout,
        stderr_callback: Callable[[str], None] = review_mod.write_stderr,
    ) -> str:
        del args, stderr_callback
        stdout_callback("revise stdout\n")
        return "COMPLETE\n"

    monkeypatch.setattr("agm.commands.refine.review_once", fake_review_once)
    monkeypatch.setattr("agm.commands.refine.revise_once", fake_revise_once)

    refine(
        RefineArgs(
            max_steps=1,
            runner=None,
            reviewer=None,
            reviser=None,
            scope=None,
            aspects=None,
            review_prompt=None,
            review_prompt_file=None,
            extra_review_prompt=None,
            extra_review_prompt_file=None,
            revise_prompt=None,
            revise_prompt_file=None,
            extra_revise_prompt=None,
            extra_revise_prompt_file=None,
            log_file=str(log_file),
        )
    )

    out = capsys.readouterr().out
    log_content = log_file.read_text(encoding="utf-8")
    assert "Step 1" in out
    assert out.index("Step 1") < out.index("review stdout")
    assert "Step 1" in log_content
    assert log_content.index("Step 1") < log_content.index("review stdout")


def test_refine_prints_logging_to_full_default_log_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agm.core.log.git_helpers.containing_root", lambda _path: None)

    def fake_review_once(
        args: ReviewArgs,
        *,
        stdout_callback: Callable[[str], None] = review_mod.write_stdout,
        stderr_callback: Callable[[str], None] = review_mod.write_stderr,
    ) -> str:
        del args, stdout_callback, stderr_callback
        return "review result\n"

    def fake_revise_once(
        args: ReviseArgs,
        *,
        stdout_callback: Callable[[str], None] = review_mod.write_stdout,
        stderr_callback: Callable[[str], None] = review_mod.write_stderr,
    ) -> str:
        del args, stdout_callback, stderr_callback
        return "COMPLETE\n"

    monkeypatch.setattr("agm.commands.refine.review_once", fake_review_once)
    monkeypatch.setattr("agm.commands.refine.revise_once", fake_revise_once)

    refine(
        RefineArgs(
            max_steps=1,
            runner=None,
            reviewer=None,
            reviser=None,
            scope=None,
            aspects=None,
            review_prompt=None,
            review_prompt_file=None,
            extra_review_prompt=None,
            extra_review_prompt_file=None,
            revise_prompt=None,
            revise_prompt_file=None,
            extra_revise_prompt=None,
            extra_revise_prompt_file=None,
        )
    )

    first_line = capsys.readouterr().out.splitlines()[0]
    assert first_line.startswith(f"Logging to {tmp_path / '.agent-files' / 'refine-'}")
    assert first_line.endswith(".log")


def test_refine_exits_when_named_config_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = _setup_home(tmp_path)
    (home / ".agm" / "config.toml").write_text('[refine]\nrunner = "base-runner"\n')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        refine(
            RefineArgs(
                max_steps=None,
                runner=None,
                reviewer=None,
                reviser=None,
                scope=None,
                aspects=None,
                review_prompt=None,
                review_prompt_file=None,
                extra_review_prompt=None,
                extra_review_prompt_file=None,
                revise_prompt=None,
                revise_prompt_file=None,
                extra_revise_prompt=None,
                extra_revise_prompt_file=None,
                command_name="fronend",
            )
        )

    err = capsys.readouterr().err
    assert "refine subcommand 'fronend' is not defined" in err
    assert "usage: agm refine [COMMAND]" in err


def test_run_wrappers_translate_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_interrupt(_args: object) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("agm.commands.review.review_once", raise_interrupt)
    with pytest.raises(SystemExit) as review_exit:
        review_mod.run(_review_args())
    assert review_exit.value.code == 130

    monkeypatch.setattr("agm.commands.revise.revise_once", raise_interrupt)
    with pytest.raises(SystemExit) as revise_exit:
        revise_mod.run(_revise_args("review.md"))
    assert revise_exit.value.code == 130

    monkeypatch.setattr("agm.commands.refine.refine", raise_interrupt)
    with pytest.raises(SystemExit) as refine_exit:
        refine_mod.run(
            RefineArgs(
                max_steps=1,
                runner=None,
                reviewer=None,
                reviser=None,
                scope=None,
                aspects=None,
                review_prompt=None,
                review_prompt_file=None,
                extra_review_prompt=None,
                extra_review_prompt_file=None,
                revise_prompt=None,
                revise_prompt_file=None,
                extra_revise_prompt=None,
                extra_revise_prompt_file=None,
            )
        )
    assert refine_exit.value.code == 130
