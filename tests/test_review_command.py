"""Tests for review, revise, and refine commands."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pytest

import agm.agent.review.review as review_pass
import agm.agent.review.revise as revise_pass
import agm.commands.refine as refine_mod
import agm.commands.review as review_mod
import agm.commands.revise as revise_mod
from agm.agent.review.review import DEFAULT_REVIEW_ASPECTS, prepare_review
from agm.agent.review.revise import prepare_revise
from agm.cli_support.args import RefineArgs, ReviewArgs, ReviseArgs
from agm.commands.refine import _write_review_file, refine
from agm.core import dry_run
from tests._git_helpers import init_repo
from tests._process_helpers import AgentCall, AgentReply, FakeAgent, fake_agent
from tests._timeouts import fail_if_slow


class _FixedDatetime:
    @classmethod
    def now(cls) -> datetime:
        return datetime(2026, 5, 13, 14, 25, 30)


def _setup_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    prompt_dir = home / ".agm" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "review.md").write_text(
        "review %{REVIEW_SCOPE} for %{REVIEW_ASPECTS}\n",
        encoding="utf-8",
    )
    (prompt_dir / "revise.md").write_text("revise @%{REVIEW_FILE}\n", encoding="utf-8")
    return home


def _isolate_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Stop git's repository search from escaping *tmp_path*.

    ``.agent-files`` is placed at the containing git root, which AGM finds by
    running git for real.  Capping the search keeps the answer the same whether
    or not the directory holding the suite's temporary files happens to sit
    inside somebody's checkout, while a repository created inside *tmp_path* is
    still found normally.
    """
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))


def _review_file_of(revise_prompt: str) -> Path:
    """The review file a rendered ``revise @<file>`` prompt points the agent at."""
    return Path(revise_prompt.split("@", maxsplit=1)[1].strip())


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


def _refine_args(**overrides: object) -> RefineArgs:
    """Build ``RefineArgs`` with every option unset except the given overrides."""
    fields: dict[str, object] = {
        "max_steps": None,
        "no_max_steps": False,
        "runner": None,
        "reviewer": "fake-reviewer",
        "reviser": "fake-reviser",
        "scope": None,
        "aspects": None,
        "review_prompt": None,
        "review_prompt_file": None,
        "extra_review_prompt": None,
        "extra_review_prompt_file": None,
        "revise_prompt": None,
        "revise_prompt_file": None,
        "extra_revise_prompt": None,
        "extra_revise_prompt_file": None,
        "no_log": True,
    }
    fields.update(overrides)
    return RefineArgs(**fields)


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
        f"review feature branch for {DEFAULT_REVIEW_ASPECTS}, performance\n"
    )


def test_prepare_review_uses_loop_runner_as_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    (home / ".agm" / "config.toml").write_text('[loop]\nrunner = "loop-runner -p"\n')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("PROJ_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")

    prepared = prepare_review(_review_args(runner=None), temp_files=[])

    assert prepared.command == ["loop-runner", "-p"]


def test_prepare_review_config_runner_wins_over_loop_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    (home / ".agm" / "config.toml").write_text(
        '[loop]\nrunner = "loop-runner -p"\n[review]\nrunner = "review-runner"\n'
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("PROJ_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")

    prepared = prepare_review(_review_args(runner=None), temp_files=[])

    assert prepared.command == ["review-runner"]


def test_prepare_review_cli_runner_wins_over_loop_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    (home / ".agm" / "config.toml").write_text(
        '[loop]\nrunner = "loop-runner -p"\n[review]\nrunner = "review-runner"\n'
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("PROJ_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")

    prepared = prepare_review(_review_args(runner="cli-runner"), temp_files=[])

    assert prepared.command == ["cli-runner"]


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
            prompt="inline %{REVIEW_SCOPE}",
            extra_prompt="extra %{REVIEW_ASPECTS}",
        ),
        temp_files=[],
    )

    assert prepared.effective_file.read_text(encoding="utf-8") == (
        f"inline {review_pass.DEFAULT_REVIEW_SCOPE}\nextra {DEFAULT_REVIEW_ASPECTS}"
    )


def test_prepare_review_uses_cli_prompt_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    prompt = tmp_path / "review-custom.md"
    prompt.write_text("custom %{REVIEW_SCOPE}", encoding="utf-8")
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
        f"custom {review_pass.DEFAULT_REVIEW_SCOPE}\nextra"
    )


def test_prepare_review_uses_config_prompt_and_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    (home / ".agm" / "config.toml").write_text(
        '[review]\nprompt = "from config %{REVIEW_SCOPE}"\nextra_prompt = "extra"\n'
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")

    prepared = prepare_review(_review_args(prompt=None, extra_prompt=None), temp_files=[])

    assert prepared.effective_file.read_text(encoding="utf-8") == (
        f"from config {review_pass.DEFAULT_REVIEW_SCOPE}\nextra"
    )


def test_prepare_review_uses_named_config_prompt_and_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    (home / ".agm" / "config.toml").write_text(
        '[review]\nextra_prompt = "base extra"\n'
        '[review.frontend]\nprompt = "frontend %{REVIEW_SCOPE}"\n'
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")

    prepared = prepare_review(_review_args(command_name="frontend"), temp_files=[])

    assert prepared.effective_file.read_text(encoding="utf-8") == (
        f"frontend {review_pass.DEFAULT_REVIEW_SCOPE}\nbase extra"
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
    assert "fronend" in err
    assert "review" in err.lower()


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
    prompt.write_text("configured %{REVIEW_FILE}", encoding="utf-8")
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
        '[revise]\nprompt = "configured %{REVIEW_FILE}"\nextra_prompt = "extra"\n'
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
        '[revise.frontend]\nprompt = "frontend %{REVIEW_FILE}"\n'
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
    err = capsys.readouterr().err
    assert "frontend" in err
    assert "REVIEW_FILE" in err


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
    assert "fronend" in err
    assert "revise" in err.lower()
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


def test_review_once_streams_output_and_removes_its_prompt_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    agent = fake_agent(monkeypatch, lambda _call: AgentReply(stdout="out", stderr="err"))

    output = review_mod.review_once(_review_args(no_review_file=True))

    captured = capsys.readouterr()
    assert captured.out == "out"
    assert captured.err == "err"
    assert output == "outerr"
    (call,) = agent.calls
    assert call.runner == ["fake-reviewer"]
    assert call.env["REVIEW_SCOPE"] == review_pass.DEFAULT_REVIEW_SCOPE
    assert call.prompt == (
        f"review {review_pass.DEFAULT_REVIEW_SCOPE} for {DEFAULT_REVIEW_ASPECTS}\n"
    )
    assert not call.prompt_file.exists()


def test_review_once_saves_an_empty_review_when_the_agent_prints_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A silent agent still produces a review — an empty one."""
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    fake_agent(monkeypatch, AgentReply())

    output = review_mod.review_once(_review_args(review_file="saved/review.md"))

    assert output == ""
    assert (tmp_path / "saved" / "review.md").read_text(encoding="utf-8") == ""


def test_review_once_saves_output_to_default_review_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    monkeypatch.setattr("agm.agent.review.review.datetime", _FixedDatetime)
    _isolate_git(monkeypatch, tmp_path)
    fake_agent(monkeypatch, "review output\n")

    output = review_mod.review_once(_review_args())

    review_file = tmp_path / ".agent-files" / "review-20260513-142530-000000.md"
    assert output == "review output\n"
    assert review_file.read_text(encoding="utf-8") == "review output\n"
    assert "review-20260513-142530-000000.md" in capsys.readouterr().out


def test_review_once_saves_the_default_review_file_at_the_git_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    """The review of a checkout lands in the checkout's own ``.agent-files``."""
    home = _setup_home(tmp_path)
    checkout = init_repo(tmp_path / "checkout", env)
    nested = checkout / "src"
    nested.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(nested)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    _isolate_git(monkeypatch, tmp_path)
    fake_agent(monkeypatch, "review output\n")

    review_mod.review_once(_review_args())

    saved = sorted((checkout / ".agent-files").glob("review-*.md"))
    assert [path.read_text(encoding="utf-8") for path in saved] == ["review output\n"]
    assert not (nested / ".agent-files").exists()


def test_review_once_honors_explicit_and_disabled_review_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    _isolate_git(monkeypatch, tmp_path)
    fake_agent(monkeypatch, "review output\n")

    review_mod.review_once(_review_args(review_file="saved/review.md"))
    review_mod.review_once(_review_args(no_review_file=True))

    assert (tmp_path / "saved" / "review.md").read_text(encoding="utf-8") == ("review output\n")
    assert not (tmp_path / ".agent-files").exists()


def test_review_once_saves_to_configured_review_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    (home / ".agm" / "config.toml").write_text(
        '[review]\nreview_file = "configured/review.md"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    _isolate_git(monkeypatch, tmp_path)
    fake_agent(monkeypatch, "review output\n")

    review_mod.review_once(_review_args())

    assert (tmp_path / "configured" / "review.md").read_text(encoding="utf-8") == (
        "review output\n"
    )
    assert not (tmp_path / ".agent-files").exists()


def test_review_once_warns_before_overwriting_existing_review_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    review_file = tmp_path / "saved" / "review.md"
    review_file.parent.mkdir()
    review_file.write_text("existing\n", encoding="utf-8")
    fake_agent(monkeypatch, "new review output\n")

    review_mod.review_once(_review_args(review_file="saved/review.md"))

    captured = capsys.readouterr()
    assert "saved/review.md" in captured.err
    assert review_file.read_text(encoding="utf-8") == "new review output\n"


def test_review_once_honors_none_and_absolute_review_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    _isolate_git(monkeypatch, tmp_path)
    fake_agent(monkeypatch, "review output\n")

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
    assert "dry-run" in captured.out
    assert "fake-reviser" in captured.out


def test_review_and_revise_interpolate_session_id_into_the_runner_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    review_file = tmp_path / "review.md"
    review_file.write_text("review findings\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SESSION_ID", "legacy-review-session")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    agent = fake_agent(monkeypatch, "agent output\n")

    assert (
        review_pass.review_once(
            _review_args(runner="fake-reviewer --session=%{SESSION_ID}", no_review_file=True)
        )
        == "agent output\n"
    )
    assert (
        revise_pass.revise_once(
            _revise_args("review.md", runner="fake-reviser --session=%{SESSION_ID}")
        )
        == "agent output\n"
    )

    assert [call.runner for call in agent.calls] == [
        ["fake-reviewer", "--session=legacy-review-session"],
        ["fake-reviser", "--session=legacy-review-session"],
    ]
    assert agent.prompts[0] == (
        f"review {review_pass.DEFAULT_REVIEW_SCOPE} for {DEFAULT_REVIEW_ASPECTS}\n"
    )
    assert _review_file_of(agent.prompts[1]) == review_file


def test_revise_once_sends_the_review_file_to_the_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Outside dry-run the reviser is really invoked, with no dry-run output."""
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    review_file = tmp_path / "review.md"
    review_file.write_text("findings\n", encoding="utf-8")
    agent = fake_agent(monkeypatch, "agent output")

    assert not dry_run.enabled()
    output = revise_mod.revise_once(_revise_args("review.md"))

    (call,) = agent.calls
    assert call.runner == ["fake-reviser"]
    assert _review_file_of(call.prompt) == review_file
    assert output == "agent output"
    assert "dry-run" not in capsys.readouterr().out


def test_revise_stream_callbacks_write_non_empty_chunks(capsys: pytest.CaptureFixture[str]) -> None:
    revise_pass.write_stdout("out")
    revise_pass.write_stderr("err")
    revise_pass.write_stdout("")
    revise_pass.write_stderr("")

    captured = capsys.readouterr()
    assert captured.out == "out"
    assert captured.err == "err"


def test_prepare_revise_uses_loop_runner_as_fallback(
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

    prepared = prepare_revise(_revise_args("review.md", runner=None), temp_files=[])

    assert prepared.command == ["loop-runner", "-p"]


def test_review_once_dry_run_preserves_generated_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    dry_run.set_enabled(True)
    try:
        review_mod.review_once(_review_args(no_review_file=True))
    finally:
        dry_run.set_enabled(False)

    captured = capsys.readouterr()
    prompt_file = Path(captured.out.rsplit("@", maxsplit=1)[1].strip())
    try:
        assert prompt_file.read_text(encoding="utf-8") == (
            f"review {review_pass.DEFAULT_REVIEW_SCOPE} for {DEFAULT_REVIEW_ASPECTS}\n"
        )
    finally:
        prompt_file.unlink(missing_ok=True)


def test_prepare_revise_config_runner_wins_over_loop_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    (home / ".agm" / "config.toml").write_text(
        '[loop]\nrunner = "loop-runner -p"\n[revise]\nrunner = "revise-runner"\n'
    )
    review_file = tmp_path / "review.md"
    review_file.write_text("review\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("PROJ_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")

    prepared = prepare_revise(_revise_args("review.md", runner=None), temp_files=[])

    assert prepared.command == ["revise-runner"]


def test_prepare_revise_cli_runner_wins_over_loop_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    (home / ".agm" / "config.toml").write_text(
        '[loop]\nrunner = "loop-runner -p"\n[revise]\nrunner = "revise-runner"\n'
    )
    review_file = tmp_path / "review.md"
    review_file.write_text("review\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("PROJ_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")

    prepared = prepare_revise(_revise_args("review.md", runner="cli-runner"), temp_files=[])

    assert prepared.command == ["cli-runner"]


def test_prepare_revise_uses_builtin_runner_when_loop_runner_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path)
    review_file = tmp_path / "review.md"
    review_file.write_text("review\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")

    prepared = prepare_revise(_revise_args("review.md", runner=None), temp_files=[])

    assert prepared.command == ["claude", "-p"]


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
    assert "dry-run" in captured.out
    assert ".agent-files/" in captured.out
    assert "review-" in captured.out


def test_write_stream_helpers_ignore_empty_chunks(
    capsys: pytest.CaptureFixture[str],
) -> None:
    review_pass.write_stdout("")
    review_pass.write_stderr("")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def _refine_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set up the prompts, working directory and PATH lookup a refine run needs."""
    home = _setup_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/bin/fake")
    _isolate_git(monkeypatch, tmp_path)
    return home


def _refine_agent(
    monkeypatch: pytest.MonkeyPatch,
    revise_statuses: Sequence[str] = (),
    *,
    review_output: str = "review result\n",
    final_status: str | None = "try again\n",
) -> FakeAgent:
    """Fake agent answering review calls and scripting the reviser's verdicts.

    Statuses are consumed one per revise call; once the script runs out every
    further revise call answers *final_status*, or fails the test when that is
    ``None`` (which also bounds an otherwise unbounded refine run).
    """
    statuses = iter(revise_statuses)

    def respond(call: AgentCall) -> str | AgentReply:
        if not _is_revise(call):
            return review_output
        status = next(statuses, final_status)
        assert status is not None, "the reviser ran more times than the script allows"
        return status

    return fake_agent(monkeypatch, respond)


def _is_revise(call: AgentCall) -> bool:
    """Whether *call* ran the revise pass rather than the review pass.

    The two default prompts differ in their first word, so the rendered prompt
    identifies the pass without depending on how the runner was named.
    """
    return call.prompt.startswith("revise")


def _passes(agent: FakeAgent) -> list[str]:
    """Which pass each recorded agent call ran, in order."""
    return ["revise" if _is_revise(call) else "review" for call in agent.calls]


def _revised_review_files(agent: FakeAgent) -> list[Path]:
    """The review file each revise call was pointed at, in order."""
    return [_review_file_of(call.prompt) for call in agent.calls if _is_revise(call)]


def test_refine_repeats_revise_for_unknown_status_and_honors_max_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _refine_home(tmp_path, monkeypatch)
    agent = _refine_agent(monkeypatch)

    refine(_refine_args(max_steps=3))

    assert _passes(agent) == ["review", "revise", "revise", "revise"]
    assert agent.runners == ["fake-reviewer", "fake-reviser", "fake-reviser", "fake-reviser"]


def test_refine_uses_default_max_steps_when_unconfigured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _refine_home(tmp_path, monkeypatch)
    agent = _refine_agent(monkeypatch)

    refine(_refine_args())

    assert _passes(agent).count("revise") == refine_mod.DEFAULT_MAX_STEPS


@pytest.mark.parametrize("max_steps", [1, 2, 5])
def test_refine_max_steps_limits_iterations_to_exact_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_steps: int
) -> None:
    _refine_home(tmp_path, monkeypatch)
    agent = _refine_agent(monkeypatch)

    refine(_refine_args(max_steps=max_steps))

    assert _passes(agent).count("revise") == max_steps


def test_refine_max_steps_one_with_continue_still_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _refine_home(tmp_path, monkeypatch)
    agent = _refine_agent(monkeypatch, ["CONTINUE\n"])

    refine(_refine_args(max_steps=1))

    assert _passes(agent) == ["review", "revise"]


def test_refine_explicit_max_steps_overrides_configured_unlimited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _refine_home(tmp_path, monkeypatch)
    (home / ".agm" / "config.toml").write_text("[refine]\nno_max_steps = true\n")
    agent = _refine_agent(monkeypatch)

    refine(_refine_args(max_steps=1))

    assert _passes(agent) == ["review", "revise"]


def test_refine_dry_run_plans_one_cycle_when_unlimited(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _refine_home(tmp_path, monkeypatch)
    dry_run.set_enabled(True)

    with fail_if_slow("unlimited refine dry-run did not terminate"):
        refine(_refine_args(no_max_steps=True))

    output = capsys.readouterr().out
    assert output.count("Step 1") == 1
    assert "Step 2" not in output
    assert output.count("dry-run: review configuration") == 1
    assert output.count("dry-run: revise configuration") == 1
    assert output.count("fake-reviewer @") == 1
    assert output.count("fake-reviser @") == 1


def test_refine_explicit_prompt_files_override_configured_inline_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _refine_home(tmp_path, monkeypatch)
    (home / ".agm" / "config.toml").write_text(
        "[refine]\n"
        'review_prompt = "configured review"\n'
        'extra_review_prompt = "configured review extra"\n'
        'revise_prompt = "configured revise"\n'
        'extra_revise_prompt = "configured revise extra"\n'
    )
    prompt_files = {
        "review_prompt_file": "file review",
        "extra_review_prompt_file": "file review extra",
        "revise_prompt_file": "file revise",
        "extra_revise_prompt_file": "file revise extra",
    }
    paths: dict[str, str] = {}
    for option, content in prompt_files.items():
        path = tmp_path / f"{option}.md"
        path.write_text(content, encoding="utf-8")
        paths[option] = str(path)

    agent = fake_agent(
        monkeypatch,
        lambda call: "COMPLETE\n" if call.runner == ["fake-reviser"] else "review\n",
    )
    refine(_refine_args(max_steps=1, **paths))

    assert agent.prompts == [
        "file review\nfile review extra",
        "file revise\nfile revise extra",
    ]


def test_refine_reviews_again_after_continue_and_keeps_the_review_otherwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``CONTINUE`` discards the review; any other status revises it again."""
    _refine_home(tmp_path, monkeypatch)
    agent = _refine_agent(
        monkeypatch,
        ["CONTINUE\n", "unclear\n", "unclear\n", "CONTINUE\n", "unclear\n"],
    )

    refine(_refine_args(max_steps=5))

    assert _passes(agent) == [
        "review",
        "revise",
        "review",
        "revise",
        "revise",
        "revise",
        "review",
        "revise",
    ]
    revised = _revised_review_files(agent)
    assert revised == [revised[0], revised[1], revised[1], revised[1], revised[4]]
    assert len(set(revised)) == 3
    assert not any(path.exists() for path in set(revised))


def test_refine_runs_a_fresh_review_after_continue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _refine_home(tmp_path, monkeypatch)
    agent = _refine_agent(monkeypatch, ["CONTINUE\n", "COMPLETE\n"])

    refine(
        _refine_args(
            max_steps=5,
            reviewer="reviewer",
            reviser="reviser",
            scope="my scope",
            aspects="my aspects",
        )
    )

    assert agent.runners == ["reviewer", "reviser", "reviewer", "reviser"]
    assert agent.prompts[0] == "review my scope for my aspects\n"
    first, second = _revised_review_files(agent)
    assert first != second


def test_refine_no_save_review_keeps_reviews_out_of_the_working_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _refine_home(tmp_path, monkeypatch)
    agent = _refine_agent(monkeypatch, ["CONTINUE\n", "COMPLETE\n"])

    refine(_refine_args(max_steps=5, save_review=False))

    assert _passes(agent).count("review") == 2
    assert not (tmp_path / ".agent-files").exists()


def test_refine_save_review_writes_every_review_to_agent_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _refine_home(tmp_path, monkeypatch)
    agent = _refine_agent(monkeypatch, ["CONTINUE\n", "COMPLETE\n"])

    refine(_refine_args(max_steps=5, save_review=True))

    assert _passes(agent).count("review") == 2
    saved = sorted((tmp_path / ".agent-files").glob("review-*.md"))
    assert [path.read_text(encoding="utf-8") for path in saved] == [
        "review result\n",
        "review result\n",
    ]


def test_refine_review_file_collects_every_review_in_one_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _refine_home(tmp_path, monkeypatch)
    agent = _refine_agent(monkeypatch, ["CONTINUE\n", "COMPLETE\n"], review_output="findings\n")

    refine(_refine_args(max_steps=5, review_file="reviews/last.md"))

    assert _passes(agent).count("review") == 2
    assert (tmp_path / "reviews" / "last.md").read_text(encoding="utf-8") == "findings\n"
    assert not (tmp_path / ".agent-files").exists()


def test_refine_leaves_missing_scope_and_aspects_for_review_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _refine_home(tmp_path, monkeypatch)
    (home / ".agm" / "config.toml").write_text("[refine.frontend]\n", encoding="utf-8")
    agent = _refine_agent(monkeypatch, ["COMPLETE\n"])

    refine(_refine_args(command_name="frontend"))

    assert agent.prompts[0] == (
        f"review {review_pass.DEFAULT_REVIEW_SCOPE} for {DEFAULT_REVIEW_ASPECTS}\n"
    )


def test_refine_uses_named_config_for_both_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _refine_home(tmp_path, monkeypatch)
    (home / ".agm" / "config.toml").write_text(
        '[refine.frontend]\nrunner = "frontend-runner"\nscope = "frontend scope"\n'
    )
    agent = _refine_agent(monkeypatch, ["COMPLETE\n"])

    refine(_refine_args(command_name="frontend", reviewer=None, reviser=None))

    assert agent.runners == ["frontend-runner", "frontend-runner"]
    assert agent.prompts[0] == f"review frontend scope for {DEFAULT_REVIEW_ASPECTS}\n"


def test_refine_writes_review_and_revise_output_to_log_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _refine_home(tmp_path, monkeypatch)
    log_file = tmp_path / "refine.log"

    def respond(call: AgentCall) -> AgentReply:
        pass_name = "revise" if _is_revise(call) else "review"
        return AgentReply(
            stdout=f"{pass_name} stdout\n",
            stderr=f"{pass_name} stderr\n",
            returncode=0,
        )

    fake_agent(monkeypatch, respond)

    refine(_refine_args(max_steps=1, no_log=False, log_file=str(log_file), save_review=False))

    log_content = log_file.read_text(encoding="utf-8")
    assert "Step 1" in log_content
    assert log_content.endswith("review stdout\nreview stderr\n\nrevise stdout\nrevise stderr\n")


def test_refine_step_header_is_printed_and_logged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _refine_home(tmp_path, monkeypatch)
    log_file = tmp_path / "refine.log"
    fake_agent(
        monkeypatch,
        lambda call: "revise stdout\n" if _is_revise(call) else "review stdout\n",
    )

    refine(_refine_args(max_steps=1, no_log=False, log_file=str(log_file), save_review=False))

    out = capsys.readouterr().out
    log_content = log_file.read_text(encoding="utf-8")
    assert "Step 1" in out
    assert out.index("Step 1") < out.index("review stdout")
    assert "Step 1" in log_content
    assert log_content.index("Step 1") < log_content.index("review stdout")


def test_refine_logs_to_a_default_file_under_agent_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _refine_home(tmp_path, monkeypatch)
    _refine_agent(monkeypatch, ["COMPLETE\n"])

    refine(_refine_args(max_steps=1, no_log=False))

    (log_file,) = sorted((tmp_path / ".agent-files").glob("refine-*.log"))
    out = capsys.readouterr().out
    assert str(Path(".agent-files") / log_file.name) in out.splitlines()[0]
    assert "Step 1" in log_file.read_text(encoding="utf-8")


def test_refine_exits_when_named_config_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = _refine_home(tmp_path, monkeypatch)
    (home / ".agm" / "config.toml").write_text('[refine]\nrunner = "base-runner"\n')
    agent = _refine_agent(monkeypatch)

    with pytest.raises(SystemExit):
        refine(_refine_args(command_name="fronend"))

    assert agent.calls == []
    err = capsys.readouterr().err
    assert "fronend" in err
    assert "refine" in err.lower()


def test_run_wrappers_translate_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _refine_home(tmp_path, monkeypatch)
    review_file = tmp_path / "review.md"
    review_file.write_text("findings\n", encoding="utf-8")

    def interrupt(_call: AgentCall) -> str:
        raise KeyboardInterrupt

    fake_agent(monkeypatch, interrupt)

    with pytest.raises(SystemExit) as review_exit:
        review_mod.run(_review_args(no_review_file=True))
    assert review_exit.value.code == 130

    with pytest.raises(SystemExit) as revise_exit:
        revise_mod.run(_revise_args("review.md"))
    assert revise_exit.value.code == 130

    with pytest.raises(SystemExit) as refine_exit:
        refine_mod.run(_refine_args(max_steps=1))
    assert refine_exit.value.code == 130


def test_refine_no_max_steps_runs_until_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _refine_home(tmp_path, monkeypatch)
    # COMPLETE only after 25 steps — well beyond the default step limit.
    agent = _refine_agent(
        monkeypatch,
        ["CONTINUE\n"] * 10 + ["unclear\n"] * 14 + ["COMPLETE\n"],
        final_status=None,
    )

    refine(_refine_args(no_max_steps=True))

    assert _passes(agent).count("revise") == 25


def test_refine_stops_when_the_review_agent_cannot_be_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _refine_home(tmp_path, monkeypatch)

    def spawn_failure(_call: AgentCall) -> str:
        raise FileNotFoundError(2, "No such file or directory", "fake-reviewer")

    agent = fake_agent(monkeypatch, spawn_failure)

    with pytest.raises(SystemExit) as exc_info:
        refine(_refine_args())

    assert exc_info.value.code == 1
    assert _passes(agent) == ["review"]


def test_refine_stops_when_the_review_runner_is_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit code 127 is a runner-configuration error, not a step to retry."""
    _refine_home(tmp_path, monkeypatch)
    agent = fake_agent(
        monkeypatch,
        lambda _call: AgentReply(stderr="command not found\n", returncode=127),
    )

    with pytest.raises(SystemExit) as exc_info:
        refine(_refine_args())

    assert exc_info.value.code == 1
    assert _passes(agent) == ["review"]


def test_refine_cleans_up_the_review_file_when_revise_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _refine_home(tmp_path, monkeypatch)

    def respond(call: AgentCall) -> AgentReply:
        if _is_revise(call):
            return AgentReply(stderr="command not found\n", returncode=127)
        return AgentReply(stdout="review result\n")

    agent = fake_agent(monkeypatch, respond)

    with pytest.raises(SystemExit) as exc_info:
        refine(_refine_args())

    assert exc_info.value.code == 1
    assert _passes(agent) == ["review", "revise"]
    assert not _revised_review_files(agent)[0].exists()
