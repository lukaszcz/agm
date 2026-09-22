"""Checking and installing Python requirements in AGM's own interpreter environment."""

from __future__ import annotations

import sys
from importlib import metadata

import pytest

from agm.core import dry_run, process
from agm.core.pyenv import (
    RequirementInstallError,
    RequirementState,
    RequirementStatus,
    install_requirements,
    requirement_status,
)

_UV = "/opt/tools/uv"


@pytest.fixture
def commands(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record foreground commands instead of running them; each succeeds."""
    recorded: list[list[str]] = []

    def run_foreground(cmd: list[str], **_kwargs: object) -> int:
        recorded.append(cmd)
        return 0

    monkeypatch.setattr(process, "run_foreground", run_foreground)
    return recorded


def satisfied(spec: str) -> bool:
    return requirement_status(spec).satisfied


def _uv_on_path(monkeypatch: pytest.MonkeyPatch, found: bool) -> None:
    monkeypatch.setattr(
        "agm.core.pyenv.shutil.which", lambda name: _UV if found and name == "uv" else None
    )


class TestInstallRequirements:
    def test_uses_uv_for_the_running_interpreter_when_on_path(
        self, monkeypatch: pytest.MonkeyPatch, commands: list[list[str]]
    ) -> None:
        _uv_on_path(monkeypatch, True)

        install_requirements(("requests>=2", "rich"))

        assert commands == [
            [_UV, "pip", "install", "--python", sys.executable, "requests>=2", "rich"]
        ]

    def test_falls_back_to_pip_of_the_running_interpreter(
        self, monkeypatch: pytest.MonkeyPatch, commands: list[list[str]]
    ) -> None:
        _uv_on_path(monkeypatch, False)

        install_requirements(("requests>=2",))

        assert commands == [[sys.executable, "-m", "pip", "install", "requests>=2"]]

    def test_nothing_to_install_runs_nothing(
        self, monkeypatch: pytest.MonkeyPatch, commands: list[list[str]]
    ) -> None:
        _uv_on_path(monkeypatch, True)

        install_requirements(())

        assert commands == []

    def test_dry_run_prints_the_command_and_runs_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        commands: list[list[str]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _uv_on_path(monkeypatch, True)
        dry_run.set_enabled(True)

        install_requirements(("requests>=2",))

        assert commands == []
        output = capsys.readouterr().out
        assert "dry-run" in output
        assert "requests>=2" in output

    def test_a_failing_installer_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _uv_on_path(monkeypatch, True)
        monkeypatch.setattr(process, "run_foreground", lambda cmd, **_kwargs: 2)

        with pytest.raises(RequirementInstallError):
            install_requirements(("requests>=2",))

    def test_a_failing_pip_fallback_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _uv_on_path(monkeypatch, False)
        monkeypatch.setattr(process, "run_foreground", lambda cmd, **_kwargs: 1)

        with pytest.raises(RequirementInstallError):
            install_requirements(("requests>=2",))

    def test_an_installer_that_cannot_start_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _uv_on_path(monkeypatch, True)

        def missing(cmd: list[str], **_kwargs: object) -> int:
            raise FileNotFoundError(cmd[0])

        monkeypatch.setattr(process, "run_foreground", missing)

        with pytest.raises(RequirementInstallError):
            install_requirements(("requests>=2",))


class TestSatisfied:
    def test_an_installed_distribution_within_the_specifier(self) -> None:
        assert satisfied("packaging>=1")

    def test_an_installed_distribution_outside_the_specifier(self) -> None:
        assert not satisfied("packaging<1")

    def test_an_absent_distribution(self) -> None:
        assert not satisfied("agm-test-absent-distribution")

    def test_a_requirement_whose_marker_does_not_apply(self) -> None:
        assert satisfied("agm-test-absent-distribution; python_version < '3'")

    def test_a_requirement_whose_marker_applies(self) -> None:
        assert not satisfied("agm-test-absent-distribution; python_version >= '3'")

    def test_names_match_under_normalization(self) -> None:
        assert satisfied("Packaging>=1")

    @pytest.mark.parametrize(
        ("installed", "spec", "expected"),
        (
            ("2.0b1", "demo>=1", True),
            ("2.0b1", "demo>=2", False),
            ("not-a-version", "demo>=1", False),
            ("not-a-version", "demo", True),
        ),
    )
    def test_installed_versions_are_matched_like_an_installer_would(
        self, monkeypatch: pytest.MonkeyPatch, installed: str, spec: str, expected: bool
    ) -> None:
        monkeypatch.setattr(metadata, "version", lambda _name: installed)

        assert satisfied(spec) is expected


def _distributions(
    monkeypatch: pytest.MonkeyPatch, installed: dict[str, tuple[str, list[str]]]
) -> None:
    """Replace the environment's distributions with *installed*: name -> (version, requires)."""

    def lookup(name: str) -> tuple[str, list[str]]:
        try:
            return installed[name.lower().replace("_", "-")]
        except KeyError:
            raise metadata.PackageNotFoundError(name) from None

    monkeypatch.setattr(metadata, "version", lambda name: lookup(name)[0])
    monkeypatch.setattr(metadata, "requires", lambda name: lookup(name)[1])


class TestSatisfiedExtras:
    _REQUIRES = [
        "base-dep>=1",
        'cli-dep>=2; extra == "cli"',
        'other-dep; extra == "other"',
        "never-dep; extra == \"cli\" and python_version < '3'",
    ]

    def test_holds_when_every_requirement_of_the_extra_holds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _distributions(monkeypatch, {"demo": ("1.0", self._REQUIRES), "cli-dep": ("2.1", [])})

        assert satisfied("demo[cli]>=1")

    def test_fails_when_a_requirement_of_the_extra_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _distributions(monkeypatch, {"demo": ("1.0", self._REQUIRES), "cli-dep": ("2.1", [])})

        assert not satisfied("demo[cli,other]")

    def test_fails_when_a_requirement_of_the_extra_is_too_old(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _distributions(monkeypatch, {"demo": ("1.0", self._REQUIRES), "cli-dep": ("1.0", [])})

        assert not satisfied("demo[cli]")

    def test_checks_extras_of_extra_requirements(self, monkeypatch: pytest.MonkeyPatch) -> None:
        demo = ["cli-dep[color]; extra == 'cli'"]
        color = ["paint; extra == 'color'"]
        _distributions(monkeypatch, {"demo": ("1.0", demo), "cli-dep": ("2.0", color)})

        assert not satisfied("demo[cli]")

    def test_cyclic_extras_terminate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        demo = ["cli-dep[back]; extra == 'cli'"]
        back = ["demo[cli]; extra == 'back'"]
        _distributions(monkeypatch, {"demo": ("1.0", demo), "cli-dep": ("2.0", back)})

        assert satisfied("demo[cli]")

    def test_a_distribution_without_declared_requirements(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(metadata, "version", lambda _name: "1.0")
        monkeypatch.setattr(metadata, "requires", lambda _name: None)

        assert satisfied("demo[cli]")

    @pytest.mark.parametrize(
        "declared",
        (
            "cli-dep>=2; extra == 'cli'",
            "not a requirement!",
            "cli-dep; extra == 'cli' and os_name ~= 'posix'",
            "cli-dep; 'cli' in extras",
        ),
    )
    def test_an_unreadable_declared_requirement_is_left_to_the_installer(
        self, monkeypatch: pytest.MonkeyPatch, declared: str
    ) -> None:
        _distributions(monkeypatch, {"demo": ("1.0", [declared]), "cli-dep": ("2.1", [])})

        assert satisfied("demo[cli]") is (declared == "cli-dep>=2; extra == 'cli'")

    def test_a_requirement_holding_without_the_extra_is_not_the_extras(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        declared = ["absent-dep; extra == 'cli' or python_version >= '3'"]
        _distributions(monkeypatch, {"demo": ("1.0", declared)})

        assert satisfied("demo[cli]")


class TestRequirementStatus:
    def test_an_installed_distribution_within_the_specifier(self) -> None:
        assert requirement_status("packaging>=1") == RequirementStatus(
            RequirementState.INSTALLED, metadata.version("packaging")
        )

    def test_an_installed_distribution_outside_the_specifier(self) -> None:
        status = requirement_status("packaging<1")

        assert status == RequirementStatus(
            RequirementState.UNSATISFIED, metadata.version("packaging")
        )
        assert not status.satisfied

    def test_an_absent_distribution(self) -> None:
        status = requirement_status("agm-test-absent-distribution>=1")

        assert status == RequirementStatus(RequirementState.MISSING)
        assert not status.satisfied

    def test_a_requirement_whose_marker_does_not_apply(self) -> None:
        status = requirement_status("packaging<1; python_version < '3'")

        assert status == RequirementStatus(RequirementState.NOT_APPLICABLE)
        assert status.satisfied
