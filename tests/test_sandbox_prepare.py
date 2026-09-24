"""Tests for the sandbox preparation library: prepare, profile, and the SRT backend."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from agm.config.general import CommandSetting, RunConfig
from agm.core import dry_run
from agm.sandbox import srt as srt_module
from agm.sandbox.backend import (
    ResolvedSettings,
    SandboxSettingsError,
    SandboxUnavailableError,
)
from agm.sandbox.prepare import (
    DEFAULT_MEMORY_LIMIT,
    DEFAULT_SWAP_LIMIT,
    ResolvedLimits,
    _resolve_raw_limit,
    _validate_limit,
    dry_run_argv,
    prepare,
    print_dry_run,
    resolve_limits,
    settings_source,
)
from agm.sandbox.profile import profile_name, profile_name_for_shell
from agm.sandbox.request import (
    DRY_RUN_SETTINGS_PLACEHOLDER,
    Default,
    PreparedSandboxCommand,
    SandboxLimits,
    SandboxRequest,
    SandboxSpec,
)
from agm.sandbox.srt import SRT_BACKEND


def _run_config(
    *,
    aliases: dict[str, str] | None = None,
    memory_limit: str | None = None,
    swap_limit: str | None = None,
    command_memory_limits: dict[str, str] | None = None,
    command_swap_limits: dict[str, str] | None = None,
    pty: bool = True,
    command_ptys: dict[str, bool] | None = None,
) -> RunConfig:
    return RunConfig(
        alias=CommandSetting(default=None, overrides=aliases or {}),
        memory=CommandSetting(default=memory_limit, overrides=command_memory_limits or {}),
        swap=CommandSetting(default=swap_limit, overrides=command_swap_limits or {}),
        pty=CommandSetting(default=pty, overrides=command_ptys or {}),
    )


def _request(
    *,
    tmp_path: Path,
    profile: str | None = "echo",
    memory: object = Default,
    swap: object = Default,
    settings_file: Path | None = None,
    patch_proj_dir: bool = True,
    alias_name: str | None = None,
    pty: bool = False,
    sandboxed: bool = True,
    proj_dir: Path | None = None,
    config_cwd: Path | None = None,
) -> SandboxRequest:
    home = tmp_path / "home"
    cwd = tmp_path / "work"
    home.mkdir(parents=True, exist_ok=True)
    cwd.mkdir(parents=True, exist_ok=True)
    return SandboxRequest(
        command=["echo", "hi"],
        cwd=cwd,
        config_cwd=cwd if config_cwd is None else config_cwd,
        env={"HOME": str(home), "PATH": "/bin"},
        home=home,
        proj_dir=proj_dir,
        spec=SandboxSpec(
            profile_name=profile,
            memory=memory,
            swap=swap,
            settings_file=settings_file,
            patch=patch_proj_dir,
        ),
        alias_name=alias_name,
        pty=pty,
        sandboxed=sandboxed,
    )


def _write_default_settings(home: Path, data: dict[str, Any] | None = None) -> Path:
    settings_dir = home / ".agm" / "sandbox"
    settings_dir.mkdir(parents=True, exist_ok=True)
    path = settings_dir / "default.json"
    path.write_text(json.dumps(data or {}), encoding="utf-8")
    return path


def _write_two_candidates(
    tmp_path: Path, name: str, *, home_data: dict[str, Any], cwd_data: dict[str, Any]
) -> tuple[Path, Path]:
    """Write matching `<name>.json` candidates under the home and cwd sandbox scopes."""

    home = tmp_path / "home"
    cwd = tmp_path / "work"
    home.mkdir(parents=True, exist_ok=True)
    cwd.mkdir(parents=True, exist_ok=True)
    home_sandbox = home / ".agm" / "sandbox"
    home_sandbox.mkdir(parents=True, exist_ok=True)
    cwd_sandbox = cwd / ".sandbox"
    cwd_sandbox.mkdir(exist_ok=True)
    (home_sandbox / f"{name}.json").write_text(json.dumps(home_data), encoding="utf-8")
    (cwd_sandbox / f"{name}.json").write_text(json.dumps(cwd_data), encoding="utf-8")
    return home, cwd


@pytest.fixture
def home_with_default_settings(tmp_path: Path) -> Iterator[Path]:
    """*tmp_path* with a default sandbox settings file and `shutil.which` mocked to succeed."""

    _write_default_settings(tmp_path / "home")
    with patch("shutil.which", return_value="/usr/bin/tool"):
        yield tmp_path


# ---------------------------------------------------------------------------
# profile_name
# ---------------------------------------------------------------------------


class TestProfileName:
    def test_plain_name(self) -> None:
        assert profile_name("claude") == "claude"

    def test_path_qualified_name(self) -> None:
        assert profile_name("/usr/bin/claude") == "claude"

    def test_empty_argv0_returns_itself(self) -> None:
        assert profile_name("") == ""


class TestProfileNameForShell:
    """``profile_name_for_shell`` derives exec's profile from the first shell word."""

    def test_plain_command(self) -> None:
        assert profile_name_for_shell("make test") == "make"

    def test_path_qualified_command(self) -> None:
        assert profile_name_for_shell("/usr/bin/make test") == "make"

    def test_quoted_first_word(self) -> None:
        assert profile_name_for_shell("'my tool' --x") == "my tool"

    def test_unsplittable_command_selects_no_name(self) -> None:
        assert profile_name_for_shell("echo 'oops") is None

    def test_empty_command_selects_no_name(self) -> None:
        assert profile_name_for_shell("") is None

    def test_assignment_prefix_is_an_unknown_name(self) -> None:
        """``VAR=1 make`` is not special-cased: its first word is the profile name."""
        assert profile_name_for_shell("VAR=1 make") == "VAR=1"

    def test_leading_paren_is_an_unknown_name(self) -> None:
        assert profile_name_for_shell("(cd sub && make)") == "(cd"


class TestSandboxLimits:
    """``SandboxLimits`` carries the shape statable without naming a command."""

    def test_a_spec_is_limits_plus_the_profile_the_limits_apply_to(self) -> None:
        spec = SandboxSpec(
            profile_name="claude",
            memory="8G",
            swap="1G",
            settings_file=Path("/tmp/x"),
            patch=False,
        )

        assert isinstance(spec, SandboxLimits)
        assert (spec.memory, spec.swap, spec.settings_file, spec.patch) == (
            "8G",
            "1G",
            Path("/tmp/x"),
            False,
        )

    def test_limits_and_spec_share_their_defaults(self) -> None:
        limits = SandboxLimits()
        spec = SandboxSpec(profile_name=None)

        for field_name in ("memory", "swap", "settings_file", "patch"):
            assert getattr(limits, field_name) == getattr(spec, field_name)

    def test_a_spec_is_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            setattr(SandboxSpec(profile_name="echo"), "profile_name", "other")


class TestSandboxLimitsForCommand:
    """``for_command`` binds profile-independent limits to one command's spec."""

    def test_binds_the_given_profile_name(self) -> None:
        limits = SandboxLimits()

        spec = limits.for_command("claude")

        assert spec == SandboxSpec(profile_name="claude")

    def test_preserves_every_other_field(self) -> None:
        limits = SandboxLimits(memory="8G", swap="1G", settings_file=Path("/tmp/x"), patch=False)

        spec = limits.for_command("codex")

        assert spec == SandboxSpec(
            profile_name="codex", memory="8G", swap="1G", settings_file=Path("/tmp/x"), patch=False
        )

    def test_a_none_profile_name_selects_no_name(self) -> None:
        """An unknown/unsplittable command falls through to the unqualified defaults."""
        spec = SandboxLimits().for_command(None)

        assert spec.profile_name is None

    def test_default_limits_stay_the_default_sentinel(self) -> None:
        spec = SandboxLimits().for_command("claude")

        assert spec.memory is Default
        assert spec.swap is Default


# ---------------------------------------------------------------------------
# prepare(): argv composition
# ---------------------------------------------------------------------------


class TestPrepareArgvComposition:
    def test_limits_prefix_plus_srt_wrapper_plus_command(
        self, home_with_default_settings: Path
    ) -> None:
        request = _request(tmp_path=home_with_default_settings)
        run_config = _run_config()
        prepared = prepare(request, run_config=run_config)
        try:
            assert prepared.argv[:4] == ["systemd-run", "--user", "--scope", "-q"]
            assert prepared.settings_path is not None
            srt_index = prepared.argv.index("srt")
            assert prepared.argv[srt_index : srt_index + 4] == [
                "srt",
                "--settings",
                str(prepared.settings_path),
                "--",
            ]
            assert prepared.argv[-3:] == ["--", "echo", "hi"]
            assert prepared.interrupt_cleanup_cmd is not None
            assert prepared.interrupt_cleanup_cmd[:3] == ["systemctl", "--user", "--no-block"]
        finally:
            prepared.close()

    def test_sandboxed_false_yields_limits_prefix_only(self, tmp_path: Path) -> None:
        request = _request(tmp_path=tmp_path, sandboxed=False)
        run_config = _run_config()
        with patch("shutil.which", return_value="/usr/bin/systemd-run"):
            prepared = prepare(request, run_config=run_config)
        try:
            assert prepared.argv[:4] == ["systemd-run", "--user", "--scope", "-q"]
            assert "srt" not in prepared.argv
            assert prepared.argv[-2:] == ["echo", "hi"]
        finally:
            prepared.close()

    def test_pty_wrapper_inside_backend_wrapper(self, home_with_default_settings: Path) -> None:
        request = _request(tmp_path=home_with_default_settings, pty=True)
        run_config = _run_config()
        prepared = prepare(request, run_config=run_config)
        try:
            srt_index = prepared.argv.index("srt")
            assert prepared.argv[srt_index : srt_index + 4] == [
                "srt",
                "--settings",
                str(prepared.settings_path),
                "--",
            ]
            assert prepared.argv[srt_index + 4 : srt_index + 8] == [
                sys.executable,
                "-m",
                "agm.sandbox.pty",
                "--",
            ]
            assert prepared.argv[-2:] == ["echo", "hi"]
        finally:
            prepared.close()

    def test_no_limits_at_all_means_no_prefix_and_no_cleanup_cmd(
        self, home_with_default_settings: Path
    ) -> None:
        request = _request(tmp_path=home_with_default_settings, memory=None, swap=None)
        run_config = _run_config()
        prepared = prepare(request, run_config=run_config)
        try:
            assert prepared.argv[:1] == ["srt"]
            assert prepared.interrupt_cleanup_cmd is None
        finally:
            prepared.close()


# ---------------------------------------------------------------------------
# prepare(): resource limit resolution
# ---------------------------------------------------------------------------


class TestLimitResolution:
    def test_default_uses_per_command_run_config_limit(
        self, home_with_default_settings: Path
    ) -> None:
        request = _request(tmp_path=home_with_default_settings)
        run_config = _run_config(command_memory_limits={"echo": "4G"})
        prepared = prepare(request, run_config=run_config)
        try:
            assert "MemoryMax=4G" in " ".join(prepared.argv)
        finally:
            prepared.close()

    def test_default_falls_back_to_unqualified_run_config_limit(
        self, home_with_default_settings: Path
    ) -> None:
        request = _request(tmp_path=home_with_default_settings)
        run_config = _run_config(memory_limit="8G")
        prepared = prepare(request, run_config=run_config)
        try:
            assert "MemoryMax=8G" in " ".join(prepared.argv)
        finally:
            prepared.close()

    def test_default_falls_back_to_builtin_default(self, home_with_default_settings: Path) -> None:
        request = _request(tmp_path=home_with_default_settings)
        run_config = _run_config()
        prepared = prepare(request, run_config=run_config)
        try:
            joined = " ".join(prepared.argv)
            assert "MemoryMax=32G" in joined
            assert "MemorySwapMax=0" in joined
        finally:
            prepared.close()

    def test_none_omits_the_limit_property(self, home_with_default_settings: Path) -> None:
        request = _request(tmp_path=home_with_default_settings, memory=None)
        run_config = _run_config(swap_limit="1G")
        prepared = prepare(request, run_config=run_config)
        try:
            joined = " ".join(prepared.argv)
            assert "MemoryMax" not in joined
            assert "MemorySwapMax=1G" in joined
        finally:
            prepared.close()

    def test_explicit_string_is_used_directly(self, home_with_default_settings: Path) -> None:
        request = _request(tmp_path=home_with_default_settings, memory="2G")
        run_config = _run_config(memory_limit="8G")
        prepared = prepare(request, run_config=run_config)
        try:
            assert "MemoryMax=2G" in " ".join(prepared.argv)
        finally:
            prepared.close()

    def test_unlimited_normalizes_to_infinity(self, home_with_default_settings: Path) -> None:
        request = _request(tmp_path=home_with_default_settings, memory="unlimited")
        run_config = _run_config()
        prepared = prepare(request, run_config=run_config)
        try:
            assert "MemoryMax=infinity" in " ".join(prepared.argv)
        finally:
            prepared.close()

    def test_invalid_limit_raises_sandbox_settings_error(self, tmp_path: Path) -> None:
        request = _request(tmp_path=tmp_path, memory="not-a-limit")
        run_config = _run_config()
        with patch("shutil.which", return_value="/usr/bin/tool"):
            with pytest.raises(SandboxSettingsError):
                prepare(request, run_config=run_config)


class TestLimitResolutionWithoutProfile:
    def test_default_memory_and_swap_use_unqualified_run_config(
        self, home_with_default_settings: Path
    ) -> None:
        request = _request(tmp_path=home_with_default_settings, profile=None, patch_proj_dir=False)
        run_config = _run_config(memory_limit="6G", swap_limit="2G")
        prepared = prepare(request, run_config=run_config)
        try:
            joined = " ".join(prepared.argv)
            assert "MemoryMax=6G" in joined
            assert "MemorySwapMax=2G" in joined
        finally:
            prepared.close()

    def test_memory_set_and_swap_none_omits_swap_flag(
        self, home_with_default_settings: Path
    ) -> None:
        request = _request(tmp_path=home_with_default_settings, memory=Default, swap=None)
        run_config = _run_config()
        prepared = prepare(request, run_config=run_config)
        try:
            joined = " ".join(prepared.argv)
            assert "MemoryMax=32G" in joined
            assert "MemorySwapMax" not in joined
        finally:
            prepared.close()


# ---------------------------------------------------------------------------
# _validate_limit: systemd resource-limit grammar (systemd.resource-control(5))
# ---------------------------------------------------------------------------


class TestValidateLimitGrammar:
    @pytest.mark.parametrize(
        "value",
        [
            "infinity",
            "unlimited",
            "UNLIMITED",
            "Unlimited",
            "80%",
            "80.5%",
            "10B",
            "500M",
            "4G 500M",
            "2T",
            "5G",
        ],
    )
    def test_accepts(self, value: str) -> None:
        _validate_limit(value)  # must not raise

    @pytest.mark.parametrize(
        "value",
        [
            "4g",  # lowercase size suffix: systemd suffixes are case-sensitive
            "INFINITY",  # only the exact-case literal `infinity` is accepted
            "5Ki",  # binary-prefix suffixes are not systemd's grammar
            "5GiB",
            "not-a-limit",
            "",
            "5X",
        ],
    )
    def test_rejects(self, value: str) -> None:
        with pytest.raises(SandboxSettingsError):
            _validate_limit(value)

    def test_unlimited_is_case_insensitive_and_normalizes_to_infinity(self) -> None:
        assert _validate_limit("UNLIMITED") == "infinity"

    def test_surrounding_whitespace_is_trimmed(self) -> None:
        assert _validate_limit("  5G  ") == "5G"


# ---------------------------------------------------------------------------
# _resolve_raw_limit(): the sentinel chain, before validation
# ---------------------------------------------------------------------------


class TestResolveRawLimit:
    """The pre-validation resolution a dry-run caller displays (`agm run --dry-run`)."""

    def test_none_stays_none(self) -> None:
        assert _resolve_raw_limit(None, configured="8G", default="32G") is None

    def test_default_uses_configured_when_present(self) -> None:
        assert _resolve_raw_limit(Default, configured="8G", default="32G") == "8G"

    def test_default_falls_back_to_built_in_default(self) -> None:
        assert _resolve_raw_limit(Default, configured=None, default="32G") == "32G"

    def test_explicit_string_is_returned_verbatim_unnormalized(self) -> None:
        assert _resolve_raw_limit(" unlimited ", configured="8G", default="32G") == " unlimited "


# ---------------------------------------------------------------------------
# resolve_limits(): the single public limit-resolution entry point
# ---------------------------------------------------------------------------


class TestResolveLimits:
    """`prepare()` and `agm run`'s dry-run detail lines share this one function."""

    def test_returns_raw_and_validated_values(self) -> None:
        spec = SandboxSpec(profile_name="echo", memory=Default, swap="unlimited")
        run_config = _run_config(command_memory_limits={"echo": "4G"})
        limits = resolve_limits(spec, run_config)
        assert limits == ResolvedLimits(
            memory_raw="4G", swap_raw="unlimited", memory="4G", swap="infinity"
        )

    def test_falls_back_to_builtin_defaults(self) -> None:
        spec = SandboxSpec(profile_name="echo")
        limits = resolve_limits(spec, _run_config())
        assert limits.memory_raw == DEFAULT_MEMORY_LIMIT
        assert limits.swap_raw == DEFAULT_SWAP_LIMIT
        assert limits.memory == DEFAULT_MEMORY_LIMIT
        assert limits.swap == DEFAULT_SWAP_LIMIT

    def test_none_limits_resolve_to_none_throughout(self) -> None:
        spec = SandboxSpec(profile_name="echo", memory=None, swap=None)
        limits = resolve_limits(spec, _run_config())
        assert limits == ResolvedLimits(memory_raw=None, swap_raw=None, memory=None, swap=None)

    def test_invalid_memory_limit_raises_sandbox_settings_error(self) -> None:
        spec = SandboxSpec(profile_name="echo", memory="not-a-limit")
        with pytest.raises(SandboxSettingsError):
            resolve_limits(spec, _run_config())

    def test_invalid_swap_limit_raises_sandbox_settings_error(self) -> None:
        spec = SandboxSpec(profile_name="echo", swap="not-a-limit")
        with pytest.raises(SandboxSettingsError):
            resolve_limits(spec, _run_config())


# ---------------------------------------------------------------------------
# prepare(): settings resolution (via the SRT backend)
# ---------------------------------------------------------------------------


class TestSettingsResolution:
    def test_settings_file_override_replaces_the_chain(self, tmp_path: Path) -> None:
        explicit = tmp_path / "custom.json"
        explicit.write_text(json.dumps({"network": {"allowedDomains": ["x.test"]}}))
        request = _request(tmp_path=tmp_path, settings_file=explicit, patch_proj_dir=False)
        run_config = _run_config()
        with patch("shutil.which", return_value="/usr/bin/tool"):
            prepared = prepare(request, run_config=run_config)
        try:
            assert prepared.settings_path == explicit
        finally:
            prepared.close()

    def test_missing_explicit_settings_file_raises(self, tmp_path: Path) -> None:
        request = _request(
            tmp_path=tmp_path, settings_file=tmp_path / "missing.json", patch_proj_dir=False
        )
        run_config = _run_config()
        with patch("shutil.which", return_value="/usr/bin/tool"):
            with pytest.raises(SandboxSettingsError) as excinfo:
                prepare(request, run_config=run_config)
        assert excinfo.value.path == tmp_path / "missing.json"

    def test_no_candidate_settings_raises(self, tmp_path: Path) -> None:
        request = _request(tmp_path=tmp_path, patch_proj_dir=False)
        run_config = _run_config()
        with patch("shutil.which", return_value="/usr/bin/tool"):
            with pytest.raises(SandboxSettingsError) as excinfo:
                prepare(request, run_config=run_config)
        assert len(excinfo.value.candidates) > 0

    def test_multiple_candidates_merge_into_a_temp_file(self, tmp_path: Path) -> None:
        _write_two_candidates(
            tmp_path,
            "echo",
            home_data={"network": {"allowedDomains": ["a.test"]}},
            cwd_data={"enabled": False},
        )

        request = _request(tmp_path=tmp_path, patch_proj_dir=False)
        run_config = _run_config()
        with patch("shutil.which", return_value="/usr/bin/tool"):
            prepared = prepare(request, run_config=run_config)
        try:
            assert prepared.settings_path is not None
            merged = json.loads(prepared.settings_path.read_text())
            assert merged["network"]["allowedDomains"] == ["a.test"]
            assert merged["enabled"] is False
        finally:
            prepared.close()
            assert not prepared.settings_path.exists()

    def test_patch_false_skips_proj_dir_patch(self, home_with_default_settings: Path) -> None:
        proj_dir = home_with_default_settings / "project"
        (proj_dir / "repo").mkdir(parents=True)
        request = _request(
            tmp_path=home_with_default_settings, proj_dir=proj_dir, patch_proj_dir=False
        )
        run_config = _run_config()
        prepared = prepare(request, run_config=run_config)
        try:
            assert prepared.settings_path is not None
            data = json.loads(prepared.settings_path.read_text())
            assert "filesystem" not in data
        finally:
            prepared.close()

    def test_patch_true_with_proj_dir_patches_write_access(
        self, home_with_default_settings: Path
    ) -> None:
        proj_dir = home_with_default_settings / "project"
        (proj_dir / "repo").mkdir(parents=True)
        request = _request(
            tmp_path=home_with_default_settings, proj_dir=proj_dir, patch_proj_dir=True
        )
        run_config = _run_config()
        prepared = prepare(request, run_config=run_config)
        try:
            assert prepared.settings_path is not None
            data = json.loads(prepared.settings_path.read_text())
            allow_write = data["filesystem"]["allowWrite"]
            assert str(proj_dir / "repo" / ".git") in allow_write
        finally:
            prepared.close()


class TestLoadSettingsFailure:
    """A malformed or unreadable settings file raises SandboxSettingsError."""

    def test_malformed_settings_file_in_merge_raises_sandbox_settings_error(
        self, tmp_path: Path
    ) -> None:
        home, cwd = _write_two_candidates(
            tmp_path, "echo", home_data={"enabled": True}, cwd_data={"enabled": False}
        )
        (home / ".agm" / "sandbox" / "echo.json").write_text("{not valid json", encoding="utf-8")

        request = _request(tmp_path=tmp_path, patch_proj_dir=False)
        with pytest.raises(SandboxSettingsError):
            SRT_BACKEND.resolve_settings(request)

    def test_malformed_single_candidate_settings_file_raises_sandbox_settings_error(
        self, tmp_path: Path
    ) -> None:
        """A single malformed candidate must not escape as a raw `json.JSONDecodeError`:
        `resolve_settings` only ever reads the file once, through the guarded loader,
        including the read `track_bwrap_artifacts` used to need."""

        home = tmp_path / "home"
        (home / ".agm" / "sandbox").mkdir(parents=True)
        (home / ".agm" / "sandbox" / "default.json").write_text("{not valid json", encoding="utf-8")

        request = _request(tmp_path=tmp_path, profile=None, patch_proj_dir=False)
        with pytest.raises(SandboxSettingsError):
            SRT_BACKEND.resolve_settings(request)

    def test_load_failure_during_patch_step_raises_sandbox_settings_error(
        self, home_with_default_settings: Path
    ) -> None:
        proj_dir = home_with_default_settings / "project"
        (proj_dir / "repo").mkdir(parents=True)
        request = _request(
            tmp_path=home_with_default_settings, proj_dir=proj_dir, patch_proj_dir=True
        )
        with patch.object(srt_module, "load_settings", side_effect=OSError("boom")):
            with pytest.raises(SandboxSettingsError):
                SRT_BACKEND.resolve_settings(request)


class TestWriteSettingsFailure:
    """A merge/patch temp-settings write failure raises SandboxSettingsError."""

    def test_temp_settings_write_failure_raises_sandbox_settings_error(
        self, home_with_default_settings: Path
    ) -> None:
        proj_dir = home_with_default_settings / "project"
        (proj_dir / "repo").mkdir(parents=True)
        request = _request(
            tmp_path=home_with_default_settings, proj_dir=proj_dir, patch_proj_dir=True
        )
        with patch.object(srt_module, "_write_json_temp", side_effect=OSError("disk full")):
            with pytest.raises(SandboxSettingsError, match="failed to write sandbox settings"):
                SRT_BACKEND.resolve_settings(request)


class TestResolveSettingsFailureCleanup:
    """A failure after a merge temp file is written must not leave it behind."""

    def test_merge_temp_file_removed_when_resolution_fails_afterwards(self, tmp_path: Path) -> None:
        _write_two_candidates(
            tmp_path, "echo", home_data={"enabled": True}, cwd_data={"enabled": False}
        )
        request = _request(tmp_path=tmp_path, patch_proj_dir=False)

        written: list[Path] = []
        real_write_json_temp = srt_module._write_json_temp

        def _tracking_write_json_temp(data: Any, temp_files: list[Path]) -> Path:
            path = real_write_json_temp(data, temp_files)
            written.append(path)
            return path

        with patch.object(srt_module, "_write_json_temp", side_effect=_tracking_write_json_temp):
            with patch.object(
                srt_module, "track_bwrap_artifacts", side_effect=RuntimeError("boom")
            ):
                with pytest.raises(RuntimeError):
                    SRT_BACKEND.resolve_settings(request)

        assert written
        assert not written[0].exists()


# ---------------------------------------------------------------------------
# close(): cleanup
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_removes_temp_settings_and_empty_artifacts(self, tmp_path: Path) -> None:
        _write_two_candidates(
            tmp_path, "echo", home_data={"enabled": True}, cwd_data={"enabled": False}
        )

        request = _request(tmp_path=tmp_path, patch_proj_dir=False)
        run_config = _run_config()
        with patch("shutil.which", return_value="/usr/bin/tool"):
            prepared = prepare(request, run_config=run_config)
        settings_path = prepared.settings_path
        assert settings_path is not None
        assert settings_path.exists()
        prepared.close()
        assert not settings_path.exists()

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        """Two candidates merge into a real temp settings file, so a second close()
        proves idempotency by trying to delete it twice rather than deleting nothing."""

        _write_two_candidates(
            tmp_path, "echo", home_data={"enabled": True}, cwd_data={"enabled": False}
        )
        request = _request(tmp_path=tmp_path, patch_proj_dir=False)
        run_config = _run_config()
        with patch("shutil.which", return_value="/usr/bin/tool"):
            prepared = prepare(request, run_config=run_config)
        settings_path = prepared.settings_path
        assert settings_path is not None
        prepared.close()
        assert not settings_path.exists()
        prepared.close()  # must not raise

    def test_close_tolerates_a_temp_settings_file_removed_out_of_band(self, tmp_path: Path) -> None:
        _write_two_candidates(
            tmp_path, "echo", home_data={"enabled": True}, cwd_data={"enabled": False}
        )
        request = _request(tmp_path=tmp_path, patch_proj_dir=False)
        run_config = _run_config()
        with patch("shutil.which", return_value="/usr/bin/tool"):
            prepared = prepare(request, run_config=run_config)
        assert prepared.settings_path is not None
        prepared.settings_path.unlink()  # simulate something already removing it
        prepared.close()  # must not raise

    def test_close_under_dry_run_removes_temp_file_and_prints_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_two_candidates(
            tmp_path, "echo", home_data={"enabled": True}, cwd_data={"enabled": False}
        )
        request = _request(tmp_path=tmp_path, patch_proj_dir=False)
        run_config = _run_config()
        with patch("shutil.which", return_value="/usr/bin/tool"):
            prepared = prepare(request, run_config=run_config)
        settings_path = prepared.settings_path
        assert settings_path is not None

        dry_run.set_enabled(True)
        try:
            prepared.close()
        finally:
            dry_run.set_enabled(False)

        captured = capsys.readouterr()
        assert not settings_path.exists()
        assert captured.out == ""
        assert captured.err == ""


class TestCloseArtifactCleanup:
    """close() removes tracked bwrap artifacts through public prepare()/close() calls."""

    def _prepare_with_deny_write(
        self, tmp_path: Path, deny_write: list[str]
    ) -> PreparedSandboxCommand:
        home = tmp_path / "home"
        cwd = tmp_path / "work"
        home.mkdir(parents=True, exist_ok=True)
        cwd.mkdir(parents=True, exist_ok=True)
        _write_default_settings(home, {"filesystem": {"denyWrite": deny_write}})
        request = _request(tmp_path=tmp_path, profile=None, patch_proj_dir=False)
        run_config = _run_config()
        with patch("shutil.which", return_value="/usr/bin/tool"):
            return prepare(request, run_config=run_config)

    def test_close_removes_empty_tracked_file(self, tmp_path: Path) -> None:
        target = tmp_path / "work" / "scratch"
        prepared = self._prepare_with_deny_write(tmp_path, [str(target)])
        target.touch()
        prepared.close()
        assert not target.exists()

    def test_close_keeps_non_empty_tracked_file(self, tmp_path: Path) -> None:
        target = tmp_path / "work" / "scratch"
        prepared = self._prepare_with_deny_write(tmp_path, [str(target)])
        target.write_text("content", encoding="utf-8")
        prepared.close()
        assert target.exists()

    def test_close_removes_empty_tracked_dir(self, tmp_path: Path) -> None:
        target_dir = tmp_path / "work" / "scratchdir"
        prepared = self._prepare_with_deny_write(tmp_path, [str(target_dir / "inner")])
        target_dir.mkdir()
        prepared.close()
        assert not target_dir.exists()

    def test_close_keeps_non_empty_tracked_dir(self, tmp_path: Path) -> None:
        target_dir = tmp_path / "work" / "scratchdir"
        prepared = self._prepare_with_deny_write(tmp_path, [str(target_dir / "inner")])
        target_dir.mkdir()
        (target_dir / "child.txt").write_text("x", encoding="utf-8")
        prepared.close()
        assert target_dir.exists()


# ---------------------------------------------------------------------------
# Backend availability errors
# ---------------------------------------------------------------------------


class TestBackendAvailability:
    def test_missing_srt_raises_sandbox_unavailable_error(self, tmp_path: Path) -> None:
        _write_default_settings(tmp_path / "home")
        request = _request(tmp_path=tmp_path, memory=None, swap=None)
        run_config = _run_config()

        def fake_which(name: str, path: str | None = None) -> str | None:
            return None if name == "srt" else "/usr/bin/" + name

        with patch("shutil.which", side_effect=fake_which):
            with pytest.raises(SandboxUnavailableError):
                prepare(request, run_config=run_config)

    def test_missing_systemd_run_raises_sandbox_unavailable_error(
        self, home_with_default_settings: Path
    ) -> None:
        request = _request(tmp_path=home_with_default_settings)
        run_config = _run_config()

        def fake_which(name: str, path: str | None = None) -> str | None:
            return None if name == "systemd-run" else "/usr/bin/" + name

        with patch("shutil.which", side_effect=fake_which):
            with pytest.raises(SandboxUnavailableError):
                prepare(request, run_config=run_config)

    def test_availability_errors_never_print_or_exit(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_default_settings(tmp_path / "home")
        request = _request(tmp_path=tmp_path, memory=None, swap=None)
        run_config = _run_config()

        with patch("shutil.which", return_value=None):
            with pytest.raises(SandboxUnavailableError):
                prepare(request, run_config=run_config)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


# ---------------------------------------------------------------------------
# NODE_USE_ENV_PROXY
# ---------------------------------------------------------------------------


class TestNodeUseEnvProxy:
    def test_defaulted_when_absent(self, home_with_default_settings: Path) -> None:
        request = _request(tmp_path=home_with_default_settings)
        run_config = _run_config()
        prepared = prepare(request, run_config=run_config)
        try:
            assert prepared.env["NODE_USE_ENV_PROXY"] == "1"
        finally:
            prepared.close()

    def test_not_overwritten_when_present(self, home_with_default_settings: Path) -> None:
        home = home_with_default_settings / "home"
        cwd = home_with_default_settings / "work"
        request = SandboxRequest(
            command=["echo", "hi"],
            cwd=cwd,
            config_cwd=cwd,
            env={"HOME": str(home), "PATH": "/bin", "NODE_USE_ENV_PROXY": "0"},
            home=home,
            proj_dir=None,
            spec=SandboxSpec(profile_name="echo"),
        )
        run_config = _run_config()
        prepared = prepare(request, run_config=run_config)
        try:
            assert prepared.env["NODE_USE_ENV_PROXY"] == "0"
        finally:
            prepared.close()

    def test_not_set_when_not_sandboxed(self, tmp_path: Path) -> None:
        request = _request(tmp_path=tmp_path, sandboxed=False, memory=None, swap=None)
        run_config = _run_config()
        with patch("shutil.which", return_value="/usr/bin/tool"):
            prepared = prepare(request, run_config=run_config)
        try:
            assert "NODE_USE_ENV_PROXY" not in prepared.env
        finally:
            prepared.close()


# ---------------------------------------------------------------------------
# dry_run_argv / print_dry_run
# ---------------------------------------------------------------------------


class TestDryRunArgv:
    def test_redacts_settings_path(self, home_with_default_settings: Path) -> None:
        request = _request(tmp_path=home_with_default_settings)
        run_config = _run_config()
        prepared = prepare(request, run_config=run_config)
        try:
            redacted = dry_run_argv(prepared)
            assert str(prepared.settings_path) not in redacted
            assert DRY_RUN_SETTINGS_PLACEHOLDER in redacted
        finally:
            prepared.close()

    def test_no_settings_path_leaves_argv_unchanged(self, tmp_path: Path) -> None:
        request = _request(tmp_path=tmp_path, sandboxed=False, memory=None, swap=None)
        run_config = _run_config()
        with patch("shutil.which", return_value="/usr/bin/tool"):
            prepared = prepare(request, run_config=run_config)
        try:
            assert dry_run_argv(prepared) == prepared.argv
        finally:
            prepared.close()


class TestPrintDryRun:
    def test_prints_labeled_command_with_redacted_settings(
        self, home_with_default_settings: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        request = _request(tmp_path=home_with_default_settings)
        run_config = _run_config()
        prepared = prepare(request, run_config=run_config)
        try:
            print_dry_run(prepared)
            captured = capsys.readouterr()
            assert "sandbox" in captured.out
            assert DRY_RUN_SETTINGS_PLACEHOLDER in captured.out
            assert str(prepared.settings_path) not in captured.out
        finally:
            prepared.close()


# ---------------------------------------------------------------------------
# prepare(resolve_settings=False): dry-run mode with no filesystem resolution
# ---------------------------------------------------------------------------


class TestPrepareNoResolveMode:
    def test_argv_carries_the_placeholder_with_no_filesystem_resolution(
        self, tmp_path: Path
    ) -> None:
        request = _request(tmp_path=tmp_path, memory=None, swap=None)
        run_config = _run_config()
        with patch("shutil.which", return_value="/usr/bin/tool"):
            prepared = prepare(request, run_config=run_config, resolve_settings=False)
        try:
            assert prepared.argv == [
                "srt",
                "--settings",
                DRY_RUN_SETTINGS_PLACEHOLDER,
                "--",
                "echo",
                "hi",
            ]
            assert prepared.settings_path is None
            # No settings scan or merge ever touched the filesystem.
            assert not (tmp_path / "home" / ".agm").exists()
        finally:
            prepared.close()

    def test_dry_run_argv_and_print_dry_run_still_work(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        request = _request(tmp_path=tmp_path, memory=None, swap=None)
        run_config = _run_config()
        with patch("shutil.which", return_value="/usr/bin/tool"):
            prepared = prepare(request, run_config=run_config, resolve_settings=False)
        try:
            assert dry_run_argv(prepared) == prepared.argv
            print_dry_run(prepared)
            captured = capsys.readouterr()
            assert DRY_RUN_SETTINGS_PLACEHOLDER in captured.out
        finally:
            prepared.close()

    def test_close_is_a_safe_no_op(self, tmp_path: Path) -> None:
        request = _request(tmp_path=tmp_path, memory=None, swap=None)
        run_config = _run_config()
        with patch("shutil.which", return_value="/usr/bin/tool"):
            prepared = prepare(request, run_config=run_config, resolve_settings=False)
        prepared.close()  # must not raise

    def test_backend_unavailable_still_raises(self, tmp_path: Path) -> None:
        request = _request(tmp_path=tmp_path, memory=None, swap=None)
        run_config = _run_config()
        with patch("shutil.which", return_value=None):
            with pytest.raises(SandboxUnavailableError):
                prepare(request, run_config=run_config, resolve_settings=False)

    def test_settings_source_and_candidates_available_without_resolving(
        self, tmp_path: Path
    ) -> None:
        merged_request = _request(tmp_path=tmp_path)
        assert settings_source(merged_request.spec) == "merged"
        candidates = SRT_BACKEND.settings_candidates(merged_request)
        assert candidates
        assert not any(path.exists() for path in candidates)

        explicit_request = _request(tmp_path=tmp_path, settings_file=tmp_path / "x.json")
        assert settings_source(explicit_request.spec) == "explicit"


# ---------------------------------------------------------------------------
# SRT backend directly
# ---------------------------------------------------------------------------


class TestSrtBackendDirect:
    def test_require_available_raises_when_missing(self) -> None:
        with patch("shutil.which", return_value=None):
            with pytest.raises(SandboxUnavailableError):
                SRT_BACKEND.require_available({"PATH": "/bin"})

    def test_require_available_passes_path(self) -> None:
        calls: list[str | None] = []

        def fake_which(name: str, path: str | None = None) -> str | None:
            calls.append(path)
            return "/usr/bin/srt"

        with patch("shutil.which", side_effect=fake_which):
            SRT_BACKEND.require_available({"PATH": "/custom"})
        assert calls == ["/custom"]

    def test_wrap_returns_srt_settings_invocation(self, tmp_path: Path) -> None:
        request = _request(tmp_path=tmp_path)
        resolved = ResolvedSettings(path=tmp_path / "settings.json")
        assert SRT_BACKEND.wrap(request, resolved) == [
            "srt",
            "--settings",
            str(tmp_path / "settings.json"),
            "--",
        ]

    def test_dry_run_wrap_returns_placeholder_invocation(self, tmp_path: Path) -> None:
        request = _request(tmp_path=tmp_path)
        assert SRT_BACKEND.dry_run_wrap(request) == [
            "srt",
            "--settings",
            DRY_RUN_SETTINGS_PLACEHOLDER,
            "--",
        ]

    def test_relative_settings_file_resolves_against_config_cwd(self, tmp_path: Path) -> None:
        """An explicit relative `settings_file` resolves against `config_cwd` --
        the host's own config context -- never the per-call `cwd`, so a
        directory a command merely runs in cannot supply the settings file a
        program names."""
        config_cwd = tmp_path / "context"
        config_cwd.mkdir(parents=True)
        request = _request(
            tmp_path=tmp_path, settings_file=Path("relative.json"), config_cwd=config_cwd
        )
        (request.cwd / "relative.json").write_text(
            json.dumps({"network": {"allowedDomains": ["evil.example"]}}), encoding="utf-8"
        )
        (config_cwd / "relative.json").write_text("{}", encoding="utf-8")

        resolved = SRT_BACKEND.resolve_settings(request)

        assert resolved.path == config_cwd / "relative.json"

    def test_missing_relative_settings_file_is_checked_against_config_cwd(
        self, tmp_path: Path
    ) -> None:
        """A relative `settings_file` that exists only at the per-call `cwd` is
        still reported missing: only `config_cwd` is ever checked."""
        config_cwd = tmp_path / "context"
        config_cwd.mkdir(parents=True)
        request = _request(
            tmp_path=tmp_path, settings_file=Path("relative.json"), config_cwd=config_cwd
        )
        (request.cwd / "relative.json").write_text("{}", encoding="utf-8")

        with pytest.raises(SandboxSettingsError) as excinfo:
            SRT_BACKEND.resolve_settings(request)

        assert excinfo.value.path == config_cwd / "relative.json"

    def test_no_profile_name_only_checks_default_json(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        cwd = tmp_path / "work"
        home.mkdir(parents=True, exist_ok=True)
        cwd.mkdir(parents=True, exist_ok=True)
        home_sandbox = home / ".agm" / "sandbox"
        home_sandbox.mkdir(parents=True)
        # A per-name file that must be ignored when there is no profile name.
        (home_sandbox / "echo.json").write_text(json.dumps({"enabled": False}))
        (home_sandbox / "default.json").write_text(json.dumps({"enabled": True}))

        request = _request(tmp_path=tmp_path, profile=None, patch_proj_dir=False)
        resolved = SRT_BACKEND.resolve_settings(request)
        assert resolved.path == home_sandbox / "default.json"

    def test_no_profile_name_with_proj_dir_checks_its_default_json_too(
        self, tmp_path: Path
    ) -> None:
        home = tmp_path / "home"
        proj_dir = tmp_path / "project"
        (home / ".agm" / "sandbox").mkdir(parents=True)
        proj_sandbox = proj_dir / "config" / "sandbox"
        proj_sandbox.mkdir(parents=True)
        proj_sandbox_default = proj_sandbox / "default.json"
        proj_sandbox_default.write_text(json.dumps({"enabled": True}))

        request = _request(tmp_path=tmp_path, profile=None, patch_proj_dir=False, proj_dir=proj_dir)
        resolved = SRT_BACKEND.resolve_settings(request)
        assert resolved.path == proj_sandbox_default

    def test_settings_candidates_matches_the_search_resolve_settings_performs(
        self, tmp_path: Path
    ) -> None:
        """`settings_candidates()` returns exactly the paths `resolve_settings()` searches,
        in the same order, and the file it selects is one of them."""

        request = _request(tmp_path=tmp_path, profile="echo")
        home_sandbox = request.home / ".agm" / "sandbox"
        home_sandbox.mkdir(parents=True)
        cwd_sandbox = request.cwd / ".sandbox"
        cwd_sandbox.mkdir()
        (cwd_sandbox / "echo.json").write_text("{}", encoding="utf-8")

        candidates = SRT_BACKEND.settings_candidates(request)
        assert candidates == [home_sandbox / "default.json", cwd_sandbox / "echo.json"]

        resolved = SRT_BACKEND.resolve_settings(request)
        assert resolved.path == candidates[-1]

    def test_resolve_settings_uses_config_cwd_not_the_per_call_cwd(self, tmp_path: Path) -> None:
        """A `.sandbox/<name>.json` at the per-call `cwd` is never a candidate,
        and never merges into the resolved settings: only `config_cwd` -- the
        host `SandboxContext`'s own directory -- is searched. Proves `cwd` and
        `config_cwd` are independent: a directory a command merely runs in
        cannot supply the settings that confine it."""
        home = tmp_path / "home"
        call_cwd = tmp_path / "data"
        config_cwd = tmp_path / "context"
        home.mkdir(parents=True)
        call_cwd.mkdir(parents=True)
        config_cwd.mkdir(parents=True)
        (home / ".agm" / "sandbox").mkdir(parents=True)
        (call_cwd / ".sandbox").mkdir()
        (call_cwd / ".sandbox" / "echo.json").write_text(
            json.dumps({"network": {"allowedDomains": ["evil.example"]}})
        )
        (config_cwd / ".sandbox").mkdir()
        (config_cwd / ".sandbox" / "echo.json").write_text("{}", encoding="utf-8")

        request = SandboxRequest(
            command=["echo", "hi"],
            cwd=call_cwd,
            config_cwd=config_cwd,
            env={"HOME": str(home), "PATH": "/bin"},
            home=home,
            proj_dir=None,
            spec=SandboxSpec(profile_name="echo"),
        )

        candidates = SRT_BACKEND.settings_candidates(request)
        assert call_cwd / ".sandbox" / "echo.json" not in candidates
        assert candidates[-1] == config_cwd / ".sandbox" / "echo.json"

        resolved = SRT_BACKEND.resolve_settings(request)
        assert resolved.path == config_cwd / ".sandbox" / "echo.json"


class TestAliasDrivenSettings:
    def test_alias_name_selects_the_alias_settings_file(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home_sandbox = home / ".agm" / "sandbox"
        home_sandbox.mkdir(parents=True)
        (home_sandbox / "myalias.json").write_text(json.dumps({"enabled": True}))

        # `profile="echo"` has no `echo.json`, so only the alias name matches.
        request = _request(
            tmp_path=tmp_path, profile="echo", alias_name="myalias", patch_proj_dir=False
        )
        resolved = SRT_BACKEND.resolve_settings(request)
        assert resolved.path == home_sandbox / "myalias.json"


# ---------------------------------------------------------------------------
# A second SandboxBackend implementation, exercising the protocol
# ---------------------------------------------------------------------------


@dataclass
class _FakeBackend:
    """A minimal, non-SRT `SandboxBackend`: proves the protocol is truly pluggable."""

    settings_path: Path
    artifact: Path

    def require_available(self, env: dict[str, str]) -> None:
        return None

    def settings_candidates(self, request: SandboxRequest) -> list[Path]:
        return [self.settings_path]

    def resolve_settings(self, request: SandboxRequest) -> ResolvedSettings:
        return ResolvedSettings(path=self.settings_path, tracked_artifacts=(self.artifact,))

    def wrap(self, request: SandboxRequest, resolved: ResolvedSettings) -> list[str]:
        return ["fake-sandbox", "--settings", str(resolved.path), "--"]

    def dry_run_wrap(self, request: SandboxRequest) -> list[str]:
        return ["fake-sandbox", "--settings", DRY_RUN_SETTINGS_PLACEHOLDER, "--"]

    def prepare_env(self, request: SandboxRequest, env: dict[str, str]) -> dict[str, str]:
        return {**env, "FAKE_SANDBOX": "1"}


class TestFakeBackendConformance:
    def test_fake_backend_wraps_command_applies_env_and_cleans_up(self, tmp_path: Path) -> None:
        settings_path = tmp_path / "fake-settings.json"
        settings_path.write_text("{}", encoding="utf-8")
        artifact = tmp_path / "artifact"
        artifact.write_bytes(b"")  # empty: close() should remove it
        backend = _FakeBackend(settings_path=settings_path, artifact=artifact)

        request = _request(tmp_path=tmp_path, memory=None, swap=None)
        run_config = _run_config()
        prepared = prepare(request, run_config=run_config, backend=backend)
        try:
            assert prepared.argv == [
                "fake-sandbox",
                "--settings",
                str(settings_path),
                "--",
                "echo",
                "hi",
            ]
            assert prepared.env["FAKE_SANDBOX"] == "1"
        finally:
            prepared.close()
        assert not artifact.exists()
