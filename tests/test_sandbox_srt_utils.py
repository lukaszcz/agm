"""Comprehensive tests for agm.config.sandbox.srt and agm.sandbox.srt."""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

import agm.core.dry_run as dry_run
from agm.config.sandbox.srt import (
    JsonDict,
    _first_missing_component,
    _normalize_path,
    json_dict,
    load_settings,
    merge_settings,
    merge_settings_chain,
    patch_for_proj_dir,
    sandbox_settings_candidates,
    sandbox_settings_path,
    track_bwrap_artifacts,
)
from agm.sandbox.srt import (
    _cleanup,
    _print_dry_run,
    _resolve_settings_path,
    _write_json_temp,
    require_srt_installed,
)

# ---------------------------------------------------------------------------
# Fixture: always reset dry-run state after each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_dry_run() -> Generator[None, None, None]:
    dry_run.set_enabled(False)
    yield
    dry_run.set_enabled(False)


# ===========================================================================
# agm.config.sandbox.srt — json_dict
# ===========================================================================


class TestJsonDict:
    def test_returns_dict_unchanged(self) -> None:
        d: dict[str, object] = {"key": "val"}
        result = json_dict(d)
        assert result is d

    def test_returns_empty_dict_for_none(self) -> None:
        assert json_dict(None) == {}

    def test_returns_empty_dict_for_string(self) -> None:
        assert json_dict("hello") == {}

    def test_returns_empty_dict_for_int(self) -> None:
        assert json_dict(42) == {}

    def test_returns_empty_dict_for_list(self) -> None:
        assert json_dict([1, 2, 3]) == {}

    def test_returns_empty_dict_for_bool(self) -> None:
        assert json_dict(True) == {}

    def test_nested_dict_is_returned(self) -> None:
        d: dict[str, object] = {"a": {"b": 1}}
        assert json_dict(d) == {"a": {"b": 1}}


# ===========================================================================
# agm.config.sandbox.srt — load_settings
# ===========================================================================


class TestLoadSettings:
    def test_loads_valid_json_dict(self, tmp_path: Path) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"enabled": True, "network": {}}), encoding="utf-8")
        result = load_settings(settings_file)
        assert result == {"enabled": True, "network": {}}

    def test_returns_empty_dict_for_json_non_object(self, tmp_path: Path) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        result = load_settings(settings_file)
        assert result == {}

    def test_returns_empty_dict_for_json_null(self, tmp_path: Path) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text("null", encoding="utf-8")
        result = load_settings(settings_file)
        assert result == {}

    def test_raises_on_invalid_json(self, tmp_path: Path) -> None:
        settings_file = tmp_path / "bad.json"
        settings_file.write_text("not json!", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_settings(settings_file)

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_settings(tmp_path / "missing.json")


# ===========================================================================
# agm.config.sandbox.srt — merge_settings
# ===========================================================================


class TestMergeSettings:
    def test_empty_dicts_produce_empty_result(self) -> None:
        assert merge_settings({}, {}) == {}

    def test_home_fields_are_preserved_when_local_is_empty(self) -> None:
        home: JsonDict = {"enabled": True, "extra": "value"}
        result = merge_settings(home, {})
        assert result["enabled"] is True
        assert result["extra"] == "value"

    def test_enabled_is_overridden_by_local(self) -> None:
        home: JsonDict = {"enabled": True}
        local: JsonDict = {"enabled": False}
        result = merge_settings(home, local)
        assert result["enabled"] is False

    def test_enabled_none_in_local_does_not_override(self) -> None:
        home: JsonDict = {"enabled": True}
        local: JsonDict = {"enabled": None}
        result = merge_settings(home, local)
        assert result["enabled"] is True

    def test_network_is_shallow_merged(self) -> None:
        home: JsonDict = {"network": {"allowedDomains": ["a.com"], "allowedPorts": [80]}}
        local: JsonDict = {"network": {"allowedDomains": ["b.com"]}}
        result = merge_settings(home, local)
        network = result["network"]
        assert isinstance(network, dict)
        assert network["allowedDomains"] == ["b.com"]
        assert network["allowedPorts"] == [80]

    def test_network_from_scratch_when_home_has_none(self) -> None:
        home: JsonDict = {}
        local: JsonDict = {"network": {"allowedDomains": ["x.com"]}}
        result = merge_settings(home, local)
        assert result["network"] == {"allowedDomains": ["x.com"]}

    def test_network_non_dict_in_local_is_ignored(self) -> None:
        home: JsonDict = {"network": {"key": "val"}}
        local: JsonDict = {"network": "not-a-dict"}
        result = merge_settings(home, local)
        assert result["network"] == {"key": "val"}

    def test_filesystem_is_shallow_merged(self) -> None:
        home: JsonDict = {"filesystem": {"allowWrite": ["/home"], "denyWrite": ["/etc"]}}
        local: JsonDict = {"filesystem": {"allowWrite": ["/local"]}}
        result = merge_settings(home, local)
        filesystem = result["filesystem"]
        assert isinstance(filesystem, dict)
        assert filesystem["allowWrite"] == ["/local"]
        assert filesystem["denyWrite"] == ["/etc"]

    def test_filesystem_from_scratch_when_home_has_none(self) -> None:
        home: JsonDict = {}
        local: JsonDict = {"filesystem": {"allowWrite": ["/work"]}}
        result = merge_settings(home, local)
        assert result["filesystem"] == {"allowWrite": ["/work"]}

    def test_ignore_violations_is_replaced_not_merged(self) -> None:
        home: JsonDict = {"ignoreViolations": {"read": True}}
        local: JsonDict = {"ignoreViolations": {"write": False}}
        result = merge_settings(home, local)
        assert result["ignoreViolations"] == {"write": False}

    def test_ignore_violations_not_overridden_if_not_dict_in_local(self) -> None:
        home: JsonDict = {"ignoreViolations": {"read": True}}
        local: JsonDict = {"ignoreViolations": "yes"}
        result = merge_settings(home, local)
        assert result["ignoreViolations"] == {"read": True}

    def test_enable_weaker_nested_sandbox_overridden_by_local(self) -> None:
        home: JsonDict = {"enableWeakerNestedSandbox": False}
        local: JsonDict = {"enableWeakerNestedSandbox": True}
        result = merge_settings(home, local)
        assert result["enableWeakerNestedSandbox"] is True

    def test_enable_weaker_nested_sandbox_none_in_local_not_overridden(self) -> None:
        home: JsonDict = {"enableWeakerNestedSandbox": True}
        local: JsonDict = {"enableWeakerNestedSandbox": None}
        result = merge_settings(home, local)
        assert result["enableWeakerNestedSandbox"] is True

    def test_enable_weaker_nested_sandbox_absent_in_local_not_overridden(self) -> None:
        home: JsonDict = {"enableWeakerNestedSandbox": False}
        local: JsonDict = {}
        result = merge_settings(home, local)
        assert result["enableWeakerNestedSandbox"] is False


# ===========================================================================
# agm.config.sandbox.srt — merge_settings_chain
# ===========================================================================


class TestMergeSettingsChain:
    def test_empty_list_returns_empty_dict(self) -> None:
        assert merge_settings_chain([]) == {}

    def test_single_element_returned_as_is(self) -> None:
        data: JsonDict = {"enabled": True}
        result = merge_settings_chain([data])
        assert result == {"enabled": True}

    def test_two_elements_are_merged(self) -> None:
        first: JsonDict = {"enabled": True, "network": {"allowedDomains": ["a.com"]}}
        second: JsonDict = {"enabled": False}
        result = merge_settings_chain([first, second])
        assert result["enabled"] is False
        assert result["network"] == {"allowedDomains": ["a.com"]}

    def test_three_elements_are_left_folded(self) -> None:
        a: JsonDict = {"enabled": True}
        b: JsonDict = {"enabled": False}
        c: JsonDict = {"enabled": True}
        result = merge_settings_chain([a, b, c])
        assert result["enabled"] is True

    def test_later_filesystem_overrides_earlier(self) -> None:
        a: JsonDict = {"filesystem": {"allowWrite": ["/a"]}}
        b: JsonDict = {"filesystem": {"allowWrite": ["/b"]}}
        result = merge_settings_chain([a, b])
        filesystem = result["filesystem"]
        assert isinstance(filesystem, dict)
        assert filesystem["allowWrite"] == ["/b"]


# ===========================================================================
# agm.config.sandbox.srt — patch_for_proj_dir
# ===========================================================================


class TestPatchForProjDir:
    def test_adds_notes_deps_and_git_dirs(self, tmp_path: Path) -> None:
        proj_dir = tmp_path / "project"
        (proj_dir / "repo").mkdir(parents=True)
        settings: JsonDict = {}
        result = patch_for_proj_dir(settings, proj_dir)
        filesystem = result["filesystem"]
        assert isinstance(filesystem, dict)
        allow_write = filesystem["allowWrite"]
        assert isinstance(allow_write, list)
        assert str(proj_dir / "notes") in allow_write
        assert str(proj_dir / "deps") in allow_write
        assert str(proj_dir / "repo" / ".git") in allow_write

    def test_preserves_existing_allow_write_entries(self, tmp_path: Path) -> None:
        proj_dir = tmp_path / "project"
        (proj_dir / "repo").mkdir(parents=True)
        settings: JsonDict = {"filesystem": {"allowWrite": ["/existing"]}}
        result = patch_for_proj_dir(settings, proj_dir)
        filesystem = result["filesystem"]
        assert isinstance(filesystem, dict)
        allow_write = filesystem["allowWrite"]
        assert "/existing" in allow_write
        assert str(proj_dir / "notes") in allow_write

    def test_does_not_duplicate_existing_entries(self, tmp_path: Path) -> None:
        proj_dir = tmp_path / "project"
        (proj_dir / "repo").mkdir(parents=True)
        notes_str = str(proj_dir / "notes")
        settings: JsonDict = {"filesystem": {"allowWrite": [notes_str]}}
        result = patch_for_proj_dir(settings, proj_dir)
        filesystem = result["filesystem"]
        assert isinstance(filesystem, dict)
        allow_write = filesystem["allowWrite"]
        assert allow_write.count(notes_str) == 1

    def test_non_list_allow_write_is_replaced(self, tmp_path: Path) -> None:
        proj_dir = tmp_path / "project"
        (proj_dir / "repo").mkdir(parents=True)
        settings: JsonDict = {"filesystem": {"allowWrite": "not-a-list"}}
        result = patch_for_proj_dir(settings, proj_dir)
        filesystem = result["filesystem"]
        assert isinstance(filesystem, dict)
        allow_write = filesystem["allowWrite"]
        assert isinstance(allow_write, list)
        assert str(proj_dir / "notes") in allow_write

    def test_creates_filesystem_key_if_absent(self, tmp_path: Path) -> None:
        proj_dir = tmp_path / "project"
        (proj_dir / "repo").mkdir(parents=True)
        settings: JsonDict = {"enabled": True}
        result = patch_for_proj_dir(settings, proj_dir)
        assert "filesystem" in result

    def test_returns_new_top_level_dict(self, tmp_path: Path) -> None:
        # patch_for_proj_dir returns a new top-level dict (shallow copy), not the same object
        proj_dir = tmp_path / "project"
        (proj_dir / "repo").mkdir(parents=True)
        settings: JsonDict = {"filesystem": {"allowWrite": ["/original"]}}
        result = patch_for_proj_dir(settings, proj_dir)
        assert result is not settings

    def test_workspace_project_uses_repo_subdir(self, tmp_path: Path) -> None:
        proj_dir = tmp_path / "workspace"
        (proj_dir / "repo").mkdir(parents=True)
        result = patch_for_proj_dir({}, proj_dir)
        filesystem = result["filesystem"]
        assert isinstance(filesystem, dict)
        allow_write = filesystem["allowWrite"]
        assert str(proj_dir / "repo" / ".git") in allow_write


# ===========================================================================
# agm.config.sandbox.srt — sandbox_settings_path
# ===========================================================================


class TestSandboxSettingsPath:
    def test_returns_command_json_if_exists(self, tmp_path: Path) -> None:
        settings_dir = tmp_path / "sandbox"
        settings_dir.mkdir()
        cmd_file = settings_dir / "claude.json"
        cmd_file.write_text("{}", encoding="utf-8")
        result = sandbox_settings_path(settings_dir, "claude")
        assert result == cmd_file

    def test_returns_alias_json_if_command_json_missing(self, tmp_path: Path) -> None:
        settings_dir = tmp_path / "sandbox"
        settings_dir.mkdir()
        alias_file = settings_dir / "claude.json"
        alias_file.write_text("{}", encoding="utf-8")
        result = sandbox_settings_path(settings_dir, "code", alias_command_name="claude")
        assert result == alias_file

    def test_returns_default_json_if_no_match(self, tmp_path: Path) -> None:
        settings_dir = tmp_path / "sandbox"
        settings_dir.mkdir()
        result = sandbox_settings_path(settings_dir, "unknown")
        assert result == settings_dir / "default.json"

    def test_command_json_takes_precedence_over_alias(self, tmp_path: Path) -> None:
        settings_dir = tmp_path / "sandbox"
        settings_dir.mkdir()
        cmd_file = settings_dir / "code.json"
        cmd_file.write_text("{}", encoding="utf-8")
        alias_file = settings_dir / "claude.json"
        alias_file.write_text("{}", encoding="utf-8")
        result = sandbox_settings_path(settings_dir, "code", alias_command_name="claude")
        assert result == cmd_file

    def test_none_alias_is_skipped(self, tmp_path: Path) -> None:
        settings_dir = tmp_path / "sandbox"
        settings_dir.mkdir()
        result = sandbox_settings_path(settings_dir, "unknown", alias_command_name=None)
        assert result == settings_dir / "default.json"

    def test_uses_basename_of_command_path(self, tmp_path: Path) -> None:
        settings_dir = tmp_path / "sandbox"
        settings_dir.mkdir()
        cmd_file = settings_dir / "node.json"
        cmd_file.write_text("{}", encoding="utf-8")
        result = sandbox_settings_path(settings_dir, "/usr/bin/node")
        assert result == cmd_file


# ===========================================================================
# agm.config.sandbox.srt — sandbox_settings_candidates
# ===========================================================================


class TestSandboxSettingsCandidates:
    def test_without_proj_dir_returns_two_candidates(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        cwd = tmp_path / "work"
        candidates = sandbox_settings_candidates(
            cwd=cwd,
            home=home,
            proj_dir=None,
            command_name="echo",
        )
        assert len(candidates) == 2

    def test_with_proj_dir_returns_three_candidates(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        cwd = tmp_path / "work"
        proj_dir = tmp_path / "project"
        candidates = sandbox_settings_candidates(
            cwd=cwd,
            home=home,
            proj_dir=proj_dir,
            command_name="echo",
        )
        assert len(candidates) == 3

    def test_first_candidate_is_from_home_agm_sandbox(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        cwd = tmp_path / "work"
        candidates = sandbox_settings_candidates(
            cwd=cwd,
            home=home,
            proj_dir=None,
            command_name="echo",
        )
        assert str(home / ".agm" / "sandbox") in str(candidates[0])

    def test_last_candidate_is_from_cwd_sandbox(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        cwd = tmp_path / "work"
        candidates = sandbox_settings_candidates(
            cwd=cwd,
            home=home,
            proj_dir=None,
            command_name="echo",
        )
        assert str(cwd / ".sandbox") in str(candidates[-1])

    def test_proj_dir_candidate_uses_project_config_dir(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        cwd = tmp_path / "work"
        proj_dir = tmp_path / "project"
        candidates = sandbox_settings_candidates(
            cwd=cwd,
            home=home,
            proj_dir=proj_dir,
            command_name="echo",
        )
        config_sandbox = proj_dir / "config" / "sandbox"
        assert any(str(config_sandbox) in str(c) for c in candidates)

    def test_command_json_used_when_file_exists(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        cwd = tmp_path / "work"
        home_sandbox = home / ".agm" / "sandbox"
        home_sandbox.mkdir(parents=True)
        (home_sandbox / "echo.json").write_text("{}", encoding="utf-8")
        candidates = sandbox_settings_candidates(
            cwd=cwd,
            home=home,
            proj_dir=None,
            command_name="echo",
        )
        assert candidates[0].name == "echo.json"

    def test_default_json_used_when_no_command_file(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        cwd = tmp_path / "work"
        candidates = sandbox_settings_candidates(
            cwd=cwd,
            home=home,
            proj_dir=None,
            command_name="unknown-cmd",
        )
        assert candidates[0].name == "default.json"


# ===========================================================================
# agm.config.sandbox.srt — _normalize_path
# ===========================================================================


class TestNormalizePath:
    def test_tilde_alone_resolves_to_home(self, tmp_path: Path) -> None:
        result = _normalize_path("~", tmp_path)
        assert result == Path.home().resolve(strict=False)

    def test_tilde_slash_expands_user(self, tmp_path: Path) -> None:
        result = _normalize_path("~/Documents", tmp_path)
        expected = Path("~/Documents").expanduser().resolve(strict=False)
        assert result == expected

    def test_absolute_path_is_used_directly(self, tmp_path: Path) -> None:
        result = _normalize_path("/etc/passwd", tmp_path)
        assert result == Path("/etc/passwd")

    def test_relative_path_resolves_against_cwd(self, tmp_path: Path) -> None:
        result = _normalize_path("subdir/file.txt", tmp_path)
        assert result == (tmp_path / "subdir" / "file.txt").resolve(strict=False)

    def test_dot_relative_path(self, tmp_path: Path) -> None:
        result = _normalize_path(".", tmp_path)
        assert result == tmp_path.resolve(strict=False)

    def test_non_tilde_leading_path_treated_as_relative(self, tmp_path: Path) -> None:
        result = _normalize_path(".gitconfig", tmp_path)
        assert result == (tmp_path / ".gitconfig").resolve(strict=False)

    def test_absolute_path_with_dotdot_is_resolved(self, tmp_path: Path) -> None:
        result = _normalize_path("/tmp/../etc", tmp_path)
        assert result == Path("/etc")


# ===========================================================================
# agm.config.sandbox.srt — _first_missing_component
# ===========================================================================


class TestFirstMissingComponent:
    def test_returns_none_when_all_components_exist(self, tmp_path: Path) -> None:
        d = tmp_path / "exists"
        d.mkdir()
        result = _first_missing_component(d, tmp_path)
        assert result is None

    def test_returns_first_missing_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "missing" / "deeper"
        result = _first_missing_component(target, tmp_path)
        assert result == tmp_path / "missing"

    def test_returns_missing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "missing_file.txt"
        result = _first_missing_component(target, tmp_path)
        assert result == target

    def test_returns_none_when_target_not_under_cwd(self, tmp_path: Path) -> None:
        target = Path("/some/other/path")
        result = _first_missing_component(target, tmp_path)
        assert result is None

    def test_partial_path_exists(self, tmp_path: Path) -> None:
        existing = tmp_path / "a"
        existing.mkdir()
        target = existing / "b" / "c"
        result = _first_missing_component(target, tmp_path)
        assert result == existing / "b"

    def test_cwd_itself_is_not_tracked(self, tmp_path: Path) -> None:
        # tmp_path itself exists; target == cwd → relative_to gives empty parts
        result = _first_missing_component(tmp_path, tmp_path)
        assert result is None


# ===========================================================================
# agm.config.sandbox.srt — track_bwrap_artifacts
# ===========================================================================


class TestTrackBwrapArtifacts:
    def _make_settings(self, tmp_path: Path, data: JsonDict) -> Path:
        p = tmp_path / "settings.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    def test_tracks_mandatory_deny_path(self, tmp_path: Path) -> None:
        settings_path = self._make_settings(tmp_path, {})
        artifacts = track_bwrap_artifacts(settings_path, tmp_path)
        # .gitconfig should be tracked as a candidate under cwd
        names = [a.name for a in artifacts]
        assert ".gitconfig" in names

    def test_tracks_filesystem_deny_write_entries(self, tmp_path: Path) -> None:
        settings_path = self._make_settings(
            tmp_path,
            {"filesystem": {"denyWrite": [".custom-deny"]}},
        )
        artifacts = track_bwrap_artifacts(settings_path, tmp_path)
        names = [a.name for a in artifacts]
        assert ".custom-deny" in names

    def test_skips_glob_patterns_in_deny_write(self, tmp_path: Path) -> None:
        settings_path = self._make_settings(
            tmp_path,
            {"filesystem": {"denyWrite": ["*.txt", "dir/**"]}},
        )
        artifacts = track_bwrap_artifacts(settings_path, tmp_path)
        names = [a.name for a in artifacts]
        assert "*.txt" not in names
        assert "dir/**" not in names

    def test_skips_paths_not_under_cwd(self, tmp_path: Path) -> None:
        settings_path = self._make_settings(
            tmp_path,
            {"filesystem": {"denyWrite": ["/absolute/outside"]}},
        )
        artifacts = track_bwrap_artifacts(settings_path, tmp_path)
        paths = [str(a) for a in artifacts]
        assert "/absolute/outside" not in paths

    def test_does_not_track_already_existing_paths(self, tmp_path: Path) -> None:
        existing = tmp_path / ".gitconfig"
        existing.write_text("", encoding="utf-8")
        settings_path = self._make_settings(tmp_path, {})
        artifacts = track_bwrap_artifacts(settings_path, tmp_path)
        # .gitconfig exists → _first_missing_component returns None → not tracked
        assert existing not in artifacts

    def test_no_duplicates_in_result(self, tmp_path: Path) -> None:
        settings_path = self._make_settings(
            tmp_path,
            {"filesystem": {"denyWrite": [".gitconfig"]}},
        )
        artifacts = track_bwrap_artifacts(settings_path, tmp_path)
        names = [a.name for a in artifacts]
        assert names.count(".gitconfig") <= 1

    def test_adds_git_hook_paths_when_dot_git_dir_exists(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        settings_path = self._make_settings(tmp_path, {})
        artifacts = track_bwrap_artifacts(settings_path, tmp_path)
        names = [a.name for a in artifacts]
        assert "hooks" in names or "config" in names


# ===========================================================================
# agm.sandbox.srt — require_srt_installed
# ===========================================================================


class TestRequireSrtInstalled:
    def test_does_not_raise_when_srt_found(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/srt"):
            require_srt_installed(None)  # should not raise

    def test_raises_system_exit_when_srt_not_found(self) -> None:
        with patch("shutil.which", return_value=None):
            with pytest.raises(SystemExit) as exc_info:
                require_srt_installed(None)
        assert exc_info.value.code == 1

    def test_passes_path_to_which(self) -> None:
        calls: list[object] = []

        def fake_which(name: str, path: str | None = None) -> str | None:
            calls.append(path)
            return "/bin/srt"

        with patch("shutil.which", side_effect=fake_which):
            require_srt_installed("/custom/bin")

        assert calls == ["/custom/bin"]

    def test_stderr_message_on_missing_srt(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("shutil.which", return_value=None):
            with pytest.raises(SystemExit):
                require_srt_installed(None)
        captured = capsys.readouterr()
        assert "srt" in captured.err


# ===========================================================================
# agm.sandbox.srt — _write_json_temp
# ===========================================================================


class TestWriteJsonTemp:
    def test_creates_file_and_appends_to_list(self, tmp_path: Path) -> None:
        temp_files: list[Path] = []
        data: JsonDict = {"key": "value", "num": 42}
        path = _write_json_temp(data, temp_files)
        assert path.is_file()
        assert temp_files == [path]
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == {"key": "value", "num": 42}

    def test_appends_to_existing_list(self, tmp_path: Path) -> None:
        existing = tmp_path / "existing.json"
        existing.write_text("{}", encoding="utf-8")
        temp_files: list[Path] = [existing]
        path = _write_json_temp({"a": 1}, temp_files)
        assert len(temp_files) == 2
        assert temp_files[0] == existing
        assert temp_files[1] == path

    def test_multiple_calls_create_distinct_files(self) -> None:
        temp_files: list[Path] = []
        path1 = _write_json_temp({"n": 1}, temp_files)
        path2 = _write_json_temp({"n": 2}, temp_files)
        assert path1 != path2
        assert len(temp_files) == 2

    def test_written_file_is_valid_json(self) -> None:
        temp_files: list[Path] = []
        data: JsonDict = {"nested": {"x": [1, 2, 3]}}
        path = _write_json_temp(data, temp_files)
        assert json.loads(path.read_text(encoding="utf-8")) == data

    def teardown_method(self) -> None:
        # Best-effort cleanup of any temp files created
        pass


# ===========================================================================
# agm.sandbox.srt — _cleanup
# ===========================================================================


class TestCleanup:
    def test_removes_temp_files(self, tmp_path: Path) -> None:
        f1 = tmp_path / "temp1.json"
        f2 = tmp_path / "temp2.json"
        f1.write_text("{}", encoding="utf-8")
        f2.write_text("{}", encoding="utf-8")
        _cleanup([f1, f2], [])
        assert not f1.exists()
        assert not f2.exists()

    def test_ignores_missing_temp_files(self, tmp_path: Path) -> None:
        missing = tmp_path / "ghost.json"
        _cleanup([missing], [])  # should not raise

    def test_removes_empty_artifact_dirs(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "artifact_dir"
        empty_dir.mkdir()
        _cleanup([], [empty_dir])
        assert not empty_dir.exists()

    def test_removes_empty_artifact_files(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "artifact_file"
        empty_file.write_bytes(b"")
        _cleanup([], [empty_file])
        assert not empty_file.exists()

    def test_does_not_remove_non_empty_artifact_files(self, tmp_path: Path) -> None:
        non_empty = tmp_path / "artifact_with_content"
        non_empty.write_text("content", encoding="utf-8")
        _cleanup([], [non_empty])
        assert non_empty.exists()

    def test_does_not_remove_non_empty_artifact_dirs(self, tmp_path: Path) -> None:
        d = tmp_path / "dir_with_child"
        d.mkdir()
        (d / "child.txt").write_text("x", encoding="utf-8")
        _cleanup([], [d])
        assert d.exists()

    def test_cleans_both_temp_files_and_artifacts(self, tmp_path: Path) -> None:
        temp_file = tmp_path / "temp.json"
        temp_file.write_text("{}", encoding="utf-8")
        artifact = tmp_path / "artifact"
        artifact.mkdir()
        _cleanup([temp_file], [artifact])
        assert not temp_file.exists()
        assert not artifact.exists()


# ===========================================================================
# agm.sandbox.srt — _resolve_settings_path
# ===========================================================================


class TestResolveSettingsPath:
    def test_explicit_settings_file_returned_directly(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        cwd = tmp_path / "work"
        settings_file = tmp_path / "custom.json"
        settings_file.write_text("{}", encoding="utf-8")
        temp_files: list[Path] = []
        result = _resolve_settings_path(
            cwd=cwd,
            home=home,
            proj_dir=None,
            command_name="echo",
            alias_command_name=None,
            settings_file=str(settings_file),
            temp_files=temp_files,
        )
        assert result == settings_file
        assert temp_files == []

    def test_relative_settings_file_resolves_against_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        cwd = tmp_path / "work"
        cwd.mkdir(parents=True)
        (cwd / "sandbox").mkdir()
        (cwd / "sandbox" / "custom.json").write_text("{}", encoding="utf-8")
        monkeypatch.chdir(cwd)
        temp_files: list[Path] = []
        result = _resolve_settings_path(
            cwd=cwd,
            home=home,
            proj_dir=None,
            command_name="echo",
            alias_command_name=None,
            settings_file="sandbox/custom.json",
            temp_files=temp_files,
        )
        assert result == cwd / "sandbox" / "custom.json"
        assert temp_files == []

    def test_explicit_settings_file_not_found_exits(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        cwd = tmp_path / "work"
        temp_files: list[Path] = []
        with pytest.raises(SystemExit) as exc_info:
            _resolve_settings_path(
                cwd=cwd,
                home=home,
                proj_dir=None,
                command_name="echo",
                alias_command_name=None,
                settings_file=str(tmp_path / "nonexistent.json"),
                temp_files=temp_files,
            )
        assert exc_info.value.code == 1

    def test_no_settings_found_exits(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        cwd = tmp_path / "work"
        cwd.mkdir()
        temp_files: list[Path] = []
        with pytest.raises(SystemExit) as exc_info:
            _resolve_settings_path(
                cwd=cwd,
                home=home,
                proj_dir=None,
                command_name="echo",
                alias_command_name=None,
                settings_file=None,
                temp_files=temp_files,
            )
        assert exc_info.value.code == 1

    def test_single_settings_file_returned_directly(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        cwd = tmp_path / "work"
        cwd.mkdir()
        home_sandbox = home / ".agm" / "sandbox"
        home_sandbox.mkdir(parents=True)
        default_settings = home_sandbox / "default.json"
        default_settings.write_text("{}", encoding="utf-8")
        temp_files: list[Path] = []
        result = _resolve_settings_path(
            cwd=cwd,
            home=home,
            proj_dir=None,
            command_name="echo",
            alias_command_name=None,
            settings_file=None,
            temp_files=temp_files,
        )
        assert result == default_settings
        assert temp_files == []

    def test_multiple_settings_files_are_merged(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        cwd = tmp_path / "work"
        cwd.mkdir()
        home_sandbox = home / ".agm" / "sandbox"
        home_sandbox.mkdir(parents=True)
        cwd_sandbox = cwd / ".sandbox"
        cwd_sandbox.mkdir()
        home_settings_data: JsonDict = {"enabled": True, "network": {"allowedDomains": ["a.com"]}}
        cwd_settings_data: JsonDict = {"enabled": False}
        (home_sandbox / "default.json").write_text(
            json.dumps(home_settings_data), encoding="utf-8"
        )
        (cwd_sandbox / "default.json").write_text(
            json.dumps(cwd_settings_data), encoding="utf-8"
        )
        temp_files: list[Path] = []
        result = _resolve_settings_path(
            cwd=cwd,
            home=home,
            proj_dir=None,
            command_name="echo",
            alias_command_name=None,
            settings_file=None,
            temp_files=temp_files,
        )
        assert len(temp_files) == 1
        assert result == temp_files[0]
        merged = json.loads(result.read_text(encoding="utf-8"))
        assert merged["enabled"] is False
        assert merged["network"]["allowedDomains"] == ["a.com"]


# ===========================================================================
# agm.sandbox.srt — _print_dry_run
# ===========================================================================


class TestPrintDryRun:
    def test_prints_sandbox_configuration_header(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dry_run.set_enabled(True)
        _print_dry_run(
            cwd=tmp_path / "work",
            home=tmp_path / "home",
            proj_dir=None,
            command=["echo", "hello"],
            command_name="echo",
            alias_command_name=None,
            settings_file=None,
            patch_proj_dir=None,
            process_prefix=[],
        )
        captured = capsys.readouterr()
        assert "sandbox" in captured.out

    def test_prints_explicit_settings_source(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dry_run.set_enabled(True)
        settings_file = str(tmp_path / "custom.json")
        _print_dry_run(
            cwd=tmp_path / "work",
            home=tmp_path / "home",
            proj_dir=None,
            command=["echo"],
            command_name="echo",
            alias_command_name=None,
            settings_file=settings_file,
            patch_proj_dir=None,
            process_prefix=[],
        )
        captured = capsys.readouterr()
        assert "explicit" in captured.out
        assert settings_file in captured.out

    def test_prints_merged_settings_source_when_no_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dry_run.set_enabled(True)
        _print_dry_run(
            cwd=tmp_path / "work",
            home=tmp_path / "home",
            proj_dir=None,
            command=["echo"],
            command_name="echo",
            alias_command_name=None,
            settings_file=None,
            patch_proj_dir=None,
            process_prefix=[],
        )
        captured = capsys.readouterr()
        assert "merged" in captured.out

    def test_prints_patch_proj_dir_path_when_set(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dry_run.set_enabled(True)
        proj_dir = tmp_path / "project"
        _print_dry_run(
            cwd=tmp_path / "work",
            home=tmp_path / "home",
            proj_dir=None,
            command=["echo"],
            command_name="echo",
            alias_command_name=None,
            settings_file=None,
            patch_proj_dir=proj_dir,
            process_prefix=[],
        )
        captured = capsys.readouterr()
        assert str(proj_dir) in captured.out

    def test_prints_disabled_when_patch_proj_dir_is_none(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dry_run.set_enabled(True)
        _print_dry_run(
            cwd=tmp_path / "work",
            home=tmp_path / "home",
            proj_dir=None,
            command=["echo"],
            command_name="echo",
            alias_command_name=None,
            settings_file=None,
            patch_proj_dir=None,
            process_prefix=[],
        )
        captured = capsys.readouterr()
        assert "disabled" in captured.out

    def test_prints_command_with_process_prefix(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dry_run.set_enabled(True)
        _print_dry_run(
            cwd=tmp_path / "work",
            home=tmp_path / "home",
            proj_dir=None,
            command=["myapp", "--flag"],
            command_name="myapp",
            alias_command_name=None,
            settings_file=None,
            patch_proj_dir=None,
            process_prefix=["systemd-run", "--user"],
        )
        captured = capsys.readouterr()
        assert "systemd-run" in captured.out
        assert "myapp" in captured.out
        assert "srt" in captured.out
