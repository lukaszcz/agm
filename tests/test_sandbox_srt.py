"""Tests for the SRT sandbox runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agm.sandbox import srt


def test_run_sandboxed_merges_patches_and_cleans_tracked_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    work = tmp_path / "work"
    proj_dir = tmp_path / "project"
    home_sandbox = home / ".agm" / "sandbox"
    local_sandbox = work / ".sandbox"
    tracked_dir = work / ".claude"
    tracked_file = work / ".gitconfig"

    home_sandbox.mkdir(parents=True)
    work.mkdir()
    (proj_dir / "repo").mkdir(parents=True)
    local_sandbox.mkdir(parents=True)

    (home_sandbox / "echo.json").write_text(
        json.dumps({"filesystem": {"allowWrite": ["/home-write"]}})
    )
    (local_sandbox / "echo.json").write_text(
        json.dumps({"network": {"allowedDomains": ["example.com"]}})
    )

    calls: dict[str, object] = {}

    def fake_run_foreground(
        cmd: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        interrupt_cleanup_cmd: list[str] | None = None,
    ) -> int:
        calls["cmd"] = cmd
        calls["cwd"] = cwd
        calls["env"] = env
        calls["interrupt_cleanup_cmd"] = interrupt_cleanup_cmd
        settings_path = Path(cmd[cmd.index("--settings") + 1])
        calls["settings"] = json.loads(settings_path.read_text())
        tracked_dir.mkdir()
        tracked_file.write_text("")
        return 0

    monkeypatch.setattr(srt, "run_foreground", fake_run_foreground)
    with pytest.raises(SystemExit) as exc_info:
        srt.run_sandboxed(
            command=["echo", "hi"],
            cwd=work,
            env={"HOME": str(home), "PATH": "/bin"},
            home=home,
            proj_dir=proj_dir,
            command_name="echo",
            alias_command_name=None,
            settings_file=None,
            patch_proj_dir=proj_dir,
            process_prefix=["systemd-run", "--user", "--scope"],
            interrupt_cleanup_cmd=["systemctl", "--user", "stop", "agm-run.scope"],
        )

    assert exc_info.value.code == 0
    assert calls["cwd"] == work
    assert calls["env"] == {"HOME": str(home), "PATH": "/bin"}
    assert calls["cmd"][0:3] == ["systemd-run", "--user", "--scope"]
    assert calls["cmd"][3:6] == ["srt", "--settings", calls["cmd"][5]]
    assert calls["cmd"][6:9] == ["--", "echo", "hi"]
    assert calls["interrupt_cleanup_cmd"] == ["systemctl", "--user", "stop", "agm-run.scope"]

    settings = calls["settings"]
    assert settings["network"]["allowedDomains"] == ["example.com"]
    assert settings["filesystem"]["allowWrite"] == [
        "/home-write",
        str(proj_dir / "notes"),
        str(proj_dir / "deps"),
        str(proj_dir / "repo" / ".git"),
    ]

    assert not tracked_dir.exists()
    assert not tracked_file.exists()
