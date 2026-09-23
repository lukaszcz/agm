"""Tests for the SRT sandbox backend's full-preparation integration.

`tests/test_sandbox_prepare.py` covers the `SrtBackend`/`prepare()` contract in
isolation; these tests exercise the same settings-merge, project-write-patch,
and `NODE_USE_ENV_PROXY` behaviour end to end through `prepare()`'s public
surface, simulating a caller that runs the prepared command and then closes it
-- the shape `commands/run.py` and future agent/`exec` callers use.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from agm.config.general import RunConfig
from agm.sandbox.prepare import prepare
from agm.sandbox.request import SandboxRequest, SandboxSpec


def _run_config() -> RunConfig:
    return RunConfig(
        aliases={},
        default_memory_limit=None,
        command_memory_limits={},
        default_swap_limit=None,
        command_swap_limits={},
        default_pty=True,
        command_ptys={},
    )


def test_prepare_merges_patches_and_cleans_tracked_artifacts(tmp_path: Path) -> None:
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

    request = SandboxRequest(
        command=["echo", "hi"],
        cwd=work,
        config_cwd=work,
        env={"HOME": str(home), "PATH": "/bin"},
        home=home,
        proj_dir=proj_dir,
        spec=SandboxSpec(profile_name="echo", memory=None, swap=None, patch=True),
    )
    with patch("shutil.which", return_value="/usr/bin/srt"):
        prepared = prepare(request, run_config=_run_config())

    try:
        assert prepared.env == {"HOME": str(home), "PATH": "/bin", "NODE_USE_ENV_PROXY": "1"}
        assert prepared.argv[:1] == ["srt"]
        assert prepared.argv[-3:] == ["--", "echo", "hi"]

        assert prepared.settings_path is not None
        settings = json.loads(prepared.settings_path.read_text())
        assert settings["network"]["allowedDomains"] == ["example.com"]
        assert settings["filesystem"]["allowWrite"] == [
            "/home-write",
            str(proj_dir / "notes"),
            str(proj_dir / "deps"),
            str(proj_dir / "repo" / ".git"),
        ]

        # Simulate the command running and leaving the tracked bwrap artifacts behind.
        tracked_dir.mkdir()
        tracked_file.write_text("")
    finally:
        prepared.close()

    assert not tracked_dir.exists()
    assert not tracked_file.exists()


def test_prepare_injects_node_use_env_proxy_when_absent(tmp_path: Path) -> None:
    """NODE_USE_ENV_PROXY=1 is injected so Node.js built-in fetch honours proxy env vars."""
    home = tmp_path / "home"
    work = tmp_path / "work"
    home_sandbox = home / ".agm" / "sandbox"
    home_sandbox.mkdir(parents=True)
    work.mkdir()
    (home_sandbox / "default.json").write_text("{}", encoding="utf-8")

    request = SandboxRequest(
        command=["echo", "hi"],
        cwd=work,
        config_cwd=work,
        env={"HOME": str(home), "PATH": "/bin"},
        home=home,
        proj_dir=None,
        spec=SandboxSpec(profile_name=None, memory=None, swap=None, patch=False),
    )
    with patch("shutil.which", return_value="/usr/bin/srt"):
        prepared = prepare(request, run_config=_run_config())
    try:
        assert prepared.env["NODE_USE_ENV_PROXY"] == "1"
    finally:
        prepared.close()


def test_prepare_preserves_existing_node_use_env_proxy(tmp_path: Path) -> None:
    """An explicitly set NODE_USE_ENV_PROXY is not overridden."""
    home = tmp_path / "home"
    work = tmp_path / "work"
    home_sandbox = home / ".agm" / "sandbox"
    home_sandbox.mkdir(parents=True)
    work.mkdir()
    (home_sandbox / "default.json").write_text("{}", encoding="utf-8")

    request = SandboxRequest(
        command=["echo", "hi"],
        cwd=work,
        config_cwd=work,
        env={"HOME": str(home), "PATH": "/bin", "NODE_USE_ENV_PROXY": "0"},
        home=home,
        proj_dir=None,
        spec=SandboxSpec(profile_name=None, memory=None, swap=None, patch=False),
    )
    with patch("shutil.which", return_value="/usr/bin/srt"):
        prepared = prepare(request, run_config=_run_config())
    try:
        assert prepared.env["NODE_USE_ENV_PROXY"] == "0"
    finally:
        prepared.close()
