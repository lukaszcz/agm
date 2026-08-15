"""Package commands report an unreadable source path instead of crashing.

``agm pkg check``, ``agm pkg create``, and ``agm pkg install`` all take a
package directory from the user, so all three must survive a path the operating
system refuses to read.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import agm.commands.pkg.check as check_command
import agm.commands.pkg.create as create_command
import agm.commands.pkg.install as install_command
from agm.cli_support.args import PkgCheckArgs, PkgCreateArgs, PkgInstallArgs
from agm.config.context import ConfigContext


def _unreadable_source(tmp_path: Path) -> Path:
    """Return a path no user can read, whatever privileges the test runner has.

    The final component exceeds the filesystem's name limit, so every access
    fails with an ``OSError`` — unlike a permission bit, which the root user
    would simply ignore.
    """
    return tmp_path / ("x" * 300)


def _use_temp_home(monkeypatch: pytest.MonkeyPatch, module: object, tmp_path: Path) -> None:
    """Point one package command's config context at a private temporary home."""
    monkeypatch.setattr(
        module,
        "current_config_context",
        lambda: ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path),
    )


def test_check_reports_an_unreadable_source_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _use_temp_home(monkeypatch, check_command, tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        check_command.run(PkgCheckArgs(directory=str(_unreadable_source(tmp_path))))

    assert exc_info.value.code == 1
    assert capsys.readouterr().err.startswith("pkg check:")


def test_create_reports_an_unreadable_source_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _use_temp_home(monkeypatch, create_command, tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        create_command.run(PkgCreateArgs(directory=str(_unreadable_source(tmp_path)), output=None))

    assert exc_info.value.code == 1
    assert capsys.readouterr().err.startswith("pkg create:")


def test_install_reports_an_unreadable_source_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _use_temp_home(monkeypatch, install_command, tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        install_command.run(
            PkgInstallArgs(source=str(_unreadable_source(tmp_path)), editable=False, shadow=False)
        )

    assert exc_info.value.code == 1
    assert capsys.readouterr().err.startswith("pkg install:")
