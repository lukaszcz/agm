"""``std/os`` process, environment, and temporary-directory operations."""

from __future__ import annotations

import json
import os
import pwd
import socket
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from agm.agl import PipelineDriver
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import RunResult
from tests._agl_helpers import agl_roots


def _roots() -> RootSet:
    return agl_roots()


def _run(source: str, *, trace_file: Path | None = None) -> RunResult:
    return PipelineDriver(get_sandbox_context=None).run(
        source, roots=_roots(), trace_file=trace_file
    )


def _exit_program(call: str) -> str:
    return f"import std/os\nprogram def main() -> unit = os::exit({call})\n"


@pytest.mark.parametrize((("call", "expected_code")), [("", 0), ("0", 0), ("255", 255)])
def test_os_exit_preserves_the_requested_system_exit_code(
    tmp_path: Path, call: str, expected_code: int
) -> None:
    trace_path = tmp_path / "trace.jsonl"

    with pytest.raises(SystemExit) as raised:
        _run(_exit_program(call), trace_file=trace_path)

    assert raised.value.code == expected_code
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["kind"] == "run_start"
    assert records[-1]["kind"] == "run_end"
    assert records[-1]["ok"] is (expected_code == 0)


@pytest.mark.parametrize("code", [-1, 256])
def test_os_exit_rejects_codes_outside_the_portable_range(code: int) -> None:
    result = _run(_exit_program(str(code)))

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "ExternError"


def test_os_metadata_uses_the_controlled_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    source = (
        "import std/os\n"
        "program def main() -> unit =\n"
        "  print(os::cwd())\n"
        "  print(os::pid())\n"
        "  print(os::hostname())\n"
    )

    result = _run(source)

    assert result.ok
    cwd, pid, hostname = capsys.readouterr().out.splitlines()
    assert cwd == str(tmp_path)
    assert int(pid) > 0
    assert hostname


@pytest.mark.parametrize((("call", "expected_code")), [("", 0), ("0", 0), ("255", 255)])
def test_os_exit_reaches_the_cli_process_and_finalizes_its_trace(
    tmp_path: Path, env: dict[str, str], call: str, expected_code: int
) -> None:
    program = tmp_path / "exit.agl"
    trace_path = tmp_path / "trace.jsonl"
    program.write_text(_exit_program(call), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from agm.cli import main; main()",
            "exec",
            "--trace-file",
            str(trace_path),
            str(program),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == expected_code
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["kind"] == "run_start"
    assert records[-1]["kind"] == "run_end"
    assert records[-1]["ok"] is (expected_code == 0)


def test_os_cwd_raises_encoding_error_when_the_working_directory_is_not_valid_unicode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "getcwd", lambda: "/tmp/h\udcffome")

    result = _run("import std/os\nprogram def main() -> unit =\n  let _ = os::cwd()\n")

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "EncodingError"
    assert result.error.fields["raw"] == "/tmp/h\\udcffome"


def test_os_hostname_raises_encoding_error_when_the_hostname_is_not_valid_unicode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: "h\udcffost")

    result = _run("import std/os\nprogram def main() -> unit =\n  let _ = os::hostname()\n")

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "EncodingError"
    assert result.error.fields["raw"] == "h\\udcffost"


def test_os_metadata_matches_the_subprocess_identity(tmp_path: Path, env: dict[str, str]) -> None:
    program = tmp_path / "metadata.agl"
    program.write_text(
        "import std/os\n"
        "program def main() -> unit =\n"
        "  print(os::cwd())\n"
        "  print(os::pid())\n"
        "  print(os::hostname())\n",
        encoding="utf-8",
    )

    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from agm.cli import main; main()",
            "exec",
            "--no-trace",
            str(program),
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate()

    assert process.returncode == 0, stderr
    cwd, pid, hostname = stdout.splitlines()
    assert cwd == str(tmp_path)
    assert int(pid) == process.pid
    assert hostname


# ---------------------------------------------------------------------------
# std/os::temp-dir
# ---------------------------------------------------------------------------


def test_os_temp_dir_returns_the_host_temporary_directory(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", None)

    result = _run("import std/os\nprogram def main() -> unit = print(os::temp-dir())")

    assert result.ok
    assert capsys.readouterr().out == f"{tempfile.gettempdir()}\n"


def test_os_temp_dir_raises_encoding_error_when_tmpdir_is_not_valid_unicode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tempfile, "gettempdir", lambda: os.fsdecode(b"/tmp/h\xffome"))

    result = _run("import std/os\nprogram def main() -> unit =\n  let _ = os::temp-dir()\n")

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "EncodingError"


# ---------------------------------------------------------------------------
# std/os::chdir
# ---------------------------------------------------------------------------


def test_os_chdir_changes_the_working_directory_and_records_environ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub").mkdir()
    source = (
        "import std/env\n"
        "import std/os\n"
        "program def main() -> unit =\n"
        "  let before = os::cwd()\n"
        '  os::chdir("sub")\n'
        "  print(os::cwd() == before)\n"
        '  print(std/env::environ.get("OLDPWD") == before)\n'
        '  print(std/env::environ.get("PWD") == os::cwd())\n'
    )

    result = _run(source)

    assert result.ok
    assert capsys.readouterr().out == "false\ntrue\ntrue\n"
    # The run's host session ends when it returns, restoring the starting directory.
    assert Path(os.getcwd()) == tmp_path


def test_os_chdir_never_writes_the_real_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OLDPWD", raising=False)
    original_pwd = os.environ.get("PWD")
    (tmp_path / "sub").mkdir()

    result = _run(
        'import std/os\nprogram def main() -> unit = os::chdir("sub")',
    )

    assert result.ok
    assert "OLDPWD" not in os.environ
    assert os.environ.get("PWD") == original_pwd


def test_os_chdir_raises_fs_error_on_a_nonexistent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = _run('import std/os\nprogram def main() -> unit = os::chdir("missing")')

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "FsError"
    assert result.error.fields["operation"] == "chdir"
    assert result.error.fields["path"] == "missing"
    assert Path(os.getcwd()) == tmp_path


def test_os_chdir_raises_fs_error_on_a_path_containing_a_nul_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = _run('import std/os\nprogram def main() -> unit = os::chdir("invalid\\u0000path")')

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "FsError"
    assert result.error.fields["operation"] == "chdir"
    assert result.error.fields["path"] == "invalid\x00path"
    assert Path(os.getcwd()) == tmp_path


def test_os_chdir_raises_encoding_error_when_the_directory_before_is_not_valid_unicode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(os, "getcwd", lambda: "/tmp/h\udcffome")

    result = _run('import std/os\nprogram def main() -> unit = os::chdir("sub")')

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "EncodingError"


def test_os_chdir_raises_encoding_error_and_restores_the_directory_when_after_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "sub"
    target.mkdir()
    real_getcwd = os.getcwd

    def fake_getcwd() -> str:
        # Only the "after" lookup (once inside `target`) is invalid, so the
        # "before" snapshot used to restore stays real regardless of when it runs.
        cwd = real_getcwd()
        return "/tmp/h\udcffome" if Path(cwd) == target else cwd

    monkeypatch.setattr(os, "getcwd", fake_getcwd)

    result = _run('import std/os\nprogram def main() -> unit = os::chdir("sub")')

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "EncodingError"
    assert Path(real_getcwd()) == tmp_path


def test_os_chdir_resolves_a_symlink_to_its_physical_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (tmp_path / "link").symlink_to(real_dir)
    source = (
        "import std/env\n"
        "import std/os\n"
        "program def main() -> unit =\n"
        '  os::chdir("link")\n'
        "  print(os::cwd())\n"
        '  print(std/env::environ.get("PWD"))\n'
        '  print(std/env::environ.get("OLDPWD"))\n'
    )

    result = _run(source)

    assert result.ok
    cwd, pwd, oldpwd = capsys.readouterr().out.splitlines()
    assert cwd == str(real_dir)
    assert pwd == str(real_dir)
    assert oldpwd == str(tmp_path)


def test_os_chdir_start_directory_restore_skips_a_directory_that_no_longer_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run-end restore never crashes the run when the start directory is gone."""
    start = tmp_path / "start"
    target = tmp_path / "target"
    start.mkdir()
    target.mkdir()
    monkeypatch.chdir(start)
    source = (
        "import std/fs\n"
        "import std/os\n"
        "program def main() -> unit =\n"
        f'  os::chdir("{target}")\n'
        f'  fs::remove("{start}")\n'
    )

    result = _run(source)

    assert result.ok
    assert Path(os.getcwd()) == target


# ---------------------------------------------------------------------------
# std/os::is-interactive, platform, cpu-count, which, user
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interactive", [True, False])
def test_os_is_interactive_reflects_both_fds_being_terminals(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], interactive: bool
) -> None:
    monkeypatch.setattr(os, "isatty", lambda _fd: interactive)

    result = _run("import std/os\nprogram def main() -> unit = print(os::is-interactive())")

    assert result.ok
    assert capsys.readouterr().out == f"{'true' if interactive else 'false'}\n"


def test_os_platform_matches_sys_platform(capsys: pytest.CaptureFixture[str]) -> None:
    result = _run("import std/os\nprogram def main() -> unit = print(os::platform())")

    assert result.ok
    assert capsys.readouterr().out == f"{sys.platform}\n"


def test_os_cpu_count_is_at_least_one(capsys: pytest.CaptureFixture[str]) -> None:
    result = _run("import std/os\nprogram def main() -> unit = print(os::cpu-count() >= 1)")

    assert result.ok
    assert capsys.readouterr().out == "true\n"


def test_os_cpu_count_falls_back_to_one_when_the_host_cannot_tell(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(os, "process_cpu_count", lambda: None)

    result = _run("import std/os\nprogram def main() -> unit = print(os::cpu-count())")

    assert result.ok
    assert capsys.readouterr().out == "1\n"


def test_os_which_finds_an_executable_on_the_given_environments_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    binary = tmp_path / "greet"
    binary.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    expected = str(binary)
    source = (
        "import std/env::Environ\nimport std/os\nprogram def main() -> unit =\n"
        f'  let env = Environ({{"PATH": "{tmp_path}"}})\n'
        f'  print(os::which("greet", env).unwrap() == "{expected}")\n'
    )

    result = _run(source)

    assert result.ok
    assert capsys.readouterr().out == "true\n"


def test_os_which_returns_none_for_an_executable_that_does_not_exist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = (
        "import std/env::Environ\nimport std/os\nprogram def main() -> unit =\n"
        f'  let env = Environ({{"PATH": "{tmp_path}"}})\n'
        '  print(os::which("definitely-not-a-real-binary-xyz", env).is-none())\n'
    )

    result = _run(source)

    assert result.ok
    assert capsys.readouterr().out == "true\n"


def test_os_which_uses_the_host_default_path_when_the_environments_path_is_absent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`os.defpath` includes `/bin`, so `sh` resolves with no `PATH` in `env`."""
    source = (
        "import std/env::Environ\nimport std/os\n"
        "program def main() -> unit =\n"
        "  let env = Environ({})\n"
        '  print(os::which("sh", env).is-some())\n'
    )

    result = _run(source)

    assert result.ok
    assert capsys.readouterr().out == "true\n"


def test_os_user_returns_the_real_users_passwd_name(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class _Entry:
        pw_name = "alice"

    monkeypatch.setattr(pwd, "getpwuid", lambda _uid: _Entry())

    result = _run(
        'import std/os\nprogram def main() -> unit = print(os::user().unwrap() == "alice")'
    )

    assert result.ok
    assert capsys.readouterr().out == "true\n"


def test_os_user_returns_none_when_the_real_uid_has_no_passwd_entry(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _raise(_uid: int) -> object:
        raise KeyError(_uid)

    monkeypatch.setattr(pwd, "getpwuid", _raise)

    result = _run("import std/os\nprogram def main() -> unit = print(os::user().is-none())")

    assert result.ok
    assert capsys.readouterr().out == "true\n"


def test_os_user_raises_encoding_error_when_the_name_is_not_valid_unicode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Entry:
        pw_name = "h\udcffost"

    monkeypatch.setattr(pwd, "getpwuid", lambda _uid: _Entry())

    result = _run("import std/os\nprogram def main() -> unit =\n  let _ = os::user()\n")

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "EncodingError"
