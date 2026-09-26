"""``std/os`` process, environment, and temporary-directory operations."""

from __future__ import annotations

import json
import os
import pwd
import shutil
import signal
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
from tests._agl_helpers import NONE_FIELD, agl_roots, repl_session, some_field
from tests._proc_helpers import wait_for_signal_ignored_script


def _roots() -> RootSet:
    return agl_roots()


def _run(source: str, *, trace_file: Path | None = None) -> RunResult:
    return PipelineDriver(get_sandbox_context=None).run(
        source, roots=_roots(), trace_file=trace_file
    )


def _exit_program(call: str) -> str:
    return f"import std/os\nprogram def main() -> unit = os::exit({call})\n"


def _write_fake_executable(path: Path, body: str) -> None:
    """Write an executable shell script at *path* with *body* as its script."""
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


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


# ---------------------------------------------------------------------------
# std/os::edit, std/os::edit-text, std/os::open, std/os::LaunchError
# ---------------------------------------------------------------------------


def _launch_program(call: str, target: str, env_literal: str) -> str:
    return (
        "import std/env::Environ\n"
        "import std/os\n"
        "program def main() -> unit =\n"
        f"  let env = {env_literal}\n"
        f'  os::{call}("{target}", env)\n'
    )


def _edit_program(file: Path, env_literal: str) -> str:
    return _launch_program("edit", str(file), env_literal)


def _open_program(target: str, env_literal: str) -> str:
    return _launch_program("open", target, env_literal)


def test_os_edit_prefers_visual_over_editor(tmp_path: Path) -> None:
    visual = tmp_path / "visual.sh"
    editor = tmp_path / "editor.sh"
    marker = tmp_path / "marker"
    _write_fake_executable(visual, f'printf visual > "{marker}"')
    _write_fake_executable(editor, f'printf editor > "{marker}"')
    file = tmp_path / "note.txt"
    file.write_text("x", encoding="utf-8")

    result = _run(_edit_program(file, f'Environ({{"VISUAL": "{visual}", "EDITOR": "{editor}"}})'))

    assert result.ok
    assert marker.read_text(encoding="utf-8") == "visual"


@pytest.mark.parametrize(
    ("visual", "editor_is_set", "expected"),
    [
        ("", True, "editor"),
        ("", False, "micro"),
        (None, False, "micro"),
    ],
)
def test_os_edit_falls_through_empty_visual_and_editor_to_the_next_candidate(
    tmp_path: Path, visual: str | None, editor_is_set: bool, expected: str
) -> None:
    """`VISUAL`/`EDITOR` set to `""` is treated the same as being unset."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "marker"
    editor = tmp_path / "editor.sh"
    _write_fake_executable(editor, f'printf editor > "{marker}"')
    _write_fake_executable(bin_dir / "micro", f'printf micro > "{marker}"')
    sh_path = shutil.which("sh")
    assert sh_path is not None
    (bin_dir / "sh").symlink_to(sh_path)
    file = tmp_path / "note.txt"
    file.write_text("x", encoding="utf-8")

    env_vars = {"EDITOR": str(editor) if editor_is_set else "", "PATH": str(bin_dir)}
    if visual is not None:
        env_vars["VISUAL"] = visual
    env_literal = "Environ({" + ", ".join(f'"{k}": "{v}"' for k, v in env_vars.items()) + "})"

    result = _run(_edit_program(file, env_literal))

    assert result.ok
    assert marker.read_text(encoding="utf-8") == expected


def test_os_edit_supports_an_editor_value_with_arguments(tmp_path: Path) -> None:
    """`EDITOR`/`VISUAL` is a shell snippet; extra words become leading arguments."""
    editor = tmp_path / "editor.sh"
    marker = tmp_path / "marker"
    _write_fake_executable(editor, f'printf "%s %s" "$1" "$2" > "{marker}"')
    file = tmp_path / "note.txt"
    file.write_text("x", encoding="utf-8")

    result = _run(_edit_program(file, f'Environ({{"EDITOR": "{editor} --flag"}})'))

    assert result.ok
    assert marker.read_text(encoding="utf-8") == f"--flag {file}"


def test_os_edit_falls_back_to_micro_then_vi_found_on_path(tmp_path: Path) -> None:
    """Hermetic: `PATH` names only `bin_dir` (plus a symlinked `sh`), so this can never
    resolve and launch a real editor installed on the host."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "marker"
    _write_fake_executable(bin_dir / "micro", f'printf micro > "{marker}"')
    _write_fake_executable(bin_dir / "vi", f'printf vi > "{marker}"')
    sh_path = shutil.which("sh")
    assert sh_path is not None
    (bin_dir / "sh").symlink_to(sh_path)
    file = tmp_path / "note.txt"
    file.write_text("x", encoding="utf-8")
    path = str(bin_dir)

    result = _run(_edit_program(file, f'Environ({{"PATH": "{path}"}})'))

    assert result.ok
    assert marker.read_text(encoding="utf-8") == "micro"

    (bin_dir / "micro").unlink()
    result = _run(_edit_program(file, f'Environ({{"PATH": "{path}"}})'))

    assert result.ok
    assert marker.read_text(encoding="utf-8") == "vi"


def test_os_edit_raises_launch_error_when_no_editor_is_found(tmp_path: Path) -> None:
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    file = tmp_path / "note.txt"
    file.write_text("x", encoding="utf-8")

    result = _run(_edit_program(file, f'Environ({{"PATH": "{empty_bin}"}})'))

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "LaunchError"
    assert result.error.fields["command"] == "vi"
    assert result.error.fields["exit-code"] == NONE_FIELD


def test_os_edit_raises_launch_error_when_the_editor_exits_non_zero(tmp_path: Path) -> None:
    editor = tmp_path / "editor.sh"
    _write_fake_executable(editor, "exit 3")
    file = tmp_path / "note.txt"
    file.write_text("x", encoding="utf-8")

    result = _run(_edit_program(file, f'Environ({{"EDITOR": "{editor}"}})'))

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "LaunchError"
    assert result.error.fields["command"] == str(editor)
    assert result.error.fields["exit-code"] == some_field(3)


def test_os_edit_raises_launch_error_with_no_exit_code_when_the_shell_reports_not_found(
    tmp_path: Path,
) -> None:
    """`sh -c` exits 127 for a command it cannot find; that means nothing was launched,
    not that the editor itself exited with status 127."""
    file = tmp_path / "note.txt"
    file.write_text("x", encoding="utf-8")
    missing_editor = str(tmp_path / "no-such-editor")

    result = _run(_edit_program(file, f'Environ({{"EDITOR": "{missing_editor}"}})'))

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "LaunchError"
    assert result.error.fields["command"] == missing_editor
    assert result.error.fields["exit-code"] == NONE_FIELD


def test_os_edit_raises_launch_error_when_sh_is_absent_from_the_given_path(
    tmp_path: Path,
) -> None:
    """A `PATH` with no `sh` on it must map to `LaunchError`, not an uncaught error."""
    editor = tmp_path / "editor.sh"
    _write_fake_executable(editor, "exit 0")
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    file = tmp_path / "note.txt"
    file.write_text("x", encoding="utf-8")

    result = _run(_edit_program(file, f'Environ({{"EDITOR": "{editor}", "PATH": "{empty_bin}"}})'))

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "LaunchError"
    assert result.error.fields["command"] == "sh"
    assert result.error.fields["exit-code"] == NONE_FIELD


def test_os_edit_gets_exactly_env_as_the_child_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The child sees only `env`'s variables — never the AGM process environment."""
    monkeypatch.setenv("AGM_TEST_PARENT_ONLY", "leak")
    editor = tmp_path / "editor.sh"
    marker = tmp_path / "marker"
    _write_fake_executable(
        editor,
        f'if [ "$AGM_TEST_CHILD_ONLY" = "present" ] && [ -z "$AGM_TEST_PARENT_ONLY" ]; then\n'
        f'  printf ok > "{marker}"\n'
        "else\n"
        f'  printf bad > "{marker}"\n'
        "fi\n",
    )
    file = tmp_path / "note.txt"
    file.write_text("x", encoding="utf-8")

    result = _run(
        _edit_program(file, f'Environ({{"EDITOR": "{editor}", "AGM_TEST_CHILD_ONLY": "present"}})')
    )

    assert result.ok
    assert marker.read_text(encoding="utf-8") == "ok"


def test_os_edit_passes_a_file_path_containing_spaces(tmp_path: Path) -> None:
    editor = tmp_path / "editor.sh"
    marker = tmp_path / "marker"
    _write_fake_executable(editor, f'printf "%s" "$1" > "{marker}"')
    file = tmp_path / "my note.txt"
    file.write_text("x", encoding="utf-8")

    result = _run(_edit_program(file, f'Environ({{"EDITOR": "{editor}"}})'))

    assert result.ok
    assert marker.read_text(encoding="utf-8") == str(file)


def test_os_edit_passes_an_absolute_path_for_a_dash_or_plus_prefixed_relative_file_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative name starting with `-`/`+` must never be parsed as an editor option."""
    monkeypatch.chdir(tmp_path)
    editor = tmp_path / "editor.sh"
    marker = tmp_path / "marker"
    _write_fake_executable(editor, f'printf "%s" "$1" > "{marker}"')
    relative = "-note.txt"
    (tmp_path / relative).write_text("x", encoding="utf-8")

    result = _run(_edit_program(Path(relative), f'Environ({{"EDITOR": "{editor}"}})'))

    assert result.ok
    assert marker.read_text(encoding="utf-8") == str(tmp_path / relative)


def test_os_edit_ignores_sigint_while_waiting_and_restores_the_handler_after(
    tmp_path: Path, default_sigint: None
) -> None:
    """The editor waits for the parent's ignore to actually be installed (see B1's
    `wait_for_signal_ignored_script`) before signalling it, avoiding a race with the
    ignore only starting once the editor itself has been spawned."""
    editor = tmp_path / "editor.sh"
    _write_fake_executable(
        editor,
        wait_for_signal_ignored_script("$AGM_TEST_SIGINT_TARGET", signal.SIGINT)
        + 'kill -INT "$AGM_TEST_SIGINT_TARGET"\nexit 0\n',
    )
    file = tmp_path / "note.txt"
    file.write_text("x", encoding="utf-8")
    previous_handler = signal.getsignal(signal.SIGINT)

    result = _run(
        _edit_program(
            file,
            f'Environ({{"EDITOR": "{editor}", "AGM_TEST_SIGINT_TARGET": "{os.getpid()}"}})',
        )
    )

    assert result.ok
    assert signal.getsignal(signal.SIGINT) is previous_handler


def test_os_edit_text_round_trips_through_the_editor_and_removes_the_temp_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    editor = tmp_path / "editor.sh"
    path_marker = tmp_path / "path-marker"
    _write_fake_executable(
        editor,
        f'printf "%s" "$1" > "{path_marker}"\nprintf " world" >> "$1"\n',
    )
    source = (
        "import std/env::Environ\n"
        "import std/os\n"
        "program def main() -> unit =\n"
        f'  let env = Environ({{"EDITOR": "{editor}"}})\n'
        '  print(os::edit-text("hello", ".txt", env))\n'
    )

    result = _run(source)

    assert result.ok
    assert capsys.readouterr().out == "hello world\n"
    temp_file = Path(path_marker.read_text(encoding="utf-8"))
    assert temp_file.name.endswith(".txt")
    assert not temp_file.exists()


def test_os_edit_text_uses_the_default_suffix_when_omitted(tmp_path: Path) -> None:
    editor = tmp_path / "editor.sh"
    path_marker = tmp_path / "path-marker"
    _write_fake_executable(editor, f'printf "%s" "$1" > "{path_marker}"')
    source = (
        "import std/env::Environ\n"
        "import std/os\n"
        "program def main() -> unit =\n"
        f'  let env = Environ({{"EDITOR": "{editor}"}})\n'
        '  let _ = os::edit-text("hello", env = env)\n'
        "  ()\n"
    )

    result = _run(source)

    assert result.ok
    temp_file = Path(path_marker.read_text(encoding="utf-8"))
    assert temp_file.name.endswith(".md")


def test_os_edit_text_leaves_the_temp_file_for_the_host_session_when_the_editor_fails(
    tmp_path: Path,
) -> None:
    """A failed `edit` leaves the temp file for the host session's own close to remove --
    observable only while the session stays open (a one-shot run's session ends, and
    removes it, the instant the run itself does)."""
    editor = tmp_path / "editor.sh"
    path_marker = tmp_path / "path-marker"
    _write_fake_executable(editor, f'printf "%s" "$1" > "{path_marker}"\nexit 5\n')
    session = repl_session()
    assert session.eval_entry("import std/env").ok
    assert session.eval_entry("import std/os").ok

    entry = session.eval_entry(
        f'let env = env::Environ({{"EDITOR": "{editor}"}})\n'
        'let _ = os::edit-text("hello", ".txt", env)\n'
    )

    assert not entry.ok
    temp_file = Path(path_marker.read_text(encoding="utf-8"))
    assert temp_file.exists()

    session.close()

    assert not temp_file.exists()


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_os_open_uses_the_platform_default_opener(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    opener_name = "open" if platform == "darwin" else "xdg-open"
    opener = tmp_path / opener_name
    marker = tmp_path / "marker"
    _write_fake_executable(opener, f'printf "%s" "$1" > "{marker}"')
    target = tmp_path / "doc.txt"
    target.write_text("x", encoding="utf-8")

    result = _run(_open_program(str(target), f'Environ({{"PATH": "{tmp_path}"}})'))

    assert result.ok
    assert marker.read_text(encoding="utf-8") == str(target)


def test_os_open_raises_launch_error_when_the_opener_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()

    result = _run(_open_program("target.txt", f'Environ({{"PATH": "{empty_bin}"}})'))

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "LaunchError"
    assert result.error.fields["command"] == "xdg-open"
    assert result.error.fields["exit-code"] == NONE_FIELD


def test_os_open_raises_launch_error_when_the_opener_exits_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    opener = tmp_path / "xdg-open"
    _write_fake_executable(opener, "exit 7")

    result = _run(_open_program("target.txt", f'Environ({{"PATH": "{tmp_path}"}})'))

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "LaunchError"
    assert result.error.fields["command"] == "xdg-open"
    assert result.error.fields["exit-code"] == some_field(7)


def test_os_open_raises_launch_error_when_the_resolved_opener_vanishes_before_exec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A which-then-exec race (the opener resolves, then is gone by the time it is spawned)
    must map to `LaunchError`, not an uncaught `FileNotFoundError`."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", lambda name, path=None: str(tmp_path / "vanished-opener"))

    result = _run(_open_program("target.txt", f'Environ({{"PATH": "{tmp_path}"}})'))

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "LaunchError"
    assert result.error.fields["command"] == "xdg-open"
    assert result.error.fields["exit-code"] == NONE_FIELD
