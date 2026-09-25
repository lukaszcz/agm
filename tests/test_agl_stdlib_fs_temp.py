"""Session-scoped temporary files and directories from ``std/fs``."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from agm.agl import PipelineDriver
from agm.agl.pipeline import RunResult
from agm.agl.semantics.values import BoolValue, Value
from agm.core import dry_run
from tests._agl_helpers import agl_roots


@pytest.fixture
def os_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the host's temporary directory at an isolated, initially empty directory."""
    directory = tmp_path / "os-temp"
    directory.mkdir()
    # ``tempfile.gettempdir`` caches process-wide; ``tempdir`` overrides the cache.
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
    monkeypatch.chdir(tmp_path)
    return directory


def _run(source: str, entry: Path, *, host_settings: dict[str, Value] | None = None) -> RunResult:
    runtime = PipelineDriver(get_sandbox_context=None)
    prepared = PipelineDriver.prepare_program(source, entry_path=entry, roots=agl_roots())
    discovery = runtime.discover_programs(prepared)
    assert discovery.compiled is not None
    return runtime.run_prepared(
        prepared,
        compiled=discovery.compiled,
        select_default_program=True,
        builtin_host_settings=host_settings,
    )


def _entries(directory: Path) -> list[Path]:
    return sorted(directory.iterdir())


def test_temp_paths_are_created_unique_in_the_os_temp_dir_and_removed_after_the_run(
    os_temp: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = _run(
        """import std/fs
import std/path
program def main() -> unit =
  let first = fs::temp-file()
  let second = fs::temp-file(suffix = ".json")
  let dir = fs::temp-dir()
  fs::write(path::join([dir, "nested/inner.txt"]), "inner")
  print(first != second)
  print(fs::read(first) == "")
  print(second.ends-with(".json"))
  print(fs::is-dir(dir))
  print(path::dirname(first) == fs::os-temp-dir())
  print(path::dirname(dir) == fs::os-temp-dir())
  print(fs::list(fs::os-temp-dir()).size())
""",
        os_temp.parent / "main.agl",
    )

    assert result.ok
    assert capsys.readouterr().out == "true\ntrue\ntrue\ntrue\ntrue\ntrue\n3\n"
    assert _entries(os_temp) == []


def test_temp_paths_are_removed_when_the_program_raises(os_temp: Path) -> None:
    result = _run(
        """import std/fs
program def main() -> unit =
  let _ = fs::temp-file()
  let _ = fs::temp-dir()
  raise Abort(message = "boom")
""",
        os_temp.parent / "main.agl",
    )

    assert not result.ok
    assert _entries(os_temp) == []


def test_temp_paths_are_removed_when_the_program_exits(os_temp: Path) -> None:
    with pytest.raises(SystemExit):
        _run(
            """import std/fs
import std/process
program def main() -> unit =
  let _ = fs::temp-file()
  process::exit(3)
""",
            os_temp.parent / "main.agl",
        )

    assert _entries(os_temp) == []


def test_temp_paths_the_program_already_removed_or_moved_are_skipped(os_temp: Path) -> None:
    result = _run(
        """import std/fs
program def main() -> unit =
  fs::remove(fs::temp-file())
  fs::remove(fs::temp-dir())
  fs::move(fs::temp-file(), "kept.txt")
""",
        os_temp.parent / "main.agl",
    )

    assert result.ok
    assert _entries(os_temp) == []
    assert (os_temp.parent / "kept.txt").is_file()


def test_debug_setting_written_by_the_program_keeps_temp_paths(os_temp: Path) -> None:
    result = _run(
        """import std/config
import std/fs
program def main() -> unit =
  let _ = fs::temp-file()
  let _ = fs::temp-dir()
  std/config::debug := true
""",
        os_temp.parent / "main.agl",
    )

    assert result.ok
    assert len(_entries(os_temp)) == 2


def test_debug_seed_keeps_temp_paths_unless_the_program_turns_it_off(os_temp: Path) -> None:
    kept = _run(
        """import std/fs
program def main() -> unit =
  let _ = fs::temp-file()
""",
        os_temp.parent / "main.agl",
        host_settings={"debug": BoolValue(True)},
    )
    assert kept.ok
    assert len(_entries(os_temp)) == 1

    removed = _run(
        """import std/config
import std/fs
program def main() -> unit =
  let _ = fs::temp-file()
  std/config::debug := false
""",
        os_temp.parent / "other.agl",
        host_settings={"debug": BoolValue(True)},
    )
    assert removed.ok
    assert len(_entries(os_temp)) == 1


def test_temp_creation_is_suppressed_and_logged_in_dry_run(
    os_temp: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dry_run.set_enabled(True)

    result = _run(
        """import std/fs
import std/path
program def main() -> unit =
  let file = fs::temp-file(suffix = ".md")
  let dir = fs::temp-dir()
  print(file != dir)
  print(path::dirname(file) == fs::os-temp-dir())
""",
        os_temp.parent / "main.agl",
    )

    assert result.ok
    assert _entries(os_temp) == []
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("dry-run: agm make-temp-file ")
    assert lines[0].endswith(".md")
    assert lines[1].startswith("dry-run: agm make-temp-dir ")
    assert lines[2:4] == ["true", "true"]


@pytest.mark.parametrize(
    ("call", "operation"),
    (
        ('fs::temp-file(suffix = "/missing/x")', "temp-file"),
        ('fs::temp-file(suffix = "\\u0000")', "temp-file"),
    ),
)
def test_invalid_temp_file_suffix_raises_fs_error(os_temp: Path, call: str, operation: str) -> None:
    result = _run(
        f"""import std/fs
program def main() -> unit =
  let _ = {call}
""",
        os_temp.parent / "main.agl",
    )

    assert not result.ok
    error = result.error
    assert error is not None
    assert error.type_name == "FsError"
    assert error.fields["operation"] == operation
    assert _entries(os_temp) == []


@pytest.mark.parametrize("function", ("temp-file", "temp-dir"))
def test_temp_creation_raises_fs_error_when_tmpdir_is_not_valid_unicode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, function: str
) -> None:
    monkeypatch.setattr(tempfile, "gettempdir", lambda: os.fsdecode(b"/tmp/h\xffome"))

    result = _run(
        f"""import std/fs
program def main() -> unit =
  let _ = fs::{function}()
""",
        tmp_path / "main.agl",
    )

    assert not result.ok
    error = result.error
    assert error is not None
    assert error.type_name == "FsError"
    assert error.fields["operation"] == function
