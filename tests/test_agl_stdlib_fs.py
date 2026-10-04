"""End-to-end behavior of the ``std/fs`` extern module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import semver

from agm.agl import PipelineDriver
from agm.agl.capabilities import HostCapabilities
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import _wire_extern_registry
from agm.agl.runtime.externs import ExternRegistry
from agm.agl.scope.program import resolve_program
from agm.agl.typecheck.program import check_program
from agm.packages.manifest import PackageManifest
from agm.packages.model import PackageInfo
from tests._agl_helpers import agl_roots
from tests.agl.module_graph import load_graph

_STDLIB = Path(__file__).resolve().parent.parent / "packages" / "stdlib"


def _run_file(source: str, path: Path, *, roots: RootSet) -> object:
    """Run *source*'s entry ``program def main() -> unit`` (no arguments)."""
    runtime = PipelineDriver(resolve_agent_spec=None, get_sandbox_context=None)
    prepared = PipelineDriver.prepare_program(source, entry_path=path, roots=roots)
    discovery = runtime.discover_programs(prepared)
    if discovery.compiled is None:
        return runtime.run_prepared(prepared)
    return runtime.run_prepared(prepared, compiled=discovery.compiled, select_default_program=True)


def test_fs_operations_use_typed_text_paths_relative_to_the_invocation_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    entry = tmp_path / "main.agl"
    (tmp_path / "listed.txt").write_text("present", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = _run_file(
        """import std/fs
program def main() -> unit =
  fs::write("created.txt", "first")
  fs::append("created.txt", " second")
  print(fs::read("created.txt"))
  print(fs::exists("created.txt"))
  print(fs::exists("missing.txt"))
  print(fs::list("."))
""",
        entry,
        roots=agl_roots(),
    )

    assert result.ok
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "first second"
    output = capsys.readouterr().out
    assert "first second\ntrue\nfalse\n" in output
    assert '"created.txt"' in output
    assert '"listed.txt"' in output


@pytest.mark.parametrize(
    ("call", "path", "operation"),
    (
        ('fs::read("missing.txt")', "missing.txt", "read"),
        ('fs::list("missing")', "missing", "list"),
        ('fs::remove("missing.txt")', "missing.txt", "remove"),
        ('fs::copy("missing.txt", "other.txt")', "missing.txt", "copy"),
        ('fs::move("missing.txt", "other.txt")', "missing.txt", "move"),
    ),
)
def test_fs_operation_errors_raise_fs_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, call: str, path: str, operation: str
) -> None:
    monkeypatch.chdir(tmp_path)
    result = _run_file(
        f"""import std/fs
program def main() -> unit =
  let _ = {call}
""",
        tmp_path / "main.agl",
        roots=agl_roots(),
    )

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "FsError"
    assert result.error.fields["path"] == path
    assert result.error.fields["operation"] == operation


@pytest.mark.parametrize(
    ("call", "path", "operation"),
    (
        ('fs::read("invalid\\u0000path")', "invalid\x00path", "read"),
        ('fs::write("invalid\\u0000path", "content")', "invalid\x00path", "write"),
        ('fs::append("invalid\\u0000path", "content")', "invalid\x00path", "append"),
        ('fs::list("invalid\\u0000path")', "invalid\x00path", "list"),
        ('fs::exists("invalid\\u0000path")', "invalid\x00path", "exists"),
        ('fs::is-file("invalid\\u0000path")', "invalid\x00path", "is-file"),
        ('fs::is-dir("invalid\\u0000path")', "invalid\x00path", "is-dir"),
        ('fs::glob("invalid\\u0000*.txt")', "invalid\x00*.txt", "glob"),
        ('fs::mkdir("invalid\\u0000path")', "invalid\x00path", "mkdir"),
        ('fs::remove("invalid\\u0000path")', "invalid\x00path", "remove"),
        ('fs::copy("invalid\\u0000path", "other.txt")', "invalid\x00path", "copy"),
        ('fs::move("invalid\\u0000path", "other.txt")', "invalid\x00path", "move"),
    ),
)
def test_fs_invalid_paths_raise_fs_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, call: str, path: str, operation: str
) -> None:
    monkeypatch.chdir(tmp_path)
    result = _run_file(
        f"""import std/fs
program def main() -> unit =
  let _ = {call}
""",
        tmp_path / "main.agl",
        roots=agl_roots(),
    )

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "FsError"
    assert result.error.fields["path"] == path
    assert result.error.fields["operation"] == operation


def test_fs_list_raises_fs_error_on_an_undecodable_directory_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / os.fsdecode(b"n\xffm")).write_bytes(b"data")

    result = _run_file(
        """import std/fs
program def main() -> unit =
  let _ = fs::list(".")
""",
        tmp_path / "main.agl",
        roots=agl_roots(),
    )

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "FsError"
    assert result.error.fields["path"] == "."
    assert result.error.fields["operation"] == "list"


def test_fs_glob_raises_fs_error_on_an_undecodable_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / os.fsdecode(b"n\xffm")).write_bytes(b"data")

    result = _run_file(
        """import std/fs
program def main() -> unit =
  let _ = fs::glob("*")
""",
        tmp_path / "main.agl",
        roots=agl_roots(),
    )

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "FsError"
    assert result.error.fields["path"] == "*"
    assert result.error.fields["operation"] == "glob"


def test_fs_try_read_of_an_invalid_path_returns_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    result = _run_file(
        """import std/fs
program def main() -> unit = print(fs::try-read("invalid\\u0000path").is-err())
""",
        tmp_path / "main.agl",
        roots=agl_roots(),
    )

    assert result.ok
    assert capsys.readouterr().out == "true\n"


def test_fs_names_its_predicates_only_by_their_public_spelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``fs`` names its predicates only as ``is-file``/``is-dir``: the Python
    companion spelling behind them does not resolve."""
    monkeypatch.chdir(tmp_path)
    result = _run_file(
        """import std/fs
program def main() -> unit =
  let _ = fs::is_file("file.txt")
""",
        tmp_path / "main.agl",
        roots=agl_roots(),
    )

    assert not result.ok
    assert result.error is None
    assert result.diagnostics


def test_fs_extensions_work_in_an_isolated_invocation_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)

    result = _run_file(
        """import std/fs
program def main() -> unit =
  fs::mkdir("nested/deep")
  fs::write("nested/source.txt", "contents")
  fs::copy("nested/source.txt", "nested/copy.txt")
  fs::move("nested/copy.txt", "nested/deep/moved.txt")
  print(fs::is-file("nested/source.txt"))
  print(fs::is-dir("nested/deep"))
  print(fs::try-read("missing.txt").is-err())
  print(fs::glob("nested/*.txt"))
  print(fs::glob("missing/*.txt"))
  fs::remove("nested/deep/moved.txt")
""",
        tmp_path / "main.agl",
        roots=agl_roots(),
    )

    assert result.ok
    assert not (tmp_path / "nested" / "deep" / "moved.txt").exists()
    assert (tmp_path / "nested" / "source.txt").read_text(encoding="utf-8") == "contents"
    assert capsys.readouterr().out == (
        f'true\ntrue\ntrue\n["{os.path.join("nested", "source.txt")}"]\n[]\n'
    )


def test_fs_write_append_and_mkdir_create_missing_parent_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)

    result = _run_file(
        """import std/fs
program def main() -> unit =
  fs::write("created/by/write/deep/file.txt", "contents")
  fs::append("created/by/append/deep/file.txt", "appended")
  fs::mkdir("created/by/mkdir/deep")
  print(fs::read("created/by/write/deep/file.txt"))
  print(fs::read("created/by/append/deep/file.txt"))
  print(fs::is-dir("created/by/mkdir/deep"))
""",
        tmp_path / "main.agl",
        roots=agl_roots(),
    )

    assert result.ok
    assert (tmp_path / "created" / "by" / "write" / "deep" / "file.txt").is_file()
    assert (tmp_path / "created" / "by" / "append" / "deep" / "file.txt").is_file()
    assert (tmp_path / "created" / "by" / "mkdir" / "deep").is_dir()
    assert capsys.readouterr().out == "contents\nappended\ntrue\n"


def test_fs_copy_and_move_create_missing_destination_parent_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)

    result = _run_file(
        """import std/fs
program def main() -> unit =
  fs::write("source.txt", "contents")
  fs::copy("source.txt", "created/by/copy/file.txt")
  fs::move("source.txt", "created/by/move/file.txt")
  print(fs::read("created/by/copy/file.txt"))
  print(fs::read("created/by/move/file.txt"))
  print(fs::exists("source.txt"))
""",
        tmp_path / "main.agl",
        roots=agl_roots(),
    )

    assert result.ok
    assert capsys.readouterr().out == "contents\ncontents\nfalse\n"


@pytest.mark.parametrize("operation", ("copy", "move"))
def test_fs_copy_and_move_of_missing_source_create_no_destination_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    monkeypatch.chdir(tmp_path)

    result = _run_file(
        f"""import std/fs
program def main() -> unit =
  fs::{operation}("missing.txt", "out/a/b.txt")
""",
        tmp_path / "main.agl",
        roots=agl_roots(),
    )

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "FsError"
    assert result.error.fields["path"] == "missing.txt"
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("operation", ("copy", "move"))
def test_fs_copy_and_move_report_uncreatable_destination_parent_by_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "source.txt").write_text("contents", encoding="utf-8")
    (tmp_path / "blocker").write_text("file", encoding="utf-8")

    result = _run_file(
        f"""import std/fs
program def main() -> unit =
  fs::{operation}("source.txt", "blocker/sub/b.txt")
""",
        tmp_path / "main.agl",
        roots=agl_roots(),
    )

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "FsError"
    assert result.error.fields["path"] == "blocker/sub/b.txt"
    assert (tmp_path / "source.txt").is_file()


def test_fs_externs_honor_the_existing_extern_capability_gate() -> None:
    graph = load_graph(
        'import std/fs\nprogram def main() -> unit =\n  let _ = fs::exists("file")\n',
        entry_path=None,
        roots=agl_roots(),
    )
    checked = check_program(resolve_program(graph), HostCapabilities(supports_extern=False))

    diagnostics = _wire_extern_registry(
        checked=checked,
        capabilities=HostCapabilities(supports_extern=False),
        registry=ExternRegistry(),
        companion_paths={
            module_id: module.companion_path for module_id, module in graph.modules.items()
        },
        packages=(),
    )

    assert len(diagnostics) == 1


def test_packaged_program_reads_a_resource_through_std_fs_in_a_temp_package(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package_root = tmp_path / "review-tools"
    entry = package_root / "src" / "read_prompt.agl"
    entry.parent.mkdir(parents=True)
    entry.write_text(
        """import std/fs
let prompt = resource("prompts/package-prompt.md")
program def main() -> unit =
  print(fs::read(prompt))
""",
        encoding="utf-8",
    )
    prompt = package_root / "prompts" / "package-prompt.md"
    prompt.parent.mkdir()
    prompt.write_text("Package prompt.\n", encoding="utf-8")
    package = PackageInfo(
        package_root,
        PackageManifest("review_tools", semver.Version.parse("0.2.0")),
    )

    result = _run_file(
        entry.read_text(encoding="utf-8"),
        entry,
        roots=RootSet(
            roots=frozenset(),
            packages=(package,),
            stdlib_roots=frozenset({_STDLIB}),
        ),
    )

    assert result.ok
    assert capsys.readouterr().out == "Package prompt.\n\n"
