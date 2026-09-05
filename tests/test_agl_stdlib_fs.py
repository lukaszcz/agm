"""End-to-end behavior of the ``std/fs`` extern module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import semver

from agm.agl import PipelineDriver
from agm.agl.capabilities import HostCapabilities
from agm.agl.modules.loader import load_graph
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import _wire_extern_registry
from agm.agl.runtime.externs import ExternRegistry
from agm.agl.scope.program import resolve_program
from agm.agl.typecheck.program import check_program
from agm.core import dry_run
from agm.packages.manifest import PackageManifest
from agm.packages.model import PackageInfo
from tests._agl_helpers import agl_roots

_STDLIB = Path(__file__).resolve().parent.parent / "stdlib"


def _run_file(source: str, path: Path, *, roots: RootSet) -> object:
    """Run *source*'s entry ``program def main() -> unit`` (no arguments)."""
    runtime = PipelineDriver()
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
        ('fs::write("missing/child.txt", "content")', "missing/child.txt", "write"),
        ('fs::append("missing/child.txt", "content")', "missing/child.txt", "append"),
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


def test_fs_writes_are_suppressed_and_logged_in_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    dry_run.set_enabled(True)

    result = _run_file(
        """import std/fs
program def main() -> unit =
  fs::write("created.txt", "first")
  fs::append("created.txt", " second")
  fs::mkdir("created")
  fs::copy("missing.txt", "copied.txt")
  fs::move("missing.txt", "moved.txt")
  fs::remove("missing.txt")
""",
        tmp_path / "main.agl",
        roots=agl_roots(),
    )

    assert result.ok
    assert not (tmp_path / "created.txt").exists()
    assert not (tmp_path / "created").exists()
    assert capsys.readouterr().out == (
        "dry-run: agm write-file created.txt\n"
        "dry-run: agm append-file created.txt\n"
        "dry-run: agm mkdir created\n"
        "dry-run: agm copy-file missing.txt copied.txt\n"
        "dry-run: agm move missing.txt moved.txt\n"
        "dry-run: agm unlink missing.txt\n"
    )


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
    )

    assert len(diagnostics) == 1


def test_packaged_program_reads_a_resource_through_std_fs_in_a_temp_package(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package_root = tmp_path / "review-tools"
    entry = package_root / "review_tools" / "read_prompt.agl"
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
            roots=frozenset({_STDLIB, package_root}),
            packages=(package,),
            stdlib_roots=frozenset({_STDLIB}),
        ),
    )

    assert result.ok
    assert capsys.readouterr().out == "Package prompt.\n\n"
