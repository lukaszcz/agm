"""End-to-end behavior of the ``std/fs`` extern module."""

from __future__ import annotations

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

_STDLIB = Path(__file__).resolve().parent.parent / "stdlib"


def _run_file(source: str, path: Path, *, roots: RootSet) -> object:
    runtime = PipelineDriver()
    prepared = PipelineDriver.prepare_program(source, entry_path=path, roots=roots)
    discovery = runtime.discover_params(prepared)
    if discovery.checked is None:
        return runtime.run_prepared(prepared)
    preflight = runtime.preflight_params(prepared, compiled=discovery.compiled)
    if not preflight.result.ok:
        return preflight.result
    (program,) = [item for item in discovery.programs if item.module.is_entry]
    assert preflight.executable is not None
    return runtime.run_prepared(
        prepared,
        compiled=discovery.compiled,
        executable=preflight.executable,
        program_symbol=preflight.executable.program_symbols[program.node_id],
    )


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
        roots=RootSet(roots=frozenset({_STDLIB})),
    )

    assert result.ok
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "first second"
    output = capsys.readouterr().out
    assert "first second\ntrue\nfalse\n" in output
    assert '"created.txt"' in output
    assert '"listed.txt"' in output


@pytest.mark.parametrize(
    ("call", "python_type"),
    (
        ('fs::read("missing.txt")', "FileNotFoundError"),
        ('fs::write("missing/child.txt", "content")', "FileNotFoundError"),
        ('fs::append("missing/child.txt", "content")', "FileNotFoundError"),
        ('fs::list("missing")', "FileNotFoundError"),
    ),
)
def test_fs_operation_errors_cross_the_extern_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, call: str, python_type: str
) -> None:
    monkeypatch.chdir(tmp_path)
    result = _run_file(
        f"""import std/fs
program def main() -> unit =
  let _ = {call}
""",
        tmp_path / "main.agl",
        roots=RootSet(roots=frozenset({_STDLIB})),
    )

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "ExternError"
    assert result.error.fields["python_type"] == python_type


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
""",
        tmp_path / "main.agl",
        roots=RootSet(roots=frozenset({_STDLIB})),
    )

    assert result.ok
    assert not (tmp_path / "created.txt").exists()
    assert capsys.readouterr().out == (
        "dry-run: agm write-file created.txt\ndry-run: agm append-file created.txt\n"
    )


def test_fs_externs_honor_the_existing_extern_capability_gate() -> None:
    graph = load_graph(
        'import std/fs\nprogram def main() -> unit =\n  let _ = fs::exists("file")\n',
        entry_path=None,
        roots=RootSet(roots=frozenset({_STDLIB})),
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


def test_packaged_program_reads_a_resource_through_std_fs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    package_root = Path(__file__).parent / "agl" / "packages" / "valid"
    package = PackageInfo(
        package_root,
        PackageManifest("review_tools", semver.Version.parse("0.2.0")),
    )
    entry = package_root / "review_tools" / "read_prompt.agl"

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
