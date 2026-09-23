"""Precompiled libraries remain reusable across programs and fresh processes."""

from __future__ import annotations

import hashlib
import os
import pickle
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from agm.agl.artifact_cache import clear_retained_artifacts
from agm.agl.modules.parsed_module_cache import clear_parsed_module_cache
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import PipelineDriver, RunResult
from tests._agl_helpers import run_inline_command


def _run(root: Path, source: str, *, dry_run: bool = False) -> subprocess.CompletedProcess[str]:
    cli = ["--dry-run"] if dry_run else []
    cmd = [
        sys.executable,
        "-m",
        "agm.cli",
        *cli,
        "exec",
        "--no-stdlib",
        "-I",
        str(root),
        "-c",
        source,
    ]
    return subprocess.run(
        cmd,
        env={**os.environ, "XDG_CACHE_HOME": str(root / "cache")},
        capture_output=True,
        text=True,
        check=False,
    )


def test_compiled_library_is_reused_by_different_programs(tmp_path: Path) -> None:
    (tmp_path / "helper.agl").write_text(
        "builtin def print[T](value: T) -> unit\ndef double(value: int) -> int = value * 2\n"
    )
    first = _run(tmp_path, "import helper::*\nprint(double(21))")
    assert first.returncode == 0, first.stderr
    assert first.stdout == "42\n"
    artifacts = {path: path.stat().st_mtime_ns for path in (tmp_path / "cache").rglob("*.ir")}
    assert artifacts
    second = _run(tmp_path, "import helper::*\nlet answer = double(3)\nprint(answer)")
    assert second.returncode == 0, second.stderr
    assert second.stdout == "6\n"
    assert {path: path.stat().st_mtime_ns for path in artifacts} == artifacts


def test_dependency_interface_edits_recompile_importers(tmp_path: Path) -> None:
    dependency = tmp_path / "dependency.agl"
    dependency.write_text("def value() = 21\n")
    (tmp_path / "helper.agl").write_text(
        "import dependency::*\nbuiltin def print[T](value: T) -> unit\ndef answer() = value()\n"
    )
    source = "import helper::*\nprint(answer())"
    first = _run(tmp_path, source)
    assert first.returncode == 0, first.stderr
    assert first.stdout == "21\n"
    dependency.write_text('def value() = "updated"\n')
    second = _run(tmp_path, source)
    assert second.returncode == 0, second.stderr
    assert second.stdout == "updated\n"


def test_rehydrated_library_matches_a_fresh_compile_across_processes(tmp_path: Path) -> None:
    """A checked module restored from disk in a fresh process behaves identically.

    The library imports ``std/agent`` and calls its ``ask`` method. Nothing
    else in this test's own source runs before the inventory pair, so its
    first run is a genuine fresh compile (the disk cache starts empty) and its
    second run genuinely rehydrates -- proven directly by the persisted
    ``checked`` entries' mtimes, not just by matching output. The ``--dry-run``
    static call-site inventory and the executed output must each agree between
    the compile that populates the cache and the one that only rehydrates it.
    """
    (tmp_path / "library.agl").write_text(
        "import std/agent::*\n"
        "def double(value: int) -> int = value * 2\n"
        'def helper(task: text) -> text = AgentCommand("impl").ask(task)\n'
    )
    inventory_source = 'import library::*\nbuiltin def print[T](value: T) -> unit\nhelper("do it")'
    first_inventory = _run(tmp_path, inventory_source, dry_run=True)
    assert first_inventory.returncode == 0, first_inventory.stderr
    assert "call-sites:" in first_inventory.stdout

    checked_entries = {
        path: path.stat().st_mtime_ns for path in (tmp_path / "cache").rglob("*.checked")
    }
    assert checked_entries  # the fresh compile above must have persisted a "checked" entry

    second_inventory = _run(tmp_path, inventory_source, dry_run=True)
    assert second_inventory.returncode == 0, second_inventory.stderr
    assert second_inventory.stdout == first_inventory.stdout
    # A cache hit rehydrates rather than recompiling: the persisted entries this
    # second, separate process reads are untouched by it.
    assert {
        path: path.stat().st_mtime_ns for path in (tmp_path / "cache").rglob("*.checked")
    } == checked_entries

    program = "import library::*\nbuiltin def print[T](value: T) -> unit\nprint(double(21))"
    executed = _run(tmp_path, program)
    assert executed.returncode == 0, executed.stderr
    assert executed.stdout == "42\n"


@pytest.fixture
def compile_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[str], RunResult]:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    def run(source: str) -> RunResult:
        clear_parsed_module_cache()
        clear_retained_artifacts()
        return run_inline_command(
            PipelineDriver(get_sandbox_context=None),
            source,
            roots=RootSet(roots=frozenset({tmp_path})),
            default_stdlib=False,
        )

    return run


def test_a_shared_cyclic_dependency_reattaches_to_fresh_resolved_modules_under_a_different_entry(
    tmp_path: Path,
    compile_again: Callable[[str], RunResult],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A checked artifact shared by a cyclic-import group reattaches to a fresh entry.

    ``left`` and ``right`` import each other and share one persisted checked
    artifact, keyed off their own content -- neither is part of the entry's
    own dependency cycle, so the same artifact is reused regardless of which
    entry imports them. A second compilation that reaches them through a
    different entry, and through the other member of the cycle, must still
    reattach the shared artifact's cross-references to its OWN fresh resolved
    modules rather than the first compilation's now-discarded ones.
    """
    (tmp_path / "left.agl").write_text(
        "import right::*\ndef via_left() -> int = 1\ndef via_right() -> int = right_value()\n"
    )
    (tmp_path / "right.agl").write_text(
        "import left::*\ndef right_value() -> int = via_left() + 2\n"
    )
    first = compile_again(
        "import left::*\nbuiltin def print[T](value: T) -> unit\nprint(via_right() + via_left())"
    )
    assert first.ok, first.diagnostics
    assert capsys.readouterr().out == "4\n"

    second = compile_again(
        "import right::*\nbuiltin def print[T](value: T) -> unit\nprint(right_value() * 10)"
    )
    assert second.ok, second.diagnostics
    assert capsys.readouterr().out == "30\n"


@pytest.mark.parametrize(
    ("declarations", "expression", "expected"),
    [
        (
            "record Box(value: int)\ntype Alias = Box\ndef box() -> Alias = Box(21)\n",
            "box().value",
            "21\n",
        ),
        (
            "def choose(value: bool) -> int = case value of | true => 21 | false => 7\n",
            "choose(true)",
            "21\n",
        ),
        ("def identity[T](value: T) -> T = value\n", 'identity("yes")', "yes\n"),
    ],
)
def test_precompiled_type_interfaces_and_bodies_agree(
    tmp_path: Path,
    compile_again: Callable[[str], RunResult],
    capsys: pytest.CaptureFixture[str],
    declarations: str,
    expression: str,
    expected: str,
) -> None:
    (tmp_path / "library.agl").write_text("builtin def print[T](value: T) -> unit\n" + declarations)
    for _ in range(2):
        result = compile_again(f"import library::*\nprint({expression})")
        assert result.ok, result.diagnostics
        assert capsys.readouterr().out == expected


@pytest.mark.parametrize(
    "replacement",
    ["def value() -> int = missing\n", 'def value() -> int = "wrong"\n', "def value( =\n"],
    ids=["scope-error", "type-error", "syntax-error"],
)
def test_invalid_dependency_edits_are_rejected_and_repairable(
    tmp_path: Path,
    compile_again: Callable[[str], RunResult],
    capsys: pytest.CaptureFixture[str],
    replacement: str,
) -> None:
    path = tmp_path / "dependency.agl"
    path.write_text("def value() -> int = 21\n")
    (tmp_path / "library.agl").write_text(
        "import dependency::*\n"
        "builtin def print[T](value: T) -> unit\n"
        "def answer() -> int = value()\n"
    )
    source = "import library::*\nprint(answer())"
    assert compile_again(source).ok
    assert capsys.readouterr().out == "21\n"
    path.write_text(replacement)
    rejected = compile_again(source)
    assert not rejected.ok
    assert rejected.diagnostics
    assert capsys.readouterr().out == ""
    path.write_text("def value() -> int = 42\n")
    repaired = compile_again(source)
    assert repaired.ok, repaired.diagnostics
    assert capsys.readouterr().out == "42\n"


@pytest.mark.parametrize("payload", [b"broken", pickle.dumps(None), pickle.dumps({})])
def test_damaged_compiled_artifacts_rebuild_safely(
    tmp_path: Path,
    compile_again: Callable[[str], RunResult],
    capsys: pytest.CaptureFixture[str],
    payload: bytes,
) -> None:
    (tmp_path / "library.agl").write_text(
        "builtin def print[T](value: T) -> unit\ndef answer() -> int = 21\n"
    )
    source = "import library::*\nprint(answer())"
    assert compile_again(source).ok
    assert capsys.readouterr().out == "21\n"
    artifacts = list((tmp_path / "cache").rglob("*"))
    for path in artifacts:
        if path.suffix in {".scope", ".checked", ".matches", ".ir"}:
            data = path.read_bytes()
            path.write_bytes(data[:32] + hashlib.sha256(payload).digest() + payload)
    result = compile_again(source)
    assert result.ok, result.diagnostics
    assert capsys.readouterr().out == "21\n"


def test_cached_ir_revalidates_resources(
    tmp_path: Path,
    compile_again: Callable[[str], RunResult],
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "library.agl").write_text(
        "builtin def print[T](value: T) -> unit\nbuiltin def resource(path: text) -> text\n"
        'def asset() -> text = resource("asset.txt")\n'
    )
    asset = tmp_path / "asset.txt"
    asset.write_text("present")
    source = "import library::*\nprint(asset())"
    assert compile_again(source).ok
    assert capsys.readouterr().out == str(asset) + "\n"
    asset.unlink()
    result = compile_again(source)
    assert not result.ok
    assert result.diagnostics
    assert capsys.readouterr().out == ""


def test_precompiled_module_initializers_run_for_each_execution(
    tmp_path: Path,
    compile_again: Callable[[str], RunResult],
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "library.agl").write_text(
        "builtin def print[T](value: T) -> unit\n"
        "var count: int = 0\n"
        "def tick() -> int =\n  count := count + 1\n  count\n"
    )
    for _ in range(2):
        result = compile_again("import library::*\nprint(tick())\nprint(tick())")
        assert result.ok, result.diagnostics
        assert capsys.readouterr().out == "1\n2\n"


def test_precompiled_externs_use_current_companion_code(
    tmp_path: Path,
    compile_again: Callable[[str], RunResult],
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "library.agl").write_text(
        "builtin def print[T](value: T) -> unit\nextern def current() -> int\n"
    )
    companion = tmp_path / "library.py"
    for value in (21, 42):
        companion.write_text(f"def current(): return {value}\n")
        result = compile_again("import library::*\nprint(current())")
        assert result.ok, result.diagnostics
        assert capsys.readouterr().out == f"{value}\n"


def test_cached_ir_tracks_resource_symlink_targets(
    tmp_path: Path,
    compile_again: Callable[[str], RunResult],
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "library.agl").write_text(
        "builtin def print[T](value: T) -> unit\nbuiltin def resource(path: text) -> text\n"
        'def asset() -> text = resource("asset.txt")\n'
    )
    asset = tmp_path / "asset.txt"
    for name in ("first.txt", "second.txt", "second.txt"):
        target = tmp_path / name
        target.touch()
        asset.unlink(missing_ok=True)
        asset.symlink_to(target)
        result = compile_again("import library::*\nprint(asset())")
        assert result.ok, result.diagnostics
        assert capsys.readouterr().out == str(target) + "\n"


@pytest.mark.parametrize("stage", ["prepared", "checked"])
def test_cache_eviction_does_not_invalidate_an_inflight_compilation(
    tmp_path: Path,
    stage: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    (tmp_path / "library.agl").write_text(
        "builtin def print[T](value: T) -> unit\ndef answer() -> int = 42\n"
    )
    driver = PipelineDriver(get_sandbox_context=None)
    source = "import library::*\nprogram def main() -> unit = print(answer())"
    roots = RootSet(roots=frozenset({tmp_path}))
    assert driver.run(source, roots=roots, default_stdlib=False).ok
    assert capsys.readouterr().out == "42\n"
    prepared = driver.prepare_program(source, roots=roots, default_stdlib=False)
    checked = driver.discover_programs(prepared).checked if stage == "checked" else None
    clear_retained_artifacts()
    result = driver.run_prepared(prepared, checked=checked, select_default_program=True)
    assert result.ok, result.diagnostics
    assert capsys.readouterr().out == "42\n"


def test_missing_home_directory_does_not_prevent_compilation(
    tmp_path: Path,
    compile_again: Callable[[str], RunResult],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XDG_CACHE_HOME")

    def unavailable() -> Path:
        raise OSError("home directory is unavailable")

    monkeypatch.setattr(Path, "home", unavailable)
    (tmp_path / "library.agl").write_text("builtin def print[T](value: T) -> unit\n")
    result = compile_again("import library::*\nprint(42)")
    assert result.ok, result.diagnostics
    assert capsys.readouterr().out == "42\n"


def test_compiled_artifacts_cannot_execute_cached_callables(
    tmp_path: Path,
    compile_again: Callable[[str], RunResult],
    capsys: pytest.CaptureFixture[str],
) -> None:
    import shlex

    (tmp_path / "library.agl").write_text("builtin def print[T](value: T) -> unit\n")
    source = "import library::*\nprint(42)"
    assert compile_again(source).ok
    assert capsys.readouterr().out == "42\n"
    marker = tmp_path / "unexpected"

    class Command:
        def __reduce__(self) -> tuple[object, tuple[str]]:
            return os.system, (f"touch {shlex.quote(str(marker))}",)

    payload = pickle.dumps(Command())
    for path in (tmp_path / "cache").rglob("*"):
        if path.suffix in {".scope", ".checked", ".matches", ".ir"}:
            path.write_bytes(path.read_bytes()[:32] + hashlib.sha256(payload).digest() + payload)
    result = compile_again(source)
    assert result.ok, result.diagnostics
    assert capsys.readouterr().out == "42\n"
    assert not marker.exists()


def test_compiled_artifacts_with_invalid_source_references_are_rebuilt(
    tmp_path: Path,
    compile_again: Callable[[str], RunResult],
    capsys: pytest.CaptureFixture[str],
) -> None:
    import io

    (tmp_path / "library.agl").write_text("builtin def print[T](value: T) -> unit\n")
    source = "import library::*\nprint(42)"
    assert compile_again(source).ok
    assert capsys.readouterr().out == "42\n"

    class BadReference(pickle.Pickler):
        def persistent_id(self, obj: object) -> int:
            return -1

    stream = io.BytesIO()
    BadReference(stream).dump(None)
    payload = stream.getvalue()
    for path in (tmp_path / "cache").rglob("*"):
        if path.suffix in {".scope", ".checked", ".matches", ".ir"}:
            path.write_bytes(path.read_bytes()[:32] + hashlib.sha256(payload).digest() + payload)
    result = compile_again(source)
    assert result.ok, result.diagnostics
    assert capsys.readouterr().out == "42\n"
