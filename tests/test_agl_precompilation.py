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


def _run(root: Path, source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "agm.cli", "exec", "--no-stdlib", "-I", str(root), "-c", source],
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


@pytest.fixture
def compile_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[str], RunResult]:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    def run(source: str) -> RunResult:
        clear_parsed_module_cache()
        clear_retained_artifacts()
        return run_inline_command(
            PipelineDriver(),
            source,
            roots=RootSet(roots=frozenset({tmp_path})),
            default_stdlib=False,
        )

    return run


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
    driver = PipelineDriver()
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
