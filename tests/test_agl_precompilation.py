"""Precompiled libraries remain reusable across programs and fresh processes."""

from __future__ import annotations

import hashlib
import os
import pickle
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from agm.agl import lower
from agm.agl.artifact_cache import clear_retained_artifacts
from agm.agl.ir.contracts import ContractPayload, TypeNode, TypeTree
from agm.agl.ir.program import ExecutableProgram
from agm.agl.matchcompile import MatchCompiledProgram
from agm.agl.modules.parsed_module_cache import clear_parsed_module_cache
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import PipelineDriver, RunResult
from tests._agl_helpers import run_inline_code


def _run(root: Path, source: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "agm.cli",
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


def _check(root: Path, source: str) -> subprocess.CompletedProcess[str]:
    program = root / "check.agl"
    program.write_text(source, encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        "agm.cli",
        "check",
        "--no-stdlib",
        "-I",
        str(root),
        str(program),
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


def test_rehydrated_library_checks_across_fresh_processes(tmp_path: Path) -> None:
    """A checked module restored in a fresh process remains statically valid.

    The library imports ``std/agent`` and calls its ``ask`` method. Its first
    check populates the disk cache; the second check rehydrates it, proven
    directly by the persisted ``checked`` entries' mtimes.
    """
    (tmp_path / "library.agl").write_text(
        "import std/agent::*\n"
        "def double(value: int) -> int = value * 2\n"
        'def helper(task: text) -> text = AgentCommand("impl").ask(task)\n'
    )
    checked_source = (
        "import library::*\nbuiltin def print[T](value: T) -> unit\n"
        'program def main() -> unit = print(helper("do it"))'
    )
    first_check = _check(tmp_path, checked_source)
    assert first_check.returncode == 0, first_check.stderr
    assert first_check.stdout == ""

    checked_entries = {
        path: path.stat().st_mtime_ns for path in (tmp_path / "cache").rglob("*.checked")
    }
    assert checked_entries

    second_check = _check(tmp_path, checked_source)
    assert second_check.returncode == 0, second_check.stderr
    assert second_check.stdout == first_check.stdout
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
        return run_inline_code(
            PipelineDriver(resolve_agent_spec=None, get_sandbox_context=None),
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
            "record Box\n  value: int\ntype Alias = Box\ndef box() -> Alias = Box(21)\n",
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


def test_precompiled_extern_target_contracts_round_trip(
    tmp_path: Path,
    compile_again: Callable[[str], RunResult],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A reloaded library still delivers its type-directed extern calls' ``TypeContract``s."""
    (tmp_path / "library.agl").write_text(
        "builtin def print[T](value: T) -> unit\n"
        "extern def query[T](question: text, context: text) -> T\n"
        "record Pair\n"
        "  left: text\n"
        "  right: text\n"
        'def direct(question: text) -> text = query(question, "d")\n'
        "def by-reference() -> (text, text) -> text = query\n"
        'def by-partial() -> (text) -> text = query::[text]("p", ?)\n'
        'def paired() -> Pair = query("l", "r")\n'
    )
    (tmp_path / "library.py").write_text(
        "import agl\n"
        "def query(contract, question, context):\n"
        "    assert isinstance(contract, agl.TypeContract)\n"
        "    if contract.kind == 'record':\n"
        "        return contract.nominal(left=question, right=context)\n"
        "    return f'{contract.label} {question} {context}'\n"
    )
    source = (
        'import library::*\nprint(direct("q"))\nprint(by-reference()("r", "c"))\n'
        'print(by-partial()("c"))\nprint(paired())'
    )
    outputs: list[str] = []
    artifacts: dict[Path, int] = {}
    for _ in range(2):
        result = compile_again(source)
        assert result.ok, result.diagnostics
        outputs.append(capsys.readouterr().out)
        if not artifacts:
            artifacts = {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*.ir")}
    assert artifacts
    assert {path: path.stat().st_mtime_ns for path in artifacts} == artifacts
    first, second = outputs
    assert first == second
    assert first.splitlines() == [
        "text q d",
        "text r c",
        "text p c",
        'Pair(left = "l", right = "r")',
    ]


def test_precompiled_extern_type_trees_round_trip(
    tmp_path: Path,
    compile_again: Callable[[str], RunResult],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reloaded library keeps its extern contracts' type trees and its types' field docs."""
    executables: list[ExecutableProgram] = []
    lower_program = lower.lower_program

    def capture(
        compiled: MatchCompiledProgram,
        *,
        contract_payloads: Mapping[int, ContractPayload] | None = None,
    ) -> ExecutableProgram:
        executables.append(lower_program(compiled, contract_payloads=contract_payloads))
        return executables[-1]

    monkeypatch.setattr(lower, "lower_program", capture)
    (tmp_path / "library.agl").write_text(
        '@doc("A team.")\nenum Team\n  | @doc("Money.") @json-name("billing") Billing\n'
        "  | Technical\n"
        "enum Tree\n  | Leaf(team: Team)\n  | Node(children: array[Tree])\n"
        'record Ticket\n  @doc("Who owns it.") team: Team\n  title: text\n'
        "extern def query[T](question: text) -> T\n"
        'def tree() -> Tree = query("q")\n'
    )
    (tmp_path / "library.py").write_text("def query(*args):\n    return args[0]\n")
    trees: list[list[TypeTree]] = []
    artifacts: dict[Path, int] = {}
    for _ in range(2):
        result = compile_again('import library::*\ndef ticket() -> Ticket = query("t")\n0')
        assert result.ok, result.diagnostics
        trees.append([request.type_tree for request in executables[-1].target_contracts.values()])
        if not artifacts:
            artifacts = {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*.ir")}
    assert artifacts
    assert {path: path.stat().st_mtime_ns for path in artifacts} == artifacts
    first, second = trees
    assert first == second
    by_defs = {tuple(key for key, _node in tree.defs): tree for tree in first}
    assert len(first) == 2 and set(by_defs) == {("Tree",), ()}
    ticket = by_defs[()].root
    assert isinstance(ticket, TypeNode)
    assert [(field.name, field.doc) for field in ticket.fields] == [
        ("team", "Who owns it."),
        ("title", None),
    ]


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
    driver = PipelineDriver(resolve_agent_spec=None, get_sandbox_context=None)
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
