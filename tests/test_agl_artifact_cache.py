"""Repeated program executions observe current source and isolated host state."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl import artifact_cache
from agm.agl.pipeline import PipelineDriver
from agm.agl.runtime.codec import TextCodec
from tests._agl_helpers import agl_roots, run_inline_command


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("print([1, 2, 3].map(fn(n: int) -> int => n * 2)[2])", "6\n"),
        ("print(case 1 of | 1 => 2 | _ => 3)", "2\n"),
        ("var counter = 0\ncounter := counter + 1\nprint(counter)", "1\n"),
    ],
    ids=["library-method", "match", "fresh-mutable-state"],
)
def test_repeated_executions_produce_the_expected_output(
    source: str, expected: str, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = PipelineDriver()
    for _ in range(2):
        result = run_inline_command(runtime, source)
        assert result.ok, result.diagnostics
        assert capsys.readouterr().out == expected


@pytest.mark.parametrize(
    ("replacement", "expected"),
    [("def helper() -> int = 7\n", "7\n"), ('def helper() -> text = "new"\n', "new\n")],
    ids=["same-size-edit", "changed-return-type"],
)
def test_edited_imports_change_the_next_execution(
    tmp_path: Path, replacement: str, expected: str, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "helper.agl"
    path.write_text("def helper() -> int = 1\n")
    roots = agl_roots(tmp_path)
    runtime = PipelineDriver()
    source = "import helper::*\nprint(helper())\n"
    assert run_inline_command(runtime, source, roots=roots).ok
    assert capsys.readouterr().out == "1\n"

    path.write_text(replacement)
    result = run_inline_command(runtime, source, roots=roots)

    assert result.ok, result.diagnostics
    assert capsys.readouterr().out == expected


def test_invalid_import_edit_is_rejected_and_can_be_repaired(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "helper.agl"
    roots = agl_roots(tmp_path)
    runtime = PipelineDriver()
    source = "import helper::*\nprint(helper())\n"
    path.write_text("def helper() -> int = 1\n")
    assert run_inline_command(runtime, source, roots=roots).ok
    assert capsys.readouterr().out == "1\n"

    path.write_text('def helper() -> int = "bad"\n')
    failed = run_inline_command(runtime, source, roots=roots)
    assert not failed.ok
    assert failed.diagnostics
    assert capsys.readouterr().out == ""

    path.write_text("def helper() -> int = 9\n")
    assert run_inline_command(runtime, source, roots=roots).ok
    assert capsys.readouterr().out == "9\n"


def test_same_named_imports_are_isolated_between_roots(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = PipelineDriver()
    for name, value in (("first", 1), ("second", 7), ("first", 1)):
        root = tmp_path / name
        root.mkdir(exist_ok=True)
        (root / "helper.agl").write_text(f"def helper() -> int = {value}\n")
        result = run_inline_command(
            runtime, "import helper::*\nprint(helper())\n", roots=agl_roots(root)
        )
        assert result.ok, result.diagnostics
        assert capsys.readouterr().out == f"{value}\n"


def test_discarding_compilation_state_preserves_program_behavior(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = "print([1, 2, 3].map(fn(n: int) -> int => n * 2)[2])"
    runtime = PipelineDriver()
    assert run_inline_command(runtime, source).ok
    assert capsys.readouterr().out == "6\n"

    artifact_cache.clear_retained_artifacts()

    assert run_inline_command(runtime, source).ok
    assert capsys.readouterr().out == "6\n"


def test_the_image_is_bounded_and_evicts_the_least_recently_used() -> None:
    """Retention is capped, so a long-lived process cannot grow without limit.

    The store is exercised directly: filling the production cap through real
    compilations would mean building hundreds of distinct standard libraries.
    """
    store: artifact_cache._ArtifactStore[str] = artifact_cache._ArtifactStore(capacity=2)
    sources: artifact_cache.Sources = ()
    store.put(("a",), sources, "first")
    store.put(("b",), sources, "second")
    assert store.get(("a",), sources) == "first"

    store.put(("c",), sources, "third")

    assert store.get(("b",), sources) is None
    assert store.get(("a",), sources) == "first"
    assert store.get(("c",), sources) == "third"


def test_custom_response_formats_do_not_leak_between_hosts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class TaggedCodec(TextCodec):
        @property
        def name(self) -> str:
            return "tagged"

    extended = PipelineDriver(agent_dispatcher=lambda request: "answer")
    extended.register_codec(TaggedCodec())
    source = 'let answer: text = ask("prompt", format = "tagged")\nprint(answer)'
    assert run_inline_command(extended, source).ok
    assert capsys.readouterr().out == "answer\n"

    ordinary = PipelineDriver(
        agent_dispatcher=lambda _: pytest.fail("unsupported format dispatched")
    )
    rejected = run_inline_command(ordinary, source)
    assert not rejected.ok
    assert rejected.diagnostics or rejected.error is not None
    assert capsys.readouterr().out == ""

    assert run_inline_command(extended, source).ok
    assert capsys.readouterr().out == "answer\n"
