"""Compiler behavior across invocations sharing a disposable frontend cache."""

from __future__ import annotations

import hashlib
import os
import pickle
import shlex
import subprocess
import sys
from functools import partial
from pathlib import Path

import pytest

from agm.agl.modules import disk_cache
from agm.agl.modules.errors import MissingExternCompanion
from agm.agl.modules.ids import STD_PRELUDE_ID
from agm.agl.modules.loader import LoadedModule, _parse_imported_module
from agm.agl.modules.parsed_module_cache import ParsedModuleCache, module_node_id_base
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import ParsedEntry, PipelineDriver


def _execute(root: Path, cache: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "agm.cli", "exec", "-c", "print marker"],
        cwd=root,
        env={**os.environ, "AGM_STDLIB": str(root), "XDG_CACHE_HOME": str(cache)},
        capture_output=True,
        text=True,
        check=False,
    )


def _library(root: Path, value: int = 41) -> Path:
    source = root / "src" / "prelude.agl"
    source.parent.mkdir(parents=True)
    source.write_text(f"builtin def print[T](value: T) -> unit\nlet marker: int = {value}\n")
    return source


def test_separate_commands_observe_source_edits_after_caching(tmp_path: Path) -> None:
    source = _library(tmp_path)
    cache = tmp_path / "cache"
    first = _execute(tmp_path, cache)
    assert first.returncode == 0, first.stderr
    assert first.stdout == "41\n"
    assert list((cache / "agm" / "agl").rglob("*.cache"))
    previous = source.stat()
    source.write_text(source.read_text().replace("41", "42"))
    os.utime(source, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    second = _execute(tmp_path, cache)
    assert second.returncode == 0, second.stderr
    assert second.stdout == "42\n"


@pytest.mark.parametrize("damage", [b"", b"incomplete cached artifact"])
def test_corrupt_cache_rebuilds_without_changing_output(tmp_path: Path, damage: bytes) -> None:
    _library(tmp_path)
    cache = tmp_path / "cache"
    assert _execute(tmp_path, cache).stdout == "41\n"
    entries = list((cache / "agm" / "agl").rglob("*.cache"))
    assert entries
    for entry in entries:
        entry.write_bytes(damage)
    repaired = _execute(tmp_path, cache)
    assert repaired.returncode == 0, repaired.stderr
    assert repaired.stdout == "41\n"


def test_unavailable_cache_does_not_prevent_execution(tmp_path: Path) -> None:
    _library(tmp_path)
    cache = tmp_path / "cache"
    cache.write_text("a file cannot hold a cache directory")
    result = _execute(tmp_path, cache)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "41\n"


def _cached_library(root: Path) -> tuple[LoadedModule, int]:
    """Open a fresh host cache using the real loader's module parser."""
    path = root / "src" / "prelude.agl"
    cache = ParsedModuleCache()
    module = cache.get_or_build(
        STD_PRELUDE_ID,
        path,
        default_stdlib=False,
        build=partial(_parse_imported_module, STD_PRELUDE_ID, path, default_stdlib=False),
    )
    # Loading the persisted artifact is the disk cache's public contract.
    restored = disk_cache.load(
        STD_PRELUDE_ID,
        path,
        path.read_text(),
        module_node_id_base(STD_PRELUDE_ID, path, path.read_text(), False),
        False,
    )
    assert restored is not None
    return module, restored[1]


def _run_cached_library(root: Path, capsys: pytest.CaptureFixture[str], expected: str) -> None:
    module, next_id = _cached_library(root)
    parsed = ParsedEntry(
        source=module.source_text,
        entry_path=module.path,
        program=module.program,
        next_id=next_id,
        spaced_qualifiers=module.spaced_qualifiers,
        diagnostics=(),
        warnings=(),
    )
    driver = PipelineDriver(get_sandbox_context=None)
    prepared = driver.prepare_parsed_entry(parsed, default_stdlib=False)
    result = driver.run_prepared(prepared, select_default_program=True)
    assert result.ok, result.diagnostics
    assert capsys.readouterr().out == expected


@pytest.mark.parametrize("literal", ["41", "1.5"])
def test_persisted_syntax_executes_in_fresh_hosts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    literal: str,
) -> None:
    source = _library(tmp_path)
    source.write_text(
        "builtin def print[T](value: T) -> unit\n"
        f"program def main() -> unit = print ({literal} + 1)\n"
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    expected = "42\n" if literal == "41" else "2.5\n"
    _run_cached_library(tmp_path, capsys, expected)
    _run_cached_library(tmp_path, capsys, expected)


@pytest.mark.parametrize(
    "payload", [None, b"", b"broken", pickle.dumps(None), pickle.dumps((None, 1))]
)
def test_invalid_artifact_payload_is_rebuilt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    payload: bytes | None,
) -> None:
    source = _library(tmp_path)
    source.write_text(source.read_text() + "\nprogram def main() -> unit = print marker\n")
    cache = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    _run_cached_library(tmp_path, capsys, "41\n")
    for entry in (cache / "agm" / "agl").glob("*.cache"):
        data = entry.read_bytes()
        entry.write_bytes(
            b"incomplete"
            if payload is None
            else data[:32] + hashlib.sha256(payload).digest() + payload
        )
    _run_cached_library(tmp_path, capsys, "41\n")


def test_artifacts_cannot_invoke_non_syntax_constructors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _library(tmp_path)
    source.write_text(source.read_text() + "\nprogram def main() -> unit = print marker\n")
    cache = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    _run_cached_library(tmp_path, capsys, "41\n")
    marker = tmp_path / "unexpected"

    class UnwantedCommand:
        def __reduce__(self) -> tuple[object, tuple[str]]:
            return os.system, (f"touch {shlex.quote(str(marker))}",)

    payload = pickle.dumps(UnwantedCommand())
    for entry in (cache / "agm" / "agl").glob("*.cache"):
        entry.write_bytes(entry.read_bytes()[:32] + hashlib.sha256(payload).digest() + payload)
    _run_cached_library(tmp_path, capsys, "41\n")
    assert not marker.exists()


def test_cached_companion_is_checked_again_when_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _library(tmp_path)
    source.write_text("extern def marker() -> int\n")
    source.with_suffix(".py").write_text("def marker(): return 41\n")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    _cached_library(tmp_path)
    source.with_suffix(".py").unlink()
    with pytest.raises(MissingExternCompanion):
        _cached_library(tmp_path)


@pytest.mark.parametrize("failure", ["read", "replace", "directory"])
def test_cache_io_failure_preserves_compilation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    source = _library(tmp_path)
    source.write_text(source.read_text() + "\nprogram def main() -> unit = print marker\n")
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    if failure == "directory":
        (tmp_path / ".cache").write_text("not a directory")
    else:
        operation = "read_bytes" if failure == "read" else "replace"
        original = getattr(Path, operation)

        def fail(path: Path, *args: object, **kwargs: object) -> object:
            if path.suffix == ".cache" or path.parent.name == "agl":
                raise OSError("cache storage is unavailable")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, operation, fail)
    driver = PipelineDriver(get_sandbox_context=None)
    result = driver.run(source.read_text(), default_stdlib=False)
    assert result.ok, result.diagnostics
    assert capsys.readouterr().out == "41\n"
    # Exercise imported-module persistence with the same unavailable storage.
    roots = RootSet(roots=frozenset(), stdlib_roots=frozenset({tmp_path}))
    result = driver.run("program def main() -> unit = print marker", roots=roots)
    assert result.ok, result.diagnostics
    assert capsys.readouterr().out == "41\n"
