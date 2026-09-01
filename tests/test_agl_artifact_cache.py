"""A one-shot compilation reuses the artifacts an earlier one left behind.

``agm exec``, ``agm check`` and every other non-REPL caller compiles a whole
program: the entry module plus everything behind it. Whatever of that is
unchanged from the last compilation in this process -- the standard library
almost always, the program's own imports whenever nothing edited them -- has its
scope resolution, type checking and compiled match sites reused rather than
recomputed.

These tests pin both halves of that contract: that the reuse fires (a later
compilation does no work for a module it already has), and that it stops
wherever the artifacts would no longer describe the modules in front of it -- a
different root set, an edited file, a different host capability set, or a
discarded image.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from agm.agl import artifact_cache
from agm.agl.artifact_cache import clear_retained_artifacts
from agm.agl.matchcompile import stage as match_stage
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.modules.roots import RootSet, assemble_roots
from agm.agl.pipeline import PipelineDriver
from agm.agl.runtime.codec import OutputCodec
from agm.agl.scope import program as scope_program
from agm.agl.typecheck import program as typecheck_program
from tests._agl_helpers import run_inline_command

_STDLIB = Path(__file__).resolve().parents[1] / "stdlib"
_HELPER_ID = ModuleId.from_path("helper")


@pytest.fixture(autouse=True)
def _cold_image() -> Iterator[None]:
    """Give every test an empty image and leave none behind."""
    clear_retained_artifacts()
    yield
    clear_retained_artifacts()


@pytest.fixture
def resolved_modules(monkeypatch: pytest.MonkeyPatch) -> list[ModuleId]:
    """Record the module ids whose bodies scope resolution actually resolves."""
    recorded: list[ModuleId] = []
    original = scope_program._Resolver

    def recording(**kwargs: object) -> object:
        module_id = kwargs["module_id"]
        assert isinstance(module_id, ModuleId)
        recorded.append(module_id)
        return original(**kwargs)

    monkeypatch.setattr(scope_program, "_Resolver", recording)
    return recorded


@pytest.fixture
def checked_modules(monkeypatch: pytest.MonkeyPatch) -> list[ModuleId]:
    """Record the module ids whose bodies type checking actually checks."""
    recorded: list[ModuleId] = []
    original = typecheck_program._check_prepared_module

    def recording(*args: object, **kwargs: object) -> object:
        module_id = kwargs["module_id"]
        assert isinstance(module_id, ModuleId)
        recorded.append(module_id)
        return original(*args, **kwargs)

    monkeypatch.setattr(typecheck_program, "_check_prepared_module", recording)
    return recorded


@pytest.fixture
def compiled_match_owners(monkeypatch: pytest.MonkeyPatch) -> list[ModuleId]:
    """Record the module ids whose match sites are compiled from source."""
    recorded: list[ModuleId] = []
    original = match_stage._compile_owner_sites

    def recording(owner: object) -> object:
        module_id = getattr(owner, "module_id", ENTRY_ID)
        assert isinstance(module_id, ModuleId)
        recorded.append(module_id)
        return original(owner)

    monkeypatch.setattr(match_stage, "_compile_owner_sites", recording)
    return recorded


def _library(module_ids: list[ModuleId]) -> list[ModuleId]:
    """Keep only the standard-library modules — the ones these cases are about."""
    return [module_id for module_id in module_ids if module_id.segments[:1] == ("std",)]


def _roots(invocation_root: Path, *, stdlib_root: Path) -> RootSet:
    """Assemble the roots a real invocation from *invocation_root* would use."""
    return assemble_roots(
        invocation_root=invocation_root,
        stdlib_root=stdlib_root,
        lib_root=None,
        configured=[],
        cli=[],
        cwd=invocation_root,
    )


def _compile(source: str, *, runtime: PipelineDriver | None = None, **kwargs: object) -> None:
    """Compile and run one inline program the way ``agm exec -c`` does."""
    result = run_inline_command(
        runtime if runtime is not None else PipelineDriver(), source, **kwargs
    )
    assert result.ok, result.diagnostics


# ---------------------------------------------------------------------------
# The reuse fires
# ---------------------------------------------------------------------------


class TestLibraryReuse:
    def test_a_later_compilation_resolves_only_its_entry(
        self, resolved_modules: list[ModuleId]
    ) -> None:
        _compile("let x = 1\n")
        resolved_modules.clear()

        _compile("let y = 2\n")

        assert _library(resolved_modules) == []

    def test_a_later_compilation_checks_only_its_entry(
        self, checked_modules: list[ModuleId]
    ) -> None:
        _compile("let x = 1\n")
        checked_modules.clear()

        _compile("let y = 2\n")

        assert _library(checked_modules) == []

    def test_a_later_compilation_compiles_only_its_own_match_sites(
        self, compiled_match_owners: list[ModuleId]
    ) -> None:
        _compile("let x = 1\n")
        compiled_match_owners.clear()

        _compile("case 1 of\n  | 1 => 2\n  | _ => 3\n")

        assert _library(compiled_match_owners) == []


# ---------------------------------------------------------------------------
# The reuse stops where the artifacts stop describing the library
# ---------------------------------------------------------------------------


class TestOrdinaryModuleReuse:
    """A user module is reused on exactly the terms a library module is.

    Nothing about the standard library makes it uniquely reusable. What makes
    an artifact reusable is that the modules it was derived from are unchanged,
    which an ordinary module satisfies just as often between two compilations
    that did not touch it.
    """

    def test_a_later_compilation_reuses_an_unchanged_user_module(
        self,
        tmp_path: Path,
        resolved_modules: list[ModuleId],
        checked_modules: list[ModuleId],
    ) -> None:
        (tmp_path / "helper.agl").write_text("def helper() -> int = 1\n")
        roots = _roots(tmp_path, stdlib_root=_STDLIB)
        _compile("import helper::*\nhelper()\n", roots=roots)
        resolved_modules.clear()
        checked_modules.clear()

        _compile("import helper::*\nhelper() + 1\n", roots=roots)

        assert _HELPER_ID not in resolved_modules
        assert _HELPER_ID not in checked_modules

    def test_an_edited_user_module_is_derived_afresh(
        self,
        tmp_path: Path,
        resolved_modules: list[ModuleId],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The rewrite below keeps the byte count, so only the text separates them."""
        path = tmp_path / "helper.agl"
        path.write_text("def helper() -> int = 1\n")
        roots = _roots(tmp_path, stdlib_root=_STDLIB)
        _compile("import helper::*\nprint(helper())\n", roots=roots)
        resolved_modules.clear()
        capsys.readouterr()

        path.write_text("def helper() -> int = 7\n")
        _compile("import helper::*\nprint(helper())\n", roots=roots)

        assert _HELPER_ID in resolved_modules
        assert "7" in capsys.readouterr().out


class _NullCodec(OutputCodec):
    """A host codec whose only purpose is to give a driver distinct capabilities."""

    @property
    def name(self) -> str:
        return "null"

    @property
    def supported_kinds(self) -> frozenset[str]:
        return frozenset({"text"})

    def decode(self, raw: str, kind: str) -> object:
        return raw


class TestLibraryInvalidation:
    def test_a_discarded_image_forces_a_cold_compilation(
        self, resolved_modules: list[ModuleId], compiled_match_owners: list[ModuleId]
    ) -> None:
        _compile("let x = 1\n")
        clear_retained_artifacts()
        resolved_modules.clear()
        compiled_match_owners.clear()

        _compile("let y = 2\n")

        assert _library(resolved_modules) != []
        assert _library(compiled_match_owners) != []

    def test_a_second_standard_library_gets_its_own_image(
        self, tmp_path: Path, resolved_modules: list[ModuleId]
    ) -> None:
        """A root set naming other files for the library derives them afresh."""
        replica = tmp_path / "stdlib"
        shutil.copytree(_STDLIB, replica)
        _compile("let x = 1\n")
        resolved_modules.clear()

        _compile("let y = 2\n", roots=_roots(tmp_path, stdlib_root=replica))

        assert _library(resolved_modules) != []

    def test_an_unrelated_extra_root_keeps_the_image(
        self, tmp_path: Path, resolved_modules: list[ModuleId]
    ) -> None:
        """Adding user modules leaves the library's own files untouched.

        The image is keyed by the identity of the modules an artifact was
        derived from, never by the root set that found them, so a compilation
        that merely searches one more directory -- and imports a module out of
        it -- still reuses every library artifact.
        """
        (tmp_path / "extra.agl").write_text("def extra() -> int = 1\n")
        roots = _roots(tmp_path, stdlib_root=_STDLIB)
        _compile("let x = 1\n", roots=roots)
        resolved_modules.clear()

        _compile("import extra::*\nextra()\n", roots=roots)

        assert _library(resolved_modules) == []

    def test_a_second_capability_set_gets_its_own_checked_image(
        self, checked_modules: list[ModuleId]
    ) -> None:
        _compile("let x = 1\n")
        checked_modules.clear()

        extended = PipelineDriver()
        extended.register_codec(_NullCodec())
        _compile("let y = 2\n", runtime=extended)

        assert _library(checked_modules) != []


# ---------------------------------------------------------------------------
# The reuse changes nothing a caller can observe
# ---------------------------------------------------------------------------


def test_library_reuse_changes_no_program_outcome(capsys: pytest.CaptureFixture[str]) -> None:
    """A program run against a warm image prints exactly what a cold one prints."""
    source = 'let doubled = [1, 2, 3].map(fn(n: int) -> int => n * 2)\nprint("%{doubled[2]}")\n'
    runtime = PipelineDriver()

    _compile(source, runtime=runtime)
    cold = capsys.readouterr().out
    _compile(source, runtime=runtime)
    warm = capsys.readouterr().out

    assert cold == warm == "6\n"


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
