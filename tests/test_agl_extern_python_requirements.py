"""A package's unsatisfied Python requirements fail its companions' loading with a diagnostic.

The fixture package declares a requirement no environment satisfies; tests make
it satisfied by standing in for the installed-distribution lookup.
"""

from __future__ import annotations

import shutil
from importlib import metadata
from pathlib import Path

import pytest

from agm.agl.modules.roots import RootSet, assemble_roots
from agm.agl.pipeline import PipelineDriver, RunResult
from agm.agl.repl.entry import EntryResult
from agm.agl.repl.session import ReplSession
from agm.agl.semantics.values import IntValue
from agm.packages.activation import ActivationIndex, ActivePackage, write_activation_index
from agm.packages.manifest import load_manifest
from agm.packages.model import PackageInfo
from tests._agl_helpers import REPO_STDLIB_ROOT, run_inline_command
from tests.agl.ir_harness import write_companion_file, write_module_file

_PACKAGE = Path(__file__).parent / "agl" / "packages" / "python_requirement"
_REQUIREMENT = "agm-test-absent-distribution>=1.0"
_SYNC = "agm pkg sync"
_INSTALL = "agm pkg install"


def _package(root: Path = _PACKAGE) -> PackageInfo:
    return PackageInfo(root, load_manifest(root / "package.toml"))


def _roots(*, loose: Path | None = None, package: Path = _PACKAGE) -> RootSet:
    return assemble_roots(
        invocation_root=None,
        stdlib_root=REPO_STDLIB_ROOT,
        lib_root=None,
        configured=[],
        cli=() if loose is None else (str(loose),),
        cwd=_PACKAGE,
        package_roots=(_package(package),),
    )


def _run(source: str, *, loose: Path | None = None, package: Path = _PACKAGE) -> RunResult:
    return run_inline_command(
        PipelineDriver(get_sandbox_context=None),
        source,
        roots=_roots(loose=loose, package=package),
        default_stdlib=False,
    )


def _install_requirement(patch: pytest.MonkeyPatch) -> None:
    """Report the fixture's required distribution as installed at a matching version."""
    real_version = metadata.version

    def version(name: str) -> str:
        return "1.0" if name == "agm-test-absent-distribution" else real_version(name)

    patch.setattr(metadata, "version", version)


@pytest.fixture()
def requirement_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_requirement(monkeypatch)


@pytest.fixture()
def package_active() -> None:
    """Activate the fixture package, editable at its own root, in the test's AGM home."""
    active = ActivePackage(_package().manifest.version, editable=_PACKAGE)
    write_activation_index(ActivationIndex({"python_requirement": active}), home=Path.home())


def _assert_requirement_diagnostic(result: RunResult | EntryResult, *, repair: str = _SYNC) -> None:
    assert result.ok is False
    assert len(result.diagnostics) == 1
    message = result.diagnostics[0].message
    assert "python_requirement" in message
    assert _REQUIREMENT in message
    assert repair in message
    assert ({_SYNC, _INSTALL} - {repair}).pop() not in message


@pytest.mark.usefixtures("package_active")
def test_unsatisfied_requirement_is_a_diagnostic_naming_package_spec_and_repair() -> None:
    result = _run("import python_requirement/double\npython_requirement/double::double(2)")

    _assert_requirement_diagnostic(result)


def test_unsatisfied_requirement_of_an_inactive_mounted_package_suggests_installing_it() -> None:
    result = _run("import python_requirement/double\npython_requirement/double::double(2)")

    _assert_requirement_diagnostic(result, repair=_INSTALL)


def test_package_active_at_another_root_suggests_installing_the_mounted_one(
    tmp_path: Path,
) -> None:
    elsewhere = tmp_path / "python_requirement"
    elsewhere.mkdir()
    (elsewhere / "package.toml").write_bytes((_PACKAGE / "package.toml").read_bytes())
    active = ActivePackage(_package().manifest.version, editable=elsewhere)
    write_activation_index(ActivationIndex({"python_requirement": active}), home=Path.home())

    result = _run("import python_requirement/double\npython_requirement/double::double(2)")

    _assert_requirement_diagnostic(result, repair=_INSTALL)


def test_unsatisfied_requirement_of_the_active_store_package_suggests_syncing() -> None:
    store_root = Path.home() / ".agm" / "packages" / "python_requirement" / "1.0.0"
    shutil.copytree(_PACKAGE, store_root)
    active = ActivePackage(_package().manifest.version)
    write_activation_index(ActivationIndex({"python_requirement": active}), home=Path.home())

    result = _run(
        "import python_requirement/double\npython_requirement/double::double(2)",
        package=store_root,
    )

    _assert_requirement_diagnostic(result)


@pytest.mark.usefixtures("package_active")
def test_requirements_are_reported_once_per_package_across_companion_modules() -> None:
    result = _run(
        "import python_requirement/double\nimport python_requirement/negate\n"
        "python_requirement/double::double(python_requirement/negate::negate(2))"
    )

    _assert_requirement_diagnostic(result)


@pytest.mark.usefixtures("requirement_installed")
def test_satisfied_requirement_loads_companions_normally() -> None:
    result = _run("import python_requirement/double\nlet r = python_requirement/double::double(21)")

    assert result.ok, result.diagnostics
    assert result.bindings["r"] == IntValue(42)


def test_package_without_reached_companions_is_not_checked() -> None:
    result = _run("import python_requirement/plain\nlet r = python_requirement/plain::plain(7)")

    assert result.ok, result.diagnostics
    assert result.bindings["r"] == IntValue(7)


def test_loose_module_companion_is_unaffected(tmp_path: Path) -> None:
    write_module_file(tmp_path, "loose", "extern def triple(x: int) -> int")
    write_companion_file(tmp_path, "loose", "def triple(x):\n    return 3 * x\n")

    result = _run("import loose\nlet r = loose::triple(3)", loose=tmp_path)

    assert result.ok, result.diagnostics
    assert result.bindings["r"] == IntValue(9)


@pytest.mark.usefixtures("package_active")
def test_fresh_driver_reusing_cached_artifacts_rechecks_the_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "import python_requirement/double\npython_requirement/double::double(1)"
    with monkeypatch.context() as patched:
        _install_requirement(patched)
        assert _run(source).ok

    _assert_requirement_diagnostic(_run(source))


def _repl() -> ReplSession:
    return ReplSession(package_roots=(_package(),), cwd=_PACKAGE, default_stdlib=False)


@pytest.mark.usefixtures("package_active")
def test_repl_entry_reports_an_unsatisfied_requirement() -> None:
    result = _repl().eval_entry("import python_requirement/double")

    _assert_requirement_diagnostic(result)


def test_repl_companion_loaded_while_satisfied_survives_the_requirement_lapsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _repl()
    with monkeypatch.context() as patched:
        _install_requirement(patched)
        assert session.eval_entry("import python_requirement/double").ok

    result = session.eval_entry("let r = python_requirement/double::double(4)")
    assert result.ok, result.diagnostics
    assert session.eval_entry("import python_requirement/negate").ok is False
