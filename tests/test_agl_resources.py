"""End-to-end resource builtin behavior."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest
import semver

from agm.agl import PipelineDriver
from agm.agl.ir.ids import Location, SourceId
from agm.agl.ir.nodes import IrResource
from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
from agm.agl.ir.validate import InvalidIrError, validate_ir
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.modules.roots import RootSet
from agm.agl.repl import ReplSession
from agm.agl.semantics.values import IntValue, TextValue
from agm.agl.syntax.resources import ResourceError, resolve_resource
from agm.packages.discipline import DisciplineError, validate_package
from agm.packages.manifest import PackageManifest
from agm.packages.model import PackageInfo
from tests._agl_helpers import agl_roots

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


def test_resource_is_a_constant_root_initializer_anchored_to_a_loose_module(
    tmp_path: Path,
) -> None:
    entry = tmp_path / "main.agl"
    resource = tmp_path / "prompt.md"
    resource.write_text("prompt", encoding="utf-8")
    source = """let prompt = resource("prompt.md")
program def main() -> unit =
  print prompt
"""

    result = _run_file(source, entry, roots=agl_roots())

    assert result.ok
    assert result.bindings["prompt"] == TextValue(str(resource.resolve()))


def test_renamed_resource_builtin_runs_through_its_original_declaration(tmp_path: Path) -> None:
    entry = tmp_path / "main.agl"
    resource = tmp_path / "prompt.md"
    resource.write_text("prompt", encoding="utf-8")
    source = """import std/core::*
import std/core::{resource as asset}
let prompt = asset("prompt.md")
program def main() -> unit =
  print prompt
"""

    result = _run_file(source, entry, roots=agl_roots())

    assert result.ok
    assert result.bindings["prompt"] == TextValue(str(resource.resolve()))


def test_resource_and_resource_dir_anchor_package_modules_to_the_package_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "review-tools"
    module = root / "review-tools" / "main.agl"
    module.parent.mkdir(parents=True)
    prompt = root / "prompts" / "review.md"
    prompt.parent.mkdir()
    prompt.write_text("prompt", encoding="utf-8")
    package = PackageInfo(root, PackageManifest("review-tools", semver.Version.parse("1.0.0")))
    source = """let prompt = resource("prompts/review.md")
let root = resource-dir()
program def main() -> unit =
  print prompt
"""

    result = _run_file(
        source,
        module,
        roots=RootSet(
            roots=frozenset({_STDLIB, root}),
            packages=(package,),
            stdlib_roots=frozenset({_STDLIB}),
        ),
    )

    assert result.ok
    assert result.bindings["prompt"] == TextValue(str(prompt.resolve()))
    assert result.bindings["root"] == TextValue(str(root.resolve()))


@pytest.mark.parametrize(
    "call",
    (
        "resource(name)",
        'resource("/tmp/prompt")',
        'resource("../prompt")',
        'resource("prompts\\\\review.md")',
        'resource("./prompt.md")',
        'resource("prompts/./review.md")',
        'resource("prompts//review.md")',
        'resource("prompts/")',
        'resource-dir("prompt.md")',
    ),
)
def test_resource_calls_require_safe_literal_paths(call: str) -> None:
    source = f"""program def main() -> unit =
  let name = "prompt.md"
  let path = {call}
  print path
"""

    result = _run_file(source, Path("main.agl"), roots=agl_roots())

    assert not result.ok
    assert result.diagnostics


@pytest.mark.parametrize("path", ("C:prompt.md", "C:..", "C:../prompt.md"))
def test_resource_rejects_windows_drive_relative_paths(path: str, tmp_path: Path) -> None:
    target = tmp_path / PurePosixPath(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("prompt", encoding="utf-8")
    source = f'''program def main() -> unit =
  print resource("{path}")
'''

    result = _run_file(
        source,
        tmp_path / "main.agl",
        roots=agl_roots(),
    )

    assert not result.ok
    assert result.diagnostics


def test_missing_resource_is_a_link_error(tmp_path: Path) -> None:
    entry = tmp_path / "main.agl"
    source = """program def main() -> unit =
  print resource("missing.md")
"""

    result = _run_file(source, entry, roots=agl_roots())

    assert not result.ok
    assert result.diagnostics


def test_repl_resource_error_is_a_source_diagnostic() -> None:
    session = ReplSession()

    result = session.eval_entry('let staged = 1\nresource("missing.md")')

    assert not result.ok
    assert result.error is None
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert (diagnostic.line, diagnostic.column, diagnostic.end_line, diagnostic.end_column) == (
        2,
        1,
        2,
        23,
    )


def test_repl_resource_error_restores_link_state() -> None:
    session = ReplSession()
    assert session.eval_entry("let keep = 1").ok
    link_snapshot = session._link_image.snapshot_state()

    result = session.eval_entry('let staged = 2\nresource("missing.md")')

    assert not result.ok
    assert session._link_image.snapshot_state() == link_snapshot
    assert [(name, value) for name, _typ, value in session.bindings()] == [("keep", IntValue(1))]
    recovered = session.eval_entry("let after = keep + 1")
    assert recovered.ok, recovered.diagnostics
    assert recovered.value == IntValue(2)


def test_resource_resolution_rejects_symlinks_that_escape_its_anchor(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "prompt.md").write_text("prompt", encoding="utf-8")
    anchor = tmp_path / "anchor"
    anchor.mkdir()
    (anchor / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ResourceError):
        resolve_resource(anchor, "linked/prompt.md")


def test_ir_resource_requires_an_absolute_path() -> None:
    source_id = SourceId(0)
    location = Location(source_id, 0, 1, 1, 0)
    program = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={ENTRY_ID: ExecutableModule(ENTRY_ID, (IrResource(location, "relative"),))},
        symbols={},
        nominals={},
        sources={source_id: SourceFile("<test>", "")},
    )

    with pytest.raises(InvalidIrError):
        validate_ir(program, deep=False)


def test_package_validation_rejects_a_missing_resource(tmp_path: Path) -> None:
    root = tmp_path / "package"
    module = root / "package" / "main.agl"
    module.parent.mkdir(parents=True)
    module.write_text(
        """program def main() -> unit =
  print resource("prompts/missing.md")
""",
        encoding="utf-8",
    )
    package = PackageInfo(root, PackageManifest("package", semver.Version.parse("1.0.0")))

    with pytest.raises(DisciplineError):
        validate_package(package)


def test_package_validation_rejects_missing_resource_reached_through_a_reexport(
    tmp_path: Path,
) -> None:
    root = tmp_path / "package"
    module_root = root / "package"
    module_root.mkdir(parents=True)
    (module_root / "resources.agl").write_text(
        "export std/core::{resource as asset}\n", encoding="utf-8"
    )
    (module_root / "main.agl").write_text(
        """import package/resources::{asset}
program def main() -> unit =
  print asset("prompts/missing.md")
""",
        encoding="utf-8",
    )
    package = PackageInfo(root, PackageManifest("package", semver.Version.parse("1.0.0")))

    with pytest.raises(DisciplineError):
        validate_package(package)


def test_package_validation_rejects_missing_resource_exposed_by_import_tail_scoped_alias(
    tmp_path: Path,
) -> None:
    root = tmp_path / "package"
    module_root = root / "package"
    module_root.mkdir(parents=True)
    (module_root / "resources.agl").write_text(
        "scope Assets\nexport std/core::{resource as asset}\nend Assets\n",
        encoding="utf-8",
    )
    (module_root / "main.agl").write_text(
        "import package/resources::*\n"
        'let prompt = Assets::asset("prompts/missing.md")\n'
        "program def main() -> unit = ()\n",
        encoding="utf-8",
    )
    package = PackageInfo(root, PackageManifest("package", semver.Version.parse("1.0.0")))

    with pytest.raises(DisciplineError):
        validate_package(package)


def test_package_validation_rejects_missing_resource_through_an_ancestor_scoped_alias(
    tmp_path: Path,
) -> None:
    root = tmp_path / "package"
    module_root = root / "package"
    module_root.mkdir(parents=True)
    (module_root / "main.agl").write_text(
        "scope Assets\n"
        "import std/core::{resource as asset}\n"
        "scope Templates\n"
        'let prompt = asset("prompts/missing.md")\n'
        "end Templates\n"
        "end Assets\n"
        "program def main() -> unit = ()\n",
        encoding="utf-8",
    )
    package = PackageInfo(root, PackageManifest("package", semver.Version.parse("1.0.0")))

    with pytest.raises(DisciplineError):
        validate_package(package)


def test_scoped_function_blocks_scoped_resource_alias_during_nested_lookup(
    tmp_path: Path,
) -> None:
    root = tmp_path / "package"
    module_root = root / "package"
    module_root.mkdir(parents=True)
    (module_root / "main.agl").write_text(
        "import std/core::{resource as asset}\n"
        "scope Assets\n"
        "import std/core::{resource as asset}\n"
        "def asset(path: text) -> text = path\n"
        "scope Templates\n"
        'let prompt = asset("prompts/missing.md")\n'
        "end Templates\n"
        "end Assets\n"
        "program def main() -> unit = ()\n",
        encoding="utf-8",
    )
    package = PackageInfo(root, PackageManifest("package", semver.Version.parse("1.0.0")))

    validate_package(package)


def test_package_validation_rejects_missing_resource_through_a_scoped_import_route(
    tmp_path: Path,
) -> None:
    root = tmp_path / "package"
    module_root = root / "package"
    module_root.mkdir(parents=True)
    (module_root / "main.agl").write_text(
        "scope Assets\n"
        "import std/core::{resource as asset}\n"
        "end Assets\n"
        'let prompt = core::asset("prompts/missing.md")\n'
        "program def main() -> unit = ()\n",
        encoding="utf-8",
    )
    package = PackageInfo(root, PackageManifest("package", semver.Version.parse("1.0.0")))

    with pytest.raises(DisciplineError):
        validate_package(package)


def test_package_validation_uses_the_resolved_resource_declaration(tmp_path: Path) -> None:
    root = tmp_path / "package"
    module_root = root / "package"
    module_root.mkdir(parents=True)
    (module_root / "main.agl").write_text(
        """import std/core::{resource as asset}
def asset(path: text) -> text = path
let local = asset("prompts/missing.md")
program def main() -> unit = ()
""",
        encoding="utf-8",
    )
    package = PackageInfo(root, PackageManifest("package", semver.Version.parse("1.0.0")))

    validate_package(package)

    (module_root / "main.agl").write_text(
        """import std/core as core
let path = core::resource("prompts/missing.md")
program def main() -> unit = ()
""",
        encoding="utf-8",
    )
    with pytest.raises(DisciplineError):
        validate_package(package)


def test_package_validation_follows_an_unaliased_qualified_resource_import(
    tmp_path: Path,
) -> None:
    root = tmp_path / "package"
    module_root = root / "package"
    module_root.mkdir(parents=True)
    (module_root / "main.agl").write_text(
        """import std/core
let path = core::resource("prompts/missing.md")
program def main() -> unit = ()
""",
        encoding="utf-8",
    )
    package = PackageInfo(root, PackageManifest("package", semver.Version.parse("1.0.0")))

    with pytest.raises(DisciplineError):
        validate_package(package)


@pytest.mark.parametrize("qualifier", ("core", "std/core", "/std/core"))
def test_package_validation_follows_qualified_resource_wildcard_imports(
    tmp_path: Path, qualifier: str
) -> None:
    root = tmp_path / "package"
    module_root = root / "package"
    module_root.mkdir(parents=True)
    (module_root / "main.agl").write_text(
        f"""import std/*
let path = {qualifier}::resource("prompts/missing.md")
program def main() -> unit = ()
""",
        encoding="utf-8",
    )
    package = PackageInfo(root, PackageManifest("package", semver.Version.parse("1.0.0")))

    with pytest.raises(DisciplineError):
        validate_package(package)


def test_package_validation_follows_a_resource_exported_by_a_wildcard(tmp_path: Path) -> None:
    root = tmp_path / "package"
    module_root = root / "package"
    module_root.mkdir(parents=True)
    (module_root / "resources.agl").write_text("export std/*\n", encoding="utf-8")
    (module_root / "main.agl").write_text(
        """import package/resources
let path = resources::resource("prompts/missing.md")
program def main() -> unit = ()
""",
        encoding="utf-8",
    )
    package = PackageInfo(root, PackageManifest("package", semver.Version.parse("1.0.0")))

    with pytest.raises(DisciplineError):
        validate_package(package)


def test_package_validation_accepts_a_reexport_that_hides_resource_builtins(tmp_path: Path) -> None:
    root = tmp_path / "package"
    module_root = root / "package"
    module_root.mkdir(parents=True)
    (module_root / "resources.agl").write_text(
        "export std/core hiding resource\n", encoding="utf-8"
    )
    (module_root / "main.agl").write_text("program def main() -> unit = ()\n", encoding="utf-8")
    package = PackageInfo(root, PackageManifest("package", semver.Version.parse("1.0.0")))

    validate_package(package)


@pytest.mark.parametrize(
    "source",
    (
        "let =",
        """program def main() -> unit =
  print resource("../outside")
""",
    ),
)
def test_package_validation_rejects_invalid_modules_and_resource_calls(
    tmp_path: Path, source: str
) -> None:
    root = tmp_path / "package"
    module = root / "package" / "main.agl"
    module.parent.mkdir(parents=True)
    module.write_text(source, encoding="utf-8")
    package = PackageInfo(root, PackageManifest("package", semver.Version.parse("1.0.0")))

    with pytest.raises(DisciplineError):
        validate_package(package)


def test_package_validation_accepts_resource_dir(tmp_path: Path) -> None:
    root = tmp_path / "package"
    module = root / "package" / "main.agl"
    module.parent.mkdir(parents=True)
    prompt = root / "prompts" / "review.md"
    prompt.parent.mkdir()
    prompt.write_text("prompt", encoding="utf-8")
    module.write_text(
        """let root = resource-dir()
let prompt = resource("prompts/review.md")
program def main() -> unit = ()
""",
        encoding="utf-8",
    )
    package = PackageInfo(root, PackageManifest("package", semver.Version.parse("1.0.0")))

    validate_package(package)
