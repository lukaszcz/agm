"""Static module-parameter inventory exposed by program discovery."""

from __future__ import annotations

from pathlib import Path

from agm.agl.modules.ids import ModuleId
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import PipelineDriver, ProgramDiscovery


def _discover(source: str, tmp_path: Path) -> ProgramDiscovery:
    """Discover source with modules rooted at *tmp_path*."""
    prepared = PipelineDriver.prepare_program(
        source,
        roots=RootSet(roots=frozenset({tmp_path})),
        default_stdlib=False,
    )
    discovery = PipelineDriver().discover_programs(prepared)
    assert discovery.diagnostics == ()
    return discovery


def test_discovery_inventories_module_params_in_declaration_order(tmp_path: Path) -> None:
    discovery = _discover(
        """
@param @opt-name("loud") @doc("Root verbosity") let verbose: bool = false
@param var retries: int = 3

scope Logging
  @param @doc("Trace logging") let trace: bool = false
  @param var level: int = 1
end Logging

program def main() -> unit = ()
""",
        tmp_path,
    )

    params = discovery.module_params[discovery.programs[0].module]

    assert [(param.name, param.cli.name, param.mutable, param.scope_path) for param in params] == [
        ("verbose", "loud", False, ()),
        ("retries", "retries", True, ()),
        ("trace", "trace", False, ("Logging",)),
        ("level", "level", True, ("Logging",)),
    ]
    assert params[0].doc == "Root verbosity"
    assert params[2].doc == "Trace logging"
    assert params[0].declaration_path == "<entry>::verbose"
    assert params[2].declaration_path == "<entry>::Logging::trace"
    assert params[0].key == (discovery.programs[0].module, (), "verbose")


def test_discovery_closure_and_params_follow_source_reachability_order(tmp_path: Path) -> None:
    (tmp_path / "first.agl").write_text("@param let first: int = 1\n", encoding="utf-8")
    (tmp_path / "second.agl").write_text(
        "import third\n@param let second: int = 2\n", encoding="utf-8"
    )
    (tmp_path / "third.agl").write_text("@param let third: int = 3\n", encoding="utf-8")

    discovery = _discover(
        """
import first
import second
@param let root: int = 0
program def main() -> unit = ()
""",
        tmp_path,
    )
    program = discovery.programs[0]

    assert program.closure == (
        program.module,
        ModuleId(("first",)),
        ModuleId(("second",)),
        ModuleId(("third",)),
    )
    assert [param.name for param in discovery.params_for(program)] == [
        "root",
        "first",
        "second",
        "third",
    ]


def test_imported_program_uses_its_own_source_reachable_closure(tmp_path: Path) -> None:
    (tmp_path / "library.agl").write_text(
        "import dependency\n@param let library: int = 1\nprogram def run() -> unit = ()\n",
        encoding="utf-8",
    )
    (tmp_path / "dependency.agl").write_text("@param let dependency: int = 2\n", encoding="utf-8")

    discovery = _discover(
        "import library\n@param let entry: int = 0\nprogram def main() -> unit = ()\n",
        tmp_path,
    )
    imported_program = next(program for program in discovery.programs if program.name == "run")

    assert imported_program.closure == (ModuleId(("library",)), ModuleId(("dependency",)))
    assert [param.name for param in discovery.params_for(imported_program)] == [
        "library",
        "dependency",
    ]
    assert discovery.module_params[ModuleId(("library",))][0].declaration_path == "library::library"


def test_discovery_has_empty_module_param_inventory_without_param_bindings(tmp_path: Path) -> None:
    (tmp_path / "library.agl").write_text("let value: int = 1\n", encoding="utf-8")

    discovery = _discover(
        "import library\nprogram def main() -> unit = ()\n",
        tmp_path,
    )

    assert all(not params for params in discovery.module_params.values())
    assert tuple(discovery.params_for(discovery.programs[0])) == ()
