"""Declaration coverage for methods on built-in receiver types."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl import PipelineDriver
from agm.agl.modules.ids import ModuleId
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import PreparedProgram
from agm.agl.semantics.types import ArrayType, DictType, IntType, TypeVarType
from agm.agl.syntax import FuncDef
from agm.agl.typecheck.env import FunctionSignature
from agm.agl.typecheck.program import CheckedProgram


def _prepare_stdlib_module(tmp_path: Path, module: str, source: str) -> PreparedProgram:
    root = tmp_path / "stdlib"
    module_path = root / f"{module}.agl"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(source, encoding="utf-8")
    if "extern def" in source:
        module_path.with_suffix(".py").write_text("", encoding="utf-8")
    return PipelineDriver.prepare_program(
        f"import {module}\nprogram def main() -> unit = ()\n",
        roots=RootSet(roots=frozenset({root})),
        default_stdlib=False,
    )


@pytest.mark.parametrize(
    "source",
    (
        "def array[int]::copy(self) -> array[int] = self\n",
        "def dict[text, array[int]]::copy(self) -> unit = ()\n",
        "def bytes[int]::copy(self) -> unit = ()\n",
    ),
)
def test_invalid_builtin_method_receivers_are_rejected(source: str) -> None:
    prepared = PipelineDriver.prepare_program(source, default_stdlib=False)
    discovery = PipelineDriver().discover_programs(prepared)
    assert discovery.checked is None
    assert discovery.diagnostics


def test_bare_array_scope_is_not_a_builtin_receiver_head() -> None:
    prepared = PipelineDriver.prepare_program(
        "def array::copy(self) -> int = 0\nprogram def main() -> unit = ()\n",
        default_stdlib=False,
    )

    discovery = PipelineDriver().discover_programs(prepared)

    assert discovery.checked is None
    assert discovery.diagnostics


def _signature_for(checked: CheckedProgram, module_path: str) -> FunctionSignature:
    module = next(
        module
        for module_id, module in checked.modules.items()
        if module_id.path_str() == module_path
    )
    declaration = module.resolved.program.body.items[0]
    assert isinstance(declaration, FuncDef)
    signature = module.type_env.get_function_signature_by_node_id(declaration.node_id)
    assert signature is not None
    return signature


@pytest.mark.parametrize(
    ("module", "source"),
    (
        ("std/array", "def array[E]::ordinary(self) -> array[E] = self\n"),
        ("std/array", "builtin def array[E]::copy(self) -> array[E]\n"),
        ("std/array", "extern def array[E]::external(self) -> array[E]\n"),
        ("std/dict", "def dict[text, V]::ordinary(self) -> dict[text, V] = self\n"),
        ("std/dict", "builtin def dict[text, V]::copy(self) -> dict[text, V]\n"),
        ("std/dict", "extern def dict[text, V]::external(self) -> dict[text, V]\n"),
        ("std/text", "def text::ordinary(self) -> text = self\n"),
        ("std/text", "builtin def text::copy(self) -> text\n"),
        ("std/text", "extern def text::external(self) -> text\n"),
        ("std/json", "def json::ordinary(self) -> json = self\n"),
        ("std/json", "builtin def json::copy(self) -> json\n"),
        ("std/json", "extern def json::external(self) -> json\n"),
        ("std/math", "def int::ordinary(self) -> int = self\n"),
        ("std/math", "builtin def int::copy(self) -> int\n"),
        ("std/math", "extern def int::external(self) -> int\n"),
        ("std/math", "def decimal::ordinary(self) -> decimal = self\n"),
        ("std/math", "builtin def decimal::copy(self) -> decimal\n"),
        ("std/math", "extern def decimal::external(self) -> decimal\n"),
        ("std/math", "def bool::ordinary(self) -> bool = self\n"),
        ("std/math", "builtin def bool::copy(self) -> bool\n"),
        ("std/math", "extern def bool::external(self) -> bool\n"),
    ),
)
def test_owning_stdlib_modules_accept_builtin_method_declarations(
    tmp_path: Path, module: str, source: str
) -> None:
    prepared = _prepare_stdlib_module(tmp_path, module, source)

    discovery = PipelineDriver().discover_programs(prepared)

    assert discovery.checked is not None, discovery.diagnostics


def test_builtin_receiver_declaration_accepts_any_module(tmp_path: Path) -> None:
    prepared = _prepare_stdlib_module(
        tmp_path,
        "std/math",
        "def array[E]::wrong-owner(self) -> array[E] = self\n",
    )

    assert prepared.resolved is not None, prepared.diagnostics

    discovery = PipelineDriver().discover_programs(prepared)

    assert discovery.checked is not None, discovery.diagnostics


def test_builtin_receiver_host_declaration_requires_a_supported_route(tmp_path: Path) -> None:
    prepared = _prepare_stdlib_module(
        tmp_path,
        "std/text",
        "builtin def text::host(self) -> text\n",
    )

    discovery = PipelineDriver().discover_programs(prepared)

    assert discovery.checked is None
    assert discovery.diagnostics


@pytest.mark.parametrize(
    ("module", "source"),
    (
        ("std/math", "builtin def int::print(self) -> json\n"),
        ("std/math", "builtin def int::copy(self) -> decimal\n"),
    ),
)
def test_builtin_receiver_host_declaration_must_match_its_route(
    tmp_path: Path, module: str, source: str
) -> None:
    prepared = _prepare_stdlib_module(tmp_path, module, source)

    discovery = PipelineDriver().discover_programs(prepared)

    assert discovery.checked is None
    assert discovery.diagnostics


def test_builtin_receiver_wildcard_uses_a_private_rigid_type_parameter(tmp_path: Path) -> None:
    prepared = _prepare_stdlib_module(
        tmp_path,
        "std/array",
        "def array[_]::count(self) -> int = 0\n",
    )

    discovery = PipelineDriver().discover_programs(prepared)

    assert discovery.checked is not None, discovery.diagnostics
    signature = _signature_for(discovery.checked, "std/array")
    (type_parameter,) = signature.type_params
    assert type_parameter.startswith("__method_type_slot_")
    assert signature.params[0].type == ArrayType(TypeVarType(type_parameter))


def test_unknown_applied_receiver_uses_its_head_name_as_its_scope(tmp_path: Path) -> None:
    prepared = _prepare_stdlib_module(
        tmp_path,
        "std/array",
        "def bytes[E]::copy(self) -> unit = ()\n",
    )

    assert prepared.resolved is not None, prepared.diagnostics
    resolved_module = prepared.resolved.modules[ModuleId.from_path("std/array")]
    (declaration,) = resolved_module.resolved.program.body.items
    assert isinstance(declaration, FuncDef)

    assert tuple(segment.name for segment in declaration.scope_path) == ("bytes",)


def test_generic_builtin_receiver_binds_its_receiver_slot(tmp_path: Path) -> None:
    prepared = _prepare_stdlib_module(
        tmp_path,
        "std/array",
        "def array[E]::map[U](self, f: (E) -> U) -> array[U] = []\n",
    )

    discovery = PipelineDriver().discover_programs(prepared)

    assert discovery.checked is not None, discovery.diagnostics
    signature = _signature_for(discovery.checked, "std/array")
    assert signature.params[0].type == ArrayType(TypeVarType("E"))
    assert signature.result == ArrayType(TypeVarType("U"))


def test_builtin_dict_receiver_wildcard_uses_a_private_rigid_type_parameter(tmp_path: Path) -> None:
    prepared = _prepare_stdlib_module(
        tmp_path,
        "std/dict",
        "def dict[text, _]::size(self) -> int = 0\n",
    )

    discovery = PipelineDriver().discover_programs(prepared)

    assert discovery.checked is not None, discovery.diagnostics
    signature = _signature_for(discovery.checked, "std/dict")
    (type_parameter,) = signature.type_params
    assert type_parameter.startswith("__method_type_slot_")
    assert signature.params[0].type == DictType(TypeVarType(type_parameter))


def test_generic_dict_receiver_binds_its_value_slot(tmp_path: Path) -> None:
    prepared = _prepare_stdlib_module(
        tmp_path,
        "std/dict",
        "def dict[text, V]::size(self) -> int = 0\n",
    )

    discovery = PipelineDriver().discover_programs(prepared)

    assert discovery.checked is not None, discovery.diagnostics
    signature = _signature_for(discovery.checked, "std/dict")
    assert signature.params[0].type == DictType(TypeVarType("V"))


def test_scalar_builtin_receiver_has_its_declared_type(tmp_path: Path) -> None:
    prepared = _prepare_stdlib_module(
        tmp_path,
        "std/math",
        "builtin def int::copy(self) -> int\n",
    )

    discovery = PipelineDriver().discover_programs(prepared)

    assert discovery.checked is not None, discovery.diagnostics
    signature = _signature_for(discovery.checked, "std/math")
    assert signature.params[0].type == IntType()
