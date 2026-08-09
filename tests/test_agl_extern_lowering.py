"""IR lowering shape for schema-free ``extern def`` declarations."""

from __future__ import annotations

from pathlib import Path

from agm.agl.capabilities import HostCapabilities
from agm.agl.ir import ExternFunctionBody, IrBind, IrMakeClosure
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import PipelineDriver

_CAPS = HostCapabilities(
    supports_shell_exec=True,
    supports_extern=True,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset({"json", "record", "enum", "array", "dict", "int", "decimal", "bool"}),
    },
)


def test_extern_descriptor_has_no_boundary_contract(tmp_path: Path) -> None:
    entry = tmp_path / "entry.agl"
    entry.write_text(
        "extern def f(value: int) -> int\nprogram def main() -> unit =\n  let result = f(1)\n"
    )
    (tmp_path / "entry.py").write_text("def f(value): return value\n")

    prepared = PipelineDriver().prepare_program(
        entry.read_text(),
        entry_path=entry,
        roots=RootSet(roots=frozenset({tmp_path})),
        default_stdlib=False,
    )
    result = PipelineDriver().preflight_params(prepared)

    assert result.result.ok
    assert result.executable is not None
    descriptor = next(desc for desc in result.executable.functions.values() if desc.is_extern)
    assert isinstance(descriptor.impl, ExternFunctionBody)
    assert descriptor.impl.name == "f"
    assert not hasattr(descriptor.impl, "contract")


def test_extern_still_initializes_as_a_function_closure(tmp_path: Path) -> None:
    entry = tmp_path / "entry.agl"
    entry.write_text("extern def f() -> unit\n")
    (tmp_path / "entry.py").write_text("def f(): return None\n")
    prepared = PipelineDriver().prepare_program(
        entry.read_text(),
        entry_path=entry,
        roots=RootSet(roots=frozenset({tmp_path})),
        default_stdlib=False,
    )
    result = PipelineDriver().preflight_params(prepared)

    assert result.executable is not None
    initializer = result.executable.modules[result.executable.entry_module].initializers[0]
    assert isinstance(initializer, IrBind)
    assert isinstance(initializer.value, IrMakeClosure)
