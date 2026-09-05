"""IR lowering shape for schema-free ``extern def`` declarations."""

from __future__ import annotations

from pathlib import Path

from agm.agl.capabilities import HostCapabilities
from agm.agl.ir import ExternFunctionBody, IrBind, IrMakeClosure
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import ArgumentPreflight, PipelineDriver
from agm.agl.runtime.arguments import ProgramArguments

_CAPS = HostCapabilities(
    supports_shell_exec=True,
    supports_extern=True,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset({"json", "record", "enum", "array", "dict", "int", "decimal", "bool"}),
    },
)


def _compile(source: str, companion: str, tmp_path: Path) -> ArgumentPreflight:
    """Run one single-module extern program through the whole pipeline."""
    entry = tmp_path / "entry.agl"
    entry.write_text(source)
    (tmp_path / "entry.py").write_text(companion)

    prepared = PipelineDriver().prepare_program(
        source,
        entry_path=entry,
        roots=RootSet(roots=frozenset({tmp_path})),
        default_stdlib=False,
    )
    runtime = PipelineDriver()
    discovery = runtime.discover_programs(prepared)
    (program,) = discovery.programs
    return runtime.preflight_arguments(
        prepared, program, ProgramArguments(positional=(), named={}), compiled=discovery.compiled
    )


def _extern_body(result: ArgumentPreflight) -> ExternFunctionBody:
    """Return the single extern implementation *result* lowered."""
    assert result.executable is not None
    descriptor = next(desc for desc in result.executable.functions.values() if desc.is_extern)
    assert isinstance(descriptor.impl, ExternFunctionBody)
    return descriptor.impl


def test_extern_descriptor_has_no_boundary_contract(tmp_path: Path) -> None:
    result = _compile(
        "extern def f(value: int) -> int\nprogram def main() -> unit =\n  let result = f(1)\n",
        "def f(value): return value\n",
        tmp_path,
    )

    assert result.result.ok
    body = _extern_body(result)
    assert body.name == "f"
    assert body.companion_name == "f"
    assert not hasattr(body, "contract")


def test_extern_still_initializes_as_a_function_closure(tmp_path: Path) -> None:
    result = _compile(
        "extern def f() -> unit\nprogram def main() -> unit = ()\n",
        "def f(): return None\n",
        tmp_path,
    )

    assert result.executable is not None
    initializer = result.executable.modules[result.executable.entry_module].initializers[0]
    assert isinstance(initializer, IrBind)
    assert isinstance(initializer.value, IrMakeClosure)


def test_extern_body_carries_the_attributed_companion_name(tmp_path: Path) -> None:
    result = _compile(
        '@extern-name("py_double")\n'
        "extern def double-it(value: int) -> int\n"
        "program def main() -> unit =\n"
        "  let result = double-it(1)\n",
        "def py_double(value): return value * 2\n",
        tmp_path,
    )

    assert result.result.ok
    body = _extern_body(result)
    assert body.name == "double-it"
    assert body.companion_name == "py_double"


def test_a_python_soft_keyword_name_compiles_through_the_pipeline(tmp_path: Path) -> None:
    result = _compile(
        "extern def match(value: int) -> int\n"
        "program def main() -> unit =\n"
        "  let result = match(1)\n",
        "def match(value): return value\n",
        tmp_path,
    )

    assert result.result.ok
    assert _extern_body(result).companion_name == "match"
