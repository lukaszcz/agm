"""General builtin-var registers and the ambient ``std/env`` binding."""

from __future__ import annotations

from pathlib import Path

from agm.agl.modules.ids import ModuleId
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import PipelineDriver
from agm.agl.repl import ReplSession
from agm.agl.semantics.values import BoolValue, DictValue, IntValue, RecordValue, TextValue
from tests._agl_helpers import agl_roots, run_inline_command

_STDLIB = Path(__file__).resolve().parent.parent / "stdlib"


def _run(source: str, **kwargs: object):
    return run_inline_command(PipelineDriver(), source, roots=agl_roots(), **kwargs)


def _environment_value(result: object, binding: str) -> RecordValue:
    value = getattr(result, "bindings")[binding]
    assert isinstance(value, RecordValue)
    return value


def test_stdlib_builtin_vars_are_keyed_by_module_and_name(tmp_path: Path) -> None:
    (tmp_path / "std").mkdir()
    (tmp_path / "std" / "state.agl").write_text("builtin var value: int = 1\n", encoding="utf-8")

    result = run_inline_command(
        PipelineDriver(),
        "import std/state\nstd/state::value := 2\nlet seen = std/state::value\nseen",
        roots=RootSet(roots=frozenset({tmp_path})),
        default_stdlib=False,
    )

    assert result.ok, result.diagnostics
    assert result.bindings["seen"].value == 2


def test_module_keyed_host_seeds_keep_same_named_builtin_vars_independent(
    tmp_path: Path,
) -> None:
    (tmp_path / "std").mkdir()
    (tmp_path / "std" / "first.agl").write_text("builtin var value: int\n", encoding="utf-8")
    (tmp_path / "std" / "second.agl").write_text("builtin var value: int\n", encoding="utf-8")

    result = run_inline_command(
        PipelineDriver(),
        "import std/first\nimport std/second\n"
        "let first = std/first::value\nlet second = std/second::value\nsecond",
        roots=RootSet(roots=frozenset({tmp_path})),
        default_stdlib=False,
        builtin_var_seeds={
            (ModuleId.from_path("std/first"), (), "value"): IntValue(1),
            (ModuleId.from_path("std/second"), (), "value"): IntValue(2),
        },
    )

    assert result.ok, result.diagnostics
    assert result.bindings["first"] == IntValue(1)
    assert result.bindings["second"] == IntValue(2)


def test_pipeline_run_accepts_module_keyed_host_seeds(tmp_path: Path) -> None:
    (tmp_path / "std").mkdir()
    (tmp_path / "std" / "state.agl").write_text("builtin var value: int\n", encoding="utf-8")

    result = PipelineDriver().run(
        "import std/state\n"
        "program def main() -> unit =\n"
        "  std/state::value := std/state::value + 1\n",
        roots=RootSet(roots=frozenset({tmp_path})),
        default_stdlib=False,
        builtin_var_seeds={(ModuleId.from_path("std/state"), (), "value"): IntValue(1)},
    )

    assert result.ok, result.diagnostics


def test_repl_accepts_scoped_module_qualified_host_seeds(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "state.agl").write_text(
        "scope First\n  builtin var value: int\nend First\n"
        "\n"
        "scope Second\n  builtin var value: int\nend Second\n",
        encoding="utf-8",
    )
    session = ReplSession(
        stdlib_root=tmp_path,
        default_stdlib=False,
        builtin_var_seeds={(ModuleId.from_path("std/state"), ("Second",), "value"): IntValue(2)},
    )

    result = session.eval_entry("import std/state\nstd/state::Second::value")

    assert result.ok, result.diagnostics
    assert result.value == IntValue(2)
    assert session.eval_entry("std/state::Second::value := 5").ok

    session.reset()
    restored = session.eval_entry("import std/state\nstd/state::Second::value")

    assert restored.ok, restored.diagnostics
    assert restored.value == IntValue(2)


def test_repl_accepts_root_engine_seeds_in_the_structured_api(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "config.agl").write_text(
        "builtin var strict-json: bool\n", encoding="utf-8"
    )
    session = ReplSession(
        stdlib_root=tmp_path,
        default_stdlib=False,
        builtin_var_seeds={(ModuleId.from_path("std/config"), (), "strict-json"): BoolValue(True)},
    )

    result = session.eval_entry("import std/config\nstd/config::strict-json")

    assert result.ok, result.diagnostics
    assert result.value == BoolValue(True)


def test_unseeded_non_engine_builtin_var_is_a_diagnostic_not_a_key_error(tmp_path: Path) -> None:
    (tmp_path / "std").mkdir()
    (tmp_path / "std" / "state.agl").write_text("builtin var value: int\n", encoding="utf-8")

    result = run_inline_command(
        PipelineDriver(),
        "import std/state\nstd/state::value",
        roots=RootSet(roots=frozenset({tmp_path})),
        default_stdlib=False,
    )

    assert not result.ok
    assert result.diagnostics
    assert "std/state::value" in result.diagnostics[0].message


def test_repl_unseeded_non_engine_builtin_var_is_a_diagnostic(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "state.agl").write_text("builtin var value: int\n", encoding="utf-8")
    session = ReplSession(stdlib_root=tmp_path, default_stdlib=False)

    result = session.eval_entry("import std/state\nstd/state::value")

    assert not result.ok
    assert result.diagnostics
    assert "std/state::value" in result.diagnostics[0].message


def test_std_env_uses_a_controlled_process_snapshot_without_mutating_os_environ() -> None:
    result = _run(
        "import std/env::*\n"
        'let before = getenv("SNAPSHOT_ONLY")\n'
        'setenv("SNAPSHOT_ONLY", "changed")\n'
        'let after = getenv("SNAPSHOT_ONLY")\n'
        "before",
        process_environment={"SNAPSHOT_ONLY": "original"},
    )

    assert result.ok, result.diagnostics
    assert result.bindings["before"] == TextValue("original")
    assert result.bindings["after"] == TextValue("changed")


def test_std_env_default_is_empty_when_no_process_snapshot_is_supplied() -> None:
    result = _run("import std/env::*\nlet env = environ\nenv")

    assert result.ok, result.diagnostics
    environ = _environment_value(result, "env")
    variables = environ.fields["vars"]
    assert isinstance(variables, DictValue)
    assert variables.entries == {}


def test_repl_reuses_its_startup_environment_snapshot() -> None:
    session = ReplSession(stdlib_root=_STDLIB, process_environment={"REPL_ONLY": "seeded"})

    assert session.open() == ()
    result = session.eval_entry('import std/env::*\ngetenv("REPL_ONLY")')

    assert result.ok, result.diagnostics
    assert result.value == TextValue("seeded")
    assert session.eval_entry('setenv("REPL_ONLY", "changed")').ok
    assert session.eval_entry('getenv("REPL_ONLY")').value == TextValue("changed")


def test_non_stdlib_library_builtin_var_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "library.agl").write_text("builtin var value: int = 1\n", encoding="utf-8")

    result = run_inline_command(
        PipelineDriver(),
        "import library\n()",
        roots=RootSet(roots=frozenset({tmp_path})),
        default_stdlib=False,
    )

    assert not result.ok
    assert result.diagnostics
