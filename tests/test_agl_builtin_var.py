"""Tests for ``builtin var`` engine-setting declarations and the ``std.config`` module.

A ``builtin var NAME : Type`` is a body-less, engine-backed, mutable binding
reserved to the ``std.config`` standard-library module. Programs read it
through a qualified reference and assign it with ``:=``.
"""

from __future__ import annotations

from pathlib import Path

from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import PipelineDriver, RunResult
from agm.agl.semantics.values import BoolValue, EnumValue, IntValue, TextValue

_STDLIB = Path(__file__).resolve().parent.parent / "stdlib"


def _run(source: str, *, default_loop_limit: int | None = None) -> RunResult:
    """Run a single-module *source* (no imports) through prepare + run_prepared."""
    rt = PipelineDriver(default_loop_limit=default_loop_limit)
    prepared = rt.prepare(source)
    result = rt.run_prepared(prepared)
    assert isinstance(result, RunResult)
    return result


def _run_graph(
    source: str,
    *,
    extra_roots: frozenset[Path] = frozenset(),
    default_loop_limit: int | None = None,
    shell_exec_timeout: float | None = None,
) -> RunResult:
    """Run *source* (with imports) through the graph pipeline against the stdlib."""
    roots = RootSet(roots=frozenset({_STDLIB}) | extra_roots)
    rt = PipelineDriver(
        default_loop_limit=default_loop_limit,
        shell_exec_timeout=shell_exec_timeout,
    )
    prepared = rt.prepare_program(source, entry_path=None, roots=roots)
    result = rt.run_prepared_graph(prepared)
    assert isinstance(result, RunResult)
    return result


def _run_with_std_config(source: str, std_config: str, root: Path) -> RunResult:
    """Run against a test ``std.config`` module without ordinary entry declarations."""
    config_path = root / "std" / "config.agl"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(std_config, encoding="utf-8")
    rt = PipelineDriver()
    prepared = rt.prepare_program(
        source,
        entry_path=None,
        roots=RootSet(roots=frozenset({root})),
        default_stdlib=False,
    )
    result = rt.run_prepared_graph(prepared)
    assert isinstance(result, RunResult)
    return result


# ---------------------------------------------------------------------------
# Builtin-var behavior through the canonical std.config declarations
# ---------------------------------------------------------------------------


class TestBuiltinVarRegisters:
    def test_read_reflects_write(self) -> None:
        result = _run_graph(
            "import std.config\n"
            "std.config::max-iters := 3\n"
            "let n = std.config::max-iters\n"
            "print n"
        )
        assert result.ok, f"expected success but got: {result.error!r}"
        assert result.bindings["n"] == IntValue(3)

    def test_strict_json_write_then_read(self) -> None:
        result = _run_graph(
            "import std.config\n"
            "std.config::strict-json := true\n"
            "let b = std.config::strict-json\n"
            "print b"
        )
        assert result.ok
        assert result.bindings["b"] == BoolValue(True)

    def test_runner_default_reads_claude(self) -> None:
        result = _run_graph(
            "import std.config\nlet r = std.config::runner\nprint r"
        )
        assert result.ok
        assert result.bindings["r"] == TextValue("claude")

    def test_runner_write_then_read(self) -> None:
        result = _run_graph(
            "import std.config\n"
            'std.config::runner := "codex"\n'
            "let r = std.config::runner\n"
            "print r"
        )
        assert result.ok
        assert result.bindings["r"] == TextValue("codex")

    def test_log_default_reads_false(self) -> None:
        result = _run_graph("import std.config\nlet l = std.config::log\nprint l")
        assert result.ok
        assert result.bindings["l"] == BoolValue(False)

    def test_max_iters_default_reads_disabled_state(self) -> None:
        """A read reports zero when the host safety valve is off."""
        result = _run_graph(
            "import std.config\nlet n = std.config::max-iters\nprint n",
            default_loop_limit=None,
        )
        assert result.ok
        assert result.bindings["n"] == IntValue(0)


# ---------------------------------------------------------------------------
# Placement + name-whitelist gate (mirrors the builtin-def gate)
# ---------------------------------------------------------------------------


class TestBuiltinVarGate:
    def test_entry_declaration_rejected(self) -> None:
        result = _run("builtin var max-iters: int\nprint 1")
        assert not result.ok
        assert result.diagnostics

    def test_unknown_key_rejected(self, tmp_path: Path) -> None:
        """The canonical module may declare only registered engine keys."""
        result = _run_with_std_config(
            "import std.config\n()",
            "builtin var bogus: int",
            tmp_path,
        )
        assert not result.ok
        assert result.diagnostics

    def test_wrong_type_rejected(self, tmp_path: Path) -> None:
        result = _run_with_std_config(
            "import std.config\n()",
            "builtin var max-iters: bool",
            tmp_path,
        )
        assert not result.ok
        assert result.diagnostics

    def test_nested_declaration_rejected(self) -> None:
        """``builtin var`` is root-only, like ``builtin def``/``def``."""
        result = _run(
            "def f() -> text =\n  builtin var runner: text\n  \"x\"\nprint f()"
        )
        assert not result.ok
        assert result.diagnostics

    def test_arbitrary_library_declaration_rejected(self, tmp_path: Path) -> None:
        """A regular library cannot expose a register-backed declaration."""
        (tmp_path / "mylib.agl").write_text(
            "builtin var max-iters: int\n", encoding="utf-8"
        )
        result = _run_graph(
            "import mylib\nprint 1",
            extra_roots=frozenset({tmp_path}),
        )
        assert not result.ok
        assert result.diagnostics


# ---------------------------------------------------------------------------
# Qualified access via std.config
# ---------------------------------------------------------------------------


class TestStdConfigQualified:
    def test_qualified_read_reflects_write(self) -> None:
        result = _run_graph(
            "import std.config\n"
            "std.config::max-iters := 3\n"
            "let n = std.config::max-iters\n"
            "print n"
        )
        assert result.ok, f"expected success but got: {result.error!r}"
        assert result.bindings["n"] == IntValue(3)

    def test_qualified_strict_json(self) -> None:
        result = _run_graph(
            "import std.config\n"
            "std.config::strict-json := true\n"
            "let b = std.config::strict-json\n"
            "print b"
        )
        assert result.ok
        assert result.bindings["b"] == BoolValue(True)

    def test_qualified_runner_default(self) -> None:
        result = _run_graph(
            "import std.config\nlet r = std.config::runner\nprint r"
        )
        assert result.ok
        assert result.bindings["r"] == TextValue("claude")

    def test_max_iters_zero_disables_valve(self) -> None:
        result = _run_graph(
            "import std.config\n"
            "std.config::max-iters := 0\n"
            "var i = 0\n"
            "do\n"
            "  i := i + 1\n"
            "until i >= 6\n"
            "print i",
            default_loop_limit=2,
        )
        assert result.ok
        assert result.bindings["i"] == IntValue(6)

    def test_log_file_some_round_trips(self) -> None:
        result = _run_graph(
            "import std.config\n"
            "std.config::log-file := Some(\"x\")\n"
            "let f = std.config::log-file\n"
            "print f"
        )
        assert result.ok, f"expected success but got: {result.error!r}"
        bound = result.bindings["f"]
        assert isinstance(bound, EnumValue)
        assert bound.variant == "Some"
        assert bound.fields["value"] == TextValue("x")

    def test_timeout_default_reads_none(self) -> None:
        result = _run_graph(
            "import std.config\nlet t = std.config::timeout\nprint t"
        )
        assert result.ok
        bound = result.bindings["t"]
        assert isinstance(bound, EnumValue)
        assert bound.variant == "None"

    def test_timeout_write_then_read_is_some(self) -> None:
        result = _run_graph(
            "import std.config\n"
            "std.config::timeout := Some(\"2m\")\n"
            "let t = std.config::timeout\n"
            "print t"
        )
        assert result.ok, f"expected success but got: {result.error!r}"
        bound = result.bindings["t"]
        assert isinstance(bound, EnumValue)
        assert bound.variant == "Some"
        assert isinstance(bound.fields["value"], TextValue)

    def test_timeout_write_none_clears_the_shell_timeout(self) -> None:
        result = _run_graph(
            "import std.config\n"
            'std.config::timeout := Some("2m")\n'
            "std.config::timeout := None\n"
            "let t = std.config::timeout\n"
            "print t"
        )
        assert result.ok, f"expected success but got: {result.error!r}"
        bound = result.bindings["t"]
        assert isinstance(bound, EnumValue)
        assert bound.variant == "None"

    def test_timeout_preserves_raw_text_through_tiny_self_assignment(self) -> None:
        result = _run_graph(
            "import std.config\n"
            'std.config::timeout := Some("0.0001s")\n'
            "std.config::timeout := std.config::timeout\n"
            "let t = std.config::timeout\n"
            "t\n"
        )

        assert result.ok, f"expected success but got: {result.error!r}"
        bound = result.bindings["t"]
        assert isinstance(bound, EnumValue)
        assert bound.fields["value"] == TextValue("0.0001s")

    def test_tiny_host_timeout_can_be_assigned_back(self) -> None:
        result = _run_graph(
            "import std.config\n"
            "std.config::timeout := std.config::timeout\n"
            "let t = std.config::timeout\n"
            "t\n",
            shell_exec_timeout=0.0000001,
        )

        assert result.ok, f"expected success but got: {result.error!r}"
        bound = result.bindings["t"]
        assert isinstance(bound, EnumValue)
        assert bound.fields["value"] == TextValue("0.0000001s")

    def test_disabled_max_iters_round_trips_without_enabling_valve(self) -> None:
        result = _run_graph(
            "import std.config\n"
            "std.config::max-iters := std.config::max-iters\n"
            "var i = 0\n"
            "do\n"
            "  i := i + 1\n"
            "until i >= 6\n"
            "print i"
        )

        assert result.ok
        assert result.bindings["i"] == IntValue(6)


# ---------------------------------------------------------------------------
# Effect-at-binding: a write takes effect from that program point forward
# ---------------------------------------------------------------------------


class TestEffectAtBinding:
    def test_loop_after_write_uses_new_limit(self) -> None:
        """A ``max-iters := 3`` before an unguarded loop raises the effective cap."""
        source = (
            "import std.config\n"
            "var i: int = 0\n"
            "std.config::max-iters := 3\n"
            "do\n"
            "  i := i + 1\n"
            "until i >= 2\n"
        )
        result = _run_graph(source, default_loop_limit=1)
        assert result.ok, f"expected success but got: {result.error!r}"

    def test_loop_before_write_uses_initial_limit(self) -> None:
        """A loop before the write uses the initial (small) cap and overflows."""
        source = (
            "import std.config\n"
            "var i: int = 0\n"
            "do\n"
            "  i := i + 1\n"
            "until i >= 5\n"
            "std.config::max-iters := 3\n"
        )
        result = _run_graph(source, default_loop_limit=1)
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "MaxIterationsExceeded"


# ---------------------------------------------------------------------------
# Qualified assignment: rejections
# ---------------------------------------------------------------------------


class TestQualifiedAssignRejections:
    def test_single_module_qualified_assign_rejected(self) -> None:
        """A qualified assignment target with no imports is undeclared."""
        result = _run("Foo::x := 1\nprint 1")
        assert not result.ok
        assert result.diagnostics

    def test_self_ref_qualified_assign_rejected(self) -> None:
        """``::name := expr`` is not a valid mutable target."""
        result = _run_graph("import std.config\n::x := 1\nprint 1")
        assert not result.ok
        assert result.diagnostics

    def test_immutable_cross_module_assign_rejected(self, tmp_path: Path) -> None:
        """Assigning a non-``builtin var`` cross-module binding is rejected."""
        (tmp_path / "mylib.agl").write_text("def foo() -> int = 1\n", encoding="utf-8")
        result = _run_graph(
            "import mylib\nmylib::foo := 3\nprint 1",
            extra_roots=frozenset({tmp_path}),
        )
        assert not result.ok
        assert result.diagnostics
