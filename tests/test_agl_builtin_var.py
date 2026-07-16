"""Tests for ``builtin var`` engine-setting declarations and the ``std.config`` module.

A ``builtin var NAME : Type`` is a body-less, runtime-register-backed, MUTABLE
binding.  Programs read it as an ordinary value and assign it with ``:=``.  The
shipped ``std.config`` library module declares the six engine keys, accessed via
qualified references (``std.config::max-iters``).  The declaration form also
works directly at an entry program's root (name-whitelisted like ``builtin def``).
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
) -> RunResult:
    """Run *source* (with imports) through the graph pipeline against the stdlib."""
    roots = RootSet(roots=frozenset({_STDLIB}) | extra_roots)
    rt = PipelineDriver(default_loop_limit=default_loop_limit)
    prepared = rt.prepare_program(source, entry_path=None, roots=roots)
    result = rt.run_prepared_graph(prepared)
    assert isinstance(result, RunResult)
    return result


# ---------------------------------------------------------------------------
# Entry-level ``builtin var`` (single module, int/bool/text keys)
# ---------------------------------------------------------------------------


class TestEntryBuiltinVar:
    def test_read_reflects_write(self) -> None:
        result = _run(
            "builtin var max-iters: int\nmax-iters := 3\nlet n = max-iters\nprint n"
        )
        assert result.ok, f"expected success but got: {result.error!r}"
        assert result.bindings["n"] == IntValue(3)

    def test_strict_json_write_then_read(self) -> None:
        result = _run(
            "builtin var strict-json: bool\n"
            "strict-json := true\n"
            "let b = strict-json\n"
            "print b"
        )
        assert result.ok
        assert result.bindings["b"] == BoolValue(True)

    def test_runner_default_reads_claude(self) -> None:
        result = _run("builtin var runner: text\nlet r = runner\nprint r")
        assert result.ok
        assert result.bindings["r"] == TextValue("claude")

    def test_runner_write_then_read(self) -> None:
        result = _run(
            "builtin var runner: text\nrunner := \"codex\"\nlet r = runner\nprint r"
        )
        assert result.ok
        assert result.bindings["r"] == TextValue("codex")

    def test_log_default_reads_false(self) -> None:
        result = _run("builtin var log: bool\nlet l = log\nprint l")
        assert result.ok
        assert result.bindings["l"] == BoolValue(False)

    def test_max_iters_default_reads_engine_default(self) -> None:
        """A read before any write reports the engine default (loop valve off)."""
        result = _run(
            "builtin var max-iters: int\nlet n = max-iters\nprint n",
            default_loop_limit=None,
        )
        assert result.ok
        assert result.bindings["n"] == IntValue(5)


# ---------------------------------------------------------------------------
# Placement + name-whitelist gate (mirrors the builtin-def gate)
# ---------------------------------------------------------------------------


class TestBuiltinVarGate:
    def test_unknown_key_rejected(self) -> None:
        """An unknown engine key is rejected (name whitelist, like builtin def)."""
        result = _run("builtin var bogus: int\nprint 1")
        assert not result.ok
        assert result.diagnostics

    def test_wrong_type_rejected(self) -> None:
        result = _run("builtin var max-iters: bool\nprint 1")
        assert not result.ok
        assert result.diagnostics

    def test_nested_declaration_rejected(self) -> None:
        """``builtin var`` is root-only, like ``builtin def``/``def``."""
        result = _run(
            "def f() -> text =\n  builtin var runner: text\n  \"x\"\nprint f()"
        )
        assert not result.ok
        assert result.diagnostics

    def test_unknown_key_in_graph_mode_rejected(self) -> None:
        """Unknown key rejected in graph mode too (covers the graph-table skip)."""
        result = _run_graph("import std.config\nbuiltin var bogus: int\nprint 1")
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

    def test_max_iters_zero_fails(self) -> None:
        result = _run_graph(
            "import std.config\nstd.config::max-iters := 0\nprint 1"
        )
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "ValueError"

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
        assert "Cannot assign" in result.diagnostics[0].message


# ---------------------------------------------------------------------------
# Coexistence: the old ``config`` mechanism is unaffected
# ---------------------------------------------------------------------------


