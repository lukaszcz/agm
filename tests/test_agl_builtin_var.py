"""Tests for ``builtin var`` declarations and ``std/config`` engine settings.

A ``builtin var NAME : Type`` is a body-less, host-backed mutable binding in a
standard-library module. ``std/config`` reserves its bindings for engine
settings. Programs read a binding through a qualified reference and assign it
with ``:=``.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from agm.agl.modules.roots import RootSet
from agm.agl.parser import parse_program
from agm.agl.pipeline import PipelineDriver, RunResult
from agm.agl.runtime.option import some_value
from agm.agl.semantics.values import BoolValue, IntValue, RecordValue, TextValue, Value
from agm.agl.syntax import BuiltinVarDecl, Call, Expr, VarRef, walk
from agm.agl.syntax.constants import is_constant_expression
from tests._agl_helpers import agent_value, run_inline_command

_STDLIB = Path(__file__).resolve().parent.parent / "stdlib"


def _run(source: str, *, default_loop_limit: int | None = None) -> RunResult:
    """Run a single-module *source* (no imports) through prepare + run_prepared."""
    rt = PipelineDriver(default_loop_limit=default_loop_limit)
    return run_inline_command(rt, source)


def _run_program(
    source: str,
    *,
    extra_roots: frozenset[Path] = frozenset(),
    default_loop_limit: int | None = None,
    shell_exec_timeout: float | None = None,
    builtin_host_settings: dict[str, Value] | None = None,
) -> RunResult:
    """Run *source* (with imports) through the program pipeline against the stdlib."""
    roots = RootSet(roots=frozenset({_STDLIB}) | extra_roots)
    rt = PipelineDriver(
        default_loop_limit=default_loop_limit,
        shell_exec_timeout=shell_exec_timeout,
    )
    return run_inline_command(rt, source, roots=roots, builtin_host_settings=builtin_host_settings)


def _run_with_std_config(
    source: str,
    std_config: str,
    root: Path,
    *,
    default_loop_limit: int | None = None,
    builtin_host_settings: dict[str, Value] | None = None,
) -> RunResult:
    """Run against a test ``std/config`` module without ordinary entry declarations."""
    config_path = root / "std" / "config.agl"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(std_config, encoding="utf-8")
    rt = PipelineDriver(default_loop_limit=default_loop_limit)
    return run_inline_command(
        rt,
        source,
        roots=RootSet(roots=frozenset({root})),
        default_stdlib=False,
        builtin_host_settings=builtin_host_settings,
    )


# ---------------------------------------------------------------------------
# Builtin-var behavior through the canonical std/config declarations
# ---------------------------------------------------------------------------


class TestBuiltinVarRegisters:
    def test_read_reflects_write(self) -> None:
        result = _run_program(
            "import std/config\nstd/config::max-iters := 3\nlet n = std/config::max-iters\nprint n"
        )
        assert result.ok, f"expected success but got: {result.error!r}"
        assert result.bindings["n"] == IntValue(3)

    def test_suffix_and_anchored_writes_target_the_engine_setting(self) -> None:
        result = _run_program(
            "import std/config\n"
            "config::max-iters := 3\n"
            "/std/config::max-iters := 4\n"
            "let n = config::max-iters\n"
            "print n"
        )
        assert result.ok, f"expected success but got: {result.error!r}"
        assert result.bindings["n"] == IntValue(4)

    def test_strict_json_write_then_read(self) -> None:
        result = _run_program(
            "import std/config::*\n"
            "std/config::strict-json := true\n"
            "let b = std/config::strict-json\n"
            "print b"
        )
        assert result.ok
        assert result.bindings["b"] == BoolValue(True)

    def test_log_default_reads_false(self) -> None:
        result = _run_program("import std/config::*\nlet l = std/config::log\nprint l")
        assert result.ok
        assert result.bindings["l"] == BoolValue(False)

    def test_max_iters_default_reads_disabled_state(self) -> None:
        """A read reports zero when the host safety valve is off."""
        result = _run_program(
            "import std/config::*\nlet n = std/config::max-iters\nprint n",
            default_loop_limit=None,
        )
        assert result.ok
        assert result.bindings["n"] == IntValue(0)


# ---------------------------------------------------------------------------
# Placement + name-whitelist gate (mirrors the builtin-def gate)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("[1]", True),
        ("[value]", False),
        ('{"key": 1}', True),
        ('{"key": value}', False),
        ("Ctor", True),
        ("value", False),
        ("Ctor::[text]", True),
        ("value::[text]", False),
        ("Ctor(1, value=1)", True),
        ("value()", False),
        ("Ctor(value)", False),
        ("Ctor(1, value=value)", False),
    ),
)
def test_constant_expression_validation(source: str, expected: bool) -> None:
    (item,) = parse_program(source).body.items
    expression = cast(Expr, item)
    constructor_node_ids: set[int] = set()

    def collect_constructor_node_id(node: object) -> None:
        if isinstance(node, VarRef) and node.name == "Ctor":
            constructor_node_ids.add(node.node_id)

    walk(expression, collect_constructor_node_id)

    assert (
        is_constant_expression(
            expression,
            is_constructor=lambda node_id: node_id in constructor_node_ids,
        )
        is expected
    )


class TestBuiltinVarDefaults:
    def test_initializer_is_the_engine_default_without_a_host_seed(self, tmp_path: Path) -> None:
        result = _run_with_std_config(
            "import std/config::*\nlet value = std/config::default-agent\nvalue",
            'builtin var default-agent: Agent = AgentCommand("declared")',
            tmp_path,
        )

        assert result.ok, f"expected success but got: {result.error!r}"
        value = result.bindings["value"]
        assert isinstance(value, RecordValue)
        assert value.display_name.rsplit("::", maxsplit=1)[-1] == "AgentCommand"
        assert value.fields["command"] == TextValue("declared")

    def test_host_seed_overrides_the_declared_default(self, tmp_path: Path) -> None:
        result = _run_with_std_config(
            "import std/config::*\nlet value = std/config::default-agent\nvalue",
            'builtin var default-agent: Agent = AgentCommand("declared")',
            tmp_path,
            builtin_host_settings={
                "default-agent": agent_value("AgentCommand", command="seeded"),
            },
        )

        assert result.ok, f"expected success but got: {result.error!r}"
        value = result.bindings["value"]
        assert isinstance(value, RecordValue)
        assert value.fields["command"] == TextValue("seeded")

    def test_initializer_must_match_the_declared_type(self, tmp_path: Path) -> None:
        result = _run_with_std_config(
            "import std/config::*\n()",
            "builtin var default-agent: Agent = false",
            tmp_path,
        )

        assert not result.ok
        assert result.diagnostics

    def test_initializer_must_be_constant(self, tmp_path: Path) -> None:
        declaration_source = 'builtin var default-agent: Agent = AgentCommand("not " + "constant")'
        (declaration,) = parse_program(declaration_source).body.items
        assert isinstance(declaration, BuiltinVarDecl)
        assert isinstance(declaration.default, Call)

        result = _run_with_std_config(
            "import std/config::*\n()",
            declaration_source,
            tmp_path,
        )

        assert not result.ok
        assert result.diagnostics


class TestBuiltinVarGate:
    def test_entry_declaration_rejected(self) -> None:
        result = _run("builtin var max-iters: int\nprint 1")
        assert not result.ok
        assert result.diagnostics

    def test_unknown_key_rejected(self, tmp_path: Path) -> None:
        """The canonical module may declare only registered engine keys."""
        result = _run_with_std_config(
            "import std/config::*\n()",
            "builtin var bogus: int",
            tmp_path,
        )
        assert not result.ok
        assert result.diagnostics

    def test_wrong_type_rejected(self, tmp_path: Path) -> None:
        result = _run_with_std_config(
            "import std/config::*\n()",
            "builtin var max-iters: bool",
            tmp_path,
        )
        assert not result.ok
        assert result.diagnostics

    def test_nested_declaration_rejected(self) -> None:
        """``builtin var`` is root-only, like ``builtin def``/``def``."""
        result = _run('def f() -> text =\n  builtin var runner: text\n  "x"\nprint f()')
        assert not result.ok
        assert result.diagnostics

    def test_arbitrary_library_declaration_rejected(self, tmp_path: Path) -> None:
        """A regular library cannot expose a register-backed declaration."""
        (tmp_path / "mylib.agl").write_text("builtin var max-iters: int\n", encoding="utf-8")
        result = _run_program(
            "import mylib\nprint 1",
            extra_roots=frozenset({tmp_path}),
        )
        assert not result.ok
        assert result.diagnostics


# ---------------------------------------------------------------------------
# `builtin var` inside a named scope region
# ---------------------------------------------------------------------------


class TestScopedBuiltinVar:
    def test_scoped_builtin_var_round_trips_through_its_full_path(self, tmp_path: Path) -> None:
        """A scoped ``std/config`` binding is read and written through its full path."""
        result = _run_with_std_config(
            "import std/config::*\n"
            "std/config::Region::max-iters := 3\n"
            "let n = std/config::Region::max-iters\n"
            "n",
            "scope Region\nbuiltin var max-iters: int\nend Region",
            tmp_path,
        )
        assert result.ok, f"expected success but got: {result.error!r}"
        assert result.bindings["n"] == IntValue(3)

    def test_scoped_builtin_var_is_bare_after_use(self, tmp_path: Path) -> None:
        result = _run_with_std_config(
            "import std/config::*\nuse std/config::Region::*\nmax-iters := 4\nlet n = max-iters\nn",
            "scope Region\nbuiltin var max-iters: int\nend Region",
            tmp_path,
        )
        assert result.ok, f"expected success but got: {result.error!r}"
        assert result.bindings["n"] == IntValue(4)

    def test_scoped_config_binding_is_not_an_engine_setting(self, tmp_path: Path) -> None:
        result = _run_with_std_config(
            "import std/config\n"
            'std/config::Region::runner := "updated"\n'
            "let value = std/config::Region::runner\n"
            "value",
            'scope Region\nbuiltin var runner: text = "declared"\nend Region',
            tmp_path,
        )

        assert result.ok, f"expected success but got: {result.error!r}"
        assert result.bindings["value"] == TextValue("updated")

    def test_scoped_same_named_settings_are_independent(self, tmp_path: Path) -> None:
        result = _run_with_std_config(
            "import std/config\n"
            "std/config::First::max-iters := 3\n"
            "let second = std/config::Second::max-iters\n"
            "second",
            "scope First\nbuiltin var max-iters: int = 1\nend First\n"
            "scope Second\nbuiltin var max-iters: int = 2\nend Second",
            tmp_path,
        )

        assert result.ok, f"expected success but got: {result.error!r}"
        assert result.bindings["second"] == IntValue(2)

    def test_scoped_engine_named_binding_does_not_apply_engine_effects(
        self, tmp_path: Path
    ) -> None:
        result = _run_with_std_config(
            "import std/config\n"
            "std/config::Region::max-iters := 3\n"
            "var i = 0\n"
            "do\n"
            "  i := i + 1\n"
            "until i >= 2\n",
            "scope Region\nbuiltin var max-iters: int = 0\nend Region",
            tmp_path,
            default_loop_limit=1,
        )

        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "MaxIterationsExceeded"

    def test_scoped_declaration_still_confined_to_std_config(self, tmp_path: Path) -> None:
        """A scoped ``builtin var`` outside ``std/config`` is rejected, same as a root one."""
        (tmp_path / "mylib.agl").write_text(
            "scope Region\nbuiltin var max-iters: int\nend Region\n", encoding="utf-8"
        )
        result = _run_program(
            "import mylib\nprint 1",
            extra_roots=frozenset({tmp_path}),
        )
        assert not result.ok
        assert result.diagnostics

    def test_existing_std_config_behavior_is_unchanged(self) -> None:
        """A root, unscoped ``builtin var`` still round-trips as before."""
        result = _run_program(
            "import std/config\nstd/config::max-iters := 3\nlet n = std/config::max-iters\nprint n"
        )
        assert result.ok, f"expected success but got: {result.error!r}"
        assert result.bindings["n"] == IntValue(3)


# ---------------------------------------------------------------------------
# Qualified access via std/config
# ---------------------------------------------------------------------------


class TestStdConfigQualified:
    def test_qualified_read_reflects_write(self) -> None:
        result = _run_program(
            "import std/config\nstd/config::max-iters := 3\nlet n = std/config::max-iters\nprint n"
        )
        assert result.ok, f"expected success but got: {result.error!r}"
        assert result.bindings["n"] == IntValue(3)

    def test_qualified_strict_json(self) -> None:
        result = _run_program(
            "import std/config::*\n"
            "std/config::strict-json := true\n"
            "let b = std/config::strict-json\n"
            "print b"
        )
        assert result.ok
        assert result.bindings["b"] == BoolValue(True)

    def test_max_iters_zero_disables_valve(self) -> None:
        result = _run_program(
            "import std/config::*\n"
            "std/config::max-iters := 0\n"
            "var i = 0\n"
            "do\n"
            "  i := i + 1\n"
            "until i >= 6\n"
            "print i",
            default_loop_limit=2,
        )
        assert result.ok
        assert result.bindings["i"] == IntValue(6)

    def test_negative_max_iters_raises_catchable_type_error(self) -> None:
        """Assigning a negative ``max-iters`` raises a catchable built-in exception.

        Regression test: the live engine-setting effect built this exception
        under an undeclared name (``"ValueError"``, never a real AgL built-in
        exception), which went unnoticed because nothing validated the name.
        The declared ``TypeError`` is what must be raised.
        """
        result = _run_program("import std/config::*\nstd/config::max-iters := -1\nprint 1")
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "TypeError"

        caught = _run_program(
            "import std/config::*\n"
            "let caught = try\n"
            "    std/config::max-iters := -1\n"
            "    false\n"
            "  catch TypeError as e =>\n"
            "    true\n"
        )
        assert caught.ok, caught.diagnostics
        assert caught.bindings["caught"] == BoolValue(True)

    def test_invalid_timeout_raises_catchable_type_error(self) -> None:
        """Assigning an unparseable ``timeout`` raises a catchable built-in exception.

        Regression test: see ``test_negative_max_iters_raises_catchable_type_error``.
        """
        result = _run_program(
            'import std/config::*\nstd/config::timeout := Some("not-a-timeout")\nprint 1'
        )
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "TypeError"

        caught = _run_program(
            "import std/config::*\n"
            "let caught = try\n"
            '    std/config::timeout := Some("not-a-timeout")\n'
            "    false\n"
            "  catch TypeError as e =>\n"
            "    true\n"
        )
        assert caught.ok, caught.diagnostics
        assert caught.bindings["caught"] == BoolValue(True)

    def test_log_file_some_round_trips(self) -> None:
        result = _run_program(
            "import std/config::*\n"
            'std/config::log-file := Some("x")\n'
            "let f = std/config::log-file\n"
            "print f"
        )
        assert result.ok, f"expected success but got: {result.error!r}"
        bound = result.bindings["f"]
        assert isinstance(bound, RecordValue)
        assert bound.display_name.rsplit("::", maxsplit=1)[-1] == "Some"
        assert bound.fields["value"] == TextValue("x")

    def test_clearing_log_file_does_not_enable_logging(self) -> None:
        result = _run_program(
            "import std/config::*\n"
            "std/config::log := false\n"
            "std/config::log-file := None\n"
            "let enabled = std/config::log\n"
            "enabled"
        )

        assert result.ok, f"expected success but got: {result.error!r}"
        assert result.bindings["enabled"] == BoolValue(False)

    def test_timeout_default_reads_none(self) -> None:
        result = _run_program("import std/config::*\nlet t = std/config::timeout\nprint t")
        assert result.ok
        bound = result.bindings["t"]
        assert isinstance(bound, RecordValue)
        assert bound.display_name.rsplit("::", maxsplit=1)[-1] == "None"

    def test_timeout_write_then_read_is_some(self) -> None:
        result = _run_program(
            "import std/config::*\n"
            'std/config::timeout := Some("2m")\n'
            "let t = std/config::timeout\n"
            "print t"
        )
        assert result.ok, f"expected success but got: {result.error!r}"
        bound = result.bindings["t"]
        assert isinstance(bound, RecordValue)
        assert bound.display_name.rsplit("::", maxsplit=1)[-1] == "Some"
        assert isinstance(bound.fields["value"], TextValue)

    def test_timeout_write_none_clears_the_shell_timeout(self) -> None:
        result = _run_program(
            "import std/config::*\n"
            'std/config::timeout := Some("2m")\n'
            "std/config::timeout := None\n"
            "let t = std/config::timeout\n"
            "print t"
        )
        assert result.ok, f"expected success but got: {result.error!r}"
        bound = result.bindings["t"]
        assert isinstance(bound, RecordValue)
        assert bound.display_name.rsplit("::", maxsplit=1)[-1] == "None"

    def test_timeout_preserves_raw_text_through_tiny_self_assignment(self) -> None:
        result = _run_program(
            "import std/config::*\n"
            'std/config::timeout := Some("0.0001s")\n'
            "std/config::timeout := std/config::timeout\n"
            "let t = std/config::timeout\n"
            "t\n"
        )

        assert result.ok, f"expected success but got: {result.error!r}"
        bound = result.bindings["t"]
        assert isinstance(bound, RecordValue)
        assert bound.fields["value"] == TextValue("0.0001s")

    def test_explicit_timeout_seed_overrides_shell_timeout(self) -> None:
        result = _run_program(
            "import std/config::*\nlet t = std/config::timeout\nt\n",
            shell_exec_timeout=2.0,
            builtin_host_settings={"timeout": some_value(TextValue("45s"))},
        )

        assert result.ok, f"expected success but got: {result.error!r}"
        bound = result.bindings["t"]
        assert isinstance(bound, RecordValue)
        assert bound.fields["value"] == TextValue("45s")

    def test_tiny_host_timeout_can_be_assigned_back(self) -> None:
        result = _run_program(
            "import std/config::*\n"
            "std/config::timeout := std/config::timeout\n"
            "let t = std/config::timeout\n"
            "t\n",
            shell_exec_timeout=0.0000001,
        )

        assert result.ok, f"expected success but got: {result.error!r}"
        bound = result.bindings["t"]
        assert isinstance(bound, RecordValue)
        assert bound.fields["value"] == TextValue("0.0000001s")

    def test_disabled_max_iters_round_trips_without_enabling_valve(self) -> None:
        result = _run_program(
            "import std/config::*\n"
            "std/config::max-iters := std/config::max-iters\n"
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
            "import std/config::*\n"
            "var i: int = 0\n"
            "std/config::max-iters := 3\n"
            "do\n"
            "  i := i + 1\n"
            "until i >= 2\n"
        )
        result = _run_program(source, default_loop_limit=1)
        assert result.ok, f"expected success but got: {result.error!r}"

    def test_loop_before_write_uses_initial_limit(self) -> None:
        """A loop before the write uses the initial (small) cap and overflows."""
        source = (
            "import std/config::*\n"
            "var i: int = 0\n"
            "do\n"
            "  i := i + 1\n"
            "until i >= 5\n"
            "std/config::max-iters := 3\n"
        )
        result = _run_program(source, default_loop_limit=1)
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "MaxIterationsExceeded"


# ---------------------------------------------------------------------------
# Bare (unqualified) assignment through an import tail
# ---------------------------------------------------------------------------


class TestBareCrossModuleAssign:
    def test_bare_write_reflected_by_bare_read(self) -> None:
        """An import makes::* a setting assignable and readable without a qualifier."""
        result = _run_program("import std/config::*\nmax-iters := 3\nlet n = max-iters\nprint n")
        assert result.ok, f"expected success but got: {result.error!r}"
        assert result.bindings["n"] == IntValue(3)

    def test_bare_write_reflected_by_qualified_read(self) -> None:
        """Bare and qualified targets denote the same binding."""
        result = _run_program(
            "import std/config::*\nmax-iters := 4\nlet n = std/config::max-iters\nprint n"
        )
        assert result.ok, f"expected success but got: {result.error!r}"
        assert result.bindings["n"] == IntValue(4)

    def test_bare_write_takes_effect_on_loop_limit(self) -> None:
        """A bare write changes engine behavior, not just the readable value."""
        result = _run_program(
            "import std/config::*\n"
            "var i: int = 0\n"
            "max-iters := 3\n"
            "do\n"
            "  i := i + 1\n"
            "until i >= 2\n",
            default_loop_limit=1,
        )
        assert result.ok, f"expected success but got: {result.error!r}"

    def test_bare_write_to_qualified_only_import_rejected(self, tmp_path: Path) -> None:
        """A qualified-only import does not expose the name for bare assignment."""
        result = _run_program("import std/config as cfg\nmax-iters := 3\nprint 1")
        assert not result.ok
        assert result.diagnostics

    def test_bare_write_to_immutable_import_rejected(self, tmp_path: Path) -> None:
        """A ``def`` exposed by an import tail is still not a valid assignment target."""
        (tmp_path / "mylib.agl").write_text("def foo() -> int = 1\n", encoding="utf-8")
        result = _run_program(
            "import mylib\nfoo := 3\nprint 1",
            extra_roots=frozenset({tmp_path}),
        )
        assert not result.ok
        assert result.diagnostics


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
        result = _run_program("import std/config::*\n::x := 1\nprint 1")
        assert not result.ok
        assert result.diagnostics

    def test_immutable_cross_module_assign_rejected(self, tmp_path: Path) -> None:
        """Assigning a non-``builtin var`` cross-module binding is rejected."""
        (tmp_path / "mylib.agl").write_text("def foo() -> int = 1\n", encoding="utf-8")
        result = _run_program(
            "import mylib\nmylib::foo := 3\nprint 1",
            extra_roots=frozenset({tmp_path}),
        )
        assert not result.ok
        assert result.diagnostics
