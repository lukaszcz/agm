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

from agm.agl.ir.builtin_nominals import NO_BUILTIN_DECLARATIONS
from agm.agl.modules.roots import RootSet
from agm.agl.parser import parse_program
from agm.agl.pipeline import PipelineDriver, RunResult
from agm.agl.runtime.option import some_value
from agm.agl.semantics.values import BoolValue, IntValue, RecordValue, TextValue, Value
from agm.agl.syntax import BuiltinVarDecl, Call, Expr, VarRef, walk
from agm.agl.syntax.constants import is_constant_expression
from tests._agl_helpers import agent_value, agl_roots, run_inline_command


def _run(source: str) -> RunResult:
    """Run a single-module *source* (no imports) through prepare + run_prepared."""
    rt = PipelineDriver()
    return run_inline_command(rt, source)


def _run_program(
    source: str,
    *,
    extra_roots: frozenset[Path] = frozenset(),
    shell_exec_timeout: float | None = None,
    builtin_host_settings: dict[str, Value] | None = None,
) -> RunResult:
    """Run *source* (with imports) through the program pipeline against the stdlib."""
    roots = agl_roots(*extra_roots)
    rt = PipelineDriver(shell_exec_timeout=shell_exec_timeout)
    return run_inline_command(rt, source, roots=roots, builtin_host_settings=builtin_host_settings)


def _run_with_std_config(
    source: str,
    std_config: str,
    root: Path,
    *,
    builtin_host_settings: dict[str, Value] | None = None,
) -> RunResult:
    """Run against a test ``std/config`` module without ordinary entry declarations."""
    config_path = root / "std" / "config.agl"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(std_config, encoding="utf-8")
    rt = PipelineDriver()
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
    def test_suffix_and_anchored_writes_target_the_engine_setting(self) -> None:
        result = _run_program(
            "import std/config\n"
            "config::strict-json := false\n"
            "/std/config::strict-json := true\n"
            "let b = config::strict-json\n"
            "print b"
        )
        assert result.ok, f"expected success but got: {result.error!r}"
        assert result.bindings["b"] == BoolValue(True)

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

    def test_a_def_declared_above_a_builtin_var_still_reads_it(self, tmp_path: Path) -> None:
        """A ``builtin var`` follows the same relaxed order a root let/var
        already does: a def declared above it may still read it."""
        result = _run_with_std_config(
            "import std/config::*\nlet b = read-strict-json()\nb",
            "def read-strict-json() -> bool = strict-json\n\nbuiltin var strict-json: bool = true",
            tmp_path,
        )
        assert result.ok, f"expected success but got: {result.error!r}"
        assert result.bindings["b"] == BoolValue(True)


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
        declaration_source = 'builtin var default-agent: Agent = AgentCommand("not %{"constant"}")'
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
        result = _run("builtin var strict-json: bool\nprint 1")
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
            "builtin var strict-json: int",
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
        (tmp_path / "mylib.agl").write_text("builtin var strict-json: bool\n", encoding="utf-8")
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
            "std/config::Region::retries := 3\n"
            "let n = std/config::Region::retries\n"
            "n",
            "scope Region\n  builtin var retries: int\nend Region",
            tmp_path,
        )
        assert result.ok, f"expected success but got: {result.error!r}"
        assert result.bindings["n"] == IntValue(3)

    def test_scoped_builtin_var_is_bare_after_use(self, tmp_path: Path) -> None:
        result = _run_with_std_config(
            "import std/config::*\nuse std/config::Region::*\nretries := 4\nlet n = retries\nn",
            "scope Region\n  builtin var retries: int\nend Region",
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
            'scope Region\n  builtin var runner: text = "declared"\nend Region',
            tmp_path,
        )

        assert result.ok, f"expected success but got: {result.error!r}"
        assert result.bindings["value"] == TextValue("updated")

    def test_scoped_same_named_settings_are_independent(self, tmp_path: Path) -> None:
        result = _run_with_std_config(
            "import std/config\n"
            "std/config::First::retries := 3\n"
            "let second = std/config::Second::retries\n"
            "second",
            "scope First\n  builtin var retries: int = 1\nend First\n"
            "\n"
            "scope Second\n  builtin var retries: int = 2\nend Second",
            tmp_path,
        )

        assert result.ok, f"expected success but got: {result.error!r}"
        assert result.bindings["second"] == IntValue(2)

    def test_scoped_engine_named_binding_does_not_apply_engine_effects(
        self, tmp_path: Path
    ) -> None:
        result = _run_with_std_config(
            "import std/config\n"
            'std/config::Region::timeout := "not-a-timeout"\n'
            "let value = std/config::Region::timeout\n"
            "value",
            'scope Region\n  builtin var timeout: text = "1s"\nend Region',
            tmp_path,
        )

        assert result.ok, f"expected success but got: {result.error!r}"
        assert result.bindings["value"] == TextValue("not-a-timeout")

    def test_a_def_declared_above_a_scoped_builtin_var_still_reads_it(self, tmp_path: Path) -> None:
        """The relaxed order also reaches a scoped ``builtin var``."""
        result = _run_with_std_config(
            "import std/config::*\nlet n = Region::read-retries()\nn",
            "scope Region\n"
            "  def read-retries() -> int = retries\n"
            "\n"
            "  builtin var retries: int = 3\n"
            "end Region",
            tmp_path,
        )
        assert result.ok, f"expected success but got: {result.error!r}"
        assert result.bindings["n"] == IntValue(3)

    def test_scoped_declaration_still_confined_to_std_config(self, tmp_path: Path) -> None:
        """A scoped ``builtin var`` outside ``std/config`` is rejected, same as a root one."""
        (tmp_path / "mylib.agl").write_text(
            "scope Region\n  builtin var retries: int\nend Region\n", encoding="utf-8"
        )
        result = _run_program(
            "import mylib\nprint 1",
            extra_roots=frozenset({tmp_path}),
        )
        assert not result.ok
        assert result.diagnostics

    def test_scoped_builtin_var_named_like_a_nested_scope_is_an_error(self, tmp_path: Path) -> None:
        """A scoped ``builtin var`` shares the same-path collision rule as a
        ``let``/``var``/``def``/type: it may not repeat a nested scope's name."""
        result = _run_with_std_config(
            "import std/config\n1",
            "scope Region\n"
            "\n"
            "  scope Sub\n"
            "    builtin var y: int\n"
            "  end Sub\n"
            "\n"
            "  builtin var Sub: int\n"
            "end Region\n",
            tmp_path,
        )
        assert not result.ok
        assert result.diagnostics

    def test_root_and_scoped_bindings_of_one_name_are_independent(self, tmp_path: Path) -> None:
        """A root binding and a same-named scoped one are separate registers."""
        result = _run_with_std_config(
            "import std/config\n"
            "std/config::strict-json := true\n"
            "std/config::Region::strict-json := false\n"
            "let root-setting = std/config::strict-json\n"
            "let scoped = std/config::Region::strict-json\n"
            "()",
            "builtin var strict-json: bool\n"
            "\n"
            "scope Region\n"
            "  builtin var strict-json: bool = false\n"
            "end Region",
            tmp_path,
        )
        assert result.ok, f"expected success but got: {result.error!r}"
        assert result.bindings["root-setting"] == BoolValue(True)
        assert result.bindings["scoped"] == BoolValue(False)


# ---------------------------------------------------------------------------
# Qualified access via std/config
# ---------------------------------------------------------------------------


class TestStdConfigQualified:
    def test_qualified_read_sees_a_bare_write_to_the_same_setting(self) -> None:
        """The bare name a wildcard import brings in addresses the same register."""
        result = _run_program(
            "import std/config::*\nstrict-json := true\nlet b = std/config::strict-json\nprint b"
        )
        assert result.ok, f"expected success but got: {result.error!r}"
        assert result.bindings["b"] == BoolValue(True)

    def test_invalid_timeout_raises_catchable_type_error(self) -> None:
        """Assigning an unparseable ``timeout`` raises a catchable built-in exception.

        Regression test: the live engine-setting effect built this exception
        under an undeclared name (``"ValueError"``, never a real AgL built-in
        exception), which went unnoticed because nothing validated the name.
        The declared ``TypeError`` is what must be raised.
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
        assert set(bound.fields) == {"value"}
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
        assert bound.fields == {}

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
        assert set(bound.fields) == {"value"}
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
        assert bound.fields == {}

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
            builtin_host_settings={
                "timeout": some_value(TextValue("45s"), nominals=NO_BUILTIN_DECLARATIONS)
            },
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


# ---------------------------------------------------------------------------
# Effect-at-binding: a write takes effect from that program point forward
# ---------------------------------------------------------------------------


_FENCED_EXEC = "let r: int = exec \"printf '```json\\n5\\n```'\"\n"


class TestEffectAtBinding:
    def test_parse_after_write_uses_new_mode(self) -> None:
        """A ``strict-json := true`` before a typed ``exec`` rejects a fenced reply."""
        source = "import std/config::*\nstd/config::strict-json := true\n" + _FENCED_EXEC
        result = _run_program(source)
        assert not result.ok
        assert result.error is not None

    def test_parse_before_write_uses_initial_mode(self) -> None:
        """A typed ``exec`` before the write still parses leniently."""
        source = "import std/config::*\n" + _FENCED_EXEC + "std/config::strict-json := true\n"
        result = _run_program(source)
        assert result.ok, f"expected success but got: {result.error!r}"
        assert result.bindings["r"] == IntValue(5)


# ---------------------------------------------------------------------------
# Bare (unqualified) assignment through an import tail
# ---------------------------------------------------------------------------


class TestBareCrossModuleAssign:
    def test_bare_write_reflected_by_bare_read(self) -> None:
        """An import makes::* a setting assignable and readable without a qualifier."""
        result = _run_program(
            "import std/config::*\nstrict-json := true\nlet b = strict-json\nprint b"
        )
        assert result.ok, f"expected success but got: {result.error!r}"
        assert result.bindings["b"] == BoolValue(True)

    def test_bare_write_reflected_by_qualified_read(self) -> None:
        """Bare and qualified targets denote the same binding."""
        result = _run_program(
            'import std/config::*\ntimeout := Some("4s")\nlet t = std/config::timeout\nprint t'
        )
        assert result.ok, f"expected success but got: {result.error!r}"
        bound = result.bindings["t"]
        assert isinstance(bound, RecordValue)
        assert bound.fields["value"] == TextValue("4s")

    def test_bare_write_takes_effect_on_the_engine(self) -> None:
        """A bare write changes engine behavior, not just the readable value."""
        result = _run_program('import std/config::*\ntimeout := Some("not-a-timeout")\nprint 1')
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "TypeError"

    def test_bare_write_to_qualified_only_import_rejected(self, tmp_path: Path) -> None:
        """A qualified-only import does not expose the name for bare assignment."""
        result = _run_program("import std/config as cfg\nstrict-json := true\nprint 1")
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
