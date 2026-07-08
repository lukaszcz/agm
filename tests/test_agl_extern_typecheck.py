"""Tests for `extern def` typechecking and call-site recording.

Covers everything from a resolved AST to a ``CheckedProgram``/``CheckedModuleGraph``
for `extern def`:
- signature checking reuses the ordinary/``builtin def`` path (kinds, zones,
  defaults, type params, no body to check).
- extern-specific header checks: Python-identifier/keyword name rule,
  builtin-name collision guard.
- the function/agent type ban anywhere in an extern's signature (type
  variables permitted).
- calls to externs type exactly like calls to ordinary declared functions.
- direct extern call sites (own-module and imported) are recorded in
  ``call_sites`` like ``ask``/``exec`` call sites.

NO contract compilation, lowering, or runtime behavior is exercised here —
externs are not executable yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.parser import parse_program
from agm.agl.scope import resolve
from agm.agl.scope.graph import resolve_graph
from agm.agl.scope.symbols import AglScopeError, ResolvedProgram, ScopeNode
from agm.agl.syntax.nodes import Block, FuncDef, Program
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck import (
    AglTypeError,
    CheckedModuleGraph,
    CheckedProgram,
    FunctionType,
    IntType,
    check,
    check_graph,
)
from tests.agl.ir_harness import make_graph_from_files, write_companion_file

_PATH = Path("/virtual/extern_typecheck.agl")

_CAPS = HostCapabilities(
    agent_names=frozenset(),
    has_default_agent=True,
    supports_shell_exec=True,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset(
            {"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}
        ),
    },
)

_TYPE_REJECTIONS_DIR = Path(__file__).resolve().parent / "agl" / "rejections" / "type"


def check_extern(source: str, capabilities: HostCapabilities | None = None) -> CheckedProgram:
    """Parse + resolve (file-backed) + check *source*, returning the CheckedProgram."""
    resolved = resolve(parse_program(source), origin_path=_PATH)
    return check(resolved, capabilities or _CAPS)


def reject_extern(source: str, capabilities: HostCapabilities | None = None) -> AglTypeError:
    """Assert that *source* is rejected with an ``AglTypeError``."""
    with pytest.raises(AglTypeError) as exc_info:
        check_extern(source, capabilities)
    return exc_info.value


def check_extern_graph(tmp_path: Path, modules: dict[str, str]) -> CheckedModuleGraph:
    """Build and typecheck a multi-module graph; returns the ``CheckedModuleGraph``."""
    graph = make_graph_from_files(tmp_path, modules)
    resolved = resolve_graph(graph)
    return check_graph(resolved, _CAPS)


# ---------------------------------------------------------------------------
# Signature parity with def / builtin def
# ---------------------------------------------------------------------------


class TestExternSignatureParity:
    def test_positional_zoned_and_named_only_params_accepted(self) -> None:
        cp = check_extern(
            "extern def f(a: int, /, b: int, @named, c: int = 1) -> int\nf(1, 2)"
        )
        sig = cp.function_signatures["f"]
        assert sig.result == IntType()
        assert len(sig.params) == 3

    def test_star_named_only_zone_accepted(self) -> None:
        check_extern("extern def f(a: int, *, b: int) -> int\nf(1, b = 2)")

    def test_default_expression_typechecks_against_param_type(self) -> None:
        check_extern('extern def f(a: int, b: text = "x") -> int\nf(1)')

    def test_default_expression_type_mismatch_rejected(self) -> None:
        reject_extern('extern def f(a: int, b: int = "x") -> int\nf(1)')

    def test_required_after_defaulted_rejected(self) -> None:
        err = reject_extern("extern def f(a: int = 1, b: int) -> int\n0")
        assert "default" in str(err).lower()

    def test_generic_extern_signature_accepted(self) -> None:
        cp = check_extern(
            "extern def reverse[T](xs: list[T]) -> list[T]\nreverse([1, 2])"
        )
        sig = cp.function_signatures["reverse"]
        assert sig.type_params == ("T",)

    def test_extern_has_no_body_to_check(self) -> None:
        # A malformed "body" cannot exist syntactically (grammar forbids it —
        # see tests/test_agl_extern_syntax.py) — this just confirms checking
        # an extern def does not choke looking for one.
        cp = check_extern("extern def f(x: int) -> int\n0")
        assert cp.function_signatures["f"].result == IntType()


# ---------------------------------------------------------------------------
# Name rules
# ---------------------------------------------------------------------------


class TestExternNameRules:
    @pytest.mark.parametrize("name", ["do-it!", "valid?", "my-func"])
    def test_non_python_identifier_rejected(self, name: str) -> None:
        err = reject_extern(f"extern def {name}(x: int) -> int\n0")
        assert "identifier" in str(err).lower()

    @pytest.mark.parametrize("name", ["class", "import", "lambda", "global"])
    def test_python_keyword_name_rejected(self, name: str) -> None:
        err = reject_extern(f"extern def {name}(x: int) -> int\n0")
        assert "keyword" in str(err).lower()

    def test_python_soft_keyword_name_accepted(self) -> None:
        check_extern("extern def match(x: int) -> int\nmatch(1)")

    def test_dunder_style_name_accepted(self) -> None:
        check_extern("extern def __init__(x: int) -> int\n__init__(1)")


# ---------------------------------------------------------------------------
# Builtin-name collision guard
# ---------------------------------------------------------------------------


class TestExternCollisionGuard:
    def test_extern_named_like_builtin_call_rejected(self) -> None:
        # A builtin *call* name (print/ask/exec/...) is reserved at the scope
        # layer, exactly like an ordinary def — extern gets no exemption
        # (unlike ``builtin def``, which names one on purpose).
        with pytest.raises(AglScopeError) as exc_info:
            check_extern('extern def print(x: text) -> text\n0')
        assert "built-in" in str(exc_info.value).lower()

    def test_extern_named_like_builtin_type_rejected(self) -> None:
        # A builtin *type* name (int/text/.../agent) is only rejected by the
        # typecheck-layer guard shared with ordinary defs.
        err = reject_extern("extern def int(x: int) -> int\n0")
        assert "built-in" in str(err).lower()


# ---------------------------------------------------------------------------
# Function/agent type ban — type variables permitted
# ---------------------------------------------------------------------------


class TestExternFunctionAgentTypeBan:
    def test_function_typed_param_rejected(self) -> None:
        err = reject_extern("extern def f(cb: (int) -> int) -> int\n0")
        assert "function" in str(err).lower()

    def test_agent_typed_param_rejected(self) -> None:
        err = reject_extern("extern def f(a: agent) -> int\n0")
        assert "agent" in str(err).lower()

    def test_function_typed_return_rejected(self) -> None:
        err = reject_extern("extern def f(x: int) -> (int) -> int\n0")
        assert "function" in str(err).lower()

    def test_agent_typed_return_rejected(self) -> None:
        err = reject_extern("extern def f(x: int) -> agent\n0")
        assert "agent" in str(err).lower()

    def test_function_type_nested_in_list_rejected(self) -> None:
        err = reject_extern("extern def f(cbs: list[(int) -> int]) -> int\n0")
        assert "function" in str(err).lower()

    def test_function_type_nested_in_dict_rejected(self) -> None:
        err = reject_extern("extern def f(cbs: dict[text, (int) -> int]) -> int\n0")
        assert "function" in str(err).lower()

    def test_function_type_nested_in_record_field_rejected(self) -> None:
        source = (
            "record Box\n  cb: (int) -> int\n"
            "extern def f(b: Box) -> int\n0"
        )
        err = reject_extern(source)
        assert "function" in str(err).lower()

    def test_function_type_nested_in_generic_record_instantiation_rejected(self) -> None:
        # `Box`'s own field never mentions `T`; the banned type only rides in
        # via the instantiation's type_args, so the check must inspect those
        # too, not just the record's declared field types.
        source = (
            "record Box[T]\n  value: int\n"
            "extern def f(b: Box[(int) -> int]) -> int\n0"
        )
        err = reject_extern(source)
        assert "function" in str(err).lower()

    def test_agent_type_nested_in_enum_variant_rejected(self) -> None:
        source = (
            "enum Holder\n  | with-agent(a: agent)\n"
            "extern def f(h: Holder) -> int\n0"
        )
        err = reject_extern(source)
        assert "agent" in str(err).lower()

    def test_function_type_nested_in_exception_field_rejected(self) -> None:
        source = (
            "exception BadExc extends Exception\n  cb: (int) -> int\n"
            "extern def f(e: BadExc) -> int\n0"
        )
        err = reject_extern(source)
        assert "function" in str(err).lower()

    def test_type_variables_permitted(self) -> None:
        check_extern("extern def id[T](x: T) -> T\nid(1)")

    def test_type_variable_nested_in_list_permitted(self) -> None:
        check_extern("extern def first[T](xs: list[T]) -> T\nfirst([1, 2])")

    def test_record_and_plain_types_permitted(self) -> None:
        source = (
            "record Box\n  value: int\n"
            "extern def get(b: Box) -> int\nget(Box(value = 1))"
        )
        check_extern(source)

    def test_generic_enum_instantiation_permitted(self) -> None:
        source = (
            "enum Option[T]\n  | none\n  | some(value: T)\n"
            "extern def f(o: Option[int]) -> int\n0"
        )
        check_extern(source)


# ---------------------------------------------------------------------------
# Calls type exactly like ordinary declared-function calls
# ---------------------------------------------------------------------------


class TestExternCallTyping:
    def test_positional_call(self) -> None:
        cp = check_extern("extern def f(x: int) -> int\nlet r = f(1)\nr")
        assert cp.node_types[_last_call_node_id(cp, "f")] == IntType()

    def test_named_and_default_call(self) -> None:
        check_extern(
            "extern def f(a: int, @named, b: int = 2) -> int\nf(1)\nf(1, b = 3)"
        )

    def test_zoned_positional_only_and_named_only_call(self) -> None:
        check_extern("extern def f(a: int, /, *, b: int) -> int\nf(1, b = 2)")

    def test_generic_inference_multiple_instantiations(self) -> None:
        source = (
            "extern def reverse[T](xs: list[T]) -> list[T]\n"
            "let a = reverse([1, 2])\n"
            'let b = reverse(["x", "y"])\n'
            "a"
        )
        check_extern(source)

    def test_call_inside_generic_function_at_rigid_type_var(self) -> None:
        source = (
            "extern def reverse[T](xs: list[T]) -> list[T]\n"
            "def wrapper[U](xs: list[U]) -> list[U] = reverse(xs)\n"
            "wrapper([1, 2])"
        )
        check_extern(source)

    def test_extern_used_as_value_has_function_type(self) -> None:
        source = "extern def f(x: int) -> int\nlet g: (int) -> int = f\ng(1)"
        cp = check_extern(source)
        assert cp.function_signatures["f"].result == IntType()
        from agm.agl.syntax.nodes import LetDecl

        program = cp.resolved.program
        g_decl = next(
            item
            for item in program.body.items
            if isinstance(item, LetDecl) and item.name == "g"
        )
        assert cp.node_types[g_decl.value.node_id] == FunctionType(
            params=(IntType(),), result=IntType()
        )

    def test_generic_extern_used_as_value(self) -> None:
        source = (
            "extern def reverse[T](xs: list[T]) -> list[T]\n"
            "let g: (list[int]) -> list[int] = reverse\ng([1, 2])"
        )
        check_extern(source)

    def test_arity_mismatch_rejected(self) -> None:
        reject_extern("extern def f(x: int) -> int\nf(1, 2)")

    def test_argument_type_mismatch_rejected(self) -> None:
        reject_extern('extern def f(x: int) -> int\nf("a")')


def _last_call_node_id(cp: CheckedProgram, name: str) -> int:
    """Find the ``node_id`` of the ``Call`` to *name* for a node-type lookup."""
    from agm.agl.syntax.nodes import Call, LetDecl, VarRef

    for item in cp.resolved.program.body.items:
        if isinstance(item, LetDecl) and isinstance(item.value, Call):
            callee = item.value.callee
            if isinstance(callee, VarRef) and callee.name == name:
                return item.value.node_id
    raise AssertionError(f"no call to {name!r} found")


# ---------------------------------------------------------------------------
# Call-site recording (dry-run inventory)
# ---------------------------------------------------------------------------


class TestExternCallSiteRecording:
    def test_single_module_extern_call_recorded(self) -> None:
        cp = check_extern("extern def f(x: int) -> int\nf(1)")
        sites = [s for s in cp.call_sites if s.callee == "f"]
        assert len(sites) == 1
        assert sites[0].codec_name == "extern"
        assert sites[0].target_type == IntType()

    def test_multiple_calls_each_recorded(self) -> None:
        cp = check_extern("extern def f(x: int) -> int\nf(1)\nf(2)")
        sites = [s for s in cp.call_sites if s.callee == "f"]
        assert len(sites) == 2

    def test_ask_exec_recording_unaffected_by_extern_presence(self) -> None:
        cp = check_extern('extern def f(x: int) -> int\nask("hi")')
        callees = [s.callee for s in cp.call_sites]
        assert callees == ["ask"]

    def test_indirect_first_class_call_recorded_at_invocation(self) -> None:
        source = "extern def f(x: int) -> int\nlet g: (int) -> int = f\ng(1)"
        cp = check_extern(source)
        sites = [s for s in cp.call_sites if s.callee == "f"]
        assert len(sites) == 1
        assert sites[0].line == 3
        assert sites[0].target_type == IntType()

    def test_partial_extern_call_recorded_at_invocation(self) -> None:
        source = "extern def f(x: int, y: int) -> int\nlet g = f(?, 2)\ng(1)"
        cp = check_extern(source)
        sites = [s for s in cp.call_sites if s.callee == "f"]
        assert len(sites) == 1
        assert sites[0].line == 3
        assert sites[0].target_type == IntType()

    def test_partial_extern_function_value_recorded_at_invocation(self) -> None:
        source = (
            "extern def f(x: int, y: int) -> int\n"
            "let g = f(?, ?)\n"
            "let h = g(?, 2)\n"
            "h(1)"
        )
        cp = check_extern(source)
        sites = [s for s in cp.call_sites if s.callee == "f"]
        assert len(sites) == 1
        assert sites[0].line == 4
        assert sites[0].target_type == IntType()

    def test_graph_mode_imported_extern_call_recorded(self, tmp_path: Path) -> None:
        write_companion_file(tmp_path / "root", "lib.mod", "def f(x):\n    return x\n")
        checked = check_extern_graph(
            tmp_path,
            {
                "entry": "import lib.mod\nlib.mod::f(1)",
                "lib.mod": "extern def f(x: int) -> int",
            },
        )
        entry_module = checked.modules[checked.entry_id]
        sites = [s for s in entry_module.call_sites if s.callee == "f"]
        assert len(sites) == 1
        assert sites[0].codec_name == "extern"
        assert sites[0].target_type == IntType()

    def test_graph_mode_own_module_extern_call_recorded(self, tmp_path: Path) -> None:
        write_companion_file(tmp_path / "root", "lib.mod", "def f(x):\n    return x\n")
        checked = check_extern_graph(
            tmp_path,
            {
                "entry": "import lib.mod\n()",
                "lib.mod": "extern def f(x: int) -> int\ndef g() -> int = f(1)",
            },
        )
        from agm.agl.modules.ids import ModuleId

        lib_mod_id = next(mid for mid in checked.modules if not mid.is_entry)
        assert isinstance(lib_mod_id, ModuleId)
        sites = [s for s in checked.modules[lib_mod_id].call_sites if s.callee == "f"]
        assert len(sites) == 1
        assert sites[0].codec_name == "extern"


# ---------------------------------------------------------------------------
# Rejection fixtures under tests/agl/rejections/type/
# ---------------------------------------------------------------------------


class TestExternRejectionFixtures:
    """Verify the extern-specific rejection fixtures at the typecheck layer.

    ``tests/test_agl_e2e.py`` globs every ``tests/agl/rejections/**/*.agl``
    file and resolves it with no backing file (inline mode), so an
    ``extern def`` in one of those fixtures is always caught by the
    file-backing placement rule before the typecheck-layer rule the fixture
    is meant to demonstrate; that generic run only asserts a rejection
    occurs, without pinning down which rule fired.  These tests give the
    fixtures a real file-backed origin so the intended typecheck-layer
    failure is actually exercised.
    """

    def _read(self, name: str) -> str:
        return (_TYPE_REJECTIONS_DIR / f"{name}.agl").read_text(encoding="utf-8")

    def test_bad_python_name_fixture(self) -> None:
        err = reject_extern(self._read("extern_bad_python_name"))
        assert "identifier" in str(err).lower()

    def test_function_typed_param_fixture(self) -> None:
        err = reject_extern(self._read("extern_function_param"))
        assert "function" in str(err).lower()


# ---------------------------------------------------------------------------
# Defensive guard unreachable from the parser
# ---------------------------------------------------------------------------


class TestExternDefensiveGuards:
    """Cover the extern-specific defensive guard unreachable from the parser.

    The grammar always requires a return type for ``extern def`` (mirroring
    ``builtin def`` — see ``extern_func_def`` in the grammar), so the
    checker's defensive check can only be exercised by constructing the AST
    directly, bypassing the parser, mirroring
    ``TestDefensiveGuards.test_builtin_funcdef_without_return_type_rejected_defensively``
    in ``test_agl_typecheck.py``.
    """

    def test_extern_funcdef_without_return_type_rejected_defensively(self) -> None:
        sp = SourceSpan(
            start_line=1, start_col=1, end_line=1, end_col=2,
            start_offset=0, end_offset=1,
        )
        fd = FuncDef(
            name="f",
            params=(),
            return_type=None,
            body=None,
            span=sp,
            node_id=1,
            is_extern=True,
        )
        block = Block(items=(fd,), span=sp, node_id=2)
        prog = Program(body=block, span=sp, node_id=3)
        resolved = ResolvedProgram(
            program=prog,
            resolution={},
            builtin_calls={},
            root_scope=ScopeNode(node_id=prog.node_id),
            declared_functions={"f": fd},
        )
        with pytest.raises(AglTypeError, match="must declare a return type"):
            check(resolved, _CAPS)
