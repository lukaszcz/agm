"""Tests for `extern def` typechecking and call-site recording.

Covers everything from a resolved AST to a ``CheckedModule``/``CheckedProgram``
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
from typing import cast

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.diagnostics import Diagnostic
from agm.agl.parser import parse_program
from agm.agl.scope import resolve_module
from agm.agl.scope.program import resolve_program
from agm.agl.scope.symbols import AglScopeError, ModuleResolution, ScopeNode
from agm.agl.semantics.types import CastSpec
from agm.agl.syntax.nodes import Block, FuncDef, Program
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck import (
    AglTypeError,
    CheckedModule,
    CheckedProgram,
    FunctionType,
    IntType,
    TextType,
    check_module,
    check_program,
)
from agm.agl.typecheck.env import CallSiteRecord, OutputContractSpec, PartialCallSpec
from tests.agl.ir_harness import make_graph_from_files, write_companion_file

_PATH = Path("/virtual/extern_typecheck.agl")

_CAPS = HostCapabilities(
    agent_names=frozenset(),
    has_default_agent=True,
    supports_shell_exec=True,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset({"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}),
    },
)

_TYPE_REJECTIONS_DIR = Path(__file__).resolve().parent / "agl" / "rejections" / "type"


def check_extern(source: str, capabilities: HostCapabilities | None = None) -> CheckedModule:
    """Parse + resolve (file-backed) + check *source*, returning the CheckedModule."""
    resolved = resolve_module(parse_program(source), origin_path=_PATH)
    return check_module(resolved, capabilities or _CAPS)


def reject_extern(source: str, capabilities: HostCapabilities | None = None) -> AglTypeError:
    """Assert that *source* is rejected with an ``AglTypeError``."""
    with pytest.raises(AglTypeError) as exc_info:
        check_extern(source, capabilities)
    return exc_info.value


def check_extern_graph(tmp_path: Path, modules: dict[str, str]) -> CheckedProgram:
    """Build and typecheck a multi-module graph; returns the ``CheckedProgram``."""
    graph = make_graph_from_files(tmp_path, modules)
    resolved = resolve_program(graph)
    return check_program(resolved, _CAPS)


# ---------------------------------------------------------------------------
# Signature parity with def / builtin def
# ---------------------------------------------------------------------------


class TestExternSignatureParity:
    def test_positional_zoned_and_named_only_params_accepted(self) -> None:
        cp = check_extern("extern def f(a: int, /, b: int, @named, c: int = 1) -> int\nf(1, 2)")
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
        cp = check_extern("extern def reverse[T](xs: list[T]) -> list[T]\nreverse([1, 2])")
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
            check_extern("extern def print(x: text) -> text\n0")
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
        source = "record Box\n  cb: (int) -> int\nextern def f(b: Box) -> int\n0"
        err = reject_extern(source)
        assert "function" in str(err).lower()

    def test_function_type_nested_in_generic_record_instantiation_rejected(self) -> None:
        # `Box`'s own field never mentions `T`; the banned type only rides in
        # via the instantiation's type_args, so the check must inspect those
        # too, not just the record's declared field types.
        source = "record Box[T]\n  value: int\nextern def f(b: Box[(int) -> int]) -> int\n0"
        err = reject_extern(source)
        assert "function" in str(err).lower()

    def test_agent_type_nested_in_enum_variant_rejected(self) -> None:
        source = "enum Holder\n  | with-agent(a: agent)\nextern def f(h: Holder) -> int\n0"
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
        source = "record Box\n  value: int\nextern def get(b: Box) -> int\nget(Box(value = 1))"
        check_extern(source)

    def test_generic_enum_instantiation_permitted(self) -> None:
        source = (
            "enum Option[T]\n  | none\n  | some(value: T)\nextern def f(o: Option[int]) -> int\n0"
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
        check_extern("extern def f(a: int, @named, b: int = 2) -> int\nf(1)\nf(1, b = 3)")

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
            item for item in program.body.items if isinstance(item, LetDecl) and item.name == "g"
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


def _last_call_node_id(cp: CheckedModule, name: str) -> int:
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

    def test_generic_direct_calls_publish_independently_concrete_metadata(self) -> None:
        cp = check_extern(
            'extern def id[T](value: T) -> T\nlet number = id(1)\nlet text = id("value")\ntext'
        )
        sites = [site for site in cp.call_sites if site.callee == "id"]
        assert [site.target_type for site in sites] == [IntType(), TextType()]
        assert cp.argument_bindings.function_param_types[_last_call_node_id(cp, "id")] == (
            IntType(),
        )

    def test_generic_extern_and_builtin_inventory_preserves_source_order(self) -> None:
        cp = check_extern(
            "extern def id[T](value: T) -> T\n"
            "def choose[T](first: T, second: T) -> T = first\n"
            'let value: int = choose(ask("answer"), id(1))\n'
            "value"
        )
        assert [site.callee for site in cp.call_sites] == ["ask", "id"]
        assert [site.target_type for site in cp.call_sites] == [IntType(), IntType()]

    def test_failed_region_rolls_back_extern_inventory_before_checker_reuse(self) -> None:
        from agm.agl.syntax.nodes import Call
        from agm.agl.typecheck.builder import _TypeBuilder
        from agm.agl.typecheck.checker import _Checker
        from agm.agl.typecheck.env import TypeEnvironment

        resolved = resolve_module(
            parse_program(
                "extern def id[T](value: T) -> T\n"
                "extern def same[T](left: T, right: T) -> T\n"
                "def choose[T](first: T, second: T) -> T = first\n"
                'choose(id(same(?, fn(value: int) -> int => value)), ask("answer"))'
            ),
            origin_path=_PATH,
        )
        env = TypeEnvironment()
        _TypeBuilder(env).collect(resolved.program)
        checker = _Checker(env, resolved, _CAPS)
        definitions = [item for item in resolved.program.body.items if isinstance(item, FuncDef)]
        for definition in definitions:
            checker._preregister_funcdef(definition)
        failed_call = resolved.program.body.items[-1]
        assert isinstance(failed_call, Call)
        # A failed boundary must retain prior published data while dropping every
        # provisional side-table delta, including append-only finalization data.
        checker._function_call_bindings[-1] = ()
        checker._constructor_call_bindings[-2] = {}
        checker._constructor_pattern_bindings[-3] = ()
        checker._partial_calls[-4] = cast(PartialCallSpec, object())
        checker._contract_specs[-5] = cast(OutputContractSpec, object())
        checker._cast_specs[-6] = cast(CastSpec, object())
        checker._extern_expr_targets[-7] = ()
        checker._extern_binding_targets[-8] = ()
        checker._call_sites.append(cast(CallSiteRecord, object()))
        checker._warnings.append(cast(Diagnostic, object()))
        before = (
            checker._function_call_bindings.copy(),
            checker._constructor_call_bindings.copy(),
            checker._constructor_pattern_bindings.copy(),
            checker._partial_calls.copy(),
            checker._contract_specs.copy(),
            checker._cast_specs.copy(),
            checker._extern_expr_targets.copy(),
            checker._extern_binding_targets.copy(),
            checker._call_sites.copy(),
            checker._warnings.copy(),
        )
        with pytest.raises(AglTypeError):
            checker._check_expr(failed_call, expected=None)
        assert (
            checker._function_call_bindings,
            checker._constructor_call_bindings,
            checker._constructor_pattern_bindings,
            checker._partial_calls,
            checker._contract_specs,
            checker._cast_specs,
            checker._extern_expr_targets,
            checker._extern_binding_targets,
            checker._call_sites,
            checker._warnings,
        ) == before
        checker._call_sites.clear()
        checker._warnings.clear()

        successful_call = failed_call.args[0]
        assert isinstance(successful_call, Call)
        assert isinstance(checker._check_expr(successful_call, expected=None), FunctionType)
        assert [site.callee for site in checker._call_sites] == ["id"]

    def test_region_finalization_zonks_only_added_extern_provenance(self) -> None:
        from agm.agl.modules.ids import ENTRY_ID
        from agm.agl.typecheck.checker import _Checker, _ExternTarget, _InferenceRegion
        from agm.agl.typecheck.env import TypeEnvironment
        from agm.agl.typecheck.inference import InferenceEngine

        checker = _Checker(TypeEnvironment(), resolve_module(parse_program("()")), _CAPS)
        target = _ExternTarget("id", IntType(), 1, ENTRY_ID)
        checker._extern_expr_targets[10] = (target,)
        checker._extern_binding_targets[11] = (target,)
        checker._return_extern_targets_stack.append([target])
        region = _InferenceRegion(
            InferenceEngine(),
            {},
            {},
            [],
            {
                "extern_expr_targets": {10, 12},
                "extern_binding_targets": {11, 13},
            },
            0,
            0,
            (0,),
        )

        checker._finalize_extern_provenance(region)

        assert checker._extern_expr_targets[10] == (target,)
        assert checker._extern_binding_targets[11] == (target,)
        assert checker._return_extern_targets_stack == [[target]]

    def test_finalization_defensively_rejects_an_unresolved_extern_obligation(self) -> None:
        from agm.agl.typecheck.checker import PendingExternCallObligation, _Checker
        from agm.agl.typecheck.env import TypeEnvironment
        from agm.agl.typecheck.inference import InferenceEngine

        checker = _Checker(TypeEnvironment(), resolve_module(parse_program("()")), _CAPS)
        unresolved = InferenceEngine().fresh("target")
        with pytest.raises(AglTypeError, match="concrete target"):
            checker._finalize_extern_call_obligation(
                PendingExternCallObligation(
                    node_id=1,
                    callee="id",
                    target_type=unresolved,
                    span=SourceSpan(1, 1, 1, 1, 0, 0),
                )
            )

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

    def test_generic_indirect_call_uses_finalized_function_provenance(self) -> None:
        cp = check_extern("extern def id[T](value: T) -> T\nlet apply: (int) -> int = id\napply(1)")
        sites = [site for site in cp.call_sites if site.callee == "id"]
        assert len(sites) == 1
        assert sites[0].line == 3
        assert sites[0].target_type == IntType()

    def test_generic_extern_provenance_is_finalized_through_a_returned_function(self) -> None:
        cp = check_extern(
            "extern def id[T](value: T) -> T\n"
            "def get() -> (int) -> int\n"
            "  return id\n"
            "let apply = get()\n"
            "apply(1)"
        )
        sites = [site for site in cp.call_sites if site.callee == "id"]
        assert len(sites) == 1
        assert sites[0].line == 5
        assert sites[0].target_type == IntType()

    def test_partial_extern_call_recorded_at_invocation(self) -> None:
        source = "extern def f(x: int, y: int) -> int\nlet g = f(?, 2)\ng(1)"
        cp = check_extern(source)
        sites = [s for s in cp.call_sites if s.callee == "f"]
        assert len(sites) == 1
        assert sites[0].line == 3
        assert sites[0].target_type == IntType()

    def test_generic_partial_call_publishes_concrete_parameter_and_result_metadata(self) -> None:
        cp = check_extern(
            "extern def same[T](left: T, right: T) -> T\n"
            "let apply: (int) -> int = same(?, 1)\n"
            "apply(2)"
        )
        partial_node_id = _last_call_node_id(cp, "same")
        assert cp.partial_calls[partial_node_id].argument_holes == (0, None)
        assert cp.argument_bindings.function_param_types[partial_node_id] == (
            IntType(),
            IntType(),
        )
        sites = [site for site in cp.call_sites if site.callee == "same"]
        assert len(sites) == 1
        assert sites[0].line == 3
        assert sites[0].target_type == IntType()

    def test_partial_extern_function_value_recorded_at_invocation(self) -> None:
        source = "extern def f(x: int, y: int) -> int\nlet g = f(?, ?)\nlet h = g(?, 2)\nh(1)"
        cp = check_extern(source)
        sites = [s for s in cp.call_sites if s.callee == "f"]
        assert len(sites) == 1
        assert sites[0].line == 4
        assert sites[0].target_type == IntType()

    def test_function_valued_if_extern_call_recorded_at_invocation(self) -> None:
        source = (
            "extern def inc(x: int) -> int\n"
            "extern def dec(x: int) -> int\n"
            "let h: (int) -> int = if true => inc | else => dec\n"
            "h(1)"
        )
        cp = check_extern(source)
        sites = [s for s in cp.call_sites if s.callee in {"inc", "dec"}]
        assert {s.callee for s in sites} == {"inc", "dec"}
        assert {s.line for s in sites} == {4}
        assert {s.target_type for s in sites} == {IntType()}

    def test_function_valued_if_deduplicates_same_extern_target(self) -> None:
        source = "extern def f(x: int) -> int\nlet h: (int) -> int = if true => f | else => f\nh(1)"
        cp = check_extern(source)
        sites = [s for s in cp.call_sites if s.callee == "f"]
        assert len(sites) == 1
        assert sites[0].line == 3

    def test_function_valued_case_extern_call_recorded_at_invocation(self) -> None:
        source = (
            "enum Choice\n"
            "  | Inc\n"
            "  | Dec\n"
            "extern def inc(x: int) -> int\n"
            "extern def dec(x: int) -> int\n"
            "let c: Choice = Inc\n"
            "let h: (int) -> int = case c of\n"
            "  | Inc() => inc\n"
            "  | Dec() => dec\n"
            "h(1)"
        )
        cp = check_extern(source)
        sites = [s for s in cp.call_sites if s.callee in {"inc", "dec"}]
        assert {s.callee for s in sites} == {"inc", "dec"}
        assert {s.line for s in sites} == {10}
        assert {s.target_type for s in sites} == {IntType()}

    def test_function_valued_block_extern_call_recorded_at_invocation(self) -> None:
        source = (
            "extern def inc(x: int) -> int\n"
            "def choose() -> (int) -> int\n"
            "  let ignored = 1\n"
            "  inc\n"
            "let h = choose()\n"
            "h(1)"
        )
        cp = check_extern(source)
        sites = [s for s in cp.call_sites if s.callee == "inc"]
        assert len(sites) == 1
        assert sites[0].line == 6
        assert sites[0].target_type == IntType()

    def test_function_valued_try_extern_call_recorded_at_invocation(self) -> None:
        source = (
            "extern def inc(x: int) -> int\n"
            "extern def dec(x: int) -> int\n"
            "def choose() -> (int) -> int = try inc catch _ => dec\n"
            "let h = choose()\n"
            "h(1)"
        )
        cp = check_extern(source)
        sites = [s for s in cp.call_sites if s.callee in {"inc", "dec"}]
        assert {s.callee for s in sites} == {"inc", "dec"}
        assert {s.line for s in sites} == {5}
        assert {s.target_type for s in sites} == {IntType()}

    def test_function_valued_return_extern_call_recorded_at_invocation(self) -> None:
        source = (
            "extern def inc(x: int) -> int\n"
            "def choose() -> (int) -> int\n"
            "  return inc\n"
            "let h = choose()\n"
            "h(1)"
        )
        cp = check_extern(source)
        sites = [s for s in cp.call_sites if s.callee == "inc"]
        assert len(sites) == 1
        assert sites[0].line == 5
        assert sites[0].target_type == IntType()

    def test_graph_mode_imported_extern_call_recorded(self, tmp_path: Path) -> None:
        write_companion_file(tmp_path / "root", "lib/mod", "def f(x):\n    return x\n")
        checked = check_extern_graph(
            tmp_path,
            {
                "entry": "import lib.mod\nlib.mod::f(1)",
                "lib/mod": "extern def f(x: int) -> int",
            },
        )
        entry_module = checked.modules[checked.entry_id]
        sites = [s for s in entry_module.call_sites if s.callee == "f"]
        assert len(sites) == 1
        assert sites[0].codec_name == "extern"
        assert sites[0].target_type == IntType()

    def test_graph_mode_generic_extern_call_has_concrete_inventory(self, tmp_path: Path) -> None:
        write_companion_file(tmp_path / "root", "lib/mod", "def id(value):\n    return value\n")
        checked = check_extern_graph(
            tmp_path,
            {
                "entry": "import lib.mod\nlib.mod::id(1)",
                "lib/mod": "extern def id[T](value: T) -> T",
            },
        )
        entry_module = checked.modules[checked.entry_id]
        sites = [site for site in entry_module.call_sites if site.callee == "id"]
        assert len(sites) == 1
        assert sites[0].target_type == IntType()

    def test_graph_mode_same_named_externs_from_different_modules_are_not_collapsed(
        self, tmp_path: Path
    ) -> None:
        write_companion_file(tmp_path / "root", "left", "def f(x):\n    return x\n")
        write_companion_file(tmp_path / "root", "right", "def f(x):\n    return x\n")
        checked = check_extern_graph(
            tmp_path,
            {
                "entry": (
                    "import left as l\n"
                    "import right as r\n"
                    "let h: (int) -> int = if true => l::f | else => r::f\n"
                    "h(1)"
                ),
                "left": "extern def f(x: int) -> int",
                "right": "extern def f(x: int) -> int",
            },
        )
        entry_module = checked.modules[checked.entry_id]
        sites = [s for s in entry_module.call_sites if s.callee == "f"]
        assert len(sites) == 2
        assert {s.line for s in sites} == {4}
        assert {s.target_type for s in sites} == {IntType()}

    def test_graph_mode_own_module_extern_call_recorded(self, tmp_path: Path) -> None:
        write_companion_file(tmp_path / "root", "lib/mod", "def f(x):\n    return x\n")
        checked = check_extern_graph(
            tmp_path,
            {
                "entry": "import lib.mod\n()",
                "lib/mod": "extern def f(x: int) -> int\ndef g() -> int = f(1)",
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
            start_line=1,
            start_col=1,
            end_line=1,
            end_col=2,
            start_offset=0,
            end_offset=1,
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
        resolved = ModuleResolution(
            program=prog,
            resolution={},
            builtin_calls={},
            root_scope=ScopeNode(node_id=prog.node_id),
            declared_functions={"f": fd},
        )
        with pytest.raises(AglTypeError, match="must declare a return type"):
            check_module(resolved, _CAPS)
