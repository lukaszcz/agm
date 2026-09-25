"""Tests for `extern def` typechecking.

Covers everything from a resolved AST to a ``CheckedModule`` for `extern def`:
- signature checking reuses the ordinary/``builtin def`` path (kinds, zones,
  defaults, type params, no body to check).
- extern-specific header checks: Python-identifier/keyword name rule,
  builtin-name collision guard.
- AgL function parameters in extern signatures are accepted for callback
  crossing; Python callables returned toward AgL remain runtime boundary
  errors, while `Agent` enum values cross as ordinary data.
- calls to externs type exactly like calls to ordinary declared functions.

NO contract compilation, lowering, or runtime behavior is exercised here —
externs are not executable yet.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.diagnostics import Diagnostic
from agm.agl.parser import parse_program
from agm.agl.scope.symbols import AglScopeError
from agm.agl.semantics.types import CastSpec
from agm.agl.syntax.nodes import Block, FuncDef
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck import (
    AglTypeError,
    CheckedModule,
    FunctionSignature,
    FunctionType,
    IntType,
)
from agm.agl.typecheck.env import (
    OutputContractSpec,
    PartialCallSpec,
)
from tests.agl.module_graph import (
    resolve_and_check_inline_program_ast,
    resolve_inline_entry,
    resolve_inline_program_ast,
)

_PATH = Path("/virtual/extern_typecheck.agl")

_CAPS = HostCapabilities(
    supports_shell_exec=True,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset({"json", "record", "enum", "array", "dict", "int", "decimal", "bool"}),
    },
)

# ``check_extern``/``resolve_program_ast`` build a hand-crafted single-module
# graph that never imports ``std/prelude`` (see
# ``tests.agl.module_graph.resolve_and_check_program_ast``), so a bare
# ``ask(...)`` call in one of these sources needs its own reachable
# declaration: a bare built-in call is classified only once it resolves to a
# ``builtin def``, exactly like any other reference.
_ASK_BUILTIN_SOURCE = (
    "builtin\n"
    "enum ParsePolicy =\n"
    "  | Abort\n"
    "  | Retry(n: int)\n"
    "\n"
    "builtin\n"
    "enum Agent =\n"
    "  | AgentCommand(command: text)\n"
    "  | AgentClaude(model: text, thinking: text)\n"
    "  | AgentCodex(model: text, thinking: text)\n"
    "  | AgentPi(provider: text, model: text, thinking: text)\n"
    "\n"
    "builtin def ask[T](\n"
    "  prompt: text,\n"
    '  agent: Agent = AgentCommand(command = "x"),\n'
    '  format: text = "",\n'
    "  strict-json: bool = false,\n"
    "  on-parse-error: ParsePolicy = ParsePolicy::Abort,\n"
    "  sandbox: AgentSandbox = Disabled,\n"
    ") -> T\n"
)


def check_extern(source: str, capabilities: HostCapabilities | None = None) -> CheckedModule:
    """Parse + resolve (file-backed) + check *source*, returning the CheckedModule.

    Uses ``resolve_and_check_program_ast``'s hand-built single-module graph
    rather than a real loaded one: *_PATH* is a virtual, non-existent file,
    deliberately exercising the scope/typecheck layers' own rule that an
    extern needs a file-backed origin -- independent of whether a companion
    ``.py`` file actually exists on disk, which only the module loader (a
    separate, already graph-tested concern; see ``test_agl_extern_syntax.py``)
    enforces. Building a real loaded graph here would hit that loader check
    first and mask the one this class means to test.
    """
    return resolve_and_check_inline_program_ast(
        parse_program(source), capabilities or _CAPS, origin_path=_PATH
    )


def reject_extern(source: str, capabilities: HostCapabilities | None = None) -> AglTypeError:
    """Assert that *source* is rejected with an ``AglTypeError``."""
    with pytest.raises(AglTypeError) as exc_info:
        check_extern(source, capabilities)
    return exc_info.value


# ---------------------------------------------------------------------------
# Signature parity with def / builtin def
# ---------------------------------------------------------------------------


class TestExternSignatureParity:
    def test_positional_zoned_and_named_only_params_accepted(self) -> None:
        cp = check_extern(
            "extern def f(@arg-pos a: int, b: int, @arg-named c: int = 1) -> int\nf(1, 2)"
        )
        sig = _extern_signature(cp, "f")
        assert sig.result == IntType()
        assert len(sig.params) == 3

    def test_named_only_zone_accepted(self) -> None:
        check_extern("extern def f(a: int, @arg-named b: int) -> int\nf(1, b = 2)")

    def test_default_expression_typechecks_against_param_type(self) -> None:
        check_extern('extern def f(a: int, b: text = "x") -> int\nf(1)')

    def test_default_expression_type_mismatch_rejected(self) -> None:
        reject_extern('extern def f(a: int, b: int = "x") -> int\nf(1)')

    def test_required_after_defaulted_rejected(self) -> None:
        err = reject_extern("extern def f(a: int = 1, b: int) -> int\n0")
        assert "default" in str(err).lower()

    def test_generic_extern_signature_accepted(self) -> None:
        cp = check_extern("extern def reverse[T](xs: array[T]) -> array[T]\nreverse([1, 2])")
        sig = _extern_signature(cp, "reverse")
        assert sig.type_params == ("T",)

    def test_extern_has_no_body_to_check(self) -> None:
        # A malformed "body" cannot exist syntactically (grammar forbids it —
        # see tests/test_agl_extern_syntax.py) — this just confirms checking
        # an extern def does not choke looking for one.
        cp = check_extern("extern def f(x: int) -> int\n0")
        assert _extern_signature(cp, "f").result == IntType()


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
# Callable signatures — type variables permitted
# ---------------------------------------------------------------------------


class TestExternCallableSignatures:
    def test_function_typed_param_accepted(self) -> None:
        check_extern("extern def f(cb: (int) -> int) -> int\n0")

    def test_agent_enum_param_and_return_are_accepted(self) -> None:
        check_extern(_ASK_BUILTIN_SOURCE + "extern def identity(a: Agent) -> Agent\n0")

    def test_function_typed_return_is_checked_at_the_runtime_boundary(self) -> None:
        check_extern("extern def f(x: int) -> (int) -> int\n0")

    def test_extern_returned_function_can_be_indirectly_invoked(self) -> None:
        check_extern("extern def choose() -> (int) -> int\nlet maker = choose\nmaker()(1)")

    def test_function_type_nested_in_array_is_accepted(self) -> None:
        check_extern("extern def f(cbs: array[(int) -> int]) -> int\n0")

    def test_function_type_nested_in_dict_is_accepted(self) -> None:
        check_extern("extern def f(cbs: dict[text, (int) -> int]) -> int\n0")

    def test_function_type_nested_in_record_field_is_accepted(self) -> None:
        source = "record Box\n  cb: (int) -> int\nextern def f(b: Box) -> int\n0"
        check_extern(source)

    def test_function_type_nested_in_generic_record_instantiation_is_accepted(self) -> None:
        source = "record Box[T]\n  value: int\nextern def f(b: Box[(int) -> int]) -> int\n0"
        check_extern(source)

    def test_function_type_nested_in_exception_field_is_accepted(self) -> None:
        source = (
            "exception BadExc extends Exception\n  cb: (int) -> int\n"
            "extern def f(e: BadExc) -> int\n0"
        )
        check_extern(source)

    def test_recursive_exception_signature_is_walked_once(self) -> None:
        check_extern(
            "exception Loop extends Exception\n  children: array[Loop]\n"
            "extern def f(value: Loop) -> int\n0"
        )

    def test_growing_generic_signature_is_rejected_before_the_type_walk(self) -> None:
        source = (
            "record Pair[A, B]\n"
            "  first: A\n"
            "  second: B\n"
            "enum Perfect[T]\n"
            "  | Single(value: T)\n"
            "  | Succ(next: Perfect[Pair[T, T]])\n"
            "extern def f(value: Perfect[int]) -> int\n"
            "0"
        )
        err = reject_extern(source)
        assert "no finite json schema" in str(err).lower()

    def test_type_variables_permitted(self) -> None:
        check_extern("extern def id[T](x: T) -> T\nid(1)")

    def test_type_variable_nested_in_array_permitted(self) -> None:
        check_extern("extern def first[T](xs: array[T]) -> T\nfirst([1, 2])")

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
        check_extern(
            "extern def f(a: int, @arg-named b: int = 2) -> int\nlet _ = f(1)\nlet _ = f(1, b = 3)"
        )

    def test_zoned_positional_only_and_named_only_call(self) -> None:
        check_extern("extern def f(@arg-pos a: int, @arg-named b: int) -> int\nf(1, b = 2)")

    def test_generic_inference_multiple_instantiations(self) -> None:
        source = (
            "extern def reverse[T](xs: array[T]) -> array[T]\n"
            "let a = reverse([1, 2])\n"
            'let b = reverse(["x", "y"])\n'
            "a"
        )
        check_extern(source)

    def test_call_inside_generic_function_at_rigid_type_var(self) -> None:
        source = (
            "extern def reverse[T](xs: array[T]) -> array[T]\n"
            "def wrapper[U](xs: array[U]) -> array[U] = reverse(xs)\n"
            "wrapper([1, 2])"
        )
        check_extern(source)

    def test_extern_used_as_value_has_function_type(self) -> None:
        source = "extern def f(x: int) -> int\nlet g: (int) -> int = f\ng(1)"
        cp = check_extern(source)
        assert _extern_signature(cp, "f").result == IntType()
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
            "extern def reverse[T](xs: array[T]) -> array[T]\n"
            "let g: (array[int]) -> array[int] = reverse\ng([1, 2])"
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


def _extern_signature(cp: CheckedModule, name: str) -> FunctionSignature:
    """Return a root extern's checked signature from the module environment."""
    definition = next(
        item
        for item in cp.resolved.program.body.items
        if isinstance(item, FuncDef) and item.is_extern and item.name == name
    )
    signature = cp.type_env.get_function_signature_by_node_id(definition.node_id)
    assert signature is not None
    return signature


# ---------------------------------------------------------------------------
# Extern provenance finalization
# ---------------------------------------------------------------------------


class TestExternProvenanceFinalization:
    def test_type_directed_extern_returned_from_a_function_keeps_its_contract(self) -> None:
        """A type-directed extern's target contract survives ``return`` through a def.

        ``get``'s declared result type instantiates ``query``'s ``T`` at ``int``
        when it is returned; that provenance is what the finalized target
        contract reports, even though the annotation lives on ``get``, not on
        the ``return`` statement itself.
        """
        cp = check_extern(
            "extern def query[T](question: text) -> T\n"
            "def get() -> (text) -> int\n"
            "  return query::[int]\n"
            "let apply = get()\n"
            'apply("q")'
        )
        (specs,) = cp.target_contract_specs.values()
        (spec,) = specs
        assert (spec.target_type, spec.codec_name) == (IntType(), "json")

    def test_failed_region_rolls_back_extern_provenance_before_checker_reuse(self) -> None:
        # Direct resolve_program_ast bypass, not resolve_and_check_entry: see
        # check_extern's docstring -- this builds a _Checker by hand from a
        # virtual origin_path with no real companion file, which is exactly
        # what a real module graph's loader would reject before this
        # internal-state assertion ever ran.
        from agm.agl.syntax.nodes import Call
        from agm.agl.typecheck.builder import _TypeBuilder
        from agm.agl.typecheck.checker import _Checker
        from agm.agl.typecheck.env import TypeEnvironment

        resolved = resolve_inline_program_ast(
            parse_program(
                _ASK_BUILTIN_SOURCE + "extern def id[T](value: T) -> T\n"
                "extern def same[T](left: T, right: T) -> T\n"
                "def choose[T](first: T, second: T) -> T = first\n"
                'choose(id(same(?, fn(value: int) -> int => value)), ask("answer"))'
            ),
            origin_path=_PATH,
        )
        env = TypeEnvironment()
        _TypeBuilder(env, attributes=resolved.attributes).collect(resolved.program)
        checker = _Checker(env, resolved, _CAPS)
        definitions = [item for item in resolved.program.body.items if isinstance(item, FuncDef)]
        for definition in definitions:
            checker._preregister_funcdef(definition)
        main = resolved.program.body.items[-1]
        assert isinstance(main, FuncDef)
        assert isinstance(main.body, Block)
        failed_call = main.body.items[-1]
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
            checker._warnings,
        ) == before
        checker._warnings.clear()

        successful_call = failed_call.args[0]
        assert isinstance(successful_call, Call)
        assert isinstance(checker._check_expr(successful_call, expected=None), FunctionType)

    def test_region_finalization_zonks_only_added_extern_provenance(self) -> None:
        from agm.agl.modules.ids import ENTRY_ID
        from agm.agl.typecheck.checker import _Checker, _ExternTarget, _InferenceRegion
        from agm.agl.typecheck.env import TypeEnvironment
        from agm.agl.typecheck.inference import InferenceEngine

        checker = _Checker(TypeEnvironment(), resolve_inline_entry("()"), _CAPS)
        target = _ExternTarget("id", IntType(), 1, ENTRY_ID)
        checker._extern_expr_targets[10] = (target,)
        checker._extern_binding_targets[11] = (target,)
        checker._return_extern_targets_stack.append([target])
        region = _InferenceRegion(
            engine=InferenceEngine(),
            node_types={},
            function_call_param_types={},
            finalization_obligations=[],
            added_side_table_keys={
                "extern_expr_targets": {10, 12},
                "extern_binding_targets": {11, 13},
            },
            warnings_start=0,
            return_target_lengths=(0,),
        )

        checker._finalize_extern_provenance(region)

        assert checker._extern_expr_targets[10] == (target,)
        assert checker._extern_binding_targets[11] == (target,)
        assert checker._return_extern_targets_stack == [[target]]

    def test_finalization_defensively_rejects_an_unresolved_extern_obligation(self) -> None:
        from agm.agl.typecheck.checker import PendingExternCallObligation, _Checker
        from agm.agl.typecheck.env import TypeEnvironment
        from agm.agl.typecheck.inference import InferenceEngine

        checker = _Checker(TypeEnvironment(), resolve_inline_entry("()"), _CAPS)
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


class TestExternDefensiveGuards:
    """Cover the extern return-type guard the grammar makes unreachable.

    ``extern_func_def`` always requires a return type, so an ``extern def``
    without one is a syntax error long before the checker sees it. The guard
    still stands because ``check_program`` accepts a caller-supplied AST, and
    this is the only way to hand it one. It mirrors
    ``TestDefensiveGuards.test_builtin_funcdef_without_return_type_rejected_defensively``
    in ``test_agl_typecheck.py``, which covers the ``builtin def`` half of the
    same rule.
    """

    def test_extern_funcdef_without_return_type_rejected_defensively(self) -> None:
        program = parse_program("extern def f() -> int\nf()")
        declaration = program.body.items[0]
        assert isinstance(declaration, FuncDef)
        stripped = replace(declaration, return_type=None)
        body = replace(program.body, items=(stripped, *program.body.items[1:]))

        with pytest.raises(AglTypeError, match="return type"):
            resolve_and_check_inline_program_ast(
                replace(program, body=body), _CAPS, origin_path=_PATH
            )
