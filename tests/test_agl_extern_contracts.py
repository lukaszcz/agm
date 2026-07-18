"""Tests for the extern boundary-contract artifact and its compiler.

Covers ``build_extern_contract`` (``agm.agl.type_schema``), which compiles a
checked extern's ``FunctionSignature`` into a typeless ``ExternContract``
(``agm.agl.ir.contracts``): every scalar/container/nominal boundary
mapping, type-variable sealing, generic-nominal instantiation, and the
static function/agent-type ban. NO runtime walkers, lowering integration, or
registry are exercised here — only the checker-types-to-typeless compiler.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.ir.contracts import (
    BoundaryDict,
    BoundaryEnum,
    BoundaryException,
    BoundaryList,
    BoundaryRecord,
    BoundaryRef,
    BoundaryScalar,
    BoundarySealVar,
    BoundaryUnit,
    BoundaryVariantShape,
    ExternContract,
    ExternParamSchema,
    ScalarKind,
)
from agm.agl.parser import parse_program
from agm.agl.scope import resolve_module
from agm.agl.semantics.type_table import create_seeded_type_table
from agm.agl.semantics.types import (
    AgentType,
    FunctionType,
    IntType,
    TextType,
)
from agm.agl.syntax.nodes import ParamKind
from agm.agl.type_schema import build_extern_contract
from agm.agl.typecheck import ParamSpec, check_module
from agm.agl.typecheck.env import AglTypeError, FunctionSignature

_PATH = Path("/virtual/extern_contracts.agl")

_CAPS = HostCapabilities(
    agent_names=frozenset(),
    has_default_agent=True,
    supports_shell_exec=True,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset({"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}),
    },
)


def build_contract(source: str, fn_name: str = "f") -> ExternContract:
    """Parse + resolve (file-backed) + check *source*, compiling ``fn_name``'s contract."""
    resolved = resolve_module(parse_program(source), origin_path=_PATH)
    cp = check_module(resolved, _CAPS)
    sig = cp.function_signatures[fn_name]
    return build_extern_contract(sig, cp.type_env.type_table)


# ---------------------------------------------------------------------------
# Scalars, unit, and containers
# ---------------------------------------------------------------------------


class TestScalarsAndContainers:
    def test_int(self) -> None:
        contract = build_contract("extern def f(x: int) -> int\n0")
        assert contract.params == (ExternParamSchema(schema=BoundaryScalar(ScalarKind.INT)),)
        assert contract.result == BoundaryScalar(ScalarKind.INT)

    def test_decimal(self) -> None:
        contract = build_contract("extern def f(x: decimal) -> decimal\n0")
        assert contract.result == BoundaryScalar(ScalarKind.DECIMAL)

    def test_bool(self) -> None:
        contract = build_contract("extern def f(x: bool) -> bool\n0")
        assert contract.result == BoundaryScalar(ScalarKind.BOOL)

    def test_text(self) -> None:
        contract = build_contract("extern def f(x: text) -> text\n0")
        assert contract.result == BoundaryScalar(ScalarKind.TEXT)

    def test_json(self) -> None:
        contract = build_contract("extern def f(x: json) -> json\n0")
        assert contract.result == BoundaryScalar(ScalarKind.JSON)

    def test_unit(self) -> None:
        contract = build_contract("extern def f(x: unit) -> unit\n0")
        assert contract.params[0].schema == BoundaryUnit()
        assert contract.result == BoundaryUnit()

    def test_list_nesting(self) -> None:
        contract = build_contract("extern def f(x: list[int]) -> list[int]\n0")
        assert contract.result == BoundaryList(BoundaryScalar(ScalarKind.INT))

    def test_dict_nesting(self) -> None:
        contract = build_contract("extern def f(x: dict[text, int]) -> dict[text, int]\n0")
        assert contract.result == BoundaryDict(BoundaryScalar(ScalarKind.INT))

    def test_list_of_dict_nesting(self) -> None:
        contract = build_contract(
            "extern def f(x: list[dict[text, int]]) -> list[dict[text, int]]\n0"
        )
        assert contract.result == BoundaryList(BoundaryDict(BoundaryScalar(ScalarKind.INT)))


# ---------------------------------------------------------------------------
# Records, enums, exceptions
# ---------------------------------------------------------------------------


class TestNominals:
    def test_record(self) -> None:
        source = "record Box\n  value: int\n  label: text\nextern def f(b: Box) -> Box\n0"
        contract = build_contract(source)
        assert isinstance(contract.result, BoundaryRecord)
        assert contract.result.display_name == "Box"
        assert contract.result.fields == (
            ("value", BoundaryScalar(ScalarKind.INT)),
            ("label", BoundaryScalar(ScalarKind.TEXT)),
        )

    def test_enum_variant_names_and_field_order(self) -> None:
        source = (
            "enum Shape\n"
            "  | circle(radius: decimal)\n"
            "  | rect(width: int, height: int)\n"
            "extern def f(s: Shape) -> Shape\n0"
        )
        contract = build_contract(source)
        assert isinstance(contract.result, BoundaryEnum)
        assert contract.result.display_name == "Shape"
        assert contract.result.variants == (
            BoundaryVariantShape(
                name="circle", fields=(("radius", BoundaryScalar(ScalarKind.DECIMAL)),)
            ),
            BoundaryVariantShape(
                name="rect",
                fields=(
                    ("width", BoundaryScalar(ScalarKind.INT)),
                    ("height", BoundaryScalar(ScalarKind.INT)),
                ),
            ),
        )

    def test_exception(self) -> None:
        source = (
            "exception BadThing extends Exception\n  detail: text\n"
            "extern def f(x: int) -> BadThing\n0"
        )
        contract = build_contract(source)
        assert isinstance(contract.result, BoundaryException)
        assert contract.result.display_name == "BadThing"
        field_names = [name for name, _ in contract.result.fields]
        assert "detail" in field_names
        assert dict(contract.result.fields)["detail"] == BoundaryScalar(ScalarKind.TEXT)


# ---------------------------------------------------------------------------
# Option[T] — an ordinary generic enum, no special-casing
# ---------------------------------------------------------------------------


class TestOption:
    _OPT = "enum Option[T]\n  | none\n  | some(value: T)\n"

    def test_option_of_int(self) -> None:
        contract = build_contract(self._OPT + "extern def f(o: Option[int]) -> int\n0")
        opt_schema = contract.params[0].schema
        assert isinstance(opt_schema, BoundaryEnum)
        assert opt_schema.display_name == "Option"
        variants = {v.name: v for v in opt_schema.variants}
        assert variants["none"].fields == ()
        assert variants["some"].fields == (("value", BoundaryScalar(ScalarKind.INT)),)

    def test_option_of_type_var(self) -> None:
        contract = build_contract(self._OPT + "extern def f[T](o: Option[T]) -> T\n0")
        opt_schema = contract.params[0].schema
        assert isinstance(opt_schema, BoundaryEnum)
        variants = {v.name: v for v in opt_schema.variants}
        assert variants["some"].fields == (("value", BoundarySealVar("T")),)
        assert contract.result == BoundarySealVar("T")


# ---------------------------------------------------------------------------
# Deep nesting
# ---------------------------------------------------------------------------


class TestDeepNesting:
    def test_list_of_dict_of_option(self) -> None:
        source = (
            "enum Option[T]\n  | none\n  | some(value: T)\n"
            "extern def f[T](x: list[dict[text, Option[T]]]) -> int\n0"
        )
        contract = build_contract(source)
        schema = contract.params[0].schema
        assert isinstance(schema, BoundaryList)
        assert isinstance(schema.element, BoundaryDict)
        inner = schema.element.value
        assert isinstance(inner, BoundaryEnum)
        variants = {v.name: v for v in inner.variants}
        assert variants["some"].fields == (("value", BoundarySealVar("T")),)


# ---------------------------------------------------------------------------
# Type parameters — sealed leaves
# ---------------------------------------------------------------------------


class TestTypeParams:
    def test_reverse_signature_yields_seal_vars(self) -> None:
        contract = build_contract(
            "extern def reverse[T](xs: list[T]) -> list[T]\n0", fn_name="reverse"
        )
        assert contract.type_params == ("T",)
        assert contract.params[0].schema == BoundaryList(BoundarySealVar("T"))
        assert contract.result == BoundaryList(BoundarySealVar("T"))

    def test_two_type_variables_stay_distinct(self) -> None:
        contract = build_contract("extern def pair[A, B](a: A, b: B) -> A\n0", fn_name="pair")
        assert contract.type_params == ("A", "B")
        assert contract.params[0].schema == BoundarySealVar("A")
        assert contract.params[1].schema == BoundarySealVar("B")
        assert contract.result == BoundarySealVar("A")


# ---------------------------------------------------------------------------
# Generic nominal instantiation — instantiated field types, not templates
# ---------------------------------------------------------------------------


class TestGenericNominalInstantiation:
    def test_generic_record_instantiated_at_concrete_type(self) -> None:
        source = "record Box[T]\n  value: T\nextern def f(b: Box[int]) -> int\n0"
        contract = build_contract(source)
        schema = contract.params[0].schema
        assert isinstance(schema, BoundaryRecord)
        assert schema.fields == (("value", BoundaryScalar(ScalarKind.INT)),)

    def test_generic_enum_instantiated_at_concrete_type(self) -> None:
        source = (
            "enum Holder[T]\n  | empty\n  | full(value: T)\nextern def f(h: Holder[text]) -> int\n0"
        )
        contract = build_contract(source)
        schema = contract.params[0].schema
        assert isinstance(schema, BoundaryEnum)
        variants = {v.name: v for v in schema.variants}
        assert variants["full"].fields == (("value", BoundaryScalar(ScalarKind.TEXT)),)


# ---------------------------------------------------------------------------
# Recursive types — shared BoundaryRef defs (same recursion plan as decode)
# ---------------------------------------------------------------------------


_TREE = "enum Tree\n  | Leaf(value: int)\n  | Node(left: Tree, right: Tree)\n"
_PERFECT = (
    "record Pair[A, B]\n  first: A\n  second: B\n"
    "enum Perfect[T]\n  | Single(value: T)\n  | Succ(next: Perfect[Pair[T, T]])\n"
)


class TestRecursiveTypes:
    def test_recursive_enum_crosses_as_boundary_ref(self) -> None:
        contract = build_contract(_TREE + "extern def f(t: Tree) -> int\n0")
        schema = contract.params[0].schema
        assert isinstance(schema, BoundaryRef)
        defs = dict(contract.defs)
        assert schema.key in defs
        body = defs[schema.key]
        assert isinstance(body, BoundaryEnum)
        node = {v.name: v for v in body.variants}["Node"]
        # Every self-occurrence — including inside the def body itself — refs out.
        assert node.fields == (
            ("left", BoundaryRef(schema.key)),
            ("right", BoundaryRef(schema.key)),
        )

    def test_recursive_record_crosses_as_boundary_ref(self) -> None:
        # A self-referential record (finite via the possibly-empty child list).
        source = (
            "record Node\n  value: int\n  children: list[Node]\nextern def f(n: Node) -> int\n0"
        )
        contract = build_contract(source)
        schema = contract.params[0].schema
        assert isinstance(schema, BoundaryRef)
        body = dict(contract.defs)[schema.key]
        assert isinstance(body, BoundaryRecord)
        assert body.fields == (
            ("value", BoundaryScalar(ScalarKind.INT)),
            ("children", BoundaryList(BoundaryRef(schema.key))),
        )

    def test_recursive_type_shared_across_param_and_result(self) -> None:
        # One plan spans all param types and the result, so a recursive type used
        # in both positions gets a single shared defs key/body.
        contract = build_contract(_TREE + "extern def f(t: Tree) -> Tree\n0")
        param_schema = contract.params[0].schema
        assert isinstance(param_schema, BoundaryRef)
        assert isinstance(contract.result, BoundaryRef)
        assert param_schema.key == contract.result.key
        assert len(contract.defs) == 1

    def test_recursive_exception_crosses_as_boundary_ref(self) -> None:
        source = (
            "exception Wrapped extends Exception\n  causes: list[Wrapped]\n"
            "extern def f(e: Wrapped) -> Wrapped\n0"
        )
        contract = build_contract(source)
        param_schema = contract.params[0].schema
        assert isinstance(param_schema, BoundaryRef)
        assert isinstance(contract.result, BoundaryRef)
        assert param_schema.key == contract.result.key
        body = dict(contract.defs)[param_schema.key]
        assert isinstance(body, BoundaryException)
        assert body.fields[-1] == ("causes", BoundaryList(BoundaryRef(param_schema.key)))

    def test_non_finite_schema_param_rejected(self) -> None:
        with pytest.raises(AglTypeError) as exc:
            build_contract(_PERFECT + "extern def f(p: Perfect[int]) -> int\n0")
        message = str(exc.value).lower()
        assert "no finite json schema" in message
        assert "extern parameter type" in message

    def test_non_finite_schema_return_rejected(self) -> None:
        with pytest.raises(AglTypeError) as exc:
            build_contract(_PERFECT + "extern def f(n: int) -> Perfect[int]\n0")
        message = str(exc.value).lower()
        assert "no finite json schema" in message
        assert "extern return type" in message


# ---------------------------------------------------------------------------
# Artifact hygiene
# ---------------------------------------------------------------------------


class TestArtifactHygiene:
    def test_contracts_hashable_and_equal_by_value(self) -> None:
        c1 = build_contract("extern def f(x: int) -> int\n0")
        c2 = build_contract("extern def f(x: int) -> int\n0")
        assert c1 == c2
        assert hash(c1) == hash(c2)
        assert {c1, c2} == {c1}

    def test_boundary_schema_nodes_hashable_and_equal_by_value(self) -> None:
        a = BoundaryList(BoundaryScalar(ScalarKind.INT))
        b = BoundaryList(BoundaryScalar(ScalarKind.INT))
        assert a == b
        assert hash(a) == hash(b)


# ---------------------------------------------------------------------------
# Function/agent-type ban — statically unreachable from source, so exercised
# by direct invocation of the compiler with a hand-built signature.
# ---------------------------------------------------------------------------


class TestFunctionAgentTypeBan:
    def test_function_typed_param_rejected(self) -> None:
        sig = FunctionSignature(
            params=(
                ParamSpec(
                    name="cb",
                    type=FunctionType(params=(IntType(),), result=IntType()),
                    kind=ParamKind.STANDARD,
                    has_default=False,
                ),
            ),
            result=IntType(),
        )
        try:
            build_extern_contract(sig, create_seeded_type_table())
        except TypeError as exc:
            assert "function" in str(exc).lower()
        else:
            raise AssertionError("expected TypeError")

    def test_agent_typed_result_rejected(self) -> None:
        sig = FunctionSignature(params=(), result=AgentType())
        try:
            build_extern_contract(sig, create_seeded_type_table())
        except TypeError as exc:
            assert "agent" in str(exc).lower()
        else:
            raise AssertionError("expected TypeError")

    def test_agent_typed_param_rejected(self) -> None:
        sig = FunctionSignature(
            params=(
                ParamSpec(name="a", type=AgentType(), kind=ParamKind.STANDARD, has_default=False),
            ),
            result=TextType(),
        )
        try:
            build_extern_contract(sig, create_seeded_type_table())
        except TypeError as exc:
            assert "agent" in str(exc).lower()
        else:
            raise AssertionError("expected TypeError")
