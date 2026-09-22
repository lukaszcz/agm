"""Type-directed externs: target type parameters and their per-occurrence resolution.

A target parameter is an extern type parameter no value parameter (receiver
included) mentions. Each occurrence (direct or method call, value reference,
partial application) resolves it from explicit type arguments or the expected
type and records one strict JSON contract spec per target parameter, in
declaration order: a call and a declared partial application at the ``Call``
node, a reference and a member partial application at the ``VarRef`` or
``FieldAccess`` naming the extern. Lowering turns each occurrence's contracts
into leading ``IrContract`` operands of the extern call, which evaluate to
opaque ``ContractValue``s the companion receives as ``agl.TypeContract``s.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from agm.agl.ir.contracts import TypeNode, TypeNodeKind, TypeNodeRef, TypeTree, TypeTreeEntry
from agm.agl.ir.ids import ContractId, FunctionId
from agm.agl.ir.nodes import (
    IrConstInt,
    IrConstText,
    IrContract,
    IrDirectCall,
    IrLoad,
    UseDefault,
)
from agm.agl.ir.program import ExecutableProgram, ExternFunctionBody
from agm.agl.parser import parse_program
from agm.agl.runtime.serialize import AglNonDataValue, value_to_json_obj
from agm.agl.scope.program import resolve_program
from agm.agl.semantics.types import (
    ArrayType,
    BoolType,
    EnumType,
    IntType,
    JsonType,
    RecordType,
    TextType,
    Type,
)
from agm.agl.semantics.values import ContractValue, TextValue
from agm.agl.syntax.nodes import (
    Block,
    Call,
    Expr,
    FieldAccess,
    FuncDef,
    LetDecl,
    TypeApply,
    VarRef,
)
from agm.agl.syntax.visitor import walk
from agm.agl.typecheck import AglTypeError, CheckedModule, FunctionSignature, check_program
from agm.agl.typecheck.env import OutputContractSpec
from tests.agl.ir_harness import (
    base_caps,
    evaluate_ir_with_externs,
    lower_extern_program,
    make_graph_from_files,
    write_companion_file,
)
from tests.agl.module_graph import resolve_and_check_inline_program_ast

_PATH = Path("/virtual/extern_targets.agl")

_QUERY = "extern def query[T](question: text) -> T\n"
_TEAM = "enum Team =\n  | Billing\n  | Technical\n"


def _check(source: str) -> CheckedModule:
    return resolve_and_check_inline_program_ast(
        parse_program(source), base_caps(), origin_path=_PATH
    )


def _reject(source: str) -> AglTypeError:
    with pytest.raises(AglTypeError) as exc_info:
        _check(source)
    return exc_info.value


def _signature(checked: CheckedModule, name: str) -> FunctionSignature:
    definition = next(
        item
        for item in checked.resolved.program.body.items
        if isinstance(item, FuncDef) and item.name == name
    )
    signature = checked.type_env.get_function_signature_by_node_id(definition.node_id)
    assert signature is not None
    return signature


def _let_value(checked: CheckedModule, binding: str) -> Expr:
    """Return the expression initializing ``let <binding>``."""
    items = list(checked.resolved.program.body.items)
    while items:
        item = items.pop()
        if isinstance(item, FuncDef) and isinstance(item.body, Block):
            items.extend(item.body.items)
        elif isinstance(item, LetDecl) and item.name == binding:
            assert item.value is not None
            return item.value
    raise AssertionError(f"no binding {binding!r}")


def _let_call(checked: CheckedModule, binding: str) -> int:
    """Return the node id of the call initializing ``let <binding>``."""
    value = _let_value(checked, binding)
    assert isinstance(value, Call)
    return value.node_id


def _let_reference(checked: CheckedModule, binding: str) -> int:
    """Return the node id naming the extern referenced by ``let <binding>``."""
    value = _let_value(checked, binding)
    if isinstance(value, TypeApply):
        value = value.expr
    assert isinstance(value, (VarRef, FieldAccess))
    return value.node_id


def _references(checked: CheckedModule, name: str) -> list[int]:
    """Return the node ids of every ``VarRef`` to *name*, in source order."""
    found: list[int] = []

    def visit(node: object) -> None:
        if isinstance(node, VarRef) and node.name == name:
            found.append(node.node_id)

    walk(checked.resolved.program, visit)
    return found


def _only_targets(checked: CheckedModule) -> tuple[Type, ...]:
    """Return the targets of the program's single recorded extern occurrence."""
    ((_node_id, specs),) = checked.target_contract_specs.items()
    return tuple(spec.target_type for spec in specs)


def _targets(checked: CheckedModule, node_id: int) -> tuple[Type, ...]:
    specs = checked.target_contract_specs[node_id]
    assert all(spec == OutputContractSpec(spec.target_type, "json", True) for spec in specs)
    return tuple(spec.target_type for spec in specs)


def _nominal_name(typ: Type) -> str:
    assert isinstance(typ, (RecordType, EnumType))
    return typ.name


class TestTargetParameterMarker:
    def test_unmentioned_type_parameter_is_a_target(self) -> None:
        checked = _check(_QUERY + "0")
        assert _signature(checked, "query").target_params == ("T",)

    def test_value_bound_type_parameter_is_not_a_target(self) -> None:
        checked = _check("extern def id[T](value: T) -> T\n0")
        assert _signature(checked, "id").target_params == ()

    def test_callback_mention_binds_a_type_parameter(self) -> None:
        checked = _check("extern def run[T](callback: (T) -> int) -> T\n0")
        assert _signature(checked, "run").target_params == ()

    def test_multiple_targets_keep_declaration_order(self) -> None:
        checked = _check("extern def f[U, A, T](value: A) -> int\n0")
        assert _signature(checked, "f").target_params == ("U", "T")

    def test_receiver_mentioned_type_parameters_are_not_targets(self) -> None:
        checked = _check("record Box[E](value: E)\nextern def Box::convert[E, T](self) -> T\n0")
        assert _signature(checked, "convert").target_params == ("T",)

    def test_non_extern_def_is_unmarked(self) -> None:
        checked = _check("def empty[T](size: int) -> array[T] = []\n0")
        assert _signature(checked, "empty").target_params == ()

    def test_standard_library_externs_are_all_value_bound(self, tmp_path: Path) -> None:
        stdlib = Path(__file__).resolve().parents[1] / "packages" / "stdlib" / "src"
        imports = "".join(f"import std/{path.stem}\n" for path in sorted(stdlib.glob("*.agl")))
        checked = check_program(
            resolve_program(make_graph_from_files(tmp_path, {"entry": imports + "()"})),
            base_caps(),
        )
        externs = [
            (module_id, item)
            for module_id, module in checked.modules.items()
            for item in module.resolved.program.body.items
            if isinstance(item, FuncDef) and item.is_extern
        ]
        assert any(item.type_param_slots for _module_id, item in externs)
        for module_id, item in externs:
            signature = checked.modules[module_id].type_env.get_function_signature_by_node_id(
                item.node_id
            )
            assert signature is not None
            assert signature.target_params == (), (module_id, item.name)


class TestDirectCallResolution:
    def test_explicit_type_argument(self) -> None:
        checked = _check(_QUERY + 'let answer = query::[int]("q")\nanswer')
        assert _targets(checked, _let_call(checked, "answer")) == (IntType(),)

    def test_annotation(self) -> None:
        checked = _check(_QUERY + 'let answer: bool = query("q")\nanswer')
        assert _targets(checked, _let_call(checked, "answer")) == (BoolType(),)

    def test_explicit_type_argument_wins_over_compatible_context(self) -> None:
        checked = _check(_QUERY + 'let answer: json = query::[int]("q")\nanswer')
        assert _targets(checked, _let_call(checked, "answer")) == (IntType(),)

    def test_assignment_target(self) -> None:
        checked = _check(_QUERY + 'var count: int = 0\ncount := query("q")\ncount')
        assert list(checked.target_contract_specs.values()) == [
            (OutputContractSpec(IntType(), "json", True),)
        ]

    def test_parameter_type(self) -> None:
        checked = _check(_QUERY + 'def use(flag: bool) -> bool = flag\nuse(query("q"))')
        assert [specs[0].target_type for specs in checked.target_contract_specs.values()] == [
            BoolType()
        ]

    def test_multiple_targets_resolve_in_declaration_order(self) -> None:
        checked = _check(
            "extern def pick[A, B](question: text) -> int\n"
            'let picked = pick::[text, bool]("q")\n'
            "picked"
        )
        assert _targets(checked, _let_call(checked, "picked")) == (TextType(), BoolType())

    def test_value_bound_parameters_are_not_contracts(self) -> None:
        checked = _check("extern def tag[V, T](value: V) -> T\nlet tagged: text = tag(1)\ntagged")
        assert _targets(checked, _let_call(checked, "tagged")) == (TextType(),)

    def test_each_occurrence_resolves_independently(self) -> None:
        checked = _check(_QUERY + 'let a: int = query("q")\nlet b: text = query("q")\nb')
        assert _targets(checked, _let_call(checked, "a")) == (IntType(),)
        assert _targets(checked, _let_call(checked, "b")) == (TextType(),)

    def test_concrete_target_inside_a_generic_def(self) -> None:
        checked = _check(_QUERY + 'def keep[U](value: U) -> int = query("q")\nkeep(true)')
        assert [specs[0].target_type for specs in checked.target_contract_specs.values()] == [
            IntType()
        ]

    def test_imported_extern(self, tmp_path: Path) -> None:
        write_companion_file(tmp_path / "root", "lib/jev", "def ask(question):\n    return 0\n")
        checked = check_program(
            resolve_program(
                make_graph_from_files(
                    tmp_path,
                    {
                        "entry": 'import lib/jev\nlet answer: int = jev::ask("q")\nanswer',
                        "lib/jev": "extern def ask[T](question: text) -> T",
                    },
                )
            ),
            base_caps(),
        )
        entry = checked.modules[checked.entry_id]
        assert _targets(entry, _let_call(entry, "answer")) == (IntType(),)


class TestMethodCallResolution:
    _BOX = "record Box[E](value: E)\nextern def Box::convert[E, T](self) -> T\n"

    def test_explicit_type_argument(self) -> None:
        checked = _check(
            self._BOX + "let box = Box(value = 1)\nlet converted = box.convert::[text]()\nconverted"
        )
        assert _targets(checked, _let_call(checked, "converted")) == (TextType(),)

    def test_annotation(self) -> None:
        checked = _check(
            self._BOX + "let box = Box(value = 1)\nlet converted: bool = box.convert()\nconverted"
        )
        assert _targets(checked, _let_call(checked, "converted")) == (BoolType(),)

    def test_builtin_receiver(self) -> None:
        checked = _check(
            "extern def int::convert[T](self) -> T\nlet converted: text = 1.convert()\nconverted"
        )
        assert _targets(checked, _let_call(checked, "converted")) == (TextType(),)

    def test_value_bound_method_records_no_contract(self) -> None:
        checked = _check(
            "record Box[E](value: E)\nextern def Box::get[E](self) -> E\n"
            "let got = Box(value = 1).get()\ngot"
        )
        assert checked.target_contract_specs == {}


class TestReferenceResolution:
    _BOX = "record Box[E](value: E)\nextern def Box::convert[E, T](self) -> T\n"

    def test_explicit_type_argument(self) -> None:
        checked = _check(_QUERY + 'let ask-int = query::[int]\nask-int("q")')
        assert _targets(checked, _let_reference(checked, "ask-int")) == (IntType(),)

    def test_expected_function_type(self) -> None:
        checked = _check(_QUERY + 'let ask-bool: (text) -> bool = query\nask-bool("q")')
        assert _targets(checked, _let_reference(checked, "ask-bool")) == (BoolType(),)

    def test_invocation_records_only_the_reference(self) -> None:
        checked = _check(_QUERY + 'let ask-int = query::[int]\nask-int("q") + ask-int("r")')
        assert list(checked.target_contract_specs) == [_let_reference(checked, "ask-int")]

    def test_argument_position(self) -> None:
        checked = _check(
            _QUERY + 'def apply(ask-it: (text) -> text) -> text = ask-it("q")\napply(query)'
        )
        (argument,) = _references(checked, "query")
        assert list(checked.target_contract_specs) == [argument]
        assert _only_targets(checked) == (TextType(),)

    def test_each_branch_occurrence_records(self) -> None:
        checked = _check(
            _QUERY + 'let ask-it = if true => query\n  | else => query::[int]\nask-it("q")'
        )
        first, second = _references(checked, "query")
        assert sorted(checked.target_contract_specs) == sorted([first, second])
        assert _targets(checked, first) == (IntType(),)
        assert _targets(checked, second) == (IntType(),)

    @pytest.mark.parametrize(
        ("binding", "target"),
        [("let f: (text) -> int = jev::ask", IntType()), ("let f = jev::ask::[bool]", BoolType())],
    )
    def test_imported_qualified_extern(self, tmp_path: Path, binding: str, target: Type) -> None:
        write_companion_file(tmp_path / "root", "lib/jev", "def ask(question):\n    return 0\n")
        checked = check_program(
            resolve_program(
                make_graph_from_files(
                    tmp_path,
                    {
                        "entry": f'import lib/jev\n{binding}\nf("q")',
                        "lib/jev": "extern def ask[T](question: text) -> T",
                    },
                )
            ),
            base_caps(),
        )
        entry = checked.modules[checked.entry_id]
        assert _targets(entry, _let_reference(entry, "f")) == (target,)

    def test_qualified_method_reference(self) -> None:
        checked = _check(
            self._BOX + "let convert-box: (Box[int]) -> text = Box::convert\n"
            "convert-box(Box(value = 1))"
        )
        assert _targets(checked, _let_reference(checked, "convert-box")) == (TextType(),)

    def test_bound_method_expected_type(self) -> None:
        checked = _check(
            self._BOX + "let box = Box(value = 1)\nlet converted: () -> bool = box.convert\n"
            "converted()"
        )
        assert _targets(checked, _let_reference(checked, "converted")) == (BoolType(),)

    def test_bound_method_explicit_type_argument(self) -> None:
        checked = _check(
            self._BOX + "let box = Box(value = 1)\nlet converted = box.convert::[text]\nconverted()"
        )
        assert _targets(checked, _let_reference(checked, "converted")) == (TextType(),)

    def test_direct_method_call_records_only_the_call(self) -> None:
        checked = _check(self._BOX + "let converted: text = Box(value = 1).convert()\nconverted")
        assert list(checked.target_contract_specs) == [_let_call(checked, "converted")]

    def test_value_bound_reference_records_no_contract(self) -> None:
        checked = _check("extern def id[T](value: T) -> T\nlet same: (int) -> int = id\nsame(1)")
        assert checked.target_contract_specs == {}


class TestPartialApplicationResolution:
    _CLASSIFY = "extern def classify[T](question: text, context: text) -> T\n"
    _BOX = "record Box[E](value: E)\nextern def Box::rate[E, T](self, question: text) -> T\n"

    def test_explicit_type_argument(self) -> None:
        checked = _check(
            self._CLASSIFY + 'let classify-q = classify::[int]("q", ?)\nclassify-q("c")'
        )
        assert _targets(checked, _let_call(checked, "classify-q")) == (IntType(),)

    def test_expected_function_type(self) -> None:
        checked = _check(
            self._CLASSIFY + 'let classify-q: (text) -> bool = classify(?, "c")\nclassify-q("q")'
        )
        assert _targets(checked, _let_call(checked, "classify-q")) == (BoolType(),)

    def test_invocation_records_only_the_partial(self) -> None:
        checked = _check(
            self._CLASSIFY + 'let classify-q = classify::[int]("q", ?)\nclassify-q("c")'
        )
        assert list(checked.target_contract_specs) == [_let_call(checked, "classify-q")]

    def test_member_partial_records_at_the_method_reference(self) -> None:
        checked = _check(
            self._BOX + "let box = Box(value = 1)\nlet rate-box: (text) -> text = box.rate(?)\n"
            'rate-box("q")'
        )
        call = _let_value(checked, "rate-box")
        assert isinstance(call, Call)
        assert list(checked.target_contract_specs) == [call.callee.node_id]
        assert _targets(checked, call.callee.node_id) == (TextType(),)

    def test_member_partial_explicit_type_argument(self) -> None:
        checked = _check(
            self._BOX + 'let box = Box(value = 1)\nlet rate-box = box.rate::[int](?)\nrate-box("q")'
        )
        call = _let_value(checked, "rate-box")
        assert isinstance(call, Call)
        assert _targets(checked, call.callee.node_id) == (IntType(),)


class TestResolutionErrors:
    def test_no_default_target(self) -> None:
        _reject(_QUERY + 'let answer = query("q")\nanswer')

    def test_no_default_target_for_a_method(self) -> None:
        _reject(
            "record Box(value: int)\nextern def Box::convert[T](self) -> T\n"
            "let converted = Box(value = 1).convert()\nconverted"
        )

    def test_no_default_for_a_target_absent_from_the_result(self) -> None:
        _reject('extern def poll[T](question: text) -> int\nlet polled = poll("q")\npolled')

    def test_contextual_type_variable_rejected(self) -> None:
        _reject(_QUERY + "def relay[U](question: text) -> U = query(question)\n0")

    def test_explicit_nested_type_variable_rejected(self) -> None:
        _reject(
            _QUERY + "def relay[U](question: text) -> array[U] = query::[array[U]](question)\n0"
        )

    def test_method_type_variable_rejected(self) -> None:
        _reject(
            "record Box(value: int)\nextern def Box::convert[T](self) -> T\n"
            "def relay[U](box: Box) -> U = box.convert()\n0"
        )

    def test_target_without_finite_schema_rejected(self) -> None:
        err = _reject(
            _QUERY + "record Pair[A, B](first: A, second: B)\n"
            "enum Perfect[T]\n"
            "  | Single(value: T)\n"
            "  | Succ(next: Perfect[Pair[T, T]])\n"
            'let grown = query::[Perfect[int]]("q")\n'
            "grown"
        )
        assert "finite json schema" in str(err).lower()

    @pytest.mark.parametrize(
        "target", ["(int) -> int", "array[(int) -> int]", "Oops", "dict[text, Oops]", "unit"]
    )
    def test_target_outside_the_json_contract_rejected(self, target: str) -> None:
        _reject(_QUERY + f'exception Oops(detail: text)\nlet answer = query::[{target}]("q")\n0')

    def test_method_function_target_rejected(self) -> None:
        _reject(
            "record Box(value: int)\nextern def Box::convert[T](self) -> T\n"
            "let converted: (int) -> int = Box(value = 1).convert()\n0"
        )

    def test_unresolved_reference(self) -> None:
        _reject(_QUERY + "let ask-it = query\n0")

    def test_unresolved_partial_application(self) -> None:
        _reject(
            "extern def classify[T](question: text, context: text) -> T\n"
            'let classify-q = classify("q", ?)\n0'
        )

    def test_unresolved_bound_method_reference(self) -> None:
        _reject(
            "record Box(value: int)\nextern def Box::convert[T](self) -> T\n"
            "let converted = Box(value = 1).convert\n0"
        )

    @pytest.mark.parametrize(
        "occurrence",
        [
            "let ask-it: (text) -> U = query\n  ask-it(question)",
            "let ask-it = query::[array[U]]\n  ask-it(question)[0]",
            "let ask-it: (text) -> U = classify(question, ?)\n  ask-it(question)",
            "let ask-it = classify::[U](?, question)\n  ask-it(question)",
            "let ask-it: () -> U = Box(value = 1).convert\n  ask-it()",
            "let ask-it: (text) -> U = Box(value = 1).rate(?)\n  ask-it(question)",
        ],
    )
    def test_value_occurrence_type_variable_rejected(self, occurrence: str) -> None:
        _reject(
            _QUERY + "extern def classify[T](question: text, context: text) -> T\n"
            "record Box(value: int)\nextern def Box::convert[T](self) -> T\n"
            "extern def Box::rate[T](self, question: text) -> T\n"
            f"def relay[U](question: text) -> U =\n  {occurrence}\n0"
        )

    @pytest.mark.parametrize(
        "occurrence", ["query::[Perfect[int]]", 'classify::[Perfect[int]]("q", ?)']
    )
    def test_value_occurrence_without_finite_schema_rejected(self, occurrence: str) -> None:
        err = _reject(
            _QUERY + "extern def classify[T](question: text, context: text) -> T\n"
            "record Pair[A, B](first: A, second: B)\n"
            "enum Perfect[T]\n"
            "  | Single(value: T)\n"
            "  | Succ(next: Perfect[Pair[T, T]])\n"
            f"let ask-it = {occurrence}\n0"
        )
        assert "finite json schema" in str(err).lower()

    @pytest.mark.parametrize(
        "occurrence",
        [
            "query::[(int) -> int]",
            "query::[unit]",
            'classify::[unit](?, "c")',
            'classify::[array[(int) -> int]]("q", ?)',
        ],
    )
    def test_value_occurrence_outside_the_json_contract_rejected(self, occurrence: str) -> None:
        _reject(
            _QUERY + "extern def classify[T](question: text, context: text) -> T\n"
            f"let ask-it = {occurrence}\n0"
        )

    def test_json_target_accepted(self) -> None:
        checked = _check(_QUERY + 'let answer = query::[json]("q")\nanswer')
        assert _targets(checked, _let_call(checked, "answer")) == (JsonType(),)


class TestNestedGenericTargets:
    def test_generic_record_result_resolves_its_argument(self) -> None:
        checked = _check(
            _TEAM + "record Choice[C](choice: C, confidence: decimal)\n"
            "extern def choose[C](question: text) -> Choice[C]\n"
            'let chosen: Choice[Team] = choose("q")\n'
            "chosen"
        )
        (target,) = _targets(checked, _let_call(checked, "chosen"))
        assert _nominal_name(target) == "Team"

    def test_generic_record_as_whole_target(self) -> None:
        checked = _check(
            _QUERY + _TEAM + "record Choice[C](choice: C, confidence: decimal)\n"
            'let chosen = query::[Choice[Team]]("q")\n'
            "chosen"
        )
        (target,) = _targets(checked, _let_call(checked, "chosen"))
        assert _nominal_name(target) == "Choice"
        assert isinstance(target, RecordType)
        assert [_nominal_name(arg) for arg in target.type_args] == ["Team"]

    def test_array_of_generic_enum(self) -> None:
        # The inline harness loads no standard library, so ``Maybe`` stands in for ``Option``.
        checked = _check(
            _QUERY + _TEAM + "enum Maybe[T]\n  | Nothing\n  | Just(value: T)\n"
            'let answers: array[Maybe[Team]] = query("q")\n'
            "answers"
        )
        (target,) = _targets(checked, _let_call(checked, "answers"))
        assert isinstance(target, ArrayType)
        assert _nominal_name(target.elem) == "Maybe"
        assert isinstance(target.elem, EnumType)
        assert [_nominal_name(arg) for arg in target.elem.type_args] == ["Team"]


class TestInventory:
    def test_call_site_record_is_unchanged(self) -> None:
        checked = _check(_QUERY + 'let answer: int = query("q")\nanswer')
        (site,) = checked.call_sites
        assert (site.callee, site.codec_name, site.target_type) == ("query", "extern", IntType())
        assert checked.contract_specs == {}

    def test_ordinary_extern_records_no_target_contract(self) -> None:
        checked = _check("extern def id[T](value: T) -> T\nlet same = id(1)\nsame")
        assert checked.target_contract_specs == {}

    def test_dry_run_inventory_is_unchanged(self, tmp_path: Path) -> None:
        program = lower_extern_program(
            _QUERY + 'let answer: int = query("q")\nanswer',
            "def query(question):\n    return 0\n",
            tmp_path,
        )
        entries = [
            entry
            for entry in program.dry_run_inventory
            if entry.module.is_entry and entry.callee == "query"
        ]
        assert [
            (entry.codec_name, entry.target_type_label, entry.has_schema) for entry in entries
        ] == [("extern", "int", False)]


# ---------------------------------------------------------------------------
# Lowering: contracts become leading extern-call operands
# ---------------------------------------------------------------------------

_COMPANION = (
    "def query(*args):\n    return args[0]\n"
    "def pick(*args):\n    return args[0]\n"
    "def classify(*args):\n    return args[0]\n"
    "def convert(*args):\n    return args[0]\n"
    "def rate(*args):\n    return args[0]\n"
    "def same(*args):\n    return args[0]\n"
    "def grade(*args):\n    return args[0]\n"
    "def tally(*args):\n    return args[0]\n"
)
_CLASSIFY = "extern def classify[T](question: text, context: text) -> T\n"
_BOX = (
    "record Box(value: int)\n"
    "extern def Box::convert[T](self) -> T\n"
    "extern def Box::rate[T](self, question: text) -> T\n"
    "let box = Box(value = 1)\n"
)


def _ir_nodes(node: object) -> Iterator[object]:
    """Yield *node* and every IR node reachable through its dataclass fields."""
    yield node
    children: tuple[object, ...]
    if isinstance(node, tuple):
        children = node
    elif is_dataclass(node) and not isinstance(node, type):
        children = tuple(getattr(node, field.name) for field in fields(node))
    else:
        return
    for child in children:
        yield from _ir_nodes(child)


def _program_nodes(program: ExecutableProgram) -> Iterator[object]:
    for module in program.modules.values():
        yield from _ir_nodes(module.initializers)
    for function in program.functions.values():
        yield from _ir_nodes(function.impl)


def _extern_id(program: ExecutableProgram, name: str) -> FunctionId:
    (function_id,) = (
        function_id
        for function_id, function in program.functions.items()
        if isinstance(function.impl, ExternFunctionBody) and function.impl.name == name
    )
    return function_id


def _extern_body(program: ExecutableProgram, function_id: FunctionId) -> ExternFunctionBody:
    impl = program.functions[function_id].impl
    assert isinstance(impl, ExternFunctionBody)
    return impl


def _extern_calls(program: ExecutableProgram, name: str) -> list[IrDirectCall]:
    """Return every direct call to extern *name*, in no particular order."""
    function_id = _extern_id(program, name)
    return [
        node
        for node in _program_nodes(program)
        if isinstance(node, IrDirectCall) and node.function_id == function_id
    ]


def _leading_contracts(program: ExecutableProgram, call: IrDirectCall) -> list[str]:
    """Return the target labels of *call*'s leading contracts, checking each is registered."""
    count = _extern_body(program, call.function_id).target_count
    assert not any(isinstance(operand, IrContract) for operand in call.arguments[count:])
    labels: list[str] = []
    for operand in call.arguments[:count]:
        assert isinstance(operand, IrContract)
        request = program.contracts[operand.contract_id]
        assert (request.codec_name, request.strict_json, request.structured_exec) == (
            "json",
            True,
            False,
        )
        labels.append(request.target_type_label)
    return labels


def _contract_ids(program: ExecutableProgram) -> list[ContractId]:
    return [node.contract_id for node in _program_nodes(program) if isinstance(node, IrContract)]


def _lower(source: str, tmp_path: Path) -> ExecutableProgram:
    return lower_extern_program(source, _COMPANION, tmp_path)


class TestTargetLowering:
    def test_target_count_follows_the_target_parameters(self, tmp_path: Path) -> None:
        program = _lower(
            _QUERY + "extern def pick[A, B](question: text) -> int\n"
            "extern def same[T](value: T) -> T\n0",
            tmp_path,
        )
        counts = {
            name: _extern_body(program, _extern_id(program, name)).target_count
            for name in ("query", "pick", "same")
        }
        assert counts == {"query": 1, "pick": 2, "same": 0}

    def test_direct_call(self, tmp_path: Path) -> None:
        program = _lower(_QUERY + 'let answer: int = query("q")\nanswer', tmp_path)
        (call,) = _extern_calls(program, "query")
        assert _leading_contracts(program, call) == ["int"]
        assert len(call.arguments) == 2

    def test_contracts_keep_declaration_order(self, tmp_path: Path) -> None:
        program = _lower(
            "extern def pick[A, B](question: text) -> int\n"
            'let picked = pick::[text, bool]("q")\npicked',
            tmp_path,
        )
        (call,) = _extern_calls(program, "pick")
        assert _leading_contracts(program, call) == ["text", "bool"]

    def test_each_occurrence_registers_its_own_contract(self, tmp_path: Path) -> None:
        program = _lower(_QUERY + 'let a: int = query("q")\nlet b: text = query("q")\nb', tmp_path)
        calls = _extern_calls(program, "query")
        assert sorted(label for call in calls for label in _leading_contracts(program, call)) == [
            "int",
            "text",
        ]
        ids = _contract_ids(program)
        assert len(ids) == len(set(ids)) == 2

    def test_omitted_argument_keeps_its_declared_default_slot(self, tmp_path: Path) -> None:
        program = _lower(
            "extern def query[T](question: text, retries: int = 1) -> T\n"
            'let answer: int = query("q")\nanswer',
            tmp_path,
        )
        (call,) = _extern_calls(program, "query")
        assert _leading_contracts(program, call) == ["int"]
        assert call.arguments[2] == UseDefault(param_index=1)

    def test_method_call(self, tmp_path: Path) -> None:
        program = _lower(_BOX + "let converted: text = box.convert()\nconverted", tmp_path)
        (call,) = _extern_calls(program, "convert")
        assert _leading_contracts(program, call) == ["text"]
        assert len(call.arguments) == 2

    def test_method_default_follows_the_leading_contract(self, tmp_path: Path) -> None:
        program = _lower(
            _BOX + "extern def Box::grade[T](self, question: text, retries: int = 1) -> T\n"
            'let graded: int = box.grade("q")\ngraded',
            tmp_path,
        )
        (call,) = _extern_calls(program, "grade")
        assert _leading_contracts(program, call) == ["int"]
        assert len(call.arguments) == 4
        assert call.arguments[3] == UseDefault(param_index=2)

    def test_named_arguments_follow_the_leading_contract(self, tmp_path: Path) -> None:
        program = _lower(
            "extern def tally[T](question: text, n: int) -> T\n"
            'let counted: int = tally(n = 3, question = "q")\ncounted',
            tmp_path,
        )
        (call,) = _extern_calls(program, "tally")
        assert _leading_contracts(program, call) == ["int"]
        question, n = call.arguments[1:]
        assert isinstance(question, IrConstText) and question.value == "q"
        assert isinstance(n, IrConstInt) and n.value == 3

    def test_reference_eta_expands(self, tmp_path: Path) -> None:
        program = _lower(
            _QUERY + 'let ask-int = query::[int]\nask-int("q") + ask-int("r")', tmp_path
        )
        (call,) = _extern_calls(program, "query")
        assert _leading_contracts(program, call) == ["int"]
        assert isinstance(call.arguments[1], IrLoad)
        assert len(_contract_ids(program)) == 1

    def test_qualified_method_reference(self, tmp_path: Path) -> None:
        program = _lower(
            _BOX + "let convert-box: (Box) -> bool = Box::convert\nconvert-box(box)", tmp_path
        )
        (call,) = _extern_calls(program, "convert")
        assert _leading_contracts(program, call) == ["bool"]
        assert isinstance(call.arguments[1], IrLoad)

    def test_bound_method_reference(self, tmp_path: Path) -> None:
        program = _lower(_BOX + "let converted = box.convert::[text]\nconverted()", tmp_path)
        (call,) = _extern_calls(program, "convert")
        assert _leading_contracts(program, call) == ["text"]
        assert len(call.arguments) == 2

    def test_declared_partial_application(self, tmp_path: Path) -> None:
        program = _lower(
            _CLASSIFY + 'let classify-q = classify::[int]("q", ?)\nclassify-q("c")', tmp_path
        )
        (call,) = _extern_calls(program, "classify")
        assert _leading_contracts(program, call) == ["int"]
        assert len(call.arguments) == 3
        assert len(_contract_ids(program)) == 1

    def test_member_partial_application(self, tmp_path: Path) -> None:
        program = _lower(_BOX + 'let rate-box = box.rate::[int](?)\nrate-box("q")', tmp_path)
        (call,) = _extern_calls(program, "rate")
        assert _leading_contracts(program, call) == ["int"]
        assert len(call.arguments) == 3

    def test_ordinary_extern_lowers_without_contracts(self, tmp_path: Path) -> None:
        program = _lower(
            "extern def same[T](value: T) -> T\nlet one = same(1)\n"
            "let f: (int) -> int = same\nf(one)",
            tmp_path,
        )
        (call,) = _extern_calls(program, "same")
        assert len(call.arguments) == 1
        assert _contract_ids(program) == []


# ---------------------------------------------------------------------------
# Runtime: contracts reach the companion as leading ``agl.TypeContract`` arguments
# ---------------------------------------------------------------------------


#: Shows each contract as ``contract:<label>#<object id>`` and a receiver as ``self``.
_ECHO = (
    "import agl\n"
    "def _show(arg):\n"
    "    if isinstance(arg, str):\n        return arg\n"
    "    if isinstance(arg, agl.TypeContract):\n"
    "        return f'contract:{arg.label}#{id(arg)}'\n"
    "    return 'self'\n"
    "def query(*args):\n    return ' '.join(map(_show, args))\n"
    "rate = convert = query\n"
)


class TestContractDelivery:
    def _run(self, source: str, tmp_path: Path) -> dict[str, str]:
        bindings, _output = evaluate_ir_with_externs(source, _ECHO, tmp_path)
        return {
            name: value.value for name, value in bindings.items() if isinstance(value, TextValue)
        }

    def test_call_passes_contract_first(self, tmp_path: Path) -> None:
        bindings = self._run(_QUERY + 'let answer: text = query("q")\nanswer', tmp_path)
        contract, question = bindings["answer"].split()
        assert contract.startswith("contract:text#")
        assert question == "q"

    def test_method_call_passes_contract_before_receiver(self, tmp_path: Path) -> None:
        bindings = self._run(_BOX + 'let rated: text = box.rate("q")\nrated', tmp_path)
        contract, receiver, question = bindings["rated"].split()
        assert contract.startswith("contract:text#")
        assert (receiver, question) == ("self", "q")

    def test_occurrences_deliver_distinct_contracts(self, tmp_path: Path) -> None:
        bindings = self._run(
            _QUERY + 'let a: text = query("q")\nlet b: text = query("q")\nb', tmp_path
        )
        assert bindings["a"] != bindings["b"]

    def test_reference_delivers_one_contract_per_occurrence(self, tmp_path: Path) -> None:
        bindings = self._run(
            _QUERY + "let ask-text = query::[text]\n"
            'let a = ask-text("q")\nlet b = ask-text("q")\nb',
            tmp_path,
        )
        assert bindings["a"] == bindings["b"]
        assert bindings["a"].endswith(" q")

    def test_partial_application_delivers_captured_contract(self, tmp_path: Path) -> None:
        bindings = self._run(
            "extern def query[T](question: text, context: text) -> T\n"
            'let ask-q = query::[text]("q", ?)\nlet a = ask-q("c")\na',
            tmp_path,
        )
        assert bindings["a"].split()[1:] == ["q", "c"]
        assert bindings["a"].startswith("contract:text#")


def test_contract_value_is_not_data() -> None:
    with pytest.raises(AglNonDataValue):
        value_to_json_obj(ContractValue(ContractId(0)))


# ---------------------------------------------------------------------------
# Contract type trees: the typeless target description of each occurrence
# ---------------------------------------------------------------------------

_DOCUMENTED_TEAM = (
    '@doc("The owning team.")\n'
    "enum Team\n"
    '  | @doc("Invoices and refunds.") @json-name("billing") Billing\n'
    "  | Technical\n"
)


def _tree(program: ExecutableProgram, name: str = "query") -> TypeTree:
    """Return the type tree of the single target contract of extern *name*'s single call."""
    (call,) = _extern_calls(program, name)
    operand = call.arguments[0]
    assert isinstance(operand, IrContract)
    tree = program.contracts[operand.contract_id].type_tree
    assert tree is not None
    return tree


def _body(tree: TypeTree, entry: TypeTreeEntry) -> TypeNode:
    """Resolve *entry* against *tree*'s definitions."""
    if isinstance(entry, TypeNodeRef):
        return dict(tree.defs)[entry.key]
    return entry


def _schema(node: TypeNode) -> object:
    return json.loads(node.schema)


def _display(program: ExecutableProgram, node: TypeNode) -> str:
    assert node.nominal is not None
    return program.nominals[node.nominal].display_name


class TestContractTypeTree:
    @pytest.mark.parametrize(
        ("target", "kind", "schema"),
        [
            ("int", TypeNodeKind.INT, {"type": "integer"}),
            ("text", TypeNodeKind.TEXT, {"type": "string"}),
            ("decimal", TypeNodeKind.DECIMAL, {"type": "number"}),
            ("bool", TypeNodeKind.BOOL, {"type": "boolean"}),
            ("json", TypeNodeKind.JSON, {}),
        ],
    )
    def test_scalar(self, tmp_path: Path, target: str, kind: TypeNodeKind, schema: object) -> None:
        tree = _tree(_lower(_QUERY + f'let answer: {target} = query("q")\n0', tmp_path))
        assert tree.defs == ()
        assert isinstance(tree.root, TypeNode)
        assert (tree.root.kind, tree.root.label, _schema(tree.root)) == (kind, target, schema)
        assert (tree.root.doc, tree.root.nominal) == (None, None)
        assert (tree.root.fields, tree.root.members) == ((), ())
        assert (tree.root.items, tree.root.values) == (None, None)

    def test_fieldless_enum_carries_docs_and_json_tags(self, tmp_path: Path) -> None:
        program = _lower(_QUERY + _DOCUMENTED_TEAM + 'let team: Team = query("q")\n0', tmp_path)
        root = _tree(program).root
        assert isinstance(root, TypeNode)
        assert (root.kind, root.label, root.doc) == (TypeNodeKind.ENUM, "Team", "The owning team.")
        assert _display(program, root) == "Team"
        schema = _schema(root)
        assert isinstance(schema, dict)
        assert [variant["properties"]["$case"] for variant in schema["oneOf"]] == [
            {"const": "billing"},
            {"const": "Technical"},
        ]
        assert [tag for tag, _member in root.members] == ["billing", "Technical"]
        (_, billing), (_, technical) = root.members
        assert (billing.kind, billing.doc) == (TypeNodeKind.MEMBER, "Invoices and refunds.")
        assert (technical.kind, technical.doc) == (TypeNodeKind.MEMBER, None)
        assert [_display(program, member) for member in (billing, technical)] == [
            "Team::Billing",
            "Team::Technical",
        ]
        assert _schema(billing) == schema["oneOf"][0]
        assert billing.fields == technical.fields == ()

    def test_record_fields_keep_order_names_tags_and_docs(self, tmp_path: Path) -> None:
        program = _lower(
            _QUERY + _DOCUMENTED_TEAM + '@doc("A triaged message.")\n'
            "record Triage\n"
            '  @doc("Is it urgent?") urgent: bool\n'
            '  @doc("Who handles it?") @json-name("owner") team: Team\n'
            "  notes: array[text]\n"
            'let triage: Triage = query("q")\n0',
            tmp_path,
        )
        root = _tree(program).root
        assert isinstance(root, TypeNode)
        assert (root.kind, root.label, root.doc) == (
            TypeNodeKind.RECORD,
            "Triage",
            "A triaged message.",
        )
        assert _display(program, root) == "Triage"
        assert [(f.json_name, f.name, f.doc) for f in root.fields] == [
            ("urgent", "urgent", "Is it urgent?"),
            ("owner", "team", "Who handles it?"),
            ("notes", "notes", None),
        ]
        urgent, team, notes = (field.node for field in root.fields)
        assert isinstance(urgent, TypeNode) and urgent.kind is TypeNodeKind.BOOL
        assert isinstance(team, TypeNode) and team.doc == "The owning team."
        assert [tag for tag, _member in team.members] == ["billing", "Technical"]
        assert isinstance(notes, TypeNode) and notes.kind is TypeNodeKind.ARRAY
        assert isinstance(notes.items, TypeNode) and notes.items.kind is TypeNodeKind.TEXT
        schema = _schema(root)
        assert isinstance(schema, dict)
        assert list(schema["properties"]) == ["urgent", "owner", "notes"]

    def test_generic_record_of_enum(self, tmp_path: Path) -> None:
        program = _lower(
            _QUERY + _DOCUMENTED_TEAM + "record Choice[C](choice: C, confidence: decimal)\n"
            'let chosen = query::[Choice[Team]]("q")\n0',
            tmp_path,
        )
        tree = _tree(program)
        root = _body(tree, tree.root)
        assert (root.kind, root.label) == (TypeNodeKind.RECORD, "Choice[Team]")
        choice, confidence = root.fields
        team = _body(tree, choice.node)
        assert (team.kind, team.label) == (TypeNodeKind.ENUM, "Team")
        assert [tag for tag, _member in team.members] == ["billing", "Technical"]
        assert isinstance(confidence.node, TypeNode)
        assert confidence.node.kind is TypeNodeKind.DECIMAL

    def test_array_and_dict_of_optional_enum(self, tmp_path: Path) -> None:
        program = _lower(
            _QUERY
            + _DOCUMENTED_TEAM
            + 'let picks: dict[text, array[Option[Team]]] = query("q")\n0',
            tmp_path,
        )
        tree = _tree(program)
        root = tree.root
        assert isinstance(root, TypeNode) and root.kind is TypeNodeKind.DICT
        assert _schema(root) == {
            "type": "object",
            "additionalProperties": _schema_of(root.values),
        }
        array = root.values
        assert isinstance(array, TypeNode) and array.kind is TypeNodeKind.ARRAY
        option = array.items
        assert isinstance(option, TypeNode)
        assert option.kind is TypeNodeKind.ENUM
        assert option.label.endswith("Option[Team]")
        (none_tag, none), (some_tag, some) = option.members
        assert (none_tag, none.fields, some_tag) == ("None", (), "Some")
        (value,) = some.fields
        assert _body(tree, value.node).doc == "The owning team."

    def test_recursive_target_refers_to_its_definition(self, tmp_path: Path) -> None:
        program = _lower(
            _QUERY + "enum Tree\n  | Leaf(value: int)\n  | Node(left: Tree, right: Tree)\n"
            'let tree: array[Tree] = query("q")\n0',
            tmp_path,
        )
        request_tree = _tree(program)
        root = request_tree.root
        assert isinstance(root, TypeNode) and root.kind is TypeNodeKind.ARRAY
        assert root.items == TypeNodeRef("Tree")
        assert [key for key, _node in request_tree.defs] == ["Tree"]
        tree = _body(request_tree, root.items)
        assert (tree.kind, tree.label) == (TypeNodeKind.ENUM, "Tree")
        (_, _leaf), (_, node) = tree.members
        assert [field.node for field in node.fields] == [TypeNodeRef("Tree")] * 2
        assert _schema(root) == {"type": "array", "items": {"$ref": "#/$defs/Tree"}}

    def test_only_type_directed_contracts_carry_a_tree(self, tmp_path: Path) -> None:
        program = _lower(
            _QUERY + 'let answer: int = query("q")\n'
            'let agent = AgentCommand("worker")\n'
            'def later() -> int = ask("How many?", agent = agent)\n0',
            tmp_path,
        )
        (call,) = _extern_calls(program, "query")
        operand = call.arguments[0]
        assert isinstance(operand, IrContract)
        (ask_id,) = set(program.contracts) - {operand.contract_id}
        assert program.contracts[operand.contract_id].type_tree is not None
        assert program.contracts[ask_id].type_tree is None

    def test_inline_member_fields_carry_docs(self, tmp_path: Path) -> None:
        program = _lower(
            _QUERY + "enum Shape\n"
            '  | Circle(@doc("The radius.") radius: decimal, label: text)\n'
            "  | Dot\n"
            'let shape: Shape = query("q")\n0',
            tmp_path,
        )
        root = _tree(program).root
        assert isinstance(root, TypeNode)
        (_, circle), (_, dot) = root.members
        assert [(f.name, f.doc) for f in circle.fields] == [
            ("radius", "The radius."),
            ("label", None),
        ]
        assert dot.fields == ()

    def test_generic_record_field_docs_survive_substitution(self, tmp_path: Path) -> None:
        program = _lower(
            _QUERY + "record Wrap[T]\n"
            '  @doc("The wrapped value.") value: T\n'
            "  count: int\n"
            'let wrapped: Wrap[int] = query("q")\n0',
            tmp_path,
        )
        root = _tree(program).root
        assert isinstance(root, TypeNode)
        assert (root.kind, root.label) == (TypeNodeKind.RECORD, "Wrap[int]")
        value, count = root.fields
        assert (value.name, value.doc, count.doc) == ("value", "The wrapped value.", None)
        assert isinstance(value.node, TypeNode) and value.node.kind is TypeNodeKind.INT


def _schema_of(entry: TypeTreeEntry | None) -> object:
    assert isinstance(entry, TypeNode)
    return _schema(entry)
