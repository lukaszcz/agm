"""Type-directed externs: target type parameters and their per-occurrence resolution.

A target parameter is an extern type parameter no value parameter (receiver
included) mentions. Each direct or method call resolves it from explicit type
arguments or the expected type and records one strict JSON contract spec per
target parameter, in declaration order.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.parser import parse_program
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
from agm.agl.syntax.nodes import Block, Call, FuncDef, LetDecl
from agm.agl.typecheck import AglTypeError, CheckedModule, FunctionSignature, check_program
from agm.agl.typecheck.env import OutputContractSpec
from tests.agl.ir_harness import (
    base_caps,
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


def _let_call(checked: CheckedModule, binding: str) -> int:
    """Return the node id of the call initializing ``let <binding>``."""
    items = list(checked.resolved.program.body.items)
    while items:
        item = items.pop()
        if isinstance(item, FuncDef) and isinstance(item.body, Block):
            items.extend(item.body.items)
        elif isinstance(item, LetDecl) and item.name == binding:
            assert isinstance(item.value, Call)
            return item.value.node_id
    raise AssertionError(f"no binding {binding!r}")


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
