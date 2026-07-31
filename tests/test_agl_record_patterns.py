"""Behavior tests for nominal record constructor patterns."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.ir.ids import NominalId
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.parser import parse_program
from agm.agl.scope import resolve_module
from agm.agl.scope.program import resolve_program
from agm.agl.scope.symbols import BinderKind
from agm.agl.semantics.types import EnumType, IntType, RecordType
from agm.agl.syntax.nodes import AsPattern, Case, ConstructorPattern, FuncDef, LetDecl, VarPattern
from agm.agl.typecheck import AglTypeError, CheckedProgram, check_module, check_program
from tests.agl.ir_harness import make_graph_from_files

_CAPS = HostCapabilities(
    agent_names=frozenset(),
    has_default_agent=True,
    supports_shell_exec=True,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset({"json", "record", "enum", "array", "dict", "int", "decimal", "bool"}),
    },
)


def accept(source: str):
    return check_module(resolve_module(parse_program(source)), _CAPS)


def reject(source: str) -> None:
    with pytest.raises(AglTypeError):
        accept(source)


def accept_graph(tmp_path: Path, modules: dict[str, str]) -> CheckedProgram:
    graph = make_graph_from_files(tmp_path, modules)
    return check_program(resolve_program(graph), _CAPS)


def reject_graph(tmp_path: Path, modules: dict[str, str]) -> None:
    with pytest.raises(AglTypeError):
        accept_graph(tmp_path, modules)


def test_record_patterns_bind_positional_named_named_only_nested_and_as_in_case_and_let() -> None:
    checked = accept(
        "record Inner\n"
        "  value: int\n"
        "record Outer(a: int, /, inner: Inner, *, label: text)\n"
        'let outer: Outer = Outer(1, Inner(value = 2), label = "ok")\n'
        "let Outer(a, inner = Inner(value = _ as value) as whole, label = label) = outer\n"
        "case outer of\n"
        "  | Outer(a, Inner(value), label) as subject => a + value\n"
        "  | _ => 0\n"
    )

    let = checked.resolved.program.body.items[3]
    assert isinstance(let, LetDecl)
    assert isinstance(let.pattern, ConstructorPattern)
    (first,) = let.pattern.positional
    fields = {field.name: field.pattern for field in let.pattern.named}
    nested = fields["inner"]
    label = fields["label"]
    assert isinstance(first, VarPattern)
    assert isinstance(nested, AsPattern)
    assert isinstance(label, VarPattern)
    assert checked.type_env.get_binding_type(first.node_id) == IntType()
    assert checked.type_env.get_binding_type(label.node_id).kind == "text"
    assert checked.type_env.get_binding_type(nested.node_id) == RecordType("Inner")
    assert checked.pattern_binding_for(first.node_id).kind is BinderKind.let_binding
    assert checked.pattern_constructor_ref_for(let.pattern.node_id) is not None

    case = checked.resolved.program.body.items[4]
    branch = case.branches[0]
    assert isinstance(branch.pattern, AsPattern)
    assert isinstance(branch.pattern.pattern, ConstructorPattern)
    assert checked.pattern_constructor_ref_for(branch.pattern.pattern.node_id) is not None


def test_generic_record_alias_pattern_uses_concrete_field_type_and_selected_constructor() -> None:
    checked = accept(
        "record Box[T]\n"
        "  value: T\n"
        "type Alias[T] = Box[T]\n"
        "type Phantom[T] = Box[int]\n"
        "let box: Alias[int] = Box(value = 1)\n"
        "let Alias(value = _ as value) = box\n"
        "let ::Alias(value = _ as self_value) = box\n"
        "let Phantom(value = _ as phantom_value) = box\n"
        "value + self_value + phantom_value\n"
    )

    let = checked.resolved.program.body.items[4]
    assert isinstance(let, LetDecl)
    assert isinstance(let.pattern, ConstructorPattern)
    (field,) = let.pattern.named
    assert isinstance(field.pattern, AsPattern)
    value = field.pattern
    assert checked.type_env.get_binding_type(value.node_id) == IntType()
    constructor = checked.pattern_constructor_ref_for(let.pattern.node_id)
    assert constructor is not None
    assert constructor.owner_name == "Alias"
    assert checked.pattern_constructor_owner_for(let.pattern.node_id) == NominalId(ENTRY_ID, "Box")


def test_generic_record_patterns_publish_owner_without_type_arguments() -> None:
    checked = accept(
        "record Box[T]\n"
        "  value: T\n"
        "let int_box: Box[int] = Box(value = 1)\n"
        "let Box(value = _) = int_box\n"
        'let text_box: Box[text] = Box(value = "x")\n'
        "let Box(value = _) = text_box\n"
        "int_box\n"
    )
    pattern_lets = [
        item
        for item in checked.resolved.program.body.items
        if isinstance(item, LetDecl) and isinstance(item.pattern, ConstructorPattern)
    ]
    assert len(pattern_lets) == 2
    assert [checked.pattern_constructor_owner_for(let.pattern.node_id) for let in pattern_lets] == [
        NominalId(ENTRY_ID, "Box"),
        NominalId(ENTRY_ID, "Box"),
    ]


def test_simple_let_name_binds_even_when_it_matches_a_nullary_constructor() -> None:
    checked = accept("enum Opt\n  | none\nlet value: Opt = none\nlet none = value\nnone\n")
    let = checked.resolved.program.body.items[2]
    assert isinstance(let, LetDecl)
    assert isinstance(let.pattern, VarPattern)
    assert checked.pattern_classifications[let.pattern.node_id] is None
    assert checked.type_env.get_binding_type(let.pattern.node_id) == EnumType("Opt")


def test_record_patterns_support_imported_and_qualified_alias_spellings(tmp_path: Path) -> None:
    checked = accept_graph(
        tmp_path,
        {
            "lib": "record Box[T]\n  value: T\ntype Alias[T] = Box[T]\n",
            "entry": (
                "import lib\n"
                "import lib as L\n"
                "open import lib\n"
                "record Local\n  value: int\n"
                "let local = Local(value = 1)\n"
                "let ::Local(value) = local\n"
                "let direct: lib::Alias[int] = lib::Box(value = 2)\n"
                "let lib::Alias(value = _ as direct_value) = direct\n"
                "let renamed: L::Alias[int] = L::Box(value = 3)\n"
                "let L::Alias(value = _ as renamed_value) = renamed\n"
                "let opened: Alias[int] = Box(value = 4)\n"
                "let Alias(value = _ as opened_value) = opened\n"
                "let generic: lib::Box[int] = lib::Box(value = 5)\n"
                "let lib::Box(value = _ as generic_value) = generic\n"
                "direct_value + renamed_value + opened_value + generic_value\n"
            ),
        },
    )

    entry = checked.modules[ENTRY_ID]
    pattern_lets = [
        item
        for item in entry.resolved.program.body.items
        if isinstance(item, LetDecl) and isinstance(item.pattern, ConstructorPattern)
    ]
    assert len(pattern_lets) == 5
    for let in pattern_lets:
        assert entry.pattern_constructor_ref_for(let.pattern.node_id) is not None


def test_local_record_owner_qualifier_selects_only_the_local_same_named_record(
    tmp_path: Path,
) -> None:
    modules = {
        "lib": "record R\n  value: int\n",
        "entry": (
            "open import lib\n"
            "record R\n  value: int\n"
            "type Alias = R\n"
            "enum Signal\n  | yes(value: int)\n"
            "def select_local(r: R) -> int = case r of | R::R(value) => value\n"
            "def select_alias(r: Alias) -> int = case r of | Alias::Alias(value) => value\n"
            "def select_enum(s: Signal) -> int = case s of | Signal::yes(value) => value\n"
        ),
    }

    accept_graph(tmp_path, modules)

    reject_graph(
        tmp_path,
        {
            "lib": modules["lib"],
            "entry": (
                "open import lib\n"
                "record R\n  value: int\n"
                "def select_imported(r: lib::R) -> int = "
                "case r of | R::R(value) => value\n"
            ),
        },
    )


def test_module_qualified_record_pattern_rejects_an_absent_named_owner(tmp_path: Path) -> None:
    reject_graph(
        tmp_path,
        {
            "lib": "record Other\n  x: int\n",
            "entry": (
                "import lib\n"
                "record Point\n"
                "  x: int\n"
                "let point = Point(x = 1)\n"
                "let lib::Point(x) = point\n"
                "x\n"
            ),
        },
    )


def test_self_qualified_record_pattern_rejects_an_absent_current_owner(tmp_path: Path) -> None:
    reject_graph(
        tmp_path,
        {
            "lib": "record Point\n  x: int\n",
            "entry": ("open import lib\nlet point = Point(x = 1)\nlet ::Point(x) = point\nx\n"),
        },
    )


def test_record_and_enum_constructor_spelling_collision_is_scrutinee_directed() -> None:
    checked = accept(
        "record Token\n"
        "  value: int\n"
        "enum Signal\n"
        "  | Token\n"
        "def select(token: Token) -> int = case token of | Token(value) => value | _ => 0\n"
    )

    func = checked.resolved.program.body.items[-1]
    assert isinstance(func, FuncDef)
    case = func.body
    assert isinstance(case, Case)
    pattern = case.branches[0].pattern
    assert isinstance(pattern, ConstructorPattern)
    selected = checked.pattern_constructor_ref_for(pattern.node_id)
    assert selected is not None
    assert selected.variant is None


@pytest.mark.parametrize(
    "source",
    (
        "record A\n  value: int\nrecord B\n  value: int\nlet a = A(value = 1)\n"
        "case a of | B(value) => value | _ => 0",
        "record Point\n  x: int\ncase 1 of | Point(x) => x | _ => 0",
        "record Point\n  x: int\nlet p = Point(x = 1)\n"
        "case p of | Point(missing = _) => 0 | _ => 1",
        "record Point\n  x: int\nlet p = Point(x = 1)\n"
        "case p of | Point(x = _, x = _) => 0 | _ => 1",
        "record Box[T]\n  value: T\ntype IntBox = Box[int]\n"
        'let box: Box[text] = Box(value = "x")\n'
        'case box of | IntBox(value) => value | _ => ""',
        'record R(x: int, /, *, label: text)\nlet r = R(1, label = "x")\n'
        "case r of | R(x = _, label = _) => 0 | _ => 1",
        "exception Boom\n  value: int\nlet b = Boom(value = 1)\n"
        "case b of | Boom(value) => value | _ => 0",
        "enum Flag\n  | value\nrecord Holder\n  value: Flag\n"
        "let holder = Holder(value = value)\ncase holder of | Holder(value) => 0 | _ => 1",
    ),
)
def test_invalid_record_constructor_patterns_are_rejected(source: str) -> None:
    reject(source)


def test_scoped_record_pattern_selects_its_scope_member() -> None:
    """``A::Point(x)`` destructures the record declared in named scope ``A``."""
    checked = accept(
        "scope A\n"
        "record Point\n"
        "  x: int\n"
        "end A\n"
        "let p: A::Point = A::Point(x = 4)\n"
        "let A::Point(x) = p\n"
        "x\n"
    )

    let_decl = checked.resolved.program.body.items[-2]
    assert isinstance(let_decl, LetDecl)
    assert isinstance(let_decl.pattern, ConstructorPattern)
    selected = checked.pattern_constructor_ref_for(let_decl.pattern.node_id)
    assert selected is not None
    assert (selected.owner_path, selected.owner_name) == (("A",), "Point")


def test_unqualified_pattern_selects_a_nominal_declared_in_the_same_scope() -> None:
    """An unqualified pattern still selects its own scope's record.

    The pattern-to-nominal ownership check must key on the structured scope
    path, not just the bare spelling: ``Bounds(low, high)`` inside
    ``scope Config`` selects ``Config::Bounds``, in both a ``case`` arm and a
    destructuring ``let``, exactly as the qualified spelling would.
    """
    checked = accept(
        "scope Config\n"
        "record Bounds(low: int, high: int)\n"
        "def pick(b: Bounds) -> int =\n"
        "  case b of\n"
        "  | Bounds(low, high) => low\n"
        "def unpick(b: Bounds) -> int =\n"
        "  let Bounds(low, high) = b\n"
        "  high\n"
        "end Config\n"
        "()\n"
    )
    region = checked.resolved.program.body.items[0]
    pick, unpick = (item for item in region.items if isinstance(item, FuncDef))
    case = pick.body.items[0]
    assert isinstance(case, Case)
    case_pattern = case.branches[0].pattern
    assert isinstance(case_pattern, ConstructorPattern)
    let_decl = unpick.body.items[0]
    assert isinstance(let_decl, LetDecl)
    assert isinstance(let_decl.pattern, ConstructorPattern)
    for pattern in (case_pattern, let_decl.pattern):
        selected = checked.pattern_constructor_ref_for(pattern.node_id)
        assert selected is not None
        assert (selected.owner_path, selected.owner_name) == (("Config",), "Bounds")


def test_scoped_record_pattern_rejects_a_same_named_root_record() -> None:
    reject(
        "record Point\n"
        "  x: int\n"
        "scope A\n"
        "record Point\n"
        "  label: text\n"
        "end A\n"
        "let p = Point(x = 1)\n"
        "case p of | A::Point(label) => 0 | _ => 1\n"
    )


def test_bare_pattern_in_a_region_is_shadowed_by_its_own_scoped_variant() -> None:
    """A same-named scoped variant shadows a root nominal for a bare pattern.

    Nearest-layer precedence is deliberate and deterministic: inside
    ``scope A``, a bare ``Point`` pattern selects ``A``'s own ``E::Point``
    variant candidate, never falling outward to the root ``Point`` record,
    so matching it against a root-typed scrutinee is a genuine mismatch.
    """
    reject(
        "record Point\n"
        "  x: int\n"
        "scope A\n"
        "enum E\n"
        "  | Point(label: text)\n"
        "def from_root(p: Point) -> int =\n"
        "  case p of\n"
        "  | Point(x) => x\n"
        "end A\n"
        "A::from_root(Point(x = 1))\n"
    )


def test_self_qualified_pattern_reaches_a_prelude_constructor() -> None:
    """``::Retry`` names the prelude variant the current module can see."""
    checked = accept(
        "let policy: ParsePolicy = Retry(n = 2)\ncase policy of | ::Retry(n) => n | _ => 0\n"
    )

    case = checked.resolved.program.body.items[-1]
    assert isinstance(case, Case)
    pattern = case.branches[0].pattern
    assert isinstance(pattern, ConstructorPattern)
    selected = checked.pattern_constructor_ref_for(pattern.node_id)
    assert selected is not None
    assert selected.variant == "Retry"


def test_route_qualified_pattern_naming_a_non_constructor_is_rejected(tmp_path: Path) -> None:
    """``lib::helper(x)`` reaches a function, not a constructor, so it cannot match."""
    reject_graph(
        tmp_path,
        {
            "lib": "def helper(x: int) -> int = x\nrecord R\n  x: int\n",
            "entry": (
                "import lib\nlet r = lib::R(x = 1)\ncase r of | lib::helper(x) => 0 | _ => 1\n"
            ),
        },
    )


def test_scoped_pattern_naming_a_non_constructor_member_is_rejected() -> None:
    """``A::helper(x)`` reaches a scope member that owns no constructor."""
    reject(
        "scope A\n"
        "def helper(x: int) -> int = x\n"
        "record Point\n"
        "  x: int\n"
        "end A\n"
        "let p: A::Point = A::Point(x = 1)\n"
        "case p of | A::helper(x) => x | _ => 0\n"
    )
