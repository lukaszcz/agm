"""Behavior tests for nominal record constructor patterns."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.scope.program import resolve_program
from agm.agl.semantics.types import EnumType
from agm.agl.syntax.nodes import Case, ConstructorPattern, FuncDef, LetDecl
from agm.agl.typecheck import AglTypeError, CheckedProgram, check_program
from tests._agl_helpers import strip_decl_ids
from tests.agl.ir_harness import make_graph_from_files
from tests.agl.module_graph import resolve_and_check_inline_entry

_CAPS = HostCapabilities(
    supports_shell_exec=True,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset({"json", "record", "enum", "array", "dict", "int", "decimal", "bool"}),
    },
)


def accept(source: str):
    return resolve_and_check_inline_entry(source, _CAPS)


def reject(source: str) -> None:
    with pytest.raises(AglTypeError):
        accept(source)


def accept_graph(tmp_path: Path, modules: dict[str, str]) -> CheckedProgram:
    graph = make_graph_from_files(tmp_path, modules)
    return check_program(resolve_program(graph), _CAPS)


def reject_graph(tmp_path: Path, modules: dict[str, str]) -> None:
    with pytest.raises(AglTypeError):
        accept_graph(tmp_path, modules)


@pytest.mark.parametrize(
    "source",
    (
        "enum Tree[T]\n"
        "  | Node(value: T)\n"
        "let tree: Tree[int] = Node(value = 1)\n"
        "case tree of | Tree[text]::Node(value) => value | _ => 0",
        "enum E\n  | M\nlet value: E = M\ncase value of | E[Unknown]::M() => 0 | _ => 1",
    ),
)
def test_qualified_enum_constructor_patterns_validate_applied_owner_arguments(source: str) -> None:
    reject(source)


def test_referenced_enum_member_aliases_match_in_patterns_and_is_tests() -> None:
    accept(
        "record R\n"
        "  value: int\n"
        "type Alias = R\n"
        "enum E = ::Alias\n"
        "let value: E = R(value = 1)\n"
        "let matched = case value of | Alias(value) => value\n"
        "let tested = value is Alias\n"
        "matched"
    )


def test_generic_referenced_enum_member_patterns_validate_the_applied_owner() -> None:
    accept(
        "record R[T]\n"
        "  value: T\n"
        "enum E[T] = ::R[T]\n"
        "let value: E[int] = R(value = 1)\n"
        "case value of | E[int]::R(value) => value"
    )


def test_enum_alias_does_not_match_a_referenced_record_member() -> None:
    reject(
        "record R\n"
        "  value: int\n"
        "enum E = ::R\n"
        "type Alias = E\n"
        "let value: E = R(value = 1)\n"
        "case value of | Alias(value) => value"
    )


def test_simple_let_name_binds_even_when_it_matches_a_nullary_constructor() -> None:
    checked = accept("enum Opt\n  | none\nlet value: Opt = none\nlet none = value\nnone\n")
    let = checked.resolved.program.body.items[2]
    assert isinstance(let, LetDecl)
    assert let.name == "none"
    assert strip_decl_ids(checked.type_env.get_binding_type(let.node_id)) == EnumType("Opt")


def test_local_record_owner_qualifier_selects_only_the_local_same_named_record(
    tmp_path: Path,
) -> None:
    modules = {
        "lib": "record R\n  value: int\n",
        "entry": (
            "import lib::*\n"
            "record R\n  value: int\n"
            "type Alias = R\n"
            "enum Signal\n  | yes(value: int)\n"
            "def select-local(r: R) -> int = case r of | R::R(value) => value\n"
            "def select-alias(r: Alias) -> int = case r of | Alias::Alias(value) => value\n"
            "def select-enum(s: Signal) -> int = case s of | Signal::yes(value) => value\n"
        ),
    }

    accept_graph(tmp_path, modules)

    reject_graph(
        tmp_path,
        {
            "lib": modules["lib"],
            "entry": (
                "import lib::*\n"
                "record R\n  value: int\n"
                "def select-imported(r: lib::R) -> int = "
                "case r of | R::R(value) => value\n"
            ),
        },
    )


def test_module_qualified_pattern_rejects_wrong_phantom_generic_enum_owner(
    tmp_path: Path,
) -> None:
    reject_graph(
        tmp_path,
        {
            "lib": "enum Other[T]\n  | none\n",
            "entry": (
                "import lib\n"
                "enum Maybe[T]\n"
                "  | none\n"
                "case Maybe::none of\n"
                "  | lib::Other::none => ()\n"
                "  | _ => ()\n"
            ),
        },
    )


def test_module_qualified_record_pattern_accepts_each_referencing_enum_owner(
    tmp_path: Path,
) -> None:
    accept_graph(
        tmp_path,
        {
            "lib": "record Shared\n  value: int\nenum First = ::Shared\nenum Second = ::Shared\n",
            "entry": (
                "import lib\n"
                "let shared: lib::Shared = lib::Shared(value = 1)\n"
                "case shared of | lib::Second::Shared(value) => value\n"
            ),
        },
    )


def test_module_qualified_record_pattern_rejects_an_unrelated_enum_owner(
    tmp_path: Path,
) -> None:
    reject_graph(
        tmp_path,
        {
            "lib": (
                "record Shared\n"
                "  value: int\n"
                "enum First = ::Shared\n"
                "enum Second = ::Shared\n"
                "enum Unrelated | Other\n"
            ),
            "entry": (
                "import lib\n"
                "let shared: lib::Shared = lib::Shared(value = 1)\n"
                "case shared of | lib::Unrelated::Shared(value) => value\n"
            ),
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
    assert selected.owner_name == "Token"


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
        "record R\n"
        "  @arg-pos x: int\n"
        '  @arg-named label: text\nlet r = R(1, label = "x")\n'
        "case r of | R(x = _, label = _) => 0 | _ => 1",
        "exception Boom\n  value: int\nlet b = Boom(value = 1)\n"
        "case b of | Boom(value) => value | _ => 0",
        "enum Flag\n  | value\nrecord Holder\n  value: Flag\n"
        "let holder = Holder(value = value)\ncase holder of | Holder(value) => 0 | _ => 1",
    ),
)
def test_invalid_record_constructor_patterns_are_rejected(source: str) -> None:
    reject(source)


def test_unqualified_pattern_selects_a_nominal_declared_in_the_same_scope() -> None:
    """An unqualified pattern still selects its own scope's record.

    The pattern-to-nominal ownership check must key on the structured scope
    path, not just the bare spelling: ``Bounds(low, high)`` inside
    ``scope Config`` selects ``Config::Bounds``.
    """
    checked = accept(
        "scope Config\n"
        "  record Bounds\n"
        "    low: int\n"
        "    high: int\n"
        "  def pick(b: Bounds) -> int =\n"
        "    case b of\n"
        "    | Bounds(low, high) => low\n"
        "end Config\n"
        "\n"
        "()\n"
    )
    region = checked.resolved.program.body.items[0]
    pick = next(item for item in region.items if isinstance(item, FuncDef))
    case = pick.body.items[0]
    assert isinstance(case, Case)
    case_pattern = case.branches[0].pattern
    assert isinstance(case_pattern, ConstructorPattern)
    selected = checked.pattern_constructor_ref_for(case_pattern.node_id)
    assert selected is not None
    assert (selected.owner_path, selected.owner_name) == (("Config",), "Bounds")


def test_scoped_record_pattern_rejects_a_same_named_root_record() -> None:
    reject(
        "record Point\n"
        "  x: int\n"
        "\n"
        "scope A\n"
        "  record Point\n"
        "    label: text\n"
        "end A\n"
        "\n"
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
        "\n"
        "scope A\n"
        "  enum E\n"
        "    | Point(label: text)\n"
        "  def from-root(p: Point) -> int =\n"
        "    case p of\n"
        "    | Point(x) => x\n"
        "end A\n"
        "\n"
        "A::from-root(Point(x = 1))\n"
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
    assert selected.owner_name == "Retry"


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
        "  def helper(x: int) -> int = x\n"
        "  record Point\n"
        "    x: int\n"
        "end A\n"
        "\n"
        "let p: A::Point = A::Point(x = 1)\n"
        "case p of | A::helper(x) => x | _ => 0\n"
    )
