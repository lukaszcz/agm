"""Behavior tests for one-level method selection over enum members and exceptions.

A member record's own methods and its counted owning enum(s)' methods form
one selection level; an exception's `extends` chain forms one level; a
same-named field and visible method at that level are a static ambiguity; a
selected enum method widens a member receiver to the enum, solving any
phantom (uncaptured) enum type parameter from arguments or the expected type,
or rejecting the call when one is left unresolved. See
``tests/test_agl_method_selection_invariant.py`` for the member<->enum and
exception descendant<->ancestor invariant this design settles; this file
exercises the individual behaviors themselves.

Every pair of same-named methods at one selection level is declared across
different modules, since a same-module pair is rejected at declaration; so
call-site ambiguity can only be observed when each declaration is legal on
its own.
"""

from __future__ import annotations

from pathlib import Path

from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.scope import AglScopeError
from agm.agl.syntax.nodes import LetDecl
from agm.agl.typecheck import (
    AglTypeError,
    ArrayType,
    CheckedModule,
    CheckedProgram,
    EnumType,
    FunctionType,
    IntType,
    TextType,
    Type,
)
from tests._agl_helpers import (
    AGL_TEST_CAPS,
    check_agl_program,
    checked_module_items,
    checked_program_selection_key,
)
from tests.agl.module_graph import resolve_and_check_repl_entry


def accept_type(source: str) -> CheckedModule:
    return resolve_and_check_repl_entry(source, AGL_TEST_CAPS)


def reject_type(source: str) -> AglTypeError | AglScopeError:
    try:
        accept_type(source)
    except (AglTypeError, AglScopeError) as exc:
        return exc
    raise AssertionError(f"expected {source!r} to be rejected")


def reject_program(tmp_path: Path, modules: dict[str, str]) -> AglTypeError | AglScopeError:
    try:
        check_agl_program(tmp_path, modules)
    except (AglTypeError, AglScopeError) as exc:
        return exc
    raise AssertionError("expected the program to be rejected")


def _final_type(checked: CheckedModule) -> Type:
    """The static type of the last top-level item's own expression."""
    last = checked_module_items(checked)[-1]
    node = last.value if isinstance(last, LetDecl) else last
    return checked.node_types[node.node_id]


def _program_final_type(checked: CheckedProgram) -> Type:
    module = checked.modules[ENTRY_ID]
    return _final_type(module)


# ---------------------------------------------------------------------------
# Enum methods are callable on nullary members, including a stdlib enum
# ---------------------------------------------------------------------------

_COLOR = 'enum Color\n  | Red\n  | Blue\ndef Color::label(self) -> text = "color"\n'


def test_enum_method_is_callable_on_a_nullary_member() -> None:
    checked = accept_type(_COLOR + "Color::Red.label()")
    assert _final_type(checked) == TextType()


def test_stdlib_enum_method_is_callable_on_its_nullary_member() -> None:
    checked = accept_type("import std/http\nhttp::Method::Get.render()")
    assert _final_type(checked) == TextType()


def test_hidden_stdlib_enum_method_route_is_not_visible_on_a_member() -> None:
    err = reject_type("import std/http hiding Method::render\nhttp::Method::Get.render()")
    assert "visible" in str(err).lower()


def test_member_only_method_is_rejected_on_an_enum_typed_receiver() -> None:
    err = reject_type(
        'enum Color\n  | Red\n  | Blue\ndef Color::Red::shade(self) -> text = "crimson"\n'
        "let c: Color = Color::Red\nc.shade()"
    )
    message = str(err).lower()
    assert "method" in message or "field" in message


# ---------------------------------------------------------------------------
# Every form of member access: bound projection, partial call, leading dot
# ---------------------------------------------------------------------------


def test_bound_projection_of_an_enum_method_on_a_member_has_a_nullary_function_type() -> None:
    checked = accept_type(_COLOR + "let f = Color::Blue.label\nf")
    assert _final_type(checked) == FunctionType((), TextType())


def test_partial_application_of_an_enum_method_on_a_member() -> None:
    checked = accept_type(
        _COLOR + 'def Color::tag(self, suffix: text) -> text = "color" ++ suffix\n'
        "let g: (text) -> text = Color::Blue.tag(?)\ng"
    )
    assert _final_type(checked) == FunctionType((TextType(),), TextType())


def test_leading_dot_enum_method_call_inside_map_over_members() -> None:
    checked = accept_type(_COLOR + "let reds: array[Color::Red] = [Color::Red]\nreds.map(.label())")
    assert _final_type(checked) == ArrayType(TextType())


# ---------------------------------------------------------------------------
# Generic widening and phantom-parameter resolution
# ---------------------------------------------------------------------------

_TREE = (
    "enum Tree[T]\n  | Leaf\n  | Node(value: T)\n"
    "def Tree::describe[T](self) -> text =\n"
    "  case self of\n"
    '    | Leaf => "leaf"\n'
    '    | Node(value) => "node %{value}"\n'
    "def Tree::or-default[T](self, fallback: T) -> T =\n"
    "  case self of\n"
    "    | Leaf => fallback\n"
    "    | Node(value) => value\n"
)


def test_generic_member_widens_with_its_own_captured_type_argument() -> None:
    checked = accept_type(_TREE + "Node(value = 1).describe()")
    assert _final_type(checked) == TextType()


def test_phantom_enum_parameter_is_solved_by_a_method_argument() -> None:
    checked = accept_type(_TREE + "Leaf.or-default(4)")
    assert _final_type(checked) == IntType()


def test_phantom_enum_parameter_is_solved_by_the_expected_result_type() -> None:
    checked = accept_type("let r: Result[int, text] = Ok(1).map(fn(x: int) => x * 2)\nr")
    result_type = _final_type(checked)
    assert isinstance(result_type, EnumType)
    assert result_type.name == "Result"
    assert result_type.type_args == (IntType(), TextType())


def test_phantom_enum_parameter_left_unresolved_is_rejected() -> None:
    err = reject_type(_TREE + "Leaf.describe()")
    assert "infer" in str(err).lower()


def test_generic_functions_rigid_type_variable_is_not_mistaken_for_a_phantom() -> None:
    """A generic caller's own rigid type variable, coincidentally spelled

    like the enum's own declared parameter, must still bind that parameter:
    it is captured, not phantom, so the payload's real type stays linked to
    the caller's `T` and cannot be solved independently from an unrelated
    argument.
    """
    err = reject_type(
        _TREE + "def g[T](n: Tree::Node[T]) -> int = n.or-default(5) + 1\ng(Node(value = 1))"
    )
    assert isinstance(err, AglTypeError)


def test_generic_functions_rigid_type_variable_widens_normally_for_a_method_with_no_phantom() -> (
    None
):
    checked = accept_type(
        _TREE + "def f[T](n: Tree::Node[T]) -> text = n.describe()\nf(Node(value = 1))"
    )
    assert _final_type(checked) == TextType()


def test_result_phantom_error_parameter_left_unresolved_is_rejected() -> None:
    err = reject_type("Ok(1).is-ok()")
    assert "infer" in str(err).lower()


def test_option_none_member_method_is_ambiguous_with_optional() -> None:
    """`Option::None` is also a member of `std/optional::Optional`, so its

    selection level carries both enums' `is-some`, an accepted ambiguity.
    """
    err = reject_type("None.is-some()")
    assert "ambiguous" in str(err).lower()


# ---------------------------------------------------------------------------
# Field vs. method kind clash, on read and on assignment
# ---------------------------------------------------------------------------

_SHAPE = (
    "enum Shape\n  | Circle(var name: text, r: int)\n  | Square(var name: text, s: int)\n"
    "def Shape::name(self) -> text =\n"
    "  case self of\n"
    "    | Circle(name) => name\n"
    "    | Square(name) => name\n"
)


def test_field_and_owning_enum_method_clash_on_read() -> None:
    err = reject_type(_SHAPE + 'Circle(name = "c", r = 1).name')
    assert "field" in str(err).lower()


def test_field_and_owning_enum_method_clash_on_assignment() -> None:
    err = reject_type(_SHAPE + 'let c = Circle(name = "c", r = 1)\nc.name := "d"')
    assert "field" in str(err).lower()


def test_enum_typed_receiver_has_no_fields_so_the_method_is_selected_cleanly() -> None:
    checked = accept_type(_SHAPE + 'let c = Circle(name = "c", r = 1)\nlet s: Shape = c\ns.name()')
    assert _final_type(checked) == TextType()


# ---------------------------------------------------------------------------
# Static methods are still not selectable with `.`
# ---------------------------------------------------------------------------


def test_static_method_is_not_selectable_on_a_member_with_dot() -> None:
    err = reject_type(
        "enum Color\n  | Red\n  | Blue\ndef Color::make() -> Color = Red\nColor::Red.make()"
    )
    message = str(err).lower()
    assert "field" in message or "method" in message


# ---------------------------------------------------------------------------
# `with` keeps a member's own record type; its enum's methods stay callable
# ---------------------------------------------------------------------------


def test_with_result_on_a_member_keeps_its_member_type_and_enum_methods() -> None:
    checked = accept_type(
        'enum Color\n  | Red(shade: text)\n  | Blue\ndef Color::label(self) -> text = "color"\n'
        'let c = Red(shade = "dark")\nlet c2 = c with shade = "light"\nc2.label()'
    )
    assert _final_type(checked) == TextType()


# ---------------------------------------------------------------------------
# Agent builtin methods still typecheck through the general member path
# ---------------------------------------------------------------------------


def test_agent_ask_still_typechecks_on_a_member_through_the_general_path() -> None:
    checked = accept_type('AgentCommand("reviewer").ask("Q")')
    assert _final_type(checked) == TextType()


# ---------------------------------------------------------------------------
# Exception `extends` chains, cross-module pairs
# ---------------------------------------------------------------------------


def test_exception_inherited_only_method_is_selected_from_the_descendant(
    tmp_path: Path,
) -> None:
    modules = {
        "errors": (
            "exception Base extends Exception\n  code: int\nexception Derived extends Base()\n"
        ),
        "methods": "import errors::*\ndef Base::status(self) -> int = self.code\n",
    }
    checked = check_agl_program(
        tmp_path,
        {
            **modules,
            "entry": (
                "import errors\nimport methods\n"
                'errors::Derived(message = "bad", code = 1).status()\n'
            ),
        },
    )
    assert _program_final_type(checked) == IntType()


# ---------------------------------------------------------------------------
# Cross-module orphan pair (palette/warm/cool), repaired by `hiding`
# ---------------------------------------------------------------------------

_PALETTE_MODULES = {
    "palette": "enum Color\n  | Red\n  | Blue\n",
    "warm": 'import palette::*\ndef Color::Red::label(self) -> text = "warm red"\n',
    "cool": 'import palette::*\ndef Color::label(self) -> text = "cool"\n',
}


def test_cross_module_orphan_pair_is_ambiguous_at_the_call_site(tmp_path: Path) -> None:
    err = reject_program(
        tmp_path,
        {
            **_PALETTE_MODULES,
            "entry": ("import palette\nimport warm\nimport cool\npalette::Color::Red.label()\n"),
        },
    )
    assert "ambiguous" in str(err).lower()


def test_hiding_one_orphans_route_repairs_the_ambiguity(tmp_path: Path) -> None:
    checked = check_agl_program(
        tmp_path,
        {
            **_PALETTE_MODULES,
            "entry": (
                "import palette\nimport warm hiding Color::Red::label\nimport cool\n"
                "palette::Color::Red.label()\n"
            ),
        },
    )
    assert _program_final_type(checked) == TextType()
    assert checked_program_selection_key(checked) == (
        ModuleId.from_path("cool"),
        ("Color",),
        "label",
    )


# ---------------------------------------------------------------------------
# A referenced record with a foreign referencing enum (lib/store)
# ---------------------------------------------------------------------------

_LIB_STORE_MODULES = {
    "lib": 'record Saved(id: int)\ndef Saved::describe(self) -> text = "saved"\n',
    "store": (
        "import lib\nenum Stored = lib::Saved | Fresh(value: int)\n"
        'def Stored::describe(self) -> text = "stored"\n'
    ),
}


def test_referenced_record_method_is_ambiguous_once_the_referencing_enum_is_imported(
    tmp_path: Path,
) -> None:
    err = reject_program(
        tmp_path,
        {
            **_LIB_STORE_MODULES,
            "entry": "import lib\nimport store\nlib::Saved(id = 1).describe()\n",
        },
    )
    assert "ambiguous" in str(err).lower()


def test_referenced_record_ambiguity_is_repaired_by_a_qualified_call(tmp_path: Path) -> None:
    checked = check_agl_program(
        tmp_path,
        {
            **_LIB_STORE_MODULES,
            "entry": (
                "import lib\nimport store\nlet s = lib::Saved(id = 1)\nlib::Saved::describe(s)\n"
            ),
        },
    )
    assert _program_final_type(checked) == TextType()


def test_referenced_record_ambiguity_is_repaired_by_widening_the_receiver(
    tmp_path: Path,
) -> None:
    checked = check_agl_program(
        tmp_path,
        {
            **_LIB_STORE_MODULES,
            "entry": (
                "import lib\nimport store\n"
                "let s = lib::Saved(id = 1)\n(s as store::Stored).describe()\n"
            ),
        },
    )
    assert _program_final_type(checked) == TextType()
    assert checked_program_selection_key(checked) == (
        ModuleId.from_path("store"),
        ("Stored",),
        "describe",
    )


def test_referenced_record_without_importing_the_referencing_enum_selects_its_own_method(
    tmp_path: Path,
) -> None:
    checked = check_agl_program(
        tmp_path,
        {**_LIB_STORE_MODULES, "entry": "import lib\nlib::Saved(id = 1).describe()\n"},
    )
    assert _program_final_type(checked) == TextType()
    assert checked_program_selection_key(checked) == (
        ModuleId.from_path("lib"),
        ("Saved",),
        "describe",
    )
