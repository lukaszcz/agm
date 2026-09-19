"""Behavior tests for one-level method selection over enum members and exceptions.

Settled design: `.agent-files/PLAN_inline_infer.md`. A member record's own
methods and its counted owning enum(s)' methods form one selection level
(S1); an exception's `extends` chain forms one level (S2); a same-named
field and visible method at that level are a static ambiguity (S3); a
selected enum method widens a member receiver to the enum, solving any
phantom (uncaptured) enum type parameter from arguments or the expected type,
or rejecting the call when one is left unresolved (S7 in the plan's §5
numbering). See ``tests/test_agl_method_selection_invariant.py`` for the
member<->enum and exception descendant<->ancestor invariant this design
settles; this file exercises the individual behaviors themselves.

Every pair of same-named methods at one selection level is declared across
different modules: a same-module pair is meant to be rejected at declaration
(§5.6, a separate, not-yet-implemented task), so call-site ambiguity can only
be observed today when each declaration is legal on its own.
"""

from __future__ import annotations

from pathlib import Path

from agm.agl.capabilities import HostCapabilities
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.scope import AglScopeError
from agm.agl.scope.program import resolve_program
from agm.agl.syntax.nodes import Block, FuncDef, Item, LetDecl
from agm.agl.typecheck import (
    AglTypeError,
    ArrayType,
    CheckedModule,
    CheckedProgram,
    FunctionType,
    IntType,
    TextType,
    Type,
    check_program,
)
from tests.agl.ir_harness import make_graph_from_files
from tests.agl.module_graph import resolve_and_check_repl_entry

_CAPS = HostCapabilities(
    supports_shell_exec=True,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset({"json", "record", "enum", "array", "dict", "int", "decimal", "bool"}),
    },
)


def accept_type(source: str) -> CheckedModule:
    return resolve_and_check_repl_entry(source, _CAPS)


def reject_type(source: str) -> AglTypeError | AglScopeError:
    try:
        accept_type(source)
    except (AglTypeError, AglScopeError) as exc:
        return exc
    raise AssertionError(f"expected {source!r} to be rejected")


def _check_program(tmp_path: Path, modules: dict[str, str]) -> CheckedProgram:
    """Build and typecheck a multi-module graph; returns the ``CheckedProgram``."""
    graph = make_graph_from_files(tmp_path, modules)
    return check_program(resolve_program(graph), _CAPS)


def reject_program(tmp_path: Path, modules: dict[str, str]) -> AglTypeError | AglScopeError:
    try:
        _check_program(tmp_path, modules)
    except (AglTypeError, AglScopeError) as exc:
        return exc
    raise AssertionError("expected the program to be rejected")


def _module_items(module: CheckedModule) -> tuple[Item, ...]:
    """Return an entry's test-only inline items, unwrapping the synthetic ``main``."""
    items = module.resolved.program.body.items
    if items and isinstance(items[-1], FuncDef) and items[-1].is_synthetic:
        return items[-1].body.items if isinstance(items[-1].body, Block) else (items[-1].body,)
    return items


def _final_type(checked: CheckedModule) -> Type:
    """The static type of the last top-level item's own expression."""
    last = _module_items(checked)[-1]
    node = last.value if isinstance(last, LetDecl) else last
    return checked.node_types[node.node_id]


def _program_final_type(checked: CheckedProgram) -> Type:
    module = checked.modules[ENTRY_ID]
    return _final_type(module)


# ---------------------------------------------------------------------------
# S1: enum methods callable on nullary members, including a stdlib enum
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
# S5: every form of member access -- bound projection, partial call, leading dot
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
# S4: generic widening and phantom-parameter resolution
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
    assert "Result" in repr(_final_type(checked))


def test_phantom_enum_parameter_left_unresolved_is_rejected() -> None:
    err = reject_type(_TREE + "Leaf.describe()")
    assert "infer" in str(err).lower()


def test_result_phantom_error_parameter_left_unresolved_is_rejected() -> None:
    err = reject_type("Ok(1).is-ok()")
    assert "infer" in str(err).lower()


def test_option_none_phantom_value_parameter_unresolved_call_is_rejected() -> None:
    """`None.is-some()` is a static error either way: unresolved `T`, or (as the

    stdlib currently stands) an ambiguity against `std/optional::Optional`,
    which also references `Option::None` as one of its own members (S1.5).
    Either is a legitimate rejection; only the error *class* is pinned.
    """
    reject_type("None.is-some()")


# ---------------------------------------------------------------------------
# S3: field vs. method kind clash, on read and on assignment
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
# S6: static methods are still not selectable with `.`
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
# S2: exception `extends` chains, cross-module pairs
# ---------------------------------------------------------------------------


def test_exception_chain_pair_is_ambiguous_on_the_descendant_and_selects_base_on_base(
    tmp_path: Path,
) -> None:
    modules = {
        "errors": (
            "exception Base extends Exception\n  code: int\nexception Derived extends Base()\n"
        ),
        "base_methods": "import errors::*\ndef Base::describe(self) -> int = self.code\n",
        "derived_methods": (
            "import errors::*\ndef Derived::describe(self) -> text = self.message\n"
        ),
    }
    ambiguous = reject_program(
        tmp_path,
        {
            **modules,
            "entry": (
                "import errors\nimport base_methods\nimport derived_methods\n"
                'errors::Derived(message = "bad", code = 1).describe()\n'
            ),
        },
    )
    assert "ambiguous" in str(ambiguous).lower()

    base_selects = _check_program(
        tmp_path,
        {
            **modules,
            "entry": (
                "import errors\nimport base_methods\nimport derived_methods\n"
                'let base: errors::Base = errors::Derived(message = "bad", code = 1)\n'
                "base.describe()\n"
            ),
        },
    )
    assert _program_final_type(base_selects) == IntType()


def test_exception_inherited_only_method_is_selected_from_the_descendant(
    tmp_path: Path,
) -> None:
    modules = {
        "errors": (
            "exception Base extends Exception\n  code: int\nexception Derived extends Base()\n"
        ),
        "methods": "import errors::*\ndef Base::status(self) -> int = self.code\n",
    }
    checked = _check_program(
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
# §4 cross-module orphan pair (palette/warm/cool), repaired by `hiding`
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
    checked = _check_program(
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


# ---------------------------------------------------------------------------
# §5.5 referenced record with a foreign referencing enum (lib/store)
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
    checked = _check_program(
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
    checked = _check_program(
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


def test_referenced_record_without_importing_the_referencing_enum_selects_its_own_method(
    tmp_path: Path,
) -> None:
    checked = _check_program(
        tmp_path,
        {**_LIB_STORE_MODULES, "entry": "import lib\nlib::Saved(id = 1).describe()\n"},
    )
    assert _program_final_type(checked) == TextType()
