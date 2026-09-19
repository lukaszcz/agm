"""Invariant suite for one-level method selection across enum members and exceptions.

Settled design: `.agent-files/PLAN_inline_infer.md`. A member record's own
methods and the methods of its counted owning enum(s) form one selection
level (S1); an exception's `extends` chain forms one level (S2). The
invariant these tests pin (§4/§5.3): changing a receiver's *static* spelling
along the member->enum or exception-descendant->ancestor axis never changes
*which* function a call selects -- it may only turn a working call into a
type error, or vice versa. Enum-to-enum subset widening (`A` vs `B` where
`A subseteq B`) is explicitly outside this invariant.

Every case that pairs two same-named methods at one selection level declares
them in different modules: a same-module pair is rejected at declaration
(§5.6, a separate task), so the only way to observe the call-site ambiguity
this suite is about is to keep each declaration legal on its own.
"""

from __future__ import annotations

from pathlib import Path

from agm.agl.capabilities import HostCapabilities
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.scope.program import resolve_program
from agm.agl.scope.symbols import AglScopeError
from agm.agl.syntax.nodes import Block, Call, FuncDef, Item
from agm.agl.typecheck import AglTypeError, CheckedModule, CheckedProgram, TextType, check_program
from tests.agl.ir_harness import make_graph_from_files

_CAPS = HostCapabilities(
    supports_shell_exec=True,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset({"json", "record", "enum", "array", "dict", "int", "decimal", "bool"}),
    },
)


def _check_program(tmp_path: Path, modules: dict[str, str]) -> CheckedProgram:
    """Build and typecheck a multi-module graph; returns the ``CheckedProgram``."""
    graph = make_graph_from_files(tmp_path, modules)
    return check_program(resolve_program(graph), _CAPS)


def _module_items(module: CheckedModule) -> tuple[Item, ...]:
    """Return an entry's test-only inline items, unwrapping the synthetic ``main``."""
    items = module.resolved.program.body.items
    if items and isinstance(items[-1], FuncDef) and items[-1].is_synthetic:
        return items[-1].body.items if isinstance(items[-1].body, Block) else (items[-1].body,)
    return items


def _run_entry(
    tmp_path: Path, modules: dict[str, str], entry: str
) -> CheckedProgram | AglTypeError | AglScopeError:
    """Typecheck *entry* as the program's entry alongside *modules*; capture rejection."""
    try:
        return _check_program(tmp_path, {**modules, "entry": entry})
    except (AglTypeError, AglScopeError) as exc:
        return exc


def _final_call(checked: CheckedProgram) -> Call:
    """The entry's final top-level statement, asserted to be a call expression."""
    call = _module_items(checked.modules[ENTRY_ID])[-1]
    assert isinstance(call, Call), f"expected the entry to end in a call, got {call!r}"
    return call


def _selection_key(checked: CheckedProgram) -> tuple[object, ...]:
    """The declaration identity (module, scope path, name) the final call selected."""
    call = _final_call(checked)
    module = checked.modules[ENTRY_ID]
    selection = module.method_selections.get(call.callee.node_id)
    assert selection is not None, "expected the final call to select a method, not a field"
    return selection.declaration_key


def _assert_same_selection_or_an_error(
    first: CheckedProgram | AglTypeError | AglScopeError,
    second: CheckedProgram | AglTypeError | AglScopeError,
) -> None:
    """The core invariant: two receiver spellings never call different functions.

    Either both spellings typecheck and select the identical declaration, or
    at least one of them is rejected as a type/scope error.
    """
    if isinstance(first, (AglTypeError, AglScopeError)) or isinstance(
        second, (AglTypeError, AglScopeError)
    ):
        return
    assert _selection_key(first) == _selection_key(second)


# ---------------------------------------------------------------------------
# S1: enum member records widen into their owning enum's method level
# ---------------------------------------------------------------------------


def test_member_and_enum_orphan_pair_selects_consistently_or_errors(tmp_path: Path) -> None:
    """A member method and its enum's method, declared in different modules.

    Bare member access sees both (an ambiguity); the enum-typed binding sees
    only the enum's own method. One side always errors, so the invariant
    holds trivially either way.
    """
    modules = {
        "palette": "enum Color\n  | Red\n  | Blue\n",
        "warm": 'import palette::*\ndef Color::Red::label(self) -> text = "warm red"\n',
        "cool": 'import palette::*\ndef Color::label(self) -> text = "cool"\n',
    }
    bare = _run_entry(
        tmp_path, modules, "import palette\nimport warm\nimport cool\npalette::Color::Red.label()\n"
    )
    typed = _run_entry(
        tmp_path,
        modules,
        "import palette\nimport warm\nimport cool\n"
        "let c: palette::Color = palette::Color::Red\nc.label()\n",
    )
    _assert_same_selection_or_an_error(bare, typed)


def test_enum_only_method_is_reachable_from_both_member_and_enum_receivers(
    tmp_path: Path,
) -> None:
    """A method declared only on the enum is, per S1, part of every member's level too."""
    declarations = 'enum Color\n  | Red\n  | Blue\ndef Color::label(self) -> text = "color"\n'
    bare = _run_entry(tmp_path, {}, declarations + "Color::Red.label()\n")
    typed = _run_entry(tmp_path, {}, declarations + "let c: Color = Color::Red\nc.label()\n")
    _assert_same_selection_or_an_error(bare, typed)


def test_member_only_method_stays_unreachable_from_an_enum_receiver(tmp_path: Path) -> None:
    """S1: an enum-typed receiver sees only the enum's own methods -- unaffected by S1-S4.

    A member-only method is selectable on the member, but "no method" on the
    enum, both before and after the one-level rule lands.
    """
    declarations = (
        'enum Color\n  | Red\n  | Blue\ndef Color::Red::shade(self) -> text = "crimson"\n'
    )
    bare = _run_entry(tmp_path, {}, declarations + "Color::Red.shade()\n")
    assert isinstance(bare, CheckedProgram)
    assert bare.modules[ENTRY_ID].node_types[_final_call(bare).node_id] == TextType()

    typed = _run_entry(tmp_path, {}, declarations + "let c: Color = Color::Red\nc.shade()\n")
    assert isinstance(typed, AglTypeError)
    message = str(typed).lower()
    assert "method" in message or "field" in message


def test_overlapping_enum_membership_widens_a_shared_members_selection_level(
    tmp_path: Path,
) -> None:
    """S1.5: a member counted by two current enums carries both enums' methods.

    `A::X` is a member of both `A` (inline) and `B` (referenced), so a bare
    `A::X.f()` sees both `A::f` and `B::f` -- an ambiguity. A narrowly typed
    receiver sees only its own enum's method.
    """
    declarations = (
        "enum A\n  | X\n  | Y\n"
        "enum B = A::X | A::Y | Z\n"
        'def A::f(self) -> text = "a"\n'
        'def B::f(self) -> text = "b"\n'
    )
    bare = _run_entry(tmp_path, {}, declarations + "A::X.f()\n")
    typed = _run_entry(tmp_path, {}, declarations + "let a: A = X\na.f()\n")
    _assert_same_selection_or_an_error(bare, typed)


def test_referenced_record_gains_a_foreign_owning_enums_method(tmp_path: Path) -> None:
    """S1.5's accepted non-locality (§5.5): a record referenced by a foreign enum.

    `lib::Saved` gains `store::Stored`'s method once `store` is reachable, so
    a bare call is ambiguous between `lib`'s own method and `store`'s; an
    explicit upcast to `Stored` picks `store`'s method unambiguously.
    """
    modules = {
        "lib": 'record Saved(id: int)\ndef Saved::describe(self) -> text = "saved"\n',
        "store": (
            "import lib\nenum Stored = lib::Saved | Fresh(value: int)\n"
            'def Stored::describe(self) -> text = "stored"\n'
        ),
    }
    bare = _run_entry(
        tmp_path, modules, "import lib\nimport store\nlib::Saved(id = 1).describe()\n"
    )
    widened = _run_entry(
        tmp_path,
        modules,
        "import lib\nimport store\n(lib::Saved(id = 1) as store::Stored).describe()\n",
    )
    _assert_same_selection_or_an_error(bare, widened)


def test_generic_member_widens_with_its_captured_type_argument(tmp_path: Path) -> None:
    """S4: `Tree::Node[int]` widens to `Tree[int]`, matching an explicit `Tree[int]` binding."""
    declarations = (
        "enum Tree[T]\n  | Leaf\n  | Node(value: T)\n"
        "def Tree::describe[T](self) -> text =\n"
        "  case self of\n"
        '    | Leaf => "leaf"\n'
        '    | Node(value) => "node %{value}"\n'
    )
    bare = _run_entry(tmp_path, {}, declarations + "Node(value = 1).describe()\n")
    typed = _run_entry(
        tmp_path,
        {},
        declarations + "let t: Tree[int] = Node(value = 1)\nt.describe()\n",
    )
    _assert_same_selection_or_an_error(bare, typed)


def test_foreign_referencing_enums_method_requires_a_reachable_route(tmp_path: Path) -> None:
    """The non-locality of S1.5 is still route-gated, exactly like an orphan method.

    `wrap`'s enum method reaches `shapes::Point` only while `wrap` (or its
    route) is imported; `hiding` its path removes it again.
    """
    modules = {
        "shapes": "record Point\n  x: int\n",
        "wrap": (
            "import shapes\nenum Wrapped = shapes::Point | Empty\n"
            'def Wrapped::describe(self) -> text = "wrapped"\n'
        ),
    }
    visible = _run_entry(
        tmp_path, modules, "import shapes\nimport wrap\nshapes::Point(x = 1).describe()\n"
    )
    hidden = _run_entry(
        tmp_path,
        modules,
        "import shapes\nimport wrap hiding Wrapped::describe\nshapes::Point(x = 1).describe()\n",
    )
    assert isinstance(hidden, AglTypeError)
    assert "visible" in str(hidden).lower()
    if isinstance(visible, CheckedProgram):
        assert visible.modules[ENTRY_ID].node_types[_final_call(visible).node_id] == TextType()


# ---------------------------------------------------------------------------
# S2: exception `extends` chains form one level
# ---------------------------------------------------------------------------


def test_exception_chain_pair_selects_consistently_or_errors(tmp_path: Path) -> None:
    """A base method and a descendant method, declared in different modules.

    A descendant-typed receiver sees both (an ambiguity); a base-typed
    receiver sees only its own chain.
    """
    modules = {
        "errors": (
            "exception Base extends Exception\n  code: int\nexception Derived extends Base()\n"
        ),
        "base_methods": "import errors::*\ndef Base::describe(self) -> int = self.code\n",
        "derived_methods": (
            "import errors::*\ndef Derived::describe(self) -> text = self.message\n"
        ),
    }
    descendant = _run_entry(
        tmp_path,
        modules,
        "import errors\nimport base_methods\nimport derived_methods\n"
        'errors::Derived(message = "bad", code = 1).describe()\n',
    )
    base = _run_entry(
        tmp_path,
        modules,
        "import errors\nimport base_methods\nimport derived_methods\n"
        'let base: errors::Base = errors::Derived(message = "bad", code = 1)\n'
        "base.describe()\n",
    )
    _assert_same_selection_or_an_error(descendant, base)


def test_inherited_only_exception_method_is_selected_from_both_levels(tmp_path: Path) -> None:
    """A method with no override anywhere in the chain: descendant and base agree."""
    modules = {
        "errors": (
            "exception Base extends Exception\n  code: int\nexception Derived extends Base()\n"
        ),
        "methods": "import errors::*\ndef Base::status(self) -> int = self.code\n",
    }
    descendant = _run_entry(
        tmp_path,
        modules,
        'import errors\nimport methods\nerrors::Derived(message = "bad", code = 1).status()\n',
    )
    base = _run_entry(
        tmp_path,
        modules,
        "import errors\nimport methods\n"
        'let base: errors::Base = errors::Derived(message = "bad", code = 1)\n'
        "base.status()\n",
    )
    assert isinstance(descendant, CheckedProgram) and isinstance(base, CheckedProgram)
    assert _selection_key(descendant) == _selection_key(base)


# ---------------------------------------------------------------------------
# S3: field vs. method kind clash, checked across receiver widening
# ---------------------------------------------------------------------------


def test_field_vs_enum_method_kind_clash_across_receiver_widening(tmp_path: Path) -> None:
    """A member field and an owning enum's same-named method: ambiguous only on the member.

    An enum-typed receiver has no fields, so the enum method is selected
    cleanly (§4 "Kind clash" example).
    """
    declarations = (
        "enum Shape\n  | Circle(name: text, r: int)\n  | Square(name: text, s: int)\n"
        "def Shape::name(self) -> text =\n"
        "  case self of\n"
        "    | Circle(name) => name\n"
        "    | Square(name) => name\n"
    )
    member_read = _run_entry(
        tmp_path, {}, declarations + 'let c = Circle(name = "c", r = 1)\nc.name\n'
    )
    enum_call = _run_entry(
        tmp_path,
        {},
        declarations + 'let c = Circle(name = "c", r = 1)\nlet s: Shape = c\ns.name()\n',
    )
    assert isinstance(enum_call, CheckedProgram)
    assert enum_call.modules[ENTRY_ID].node_types[_final_call(enum_call).node_id] == TextType()
    if isinstance(member_read, (AglTypeError, AglScopeError)):
        message = str(member_read).lower()
        assert "field" in message or "ambiguous" in message


# ---------------------------------------------------------------------------
# Outside the invariant: enum-to-enum subset widening
# ---------------------------------------------------------------------------


def test_enum_to_enum_subset_widening_is_outside_the_invariant(tmp_path: Path) -> None:
    """§5.3: `A` and `B` are distinct nominal types even when `A subseteq B`.

    Unlike member->enum or descendant->ancestor, this axis is explicitly
    *not* covered by the invariant: the two typed receivers legitimately
    select different functions.
    """
    declarations = (
        "enum A\n  | X\n  | Y\n"
        "enum B = A::X | A::Y | Z\n"
        'def A::f(self) -> text = "a"\n'
        'def B::f(self) -> text = "b"\n'
    )
    as_a = _run_entry(tmp_path, {}, declarations + "let a: A = X\na.f()\n")
    as_b = _run_entry(tmp_path, {}, declarations + "let a: A = X\n(a as B).f()\n")
    assert isinstance(as_a, CheckedProgram) and isinstance(as_b, CheckedProgram)
    assert _selection_key(as_a) != _selection_key(as_b)
