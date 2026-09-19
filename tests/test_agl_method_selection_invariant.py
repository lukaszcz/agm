"""Invariant suite for one-level method selection across enum members and exceptions.

A member record's own methods and the methods of its counted owning enum(s)
form one selection level; an exception's `extends` chain forms one level.
The invariant these tests pin: changing a receiver's *static* spelling along
the member->enum or exception-descendant->ancestor axis never changes
*which* function a call selects -- it may only turn a working call into a
type error, or vice versa. Enum-to-enum subset widening (`A` vs `B` where
`A subseteq B`) is explicitly outside this invariant.

Every case that pairs two same-named methods at one selection level declares
them in different modules, since a same-module pair is rejected at
declaration, so the only way to observe the call-site ambiguity this suite
is about is to keep each declaration legal on its own.
"""

from __future__ import annotations

from pathlib import Path

from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.scope.symbols import AglScopeError
from agm.agl.typecheck import AglTypeError, CheckedProgram, TextType
from tests._agl_helpers import check_agl_program, checked_program_selection_key, final_entry_call

Selectable = CheckedProgram | AglTypeError | AglScopeError


def _run_entry(tmp_path: Path, modules: dict[str, str], entry: str) -> Selectable:
    """Typecheck *entry* as the program's entry alongside *modules*; capture rejection."""
    try:
        return check_agl_program(tmp_path, {**modules, "entry": entry})
    except (AglTypeError, AglScopeError) as exc:
        return exc


def _assert_selects(
    result: Selectable, module: ModuleId, scope_path: tuple[str, ...], name: str
) -> None:
    """Assert *result* is a successful check that selected the named declaration."""
    assert isinstance(result, CheckedProgram), f"expected a successful check, got {result!r}"
    assert checked_program_selection_key(result) == (module, scope_path, name)


def _assert_rejected(result: Selectable, keyword: str) -> None:
    """Assert *result* is a type/scope error whose message contains *keyword*."""
    assert isinstance(result, (AglTypeError, AglScopeError)), (
        f"expected a rejection, got {result!r}"
    )
    assert keyword in str(result).lower()


# ---------------------------------------------------------------------------
# Enum member records widen into their owning enum's method level
# ---------------------------------------------------------------------------


def test_member_and_enum_orphan_pair_selects_consistently_or_errors(tmp_path: Path) -> None:
    """A member method and its enum's method, declared in different modules.

    Bare member access sees both -- an ambiguity. Both an enum-typed binding
    and an identity upcast to the enum see only the enum's own method.
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
    upcast = _run_entry(
        tmp_path,
        modules,
        "import palette\nimport warm\nimport cool\n"
        "(palette::Color::Red as palette::Color).label()\n",
    )
    _assert_rejected(bare, "ambiguous")
    cool_label = (ModuleId.from_path("cool"), ("Color",), "label")
    _assert_selects(typed, *cool_label)
    _assert_selects(upcast, *cool_label)


def test_enum_only_method_is_reachable_from_both_member_and_enum_receivers(
    tmp_path: Path,
) -> None:
    """A method declared only on the enum is reachable from a member's level too."""
    declarations = 'enum Color\n  | Red\n  | Blue\ndef Color::label(self) -> text = "color"\n'
    bare = _run_entry(tmp_path, {}, declarations + "Color::Red.label()\n")
    typed = _run_entry(tmp_path, {}, declarations + "let c: Color = Color::Red\nc.label()\n")
    upcast = _run_entry(tmp_path, {}, declarations + "(Color::Red as Color).label()\n")
    label = (ENTRY_ID, ("Color",), "label")
    _assert_selects(bare, *label)
    _assert_selects(typed, *label)
    _assert_selects(upcast, *label)


def test_member_only_method_stays_unreachable_from_an_enum_receiver(tmp_path: Path) -> None:
    """An enum-typed receiver sees only the enum's own methods.

    A member-only method is selectable on the member, but "no method" on
    both an enum-typed binding and an identity upcast to the enum.
    """
    declarations = (
        'enum Color\n  | Red\n  | Blue\ndef Color::Red::shade(self) -> text = "crimson"\n'
    )
    bare = _run_entry(tmp_path, {}, declarations + "Color::Red.shade()\n")
    _assert_selects(bare, ENTRY_ID, ("Color", "Red"), "shade")

    typed = _run_entry(tmp_path, {}, declarations + "let c: Color = Color::Red\nc.shade()\n")
    _assert_rejected(typed, "method")

    upcast = _run_entry(tmp_path, {}, declarations + "(Color::Red as Color).shade()\n")
    _assert_rejected(upcast, "method")


def test_overlapping_enum_membership_widens_a_shared_members_selection_level(
    tmp_path: Path,
) -> None:
    """A member counted by two current enums carries both enums' methods.

    `A::X` is a member of both `A` (inline) and `B` (referenced), so a bare
    `A::X.f()` sees both `A::f` and `B::f` -- an ambiguity. A narrowly typed
    receiver, or an identity upcast to it, sees only its own enum's method.
    """
    declarations = (
        "enum A\n  | X\n  | Y\n"
        "enum B = A::X | A::Y | Z\n"
        'def A::f(self) -> text = "a"\n'
        'def B::f(self) -> text = "b"\n'
    )
    bare = _run_entry(tmp_path, {}, declarations + "A::X.f()\n")
    typed = _run_entry(tmp_path, {}, declarations + "let a: A = X\na.f()\n")
    upcast = _run_entry(tmp_path, {}, declarations + "(X as A).f()\n")
    _assert_rejected(bare, "ambiguous")
    a_f = (ENTRY_ID, ("A",), "f")
    _assert_selects(typed, *a_f)
    _assert_selects(upcast, *a_f)


def test_referenced_record_gains_a_foreign_owning_enums_method(tmp_path: Path) -> None:
    """A record referenced by a foreign enum gains that enum's method.

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
    _assert_rejected(bare, "ambiguous")
    _assert_selects(widened, ModuleId.from_path("store"), ("Stored",), "describe")


def test_generic_member_widens_with_its_captured_type_argument(tmp_path: Path) -> None:
    """`Tree::Node[int]` widens to `Tree[int]`, matching an explicit `Tree[int]` binding."""
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
    upcast = _run_entry(
        tmp_path,
        {},
        declarations + "(Node(value = 1) as Tree[int]).describe()\n",
    )
    describe = (ENTRY_ID, ("Tree",), "describe")
    _assert_selects(bare, *describe)
    _assert_selects(typed, *describe)
    _assert_selects(upcast, *describe)


def test_foreign_referencing_enums_method_requires_a_reachable_route(tmp_path: Path) -> None:
    """A foreign enum's method on a referenced record is still route-gated, like an orphan method.

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
    upcast = _run_entry(
        tmp_path,
        modules,
        "import shapes\nimport wrap\n(shapes::Point(x = 1) as wrap::Wrapped).describe()\n",
    )
    hidden = _run_entry(
        tmp_path,
        modules,
        "import shapes\nimport wrap hiding Wrapped::describe\nshapes::Point(x = 1).describe()\n",
    )
    wrapped_describe = (ModuleId.from_path("wrap"), ("Wrapped",), "describe")
    _assert_selects(visible, *wrapped_describe)
    _assert_selects(upcast, *wrapped_describe)
    _assert_rejected(hidden, "visible")


# ---------------------------------------------------------------------------
# Exception `extends` chains form one level
# ---------------------------------------------------------------------------


def test_exception_chain_pair_selects_consistently_or_errors(tmp_path: Path) -> None:
    """A base method and a descendant method, declared in different modules.

    A descendant-typed receiver sees both -- an ambiguity. Both a base-typed
    binding and an identity upcast to the base see only the base's chain.
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
    upcast = _run_entry(
        tmp_path,
        modules,
        "import errors\nimport base_methods\nimport derived_methods\n"
        '(errors::Derived(message = "bad", code = 1) as errors::Base).describe()\n',
    )
    _assert_rejected(descendant, "ambiguous")
    base_describe = (ModuleId.from_path("base_methods"), ("Base",), "describe")
    _assert_selects(base, *base_describe)
    _assert_selects(upcast, *base_describe)


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
    status = (ModuleId.from_path("methods"), ("Base",), "status")
    _assert_selects(descendant, *status)
    _assert_selects(base, *status)


# ---------------------------------------------------------------------------
# Field vs. method kind clash, checked across receiver widening
# ---------------------------------------------------------------------------


def test_field_vs_enum_method_kind_clash_across_receiver_widening(tmp_path: Path) -> None:
    """A member field and an owning enum's same-named method: ambiguous only on the member.

    An enum-typed receiver has no fields, so the enum method is selected
    cleanly.
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
    _assert_rejected(member_read, "ambiguous")
    assert isinstance(enum_call, CheckedProgram)
    assert enum_call.modules[ENTRY_ID].node_types[final_entry_call(enum_call).node_id] == TextType()


# ---------------------------------------------------------------------------
# Outside the invariant: enum-to-enum subset widening
# ---------------------------------------------------------------------------


def test_enum_to_enum_subset_widening_is_outside_the_invariant(tmp_path: Path) -> None:
    """`A` and `B` are distinct nominal types even when `A subseteq B`.

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
    _assert_selects(as_a, ENTRY_ID, ("A",), "f")
    _assert_selects(as_b, ENTRY_ID, ("B",), "f")
