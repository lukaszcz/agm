"""Module-local constant folding over a parsed module's AST alone."""

from __future__ import annotations

from agm.agl.parser import parse_program
from agm.agl.syntax.module_constants import FoldFailure, ModuleConstants
from agm.agl.syntax.nodes import FuncDef, LetDecl, static_items


def _fold(source: str, expression: str, *, scope: str = "") -> str | FoldFailure:
    """Fold *expression*, written at module root inside *scope*, against *source*."""
    prefix = f"scope {scope}\n  " if scope else ""
    suffix = f"\nend {scope}\n" if scope else "\n"
    program = parse_program(f"{source}\n{prefix}let probe = {expression}{suffix}")
    constants = ModuleConstants(program)
    (probe,) = (
        item
        for item in static_items(program.body.items)
        if isinstance(item, LetDecl) and item.name == "probe"
    )
    return constants.fold_text(probe.value, scope_path=(scope,) if scope else ())


class TestScalarHoles:
    def test_every_scalar_kind_renders_the_way_it_prints(self) -> None:
        source = 'let name = "ada"\nlet count = 3\nlet ratio = 1.50\nlet precise = true\n'

        folded = _fold(source, '"%{name} %{count} %{ratio} %{precise}"')

        assert folded == "ada 3 1.5 true"

    def test_a_bare_reference_is_its_own_text(self) -> None:
        assert _fold('let prose = "hello"\n', "prose") == "hello"

    def test_a_reference_chain_folds_through(self) -> None:
        source = 'let base = "core"\nlet derived = "%{base}+extra"\n'

        assert _fold(source, '"[%{derived}]"') == "[core+extra]"

    def test_a_negated_number_and_a_negated_bool_fold(self) -> None:
        source = "let offset = -2\nlet shifted = -1.25\nlet off = not true\n"

        assert _fold(source, '"%{offset} %{shifted} %{off}"') == "-2 -1.25 false"

    def test_a_forward_reference_folds(self) -> None:
        """A constant denotes its value, so declaration order does not matter."""
        assert _fold('let later = "tail"\n', '"%{later}"') == "tail"

    def test_an_escaped_hole_stays_literal(self) -> None:
        assert _fold('let name = "ada"\n', r'"\%{name} is %{name}"') == "%{name} is ada"

    def test_a_triple_quoted_template_folds_its_holes(self) -> None:
        source = 'let topic = "scopes"\n'

        assert _fold(source, '"""\n  Read about %{topic}.\n  """') == "Read about scopes."

    def test_a_non_text_constant_is_not_a_text_argument(self) -> None:
        folded = _fold("let count = 3\n", "count")

        assert isinstance(folded, FoldFailure)
        assert "text constant" in folded.message


class TestNameResolution:
    def test_a_scope_region_sees_its_own_members_first(self) -> None:
        source = 'let label = "root"\n\nscope Inner\n  let label = "inner"\nend Inner\n'

        assert _fold(source, '"%{label}"', scope="Inner") == "inner"

    def test_a_scope_region_falls_outward_to_the_module_root(self) -> None:
        source = 'let label = "root"\n'

        assert _fold(source, '"%{label}"', scope="Inner") == "root"

    def test_an_anchored_reference_bypasses_a_nearer_member(self) -> None:
        source = 'let label = "root"\n\nscope Inner\n  let label = "inner"\nend Inner\n'

        assert _fold(source, '"%{::label}"', scope="Inner") == "root"

    def test_a_scope_path_reaches_a_member_from_the_root(self) -> None:
        source = 'scope Inner\n  let label = "inner"\nend Inner\n'

        assert _fold(source, '"%{Inner::label}"') == "inner"

    def test_a_declaration_path_shorthand_declares_into_its_scope(self) -> None:
        source = 'scope Inner\nend Inner\n\nlet Inner::label = "shorthand"\n'

        assert _fold(source, '"%{Inner::label}"') == "shorthand"

    def test_a_module_route_is_rejected_as_another_module(self) -> None:
        folded = _fold("import std/config\n", '"%{/std/config::strict-json}"')

        assert isinstance(folded, FoldFailure)
        assert "another module" in folded.message

    def test_an_unknown_name_names_no_constant(self) -> None:
        folded = _fold("", '"%{missing}"')

        assert isinstance(folded, FoldFailure)
        assert "no constant of this module" in folded.message


class TestFoldFailures:
    def test_a_self_referential_constant_is_reported_as_a_cycle(self) -> None:
        folded = _fold('let loop-a = "%{loop-b}"\nlet loop-b = "%{loop-a}"\n', '"%{loop-a}"')

        assert isinstance(folded, FoldFailure)
        assert "in terms of itself" in folded.message

    def test_a_container_constant_does_not_render(self) -> None:
        folded = _fold('let items = ["a", "b"]\n', '"%{items}"')

        assert isinstance(folded, FoldFailure)
        assert "folds into text" in folded.message

    def test_a_call_is_not_a_constant(self) -> None:
        folded = _fold("def size() -> int = 1\n", '"%{size()}"')

        assert isinstance(folded, FoldFailure)
        assert "not a constant expression" in folded.message

    def test_an_environment_hole_is_not_a_constant(self) -> None:
        folded = _fold("", '"%{1}${HOME}"')

        assert isinstance(folded, FoldFailure)

    def test_a_failure_under_an_operator_is_reported_as_itself(self) -> None:
        negated = _fold("", '"%{-missing}"')
        inverted = _fold("", '"%{not missing}"')

        assert isinstance(negated, FoldFailure)
        assert isinstance(inverted, FoldFailure)
        assert "no constant of this module" in negated.message
        assert "no constant of this module" in inverted.message

    def test_negation_of_a_non_number_does_not_fold(self) -> None:
        folded = _fold("let flag = true\n", '"%{-flag}"')

        assert isinstance(folded, FoldFailure)
        assert "number" in folded.message

    def test_logical_negation_of_a_non_bool_does_not_fold(self) -> None:
        folded = _fold("let count = 1\n", '"%{not count}"')

        assert isinstance(folded, FoldFailure)
        assert "bool" in folded.message


class TestStaticScopePaths:
    def test_a_static_item_reports_the_scope_that_contains_it(self) -> None:
        program = parse_program(
            "scope Outer\n\n  scope Inner\n    def work() -> unit = ()\n  end Inner\nend Outer\n"
            "\n"
            "program def main() -> unit = ()\n"
        )
        constants = ModuleConstants(program)
        (work,) = (
            item
            for item in static_items(program.body.items)
            if isinstance(item, FuncDef) and item.name == "work"
        )

        assert constants.static_scope_path_of(work.node_id) == ("Outer", "Inner")

    def test_a_node_that_is_not_a_static_item_reports_nothing(self) -> None:
        program = parse_program("program def main() -> unit = ()\n")
        constants = ModuleConstants(program)

        assert constants.static_scope_path_of(-1) is None

    def test_a_builtin_var_is_a_static_item_but_not_a_constant(self) -> None:
        program = parse_program(
            "builtin var level: int\nprogram def main() -> unit = print level\n"
        )
        constants = ModuleConstants(program)
        probe = parse_program('let probe = "%{level}"\nprogram def main() -> unit = ()\n')
        (binding,) = (
            item
            for item in static_items(probe.body.items)
            if isinstance(item, LetDecl) and item.name == "probe"
        )

        folded = constants.fold_text(binding.value, scope_path=())

        assert isinstance(folded, FoldFailure)
        assert "no constant of this module" in folded.message


def test_a_decimal_constant_never_folds_to_scientific_notation() -> None:
    assert _fold("let big = 100.00\n", '"%{big}"') == "100"
    assert _fold("let small = 0.500\n", '"%{small}"') == "0.5"
