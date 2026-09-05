"""Tests for declaration-attribute recognition in the AgL scope pass.

Every attribute a declaration carries is validated against the built-in
attribute catalog while names are resolved, and the ``@arg-*`` attributes are
turned into the resolved program's ``param_zones`` table. These tests assert
on that table's contents and pin the phase the rejections come from; the
rejection fixtures under ``tests/agl/rejections/scope/`` cover the same
diagnostics as whole programs.
"""

from __future__ import annotations

import pytest

from agm.agl.scope import AglScopeError, ModuleResolution
from agm.agl.syntax.nodes import (
    Attribute,
    EnumDef,
    ExceptionDef,
    FuncDef,
    Lambda,
    LetDecl,
    Param,
    RecordDef,
    StringLit,
    VariantDef,
)
from agm.agl.syntax.visitor import walk
from agm.agl.zones import ParamZone
from tests.agl.module_graph import resolve_entry

POSITIONAL_ONLY = ParamZone.POSITIONAL_ONLY
STANDARD = ParamZone.STANDARD
NAMED_ONLY = ParamZone.NAMED_ONLY


def _entries(resolution: ModuleResolution, owner_name: str) -> tuple[Param, ...]:
    """Return the parameter or field list of the named declaration."""
    found: list[tuple[Param, ...]] = []

    def visit(node: object) -> None:
        if isinstance(node, FuncDef) and node.name == owner_name:
            found.append(node.params)
        elif isinstance(node, (RecordDef, ExceptionDef, VariantDef)) and node.name == owner_name:
            found.append(node.fields)

    walk(resolution.program, visit)
    assert len(found) == 1, f"expected exactly one declaration named {owner_name!r}"
    return found[0]


def _zones(resolution: ModuleResolution, owner_name: str) -> dict[str, ParamZone]:
    """Return the resolved zone of every entry of the named declaration."""
    return {
        entry.name: resolution.param_zones[entry.node_id]
        for entry in _entries(resolution, owner_name)
    }


def _lambda_zones(resolution: ModuleResolution) -> dict[str, ParamZone]:
    """Return the resolved zones of the single lambda in the program."""
    lambdas: list[Lambda] = []

    def visit(node: object) -> None:
        if isinstance(node, Lambda):
            lambdas.append(node)

    walk(resolution.program, visit)
    assert len(lambdas) == 1
    return {p.name: resolution.param_zones[p.node_id] for p in lambdas[0].params}


def _doc_arguments(attributes: tuple[Attribute, ...]) -> list[str]:
    """Return the text argument of every ``@doc`` among *attributes*."""
    return [
        arg.value
        for attribute in attributes
        if attribute.name == "doc"
        for arg in attribute.args
        if isinstance(arg, StringLit)
    ]


class TestParameterZones:
    """Per-entry ``@arg-*`` attributes and the form defaults they override."""

    def test_a_function_defaults_every_parameter_to_standard(self) -> None:
        resolution = resolve_entry("def f(a: int, b: int) -> int = a + b\n")

        assert _zones(resolution, "f") == {"a": STANDARD, "b": STANDARD}

    def test_a_parameter_attribute_sets_that_entrys_zone(self) -> None:
        resolution = resolve_entry(
            "def f(@arg-pos a: int, b: int, @arg-named c: int) -> int = a + b + c\n"
        )

        assert _zones(resolution, "f") == {
            "a": POSITIONAL_ONLY,
            "b": STANDARD,
            "c": NAMED_ONLY,
        }

    def test_a_declaration_attribute_sets_the_list_default(self) -> None:
        resolution = resolve_entry("@arg-named\ndef f(a: int, b: int) -> int = a + b\n")

        assert _zones(resolution, "f") == {"a": NAMED_ONLY, "b": NAMED_ONLY}

    def test_an_entry_attribute_overrides_the_declaration_default(self) -> None:
        resolution = resolve_entry("@arg-named\ndef f(@arg-pos a: int, b: int) -> int = a + b\n")

        assert _zones(resolution, "f") == {"a": POSITIONAL_ONLY, "b": NAMED_ONLY}

    def test_a_program_defaults_its_parameters_to_named_only(self) -> None:
        resolution = resolve_entry("program def main(a: int) -> unit = print a\n")

        assert _zones(resolution, "main") == {"a": NAMED_ONLY}

    def test_a_program_declaration_attribute_overrides_the_named_only_default(self) -> None:
        resolution = resolve_entry("@arg-pos\nprogram def main(a: int) -> unit = print a\n")

        assert _zones(resolution, "main") == {"a": POSITIONAL_ONLY}

    def test_a_receiver_is_positional_only(self) -> None:
        resolution = resolve_entry(
            "record Point\n  x: int\n\ndef Point::shifted(self, by: int) -> int = self.x + by\n"
        )

        assert _zones(resolution, "shifted") == {"self": POSITIONAL_ONLY, "by": STANDARD}

    def test_a_receiver_sits_ahead_of_a_positional_only_parameter(self) -> None:
        resolution = resolve_entry(
            "record Point\n"
            "  x: int\n"
            "\n"
            "def Point::shifted(self, @arg-pos by: int) -> int = self.x + by\n"
        )

        assert _zones(resolution, "shifted") == {
            "self": POSITIONAL_ONLY,
            "by": POSITIONAL_ONLY,
        }

    def test_a_zone_attribute_is_found_past_an_unrelated_one(self) -> None:
        resolution = resolve_entry('def f(@doc("a") @arg-named a: int) -> int = a\n')

        assert _zones(resolution, "f") == {"a": NAMED_ONLY}

    def test_a_lambda_parameter_carries_its_own_zone(self) -> None:
        resolution = resolve_entry(
            "def f() -> int =\n"
            "  let g = fn(a: int, @arg-named b: int) -> int => a + b\n"
            "  g(1, b = 2)\n"
        )

        assert _lambda_zones(resolution) == {"a": STANDARD, "b": NAMED_ONLY}


class TestFieldZones:
    """Record, exception and enum-member field lists follow the same rules."""

    def test_record_fields_default_to_standard(self) -> None:
        resolution = resolve_entry("record R\n  x: int\n  y: int\n")

        assert _zones(resolution, "R") == {"x": STANDARD, "y": STANDARD}

    def test_a_record_attribute_sets_its_field_default(self) -> None:
        resolution = resolve_entry("@arg-named\nrecord R\n  @arg-std x: int\n  y: int\n")

        assert _zones(resolution, "R") == {"x": STANDARD, "y": NAMED_ONLY}

    def test_an_exception_attribute_sets_its_field_default(self) -> None:
        resolution = resolve_entry(
            "@arg-named\nexception Failure extends Exception\n  detail: text\n"
        )

        assert _zones(resolution, "Failure") == {"detail": NAMED_ONLY}

    def test_an_enum_member_attribute_sets_its_field_default(self) -> None:
        resolution = resolve_entry(
            "enum Shape =\n  | @arg-named Circle(radius: int)\n  | Square(side: int)\n"
        )

        assert _zones(resolution, "Circle") == {"radius": NAMED_ONLY}
        assert _zones(resolution, "Square") == {"side": STANDARD}


class TestAttributeDiagnostics:
    """Diagnostics that need no whole-program pipeline to observe."""

    def test_an_unknown_attribute_is_rejected(self) -> None:
        with pytest.raises(AglScopeError, match="nonesuch"):
            resolve_entry("@nonesuch\ndef f() -> int = 1\n")

    def test_an_attribute_on_a_disallowed_target_is_rejected(self) -> None:
        with pytest.raises(AglScopeError, match="opt-name"):
            resolve_entry('@opt-name("x")\nrecord R\n  x: int\n')

    def test_an_enum_declaration_admits_a_doc_attribute(self) -> None:
        resolution = resolve_entry('@doc("shapes")\nenum Shape =\n  | Dot\n  | Dash\n')

        enums = [item for item in resolution.program.body.items if isinstance(item, EnumDef)]
        assert [enum.name for enum in enums] == ["Shape"]
        assert _doc_arguments(enums[0].attributes) == ["shapes"]

    def test_a_zone_attribute_on_an_enum_declaration_is_rejected(self) -> None:
        with pytest.raises(AglScopeError, match="arg-named"):
            resolve_entry("@arg-named\nenum Shape =\n  | Dot\n  | Dash\n")

    def test_a_binding_admits_a_doc_attribute(self) -> None:
        resolution = resolve_entry('@doc("the answer")\nlet answer = 42\n')

        lets = [item for item in resolution.program.body.items if isinstance(item, LetDecl)]
        assert len(lets) == 1
        assert _doc_arguments(lets[0].attributes) == ["the answer"]

    def test_a_zone_attribute_on_a_binding_is_rejected(self) -> None:
        with pytest.raises(AglScopeError, match="arg-pos"):
            resolve_entry("@arg-pos\nlet answer = 42\n")

    def test_a_zone_attribute_out_of_order_is_rejected(self) -> None:
        with pytest.raises(AglScopeError, match="ordered"):
            resolve_entry("def f(a: int, @arg-pos b: int) -> int = a + b\n")

    def test_a_zone_attribute_on_a_receiver_is_rejected(self) -> None:
        with pytest.raises(AglScopeError, match="shifted"):
            resolve_entry(
                "record Point\n"
                "  x: int\n"
                "\n"
                "def Point::shifted(@arg-std self, by: int) -> int = self.x + by\n"
            )

    def test_a_type_alias_admits_a_doc_attribute(self) -> None:
        resolution = resolve_entry('@doc("a count")\ntype Count = int\n')

        assert resolution.param_zones == {}
