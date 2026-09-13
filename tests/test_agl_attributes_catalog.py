"""Contract tests for the built-in AgL attribute catalog.

The catalog is pure data: it names every built-in attribute, the declaration
kinds each one may sit on, the shape of its arguments, whether it may repeat,
and which attributes it excludes. Recognition itself lives in the scope pass,
so these tests pin the data rather than any validation behaviour.
"""

from __future__ import annotations

import dataclasses
import operator

import pytest

from agm.agl.attributes import (
    BUILTIN_ATTRIBUTES,
    AttributeArguments,
    AttributeSpec,
    AttributeTarget,
)

ZONE_ATTRIBUTES = ("arg-pos", "arg-std", "arg-named")
OPTION_ATTRIBUTES = ("opt-short", "opt-name", "opt-env", "opt-metavar", "opt-hidden")
COMMAND_ATTRIBUTES = ("command", "description", "help")


class TestCatalogContents:
    def test_every_builtin_attribute_is_listed(self) -> None:
        assert set(BUILTIN_ATTRIBUTES) == {
            *ZONE_ATTRIBUTES,
            *OPTION_ATTRIBUTES,
            *COMMAND_ATTRIBUTES,
            "extern-name",
            "doc",
        }

    def test_specs_are_keyed_by_their_own_name(self) -> None:
        assert all(name == spec.name for name, spec in BUILTIN_ATTRIBUTES.items())

    def test_specs_are_frozen_and_hashable(self) -> None:
        spec = BUILTIN_ATTRIBUTES["doc"]
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(spec, "name", "other")
        assert hash(spec) == hash(BUILTIN_ATTRIBUTES["doc"])

    def test_the_catalog_is_read_only(self) -> None:
        with pytest.raises(TypeError):
            operator.setitem(BUILTIN_ATTRIBUTES, "doc", BUILTIN_ATTRIBUTES["arg-pos"])


class TestZoneAttributes:
    @pytest.mark.parametrize("name", ZONE_ATTRIBUTES)
    def test_zone_attributes_take_no_arguments(self, name: str) -> None:
        assert BUILTIN_ATTRIBUTES[name].arguments is AttributeArguments.NONE

    @pytest.mark.parametrize("name", ZONE_ATTRIBUTES)
    def test_zone_attributes_sit_on_entries_and_their_owners(self, name: str) -> None:
        assert BUILTIN_ATTRIBUTES[name].targets == frozenset(
            {
                AttributeTarget.PARAMETER,
                AttributeTarget.PROGRAM_PARAMETER,
                AttributeTarget.FIELD,
                AttributeTarget.FUNCTION,
                AttributeTarget.PROGRAM,
                AttributeTarget.EXTERN,
                AttributeTarget.BUILTIN_FUNCTION,
                AttributeTarget.RECORD,
                AttributeTarget.EXCEPTION,
                AttributeTarget.ENUM_MEMBER,
            }
        )

    @pytest.mark.parametrize("name", ZONE_ATTRIBUTES)
    def test_each_zone_attribute_excludes_the_other_two(self, name: str) -> None:
        assert set(BUILTIN_ATTRIBUTES[name].conflicts) == {
            other for other in ZONE_ATTRIBUTES if other != name
        }


class TestExternAndOptionAttributes:
    def test_extern_name_takes_one_text_on_externs_only(self) -> None:
        spec = BUILTIN_ATTRIBUTES["extern-name"]
        assert spec.targets == frozenset({AttributeTarget.EXTERN})
        assert spec.arguments is AttributeArguments.ONE_TEXT

    @pytest.mark.parametrize("name", OPTION_ATTRIBUTES)
    def test_option_attributes_sit_on_program_parameters_only(self, name: str) -> None:
        assert BUILTIN_ATTRIBUTES[name].targets == frozenset({AttributeTarget.PROGRAM_PARAMETER})

    @pytest.mark.parametrize("name", ("opt-short", "opt-name", "opt-env", "opt-metavar"))
    def test_valued_option_attributes_take_one_text(self, name: str) -> None:
        assert BUILTIN_ATTRIBUTES[name].arguments is AttributeArguments.ONE_TEXT

    def test_opt_hidden_is_a_bare_marker(self) -> None:
        assert BUILTIN_ATTRIBUTES["opt-hidden"].arguments is AttributeArguments.NONE

    def test_no_builtin_attribute_conflicts_outside_the_zone_family(self) -> None:
        for name, spec in BUILTIN_ATTRIBUTES.items():
            if name not in ZONE_ATTRIBUTES:
                assert spec.conflicts == ()


class TestCommandAttributes:
    """The attributes registering a program as a package command."""

    @pytest.mark.parametrize("name", COMMAND_ATTRIBUTES)
    def test_command_attributes_sit_on_programs_only(self, name: str) -> None:
        assert BUILTIN_ATTRIBUTES[name].targets == frozenset({AttributeTarget.PROGRAM})

    @pytest.mark.parametrize("name", COMMAND_ATTRIBUTES)
    def test_command_attributes_take_one_text(self, name: str) -> None:
        assert BUILTIN_ATTRIBUTES[name].arguments is AttributeArguments.ONE_TEXT


class TestDocAttribute:
    def test_doc_takes_one_text_on_every_target(self) -> None:
        spec = BUILTIN_ATTRIBUTES["doc"]
        assert spec.arguments is AttributeArguments.ONE_TEXT
        assert spec.targets == frozenset(AttributeTarget)


class TestSpecDefaults:
    def test_a_spec_defaults_to_an_unconstrained_argument_without_conflicts(self) -> None:
        spec = AttributeSpec(
            name="example",
            targets=frozenset({AttributeTarget.RECORD}),
            arguments=AttributeArguments.NONE,
        )
        assert spec.conflicts == ()
        assert spec.pattern is None
        assert spec.expected is None
