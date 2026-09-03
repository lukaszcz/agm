"""Tests for ``cli_support.program_options``: the type-directed CLI projection
shared by engine keys and program parameters, and ``ProgramOptionMap``.
"""

from __future__ import annotations

import pytest

from agm.agl.ir.reserved_nominals import require_reserved_nominal_id
from agm.agl.ir.zones import ParamZone
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.runtime.arguments import decode_param_value
from agm.agl.runtime.option import none_value, some_value
from agm.agl.runtime.types import ProgramParamInfo
from agm.agl.semantics.type_table import create_seeded_type_table
from agm.agl.semantics.types import (
    RESERVED_ID,
    ArrayType,
    BoolType,
    EnumType,
    IntType,
    TextType,
    Type,
)
from agm.agl.semantics.values import ArrayValue, BoolValue, IntValue, TextValue, Value
from agm.agl.syntax.spans import SourceSpan
from agm.agl.type_schema import build_param_decoder
from agm.cli_support.program_options import (
    DuplicateOptionFlagError,
    ProgramOptionMap,
    ReservedFlagError,
    ValueForm,
    build_program_option_map,
    engine_key_flags,
    project_option,
)
from tests._agl_helpers import next_decl_id

_SPAN = SourceSpan(1, 1, 1, 2, 0, 1)
_TABLE = create_seeded_type_table()


def _option_type(inner: Type) -> EnumType:
    return EnumType(
        name="Option",
        type_args=(inner,),
        module_id=RESERVED_ID,
        decl_id=require_reserved_nominal_id("Option"),
    )


def _param(
    name: str,
    typ: Type,
    kind: ParamZone = ParamZone.NAMED_ONLY,
    *,
    has_default: bool = False,
) -> ProgramParamInfo:
    return ProgramParamInfo(name=name, kind=kind, type=typ, has_default=has_default, span=_SPAN)


# ---------------------------------------------------------------------------
# project_option
# ---------------------------------------------------------------------------


class TestProjectOption:
    def test_bool_projects_a_flag_pair_with_no_value(self) -> None:
        projected = project_option("verbose", BoolType())

        assert projected.flags == ("--verbose",)
        assert projected.negative_flags == ("--no-verbose",)
        assert projected.takes_value is False
        assert projected.value_form is ValueForm.BOOL

    def test_text_projects_a_verbatim_value_flag_with_no_negative(self) -> None:
        projected = project_option("name", TextType())

        assert projected.flags == ("--name",)
        assert projected.negative_flags == ()
        assert projected.takes_value is True
        assert projected.value_form is ValueForm.TEXT

    def test_option_projects_a_flag_pair_taking_a_value(self) -> None:
        projected = project_option("region", _option_type(TextType()))

        assert projected.flags == ("--region",)
        assert projected.negative_flags == ("--no-region",)
        assert projected.takes_value is True
        assert projected.value_form is ValueForm.OPTION

    def test_option_of_bool_still_projects_the_option_form(self) -> None:
        projected = project_option("flag", _option_type(BoolType()))

        assert projected.value_form is ValueForm.OPTION
        assert projected.negative_flags == ("--no-flag",)

    def test_every_other_type_projects_a_json_value_with_no_negative(self) -> None:
        for typ in (IntType(), ArrayType(elem=TextType())):
            projected = project_option("count", typ)
            assert projected.negative_flags == ()
            assert projected.takes_value is True
            assert projected.value_form is ValueForm.JSON

    def test_flag_spelling_preserves_the_name_verbatim(self) -> None:
        assert project_option("my_flag", TextType()).flags == ("--my_flag",)

    def test_entry_module_enum_named_option_projects_as_json_not_option(self) -> None:
        """A user-declared ``enum Option[T]`` in the entry module shares the
        name but not the identity with the standard ``Option`` — it must not
        be misprojected as the ``Option`` shape (see ``is_standard_option_enum``)."""
        user_option = EnumType(
            name="Option", type_args=(TextType(),), module_id=ENTRY_ID, decl_id=next_decl_id()
        )

        projected = project_option("thing", user_option)

        assert projected.value_form is ValueForm.JSON
        assert projected.negative_flags == ()
        assert projected.takes_value is True


# ---------------------------------------------------------------------------
# engine_key_flags
# ---------------------------------------------------------------------------


class TestEngineKeyFlags:
    def test_bool_engine_key_contributes_both_polarities(self) -> None:
        flags = engine_key_flags()
        assert "--log" in flags
        assert "--no-log" in flags

    def test_option_engine_key_contributes_both_polarities(self) -> None:
        flags = engine_key_flags()
        assert "--timeout" in flags
        assert "--no-timeout" in flags

    def test_scalar_engine_key_contributes_only_its_positive_flag(self) -> None:
        flags = engine_key_flags()
        assert "--max-iters" in flags
        assert "--no-max-iters" not in flags

    def test_agent_engine_key_contributes_only_its_positive_flag(self) -> None:
        flags = engine_key_flags()
        assert "--default-agent" in flags
        assert "--no-default-agent" not in flags

    def test_reserved_flags_includes_every_engine_key_flag(self) -> None:
        from agm.cli_support.exec_params import RESERVED_FLAGS

        assert engine_key_flags() <= RESERVED_FLAGS


# ---------------------------------------------------------------------------
# build_program_option_map: shape and reservation
# ---------------------------------------------------------------------------


class TestBuildProgramOptionMap:
    def test_positional_only_and_standard_params_are_positional_in_order(self) -> None:
        params = (
            _param("first", TextType(), ParamZone.POSITIONAL_ONLY),
            _param("second", TextType(), ParamZone.STANDARD),
            _param("third", TextType(), ParamZone.NAMED_ONLY),
        )
        result = build_program_option_map(params)

        assert isinstance(result, ProgramOptionMap)
        assert [p.name for p in result.positional] == ["first", "second"]

    def test_standard_and_named_only_params_are_options(self) -> None:
        params = (
            _param("first", TextType(), ParamZone.POSITIONAL_ONLY),
            _param("second", TextType(), ParamZone.STANDARD),
            _param("third", TextType(), ParamZone.NAMED_ONLY),
        )
        result = build_program_option_map(params)

        assert isinstance(result, ProgramOptionMap)
        assert [p.name for p, _projected in result.options] == ["second", "third"]

    def test_positional_only_param_is_never_reserved_checked(self) -> None:
        params = (_param("help", TextType(), ParamZone.POSITIONAL_ONLY),)
        result = build_program_option_map(params)

        assert isinstance(result, ProgramOptionMap)

    def test_named_param_colliding_with_a_builtin_flag_is_rejected(self) -> None:
        params = (_param("help", TextType(), ParamZone.NAMED_ONLY),)
        result = build_program_option_map(params)

        assert result == ReservedFlagError(parameter="help", flag="--help")

    def test_named_param_colliding_with_an_engine_key_is_rejected(self) -> None:
        params = (_param("timeout", TextType(), ParamZone.NAMED_ONLY),)
        result = build_program_option_map(params)

        assert result == ReservedFlagError(parameter="timeout", flag="--timeout")

    def test_bool_param_colliding_only_via_its_negative_flag_is_rejected(self) -> None:
        # ``--no-stdlib`` is a reserved built-in host flag with no paired
        # ``--stdlib``, so only a bool param's negative form collides.
        params = (_param("stdlib", BoolType(), ParamZone.NAMED_ONLY),)
        result = build_program_option_map(params)

        assert result == ReservedFlagError(parameter="stdlib", flag="--no-stdlib")

    def test_bool_negative_colliding_with_another_params_positive_flag_is_rejected(self) -> None:
        params = (
            _param("cache", BoolType(), ParamZone.NAMED_ONLY),
            _param("no-cache", BoolType(), ParamZone.NAMED_ONLY),
        )
        result = build_program_option_map(params)

        assert result == DuplicateOptionFlagError(
            first_parameter="cache", second_parameter="no-cache", flag="--no-cache"
        )

    def test_option_negative_colliding_with_another_bool_params_positive_flag_is_rejected(
        self,
    ) -> None:
        params = (
            _param("region", _option_type(TextType()), ParamZone.NAMED_ONLY),
            _param("no-region", BoolType(), ParamZone.NAMED_ONLY),
        )
        result = build_program_option_map(params)

        assert result == DuplicateOptionFlagError(
            first_parameter="region", second_parameter="no-region", flag="--no-region"
        )


# ---------------------------------------------------------------------------
# ProgramOptionMap.parse_tokens
# ---------------------------------------------------------------------------


def _map(*params: ProgramParamInfo) -> ProgramOptionMap:
    result = build_program_option_map(params)
    assert isinstance(result, ProgramOptionMap)
    return result


class TestParseTokensText:
    def test_named_value_flag(self) -> None:
        option_map = _map(_param("name", TextType()))
        args = option_map.parse_tokens(["--name", "hello"])
        assert args.positional == ()
        assert args.named == {"name": "hello"}

    def test_equals_form(self) -> None:
        option_map = _map(_param("name", TextType()))
        args = option_map.parse_tokens(["--name=hello"])
        assert args.named == {"name": "hello"}

    def test_equals_form_preserves_embedded_equals(self) -> None:
        option_map = _map(_param("expr", TextType()))
        args = option_map.parse_tokens(["--expr=a=b"])
        assert args.named == {"expr": "a=b"}

    def test_missing_value_raises(self) -> None:
        option_map = _map(_param("name", TextType()))
        with pytest.raises(ValueError, match="requires a value"):
            option_map.parse_tokens(["--name"])

    def test_equals_form_with_empty_value_yields_empty_string(self) -> None:
        option_map = _map(_param("name", TextType()))
        args = option_map.parse_tokens(["--name="])
        assert args.named == {"name": ""}


class TestParseTokensBool:
    def test_positive_flag_is_true(self) -> None:
        option_map = _map(_param("verbose", BoolType()))
        args = option_map.parse_tokens(["--verbose"])
        assert args.named == {"verbose": True}

    def test_negative_flag_is_false(self) -> None:
        option_map = _map(_param("verbose", BoolType()))
        args = option_map.parse_tokens(["--no-verbose"])
        assert args.named == {"verbose": False}

    def test_value_for_a_bool_flag_is_rejected(self) -> None:
        option_map = _map(_param("verbose", BoolType()))
        with pytest.raises(ValueError, match="does not take a value"):
            option_map.parse_tokens(["--verbose=true"])

    def test_negative_flag_with_value_is_rejected(self) -> None:
        option_map = _map(_param("verbose", BoolType()))
        with pytest.raises(ValueError, match="does not take a value"):
            option_map.parse_tokens(["--no-verbose=true"])


class TestParseTokensJson:
    def test_json_form_value_stays_a_raw_string(self) -> None:
        option_map = _map(_param("count", IntType()))
        args = option_map.parse_tokens(["--count", "42"])
        assert args.named == {"count": "42"}

    def test_array_typed_value_stays_a_raw_string(self) -> None:
        option_map = _map(_param("tags", ArrayType(elem=TextType())))
        args = option_map.parse_tokens(["--tags", '["a","b"]'])
        assert args.named == {"tags": '["a","b"]'}

    def test_equals_form_with_empty_value_yields_empty_string(self) -> None:
        option_map = _map(_param("count", IntType()))
        args = option_map.parse_tokens(["--count="])
        assert args.named == {"count": ""}


class TestParseTokensOption:
    def test_positive_text_option_wraps_the_value_verbatim(self) -> None:
        option_map = _map(_param("region", _option_type(TextType())))
        args = option_map.parse_tokens(["--region", "eu"])
        assert args.named == {"region": {"$case": "Some", "value": "eu"}}

    def test_negative_option_flag_is_none(self) -> None:
        option_map = _map(_param("region", _option_type(TextType())))
        args = option_map.parse_tokens(["--no-region"])
        assert args.named == {"region": {"$case": "None"}}

    def test_positive_int_option_json_parses_the_value(self) -> None:
        option_map = _map(_param("count", _option_type(IntType())))
        args = option_map.parse_tokens(["--count", "5"])
        assert args.named == {"count": {"$case": "Some", "value": 5}}

    def test_positive_bool_option_json_parses_the_value(self) -> None:
        option_map = _map(_param("flag", _option_type(BoolType())))
        args = option_map.parse_tokens(["--flag", "true"])
        assert args.named == {"flag": {"$case": "Some", "value": True}}

    def test_positive_nested_option_json_parses_the_whole_inner_shape(self) -> None:
        option_map = _map(_param("nested", _option_type(_option_type(IntType()))))
        args = option_map.parse_tokens(["--nested", '{"$case": "Some", "value": 7}'])
        assert args.named == {"nested": {"$case": "Some", "value": {"$case": "Some", "value": 7}}}

    def test_malformed_option_value_raises_immediately(self) -> None:
        option_map = _map(_param("count", _option_type(IntType())))
        with pytest.raises(ValueError, match="not valid JSON"):
            option_map.parse_tokens(["--count", "not-json"])

    def test_malformed_option_value_error_names_the_flag(self) -> None:
        option_map = _map(_param("count", _option_type(IntType())))
        with pytest.raises(ValueError, match=r"Option '--count'"):
            option_map.parse_tokens(["--count", "not-json"])

    def test_text_option_empty_inline_value_is_a_verbatim_empty_string(self) -> None:
        option_map = _map(_param("region", _option_type(TextType())))
        args = option_map.parse_tokens(["--region="])
        assert args.named == {"region": {"$case": "Some", "value": ""}}

    def test_json_option_empty_inline_value_raises(self) -> None:
        option_map = _map(_param("count", _option_type(IntType())))
        with pytest.raises(ValueError, match="not valid JSON"):
            option_map.parse_tokens(["--count="])


class TestParseTokensPositional:
    def test_positional_tokens_collected_in_order(self) -> None:
        option_map = _map(
            _param("first", TextType(), ParamZone.POSITIONAL_ONLY),
            _param("second", TextType(), ParamZone.POSITIONAL_ONLY),
        )
        args = option_map.parse_tokens(["a", "b"])
        assert args.positional == ("a", "b")
        assert args.named == {}

    def test_positional_and_options_interleave(self) -> None:
        option_map = _map(
            _param("first", TextType(), ParamZone.POSITIONAL_ONLY),
            _param("name", TextType(), ParamZone.NAMED_ONLY),
        )
        args = option_map.parse_tokens(["a", "--name", "hello"])
        assert args.positional == ("a",)
        assert args.named == {"name": "hello"}

    def test_more_positional_tokens_than_declared_are_collected_unconditionally(self) -> None:
        """No cap here: an excess positional argument is diagnosed downstream
        by the shared zone binder (``bind_program_arguments``), which reports
        a different message depending on whether the program declares any
        named-only parameters — this parser defers to it rather than
        duplicating (and pre-empting) that diagnosis."""
        option_map = _map(_param("first", TextType(), ParamZone.POSITIONAL_ONLY))
        args = option_map.parse_tokens(["a", "b"])
        assert args.positional == ("a", "b")

    def test_bare_token_with_no_positional_params_is_still_collected(self) -> None:
        option_map = _map(_param("name", TextType(), ParamZone.NAMED_ONLY))
        args = option_map.parse_tokens(["bare"])
        assert args.positional == ("bare",)
        assert args.named == {}

    def test_single_dash_token_is_positional(self) -> None:
        """A single-dash spelling — a negative-number JSON value, or any
        other token this parser recognizes no short-option form for — is
        always collected positionally, never mistaken for a flag."""
        option_map = _map(_param("first", TextType(), ParamZone.POSITIONAL_ONLY))
        args = option_map.parse_tokens(["-5"])
        assert args.positional == ("-5",)

    def test_single_dash_short_flag_shaped_token_is_positional(self) -> None:
        option_map = _map(_param("first", TextType(), ParamZone.POSITIONAL_ONLY))
        args = option_map.parse_tokens(["-x"])
        assert args.positional == ("-x",)

    def test_double_dash_ends_options(self) -> None:
        option_map = _map(_param("first", TextType(), ParamZone.POSITIONAL_ONLY))
        args = option_map.parse_tokens(["--", "--not-a-flag"])
        assert args.positional == ("--not-a-flag",)

    def test_double_dash_with_no_positional_params_still_collects_the_rest(self) -> None:
        option_map = _map()
        args = option_map.parse_tokens(["--", "x"])
        assert args.positional == ("x",)


class TestParseTokensErrors:
    def test_unknown_flag_raises(self) -> None:
        option_map = _map(_param("name", TextType()))
        with pytest.raises(ValueError, match="Unknown option"):
            option_map.parse_tokens(["--unknown"])

    def test_unknown_flag_in_equals_form_raises(self) -> None:
        option_map = _map(_param("name", TextType()))
        with pytest.raises(ValueError, match="Unknown option"):
            option_map.parse_tokens(["--unknown=value"])

    def test_duplicate_flag_raises(self) -> None:
        option_map = _map(_param("name", TextType()))
        with pytest.raises(ValueError, match="more than once"):
            option_map.parse_tokens(["--name", "a", "--name", "b"])

    def test_duplicate_bool_via_negative_form_raises(self) -> None:
        option_map = _map(_param("flag", BoolType()))
        with pytest.raises(ValueError, match="more than once"):
            option_map.parse_tokens(["--flag", "--no-flag"])

    def test_empty_tokens_returns_empty_arguments(self) -> None:
        option_map = _map(_param("name", TextType()))
        args = option_map.parse_tokens([])
        assert args.positional == ()
        assert args.named == {}

    def test_no_params_empty_tokens(self) -> None:
        option_map = _map()
        args = option_map.parse_tokens([])
        assert args.positional == ()
        assert args.named == {}


# ---------------------------------------------------------------------------
# The decode seam: every raw value ``parse_tokens`` emits must decode through
# ``runtime.arguments.decode_param_value`` (regression protection for the
# module's most important contract — the seam itself is already correct).
# ---------------------------------------------------------------------------


def _decode(typ: Type, raw: object) -> Value:
    decoder = build_param_decoder(typ, _TABLE)
    return decode_param_value(decoder, raw)


_DECODE_SEAM_CASES: tuple[tuple[str, Type, tuple[str, ...], Value], ...] = (
    ("name", TextType(), ("--name", "hello"), TextValue("hello")),
    ("count", IntType(), ("--count", "42"), IntValue(42)),
    (
        "tags",
        ArrayType(elem=TextType()),
        ("--tags", '["a","b"]'),
        ArrayValue([TextValue("a"), TextValue("b")]),
    ),
    ("verbose", BoolType(), ("--verbose",), BoolValue(True)),
    ("verbose", BoolType(), ("--no-verbose",), BoolValue(False)),
    ("region", _option_type(TextType()), ("--region", "eu"), some_value(TextValue("eu"))),
    ("count", _option_type(IntType()), ("--count", "5"), some_value(IntValue(5))),
    ("flag", _option_type(BoolType()), ("--flag", "true"), some_value(BoolValue(True))),
    (
        "nested",
        _option_type(_option_type(IntType())),
        ("--nested", '{"$case": "None"}'),
        some_value(none_value()),
    ),
    ("region", _option_type(TextType()), ("--no-region",), none_value()),
)


class TestDecodeSeam:
    @pytest.mark.parametrize(("name", "typ", "tokens", "expected"), _DECODE_SEAM_CASES)
    def test_raw_value_decodes_to_the_expected_value(
        self, name: str, typ: Type, tokens: tuple[str, ...], expected: Value
    ) -> None:
        option_map = _map(_param(name, typ))
        args = option_map.parse_tokens(list(tokens))
        assert _decode(typ, args.named[name]) == expected


# ---------------------------------------------------------------------------
# usage_line / render_help_section / completion_items
# ---------------------------------------------------------------------------


class TestUsageLine:
    def test_required_positional_uses_angle_brackets(self) -> None:
        option_map = _map(_param("file", TextType(), ParamZone.POSITIONAL_ONLY))
        assert option_map.usage_line("main") == "main <file>"

    def test_defaulted_positional_uses_square_brackets(self) -> None:
        option_map = _map(_param("file", TextType(), ParamZone.POSITIONAL_ONLY, has_default=True))
        assert option_map.usage_line("main") == "main [file]"

    def test_options_are_summarized_when_present(self) -> None:
        option_map = _map(_param("name", TextType(), ParamZone.NAMED_ONLY))
        assert option_map.usage_line("main") == "main [OPTIONS]"

    def test_no_positional_or_options_is_just_the_program_name(self) -> None:
        option_map = _map()
        assert option_map.usage_line("main") == "main"


class TestOptionLines:
    def test_no_options_describes_nothing(self) -> None:
        option_map = _map(_param("file", TextType(), ParamZone.POSITIONAL_ONLY))
        assert option_map.option_lines() == ()

    def test_one_line_per_name_addressable_parameter(self) -> None:
        option_map = _map(_param("name", TextType()), _param("verbose", BoolType()))
        described = option_map.option_lines()
        assert len(described) == 2
        assert described[0].startswith("--name")
        assert described[1].startswith("--verbose")

    def test_lines_are_unindented_and_carry_no_header(self) -> None:
        option_map = _map(_param("name", TextType()))
        described = option_map.option_lines()
        assert "Options:" not in described
        assert described[0] == described[0].lstrip()

    def test_render_help_section_indents_the_same_lines_under_a_header(self) -> None:
        option_map = _map(_param("name", TextType()), _param("verbose", BoolType()))
        section = option_map.render_help_section().splitlines()
        assert section[0] == "Options:"
        assert section[1:] == [f"  {line}" for line in option_map.option_lines()]


class TestRenderHelpSection:
    def test_no_options_renders_nothing(self) -> None:
        option_map = _map(_param("file", TextType(), ParamZone.POSITIONAL_ONLY))
        assert option_map.render_help_section() == ""

    def test_starts_with_the_options_header(self) -> None:
        option_map = _map(_param("name", TextType()))
        assert option_map.render_help_section().startswith("Options:\n")

    def test_required_option_is_marked_required(self) -> None:
        option_map = _map(_param("name", TextType(), has_default=False))
        section = option_map.render_help_section()
        assert "--name" in section
        assert "(required)" in section

    def test_defaulted_option_is_marked_optional(self) -> None:
        option_map = _map(_param("name", TextType(), has_default=True))
        section = option_map.render_help_section()
        assert "(required)" not in section
        assert "optional" in section
        assert "default" in section

    def test_bool_option_shows_both_flag_polarities_and_no_value_placeholder(self) -> None:
        option_map = _map(_param("verbose", BoolType()))
        section = option_map.render_help_section()
        assert "--verbose" in section
        assert "--no-verbose" in section
        assert "VALUE" not in section

    def test_option_shape_shows_both_polarities_and_a_value_placeholder(self) -> None:
        option_map = _map(_param("region", _option_type(TextType())))
        section = option_map.render_help_section()
        assert "--region" in section
        assert "--no-region" in section
        assert "VALUE" in section

    def test_plain_value_shape_shows_no_negative_and_a_value_placeholder(self) -> None:
        option_map = _map(_param("name", TextType()))
        section = option_map.render_help_section()
        assert "--name VALUE" in section
        assert "--no-name" not in section


class TestCompletionItems:
    def test_includes_positive_and_negative_flags(self) -> None:
        option_map = _map(_param("verbose", BoolType()))
        assert option_map.completion_items() == ("--verbose", "--no-verbose")

    def test_positional_only_params_contribute_no_items(self) -> None:
        option_map = _map(_param("file", TextType(), ParamZone.POSITIONAL_ONLY))
        assert option_map.completion_items() == ()

    def test_text_option_contributes_only_its_positive_flag(self) -> None:
        option_map = _map(_param("name", TextType()))
        assert option_map.completion_items() == ("--name",)


class TestValueTakingFlags:
    def test_bool_option_contributes_no_value_taking_flag(self) -> None:
        option_map = _map(_param("verbose", BoolType()))
        assert option_map.value_taking_flags() == frozenset()

    def test_positional_only_params_contribute_no_value_taking_flag(self) -> None:
        option_map = _map(_param("file", TextType(), ParamZone.POSITIONAL_ONLY))
        assert option_map.value_taking_flags() == frozenset()

    def test_text_and_option_shapes_contribute_their_positive_flag_only(self) -> None:
        option_map = _map(_param("name", TextType()), _param("region", _option_type(TextType())))
        assert option_map.value_taking_flags() == frozenset({"--name", "--region"})


class TestProgramOptionMapOrNone:
    def test_returns_the_built_map_for_a_valid_signature(self) -> None:
        from agm.cli_support.program_options import program_option_map_or_none

        option_map = program_option_map_or_none((_param("name", TextType()),))

        assert isinstance(option_map, ProgramOptionMap)
        assert option_map.completion_items() == ("--name",)

    def test_degrades_to_none_on_a_reservation_collision(self) -> None:
        from agm.cli_support.program_options import program_option_map_or_none

        assert program_option_map_or_none((_param("help", TextType()),)) is None


class TestShortHelpRequested:
    def test_bare_short_flag_is_a_help_request(self) -> None:
        from agm.cli_support.program_options import short_help_requested

        assert short_help_requested(["-h"], value_flags=frozenset()) is True

    def test_short_flag_consumed_as_a_preceding_value_flags_value_is_not_a_help_request(
        self,
    ) -> None:
        from agm.cli_support.program_options import short_help_requested

        assert short_help_requested(["--name", "-h"], value_flags=frozenset({"--name"})) is False

    def test_short_flag_as_the_first_positional_is_a_help_request(self) -> None:
        from agm.cli_support.program_options import short_help_requested

        assert short_help_requested(["-h", "extra"], value_flags=frozenset()) is True

    def test_short_flag_after_end_of_options_marker_is_not_a_help_request(self) -> None:
        from agm.cli_support.program_options import short_help_requested

        assert short_help_requested(["--", "-h"], value_flags=frozenset()) is False
