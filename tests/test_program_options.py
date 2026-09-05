"""Tests for ``cli_support.program_options``: the type-directed CLI projection
shared by engine keys and program parameters, and the ``click`` command a
``program def`` is built into.
"""

from __future__ import annotations

import click
import pytest

from agm.agl.attributes import ProgramOptionSpec
from agm.agl.ir.reserved_nominals import require_reserved_nominal_id
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.runtime.arguments import decode_param_value
from agm.agl.runtime.option import none_value, some_value
from agm.agl.runtime.types import ProgramDeclInfo, ProgramParamInfo
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
from agm.agl.zones import ParamZone
from agm.cli_support.program_options import (
    DuplicateOptionFlagError,
    ProgramCommand,
    ReservedFlagError,
    ValueForm,
    build_program_command,
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
    external: str | None = None,
    short: str | None = None,
    env: str | None = None,
    metavar: str | None = None,
    hidden: bool = False,
    doc: str | None = None,
) -> ProgramParamInfo:
    return ProgramParamInfo(
        name=name,
        kind=kind,
        type=typ,
        has_default=has_default,
        span=_SPAN,
        cli=ProgramOptionSpec(
            name=name if external is None else external,
            short=short,
            env=env,
            metavar=metavar,
            hidden=hidden,
            doc=doc,
        ),
    )


def _program(*params: ProgramParamInfo, doc: str | None = None) -> ProgramDeclInfo:
    return ProgramDeclInfo(
        module=ENTRY_ID,
        scope_path=(),
        name="main",
        node_id=1,
        span=_SPAN,
        parameters=params,
        is_entry=True,
        doc=doc,
    )


def _command(*params: ProgramParamInfo, doc: str | None = None) -> ProgramCommand:
    result = build_program_command(_program(*params, doc=doc))
    assert isinstance(result, ProgramCommand)
    return result


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
        assert project_option("my-flag", TextType()).flags == ("--my-flag",)

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
        from agm.cli_support.program_options import RESERVED_FLAGS

        assert engine_key_flags() <= RESERVED_FLAGS

    def test_every_flag_the_exec_command_declares_is_reserved(self) -> None:
        """The reserved set is derived from ``agm exec``'s own declarations,
        not restated: an unreserved host flag would silently shadow the
        program parameter advertised under it.
        """
        import typer.main

        from agm import cli
        from agm.cli_support.program_options import RESERVED_FLAGS

        group = typer.main.get_command(cli.app)
        assert isinstance(group, click.Group)
        exec_command = group.commands["exec"]
        declared = {
            flag
            for param in exec_command.params
            if isinstance(param, click.Option)
            for flag in (*param.opts, *param.secondary_opts)
        }

        assert declared <= RESERVED_FLAGS


# ---------------------------------------------------------------------------
# build_program_command: shape and reservation
# ---------------------------------------------------------------------------


class TestBuildProgramCommand:
    def test_positional_only_and_standard_params_are_positional_in_order(self) -> None:
        command = _command(
            _param("first", TextType(), ParamZone.POSITIONAL_ONLY),
            _param("second", TextType(), ParamZone.STANDARD),
            _param("third", TextType(), ParamZone.NAMED_ONLY),
        )

        assert [p.name for p in command.positional] == ["first", "second"]

    def test_standard_and_named_only_params_are_options(self) -> None:
        command = _command(
            _param("first", TextType(), ParamZone.POSITIONAL_ONLY),
            _param("second", TextType(), ParamZone.STANDARD),
            _param("third", TextType(), ParamZone.NAMED_ONLY),
        )

        assert [p.name for p, _projected in command.options] == ["second", "third"]

    def test_the_command_is_named_after_the_program_declaration(self) -> None:
        assert _command().command.name == "main"

    def test_the_programs_own_doc_becomes_the_command_help(self) -> None:
        assert _command(doc="Greets an addressee.").command.help == "Greets an addressee."

    def test_positional_only_param_is_never_reserved_checked(self) -> None:
        assert isinstance(
            build_program_command(_program(_param("help", TextType(), ParamZone.POSITIONAL_ONLY))),
            ProgramCommand,
        )

    def test_named_param_colliding_with_a_builtin_flag_is_rejected(self) -> None:
        result = build_program_command(_program(_param("help", TextType())))

        assert result == ReservedFlagError(parameter="help", flag="--help")

    def test_named_param_colliding_with_an_engine_key_is_rejected(self) -> None:
        result = build_program_command(_program(_param("timeout", TextType())))

        assert result == ReservedFlagError(parameter="timeout", flag="--timeout")

    def test_bool_param_colliding_only_via_its_negative_flag_is_rejected(self) -> None:
        # ``--no-stdlib`` is a reserved built-in host flag with no paired
        # ``--stdlib``, so only a bool param's negative form collides.
        result = build_program_command(_program(_param("stdlib", BoolType())))

        assert result == ReservedFlagError(parameter="stdlib", flag="--no-stdlib")

    def test_bool_negative_colliding_with_another_params_positive_flag_is_rejected(self) -> None:
        result = build_program_command(
            _program(_param("cache", BoolType()), _param("no-cache", BoolType()))
        )

        assert result == DuplicateOptionFlagError(
            first_parameter="cache", second_parameter="no-cache", flag="--no-cache"
        )

    def test_option_negative_colliding_with_another_bool_params_positive_flag_is_rejected(
        self,
    ) -> None:
        result = build_program_command(
            _program(
                _param("region", _option_type(TextType())),
                _param("no-region", BoolType()),
            )
        )

        assert result == DuplicateOptionFlagError(
            first_parameter="region", second_parameter="no-region", flag="--no-region"
        )

    def test_external_name_is_what_the_reservation_check_sees(self) -> None:
        result = build_program_command(_program(_param("tag", TextType(), external="help")))

        assert result == ReservedFlagError(parameter="tag", flag="--help")

    def test_two_params_renamed_onto_the_same_external_name_collide(self) -> None:
        result = build_program_command(
            _program(
                _param("first", TextType(), external="shared"),
                _param("second", TextType(), external="shared"),
            )
        )

        assert result == DuplicateOptionFlagError(
            first_parameter="first", second_parameter="second", flag="--shared"
        )

    def test_short_option_colliding_with_a_reserved_host_short_is_rejected(self) -> None:
        result = build_program_command(_program(_param("path", TextType(), short="p")))

        assert result == ReservedFlagError(parameter="path", flag="-p")

    def test_short_option_colliding_with_another_params_short_is_rejected(self) -> None:
        result = build_program_command(
            _program(
                _param("alpha", TextType(), short="a"),
                _param("author", TextType(), short="a"),
            )
        )

        assert result == DuplicateOptionFlagError(
            first_parameter="alpha", second_parameter="author", flag="-a"
        )

    def test_distinct_shorts_are_accepted(self) -> None:
        command = _command(
            _param("alpha", TextType(), short="a"),
            _param("beta", TextType(), short="b"),
        )

        assert [p.name for p, _projected in command.options] == ["alpha", "beta"]


class TestCommandParameterPresentation:
    """``@opt-*`` presentation facts reach the built command's own parameters."""

    def _option(self, command: ProgramCommand, flag: str) -> object:
        return next(param for param in command.command.params if flag in getattr(param, "opts", ()))

    def test_metavar_reaches_the_option(self) -> None:
        command = _command(_param("count", IntType(), metavar="N"))

        assert getattr(self._option(command, "--count"), "metavar") == "N"

    def test_doc_becomes_the_options_help(self) -> None:
        command = _command(_param("name", TextType(), doc="who to greet"))

        assert getattr(self._option(command, "--name"), "help") == "who to greet"

    def test_hidden_reaches_the_option(self) -> None:
        command = _command(_param("secret", TextType(), hidden=True))

        assert getattr(self._option(command, "--secret"), "hidden") is True

    def test_env_becomes_the_options_envvar(self) -> None:
        command = _command(_param("who", TextType(), env="GREET_WHO"))

        assert getattr(self._option(command, "--who"), "envvar") == "GREET_WHO"

    def test_short_option_is_declared_beside_the_long_one(self) -> None:
        command = _command(_param("name", TextType(), short="n"))

        assert "-n" in getattr(self._option(command, "--name"), "opts")

    def test_external_name_spells_the_flag(self) -> None:
        command = _command(_param("who", TextType(), external="addressee"))

        assert getattr(self._option(command, "--addressee"), "opts") == ["--addressee"]


# ---------------------------------------------------------------------------
# ProgramCommand.parse
# ---------------------------------------------------------------------------


class TestParseText:
    def test_named_value_flag(self) -> None:
        args = _command(_param("name", TextType())).parse(["--name", "hello"])
        assert args.positional == ()
        assert args.named == {"name": "hello"}

    def test_equals_form(self) -> None:
        args = _command(_param("name", TextType())).parse(["--name=hello"])
        assert args.named == {"name": "hello"}

    def test_equals_form_preserves_embedded_equals(self) -> None:
        args = _command(_param("expr", TextType())).parse(["--expr=a=b"])
        assert args.named == {"expr": "a=b"}

    def test_missing_value_raises(self) -> None:
        with pytest.raises(ValueError):
            _command(_param("name", TextType())).parse(["--name"])

    def test_equals_form_with_empty_value_yields_empty_string(self) -> None:
        args = _command(_param("name", TextType())).parse(["--name="])
        assert args.named == {"name": ""}

    def test_a_flag_shaped_token_is_taken_as_the_preceding_options_value(self) -> None:
        """Click's convention: an option that requires a value consumes the
        next token whatever it looks like, so ``--msg --x`` supplies ``--x``."""
        args = _command(_param("msg", TextType())).parse(["--msg", "--x"])
        assert args.named == {"msg": "--x"}

    def test_a_repeated_option_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="more than once"):
            _command(_param("name", TextType())).parse(["--name", "a", "--name", "b"])

    def test_a_repeated_option_error_names_the_flag(self) -> None:
        with pytest.raises(ValueError, match=r"'--name'"):
            _command(_param("name", TextType())).parse(["--name", "a", "--name", "b"])


class TestParseShortOptions:
    def test_separate_value(self) -> None:
        args = _command(_param("name", TextType(), short="n")).parse(["-n", "hello"])
        assert args.named == {"name": "hello"}

    def test_attached_value(self) -> None:
        args = _command(_param("name", TextType(), short="n")).parse(["-nhello"])
        assert args.named == {"name": "hello"}

    def test_bundled_bool_flags(self) -> None:
        args = _command(
            _param("alpha", BoolType(), short="a"),
            _param("beta", BoolType(), short="b"),
        ).parse(["-ab"])
        assert args.named == {"alpha": True, "beta": True}

    def test_bundled_bool_flag_followed_by_a_value_option(self) -> None:
        args = _command(
            _param("alpha", BoolType(), short="a"),
            _param("name", TextType(), short="n"),
        ).parse(["-an", "hello"])
        assert args.named == {"alpha": True, "name": "hello"}

    def test_a_repetition_across_the_short_and_long_spellings_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="more than once"):
            _command(_param("name", TextType(), short="n")).parse(["-n", "a", "--name", "b"])

    def test_short_option_for_an_externally_renamed_parameter(self) -> None:
        args = _command(_param("who", TextType(), external="addressee", short="a")).parse(
            ["-a", "agm"]
        )
        assert args.named == {"who": "agm"}


class TestParseBool:
    def test_positive_flag_is_true(self) -> None:
        args = _command(_param("verbose", BoolType())).parse(["--verbose"])
        assert args.named == {"verbose": True}

    def test_negative_flag_is_false(self) -> None:
        args = _command(_param("verbose", BoolType())).parse(["--no-verbose"])
        assert args.named == {"verbose": False}

    def test_value_for_a_bool_flag_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            _command(_param("verbose", BoolType())).parse(["--verbose=true"])

    def test_negative_flag_with_value_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            _command(_param("verbose", BoolType())).parse(["--no-verbose=true"])

    def test_a_repeated_positive_flag_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"'--verbose' specified more than once"):
            _command(_param("verbose", BoolType())).parse(["--verbose", "--verbose"])

    def test_both_polarities_are_rejected_as_one_repetition(self) -> None:
        with pytest.raises(ValueError, match=r"'--no-verbose' specified more than once"):
            _command(_param("verbose", BoolType())).parse(["--verbose", "--no-verbose"])

    def test_the_negative_then_positive_order_names_the_positive_spelling(self) -> None:
        with pytest.raises(ValueError, match=r"'--verbose' specified more than once"):
            _command(_param("verbose", BoolType())).parse(["--no-verbose", "--verbose"])

    def test_a_repeated_negative_flag_names_the_spelling_as_typed(self) -> None:
        with pytest.raises(ValueError, match=r"'--no-verbose' specified more than once"):
            _command(_param("verbose", BoolType())).parse(["--no-verbose", "--no-verbose"])

    def test_omitted_bool_supplies_nothing(self) -> None:
        args = _command(_param("verbose", BoolType())).parse([])
        assert args.named == {}


class TestParseJson:
    def test_json_form_value_stays_a_raw_string(self) -> None:
        args = _command(_param("count", IntType())).parse(["--count", "42"])
        assert args.named == {"count": "42"}

    def test_array_typed_value_stays_a_raw_string(self) -> None:
        args = _command(_param("tags", ArrayType(elem=TextType()))).parse(["--tags", '["a","b"]'])
        assert args.named == {"tags": '["a","b"]'}

    def test_equals_form_with_empty_value_yields_empty_string(self) -> None:
        args = _command(_param("count", IntType())).parse(["--count="])
        assert args.named == {"count": ""}

    def test_a_repeated_option_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"'--n' specified more than once"):
            _command(_param("n", IntType())).parse(["--n", "1", "--n", "2"])


class TestParseOption:
    def test_positive_text_option_wraps_the_value_verbatim(self) -> None:
        args = _command(_param("region", _option_type(TextType()))).parse(["--region", "eu"])
        assert args.named == {"region": {"$case": "Some", "value": "eu"}}

    def test_negative_option_flag_is_none(self) -> None:
        args = _command(_param("region", _option_type(TextType()))).parse(["--no-region"])
        assert args.named == {"region": {"$case": "None"}}

    def test_positive_int_option_json_parses_the_value(self) -> None:
        args = _command(_param("count", _option_type(IntType()))).parse(["--count", "5"])
        assert args.named == {"count": {"$case": "Some", "value": 5}}

    def test_positive_bool_option_json_parses_the_value(self) -> None:
        args = _command(_param("flag", _option_type(BoolType()))).parse(["--flag", "true"])
        assert args.named == {"flag": {"$case": "Some", "value": True}}

    def test_positive_nested_option_json_parses_the_whole_inner_shape(self) -> None:
        args = _command(_param("nested", _option_type(_option_type(IntType())))).parse(
            ["--nested", '{"$case": "Some", "value": 7}']
        )
        assert args.named == {"nested": {"$case": "Some", "value": {"$case": "Some", "value": 7}}}

    def test_malformed_option_value_raises_immediately(self) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            _command(_param("count", _option_type(IntType()))).parse(["--count", "not-json"])

    def test_malformed_option_value_error_names_the_flag(self) -> None:
        with pytest.raises(ValueError, match=r"--count"):
            _command(_param("count", _option_type(IntType()))).parse(["--count", "not-json"])

    def test_text_option_empty_inline_value_is_a_verbatim_empty_string(self) -> None:
        args = _command(_param("region", _option_type(TextType()))).parse(["--region="])
        assert args.named == {"region": {"$case": "Some", "value": ""}}

    def test_json_option_empty_inline_value_raises(self) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            _command(_param("count", _option_type(IntType()))).parse(["--count="])

    def test_a_repeated_positive_flag_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"'--region' specified more than once"):
            _command(_param("region", _option_type(TextType()))).parse(
                ["--region", "a", "--region", "b"]
            )

    def test_a_repeated_negative_flag_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"'--no-region' specified more than once"):
            _command(_param("region", _option_type(TextType()))).parse(
                ["--no-region", "--no-region"]
            )

    def test_both_polarities_together_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            _command(_param("region", _option_type(TextType()))).parse(
                ["--region", "eu", "--no-region"]
            )

    def test_omitted_option_supplies_nothing(self) -> None:
        args = _command(_param("region", _option_type(TextType()))).parse([])
        assert args.named == {}


class TestParsePositional:
    def test_positional_tokens_collected_in_order(self) -> None:
        args = _command(
            _param("first", TextType(), ParamZone.POSITIONAL_ONLY),
            _param("second", TextType(), ParamZone.POSITIONAL_ONLY),
        ).parse(["a", "b"])
        assert args.positional == ("a", "b")
        assert args.named == {}

    def test_positional_and_options_interleave(self) -> None:
        args = _command(
            _param("first", TextType(), ParamZone.POSITIONAL_ONLY),
            _param("name", TextType()),
        ).parse(["a", "--name", "hello"])
        assert args.positional == ("a",)
        assert args.named == {"name": "hello"}

    def test_more_positional_tokens_than_declared_are_collected_unconditionally(self) -> None:
        """No cap here: an excess positional argument is diagnosed downstream
        by the shared zone binder (``bind_program_arguments``), which reports
        a different message depending on whether the program declares any
        named-only parameters — this parser defers to it rather than
        duplicating (and pre-empting) that diagnosis."""
        args = _command(_param("first", TextType(), ParamZone.POSITIONAL_ONLY)).parse(["a", "b"])
        assert args.positional == ("a", "b")

    def test_bare_token_with_no_positional_params_is_still_collected(self) -> None:
        args = _command(_param("name", TextType())).parse(["bare"])
        assert args.positional == ("bare",)
        assert args.named == {}

    def test_double_dash_ends_options(self) -> None:
        args = _command(_param("first", TextType(), ParamZone.POSITIONAL_ONLY)).parse(
            ["--", "--not-a-flag"]
        )
        assert args.positional == ("--not-a-flag",)

    def test_double_dash_with_no_positional_params_still_collects_the_rest(self) -> None:
        assert _command().parse(["--", "x"]).positional == ("x",)

    def test_a_bare_leading_dash_token_is_an_unknown_option(self) -> None:
        """``@opt-short`` makes single-dash spellings real options, so a bare
        ``-5`` is an unknown option rather than a positional token; a
        leading-dash value is spelled after ``--``, or attached to its own
        option (see the two tests below)."""
        with pytest.raises(ValueError):
            _command(_param("first", IntType(), ParamZone.POSITIONAL_ONLY)).parse(["-5"])

    def test_a_letter_shaped_single_dash_token_is_an_unknown_option(self) -> None:
        """The letter shape of the same rule: ``-x`` is a short option Click
        knows nothing about, not a positional token."""
        with pytest.raises(ValueError, match="-x"):
            _command(_param("first", TextType(), ParamZone.POSITIONAL_ONLY)).parse(["-x"])

    def test_a_leading_dash_positional_is_spelled_after_the_end_of_options_marker(self) -> None:
        args = _command(_param("first", IntType(), ParamZone.POSITIONAL_ONLY)).parse(["--", "-5"])
        assert args.positional == ("-5",)

    def test_a_leading_dash_option_value_is_spelled_inline_or_as_the_next_token(self) -> None:
        param = _param("n", IntType())
        assert _command(param).parse(["--n=-5"]).named == {"n": "-5"}
        assert _command(param).parse(["--n", "-5"]).named == {"n": "-5"}


class TestParseErrors:
    def test_unknown_flag_raises(self) -> None:
        with pytest.raises(ValueError):
            _command(_param("name", TextType())).parse(["--unknown"])

    def test_unknown_flag_in_equals_form_raises(self) -> None:
        with pytest.raises(ValueError):
            _command(_param("name", TextType())).parse(["--unknown=value"])

    def test_unknown_flag_error_names_the_flag(self) -> None:
        with pytest.raises(ValueError, match="--unknown"):
            _command(_param("name", TextType())).parse(["--unknown"])

    def test_empty_tokens_returns_empty_arguments(self) -> None:
        args = _command(_param("name", TextType())).parse([])
        assert args.positional == ()
        assert args.named == {}

    def test_no_params_empty_tokens(self) -> None:
        args = _command().parse([])
        assert args.positional == ()
        assert args.named == {}


class TestParseEnvironment:
    def test_env_supplies_an_omitted_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GREET_WHO", "world")
        args = _command(_param("who", TextType(), env="GREET_WHO")).parse([])
        assert args.named == {"who": "world"}

    def test_a_cli_token_overrides_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GREET_WHO", "world")
        args = _command(_param("who", TextType(), env="GREET_WHO")).parse(["--who", "agm"])
        assert args.named == {"who": "agm"}

    def test_env_supplies_a_bool_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VERBOSE", "true")
        args = _command(_param("verbose", BoolType(), env="VERBOSE")).parse([])
        assert args.named == {"verbose": True}

    def test_env_supplies_an_option_value_wrapped_some(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REGION", "eu")
        args = _command(_param("region", _option_type(TextType()), env="REGION")).parse([])
        assert args.named == {"region": {"$case": "Some", "value": "eu"}}

    def test_an_unset_variable_supplies_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GREET_WHO", raising=False)
        args = _command(_param("who", TextType(), env="GREET_WHO")).parse([])
        assert args.named == {}

    def test_an_environment_value_is_taken_verbatim_as_one_occurrence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A path-separator-bearing value is one value, not several: an
        environment fallback can never look like a repeated flag."""
        monkeypatch.setenv("GREET_WHO", "a:b")
        args = _command(_param("who", TextType(), env="GREET_WHO")).parse([])
        assert args.named == {"who": "a:b"}

    def test_the_negative_flag_beats_an_environment_supplied_option_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A CLI token outranks an environment value, so ``--no-x`` against an
        ``@opt-env``-supplied positive binds ``None`` rather than colliding."""
        monkeypatch.setenv("REGION", "eu")
        args = _command(_param("region", _option_type(TextType()), env="REGION")).parse(
            ["--no-region"]
        )
        assert args.named == {"region": {"$case": "None"}}


# ---------------------------------------------------------------------------
# The decode seam: every raw value ``parse`` emits must decode through
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
        args = _command(_param(name, typ)).parse(list(tokens))
        assert _decode(typ, args.named[name]) == expected


# ---------------------------------------------------------------------------
# usage_line / render_help_section / completion_items
# ---------------------------------------------------------------------------


class TestUsageLine:
    def test_required_positional_uses_angle_brackets(self) -> None:
        command = _command(_param("file", TextType(), ParamZone.POSITIONAL_ONLY))
        assert command.usage_line("main") == "main <file>"

    def test_defaulted_positional_uses_square_brackets(self) -> None:
        command = _command(_param("file", TextType(), ParamZone.POSITIONAL_ONLY, has_default=True))
        assert command.usage_line("main") == "main [file]"

    def test_options_are_summarized_when_present(self) -> None:
        assert _command(_param("name", TextType())).usage_line("main") == "main [OPTIONS]"

    def test_no_positional_or_options_is_just_the_program_name(self) -> None:
        assert _command().usage_line("main") == "main"

    def test_a_renamed_standard_parameter_shows_its_external_name(self) -> None:
        command = _command(_param("who", TextType(), ParamZone.STANDARD, external="addressee"))
        assert command.usage_line("main") == "main <addressee> [OPTIONS]"


class TestOptionLines:
    def test_no_options_describes_nothing(self) -> None:
        assert _command(_param("file", TextType(), ParamZone.POSITIONAL_ONLY)).option_lines() == ()

    def test_one_line_per_name_addressable_parameter(self) -> None:
        described = _command(
            _param("name", TextType()), _param("verbose", BoolType())
        ).option_lines()
        assert len(described) == 2
        assert described[0].startswith("--name")
        assert described[1].startswith("--verbose")

    def test_lines_are_unindented_and_carry_no_header(self) -> None:
        described = _command(_param("name", TextType())).option_lines()
        assert "Options:" not in described
        assert described[0] == described[0].lstrip()

    def test_render_help_section_indents_the_same_lines_under_a_header(self) -> None:
        command = _command(_param("name", TextType()), _param("verbose", BoolType()))
        section = command.render_help_section().splitlines()
        assert section[0] == "Options:"
        assert section[1:] == [f"  {line}" for line in command.option_lines()]

    def test_a_renamed_parameter_is_described_by_its_external_flag(self) -> None:
        described = _command(_param("who", TextType(), external="addressee")).option_lines()
        assert described[0].startswith("--addressee")


class TestRenderHelpSection:
    def test_no_options_renders_nothing(self) -> None:
        command = _command(_param("file", TextType(), ParamZone.POSITIONAL_ONLY))
        assert command.render_help_section() == ""

    def test_starts_with_the_options_header(self) -> None:
        assert _command(_param("name", TextType())).render_help_section().startswith("Options:\n")

    def test_required_option_is_marked_required(self) -> None:
        section = _command(_param("name", TextType(), has_default=False)).render_help_section()
        assert "--name" in section
        assert "(required)" in section

    def test_defaulted_option_is_marked_optional(self) -> None:
        section = _command(_param("name", TextType(), has_default=True)).render_help_section()
        assert "(required)" not in section
        assert "optional" in section
        assert "default" in section

    def test_bool_option_shows_both_flag_polarities_and_no_value_placeholder(self) -> None:
        section = _command(_param("verbose", BoolType())).render_help_section()
        assert "--verbose" in section
        assert "--no-verbose" in section
        assert "VALUE" not in section

    def test_option_shape_shows_both_polarities_and_a_value_placeholder(self) -> None:
        section = _command(_param("region", _option_type(TextType()))).render_help_section()
        assert "--region" in section
        assert "--no-region" in section
        assert "VALUE" in section

    def test_plain_value_shape_shows_no_negative_and_a_value_placeholder(self) -> None:
        section = _command(_param("name", TextType())).render_help_section()
        assert "--name VALUE" in section
        assert "--no-name" not in section


class TestPositionalOnlyNames:
    def test_only_positional_only_parameters_are_listed(self) -> None:
        command = _command(
            _param("first", TextType(), ParamZone.POSITIONAL_ONLY),
            _param("second", TextType(), ParamZone.STANDARD),
        )
        assert command.positional_only_names() == frozenset({"first"})


class TestCompletionItems:
    def test_includes_positive_and_negative_flags(self) -> None:
        assert _command(_param("verbose", BoolType())).completion_items() == (
            "--verbose",
            "--no-verbose",
        )

    def test_positional_only_params_contribute_no_items(self) -> None:
        command = _command(_param("file", TextType(), ParamZone.POSITIONAL_ONLY))
        assert command.completion_items() == ()

    def test_text_option_contributes_only_its_positive_flag(self) -> None:
        assert _command(_param("name", TextType())).completion_items() == ("--name",)

    def test_a_renamed_parameter_is_completed_by_its_external_flag(self) -> None:
        command = _command(_param("who", TextType(), external="addressee"))
        assert command.completion_items() == ("--addressee",)


class TestValueTakingFlags:
    def test_bool_option_contributes_no_value_taking_flag(self) -> None:
        assert _command(_param("verbose", BoolType())).value_taking_flags() == frozenset()

    def test_positional_only_params_contribute_no_value_taking_flag(self) -> None:
        command = _command(_param("file", TextType(), ParamZone.POSITIONAL_ONLY))
        assert command.value_taking_flags() == frozenset()

    def test_text_and_option_shapes_contribute_their_positive_flag_only(self) -> None:
        command = _command(_param("name", TextType()), _param("region", _option_type(TextType())))
        assert command.value_taking_flags() == frozenset({"--name", "--region"})

    def test_a_short_spelling_of_a_value_taking_option_is_included(self) -> None:
        command = _command(_param("name", TextType(), short="n"))
        assert command.value_taking_flags() == frozenset({"--name", "-n"})


class TestProgramCommandFor:
    def test_returns_the_built_command_for_a_valid_signature(self) -> None:
        from agm.cli_support.program_options import program_command_for

        command = program_command_for(_program(_param("name", TextType())))

        assert isinstance(command, ProgramCommand)
        assert command.completion_items() == ("--name",)

    def test_degrades_to_none_on_a_reservation_collision(self) -> None:
        from agm.cli_support.program_options import program_command_for

        assert program_command_for(_program(_param("help", TextType()))) is None

    def test_no_program_is_none(self) -> None:
        from agm.cli_support.program_options import program_command_for

        assert program_command_for(None) is None


class TestProgramValueTakingFlags:
    def test_no_program_has_no_value_taking_flags(self) -> None:
        from agm.cli_support.program_options import program_value_taking_flags

        assert program_value_taking_flags(None) == frozenset()

    def test_a_programs_value_taking_flags_are_reported(self) -> None:
        from agm.cli_support.program_options import program_value_taking_flags

        assert program_value_taking_flags(_program(_param("name", TextType()))) == frozenset(
            {"--name"}
        )


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

    def test_a_flag_never_serves_as_another_flags_value(self) -> None:
        """``--a --b -h`` binds ``-h`` to ``--b``."""
        from agm.cli_support.program_options import short_help_requested

        assert (
            short_help_requested(["--a", "--b", "-h"], value_flags=frozenset({"--a", "--b"}))
            is False
        )
