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
    BUILTIN_PRELUDE_TYPES,
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
    EXEC_RESERVED_FLAGS,
    REGISTERED_RESERVED_FLAGS,
    DuplicateOptionFlagError,
    ExecTail,
    ProgramCommand,
    ProgramHelpRequested,
    ReservedFlagError,
    ValueForm,
    build_program_command,
    engine_key_flags,
    exec_program_name,
    native_raw_value,
    program_help_requested,
    project_option,
    protect_host_option_values,
    split_exec_tail,
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
    result = build_program_command(_program(*params, doc=doc), EXEC_RESERVED_FLAGS)
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

    def test_agent_projects_a_host_agent_value(self) -> None:
        projected = project_option("worker", BUILTIN_PRELUDE_TYPES["Agent"])

        assert projected.negative_flags == ()
        assert projected.takes_value is True
        assert projected.value_form is ValueForm.AGENT

    def test_every_other_type_projects_a_json_value_with_no_negative(self) -> None:
        for typ in (IntType(), ArrayType(elem=TextType())):
            projected = project_option("count", typ)
            assert projected.negative_flags == ()
            assert projected.takes_value is True
            assert projected.value_form is ValueForm.JSON

    def test_user_enum_named_agent_still_projects_as_json(self) -> None:
        user_agent = EnumType(name="Agent", module_id=ENTRY_ID, decl_id=next_decl_id())

        assert project_option("worker", user_agent).value_form is ValueForm.JSON

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

    def test_exec_reserves_every_engine_key_flag(self) -> None:
        assert engine_key_flags() <= EXEC_RESERVED_FLAGS

    def test_every_flag_the_exec_command_declares_is_reserved(self) -> None:
        """The reserved set is derived from ``agm exec``'s own declarations,
        not restated: an unreserved host flag would silently shadow the
        program parameter advertised under it.
        """
        import typer.main

        from agm import cli

        group = typer.main.get_command(cli.app)
        assert isinstance(group, click.Group)
        exec_command = group.commands["exec"]
        declared = {
            flag
            for param in exec_command.params
            if isinstance(param, click.Option)
            for flag in (*param.opts, *param.secondary_opts)
        }

        assert declared <= EXEC_RESERVED_FLAGS

    def test_every_flag_a_registered_command_declares_is_reserved(self) -> None:
        """Derived from the registered command's own declarations, as for ``agm exec``."""
        from agm.cli_dispatch import RegisteredProgramCommand
        from agm.packages.activation import CommandRegistration

        command = RegisteredProgramCommand(
            "tools run", CommandRegistration("tools", "tools/run::main")
        )
        declared = {
            flag
            for param in command.params
            if isinstance(param, click.Option)
            for flag in (*param.opts, *param.secondary_opts)
        }

        assert declared <= REGISTERED_RESERVED_FLAGS


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
            build_program_command(
                _program(_param("help", TextType(), ParamZone.POSITIONAL_ONLY)), EXEC_RESERVED_FLAGS
            ),
            ProgramCommand,
        )

    def test_named_param_colliding_with_a_builtin_flag_is_rejected(self) -> None:
        result = build_program_command(_program(_param("help", TextType())), EXEC_RESERVED_FLAGS)

        assert result == ReservedFlagError(parameter="help", flag="--help")

    def test_named_param_colliding_with_an_engine_key_is_rejected(self) -> None:
        result = build_program_command(_program(_param("timeout", TextType())), EXEC_RESERVED_FLAGS)

        assert result == ReservedFlagError(parameter="timeout", flag="--timeout")

    def test_bool_param_colliding_only_via_its_negative_flag_is_rejected(self) -> None:
        # ``--no-stdlib`` is a reserved built-in host flag with no paired
        # ``--stdlib``, so only a bool param's negative form collides.
        result = build_program_command(_program(_param("stdlib", BoolType())), EXEC_RESERVED_FLAGS)

        assert result == ReservedFlagError(parameter="stdlib", flag="--no-stdlib")

    def test_bool_negative_colliding_with_another_params_positive_flag_is_rejected(self) -> None:
        result = build_program_command(
            _program(_param("cache", BoolType()), _param("no-cache", BoolType())),
            EXEC_RESERVED_FLAGS,
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
            ),
            EXEC_RESERVED_FLAGS,
        )

        assert result == DuplicateOptionFlagError(
            first_parameter="region", second_parameter="no-region", flag="--no-region"
        )

    def test_external_name_is_what_the_reservation_check_sees(self) -> None:
        result = build_program_command(
            _program(_param("tag", TextType(), external="help")), EXEC_RESERVED_FLAGS
        )

        assert result == ReservedFlagError(parameter="tag", flag="--help")

    def test_two_params_renamed_onto_the_same_external_name_collide(self) -> None:
        result = build_program_command(
            _program(
                _param("first", TextType(), external="shared"),
                _param("second", TextType(), external="shared"),
            ),
            EXEC_RESERVED_FLAGS,
        )

        assert result == DuplicateOptionFlagError(
            first_parameter="first", second_parameter="second", flag="--shared"
        )

    def test_short_option_colliding_with_a_reserved_host_short_is_rejected(self) -> None:
        result = build_program_command(
            _program(_param("path", TextType(), short="p")), EXEC_RESERVED_FLAGS
        )

        assert result == ReservedFlagError(parameter="path", flag="-p")

    def test_exec_only_flags_are_free_on_a_registered_command(self) -> None:
        result = build_program_command(
            _program(
                _param("agent", BUILTIN_PRELUDE_TYPES["Agent"]),
                _param("stdlib", BoolType()),
                _param("path", TextType(), short="p"),
            ),
            REGISTERED_RESERVED_FLAGS,
        )

        assert isinstance(result, ProgramCommand)

    def test_exec_leaves_the_agent_flag_to_the_program(self) -> None:
        result = build_program_command(
            _program(_param("agent", BUILTIN_PRELUDE_TYPES["Agent"])), EXEC_RESERVED_FLAGS
        )

        assert isinstance(result, ProgramCommand)

    def test_registered_command_reserves_its_own_dry_run_flag(self) -> None:
        result = build_program_command(
            _program(_param("dry-run", BoolType())), REGISTERED_RESERVED_FLAGS
        )

        assert result == ReservedFlagError(parameter="dry-run", flag="--dry-run")

    @pytest.mark.parametrize("reserved", [EXEC_RESERVED_FLAGS, REGISTERED_RESERVED_FLAGS])
    def test_help_flags_are_reserved_on_every_surface(self, reserved: frozenset[str]) -> None:
        result = build_program_command(_program(_param("tag", TextType(), short="h")), reserved)

        assert result == ReservedFlagError(parameter="tag", flag="-h")

    def test_short_option_colliding_with_another_params_short_is_rejected(self) -> None:
        result = build_program_command(
            _program(
                _param("alpha", TextType(), short="a"),
                _param("author", TextType(), short="a"),
            ),
            EXEC_RESERVED_FLAGS,
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

    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            (
                "claude/sonnet-custom",
                {
                    "$case": "AgentClaude",
                    "model": "sonnet",
                    "thinking": "custom",
                },
            ),
            (
                '{"$case":"AgentCommand","command":"worker --flag"}',
                {"$case": "AgentCommand", "command": "worker --flag"},
            ),
        ],
    )
    def test_positive_agent_option_accepts_host_syntax_and_tagged_json(
        self, token: str, expected: dict[str, object]
    ) -> None:
        args = _command(_param("worker", _option_type(BUILTIN_PRELUDE_TYPES["Agent"]))).parse(
            ["--worker", token]
        )

        assert args.named == {"worker": {"$case": "Some", "value": expected}}

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

    def test_a_positional_standard_argument_overrides_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GREET_WHO", "world")
        args = _command(_param("who", TextType(), ParamZone.STANDARD, env="GREET_WHO")).parse(
            ["agm"]
        )
        assert args.positional == ("agm",)
        assert args.named == {}

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


class TestNativeRawValue:
    def test_config_agent_option_accepts_host_syntax(self) -> None:
        projected = project_option("worker", _option_type(BUILTIN_PRELUDE_TYPES["Agent"]))

        assert native_raw_value(projected, "codex/o3-high") == {
            "$case": "Some",
            "value": {"$case": "AgentCodex", "model": "o3", "thinking": "high"},
        }

    def test_config_agent_option_keeps_a_native_tagged_object(self) -> None:
        projected = project_option("worker", _option_type(BUILTIN_PRELUDE_TYPES["Agent"]))
        tagged = {"$case": "AgentCommand", "command": "worker"}

        assert native_raw_value(projected, tagged) == {"$case": "Some", "value": tagged}


class TestDecodeSeam:
    @pytest.mark.parametrize(("name", "typ", "tokens", "expected"), _DECODE_SEAM_CASES)
    def test_raw_value_decodes_to_the_expected_value(
        self, name: str, typ: Type, tokens: tuple[str, ...], expected: Value
    ) -> None:
        args = _command(_param(name, typ)).parse(list(tokens))
        assert _decode(typ, args.named[name]) == expected


# ---------------------------------------------------------------------------
# help rendering, help recognition, and completion spellings
# ---------------------------------------------------------------------------


class TestRenderHelp:
    def test_usage_line_names_the_invocation_and_its_positional_slots(self) -> None:
        command = _command(
            _param("file", TextType(), ParamZone.POSITIONAL_ONLY),
            _param("tag", TextType(), ParamZone.STANDARD, has_default=True),
        )

        usage = command.render_help("agm exec prog.agl").splitlines()[0]

        assert usage.startswith("Usage: agm exec prog.agl")
        assert "<file>" in usage
        assert "[tag]" in usage
        assert "_positional" not in usage.lower()

    def test_a_renamed_standard_parameter_shows_its_external_name(self) -> None:
        command = _command(_param("who", TextType(), ParamZone.STANDARD, external="addressee"))

        usage = command.render_help("main").splitlines()[0]

        assert "<addressee>" in usage
        assert "who" not in usage

    def test_a_hidden_standard_parameter_keeps_its_positional_slot(self) -> None:
        """Hiding is about the ``--name`` entry only: the parameter still takes a
        positional token, so the usage line must keep showing its slot."""
        command = _command(
            _param("secret", TextType(), ParamZone.STANDARD, hidden=True),
            _param("name", TextType(), ParamZone.STANDARD),
        )

        text = command.render_help("main")
        usage = text.splitlines()[0]

        assert "<secret>" in usage
        assert "<name>" in usage
        assert "--secret" not in text

    def test_a_declared_metavar_names_the_positional_slot(self) -> None:
        command = _command(
            _param("file", TextType(), ParamZone.POSITIONAL_ONLY, metavar="PATH"),
            _param("tag", TextType(), ParamZone.STANDARD, has_default=True, metavar="LABEL"),
        )

        usage = command.render_help("main").splitlines()[0]

        assert "<PATH>" in usage
        assert "[LABEL]" in usage
        assert "file" not in usage
        assert "tag" not in usage

    def test_the_program_doc_is_the_description(self) -> None:
        command = _command(_param("tag", TextType()), doc="Tags one artifact.")

        assert "Tags one artifact." in command.render_help("main")

    def test_a_supplied_description_overrides_the_program_doc(self) -> None:
        command = _command(_param("tag", TextType()), doc="Tags one artifact.")

        text = command.render_help("main", description="Publish a subject")

        assert "Publish a subject" in text
        assert "Tags one artifact." not in text

    def test_each_parameter_doc_is_rendered_beside_its_flag(self) -> None:
        command = _command(_param("tag", TextType(), doc="The tag to apply."))

        text = command.render_help("main")

        assert "--tag" in text
        assert "The tag to apply." in text

    def test_a_declared_metavar_stands_for_the_value(self) -> None:
        command = _command(_param("tag", TextType(), metavar="LABEL"))

        assert "LABEL" in command.render_help("main")

    def test_a_hidden_parameter_is_absent_from_help(self) -> None:
        command = _command(
            _param("tag", TextType()), _param("secret", TextType(), hidden=True, doc="Internal.")
        )

        text = command.render_help("main")

        assert "--tag" in text
        assert "--secret" not in text
        assert "Internal." not in text

    def test_a_short_spelling_is_listed_with_its_long_flag(self) -> None:
        text = _command(_param("tag", TextType(), short="t")).render_help("main")

        assert "-t" in text
        assert "--tag" in text

    def test_both_polarities_of_a_bool_parameter_are_listed(self) -> None:
        text = _command(_param("verbose", BoolType())).render_help("main")

        assert "--verbose" in text
        assert "--no-verbose" in text

    def test_the_help_flags_themselves_are_listed(self) -> None:
        text = _command(_param("tag", TextType())).render_help("main")

        assert "--help" in text
        assert "-h" in text

    def test_extra_options_are_listed_beside_the_program_flags(self) -> None:
        text = _command(_param("tag", TextType())).render_help(
            "main", extra_options=(click.Option(["--dry-run"], is_flag=True, help="Check only."),)
        )

        assert "--tag" in text
        assert "--dry-run" in text
        assert "Check only." in text

    def test_a_parameterless_program_renders_usage_and_options_only(self) -> None:
        text = _command().render_help("main")

        assert text.splitlines()[0].startswith("Usage: main")
        assert "--help" in text


class TestRenderProgramHelp:
    def test_a_built_command_renders_its_own_help(self) -> None:
        from agm.cli_support.program_options import render_program_help

        text = render_program_help(_command(_param("tag", TextType())), program_name="main")

        assert "--tag" in text

    def test_without_a_command_the_surface_is_still_rendered(self) -> None:
        from agm.cli_support.program_options import render_program_help

        text = render_program_help(
            None,
            program_name="agm tools lint",
            description="Lint package inputs",
            extra_options=(click.Option(["--dry-run"], is_flag=True),),
        )

        assert text.splitlines()[0].startswith("Usage: agm tools lint")
        assert "Lint package inputs" in text
        assert "--dry-run" in text


class TestProgramHelpRequested:
    """Click owns ``-h``/``--help``, including where a token is a value."""

    def _command_with_shorts(self) -> ProgramCommand:
        return _command(
            _param("verbose", BoolType(), short="v"),
            _param("addressee", TextType(), short="a"),
        )

    def test_a_bare_short_flag_asks_for_help(self) -> None:
        assert program_help_requested(["-h"], self._command_with_shorts()) is True

    def test_a_long_flag_asks_for_help(self) -> None:
        assert program_help_requested(["--help"], self._command_with_shorts()) is True

    def test_a_short_flag_bundled_with_a_bool_short_asks_for_help(self) -> None:
        assert program_help_requested(["-vh"], self._command_with_shorts()) is True

    def test_a_value_taking_bundle_consumes_the_following_short_flag(self) -> None:
        """``-va -h`` binds ``-h`` as ``-a``'s value, exactly as Click parses it."""
        assert program_help_requested(["-va", "-h"], self._command_with_shorts()) is False

    def test_a_value_taking_flags_own_value_is_not_a_help_request(self) -> None:
        assert program_help_requested(["--addressee", "-h"], self._command_with_shorts()) is False

    def test_a_token_past_the_end_of_options_marker_is_positional(self) -> None:
        assert program_help_requested(["--", "-h"], self._command_with_shorts()) is False

    def test_a_usage_error_before_the_flag_is_not_a_help_request(self) -> None:
        assert program_help_requested(["--nope", "-h"], self._command_with_shorts()) is False

    def test_without_a_command_a_bare_flag_still_asks_for_help(self) -> None:
        assert program_help_requested(["-h"], None) is True

    def test_without_a_command_an_unknown_option_is_still_tolerated(self) -> None:
        assert program_help_requested(["--tag", "x", "--help"], None) is True

    def test_without_a_command_the_end_of_options_marker_still_applies(self) -> None:
        assert program_help_requested(["--", "-h"], None) is False


class TestParseHelpRequest:
    def test_parsing_a_help_token_raises_the_help_request(self) -> None:
        command = _command(_param("tag", TextType()))

        with pytest.raises(ProgramHelpRequested) as exc_info:
            command.parse(["-h"])

        assert exc_info.value.command is command


class TestPositionalOnlyNames:
    def test_only_positional_only_parameters_are_listed(self) -> None:
        command = _command(
            _param("first", TextType(), ParamZone.POSITIONAL_ONLY),
            _param("second", TextType(), ParamZone.STANDARD),
        )
        assert command.positional_only_names() == frozenset({"first"})


class TestDefaultMetavar:
    """A value-taking option's placeholder names how its token is read."""

    def test_a_text_parameter_reads_its_token_verbatim(self) -> None:
        help_text = _command(_param("tag", TextType())).render_help("prog")

        assert "--tag TEXT" in help_text

    def test_a_json_form_parameter_announces_its_token_as_json(self) -> None:
        help_text = _command(_param("count", IntType())).render_help("prog")

        assert "--count JSON" in help_text

    def test_an_option_parameter_follows_its_inner_type(self) -> None:
        help_text = _command(
            _param("tag", _option_type(TextType())), _param("count", _option_type(IntType()))
        ).render_help("prog")

        assert "--tag TEXT" in help_text
        assert "--count JSON" in help_text

    def test_an_explicit_metavar_replaces_the_default(self) -> None:
        help_text = _command(_param("count", IntType(), metavar="N")).render_help("prog")

        assert "--count N" in help_text


class TestOptionSpellings:
    def test_both_polarities_of_a_bool_parameter_are_completable(self) -> None:
        spellings = _command(_param("verbose", BoolType())).option_spellings()

        assert "--verbose" in spellings
        assert "--no-verbose" in spellings

    def test_a_short_spelling_is_completable(self) -> None:
        assert "-t" in _command(_param("tag", TextType(), short="t")).option_spellings()

    def test_a_hidden_parameter_contributes_no_spelling(self) -> None:
        spellings = _command(
            _param("tag", TextType()), _param("secret", TextType(), hidden=True)
        ).option_spellings()

        assert "--tag" in spellings
        assert "--secret" not in spellings

    def test_a_positional_only_parameter_contributes_no_spelling(self) -> None:
        command = _command(_param("file", TextType(), ParamZone.POSITIONAL_ONLY))
        spellings = command.option_spellings()

        assert not any(spelling.startswith("--file") for spelling in spellings)
        assert "_positionals" not in spellings

    def test_a_renamed_parameter_is_completed_by_its_external_flag(self) -> None:
        spellings = _command(_param("who", TextType(), external="addressee")).option_spellings()

        assert "--addressee" in spellings
        assert "--who" not in spellings

    def test_the_help_flags_are_completable(self) -> None:
        spellings = _command(_param("tag", TextType())).option_spellings()

        assert "--help" in spellings
        assert "-h" in spellings


class TestProgramCommandFor:
    def test_returns_the_built_command_for_a_valid_signature(self) -> None:
        from agm.cli_support.program_options import program_command_for

        command = program_command_for(_program(_param("name", TextType())), EXEC_RESERVED_FLAGS)

        assert isinstance(command, ProgramCommand)
        assert "--name" in command.option_spellings()

    def test_degrades_to_none_on_a_reservation_collision(self) -> None:
        from agm.cli_support.program_options import program_command_for

        assert (
            program_command_for(_program(_param("help", TextType())), EXEC_RESERVED_FLAGS) is None
        )

    def test_no_program_is_none(self) -> None:
        from agm.cli_support.program_options import program_command_for

        assert program_command_for(None, EXEC_RESERVED_FLAGS) is None


class TestHostLookingProgramValues:
    """A host must not consume tokens that a program option owns as values."""

    def test_finds_every_separate_value_form(self) -> None:
        command = _command(
            _param("verbose", BoolType()),
            _param("message", TextType()),
            _param("region", _option_type(TextType())),
            _param("tag", TextType(), short="t"),
        )
        tokens = [
            "word",
            "--verbose",
            "--message",
            "--dry-run",
            "--region",
            "--no-stdlib",
            "--no-region",
            "-t",
            "--log",
            "-tagm",
            "--message=inline",
            "--",
            "--message",
        ]

        assert command.value_token_indexes(tokens) == frozenset({3, 5, 8})

    def test_incomplete_and_unknown_short_options_claim_no_value(self) -> None:
        command = _command(_param("message", TextType()), _param("tag", TextType(), short="t"))

        assert command.value_token_indexes(["--message"]) == frozenset()
        assert command.value_token_indexes(["-x", "-t"]) == frozenset()

    def test_protection_only_rewrites_host_spelled_values(self) -> None:
        command = _command(_param("message", TextType()))

        protected, replacements = protect_host_option_values(
            ["--message", "--dry-run"], command, frozenset({"--dry-run"})
        )

        assert protected == ["--message", "agm-program-value-1"]
        assert replacements == {"agm-program-value-1": "--dry-run"}

    @pytest.mark.parametrize("value", ["-pnot-a-program", "-csource", "-Idir"])
    def test_protection_recognizes_attached_host_short_option_values(self, value: str) -> None:
        command = _command(_param("message", TextType()))

        protected, replacements = protect_host_option_values(
            ["--message", value], command, {"-p": True, "-c": True, "-I": True}
        )

        assert protected == ["--message", "agm-program-value-1"]
        assert replacements == {"agm-program-value-1": value}

    def test_host_option_value_is_not_scanned_as_a_program_option(self) -> None:
        command = _command(_param("message", TextType()))

        protected, replacements = protect_host_option_values(
            ["--log-file", "--message", "--dry-run"],
            command,
            {"--log-file": True, "--dry-run": False},
        )

        assert protected == ["--log-file", "--message", "--dry-run"]
        assert replacements == {}

    def test_bare_marker_used_as_a_program_value_is_protected(self) -> None:
        command = _command(_param("message", TextType()))

        protected, replacements = protect_host_option_values(
            ["--message", "--", "--dry-run"], command, {"--dry-run": False}
        )

        assert protected == ["--message", "agm-program-value-1", "--dry-run"]
        assert replacements == {"agm-program-value-1": "--"}

    def test_preview_placeholder_does_not_collide_with_a_literal_token(self) -> None:
        from agm.cli_support.program_options import protect_potential_program_values

        assert protect_potential_program_values(
            ["--message", "--dry-run", "agm-program-preview-1"],
            {"--dry-run": False},
        ) == ["--message", "agm-program-preview-1-1", "agm-program-preview-1"]

    def test_value_scan_walks_past_a_short_flag_in_a_bundle(self) -> None:
        command = _command(
            _param("verbose", BoolType(), short="v"),
            _param("tag", TextType(), short="t"),
        )

        assert command.value_token_indexes(["-vt", "value"]) == frozenset({1})

    def test_protection_uses_a_placeholder_distinct_from_literal_arguments(self) -> None:
        command = _command(_param("message", TextType()))

        protected, replacements = protect_host_option_values(
            ["--message", "--dry-run", "agm-program-value-1"],
            command,
            frozenset({"--dry-run"}),
        )

        assert protected == ["--message", "agm-program-value-1-1", "agm-program-value-1"]
        assert replacements == {"agm-program-value-1-1": "--dry-run"}

    def test_protection_leaves_unselected_and_non_host_values_unchanged(self) -> None:
        command = _command(_param("message", TextType()))

        assert protect_host_option_values(
            ["--message", "value"], command, frozenset({"--dry-run"})
        ) == (
            ["--message", "value"],
            {},
        )
        assert protect_host_option_values(
            ["--message", "--dry-run"], None, frozenset({"--dry-run"})
        ) == (
            ["--message", "--dry-run"],
            {},
        )


class TestExecProgramName:
    def test_a_file_invocation_names_its_source(self) -> None:
        assert exec_program_name(file="prog.agl", program=None) == "agm exec prog.agl"

    def test_a_selected_program_is_part_of_the_invocation(self) -> None:
        assert (
            exec_program_name(file="prog.agl", program="review::main")
            == "agm exec prog.agl -p review::main"
        )

    def test_an_inline_source_is_named_by_its_option(self) -> None:
        assert exec_program_name(file=None, program=None) == "agm exec -c COMMAND"


# ---------------------------------------------------------------------------
# split_exec_tail / retain_end_of_options
# ---------------------------------------------------------------------------


class TestSplitExecTail:
    """``agm exec``'s FILE argument, derived from the tail Click leaves."""

    def _split(self, *tail: str) -> ExecTail:
        from agm.cli_support.program_options import split_exec_tail

        return split_exec_tail(tail)

    def test_a_lone_token_is_the_file(self) -> None:
        assert self._split("prog.agl") == ExecTail(file="prog.agl", tokens=())

    def test_tokens_after_the_file_belong_to_the_program(self) -> None:
        assert self._split("prog.agl", "--name", "x") == ExecTail(
            file="prog.agl", tokens=("--name", "x")
        )

    def test_a_host_help_flag_before_the_file_is_not_the_file(self) -> None:
        assert self._split("-h", "prog.agl") == ExecTail(file="prog.agl", tokens=("-h",))

    def test_a_program_option_before_the_file_takes_its_value_with_it(self) -> None:
        assert self._split("-h", "--name", "x", "prog.agl") == ExecTail(
            file="prog.agl", tokens=("-h", "--name", "x")
        )

    def test_a_short_program_option_before_the_file_takes_its_value_with_it(self) -> None:
        assert self._split("-h", "-n", "x", "prog.agl") == ExecTail(
            file="prog.agl", tokens=("-h", "-n", "x")
        )

    @pytest.mark.parametrize(
        ("tail", "param"),
        (
            (("--verbose", "prog.agl"), _param("verbose", BoolType())),
            (("-nvalue", "prog.agl"), _param("name", TextType(), short="n")),
            (("--name", "--x", "prog.agl"), _param("name", TextType())),
        ),
    )
    def test_an_ambiguous_pre_file_option_uses_the_programs_own_arity(
        self, tail: tuple[str, ...], param: ProgramParamInfo
    ) -> None:
        command = _command(param)
        selected = split_exec_tail(
            tail,
            program_command_for_file=lambda file: command if file == "prog.agl" else None,
        )

        assert selected == ExecTail(file="prog.agl", tokens=tail[:-1])

    def test_an_inline_value_leaves_the_next_token_as_the_file(self) -> None:
        assert self._split("--name=x", "prog.agl") == ExecTail(
            file="prog.agl", tokens=("--name=x",)
        )

    def test_a_host_option_spelled_inline_is_still_the_hosts(self) -> None:
        assert self._split("--program=main", "prog.agl") == ExecTail(
            file="prog.agl", tokens=("--program=main",)
        )

    def test_no_file_at_all(self) -> None:
        assert self._split("--help") == ExecTail(file=None, tokens=("--help",))

    def test_a_lone_dash_is_a_value_not_an_option(self) -> None:
        assert self._split("-") == ExecTail(file="-", tokens=())

    def test_the_token_after_the_marker_is_the_file_however_it_is_spelled(self) -> None:
        assert self._split("--", "--weird.agl", "-h") == ExecTail(
            file="--weird.agl", tokens=("-h",)
        )

    def test_a_marker_after_the_file_is_forwarded_to_the_program(self) -> None:
        """The FILE was named before the marker, so the marker was written for
        the program and reaches its own parser in place."""
        assert self._split("prog.agl", "--", "-x") == ExecTail(file="prog.agl", tokens=("--", "-x"))

    def test_a_marker_naming_a_flag_shaped_file_is_the_one_the_host_consumes(self) -> None:
        assert self._split("--", "--weird.agl", "--", "-x") == ExecTail(
            file="--weird.agl", tokens=("--", "-x")
        )

    def test_a_trailing_marker_names_no_file(self) -> None:
        assert self._split("-h", "--") == ExecTail(file=None, tokens=("-h", "--"))

    def test_a_pre_file_option_uses_the_programs_arity_past_a_marker(self) -> None:
        """The marker ends host option scanning; it does not stop the selected
        program's own arity from settling which earlier token is the FILE."""
        command = _command(_param("verbose", BoolType()))

        selected = split_exec_tail(
            ("--verbose", "prog.agl", "--", "World"),
            program_command_for_file=lambda file: command if file == "prog.agl" else None,
        )

        assert selected == ExecTail(file="prog.agl", tokens=("--verbose", "--", "World"))

    def test_a_marker_naming_a_program_outranks_the_pre_marker_probe(self) -> None:
        """A marker the reader wrote is an explicit statement of which token is
        the FILE, so it settles the split before the pre-marker probe — whose
        right-to-left guess would otherwise claim a program flag's value."""
        command = _command(_param("input", TextType()))

        def resolve(file: str) -> ProgramCommand | None:
            return command if file in {"inp.agl", "--weird.agl"} else None

        selected = split_exec_tail(
            ("--input", "inp.agl", "--", "--weird.agl"), program_command_for_file=resolve
        )

        assert selected == ExecTail(file="--weird.agl", tokens=("--input", "inp.agl"))

    def test_a_trailing_marker_names_no_file_even_with_a_resolver(self) -> None:
        """There is no token after the marker to name, so the pre-marker tokens
        are all there is — and they name no program here."""
        command = _command(_param("verbose", BoolType()))

        selected = split_exec_tail(
            ("-h", "--"),
            program_command_for_file=lambda file: command if file == "prog.agl" else None,
        )

        assert selected == ExecTail(file=None, tokens=("-h", "--"))


class TestRetainEndOfOptions:
    """The marker Click removes is doubled so the split can still see it."""

    def test_a_marker_is_doubled_in_place(self) -> None:
        from agm.cli_support.program_options import retain_end_of_options

        assert retain_end_of_options(["-p", "main", "--", "x"]) == ["-p", "main", "--", "--", "x"]

    def test_only_the_first_marker_is_doubled(self) -> None:
        from agm.cli_support.program_options import retain_end_of_options

        assert retain_end_of_options(["--", "--", "x"]) == ["--", "--", "--", "x"]

    def test_a_tail_without_a_marker_is_unchanged(self) -> None:
        from agm.cli_support.program_options import retain_end_of_options

        assert retain_end_of_options(["prog.agl", "-h"]) == ["prog.agl", "-h"]

    def test_a_marker_owned_by_a_host_option_is_not_doubled(self) -> None:
        from agm.cli_support.program_options import retain_end_of_options

        assert retain_end_of_options(
            ["--log-file", "--", "--no-stdlib", "prog.agl"],
            {"--log-file": True, "--no-stdlib": False},
        ) == ["--log-file", "--", "--no-stdlib", "prog.agl"]

    def test_a_host_short_bundle_can_end_in_a_value_option(self) -> None:
        from agm.cli_support.program_options import retain_end_of_options

        assert retain_end_of_options(["-hp", "value", "--"], {"-h": False, "-p": True}) == [
            "-hp",
            "value",
            "--",
            "--",
        ]
