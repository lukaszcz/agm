"""Tests for the exec_params helper module (M4: per-param CLI options for agm exec)."""

from __future__ import annotations

import pytest

from agm.agl.runtime.types import ParamDeclInfo
from agm.agl.semantics.types import BoolType, IntType, ListType, TextType


def _make_param(
    name: str,
    typ: object = None,
    *,
    has_default: bool = False,
    line: int = 1,
    col: int = 1,
) -> ParamDeclInfo:
    """Build a ``ParamDeclInfo`` for testing."""
    if typ is None:
        typ = TextType()
    return ParamDeclInfo(name=name, type=typ, has_default=has_default, line=line, col=col)


# ---------------------------------------------------------------------------
# param_flag
# ---------------------------------------------------------------------------


class TestParamFlag:
    def test_simple_name(self) -> None:
        from agm.cli_support.exec_params import param_flag

        assert param_flag("msg") == "--msg"

    def test_underscore_preserved(self) -> None:
        from agm.cli_support.exec_params import param_flag

        assert param_flag("my_param") == "--my_param"

    def test_hyphen_name(self) -> None:
        from agm.cli_support.exec_params import param_flag

        assert param_flag("my-param") == "--my-param"


# ---------------------------------------------------------------------------
# check_param_collisions
# ---------------------------------------------------------------------------


class TestCheckParamCollisions:
    def test_clean_params_no_errors(self) -> None:
        from agm.cli_support.exec_params import check_param_collisions

        params = (_make_param("msg"), _make_param("count"), _make_param("verbose", BoolType()))
        assert check_param_collisions(params) == []

    def test_exact_collision_with_max_iters(self) -> None:
        from agm.cli_support.exec_params import check_param_collisions

        # exact: --max-iters is reserved; param name max_iters normalizes to --max-iters
        params = (_make_param("max_iters"),)
        errors = check_param_collisions(params)
        assert len(errors) == 1
        assert "max_iters" in errors[0]

    def test_exact_collision_with_command(self) -> None:
        from agm.cli_support.exec_params import check_param_collisions

        params = (_make_param("command"),)
        errors = check_param_collisions(params)
        assert len(errors) == 1
        assert "command" in errors[0]

    def test_normalized_collision_strict_json(self) -> None:
        from agm.cli_support.exec_params import check_param_collisions

        # param strict_json → --strict_json normalizes to --strict-json (reserved)
        params = (_make_param("strict_json"),)
        errors = check_param_collisions(params)
        assert len(errors) == 1
        assert "strict_json" in errors[0]

    def test_bool_no_prefix_collision_no_log(self) -> None:
        from agm.cli_support.exec_params import check_param_collisions

        # bool param "log" → generates --no-log which collides with reserved --no-log
        params = (_make_param("log", BoolType()),)
        errors = check_param_collisions(params)
        # must report the --no-log collision
        assert any("no-log" in e or "--no-log" in e for e in errors)

    def test_bool_param_positive_flag_no_collision(self) -> None:
        from agm.cli_support.exec_params import check_param_collisions

        # bool param "verbose" — neither --verbose nor --no-verbose is reserved
        params = (_make_param("verbose", BoolType()),)
        assert check_param_collisions(params) == []

    def test_runner_collision(self) -> None:
        from agm.cli_support.exec_params import check_param_collisions

        params = (_make_param("runner"),)
        errors = check_param_collisions(params)
        assert len(errors) == 1
        assert "runner" in errors[0]

    def test_dry_run_normalized_collision(self) -> None:
        from agm.cli_support.exec_params import check_param_collisions

        # param dry_run → --dry_run normalizes to --dry-run (reserved)
        params = (_make_param("dry_run"),)
        errors = check_param_collisions(params)
        assert len(errors) == 1
        assert "dry_run" in errors[0]

    def test_multiple_collisions_reported(self) -> None:
        from agm.cli_support.exec_params import check_param_collisions

        params = (_make_param("runner"), _make_param("max_iters"))
        errors = check_param_collisions(params)
        assert len(errors) == 2

    def test_line_number_in_error(self) -> None:
        from agm.cli_support.exec_params import check_param_collisions

        params = (_make_param("runner", line=7),)
        errors = check_param_collisions(params)
        assert "7" in errors[0]

    def test_module_path_collision(self) -> None:
        from agm.cli_support.exec_params import check_param_collisions

        # param module_path → --module_path normalizes to --module-path (reserved)
        params = (_make_param("module_path"),)
        errors = check_param_collisions(params)
        assert len(errors) == 1
        assert "module_path" in errors[0]

    def test_log_collision(self) -> None:
        from agm.cli_support.exec_params import check_param_collisions

        # param log → --log collides with reserved --log
        params = (_make_param("log"),)
        errors = check_param_collisions(params)
        assert len(errors) >= 1
        assert any("log" in e for e in errors)


# ---------------------------------------------------------------------------
# parse_param_tokens
# ---------------------------------------------------------------------------


class TestParseParamTokens:
    def _text_param(self, name: str = "name", **kw: object) -> ParamDeclInfo:
        return _make_param(name, TextType(), **kw)

    def _bool_param(self, name: str = "verbose", **kw: object) -> ParamDeclInfo:
        return _make_param(name, BoolType(), **kw)

    def _int_param(self, name: str = "count", **kw: object) -> ParamDeclInfo:
        return _make_param(name, IntType(), **kw)

    def test_text_value(self) -> None:
        from agm.cli_support.exec_params import parse_param_tokens

        params = (self._text_param("name"),)
        result = parse_param_tokens(params, ["--name", "hello"])
        assert result == {"name": "hello"}

    def test_equals_form(self) -> None:
        from agm.cli_support.exec_params import parse_param_tokens

        params = (self._text_param("name"),)
        result = parse_param_tokens(params, ["--name=hello"])
        assert result == {"name": "hello"}

    def test_bool_true(self) -> None:
        from agm.cli_support.exec_params import parse_param_tokens

        params = (self._bool_param("verbose"),)
        result = parse_param_tokens(params, ["--verbose"])
        assert result == {"verbose": True}

    def test_bool_false(self) -> None:
        from agm.cli_support.exec_params import parse_param_tokens

        params = (self._bool_param("verbose"),)
        result = parse_param_tokens(params, ["--no-verbose"])
        assert result == {"verbose": False}

    def test_int_value_as_string(self) -> None:
        from agm.cli_support.exec_params import parse_param_tokens

        # parse_param_tokens returns raw str; runtime handles coercion
        params = (self._int_param("count"),)
        result = parse_param_tokens(params, ["--count", "42"])
        assert result == {"count": "42"}

    def test_json_string_passthrough(self) -> None:
        from agm.cli_support.exec_params import parse_param_tokens

        params = (_make_param("tags", ListType(elem=TextType())),)
        result = parse_param_tokens(params, ["--tags", '["a","b"]'])
        assert result == {"tags": '["a","b"]'}

    def test_multiple_params(self) -> None:
        from agm.cli_support.exec_params import parse_param_tokens

        params = (self._text_param("name"), self._int_param("count"))
        result = parse_param_tokens(params, ["--name", "alice", "--count", "5"])
        assert result == {"name": "alice", "count": "5"}

    def test_empty_tokens_returns_empty_dict(self) -> None:
        from agm.cli_support.exec_params import parse_param_tokens

        params = (self._text_param("name"),)
        assert parse_param_tokens(params, []) == {}

    def test_unknown_flag_raises_value_error(self) -> None:
        from agm.cli_support.exec_params import parse_param_tokens

        params = (self._text_param("name"),)
        with pytest.raises(ValueError, match="--unknown"):
            parse_param_tokens(params, ["--unknown"])

    def test_duplicate_param_raises_value_error(self) -> None:
        from agm.cli_support.exec_params import parse_param_tokens

        params = (self._text_param("name"),)
        with pytest.raises(ValueError):
            parse_param_tokens(params, ["--name", "a", "--name", "b"])

    def test_missing_value_raises_value_error(self) -> None:
        from agm.cli_support.exec_params import parse_param_tokens

        params = (self._text_param("name"),)
        with pytest.raises(ValueError):
            parse_param_tokens(params, ["--name"])

    def test_equals_form_value_with_equals_in_value(self) -> None:
        from agm.cli_support.exec_params import parse_param_tokens

        params = (self._text_param("expr"),)
        result = parse_param_tokens(params, ["--expr=a=b"])
        assert result == {"expr": "a=b"}

    def test_underscore_name_param(self) -> None:
        from agm.cli_support.exec_params import parse_param_tokens

        params = (_make_param("my_param", TextType()),)
        result = parse_param_tokens(params, ["--my_param", "hello"])
        assert result == {"my_param": "hello"}

    def test_bool_false_via_no_prefix(self) -> None:
        from agm.cli_support.exec_params import parse_param_tokens

        params = (_make_param("flag", BoolType()),)
        result = parse_param_tokens(params, ["--no-flag"])
        assert result == {"flag": False}

    def test_positional_tokens_ignored(self) -> None:
        """Non-option tokens (like the FILE arg) are silently skipped."""
        from agm.cli_support.exec_params import parse_param_tokens

        params = (self._text_param("name"),)
        # 'some_file.agl' does not start with '--', should be ignored
        result = parse_param_tokens(params, ["some_file.agl", "--name", "hello"])
        assert result == {"name": "hello"}

    def test_no_params_empty_tokens(self) -> None:
        from agm.cli_support.exec_params import parse_param_tokens

        assert parse_param_tokens((), []) == {}

    def test_bool_duplicate_via_no_prefix_raises(self) -> None:
        from agm.cli_support.exec_params import parse_param_tokens

        params = (_make_param("flag", BoolType()),)
        with pytest.raises(ValueError):
            parse_param_tokens(params, ["--flag", "--no-flag"])

    def test_unknown_flag_in_equals_form_raises(self) -> None:
        from agm.cli_support.exec_params import parse_param_tokens

        params = (self._text_param("name"),)
        with pytest.raises(ValueError, match="--unknown"):
            parse_param_tokens(params, ["--unknown=value"])

    def test_bool_flag_with_equals_value_raises(self) -> None:
        from agm.cli_support.exec_params import parse_param_tokens

        params = (_make_param("flag", BoolType()),)
        with pytest.raises(ValueError):
            parse_param_tokens(params, ["--flag=true"])

    def test_duplicate_equals_form_raises(self) -> None:
        from agm.cli_support.exec_params import parse_param_tokens

        params = (self._text_param("name"),)
        with pytest.raises(ValueError):
            parse_param_tokens(params, ["--name=a", "--name=b"])


# ---------------------------------------------------------------------------
# render_param_help_section
# ---------------------------------------------------------------------------


class TestRenderParamHelpSection:
    def test_empty_params_returns_empty_string(self) -> None:
        from agm.cli_support.exec_params import render_param_help_section

        assert render_param_help_section(()) == ""

    def test_text_param_required(self) -> None:
        from agm.cli_support.exec_params import render_param_help_section

        params = (_make_param("msg", TextType(), has_default=False),)
        section = render_param_help_section(params)
        assert "Program parameters:" in section
        assert "--msg" in section
        assert "required" in section.lower()

    def test_text_param_with_default(self) -> None:
        from agm.cli_support.exec_params import render_param_help_section

        params = (_make_param("msg", TextType(), has_default=True),)
        section = render_param_help_section(params)
        assert "--msg" in section
        assert "optional" in section.lower() or "default" in section.lower()

    def test_bool_param_shows_no_prefix_form(self) -> None:
        from agm.cli_support.exec_params import render_param_help_section

        params = (_make_param("verbose", BoolType(), has_default=False),)
        section = render_param_help_section(params)
        assert "--verbose/--no-verbose" in section

    def test_multiple_params_all_present(self) -> None:
        from agm.cli_support.exec_params import render_param_help_section

        params = (
            _make_param("msg", TextType()),
            _make_param("count", IntType(), has_default=True),
            _make_param("verbose", BoolType()),
        )
        section = render_param_help_section(params)
        assert "--msg" in section
        assert "--count" in section
        assert "--verbose/--no-verbose" in section

    def test_section_starts_with_header(self) -> None:
        from agm.cli_support.exec_params import render_param_help_section

        params = (_make_param("x", TextType()),)
        section = render_param_help_section(params)
        assert section.startswith("Program parameters:")


# ---------------------------------------------------------------------------
# resolve_param_values (M5)
# ---------------------------------------------------------------------------


class TestResolveParamValues:
    """Unit tests for ``resolve_param_values`` — merging config + CLI values."""

    def test_cli_wins_over_config(self) -> None:
        from agm.cli_support.exec_params import resolve_param_values

        external, warnings = resolve_param_values(
            {"msg"},
            {"msg": "from_config"},
            {"msg": "from_cli"},
        )
        assert external == {"msg": "from_cli"}
        assert warnings == []

    def test_config_used_when_no_cli_value(self) -> None:
        from agm.cli_support.exec_params import resolve_param_values

        external, warnings = resolve_param_values(
            {"msg"},
            {"msg": "from_config"},
            {},
        )
        assert external == {"msg": "from_config"}
        assert warnings == []

    def test_cli_only_no_config(self) -> None:
        from agm.cli_support.exec_params import resolve_param_values

        external, warnings = resolve_param_values(
            {"msg"},
            {},
            {"msg": "from_cli"},
        )
        assert external == {"msg": "from_cli"}
        assert warnings == []

    def test_empty_both_returns_empty(self) -> None:
        from agm.cli_support.exec_params import resolve_param_values

        external, warnings = resolve_param_values(frozenset({"msg"}), {}, {})
        assert external == {}
        assert warnings == []

    def test_undeclared_config_key_warns_and_excluded(self) -> None:
        from agm.cli_support.exec_params import resolve_param_values

        external, warnings = resolve_param_values(
            {"msg"},
            {"msg": "hi", "typo_key": "oops"},
            {},
        )
        # Only declared key is in result.
        assert external == {"msg": "hi"}
        assert "typo_key" not in external
        # Warning emitted for the undeclared key.
        assert len(warnings) == 1
        assert "typo_key" in warnings[0]

    def test_multiple_undeclared_keys_each_warn(self) -> None:
        from agm.cli_support.exec_params import resolve_param_values

        external, warnings = resolve_param_values(
            frozenset(),
            {"a": 1, "b": 2},
            {},
        )
        assert external == {}
        assert len(warnings) == 2
        assert any("a" in w for w in warnings)
        assert any("b" in w for w in warnings)

    def test_program_name_in_warning_message(self) -> None:
        from agm.cli_support.exec_params import resolve_param_values

        _, warnings = resolve_param_values(
            frozenset(),
            {"bad_key": "x"},
            {},
            program_name="my_workflow",
        )
        assert len(warnings) == 1
        assert "my_workflow" in warnings[0]
        assert "bad_key" in warnings[0]

    def test_program_name_none_still_warns(self) -> None:
        from agm.cli_support.exec_params import resolve_param_values

        _, warnings = resolve_param_values(
            frozenset(),
            {"bad_key": "x"},
            {},
            program_name=None,
        )
        assert len(warnings) == 1
        assert "bad_key" in warnings[0]

    def test_native_bool_preserved(self) -> None:
        from agm.cli_support.exec_params import resolve_param_values

        external, _ = resolve_param_values({"flag"}, {"flag": True}, {})
        assert external["flag"] is True
        assert isinstance(external["flag"], bool)

    def test_native_int_preserved(self) -> None:
        from agm.cli_support.exec_params import resolve_param_values

        external, _ = resolve_param_values({"count"}, {"count": 42}, {})
        assert external["count"] == 42
        assert isinstance(external["count"], int)

    def test_cli_overrides_config_all_declared(self) -> None:
        """CLI overrides config; declared-only passthrough for multiple params."""
        from agm.cli_support.exec_params import resolve_param_values

        config = {"a": "config_a", "b": "config_b", "c": "config_c"}
        cli = {"b": "cli_b"}
        external, warnings = resolve_param_values({"a", "b", "c"}, config, cli)
        assert external["a"] == "config_a"
        assert external["b"] == "cli_b"  # CLI wins
        assert external["c"] == "config_c"
        assert warnings == []

    def test_frozenset_accepted(self) -> None:
        from agm.cli_support.exec_params import resolve_param_values

        external, warnings = resolve_param_values(
            frozenset({"msg"}),
            {"msg": "hello"},
            {},
        )
        assert external == {"msg": "hello"}
        assert warnings == []
