"""Tests for the exec_params helper module."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.runtime.types import ParamDeclInfo
from agm.agl.semantics.types import ArrayType, BoolType, IntType, TextType


def _make_param(
    name: str,
    typ: object = None,
    *,
    has_default: bool = False,
    line: int = 1,
    col: int = 1,
    module_segments: tuple[str, ...] = (),
) -> ParamDeclInfo:
    """Build a ``ParamDeclInfo`` for testing."""
    if typ is None:
        typ = TextType()
    return ParamDeclInfo(
        name=name,
        type=typ,
        has_default=has_default,
        line=line,
        col=col,
        module_segments=module_segments,
    )


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------


class TestSourceDiscovery:
    def test_discovers_param_using_the_standard_library(self) -> None:
        from agm.cli_support.exec_params import discover_params_from_source

        params = discover_params_from_source(
            "param selected: Option[bool] = Option::Some(value = true)\n"
            "program def main() -> unit = ()"
        )

        assert [param.name for param in params] == ["selected"]

    def test_inline_source_wraps_before_discovering_params(self) -> None:
        from agm.cli_support.exec_params import discover_params_from_source

        params = discover_params_from_source(
            "param count: int = 1\nlet next = count + 1\nprint next",
            inline_source=True,
        )

        assert [param.name for param in params] == ["count"]

    def test_file_source_discovers_imported_module_params(self, tmp_path: Path) -> None:
        from agm.cli_support.exec_params import discover_params_from_source

        entry_path = tmp_path / "program.agl"
        entry_path.write_text("import settings\nprogram def main() -> unit = ()\n")
        (tmp_path / "settings.agl").write_text('param region: text = "eu"\n')

        params = discover_params_from_source(entry_path.read_text(), entry_path=entry_path)

        assert [(param.module_segments, param.name) for param in params] == [
            (("settings",), "region")
        ]

    def test_invalid_inline_source_degrades_to_no_params(self) -> None:
        from agm.cli_support.exec_params import discover_params_from_source

        assert discover_params_from_source("param count: int =", inline_source=True) == ()

    def test_scoped_param_discovers_under_its_full_path_spelling(self) -> None:
        """A scoped param's external key is its full path, e.g. 'Deploy::region'."""
        from agm.cli_support.exec_params import discover_params_from_source

        params = discover_params_from_source(
            'scope Deploy\nparam region: text = "eu"\nend Deploy\nprogram def main() -> unit = ()'
        )

        assert [param.name for param in params] == ["Deploy::region"]

    def test_nested_scoped_param_discovers_under_its_full_path_spelling(self) -> None:
        from agm.cli_support.exec_params import discover_params_from_source

        params = discover_params_from_source(
            "scope A\nscope B\nparam x: int\nend B\nend A\nprogram def main() -> unit = ()"
        )

        assert [param.name for param in params] == ["A::B::x"]

    def test_root_and_scoped_params_are_both_discovered(self) -> None:
        from agm.cli_support.exec_params import discover_params_from_source

        params = discover_params_from_source(
            'param increment: int\nscope Deploy\nparam region: text = "eu"\nend Deploy\n'
            "program def main() -> unit = ()"
        )

        assert {param.name for param in params} == {"increment", "Deploy::region"}


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

        params = (_make_param("tags", ArrayType(elem=TextType())),)
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

    def test_module_qualified_spelling_resolves_to_the_short_param_key(self) -> None:
        from agm.cli_support.exec_params import parse_param_tokens

        params = (self._text_param("Deploy::region", module_segments=("pkg", "deploy")),)

        assert parse_param_tokens(params, ["--Deploy::region", "eu"]) == {"Deploy::region": "eu"}
        assert parse_param_tokens(params, ["--pkg/deploy::Deploy::region", "us"]) == {
            "Deploy::region": "us"
        }

    def test_reserved_short_spelling_requires_module_qualification(self) -> None:
        from agm.cli_support.exec_params import parse_param_tokens

        params = (self._text_param("max-iters", module_segments=("pkg", "tuning")),)

        with pytest.raises(ValueError, match="pkg/tuning::max-iters"):
            parse_param_tokens(params, ["--max-iters", "5"])
        assert parse_param_tokens(params, ["--pkg/tuning::max-iters", "5"]) == {
            "pkg/tuning::max-iters": "5"
        }

    def test_short_spelling_collision_requires_module_qualification(self) -> None:
        from agm.cli_support.exec_params import parse_param_tokens

        params = (
            self._text_param("Deploy::region", module_segments=("pkg", "one")),
            self._text_param("Deploy::region", module_segments=("pkg", "two")),
        )

        with pytest.raises(ValueError) as exc_info:
            parse_param_tokens(params, ["--Deploy::region", "eu"])
        assert "pkg/one::Deploy::region" in str(exc_info.value)
        assert "pkg/two::Deploy::region" in str(exc_info.value)
        assert parse_param_tokens(params, ["--pkg/one::Deploy::region", "eu"]) == {
            "pkg/one::Deploy::region": "eu"
        }
        with pytest.raises(ValueError) as equals_exc_info:
            parse_param_tokens(params, ["--Deploy::region=eu"])
        assert "pkg/one::Deploy::region" in str(equals_exc_info.value)

    def test_bool_short_collision_rejects_the_negative_spelling(self) -> None:
        from agm.cli_support.exec_params import parse_param_tokens

        params = (
            self._bool_param("verbose", module_segments=("pkg", "one")),
            self._bool_param("verbose", module_segments=("pkg", "two")),
        )

        with pytest.raises(ValueError, match="pkg/one::verbose"):
            parse_param_tokens(params, ["--no-verbose"])


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

    def test_reserved_param_renders_only_its_module_qualified_spelling(self) -> None:
        from agm.cli_support.exec_params import render_param_help_section

        section = render_param_help_section(
            (_make_param("max-iters", IntType(), module_segments=("pkg", "tuning")),)
        )

        assert "--pkg/tuning::max-iters" in section
        assert "--max-iters INT" not in section

    def test_ambiguous_params_render_module_qualified_spellings(self) -> None:
        from agm.cli_support.exec_params import render_param_help_section

        params = (
            _make_param("region", TextType(), module_segments=("pkg", "one")),
            _make_param("region", TextType(), module_segments=("pkg", "two")),
        )

        section = render_param_help_section(params)
        assert "--pkg/one::region" in section
        assert "--pkg/two::region" in section
