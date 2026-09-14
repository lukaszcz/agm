"""Unit tests for ExternalName: the shared @name/@json-name data model.

Covers .json() (effective JSON name: json_name ?? name ?? declared) and
.value_names() (declared, plus the alias when it differs) directly, since
these are pure leaf methods consumed by the type builder, schema, encode,
and decode passes without re-deriving the precedence rule.
"""

from __future__ import annotations

from agm.agl.semantics.external_names import NO_EXTERNAL_NAME, ExternalName


class TestJson:
    def test_no_override_falls_back_to_declared(self) -> None:
        assert ExternalName().json("width") == "width"

    def test_name_only_becomes_json_name(self) -> None:
        assert ExternalName(name="w").json("width") == "w"

    def test_json_name_only_overrides_declared(self) -> None:
        assert ExternalName(json_name="w_json").json("width") == "w_json"

    def test_json_name_wins_over_name(self) -> None:
        assert ExternalName(name="w", json_name="w_json").json("width") == "w_json"


class TestValueNames:
    def test_no_alias_returns_only_declared(self) -> None:
        assert ExternalName().value_names("width") == ("width",)

    def test_json_name_only_does_not_add_a_value_alias(self) -> None:
        assert ExternalName(json_name="w_json").value_names("width") == ("width",)

    def test_alias_distinct_from_declared_is_appended(self) -> None:
        assert ExternalName(name="w").value_names("width") == ("width", "w")

    def test_alias_equal_to_declared_is_not_duplicated(self) -> None:
        assert ExternalName(name="width").value_names("width") == ("width",)


class TestNoExternalName:
    def test_is_the_all_defaults_sentinel(self) -> None:
        assert NO_EXTERNAL_NAME == ExternalName()
        assert NO_EXTERNAL_NAME.json("x") == "x"
        assert NO_EXTERNAL_NAME.value_names("x") == ("x",)
