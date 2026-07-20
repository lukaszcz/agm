"""Tests for ModuleId in src/agm/agl/modules/ids.py."""

from __future__ import annotations

import pytest

from agm.agl.modules.ids import ENTRY_ID, PRELUDE_ID, STD_CONFIG_ID, STD_CORE_ID, ModuleId


class TestModuleIdConstruction:
    def test_from_path_simple(self) -> None:
        mid = ModuleId.from_path("foo")
        assert mid.segments == ("foo",)

    def test_from_path_nested(self) -> None:
        mid = ModuleId.from_path("foo/bar/baz")
        assert mid.segments == ("foo", "bar", "baz")

    def test_segments_stored_as_tuple(self) -> None:
        mid = ModuleId(segments=("a", "b"))
        assert isinstance(mid.segments, tuple)

    def test_frozen(self) -> None:
        mid = ModuleId.from_path("foo")
        with pytest.raises((AttributeError, TypeError)):
            setattr(mid, "segments", ("bar",))

    def test_equality_and_hash(self) -> None:
        a = ModuleId.from_path("foo/bar")
        b = ModuleId.from_path("foo/bar")
        assert a == b
        assert hash(a) == hash(b)

    def test_inequality(self) -> None:
        a = ModuleId.from_path("foo/bar")
        b = ModuleId.from_path("foo/baz")
        assert a != b


class TestModuleIdPathStr:
    def test_single_segment(self) -> None:
        assert ModuleId(segments=("foo",)).path_str() == "foo"

    def test_multiple_segments(self) -> None:
        assert ModuleId(segments=("foo", "bar", "baz")).path_str() == "foo/bar/baz"


class TestModuleIdRelpath:
    def test_single_segment(self) -> None:
        assert ModuleId(segments=("foo",)).relpath() == "foo.agl"

    def test_nested_segments(self) -> None:
        assert ModuleId(segments=("foo", "bar", "baz")).relpath() == "foo/bar/baz.agl"

    def test_uses_forward_slash(self) -> None:
        """relpath() must use '/' (os-independent) not os.sep."""
        rp = ModuleId(segments=("a", "b")).relpath()
        assert "/" in rp
        assert "\\" not in rp

    def test_ends_with_agl_extension(self) -> None:
        rp = ModuleId(segments=("foo", "bar")).relpath()
        assert rp.endswith(".agl")


class TestModuleIdFromPathValidation:
    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            ModuleId.from_path("")

    def test_leading_slash_raises(self) -> None:
        with pytest.raises(ValueError):
            ModuleId.from_path("/foo")

    def test_trailing_slash_raises(self) -> None:
        with pytest.raises(ValueError):
            ModuleId.from_path("foo/")

    def test_double_slash_raises(self) -> None:
        with pytest.raises(ValueError):
            ModuleId.from_path("foo//bar")

    def test_empty_segment_raises(self) -> None:
        with pytest.raises(ValueError):
            ModuleId.from_path("foo//baz")

    def test_segment_starting_with_digit_raises(self) -> None:
        with pytest.raises(ValueError):
            ModuleId.from_path("1foo")

    def test_segment_with_hyphen_raises(self) -> None:
        with pytest.raises(ValueError):
            ModuleId.from_path("foo-bar")

    def test_segment_with_space_raises(self) -> None:
        with pytest.raises(ValueError):
            ModuleId.from_path("foo bar")

    def test_segment_with_dot_raises(self) -> None:
        with pytest.raises(ValueError):
            ModuleId.from_path("foo.bar")

    def test_valid_underscore(self) -> None:
        mid = ModuleId.from_path("foo_bar")
        assert mid.segments == ("foo_bar",)

    def test_valid_uppercase(self) -> None:
        mid = ModuleId.from_path("FooBar")
        assert mid.segments == ("FooBar",)

    def test_valid_mixed_case_nested(self) -> None:
        mid = ModuleId.from_path("Foo/bar_baz/Qux2")
        assert mid.segments == ("Foo", "bar_baz", "Qux2")

    def test_single_identifier(self) -> None:
        mid = ModuleId.from_path("x")
        assert mid.segments == ("x",)

    def test_segment_with_only_underscore(self) -> None:
        mid = ModuleId.from_path("_")
        assert mid.segments == ("_",)

    def test_segment_starting_with_underscore(self) -> None:
        mid = ModuleId.from_path("_private/module")
        assert mid.segments == ("_private", "module")


class TestModuleIdRoundTrip:
    def test_slash_path_roundtrip(self) -> None:
        original = "foo/bar/baz"
        mid = ModuleId.from_path(original)
        assert mid.path_str() == original

    def test_from_path_relpath_roundtrip(self) -> None:
        mid = ModuleId.from_path("foo/bar/baz")
        assert mid.relpath() == "foo/bar/baz.agl"


class TestStandardLibraryIds:
    def test_standard_library_ids_use_slash_paths(self) -> None:
        assert STD_CORE_ID.path_str() == "std/core"
        assert STD_CONFIG_ID.path_str() == "std/config"


class TestSentinelIds:
    @pytest.mark.parametrize("sentinel", [ENTRY_ID, PRELUDE_ID])
    def test_sentinel_is_module_id(self, sentinel: ModuleId) -> None:
        assert isinstance(sentinel, ModuleId)

    def test_entry_id_is_entry(self) -> None:
        assert ENTRY_ID.is_entry

    def test_prelude_id_is_not_entry(self) -> None:
        assert not PRELUDE_ID.is_entry

    def test_non_entry_is_not_entry(self) -> None:
        mid = ModuleId.from_path("foo")
        assert not mid.is_entry

    def test_entry_id_not_equal_to_user_module(self) -> None:
        mid = ModuleId.from_path("main")
        assert ENTRY_ID != mid

    @pytest.mark.parametrize("sentinel", [ENTRY_ID, PRELUDE_ID])
    def test_sentinel_path_is_not_round_trippable(self, sentinel: ModuleId) -> None:
        with pytest.raises(ValueError):
            ModuleId.from_path(sentinel.path_str())

    def test_entry_id_is_sentinel(self) -> None:
        """ENTRY_ID should be a stable singleton-like object."""
        from agm.agl.modules.ids import ENTRY_ID as e2

        assert ENTRY_ID is e2

    def test_entry_id_is_hashable(self) -> None:
        d: dict[ModuleId, str] = {}
        d[ENTRY_ID] = "entry"
        assert d[ENTRY_ID] == "entry"
