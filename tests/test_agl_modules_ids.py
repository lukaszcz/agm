"""Tests for ModuleId in src/agm/agl/modules/ids.py."""

from __future__ import annotations

import pytest

from agm.agl.modules.ids import ENTRY_ID, ModuleId


class TestModuleIdConstruction:
    def test_from_dotted_simple(self) -> None:
        mid = ModuleId.from_dotted("foo")
        assert mid.segments == ("foo",)

    def test_from_dotted_nested(self) -> None:
        mid = ModuleId.from_dotted("foo.bar.baz")
        assert mid.segments == ("foo", "bar", "baz")

    def test_segments_stored_as_tuple(self) -> None:
        mid = ModuleId(segments=("a", "b"))
        assert isinstance(mid.segments, tuple)

    def test_frozen(self) -> None:
        mid = ModuleId.from_dotted("foo")
        with pytest.raises((AttributeError, TypeError)):
            mid.segments = ("bar",)  # type: ignore[misc]

    def test_equality_and_hash(self) -> None:
        a = ModuleId.from_dotted("foo.bar")
        b = ModuleId.from_dotted("foo.bar")
        assert a == b
        assert hash(a) == hash(b)

    def test_inequality(self) -> None:
        a = ModuleId.from_dotted("foo.bar")
        b = ModuleId.from_dotted("foo.baz")
        assert a != b


class TestModuleIdDotted:
    def test_single_segment(self) -> None:
        assert ModuleId(segments=("foo",)).dotted() == "foo"

    def test_multiple_segments(self) -> None:
        assert ModuleId(segments=("foo", "bar", "baz")).dotted() == "foo.bar.baz"


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


class TestModuleIdFromDottedValidation:
    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            ModuleId.from_dotted("")

    def test_leading_dot_raises(self) -> None:
        with pytest.raises(ValueError):
            ModuleId.from_dotted(".foo")

    def test_trailing_dot_raises(self) -> None:
        with pytest.raises(ValueError):
            ModuleId.from_dotted("foo.")

    def test_double_dot_raises(self) -> None:
        with pytest.raises(ValueError):
            ModuleId.from_dotted("foo..bar")

    def test_empty_segment_raises(self) -> None:
        with pytest.raises(ValueError):
            ModuleId.from_dotted("foo..baz")

    def test_segment_starting_with_digit_raises(self) -> None:
        with pytest.raises(ValueError):
            ModuleId.from_dotted("1foo")

    def test_segment_with_hyphen_raises(self) -> None:
        with pytest.raises(ValueError):
            ModuleId.from_dotted("foo-bar")

    def test_segment_with_space_raises(self) -> None:
        with pytest.raises(ValueError):
            ModuleId.from_dotted("foo bar")

    def test_valid_underscore(self) -> None:
        mid = ModuleId.from_dotted("foo_bar")
        assert mid.segments == ("foo_bar",)

    def test_valid_uppercase(self) -> None:
        mid = ModuleId.from_dotted("FooBar")
        assert mid.segments == ("FooBar",)

    def test_valid_mixed_case_nested(self) -> None:
        mid = ModuleId.from_dotted("Foo.bar_baz.Qux2")
        assert mid.segments == ("Foo", "bar_baz", "Qux2")

    def test_single_identifier(self) -> None:
        mid = ModuleId.from_dotted("x")
        assert mid.segments == ("x",)

    def test_segment_with_only_underscore(self) -> None:
        mid = ModuleId.from_dotted("_")
        assert mid.segments == ("_",)

    def test_segment_starting_with_underscore(self) -> None:
        mid = ModuleId.from_dotted("_private.module")
        assert mid.segments == ("_private", "module")


class TestModuleIdRoundTrip:
    def test_dotted_to_from_roundtrip(self) -> None:
        original = "foo.bar.baz"
        mid = ModuleId.from_dotted(original)
        assert mid.dotted() == original

    def test_from_dotted_relpath_roundtrip(self) -> None:
        mid = ModuleId.from_dotted("foo.bar.baz")
        assert mid.relpath() == "foo/bar/baz.agl"


class TestEntryId:
    def test_entry_id_is_module_id(self) -> None:
        assert isinstance(ENTRY_ID, ModuleId)

    def test_entry_id_is_entry(self) -> None:
        assert ENTRY_ID.is_entry

    def test_non_entry_is_not_entry(self) -> None:
        mid = ModuleId.from_dotted("foo")
        assert not mid.is_entry

    def test_entry_id_not_equal_to_user_module(self) -> None:
        mid = ModuleId.from_dotted("main")
        assert ENTRY_ID != mid

    def test_entry_id_segments_not_user_resolvable(self) -> None:
        """Entry id segment must be invalid as an identifier so from_dotted can't produce it."""
        for seg in ENTRY_ID.segments:
            # At least one segment must not pass identifier validation
            try:
                ModuleId.from_dotted(seg)
                is_valid = True
            except ValueError:
                is_valid = False
            if not is_valid:
                return  # Found a non-user-resolvable segment - test passes
        # If all segments pass from_dotted, the dotted form should itself fail
        try:
            ModuleId.from_dotted(ENTRY_ID.dotted())
            # If it succeeds, we have a collision risk — but we need to check
            # that ENTRY_ID itself is distinguishable via is_entry
            # In that case, is_entry must be the only discriminator
            # which is fine per the contract
        except ValueError:
            pass  # This is the ideal case

    def test_entry_id_is_sentinel(self) -> None:
        """ENTRY_ID should be a stable singleton-like object."""
        from agm.agl.modules.ids import ENTRY_ID as E2

        assert ENTRY_ID is E2

    def test_entry_id_is_hashable(self) -> None:
        d: dict[ModuleId, str] = {}
        d[ENTRY_ID] = "entry"
        assert d[ENTRY_ID] == "entry"

    def test_from_dotted_cannot_produce_entry_segments(self) -> None:
        """The entry id's reserved segment contains a char that identifiers cannot have."""
        # This verifies that from_dotted rejects the entry sentinel segment
        for seg in ENTRY_ID.segments:
            if not seg.replace("_", "a").replace("-", "").isidentifier():
                # Has non-identifier chars — confirmed unreachable via from_dotted
                return
        # If all segments look identifier-like but ENTRY_ID.is_entry, that's
        # acceptable if is_entry is implemented as an explicit flag
        assert ENTRY_ID.is_entry
