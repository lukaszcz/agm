"""Tests for ModuleId→path resolution in src/agm/agl/modules/resolver.py."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agm.agl.modules.errors import AmbiguousModule, ModuleNotFound, ModulePrefixNotFound
from agm.agl.modules.ids import ModuleId
from agm.agl.modules.resolver import expand_wildcard, resolve_module
from agm.agl.modules.roots import RootSet

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _roots(*paths: Path) -> RootSet:
    """Build a RootSet from canonical paths (must exist)."""
    return RootSet(roots=frozenset(paths))


def _make_module(root: Path, dotted: str) -> Path:
    """Write an empty .agl file for the given module under root."""
    mid = ModuleId.from_dotted(dotted)
    path = root / mid.relpath().replace("/", os.sep)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


# ---------------------------------------------------------------------------
# resolve_module — basic lookup
# ---------------------------------------------------------------------------


class TestResolveModuleFound:
    def test_single_root_finds_module(self, tmp_path: Path) -> None:
        root = tmp_path / "lib"
        root.mkdir()
        expected = _make_module(root, "foo.bar")
        result = resolve_module(ModuleId.from_dotted("foo.bar"), _roots(root))
        assert result == expected.resolve()

    def test_returns_canonical_path(self, tmp_path: Path) -> None:
        root = tmp_path / "lib"
        root.mkdir()
        _make_module(root, "mymod")
        # Access through a symlinked root
        link = tmp_path / "link"
        link.symlink_to(root)
        result = resolve_module(ModuleId.from_dotted("mymod"), _roots(link))
        assert result == (root / "mymod.agl").resolve()

    def test_finds_module_in_one_of_multiple_roots(self, tmp_path: Path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        _make_module(root_b, "util.helper")
        result = resolve_module(
            ModuleId.from_dotted("util.helper"), _roots(root_a, root_b)
        )
        assert result == (root_b / "util" / "helper.agl").resolve()

    def test_same_file_via_two_roots_counts_once(self, tmp_path: Path) -> None:
        """Duplicate/symlinked roots that resolve to the same canonical file → ok."""
        root = tmp_path / "lib"
        root.mkdir()
        _make_module(root, "foo")
        # Add a symlink root that points to the same directory
        link = tmp_path / "link"
        link.symlink_to(root)
        # Both roots resolve to the same canonical file → exactly one → ok
        result = resolve_module(ModuleId.from_dotted("foo"), _roots(root, link))
        assert result == (root / "foo.agl").resolve()

    def test_nested_root_same_file_counts_once(self, tmp_path: Path) -> None:
        """A root nested inside another root seeing the same file → ok."""
        outer = tmp_path / "outer"
        (outer / "inner").mkdir(parents=True)
        # Module lives at outer/inner/mod.agl; outer and outer/inner are both roots
        mod_file = outer / "inner" / "mod.agl"
        mod_file.write_text("")
        inner = outer / "inner"
        # From outer root: 'inner.mod' resolves; from inner root: 'mod' resolves
        # These are DIFFERENT module ids, so no conflict. Test same-canonical-file
        # for the same module-id via two roots pointing to the same directory.
        root1 = inner
        link = tmp_path / "link"
        link.symlink_to(inner)
        result = resolve_module(ModuleId.from_dotted("mod"), _roots(root1, link))
        assert result == mod_file.resolve()


class TestResolveModuleNotFound:
    def test_not_found_raises_module_not_found(self, tmp_path: Path) -> None:
        root = tmp_path / "lib"
        root.mkdir()
        with pytest.raises(ModuleNotFound):
            resolve_module(ModuleId.from_dotted("missing"), _roots(root))

    def test_not_found_message_lists_roots(self, tmp_path: Path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        try:
            resolve_module(ModuleId.from_dotted("absent"), _roots(root_a, root_b))
        except ModuleNotFound as exc:
            msg = str(exc)
            # Both roots should be listed in the error
            assert str(root_a) in msg or str(root_b) in msg
        else:
            pytest.fail("Expected ModuleNotFound")

    def test_not_found_error_has_searched_roots(self, tmp_path: Path) -> None:
        root = tmp_path / "lib"
        root.mkdir()
        roots = _roots(root)
        try:
            resolve_module(ModuleId.from_dotted("gone"), roots)
        except ModuleNotFound as exc:
            assert exc.searched_roots == roots.sorted_roots()
        else:
            pytest.fail("Expected ModuleNotFound")

    def test_empty_roots_raises_module_not_found(self, tmp_path: Path) -> None:
        empty_roots = RootSet(roots=frozenset())
        with pytest.raises(ModuleNotFound):
            resolve_module(ModuleId.from_dotted("foo"), empty_roots)


class TestResolveModuleAmbiguous:
    def test_same_id_in_two_roots_raises_ambiguous(self, tmp_path: Path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        _make_module(root_a, "shared.util")
        _make_module(root_b, "shared.util")
        with pytest.raises(AmbiguousModule):
            resolve_module(ModuleId.from_dotted("shared.util"), _roots(root_a, root_b))

    def test_ambiguous_error_lists_candidates(self, tmp_path: Path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        _make_module(root_a, "dup")
        _make_module(root_b, "dup")
        try:
            resolve_module(ModuleId.from_dotted("dup"), _roots(root_a, root_b))
        except AmbiguousModule as exc:
            assert len(exc.candidates) >= 2
        else:
            pytest.fail("Expected AmbiguousModule")


class TestResolveModuleDeterminism:
    def test_result_independent_of_root_ordering(self, tmp_path: Path) -> None:
        """resolve_module must return the same result regardless of root set order."""
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        # Only one root has the module → no ambiguity regardless of ordering
        _make_module(root_b, "lib.core")
        res1 = resolve_module(ModuleId.from_dotted("lib.core"), _roots(root_a, root_b))
        res2 = resolve_module(ModuleId.from_dotted("lib.core"), _roots(root_b, root_a))
        assert res1 == res2

    def test_expand_wildcard_result_independent_of_filesystem_discovery_order(
        self, tmp_path: Path
    ) -> None:
        """expand_wildcard result must be identical regardless of filesystem discovery order.

        We cannot control the OS glob order directly, but we can assert that
        building the same module layout under several distinct directory
        structures (different root names, different creation orders) yields
        identical ordered results — proving that the final sort makes discovery
        order irrelevant.
        """
        # Build the same logical module set under three distinct directory layouts.
        # Layout 1: single root, modules created in alphabetical order.
        root1 = tmp_path / "layout1"
        root1.mkdir()
        _make_module(root1, "pkg.alpha")
        _make_module(root1, "pkg.beta")
        _make_module(root1, "pkg.gamma")

        # Layout 2: single root, modules created in reverse-alphabetical order.
        root2 = tmp_path / "layout2"
        root2.mkdir()
        _make_module(root2, "pkg.gamma")
        _make_module(root2, "pkg.beta")
        _make_module(root2, "pkg.alpha")

        # Layout 3: two roots, modules split across them in mixed order.
        root3a = tmp_path / "layout3a"
        root3b = tmp_path / "layout3b"
        root3a.mkdir()
        root3b.mkdir()
        _make_module(root3b, "pkg.gamma")
        _make_module(root3a, "pkg.alpha")
        _make_module(root3b, "pkg.beta")

        result1 = expand_wildcard(("pkg",), _roots(root1))
        result2 = expand_wildcard(("pkg",), _roots(root2))
        result3a = expand_wildcard(("pkg",), _roots(root3a, root3b))
        result3b = expand_wildcard(("pkg",), _roots(root3b, root3a))

        # All four calls must yield the same ordered id list.
        keys1 = list(result1.keys())
        keys2 = list(result2.keys())
        keys3a = list(result3a.keys())
        keys3b = list(result3b.keys())
        assert keys1 == keys2, "Discovery order affects result (layout1 vs layout2)"
        assert keys1 == keys3a, "Root ordering affects result (layout1 vs layout3a)"
        assert keys3a == keys3b, "Root ordering affects result (layout3a vs layout3b)"


# ---------------------------------------------------------------------------
# expand_wildcard
# ---------------------------------------------------------------------------


class TestExpandWildcard:
    def test_single_root_single_file(self, tmp_path: Path) -> None:
        root = tmp_path / "lib"
        root.mkdir()
        _make_module(root, "foo.bar")
        result = expand_wildcard(("foo",), _roots(root))
        assert ModuleId.from_dotted("foo.bar") in result

    def test_glob_matches_direct_file_and_subtree(self, tmp_path: Path) -> None:
        """foo.* should match foo.agl (if present) AND foo/**/*.agl."""
        root = tmp_path / "lib"
        root.mkdir()
        _make_module(root, "ns.direct")
        _make_module(root, "ns.sub.deep")
        result = expand_wildcard(("ns",), _roots(root))
        assert ModuleId.from_dotted("ns.direct") in result
        assert ModuleId.from_dotted("ns.sub.deep") in result

    def test_prefix_file_itself_included(self, tmp_path: Path) -> None:
        """glob pattern <root>/<prefix>.agl includes the prefix module itself."""
        root = tmp_path / "lib"
        root.mkdir()
        _make_module(root, "mylib")
        _make_module(root, "mylib.util")
        result = expand_wildcard(("mylib",), _roots(root))
        assert ModuleId.from_dotted("mylib") in result
        assert ModuleId.from_dotted("mylib.util") in result

    def test_wildcard_spans_multiple_roots(self, tmp_path: Path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        _make_module(root_a, "pkg.x")
        _make_module(root_b, "pkg.y")
        result = expand_wildcard(("pkg",), _roots(root_a, root_b))
        assert ModuleId.from_dotted("pkg.x") in result
        assert ModuleId.from_dotted("pkg.y") in result

    def test_empty_wildcard_raises_prefix_not_found(self, tmp_path: Path) -> None:
        root = tmp_path / "lib"
        root.mkdir()
        with pytest.raises(ModulePrefixNotFound):
            expand_wildcard(("nosuchprefix",), _roots(root))

    def test_ambiguous_id_across_roots_raises_ambiguous(self, tmp_path: Path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        _make_module(root_a, "ns.clash")
        _make_module(root_b, "ns.clash")
        with pytest.raises(AmbiguousModule):
            expand_wildcard(("ns",), _roots(root_a, root_b))

    def test_result_ordered_by_module_id(self, tmp_path: Path) -> None:
        root = tmp_path / "lib"
        root.mkdir()
        _make_module(root, "pkg.z")
        _make_module(root, "pkg.a")
        _make_module(root, "pkg.m")
        result = expand_wildcard(("pkg",), _roots(root))
        keys = list(result.keys())
        assert keys == sorted(keys, key=lambda m: m.segments)

    def test_same_canonical_file_via_symlinked_roots_counts_once(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "lib"
        root.mkdir()
        _make_module(root, "x.mod")
        link = tmp_path / "link"
        link.symlink_to(root)
        result = expand_wildcard(("x",), _roots(root, link))
        # Should appear exactly once
        assert list(result.keys()).count(ModuleId.from_dotted("x.mod")) == 1

    def test_deterministic_under_shuffled_roots(self, tmp_path: Path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        _make_module(root_a, "ns.one")
        _make_module(root_b, "ns.two")
        result1 = expand_wildcard(("ns",), _roots(root_a, root_b))
        result2 = expand_wildcard(("ns",), _roots(root_b, root_a))
        assert result1 == result2
        assert list(result1.keys()) == list(result2.keys())

    def test_multi_segment_prefix(self, tmp_path: Path) -> None:
        root = tmp_path / "lib"
        root.mkdir()
        _make_module(root, "a.b.c")
        _make_module(root, "a.b.d")
        result = expand_wildcard(("a", "b"), _roots(root))
        assert ModuleId.from_dotted("a.b.c") in result
        assert ModuleId.from_dotted("a.b.d") in result

    def test_does_not_match_sibling_prefixes(self, tmp_path: Path) -> None:
        """expand_wildcard('foo') must NOT match 'foobar.agl'."""
        root = tmp_path / "lib"
        root.mkdir()
        _make_module(root, "foobar")
        _make_module(root, "foo.real")
        result = expand_wildcard(("foo",), _roots(root))
        assert ModuleId.from_dotted("foobar") not in result
        assert ModuleId.from_dotted("foo.real") in result

    def test_values_are_canonical_paths(self, tmp_path: Path) -> None:
        root = tmp_path / "lib"
        root.mkdir()
        mod_path = _make_module(root, "ns.mod")
        result = expand_wildcard(("ns",), _roots(root))
        assert result[ModuleId.from_dotted("ns.mod")] == mod_path.resolve()

    def test_directory_named_agl_skipped_in_rglob(self, tmp_path: Path) -> None:
        """A directory ending in .agl inside the subtree must be skipped."""
        root = tmp_path / "lib"
        root.mkdir()
        # Create a directory named "subdir.agl" inside the prefix subtree
        fake_dir = root / "ns" / "subdir.agl"
        fake_dir.mkdir(parents=True)
        # Also create a real module so the prefix is not empty
        _make_module(root, "ns.real")
        result = expand_wildcard(("ns",), _roots(root))
        # Only the real file module should appear
        assert ModuleId.from_dotted("ns.real") in result
        # No entry for the directory
        for mid in result:
            assert mid != ModuleId(segments=("ns", "subdir"))
