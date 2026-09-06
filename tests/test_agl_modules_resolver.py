"""Tests for ModuleId→path resolution in src/agm/agl/modules/resolver.py."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agm.agl.modules.errors import AmbiguousModule, ModuleNotFound, ModulePrefixNotFound
from agm.agl.modules.ids import ModuleId
from agm.agl.modules.resolver import expand_wildcard, resolve_module
from agm.agl.modules.roots import RootSet, assemble_roots
from agm.packages.layout import MODULE_TREE_DIRNAME
from agm.packages.manifest import load_manifest
from agm.packages.model import PackageInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _roots(*paths: Path) -> RootSet:
    """Build a RootSet from canonical paths (must exist)."""
    return RootSet(roots=frozenset(paths))


def _make_module(root: Path, module_path: str) -> Path:
    """Write an empty .agl file for the given module path under root."""
    mid = ModuleId.from_path(module_path)
    path = root / mid.relpath().replace("/", os.sep)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


def _mounted_package_roots(tmp_path: Path, *, also_loose: bool = False) -> RootSet:
    """Mount a package whose root also contains AgL files outside its module tree."""
    invocation = tmp_path / "invocation"
    invocation.mkdir()
    package_root = tmp_path / "package"
    (package_root / MODULE_TREE_DIRNAME).mkdir(parents=True)
    manifest_path = package_root / "package.toml"
    manifest_path.write_text('[package]\nname = "demo"\nversion = "1.0.0"\n')
    package = PackageInfo(package_root, load_manifest(manifest_path))
    _make_module(package_root, "assets/rogue")
    _make_module(package_root, "demo")
    _make_module(package_root, "demo/main")
    _make_module(package.module_root, "main")
    _make_module(package.module_root, "sub/deep")
    return assemble_roots(
        invocation_root=invocation,
        lib_root=None,
        configured=(),
        cli=(str(package_root),) if also_loose else (),
        cwd=tmp_path,
        package_roots=(package,),
    )


def _stdlib_roots(tmp_path: Path) -> RootSet:
    """Select a standard-library root holding one module and one stray file."""
    invocation = tmp_path / "invocation"
    invocation.mkdir()
    stdlib_root = tmp_path / "stdlib"
    (stdlib_root / MODULE_TREE_DIRNAME).mkdir(parents=True)
    _make_module(stdlib_root / MODULE_TREE_DIRNAME, "prelude")
    _make_module(stdlib_root, "stray")
    return assemble_roots(
        invocation_root=invocation,
        stdlib_root=stdlib_root,
        lib_root=None,
        configured=(),
        cli=(),
        cwd=tmp_path,
    )


# ---------------------------------------------------------------------------
# resolve_module — basic lookup
# ---------------------------------------------------------------------------


class TestResolveModuleFound:
    def test_single_root_finds_module(self, tmp_path: Path) -> None:
        root = tmp_path / "lib"
        root.mkdir()
        expected = _make_module(root, "foo/bar")
        result = resolve_module(ModuleId.from_path("foo/bar"), _roots(root))
        assert result == expected.resolve()

    def test_returns_canonical_path(self, tmp_path: Path) -> None:
        root = tmp_path / "lib"
        root.mkdir()
        _make_module(root, "mymod")
        # Access through a symlinked root
        link = tmp_path / "link"
        link.symlink_to(root)
        result = resolve_module(ModuleId.from_path("mymod"), _roots(link))
        assert result == (root / "mymod.agl").resolve()

    def test_finds_module_in_one_of_multiple_roots(self, tmp_path: Path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        _make_module(root_b, "util/helper")
        result = resolve_module(ModuleId.from_path("util/helper"), _roots(root_a, root_b))
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
        result = resolve_module(ModuleId.from_path("foo"), _roots(root, link))
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
        result = resolve_module(ModuleId.from_path("mod"), _roots(root1, link))
        assert result == mod_file.resolve()


class TestPackageMounts:
    """A mounted package resolves ``<name>/<rest>`` inside its module tree."""

    def test_a_mounted_module_resolves_under_the_package_name(self, tmp_path: Path) -> None:
        roots = _mounted_package_roots(tmp_path)
        module_root = (tmp_path / "package" / MODULE_TREE_DIRNAME).resolve()

        assert resolve_module(ModuleId.from_path("demo/main"), roots) == module_root / "main.agl"
        assert (
            resolve_module(ModuleId.from_path("demo/sub/deep"), roots)
            == module_root / "sub" / "deep.agl"
        )

    def test_a_file_beside_the_module_tree_is_not_a_module(self, tmp_path: Path) -> None:
        roots = _mounted_package_roots(tmp_path)

        with pytest.raises(ModuleNotFound):
            resolve_module(ModuleId.from_path("assets/rogue"), roots)
        with pytest.raises(ModuleNotFound):
            resolve_module(ModuleId.from_path("demo"), roots)

    def test_a_loose_root_over_a_package_root_makes_its_own_tree_ambiguous(
        self, tmp_path: Path
    ) -> None:
        # The package root holds a directory named after the package; supplying
        # that root loosely makes its files modules too, and the same id then
        # names two distinct files.
        roots = _mounted_package_roots(tmp_path, also_loose=True)

        with pytest.raises(AmbiguousModule):
            resolve_module(ModuleId.from_path("demo/main"), roots)

    def test_not_found_lists_the_searched_module_trees(self, tmp_path: Path) -> None:
        roots = _mounted_package_roots(tmp_path)

        with pytest.raises(ModuleNotFound) as exc_info:
            resolve_module(ModuleId.from_path("demo/absent"), roots)

        assert exc_info.value.searched_roots == (
            (tmp_path / "invocation").resolve(),
            (tmp_path / "package" / MODULE_TREE_DIRNAME).resolve(),
        )

    def test_not_found_lists_only_the_roots_that_were_searched(self, tmp_path: Path) -> None:
        # Nothing mounts "assets", so only the loose invocation root is
        # searched -- and no module tree may be named as if it had been.
        roots = _mounted_package_roots(tmp_path)

        with pytest.raises(ModuleNotFound) as exc_info:
            resolve_module(ModuleId.from_path("assets/absent"), roots)

        assert exc_info.value.searched_roots == ((tmp_path / "invocation").resolve(),)


class TestStandardLibraryMount:
    """The selected standard-library root mounts ``std`` and nothing else."""

    def test_std_resolves_inside_the_library_module_tree(self, tmp_path: Path) -> None:
        roots = _stdlib_roots(tmp_path)

        assert (
            resolve_module(ModuleId.from_path("std/prelude"), roots)
            == (tmp_path / "stdlib" / MODULE_TREE_DIRNAME / "prelude.agl").resolve()
        )

    def test_the_library_root_is_not_loose(self, tmp_path: Path) -> None:
        roots = _stdlib_roots(tmp_path)

        with pytest.raises(ModuleNotFound):
            resolve_module(ModuleId.from_path("stray"), roots)


class TestResolveModuleNotFound:
    def test_not_found_raises_module_not_found(self, tmp_path: Path) -> None:
        root = tmp_path / "lib"
        root.mkdir()
        with pytest.raises(ModuleNotFound):
            resolve_module(ModuleId.from_path("missing"), _roots(root))

    def test_not_found_message_lists_roots(self, tmp_path: Path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        try:
            resolve_module(ModuleId.from_path("absent"), _roots(root_a, root_b))
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
            resolve_module(ModuleId.from_path("gone"), roots)
        except ModuleNotFound as exc:
            assert exc.searched_roots == roots.sorted_roots()
        else:
            pytest.fail("Expected ModuleNotFound")

    def test_empty_roots_raises_module_not_found(self, tmp_path: Path) -> None:
        empty_roots = RootSet(roots=frozenset())
        with pytest.raises(ModuleNotFound):
            resolve_module(ModuleId.from_path("foo"), empty_roots)


class TestResolveModuleAmbiguous:
    def test_same_id_in_two_roots_raises_ambiguous(self, tmp_path: Path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        _make_module(root_a, "shared/util")
        _make_module(root_b, "shared/util")
        with pytest.raises(AmbiguousModule):
            resolve_module(ModuleId.from_path("shared/util"), _roots(root_a, root_b))

    def test_ambiguous_error_lists_candidates(self, tmp_path: Path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        _make_module(root_a, "dup")
        _make_module(root_b, "dup")
        try:
            resolve_module(ModuleId.from_path("dup"), _roots(root_a, root_b))
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
        _make_module(root_b, "lib/core")
        res1 = resolve_module(ModuleId.from_path("lib/core"), _roots(root_a, root_b))
        res2 = resolve_module(ModuleId.from_path("lib/core"), _roots(root_b, root_a))
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
        _make_module(root1, "pkg/alpha")
        _make_module(root1, "pkg/beta")
        _make_module(root1, "pkg/gamma")

        # Layout 2: single root, modules created in reverse-alphabetical order.
        root2 = tmp_path / "layout2"
        root2.mkdir()
        _make_module(root2, "pkg/gamma")
        _make_module(root2, "pkg/beta")
        _make_module(root2, "pkg/alpha")

        # Layout 3: two roots, modules split across them in mixed order.
        root3a = tmp_path / "layout3a"
        root3b = tmp_path / "layout3b"
        root3a.mkdir()
        root3b.mkdir()
        _make_module(root3b, "pkg/gamma")
        _make_module(root3a, "pkg/alpha")
        _make_module(root3b, "pkg/beta")

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
    def test_a_package_wildcard_spans_its_whole_module_tree(self, tmp_path: Path) -> None:
        roots = _mounted_package_roots(tmp_path)
        module_root = (tmp_path / "package" / MODULE_TREE_DIRNAME).resolve()

        with pytest.raises(ModulePrefixNotFound):
            expand_wildcard(("assets",), roots)
        assert expand_wildcard(("demo",), roots) == {
            ModuleId.from_path("demo/main"): module_root / "main.agl",
            ModuleId.from_path("demo/sub/deep"): module_root / "sub" / "deep.agl",
        }

    def test_a_wildcard_below_the_package_name_spans_that_subtree(self, tmp_path: Path) -> None:
        roots = _mounted_package_roots(tmp_path)
        module_root = (tmp_path / "package" / MODULE_TREE_DIRNAME).resolve()
        _make_module(module_root, "sub")

        assert expand_wildcard(("demo", "sub"), roots) == {
            ModuleId.from_path("demo/sub"): module_root / "sub.agl",
            ModuleId.from_path("demo/sub/deep"): module_root / "sub" / "deep.agl",
        }

    def test_a_standard_library_wildcard_stays_inside_the_module_tree(self, tmp_path: Path) -> None:
        roots = _stdlib_roots(tmp_path)

        assert expand_wildcard(("std",), roots) == {
            ModuleId.from_path("std/prelude"): (
                tmp_path / "stdlib" / MODULE_TREE_DIRNAME / "prelude.agl"
            ).resolve()
        }

    def test_explicit_loose_root_still_exposes_all_files(self, tmp_path: Path) -> None:
        roots = _mounted_package_roots(tmp_path, also_loose=True)

        module_id = ModuleId.from_path("assets/rogue")
        assert resolve_module(module_id, roots).name == "rogue.agl"
        assert expand_wildcard(("assets",), roots) == {
            module_id: (tmp_path / "package" / "assets" / "rogue.agl").resolve()
        }

    def test_a_loose_root_over_a_package_root_makes_its_wildcard_ambiguous(
        self, tmp_path: Path
    ) -> None:
        roots = _mounted_package_roots(tmp_path, also_loose=True)

        with pytest.raises(AmbiguousModule):
            expand_wildcard(("demo",), roots)

    def test_single_root_single_file(self, tmp_path: Path) -> None:
        root = tmp_path / "lib"
        root.mkdir()
        _make_module(root, "foo/bar")
        result = expand_wildcard(("foo",), _roots(root))
        assert ModuleId.from_path("foo/bar") in result

    def test_glob_matches_direct_file_and_subtree(self, tmp_path: Path) -> None:
        """foo/* should match foo.agl (if present) AND foo/**/*.agl."""
        root = tmp_path / "lib"
        root.mkdir()
        _make_module(root, "ns/direct")
        _make_module(root, "ns/sub/deep")
        result = expand_wildcard(("ns",), _roots(root))
        assert ModuleId.from_path("ns/direct") in result
        assert ModuleId.from_path("ns/sub/deep") in result

    def test_prefix_file_itself_included(self, tmp_path: Path) -> None:
        """glob pattern <root>/<prefix>.agl includes the prefix module itself."""
        root = tmp_path / "lib"
        root.mkdir()
        _make_module(root, "mylib")
        _make_module(root, "mylib/util")
        result = expand_wildcard(("mylib",), _roots(root))
        assert ModuleId.from_path("mylib") in result
        assert ModuleId.from_path("mylib/util") in result

    def test_wildcard_spans_multiple_roots(self, tmp_path: Path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        _make_module(root_a, "pkg/x")
        _make_module(root_b, "pkg/y")
        result = expand_wildcard(("pkg",), _roots(root_a, root_b))
        assert ModuleId.from_path("pkg/x") in result
        assert ModuleId.from_path("pkg/y") in result

    def test_empty_wildcard_raises_prefix_not_found(self, tmp_path: Path) -> None:
        root = tmp_path / "lib"
        root.mkdir()
        with pytest.raises(ModulePrefixNotFound) as raised:
            expand_wildcard(("nosuchprefix",), _roots(root))
        assert "nosuchprefix/*" in str(raised.value)

    def test_ambiguous_id_across_roots_raises_ambiguous(self, tmp_path: Path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        _make_module(root_a, "ns/clash")
        _make_module(root_b, "ns/clash")
        with pytest.raises(AmbiguousModule):
            expand_wildcard(("ns",), _roots(root_a, root_b))

    def test_result_ordered_by_module_id(self, tmp_path: Path) -> None:
        root = tmp_path / "lib"
        root.mkdir()
        _make_module(root, "pkg/z")
        _make_module(root, "pkg/a")
        _make_module(root, "pkg/m")
        result = expand_wildcard(("pkg",), _roots(root))
        keys = list(result.keys())
        assert keys == sorted(keys, key=lambda m: m.segments)

    def test_same_canonical_file_via_symlinked_roots_counts_once(self, tmp_path: Path) -> None:
        root = tmp_path / "lib"
        root.mkdir()
        _make_module(root, "x/mod")
        link = tmp_path / "link"
        link.symlink_to(root)
        result = expand_wildcard(("x",), _roots(root, link))
        # Should appear exactly once
        assert list(result.keys()).count(ModuleId.from_path("x/mod")) == 1

    def test_deterministic_under_shuffled_roots(self, tmp_path: Path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        _make_module(root_a, "ns/one")
        _make_module(root_b, "ns/two")
        result1 = expand_wildcard(("ns",), _roots(root_a, root_b))
        result2 = expand_wildcard(("ns",), _roots(root_b, root_a))
        assert result1 == result2
        assert list(result1.keys()) == list(result2.keys())

    def test_multi_segment_prefix(self, tmp_path: Path) -> None:
        root = tmp_path / "lib"
        root.mkdir()
        _make_module(root, "a/b/c")
        _make_module(root, "a/b/d")
        result = expand_wildcard(("a", "b"), _roots(root))
        assert ModuleId.from_path("a/b/c") in result
        assert ModuleId.from_path("a/b/d") in result

    def test_does_not_match_sibling_prefixes(self, tmp_path: Path) -> None:
        """expand_wildcard('foo') must NOT match 'foobar.agl'."""
        root = tmp_path / "lib"
        root.mkdir()
        _make_module(root, "foobar")
        _make_module(root, "foo/real")
        result = expand_wildcard(("foo",), _roots(root))
        assert ModuleId.from_path("foobar") not in result
        assert ModuleId.from_path("foo/real") in result

    def test_values_are_canonical_paths(self, tmp_path: Path) -> None:
        root = tmp_path / "lib"
        root.mkdir()
        mod_path = _make_module(root, "ns/mod")
        result = expand_wildcard(("ns",), _roots(root))
        assert result[ModuleId.from_path("ns/mod")] == mod_path.resolve()

    def test_directory_named_agl_skipped_in_rglob(self, tmp_path: Path) -> None:
        """A directory ending in .agl inside the subtree must be skipped."""
        root = tmp_path / "lib"
        root.mkdir()
        # Create a directory named "subdir.agl" inside the prefix subtree
        fake_dir = root / "ns" / "subdir.agl"
        fake_dir.mkdir(parents=True)
        # Also create a real module so the prefix is not empty
        _make_module(root, "ns/real")
        result = expand_wildcard(("ns",), _roots(root))
        # Only the real file module should appear
        assert ModuleId.from_path("ns/real") in result
        # No entry for the directory
        for mid in result:
            assert mid != ModuleId(segments=("ns", "subdir"))
