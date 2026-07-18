"""Tests for the pure import-resolution model: ``build_import_env``.

Exhaustively covers every  form (from the AgL module system) and every
static error case.  All tests operate purely on data — no parser, no resolver.
"""

from __future__ import annotations

import pytest

from agm.agl.modules.ids import ModuleId
from agm.agl.scope.imports import (
    ImportEnv,
    ImportTarget,
    SingleTarget,
    WildcardTarget,
    build_import_env,
)
from agm.agl.scope.symbols import AglScopeError
from agm.agl.syntax.nodes import ImportDecl, ImportItem
from agm.agl.syntax.spans import UNKNOWN_SOURCE, SourceSpan
from agm.agl.syntax.types import ImportMode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _span(line: int = 1) -> SourceSpan:
    """Build a minimal SourceSpan for test nodes."""
    return SourceSpan(
        start_line=line,
        start_col=1,
        end_line=line,
        end_col=10,
        start_offset=0,
        end_offset=10,
        source=UNKNOWN_SOURCE,
    )


_NID: int = 0  # global counter (tests run sequentially; node IDs just need to be unique)


def _nid() -> int:
    global _NID
    _NID += 1
    return _NID


def _decl(
    module_path: tuple[str, ...],
    *,
    wildcard: bool = False,
    qualified: bool = False,
    alias: str | None = None,
    mode: ImportMode = ImportMode.ALL,
    items: tuple[ImportItem, ...] = (),
    line: int = 1,
) -> ImportDecl:
    return ImportDecl(
        module_path=module_path,
        wildcard=wildcard,
        qualified=qualified,
        alias=alias,
        mode=mode,
        items=items,
        span=_span(line),
        node_id=_nid(),
    )


def _item(name: str, rename: str | None = None) -> ImportItem:
    return ImportItem(name=name, rename=rename, span=_span(), node_id=_nid())


def _mid(*segments: str) -> ModuleId:
    return ModuleId(segments=segments)


# Module IDs used throughout
FOO = _mid("foo")
BAR = _mid("bar")
FOO_BAR = _mid("foo", "bar")
FOO_BAZ = _mid("foo", "baz")
FOO_QUX = _mid("foo", "qux")
FOO_BAR_CHILD = _mid("foo", "bar", "child")
CURRENT = _mid("current")


def _exports(*names: str, mid: ModuleId) -> dict[str, tuple[ModuleId, str]]:
    """Build a simple export map where all names are locally defined in *mid*."""
    return {n: (mid, n) for n in names}


def _build(
    decls: list[ImportDecl],
    targets: dict[int, ImportTarget],
    exports: dict[ModuleId, dict[str, tuple[ModuleId, str]]],
    *,
    current: ModuleId = CURRENT,
) -> ImportEnv:
    return build_import_env(
        current_module=current,
        decls=tuple(decls),
        targets=targets,
        exports=exports,
    )


# ---------------------------------------------------------------------------
# 1.  Bare import — open, all exports
# ---------------------------------------------------------------------------


class TestBareImport:
    """import foo.bar  →  all exports unqualified + qualified via foo.bar handle."""

    def test_unqualified_all_exports(self) -> None:
        d = _decl(("foo", "bar"))
        env = _build(
            [d],
            {d.node_id: SingleTarget(FOO_BAR)},
            {FOO_BAR: _exports("x", "y", mid=FOO_BAR)},
        )
        assert env.unqualified["x"] == frozenset({(FOO_BAR, "x")})
        assert env.unqualified["y"] == frozenset({(FOO_BAR, "y")})

    def test_qualified_handle_is_dotpath(self) -> None:
        d = _decl(("foo", "bar"))
        env = _build(
            [d],
            {d.node_id: SingleTarget(FOO_BAR)},
            {FOO_BAR: _exports("x", mid=FOO_BAR)},
        )
        assert ("foo", "bar") in env.qualified
        assert env.qualified[("foo", "bar")]["x"] == (FOO_BAR, "x")

    def test_empty_exports_leaves_empty_env(self) -> None:
        d = _decl(("foo", "bar"))
        env = _build(
            [d],
            {d.node_id: SingleTarget(FOO_BAR)},
            {FOO_BAR: {}},
        )
        assert "x" not in env.unqualified
        assert env.qualified.get(("foo", "bar"), {}) == {}


# ---------------------------------------------------------------------------
# 2.  Qualified import — no unqualified injection
# ---------------------------------------------------------------------------


class TestQualifiedImport:
    """import foo.bar qualified  →  no unqualified; qualified via foo.bar handle."""

    def test_no_unqualified_injection(self) -> None:
        d = _decl(("foo", "bar"), qualified=True)
        env = _build(
            [d],
            {d.node_id: SingleTarget(FOO_BAR)},
            {FOO_BAR: _exports("x", mid=FOO_BAR)},
        )
        assert "x" not in env.unqualified

    def test_qualified_handle_present(self) -> None:
        d = _decl(("foo", "bar"), qualified=True)
        env = _build(
            [d],
            {d.node_id: SingleTarget(FOO_BAR)},
            {FOO_BAR: _exports("x", mid=FOO_BAR)},
        )
        assert env.qualified[("foo", "bar")]["x"] == (FOO_BAR, "x")


# ---------------------------------------------------------------------------
# 3.  Alias import — alias replaces dotpath qualifier
# ---------------------------------------------------------------------------


class TestAliasImport:
    """import foo.bar as A  →  unqualified + qualified via ("A",) only (no foo.bar handle)."""

    def test_alias_handle_present(self) -> None:
        d = _decl(("foo", "bar"), alias="A")
        env = _build(
            [d],
            {d.node_id: SingleTarget(FOO_BAR)},
            {FOO_BAR: _exports("x", mid=FOO_BAR)},
        )
        assert ("A",) in env.qualified
        assert env.qualified[("A",)]["x"] == (FOO_BAR, "x")

    def test_dotpath_handle_absent(self) -> None:
        d = _decl(("foo", "bar"), alias="A")
        env = _build(
            [d],
            {d.node_id: SingleTarget(FOO_BAR)},
            {FOO_BAR: _exports("x", mid=FOO_BAR)},
        )
        assert ("foo", "bar") not in env.qualified

    def test_unqualified_still_injected(self) -> None:
        d = _decl(("foo", "bar"), alias="A")
        env = _build(
            [d],
            {d.node_id: SingleTarget(FOO_BAR)},
            {FOO_BAR: _exports("x", mid=FOO_BAR)},
        )
        assert env.unqualified["x"] == frozenset({(FOO_BAR, "x")})


# ---------------------------------------------------------------------------
# 4.  Qualified + alias  — nothing unqualified; alias handle only
# ---------------------------------------------------------------------------


class TestQualifiedAliasImport:
    """import foo.bar qualified as A  →  no unqualified; qualified via ("A",) only."""

    def test_no_unqualified_with_qualified_alias(self) -> None:
        d = _decl(("foo", "bar"), qualified=True, alias="A")
        env = _build(
            [d],
            {d.node_id: SingleTarget(FOO_BAR)},
            {FOO_BAR: _exports("x", mid=FOO_BAR)},
        )
        assert "x" not in env.unqualified
        assert env.qualified[("A",)]["x"] == (FOO_BAR, "x")
        assert ("foo", "bar") not in env.qualified


# ---------------------------------------------------------------------------
# 5.  Using import — only listed names
# ---------------------------------------------------------------------------


class TestUsingImport:
    """import foo.bar using x, y  →  only x, y visible."""

    def test_only_listed_names_in_unqualified(self) -> None:
        d = _decl(
            ("foo", "bar"),
            mode=ImportMode.USING,
            items=(
                _item("x"),
                _item("y"),
            ),
        )
        env = _build(
            [d],
            {d.node_id: SingleTarget(FOO_BAR)},
            {FOO_BAR: _exports("x", "y", "z", mid=FOO_BAR)},
        )
        assert "x" in env.unqualified
        assert "y" in env.unqualified
        assert "z" not in env.unqualified

    def test_only_listed_names_in_qualified(self) -> None:
        d = _decl(
            ("foo", "bar"),
            mode=ImportMode.USING,
            items=(_item("x"),),
        )
        env = _build(
            [d],
            {d.node_id: SingleTarget(FOO_BAR)},
            {FOO_BAR: _exports("x", "z", mid=FOO_BAR)},
        )
        assert "x" in env.qualified[("foo", "bar")]
        assert "z" not in env.qualified.get(("foo", "bar"), {})

    def test_using_nonexported_single_raises(self) -> None:
        d = _decl(
            ("foo", "bar"),
            mode=ImportMode.USING,
            items=(_item("nonexistent"),),
        )
        with pytest.raises(AglScopeError, match="nonexistent"):
            _build(
                [d],
                {d.node_id: SingleTarget(FOO_BAR)},
                {FOO_BAR: _exports("x", mid=FOO_BAR)},
            )


# ---------------------------------------------------------------------------
# 6.  Hiding import — all except listed
# ---------------------------------------------------------------------------


class TestHidingImport:
    """import foo.bar hiding x  →  all except x."""

    def test_all_except_hidden_in_unqualified(self) -> None:
        d = _decl(
            ("foo", "bar"),
            mode=ImportMode.HIDING,
            items=(_item("x"),),
        )
        env = _build(
            [d],
            {d.node_id: SingleTarget(FOO_BAR)},
            {FOO_BAR: _exports("x", "y", "z", mid=FOO_BAR)},
        )
        assert "x" not in env.unqualified
        assert "y" in env.unqualified
        assert "z" in env.unqualified

    def test_hidden_absent_from_qualified(self) -> None:
        d = _decl(
            ("foo", "bar"),
            mode=ImportMode.HIDING,
            items=(_item("x"),),
        )
        env = _build(
            [d],
            {d.node_id: SingleTarget(FOO_BAR)},
            {FOO_BAR: _exports("x", "y", mid=FOO_BAR)},
        )
        assert "x" not in env.qualified.get(("foo", "bar"), {})
        assert "y" in env.qualified[("foo", "bar")]

    def test_hiding_nonexported_single_raises(self) -> None:
        d = _decl(
            ("foo", "bar"),
            mode=ImportMode.HIDING,
            items=(_item("nonexistent"),),
        )
        with pytest.raises(AglScopeError, match="nonexistent"):
            _build(
                [d],
                {d.node_id: SingleTarget(FOO_BAR)},
                {FOO_BAR: _exports("x", mid=FOO_BAR)},
            )


# ---------------------------------------------------------------------------
# 7.  Using + rename  — canonical rename
# ---------------------------------------------------------------------------


class TestUsingRename:
    """import foo.bar using x as X  →  exposed name is X everywhere."""

    def test_renamed_unqualified(self) -> None:
        d = _decl(
            ("foo", "bar"),
            mode=ImportMode.USING,
            items=(_item("x", rename="X"),),
        )
        env = _build(
            [d],
            {d.node_id: SingleTarget(FOO_BAR)},
            {FOO_BAR: _exports("x", mid=FOO_BAR)},
        )
        # Original name not exposed
        assert "x" not in env.unqualified
        # Renamed name is exposed, pointing to (FOO_BAR, "x")
        assert env.unqualified["X"] == frozenset({(FOO_BAR, "x")})

    def test_renamed_qualified(self) -> None:
        d = _decl(
            ("foo", "bar"),
            mode=ImportMode.USING,
            items=(_item("x", rename="X"),),
        )
        env = _build(
            [d],
            {d.node_id: SingleTarget(FOO_BAR)},
            {FOO_BAR: _exports("x", mid=FOO_BAR)},
        )
        # Qualified access also uses the new name X
        assert "x" not in env.qualified.get(("foo", "bar"), {})
        assert env.qualified[("foo", "bar")]["X"] == (FOO_BAR, "x")

    def test_rename_nonexported_single_raises(self) -> None:
        d = _decl(
            ("foo", "bar"),
            mode=ImportMode.USING,
            items=(_item("missing", rename="M"),),
        )
        with pytest.raises(AglScopeError, match="missing"):
            _build(
                [d],
                {d.node_id: SingleTarget(FOO_BAR)},
                {FOO_BAR: _exports("x", mid=FOO_BAR)},
            )


# ---------------------------------------------------------------------------
# 8.  Qualified using  — using + no unqualified injection
# ---------------------------------------------------------------------------


class TestQualifiedUsing:
    """import foo.bar qualified using x  →  no unqualified; only x in qualified."""

    def test_no_unqualified(self) -> None:
        d = _decl(
            ("foo", "bar"),
            qualified=True,
            mode=ImportMode.USING,
            items=(_item("x"),),
        )
        env = _build(
            [d],
            {d.node_id: SingleTarget(FOO_BAR)},
            {FOO_BAR: _exports("x", "y", mid=FOO_BAR)},
        )
        assert "x" not in env.unqualified
        assert "y" not in env.unqualified

    def test_only_x_in_qualified(self) -> None:
        d = _decl(
            ("foo", "bar"),
            qualified=True,
            mode=ImportMode.USING,
            items=(_item("x"),),
        )
        env = _build(
            [d],
            {d.node_id: SingleTarget(FOO_BAR)},
            {FOO_BAR: _exports("x", "y", mid=FOO_BAR)},
        )
        h = env.qualified[("foo", "bar")]
        assert "x" in h
        assert "y" not in h


# ---------------------------------------------------------------------------
# 9.  Qualified as + using
# ---------------------------------------------------------------------------


class TestQualifiedAliasUsing:
    """import foo.bar qualified as A using x  →  no unqualified; x in A:: handle."""

    def test_qualified_alias_using(self) -> None:
        d = _decl(
            ("foo", "bar"),
            qualified=True,
            alias="A",
            mode=ImportMode.USING,
            items=(_item("x"),),
        )
        env = _build(
            [d],
            {d.node_id: SingleTarget(FOO_BAR)},
            {FOO_BAR: _exports("x", "y", mid=FOO_BAR)},
        )
        assert "x" not in env.unqualified
        assert ("foo", "bar") not in env.qualified
        assert env.qualified[("A",)]["x"] == (FOO_BAR, "x")
        assert "y" not in env.qualified.get(("A",), {})


# ---------------------------------------------------------------------------
# 10.  Qualified hiding
# ---------------------------------------------------------------------------


class TestQualifiedHiding:
    """import foo.bar qualified hiding x  →  no unqualified; all except x in qualified."""

    def test_qualified_hiding(self) -> None:
        d = _decl(
            ("foo", "bar"),
            qualified=True,
            mode=ImportMode.HIDING,
            items=(_item("x"),),
        )
        env = _build(
            [d],
            {d.node_id: SingleTarget(FOO_BAR)},
            {FOO_BAR: _exports("x", "y", "z", mid=FOO_BAR)},
        )
        assert "x" not in env.unqualified
        assert "y" not in env.unqualified
        h = env.qualified[("foo", "bar")]
        assert "x" not in h
        assert "y" in h
        assert "z" in h


# ---------------------------------------------------------------------------
# 11.  Wildcard — ALL
# ---------------------------------------------------------------------------


class TestWildcardAll:
    """import foo.*  →  each matched module's exports, full-path qualified."""

    def test_unqualified_from_multiple_modules(self) -> None:
        d = _decl(("foo",), wildcard=True)
        env = _build(
            [d],
            {d.node_id: WildcardTarget(frozenset({FOO_BAR, FOO_BAZ}))},
            {
                FOO_BAR: _exports("x", mid=FOO_BAR),
                FOO_BAZ: _exports("y", mid=FOO_BAZ),
            },
        )
        assert env.unqualified["x"] == frozenset({(FOO_BAR, "x")})
        assert env.unqualified["y"] == frozenset({(FOO_BAZ, "y")})

    def test_qualified_handles_are_full_paths(self) -> None:
        d = _decl(("foo",), wildcard=True)
        env = _build(
            [d],
            {d.node_id: WildcardTarget(frozenset({FOO_BAR, FOO_BAZ}))},
            {
                FOO_BAR: _exports("x", mid=FOO_BAR),
                FOO_BAZ: _exports("y", mid=FOO_BAZ),
            },
        )
        assert env.qualified[("foo", "bar")]["x"] == (FOO_BAR, "x")
        assert env.qualified[("foo", "baz")]["y"] == (FOO_BAZ, "y")

    def test_clash_accumulates_in_unqualified(self) -> None:
        """Same name from two modules → frozenset with 2 QNames (deferred clash-on-use)."""
        d = _decl(("foo",), wildcard=True)
        env = _build(
            [d],
            {d.node_id: WildcardTarget(frozenset({FOO_BAR, FOO_BAZ}))},
            {
                FOO_BAR: _exports("x", mid=FOO_BAR),
                FOO_BAZ: _exports("x", mid=FOO_BAZ),
            },
        )
        assert env.unqualified["x"] == frozenset({(FOO_BAR, "x"), (FOO_BAZ, "x")})


# ---------------------------------------------------------------------------
# 12.  Wildcard — qualified (no unqualified injection)
# ---------------------------------------------------------------------------


class TestWildcardQualified:
    def test_no_unqualified(self) -> None:
        d = _decl(("foo",), wildcard=True, qualified=True)
        env = _build(
            [d],
            {d.node_id: WildcardTarget(frozenset({FOO_BAR}))},
            {FOO_BAR: _exports("x", mid=FOO_BAR)},
        )
        assert "x" not in env.unqualified
        assert env.qualified[("foo", "bar")]["x"] == (FOO_BAR, "x")


# ---------------------------------------------------------------------------
# 13.  Wildcard — using  (per-module selective import)
# ---------------------------------------------------------------------------


class TestWildcardUsing:
    """import foo.* using x  →  only x from each module that exports it."""

    def test_name_from_exporting_module_only(self) -> None:
        d = _decl(("foo",), wildcard=True, mode=ImportMode.USING, items=(_item("x"),))
        env = _build(
            [d],
            {d.node_id: WildcardTarget(frozenset({FOO_BAR, FOO_BAZ}))},
            {
                FOO_BAR: _exports("x", "y", mid=FOO_BAR),
                FOO_BAZ: _exports("y", mid=FOO_BAZ),  # does NOT export x
            },
        )
        assert env.unqualified["x"] == frozenset({(FOO_BAR, "x")})
        assert "y" not in env.unqualified

    def test_wildcard_using_all_miss_raises(self) -> None:
        """Listed name not exported by ANY matched module → error."""
        d = _decl(("foo",), wildcard=True, mode=ImportMode.USING, items=(_item("z"),))
        with pytest.raises(AglScopeError, match="z"):
            _build(
                [d],
                {d.node_id: WildcardTarget(frozenset({FOO_BAR, FOO_BAZ}))},
                {
                    FOO_BAR: _exports("x", mid=FOO_BAR),
                    FOO_BAZ: _exports("y", mid=FOO_BAZ),
                },
            )

    def test_wildcard_partial_miss_not_error(self) -> None:
        """One matched module lacks the listed name — NOT an error as long as another exports it."""
        d = _decl(("foo",), wildcard=True, mode=ImportMode.USING, items=(_item("x"),))
        env = _build(
            [d],
            {d.node_id: WildcardTarget(frozenset({FOO_BAR, FOO_BAZ}))},
            {
                FOO_BAR: _exports("x", mid=FOO_BAR),
                FOO_BAZ: {},  # doesn't export x — not an error
            },
        )
        assert env.unqualified["x"] == frozenset({(FOO_BAR, "x")})


# ---------------------------------------------------------------------------
# 14.  Wildcard — hiding
# ---------------------------------------------------------------------------


class TestWildcardHiding:
    """import foo.* hiding x  →  all except x from each module."""

    def test_hidden_name_absent(self) -> None:
        d = _decl(("foo",), wildcard=True, mode=ImportMode.HIDING, items=(_item("x"),))
        env = _build(
            [d],
            {d.node_id: WildcardTarget(frozenset({FOO_BAR}))},
            {FOO_BAR: _exports("x", "y", mid=FOO_BAR)},
        )
        assert "x" not in env.unqualified
        assert "y" in env.unqualified

    def test_wildcard_hiding_all_miss_raises(self) -> None:
        """Hidden name not exported by any matched module → error."""
        d = _decl(("foo",), wildcard=True, mode=ImportMode.HIDING, items=(_item("z"),))
        with pytest.raises(AglScopeError, match="z"):
            _build(
                [d],
                {d.node_id: WildcardTarget(frozenset({FOO_BAR}))},
                {FOO_BAR: _exports("x", mid=FOO_BAR)},
            )

    def test_wildcard_hiding_partial_miss_not_error(self) -> None:
        """One matched module doesn't export the hidden name — NOT an error."""
        d = _decl(("foo",), wildcard=True, mode=ImportMode.HIDING, items=(_item("x"),))
        env = _build(
            [d],
            {d.node_id: WildcardTarget(frozenset({FOO_BAR, FOO_BAZ}))},
            {
                FOO_BAR: _exports("x", "y", mid=FOO_BAR),
                FOO_BAZ: _exports("y", mid=FOO_BAZ),  # doesn't export x — fine
            },
        )
        assert "x" not in env.unqualified
        assert "y" in env.unqualified


# ---------------------------------------------------------------------------
# 15.  Wildcard + as alias  — prefix re-rooting
# ---------------------------------------------------------------------------


class TestWildcardAlias:
    """import foo.bar.* as A  →  foo.bar→("A",), foo.bar.child→("A","child")."""

    def test_exact_prefix_match_reroooted(self) -> None:
        """foo.bar itself → ("A",)."""
        d = _decl(("foo", "bar"), wildcard=True, alias="A")
        env = _build(
            [d],
            {d.node_id: WildcardTarget(frozenset({FOO_BAR}))},
            {FOO_BAR: _exports("x", mid=FOO_BAR)},
        )
        assert ("A",) in env.qualified
        assert env.qualified[("A",)]["x"] == (FOO_BAR, "x")
        # original path should not be present
        assert ("foo", "bar") not in env.qualified

    def test_child_rerooted(self) -> None:
        """foo.bar.child → ("A", "child")."""
        d = _decl(("foo", "bar"), wildcard=True, alias="A")
        env = _build(
            [d],
            {d.node_id: WildcardTarget(frozenset({FOO_BAR_CHILD}))},
            {FOO_BAR_CHILD: _exports("y", mid=FOO_BAR_CHILD)},
        )
        assert ("A", "child") in env.qualified
        assert env.qualified[("A", "child")]["y"] == (FOO_BAR_CHILD, "y")

    def test_unqualified_injected_for_open_wildcard_alias(self) -> None:
        """Wildcard + alias but NOT qualified → unqualified also injected."""
        d = _decl(("foo", "bar"), wildcard=True, alias="A")
        env = _build(
            [d],
            {d.node_id: WildcardTarget(frozenset({FOO_BAR}))},
            {FOO_BAR: _exports("x", mid=FOO_BAR)},
        )
        assert env.unqualified["x"] == frozenset({(FOO_BAR, "x")})

    def test_qualified_wildcard_alias_no_unqualified(self) -> None:
        d = _decl(("foo", "bar"), wildcard=True, alias="A", qualified=True)
        env = _build(
            [d],
            {d.node_id: WildcardTarget(frozenset({FOO_BAR}))},
            {FOO_BAR: _exports("x", mid=FOO_BAR)},
        )
        assert "x" not in env.unqualified
        assert env.qualified[("A",)]["x"] == (FOO_BAR, "x")


# ---------------------------------------------------------------------------
# 16.  Merge — multiple import declarations (lenient)
# ---------------------------------------------------------------------------


class TestMerge:
    """Multiple import declarations union their effects."""

    def test_merge_idempotent_same_mapping(self) -> None:
        """Two identical declarations → same env as one."""
        d1 = _decl(("foo", "bar"))
        d2 = _decl(("foo", "bar"))
        env = _build(
            [d1, d2],
            {d1.node_id: SingleTarget(FOO_BAR), d2.node_id: SingleTarget(FOO_BAR)},
            {FOO_BAR: _exports("x", mid=FOO_BAR)},
        )
        assert env.unqualified["x"] == frozenset({(FOO_BAR, "x")})
        assert env.qualified[("foo", "bar")]["x"] == (FOO_BAR, "x")

    def test_merge_two_different_modules(self) -> None:
        d1 = _decl(("foo",))
        d2 = _decl(("bar",))
        env = _build(
            [d1, d2],
            {d1.node_id: SingleTarget(FOO), d2.node_id: SingleTarget(BAR)},
            {FOO: _exports("a", mid=FOO), BAR: _exports("b", mid=BAR)},
        )
        assert env.unqualified["a"] == frozenset({(FOO, "a")})
        assert env.unqualified["b"] == frozenset({(BAR, "b")})

    def test_merge_clash_accumulates(self) -> None:
        """Same bare name from two different module declarations → deferred clash."""
        d1 = _decl(("foo",))
        d2 = _decl(("bar",))
        env = _build(
            [d1, d2],
            {d1.node_id: SingleTarget(FOO), d2.node_id: SingleTarget(BAR)},
            {FOO: _exports("x", mid=FOO), BAR: _exports("x", mid=BAR)},
        )
        assert env.unqualified["x"] == frozenset({(FOO, "x"), (BAR, "x")})

    def test_merge_qualified_conflict_raises(self) -> None:
        """Two decls mapping same (handle, name) to different QNames → static error."""
        d1 = _decl(("foo",), alias="A")
        d2 = _decl(("bar",), alias="A")
        # Both try to put A::x → different modules
        with pytest.raises(AglScopeError):
            _build(
                [d1, d2],
                {d1.node_id: SingleTarget(FOO), d2.node_id: SingleTarget(BAR)},
                {FOO: _exports("x", mid=FOO), BAR: _exports("x", mid=BAR)},
            )

    def test_merge_different_handles_no_conflict(self) -> None:
        """Different qualifier handles, even same module — fine."""
        d1 = _decl(("foo",))
        d2 = _decl(("foo",), alias="F")
        env = _build(
            [d1, d2],
            {d1.node_id: SingleTarget(FOO), d2.node_id: SingleTarget(FOO)},
            {FOO: _exports("x", mid=FOO)},
        )
        assert env.qualified[("foo",)]["x"] == (FOO, "x")
        assert env.qualified[("F",)]["x"] == (FOO, "x")

    def test_merge_open_and_qualified_same_module(self) -> None:
        """import foo + import foo qualified  →  unqualified from the open one."""
        d1 = _decl(("foo",))
        d2 = _decl(("foo",), qualified=True)
        env = _build(
            [d1, d2],
            {d1.node_id: SingleTarget(FOO), d2.node_id: SingleTarget(FOO)},
            {FOO: _exports("x", mid=FOO)},
        )
        assert env.unqualified["x"] == frozenset({(FOO, "x")})
        assert env.qualified[("foo",)]["x"] == (FOO, "x")


# ---------------------------------------------------------------------------
# 17.  Duplicate-alias static error
# ---------------------------------------------------------------------------


class TestDuplicateAlias:
    """import foo as A + import bar as A  →  AglScopeError."""

    def test_duplicate_alias_to_different_modules_raises(self) -> None:
        d1 = _decl(("foo",), alias="A")
        d2 = _decl(("bar",), alias="A")
        with pytest.raises(AglScopeError, match="A"):
            _build(
                [d1, d2],
                {d1.node_id: SingleTarget(FOO), d2.node_id: SingleTarget(BAR)},
                {FOO: _exports("x", mid=FOO), BAR: _exports("y", mid=BAR)},
            )

    def test_same_alias_to_same_module_is_idempotent(self) -> None:
        """import foo as A + import foo as A  →  fine (idempotent)."""
        d1 = _decl(("foo",), alias="A")
        d2 = _decl(("foo",), alias="A")
        env = _build(
            [d1, d2],
            {d1.node_id: SingleTarget(FOO), d2.node_id: SingleTarget(FOO)},
            {FOO: _exports("x", mid=FOO)},
        )
        assert env.qualified[("A",)]["x"] == (FOO, "x")

    def test_alias_message_names_both_modules(self) -> None:
        """Error message should mention both conflicting modules."""
        d1 = _decl(("foo",), alias="A", line=1)
        d2 = _decl(("bar",), alias="A", line=2)
        with pytest.raises(AglScopeError) as exc_info:
            _build(
                [d1, d2],
                {d1.node_id: SingleTarget(FOO), d2.node_id: SingleTarget(BAR)},
                {FOO: {}, BAR: {}},
            )
        msg = str(exc_info.value)
        assert "A" in msg


# ---------------------------------------------------------------------------
# 18.  Alias-root collision static error
# ---------------------------------------------------------------------------


class TestAliasRootCollision:
    """import foo + import bar.baz as foo  →  alias root collides with module-path root."""

    def test_alias_root_equals_dotpath_root_different_module_raises(self) -> None:
        d1 = _decl(("foo",))
        d2 = _decl(("bar", "baz"), alias="foo")
        with pytest.raises(AglScopeError, match="foo"):
            _build(
                [d1, d2],
                {d1.node_id: SingleTarget(FOO), d2.node_id: SingleTarget(BAR)},
                {FOO: _exports("x", mid=FOO), BAR: _exports("y", mid=BAR)},
            )

    def test_no_collision_when_both_same_module(self) -> None:
        """import foo + import foo as foo  →  fine (same root, same alias)."""
        d1 = _decl(("foo",))
        d2 = _decl(("foo",), alias="foo")
        env = _build(
            [d1, d2],
            {d1.node_id: SingleTarget(FOO), d2.node_id: SingleTarget(FOO)},
            {FOO: _exports("x", mid=FOO)},
        )
        assert env.qualified[("foo",)]["x"] == (FOO, "x")

    def test_alias_root_collision_different_position(self) -> None:
        """import bar.baz as foo + import foo  →  error regardless of order."""
        d1 = _decl(("bar", "baz"), alias="foo")
        d2 = _decl(("foo",))
        with pytest.raises(AglScopeError, match="foo"):
            _build(
                [d1, d2],
                {d1.node_id: SingleTarget(BAR), d2.node_id: SingleTarget(FOO)},
                {BAR: _exports("y", mid=BAR), FOO: _exports("x", mid=FOO)},
            )

    def test_sibling_imports_same_root_no_collision(self) -> None:
        """import foo.bar + import foo.baz  →  siblings sharing root 'foo' must NOT collide."""
        d1 = _decl(("foo", "bar"))
        d2 = _decl(("foo", "baz"))
        # Should resolve cleanly — no alias involved, just two plain dotpath imports
        env = _build(
            [d1, d2],
            {d1.node_id: SingleTarget(FOO_BAR), d2.node_id: SingleTarget(FOO_BAZ)},
            {FOO_BAR: _exports("x", mid=FOO_BAR), FOO_BAZ: _exports("y", mid=FOO_BAZ)},
        )
        assert env.unqualified["x"] == frozenset({(FOO_BAR, "x")})
        assert env.unqualified["y"] == frozenset({(FOO_BAZ, "y")})

    def test_parent_and_child_import_no_collision(self) -> None:
        """import foo + import foo.bar  →  parent + child sharing root 'foo' must NOT collide."""
        d1 = _decl(("foo",))
        d2 = _decl(("foo", "bar"))
        env = _build(
            [d1, d2],
            {d1.node_id: SingleTarget(FOO), d2.node_id: SingleTarget(FOO_BAR)},
            {FOO: _exports("a", mid=FOO), FOO_BAR: _exports("b", mid=FOO_BAR)},
        )
        assert env.unqualified["a"] == frozenset({(FOO, "a")})
        assert env.unqualified["b"] == frozenset({(FOO_BAR, "b")})
        # foo:: and foo.bar:: handles must both be present
        assert env.qualified[("foo",)]["a"] == (FOO, "a")
        assert env.qualified[("foo", "bar")]["b"] == (FOO_BAR, "b")


# ---------------------------------------------------------------------------
# 19.  S bounds qualified access
# ---------------------------------------------------------------------------


class TestSBoundsQualified:
    """Names not in S are absent from qualified table."""

    def test_using_bounds_qualified(self) -> None:
        d = _decl(
            ("foo", "bar"),
            mode=ImportMode.USING,
            items=(_item("x"),),
        )
        env = _build(
            [d],
            {d.node_id: SingleTarget(FOO_BAR)},
            {FOO_BAR: _exports("x", "z", mid=FOO_BAR)},
        )
        assert "x" in env.qualified[("foo", "bar")]
        assert "z" not in env.qualified.get(("foo", "bar"), {})

    def test_hiding_bounds_qualified(self) -> None:
        d = _decl(
            ("foo", "bar"),
            mode=ImportMode.HIDING,
            items=(_item("x"),),
        )
        env = _build(
            [d],
            {d.node_id: SingleTarget(FOO_BAR)},
            {FOO_BAR: _exports("x", "y", mid=FOO_BAR)},
        )
        h = env.qualified.get(("foo", "bar"), {})
        assert "x" not in h
        assert "y" in h


# ---------------------------------------------------------------------------
# 20.  Empty ImportEnv for empty decls
# ---------------------------------------------------------------------------


class TestEmptyDecls:
    def test_empty_decls_empty_env(self) -> None:
        env = build_import_env(
            current_module=CURRENT,
            decls=(),
            targets={},
            exports={},
        )
        assert len(env.unqualified) == 0
        assert len(env.qualified) == 0


# ---------------------------------------------------------------------------
# 21.  Wildcard using + rename (canonical rename in wildcard)
# ---------------------------------------------------------------------------


class TestWildcardUsingRename:
    def test_wildcard_using_rename(self) -> None:
        d = _decl(
            ("foo",),
            wildcard=True,
            mode=ImportMode.USING,
            items=(_item("x", rename="X"),),
        )
        env = _build(
            [d],
            {d.node_id: WildcardTarget(frozenset({FOO_BAR, FOO_BAZ}))},
            {
                FOO_BAR: _exports("x", mid=FOO_BAR),
                FOO_BAZ: _exports("x", "y", mid=FOO_BAZ),
            },
        )
        # Exposed name is X, original x not exposed
        assert "x" not in env.unqualified
        assert "X" in env.unqualified
        # Both modules contribute to X
        assert env.unqualified["X"] == frozenset({(FOO_BAR, "x"), (FOO_BAZ, "x")})

    def test_wildcard_using_rename_all_miss_raises(self) -> None:
        """Name not exported by any matched module → error, even with rename."""
        d = _decl(
            ("foo",),
            wildcard=True,
            mode=ImportMode.USING,
            items=(_item("missing", rename="M"),),
        )
        with pytest.raises(AglScopeError, match="missing"):
            _build(
                [d],
                {d.node_id: WildcardTarget(frozenset({FOO_BAR}))},
                {FOO_BAR: _exports("x", mid=FOO_BAR)},
            )


# ---------------------------------------------------------------------------
# 22.  ImportEnv is a frozen dataclass
# ---------------------------------------------------------------------------


class TestImportEnvImmutability:
    def test_import_env_frozen(self) -> None:
        env = build_import_env(
            current_module=CURRENT,
            decls=(),
            targets={},
            exports={},
        )
        with pytest.raises((AttributeError, TypeError)):
            setattr(env, "unqualified", {})


# ---------------------------------------------------------------------------
# 23.  No empty-segment handle in qualified (:: self-ref is resolver's job)
# ---------------------------------------------------------------------------


class TestNoEmptySegmentHandle:
    def test_no_empty_tuple_key_in_qualified(self) -> None:
        d = _decl(("foo",))
        env = _build(
            [d],
            {d.node_id: SingleTarget(FOO)},
            {FOO: _exports("x", mid=FOO)},
        )
        assert () not in env.qualified


# ---------------------------------------------------------------------------
# 24.  Multiple decls, qualified conflict in merged view
# ---------------------------------------------------------------------------


class TestMergeQualifiedConflict:
    """Two import decls: same qualified handle + name but different QNames → static error."""

    def test_conflict_via_different_using_selections(self) -> None:
        """import foo as A using x  +  import bar as A using x  →  conflict."""
        d1 = _decl(("foo",), alias="A", mode=ImportMode.USING, items=(_item("x"),))
        d2 = _decl(("bar",), alias="A", mode=ImportMode.USING, items=(_item("x"),))
        with pytest.raises(AglScopeError):
            _build(
                [d1, d2],
                {d1.node_id: SingleTarget(FOO), d2.node_id: SingleTarget(BAR)},
                {FOO: _exports("x", mid=FOO), BAR: _exports("x", mid=BAR)},
            )


# ---------------------------------------------------------------------------
# 25.  Wildcard handles: no original dotpath handle when alias is present
# ---------------------------------------------------------------------------


class TestWildcardAliasSuppressesOriginalHandle:
    def test_original_handle_absent_with_wildcard_alias(self) -> None:
        """import foo.bar.* as A: foo.bar.child  →  ("A","child"), NOT ("foo","bar","child")."""
        d = _decl(("foo", "bar"), wildcard=True, alias="A")
        env = _build(
            [d],
            {d.node_id: WildcardTarget(frozenset({FOO_BAR_CHILD}))},
            {FOO_BAR_CHILD: _exports("z", mid=FOO_BAR_CHILD)},
        )
        assert ("foo", "bar", "child") not in env.qualified
        assert ("A", "child") in env.qualified
        assert env.qualified[("A", "child")]["z"] == (FOO_BAR_CHILD, "z")


# ---------------------------------------------------------------------------
# 26.  Deterministic sorting in error messages
# ---------------------------------------------------------------------------


class TestErrorMessageDeterminism:
    def test_using_error_message_names_module(self) -> None:
        d = _decl(("foo", "bar"), mode=ImportMode.USING, items=(_item("ghost"),))
        with pytest.raises(AglScopeError) as exc_info:
            _build(
                [d],
                {d.node_id: SingleTarget(FOO_BAR)},
                {FOO_BAR: _exports("x", mid=FOO_BAR)},
            )
        msg = str(exc_info.value)
        assert "ghost" in msg
        assert "foo/bar" in msg

    def test_hiding_error_message_names_module(self) -> None:
        d = _decl(("foo", "bar"), mode=ImportMode.HIDING, items=(_item("ghost"),))
        with pytest.raises(AglScopeError) as exc_info:
            _build(
                [d],
                {d.node_id: SingleTarget(FOO_BAR)},
                {FOO_BAR: _exports("x", mid=FOO_BAR)},
            )
        msg = str(exc_info.value)
        assert "ghost" in msg
        assert "foo/bar" in msg

    def test_wildcard_using_error_names_missing_name(self) -> None:
        d = _decl(("foo",), wildcard=True, mode=ImportMode.USING, items=(_item("ghost"),))
        with pytest.raises(AglScopeError) as exc_info:
            _build(
                [d],
                {d.node_id: WildcardTarget(frozenset({FOO_BAR, FOO_BAZ}))},
                {FOO_BAR: _exports("x", mid=FOO_BAR), FOO_BAZ: _exports("y", mid=FOO_BAZ)},
            )
        msg = str(exc_info.value)
        assert "ghost" in msg


# ---------------------------------------------------------------------------
# 27.  Qualified conflict via rename in single-module import
# ---------------------------------------------------------------------------


class TestSingleQualifiedConflictViaRename:
    """Two decls to same module, same handle, different src names under same exposed name."""

    def test_single_rename_conflict_same_handle(self) -> None:
        """import foo using x as Z  +  import foo using y as Z  →  static qualified conflict."""
        d1 = _decl(("foo",), mode=ImportMode.USING, items=(_item("x", rename="Z"),))
        d2 = _decl(("foo",), mode=ImportMode.USING, items=(_item("y", rename="Z"),))
        with pytest.raises(AglScopeError):
            _build(
                [d1, d2],
                {d1.node_id: SingleTarget(FOO), d2.node_id: SingleTarget(FOO)},
                {FOO: _exports("x", "y", mid=FOO)},
            )


# ---------------------------------------------------------------------------
# 28.  Wildcard qualified conflict across two wildcard decls
# ---------------------------------------------------------------------------


class TestWildcardQualifiedConflictAcrossDecls:
    """Two wildcard decls with re-rooted aliases overlap on same handle+name → static conflict."""

    def test_two_wildcards_same_alias_root_different_modules(self) -> None:
        """import foo.* as A + import bar.* as A: both produce ("A","baz")::x → conflict."""
        # foo.baz→("A","baz"), name x → (FOO_BAZ, "x")
        # bar.baz→("A","baz"), name x → (BAR_BAZ, "x")  [different QName]
        bar_baz = _mid("bar", "baz")
        d1 = _decl(("foo",), wildcard=True, alias="A")
        d2 = _decl(("bar",), wildcard=True, alias="A")
        with pytest.raises(AglScopeError):
            _build(
                [d1, d2],
                {
                    d1.node_id: WildcardTarget(frozenset({FOO_BAZ})),
                    d2.node_id: WildcardTarget(frozenset({bar_baz})),
                },
                {FOO_BAZ: _exports("x", mid=FOO_BAZ), bar_baz: _exports("x", mid=bar_baz)},
            )
