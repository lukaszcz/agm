"""Tests for the contribution import environment and qualifier resolution."""

from __future__ import annotations

import pytest

from agm.agl.modules.ids import ModuleId
from agm.agl.scope.imports import (
    ImportEnv,
    ModuleContribution,
    QualResolutionAmbiguous,
    QualResolutionFound,
    QualResolutionMissingMember,
    QualResolutionUnknownQualifier,
    SingleTarget,
    WildcardTarget,
    build_import_env,
    resolve_qualified,
)
from agm.agl.scope.symbols import AglScopeError
from agm.agl.syntax.nodes import ImportDecl, ImportItem, ScopeSegment
from agm.agl.syntax.spans import UNKNOWN_SOURCE, SourceSpan
from agm.agl.syntax.types import ImportMode


def _span() -> SourceSpan:
    return SourceSpan(1, 1, 1, 1, 0, 0, UNKNOWN_SOURCE)


_next_node_id = 0


def _node_id() -> int:
    global _next_node_id
    _next_node_id += 1
    return _next_node_id


def _decl(
    path: str,
    *,
    alias: str | None = None,
    wildcard: bool = False,
    is_open: bool = False,
    mode: ImportMode = ImportMode.ALL,
    items: tuple[ImportItem, ...] = (),
    scope_path: tuple[ScopeSegment, ...] = (),
) -> ImportDecl:
    return ImportDecl(
        module_path=tuple(path.split("/")),
        wildcard=wildcard,
        is_open=is_open,
        alias=alias,
        mode=mode,
        items=items,
        span=_span(),
        node_id=_node_id(),
        scope_path=scope_path,
    )


def _region(*names: str) -> tuple[ScopeSegment, ...]:
    return tuple(ScopeSegment(name=name, span=_span(), node_id=_node_id()) for name in names)


def _item(name: str, rename: str | None = None) -> ImportItem:
    return ImportItem(name, rename, _span(), _node_id())


def _module(path: str) -> ModuleId:
    return ModuleId.from_path(path)


def _exports(path: str, *names: str) -> dict[str, tuple[ModuleId, str]]:
    module = _module(path)
    return {name: (module, name) for name in names}


def _build(
    decls: list[ImportDecl],
    targets: dict[int, SingleTarget | WildcardTarget],
    exports: dict[ModuleId, dict[str, tuple[ModuleId, str]]],
) -> ImportEnv:
    return build_import_env(tuple(decls), targets, exports)


class TestContributionSets:
    """Every import contributes exactly its selected set S."""

    @pytest.mark.parametrize(
        ("mode", "items", "expected"),
        [
            (ImportMode.ALL, (), {"one", "two", "three"}),
            (ImportMode.USING, (_item("one"), _item("three")), {"one", "three"}),
            (ImportMode.HIDING, (_item("two"),), {"one", "three"}),
        ],
    )
    def test_one_set_by_mode(
        self, mode: ImportMode, items: tuple[ImportItem, ...], expected: set[str]
    ) -> None:
        decl = _decl("lib/api", mode=mode, items=items)
        module = _module("lib/api")
        env = _build(
            [decl],
            {decl.node_id: SingleTarget(module)},
            {module: _exports("lib/api", "one", "two", "three")},
        )

        assert set(env.contributions[module].members) == expected
        assert set(env.unqualified) == ({"one", "three"} if mode is ImportMode.USING else set())

    def test_open_injects_the_full_selected_set_bare(self) -> None:
        decl = _decl("lib/api", is_open=True, mode=ImportMode.HIDING, items=(_item("secret"),))
        module = _module("lib/api")
        env = _build(
            [decl],
            {decl.node_id: SingleTarget(module)},
            {module: _exports("lib/api", "public", "secret")},
        )

        assert set(env.unqualified) == {"public"}
        assert env.unqualified["public"] == frozenset({(module, "public")})

    def test_region_scoped_open_import_keeps_bare_names_out_of_unqualified(self) -> None:
        """A region-scoped ``open import`` snapshots into ``decl_bare``, not module-wide."""
        decl = _decl("lib/api", is_open=True, scope_path=_region("A"))
        module = _module("lib/api")
        env = _build(
            [decl],
            {decl.node_id: SingleTarget(module)},
            {module: _exports("lib/api", "one", "two")},
        )

        assert set(env.unqualified) == set()
        assert dict(env.decl_bare[decl.node_id]) == {
            "one": frozenset({(module, "one")}),
            "two": frozenset({(module, "two")}),
        }
        # The qualifier route stays module-wide regardless of the scope path.
        assert env.contributions[module].path_enabled is True
        assert set(env.contributions[module].members) == {"one", "two"}

    def test_region_scoped_wildcard_keeps_every_origin_of_a_duplicate_bare_name(self) -> None:
        """A wildcard expanding to several modules doesn't hard-error on a shared bare name.

        Two sibling modules exporting the same name is only an ambiguity at
        the name's first use (mirrors the module-wide ``unqualified`` table's
        own policy), never a build-time error.
        """
        decl = _decl("pkg", wildcard=True, is_open=True, scope_path=_region("A"))
        one = _module("pkg/one")
        two = _module("pkg/two")
        env = _build(
            [decl],
            {decl.node_id: WildcardTarget(frozenset({one, two}))},
            {one: _exports("pkg/one", "alpha"), two: _exports("pkg/two", "alpha")},
        )

        assert dict(env.decl_bare[decl.node_id]) == {
            "alpha": frozenset({(one, "alpha"), (two, "alpha")}),
        }

    def test_region_scoped_plain_import_contributes_no_bare_names(self) -> None:
        """A scoped plain ``import`` (no ``open``/``using``) never populates ``decl_bare``."""
        decl = _decl("lib/api", scope_path=_region("A"))
        module = _module("lib/api")
        env = _build(
            [decl],
            {decl.node_id: SingleTarget(module)},
            {module: _exports("lib/api", "one")},
        )

        assert env.decl_bare.get(decl.node_id, {}) == {}
        assert env.contributions[module].path_enabled is True

    def test_two_region_scoped_decls_keep_independent_bare_sets(self) -> None:
        """Each region-scoped declaration's bare selection is kept separate by node id."""
        first = _decl("lib/api", is_open=True, mode=ImportMode.HIDING, scope_path=_region("A"))
        second = _decl(
            "lib/api",
            mode=ImportMode.USING,
            items=(_item("two"),),
            scope_path=_region("B"),
        )
        module = _module("lib/api")
        env = _build(
            [first, second],
            {first.node_id: SingleTarget(module), second.node_id: SingleTarget(module)},
            {module: _exports("lib/api", "one", "two")},
        )

        assert set(env.unqualified) == set()
        assert dict(env.decl_bare[first.node_id]) == {
            "one": frozenset({(module, "one")}),
            "two": frozenset({(module, "two")}),
        }
        assert dict(env.decl_bare[second.node_id]) == {"two": frozenset({(module, "two")})}


class TestContributionImmutability:
    def test_public_mappings_are_immutable_snapshots(self) -> None:
        module = _module("lib/api")
        original = (module, "original")
        members = {"public": original}
        contribution = ModuleContribution(
            module=module,
            members=members,
            bare_names=frozenset({"public"}),
            path_enabled=True,
            aliases=frozenset(),
        )
        contributions = {module: contribution}
        unqualified = {"public": frozenset({original})}
        env = ImportEnv(contributions=contributions, unqualified=unqualified)

        members["later"] = (module, "later")
        contributions[_module("other/api")] = contribution
        unqualified["later"] = frozenset({(module, "later")})

        assert dict(contribution.members) == {"public": original}
        assert dict(env.contributions) == {module: contribution}
        assert dict(env.unqualified) == {"public": frozenset({original})}
        with pytest.raises(TypeError):
            contribution.members["later"] = (module, "later")
        with pytest.raises(TypeError):
            env.contributions[module] = contribution
        with pytest.raises(TypeError):
            env.unqualified["later"] = frozenset({(module, "later")})

    def test_decl_bare_is_an_immutable_snapshot(self) -> None:
        module = _module("lib/api")
        bare = {"one": frozenset({(module, "one")})}
        decl_bare = {1: bare}
        env = ImportEnv(contributions={}, unqualified={}, decl_bare=decl_bare)

        bare["two"] = frozenset({(module, "two")})
        decl_bare[2] = {"three": frozenset({(module, "three")})}

        assert dict(env.decl_bare[1]) == {"one": frozenset({(module, "one")})}
        assert set(env.decl_bare) == {1}
        with pytest.raises(TypeError):
            env.decl_bare[1]["two"] = frozenset({(module, "two")})
        with pytest.raises(TypeError):
            env.decl_bare[3] = {}


class TestContributionMerging:
    def test_same_module_unions_sets_and_bare_injection(self) -> None:
        plain = _decl("lib/api", mode=ImportMode.HIDING, items=(_item("hidden"),))
        selected = _decl("lib/api", mode=ImportMode.USING, items=(_item("hidden", "revealed"),))
        module = _module("lib/api")
        env = _build(
            [plain, selected],
            {plain.node_id: SingleTarget(module), selected.node_id: SingleTarget(module)},
            {module: _exports("lib/api", "shown", "hidden", "other")},
        )

        contribution = env.contributions[module]
        assert set(contribution.members) == {"shown", "other", "revealed"}
        assert contribution.bare_names == frozenset({"revealed"})
        assert env.unqualified["revealed"] == frozenset({(module, "hidden")})

    def test_conflicting_exposed_names_from_one_module_are_rejected(self) -> None:
        first = _decl("lib/api", mode=ImportMode.USING, items=(_item("one", "item"),))
        second = _decl("lib/api", mode=ImportMode.USING, items=(_item("two", "item"),))
        module = _module("lib/api")

        with pytest.raises(AglScopeError):
            _build(
                [first, second],
                {first.node_id: SingleTarget(module), second.node_id: SingleTarget(module)},
                {module: _exports("lib/api", "one", "two")},
            )


class TestWildcardDistribution:
    def test_wildcard_distributes_contributions_per_module(self) -> None:
        decl = _decl("pkg", wildcard=True, mode=ImportMode.HIDING, items=(_item("hidden"),))
        left = _module("pkg/left")
        right = _module("pkg/right")
        env = _build(
            [decl],
            {decl.node_id: WildcardTarget(frozenset({left, right}))},
            {
                left: _exports("pkg/left", "left", "hidden"),
                right: _exports("pkg/right", "right", "hidden"),
            },
        )

        assert set(env.contributions[left].members) == {"left"}
        assert set(env.contributions[right].members) == {"right"}

    def test_wildcard_using_renames_each_module_canonically(self) -> None:
        decl = _decl("pkg", wildcard=True, mode=ImportMode.USING, items=(_item("shared", "api"),))
        left = _module("pkg/left")
        right = _module("pkg/right")
        env = _build(
            [decl],
            {decl.node_id: WildcardTarget(frozenset({left, right}))},
            {
                left: _exports("pkg/left", "shared"),
                right: _exports("pkg/right", "shared"),
            },
        )

        assert env.contributions[left].members == {"api": (left, "shared")}
        assert env.contributions[right].members == {"api": (right, "shared")}
        assert env.unqualified["api"] == frozenset({(left, "shared"), (right, "shared")})

    def test_wildcard_using_requires_every_matched_module_to_export_each_name(self) -> None:
        decl = _decl("pkg", wildcard=True, mode=ImportMode.USING, items=(_item("shared"),))
        left = _module("pkg/left")
        right = _module("pkg/right")

        with pytest.raises(AglScopeError):
            _build(
                [decl],
                {decl.node_id: WildcardTarget(frozenset({left, right}))},
                {left: _exports("pkg/left", "shared"), right: _exports("pkg/right", "other")},
            )

    def test_wildcard_using_keeps_all_origins_for_a_bare_name_clash(self) -> None:
        decl = _decl("pkg", wildcard=True, mode=ImportMode.USING, items=(_item("shared"),))
        left = _module("pkg/left")
        right = _module("pkg/right")
        env = _build(
            [decl],
            {decl.node_id: WildcardTarget(frozenset({left, right}))},
            {
                left: _exports("pkg/left", "shared"),
                right: _exports("pkg/right", "shared"),
            },
        )

        assert env.unqualified["shared"] == frozenset({(left, "shared"), (right, "shared")})

    def test_wildcard_open_distributes_bare_names(self) -> None:
        decl = _decl("pkg", wildcard=True, is_open=True)
        left = _module("pkg/left")
        right = _module("pkg/right")
        env = _build(
            [decl],
            {decl.node_id: WildcardTarget(frozenset({left, right}))},
            {left: _exports("pkg/left", "left"), right: _exports("pkg/right", "right")},
        )

        assert env.unqualified == {
            "left": frozenset({(left, "left")}),
            "right": frozenset({(right, "right")}),
        }


class TestDeterministicConstruction:
    def test_public_mappings_have_stable_module_and_name_order(self) -> None:
        decl = _decl("pkg", wildcard=True, is_open=True)
        left = _module("pkg/left")
        right = _module("pkg/right")
        env = _build(
            [decl],
            {decl.node_id: WildcardTarget(frozenset({right, left}))},
            {
                left: _exports("pkg/left", "zeta", "alpha"),
                right: _exports("pkg/right", "zeta", "alpha"),
            },
        )

        assert list(env.contributions) == [left, right]
        assert list(env.contributions[left].members) == ["alpha", "zeta"]
        assert list(env.unqualified) == ["alpha", "zeta"]


class TestQualifierRoutes:
    def test_wildcard_alias_forms_a_merged_facade(self) -> None:
        decl = _decl("pkg", wildcard=True, alias="api")
        first_module = _module("pkg/first")
        second_module = _module("pkg/second")
        env = _build(
            [decl],
            {decl.node_id: WildcardTarget(frozenset({first_module, second_module}))},
            {
                first_module: _exports("pkg/first", "first"),
                second_module: _exports("pkg/second", "second"),
            },
        )

        assert resolve_qualified(env, ("api",), "first") == QualResolutionFound(
            first_module, (first_module, "first")
        )
        assert resolve_qualified(env, ("api",), "second") == QualResolutionFound(
            second_module, (second_module, "second")
        )

    def test_alias_forms_a_merged_facade_with_member_filtering(self) -> None:
        first = _decl("pkg/first", alias="api")
        second = _decl("pkg/second", alias="api")
        first_module = _module("pkg/first")
        second_module = _module("pkg/second")
        env = _build(
            [first, second],
            {
                first.node_id: SingleTarget(first_module),
                second.node_id: SingleTarget(second_module),
            },
            {
                first_module: _exports("pkg/first", "first", "same"),
                second_module: _exports("pkg/second", "second", "same"),
            },
        )

        assert resolve_qualified(env, ("api",), "first") == QualResolutionFound(
            first_module, (first_module, "first")
        )
        assert resolve_qualified(env, ("api",), "same") == QualResolutionAmbiguous(
            ("api",), "same", (first_module, second_module)
        )

    def test_alias_only_route_is_excluded_from_suffixes_and_anchors(self) -> None:
        decl = _decl("std/config", alias="settings")
        module = _module("std/config")
        env = _build(
            [decl],
            {decl.node_id: SingleTarget(module)},
            {module: _exports("std/config", "timeout")},
        )

        assert isinstance(
            resolve_qualified(env, ("config",), "timeout"), QualResolutionUnknownQualifier
        )
        assert isinstance(
            resolve_qualified(env, ("std", "config"), "timeout", anchored=True),
            QualResolutionUnknownQualifier,
        )
        assert resolve_qualified(env, ("settings",), "timeout") == QualResolutionFound(
            module, (module, "timeout")
        )

    def test_plain_and_alias_routes_for_the_same_module_are_both_available(self) -> None:
        plain = _decl("std/config")
        alias = _decl(
            "std/config", alias="settings", mode=ImportMode.USING, items=(_item("timeout"),)
        )
        module = _module("std/config")
        env = _build(
            [plain, alias],
            {plain.node_id: SingleTarget(module), alias.node_id: SingleTarget(module)},
            {module: _exports("std/config", "timeout")},
        )

        assert resolve_qualified(env, ("config",), "timeout") == QualResolutionFound(
            module, (module, "timeout")
        )
        assert resolve_qualified(env, ("settings",), "timeout") == QualResolutionFound(
            module, (module, "timeout")
        )


class TestSuffixResolution:
    def test_full_and_proper_path_suffixes_resolve(self) -> None:
        decl = _decl("std/list/config")
        module = _module("std/list/config")
        env = _build(
            [decl],
            {decl.node_id: SingleTarget(module)},
            {module: _exports("std/list/config", "option")},
        )

        for qualifier in (("std", "list", "config"), ("list", "config"), ("config",)):
            assert resolve_qualified(env, qualifier, "option") == QualResolutionFound(
                module, (module, "option")
            )

    def test_anchor_requires_the_exact_plain_path(self) -> None:
        base = _decl("std/config")
        nested = _decl("new/std/config")
        base_module = _module("std/config")
        nested_module = _module("new/std/config")
        env = _build(
            [base, nested],
            {base.node_id: SingleTarget(base_module), nested.node_id: SingleTarget(nested_module)},
            {
                base_module: _exports("std/config", "option"),
                nested_module: _exports("new/std/config", "option"),
            },
        )

        assert resolve_qualified(
            env, ("std", "config"), "option", anchored=True
        ) == QualResolutionFound(base_module, (base_module, "option"))

    def test_member_filtering_rescues_a_shared_suffix(self) -> None:
        left = _decl("std/config")
        right = _decl("extra/config")
        left_module = _module("std/config")
        right_module = _module("extra/config")
        env = _build(
            [left, right],
            {left.node_id: SingleTarget(left_module), right.node_id: SingleTarget(right_module)},
            {
                left_module: _exports("std/config", "timeout"),
                right_module: _exports("extra/config", "retries"),
            },
        )

        assert resolve_qualified(env, ("config",), "timeout") == QualResolutionFound(
            left_module, (left_module, "timeout")
        )
        assert resolve_qualified(env, ("config",), "missing") == QualResolutionMissingMember(
            ("config",), "missing", (right_module, left_module)
        )

    def test_all_resolution_verdicts_and_no_preference_order(self) -> None:
        base = _decl("std/config")
        nested = _decl("new/std/config")
        alias = _decl("extra/log", alias="config")
        base_module = _module("std/config")
        nested_module = _module("new/std/config")
        alias_module = _module("extra/log")
        env = _build(
            [base, nested, alias],
            {
                base.node_id: SingleTarget(base_module),
                nested.node_id: SingleTarget(nested_module),
                alias.node_id: SingleTarget(alias_module),
            },
            {
                base_module: _exports("std/config", "shared"),
                nested_module: _exports("new/std/config", "shared"),
                alias_module: _exports("extra/log", "shared", "alias_only"),
            },
        )

        assert isinstance(
            resolve_qualified(env, ("unknown",), "shared"), QualResolutionUnknownQualifier
        )
        assert resolve_qualified(env, ("config",), "absent") == QualResolutionMissingMember(
            ("config",), "absent", (alias_module, nested_module, base_module)
        )
        assert resolve_qualified(env, ("config",), "shared") == QualResolutionAmbiguous(
            ("config",), "shared", (alias_module, nested_module, base_module)
        )
        assert resolve_qualified(env, ("config",), "alias_only") == QualResolutionFound(
            alias_module, (alias_module, "alias_only")
        )
        assert resolve_qualified(env, ("std", "config"), "shared") == QualResolutionAmbiguous(
            ("std", "config"), "shared", (nested_module, base_module)
        )
