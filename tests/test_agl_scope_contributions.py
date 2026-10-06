"""Tests for contribution construction, scoping, and route aliases."""

from __future__ import annotations

from agm.agl.modules.ids import ModuleId
from agm.agl.scope.imports import (
    ImportEnv,
    ImportTarget,
    NameAtom,
    QName,
    SingleTarget,
    WildcardTarget,
    build_import_env,
    qualifier_candidates,
    qualifier_member_ways,
)
from agm.agl.scope.symbols import (
    BinderKind,
    BindingRef,
    ContributionLayer,
    ScopeNode,
)
from agm.agl.syntax.nodes import ImportDecl, ImportItem, ScopeSegment
from tests._agl_helpers import dummy_span


def resolve_bare_contribution(scope: ScopeNode, name: NameAtom) -> set[BindingRef] | None:
    """Return the candidates of the first layer, from *scope* outward, contributing *name*."""
    layer: ScopeNode | None = scope
    while layer is not None:
        stored = layer.bare_contributions.get(name)
        if stored:
            return set(stored)
        layer = layer.parent
    return None


_next_node_id = 0


def _node_id() -> int:
    global _next_node_id
    _next_node_id += 1
    return _next_node_id


def _decl(
    path: str,
    *,
    wildcard: bool = False,
    alias: str | None = None,
    tail: tuple[ImportItem, ...] | None = None,
    hidden: tuple[ImportItem, ...] = (),
    scope_path: tuple[ScopeSegment, ...] = (),
) -> ImportDecl:
    return ImportDecl(
        module_path=tuple(path.split("/")),
        wildcard=wildcard,
        alias=alias,
        tail=tail,
        hidden=hidden,
        span=dummy_span(),
        node_id=_node_id(),
        scope_path=scope_path,
    )


def _item(name: str, rename: str | None = None) -> ImportItem:
    return ImportItem(name, rename, dummy_span(), _node_id())


def _region(name: str) -> tuple[ScopeSegment, ...]:
    return (ScopeSegment(name, dummy_span(), _node_id()),)


def _module(path: str) -> ModuleId:
    return ModuleId.from_path(path)


def _exports(path: str, *names: str) -> dict[NameAtom, QName]:
    module = _module(path)
    return {name: (module, name) for name in names}


def _build(
    decls: tuple[ImportDecl, ...],
    targets: dict[int, ImportTarget],
    exports: dict[ModuleId, dict[NameAtom, QName]],
) -> ImportEnv:
    return build_import_env(decls, targets, exports, {module: {} for module in exports})


def test_region_tailed_import_keeps_its_bare_contribution_regional() -> None:
    decl = _decl("lib/api", tail=(_item("one"),), scope_path=_region("A"))
    module = _module("lib/api")

    env = _build(
        (decl,),
        {decl.node_id: SingleTarget(module)},
        {module: _exports("lib/api", "one", "two")},
    )

    assert env.unqualified == {}
    assert {atom: set(origins) for atom, origins in env.decl_bare_ways[decl.node_id].items()} == {
        "one": {(module, "one")}
    }
    assert set(env.contributions[module].members) == {"one", "two"}


def test_alias_route_retains_full_surface_and_records_its_own_hiding() -> None:
    decl = _decl("std/config", alias="settings", hidden=(_item("debug"),))
    module = _module("std/config")

    env = _build(
        (decl,),
        {decl.node_id: SingleTarget(module)},
        {module: _exports("std/config", "timeout", "debug")},
    )

    assert set(qualifier_member_ways(env, ("settings",), "timeout")) == {(module, "timeout")}
    assert "debug" in env.contributions[module].members
    assert {item.declaration for item in env.decl_hiding[decl.node_id]} == {(module, "debug")}


def test_alias_route_does_not_also_contribute_the_module_suffix() -> None:
    decl = _decl("std/config", alias="settings")
    module = _module("std/config")

    env = _build(
        (decl,),
        {decl.node_id: SingleTarget(module)},
        {module: _exports("std/config", "timeout")},
    )

    assert set(qualifier_member_ways(env, ("settings",), "timeout")) == {(module, "timeout")}
    assert qualifier_candidates(env, ("config",), anchored=False) == ()


def test_alias_hiding_is_recorded_on_the_alias_declaration_only() -> None:
    alias = _decl("std/config", alias="settings", hidden=(_item("debug"),))
    plain = _decl("std/config")
    module = _module("std/config")

    env = _build(
        (alias, plain),
        {alias.node_id: SingleTarget(module), plain.node_id: SingleTarget(module)},
        {module: _exports("std/config", "timeout", "debug")},
    )

    assert set(qualifier_member_ways(env, ("config",), "debug")) == {(module, "debug")}
    assert plain.node_id not in env.decl_hiding
    ways = env.contributions[module].routes["settings"].member_ways["debug"]
    assert {way.node_id for way in ways} == {alias.node_id}


def test_regional_tail_bare_contributions_narrow_at_the_scope_seam() -> None:
    left_decl = _decl("left/api", tail=(_item("selected"),), scope_path=_region("Left"))
    right_decl = _decl("right/api", tail=(_item("selected"),), scope_path=_region("Right"))
    left_module = _module("left/api")
    right_module = _module("right/api")
    env = _build(
        (left_decl, right_decl),
        {
            left_decl.node_id: SingleTarget(left_module),
            right_decl.node_id: SingleTarget(right_module),
        },
        {
            left_module: _exports("left/api", "selected"),
            right_module: _exports("right/api", "selected"),
        },
    )
    root = ScopeNode(node_id=0)
    left_scope = ScopeNode(node_id=1, parent=root, scope_path=("Left",))
    right_scope = ScopeNode(node_id=2, parent=root, scope_path=("Right",))

    for scope, decl in ((left_scope, left_decl), (right_scope, right_decl)):
        for name, qnames in env.decl_bare_ways[decl.node_id].items():
            for module, _source in qnames:
                binding_name = name if isinstance(name, str) else name[-1]
                scope.contribute_bare(
                    name,
                    BindingRef(
                        binding_name,
                        False,
                        dummy_span(),
                        decl.node_id,
                        BinderKind.function_binding,
                        module,
                    ),
                    ContributionLayer.IMPORTED,
                )

    assert resolve_bare_contribution(root, "selected") is None
    assert {ref.module_id for ref in resolve_bare_contribution(left_scope, "selected") or ()} == {
        left_module
    }
    assert {ref.module_id for ref in resolve_bare_contribution(right_scope, "selected") or ()} == {
        right_module
    }


def test_import_tail_and_use_route_of_the_same_origin_are_not_ambiguous() -> None:
    decl = _decl("lib/api", tail=(_item("selected"),))
    module = _module("lib/api")
    root = ScopeNode(node_id=0)
    root.contribute_bare(
        "selected",
        BindingRef(
            "selected",
            False,
            dummy_span(),
            decl.node_id,
            BinderKind.function_binding,
            module,
        ),
        ContributionLayer.IMPORTED,
    )

    candidates = resolve_bare_contribution(root, "selected")

    assert candidates is not None
    assert len(candidates) == 1


def test_wildcard_tails_apply_bare_contributions_per_module() -> None:
    decl = _decl("pkg", wildcard=True, tail=(_item("shared", "api"),))
    left = _module("pkg/left")
    right = _module("pkg/right")

    env = _build(
        (decl,),
        {decl.node_id: WildcardTarget(frozenset({left, right}))},
        {
            left: _exports("pkg/left", "shared", "left"),
            right: _exports("pkg/right", "shared", "right"),
        },
    )

    assert env.unqualified["shared"] == frozenset({(left, "shared"), (right, "shared")})
    assert env.unqualified["api"] == frozenset({(left, "shared"), (right, "shared")})
    assert set(env.contributions[left].members) == {"shared", "left"}
    assert set(env.contributions[right].members) == {"shared", "right"}
