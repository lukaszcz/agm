"""Tests for contribution construction, scoping, and route aliases."""

from __future__ import annotations

from agm.agl.modules.ids import ModuleId
from agm.agl.scope.imports import (
    NameAtom,
    QName,
    QualResolutionFound,
    QualResolutionUnknownQualifier,
    SingleTarget,
    WildcardTarget,
    build_import_env,
    resolve_qualified,
)
from agm.agl.scope.symbols import (
    BinderKind,
    BindingRef,
    ScopeNode,
    resolve_bare_contribution_layer,
)
from agm.agl.syntax.nodes import ImportDecl, ImportItem, ScopeSegment
from agm.agl.syntax.spans import UNKNOWN_SOURCE, SourceSpan


def _span() -> SourceSpan:
    return SourceSpan(1, 1, 1, 1, 0, 0, UNKNOWN_SOURCE)


def resolve_bare_contribution(scope: ScopeNode, name: NameAtom) -> set[BindingRef] | None:
    """Return just the candidates the nearest contributing layer holds for *name*."""
    resolved = resolve_bare_contribution_layer(scope, name)
    return None if resolved is None else resolved[1]


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
        span=_span(),
        node_id=_node_id(),
        scope_path=scope_path,
    )


def _item(name: str, rename: str | None = None) -> ImportItem:
    return ImportItem(name, rename, _span(), _node_id())


def _region(name: str) -> tuple[ScopeSegment, ...]:
    return (ScopeSegment(name, _span(), _node_id()),)


def _module(path: str) -> ModuleId:
    return ModuleId.from_path(path)


def _exports(path: str, *names: str) -> dict[NameAtom, QName]:
    module = _module(path)
    return {name: (module, name) for name in names}


def test_region_tailed_import_keeps_its_bare_contribution_regional() -> None:
    decl = _decl("lib/api", tail=(_item("one"),), scope_path=_region("A"))
    module = _module("lib/api")

    env = build_import_env(
        (decl,),
        {decl.node_id: SingleTarget(module)},
        {module: _exports("lib/api", "one", "two")},
    )

    assert env.unqualified == {}
    assert dict(env.decl_bare[decl.node_id]) == {"one": frozenset({(module, "one")})}
    assert set(env.contributions[module].members) == {"one", "two"}


def test_alias_route_retains_full_surface_except_its_own_hiding() -> None:
    decl = _decl("std/config", alias="settings", hidden=(_item("debug"),))
    module = _module("std/config")

    env = build_import_env(
        (decl,),
        {decl.node_id: SingleTarget(module)},
        {module: _exports("std/config", "timeout", "debug")},
    )

    assert resolve_qualified(env, ("settings",), "timeout") == QualResolutionFound(
        module, (module, "timeout")
    )
    assert "debug" not in env.contributions[module].members


def test_alias_route_does_not_also_contribute_the_module_suffix() -> None:
    decl = _decl("std/config", alias="settings")
    module = _module("std/config")

    env = build_import_env(
        (decl,),
        {decl.node_id: SingleTarget(module)},
        {module: _exports("std/config", "timeout")},
    )

    assert resolve_qualified(env, ("settings",), "timeout") == QualResolutionFound(
        module, (module, "timeout")
    )
    assert isinstance(
        resolve_qualified(env, ("config",), "timeout"), QualResolutionUnknownQualifier
    )


def test_alias_hiding_remains_limited_to_the_alias_declaration() -> None:
    alias = _decl("std/config", alias="settings", hidden=(_item("debug"),))
    plain = _decl("std/config")
    module = _module("std/config")

    env = build_import_env(
        (alias, plain),
        {alias.node_id: SingleTarget(module), plain.node_id: SingleTarget(module)},
        {module: _exports("std/config", "timeout", "debug")},
    )

    assert resolve_qualified(env, ("config",), "debug") == QualResolutionFound(
        module, (module, "debug")
    )
    assert "debug" not in env.contributions[module].alias_members["settings"]


def test_regional_tail_bare_contributions_narrow_at_the_scope_seam() -> None:
    left_decl = _decl("left/api", tail=(_item("selected"),), scope_path=_region("Left"))
    right_decl = _decl("right/api", tail=(_item("selected"),), scope_path=_region("Right"))
    left_module = _module("left/api")
    right_module = _module("right/api")
    env = build_import_env(
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
        for name, qnames in env.decl_bare[decl.node_id].items():
            for module, _source in qnames:
                binding_name = name if isinstance(name, str) else name[-1]
                scope.contribute_bare(
                    name,
                    BindingRef(
                        binding_name,
                        False,
                        _span(),
                        decl.node_id,
                        BinderKind.function_binding,
                        module,
                    ),
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
            _span(),
            decl.node_id,
            BinderKind.function_binding,
            module,
        ),
    )

    candidates = resolve_bare_contribution(root, "selected")

    assert candidates is not None
    assert len(candidates) == 1


def test_wildcard_tails_apply_bare_contributions_per_module() -> None:
    decl = _decl("pkg", wildcard=True, tail=(_item("shared", "api"),))
    left = _module("pkg/left")
    right = _module("pkg/right")

    env = build_import_env(
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
