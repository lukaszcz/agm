"""Unit tests for import contributions and qualified routes."""

from __future__ import annotations

import pytest

from agm.agl.modules.ids import ModuleId
from agm.agl.scope.imports import (
    ImportEnv,
    ModuleContribution,
    NameAtom,
    QName,
    RouteSurface,
    SingleTarget,
    build_import_env,
    qualifier_member_ways,
    qualifier_members,
    validate_import_items,
)
from agm.agl.scope.symbols import AglScopeError
from agm.agl.syntax.nodes import ImportDecl, ImportItem
from tests._agl_helpers import dummy_span

_next_node_id = 0


def _node_id() -> int:
    global _next_node_id
    _next_node_id += 1
    return _next_node_id


def _decl(
    path: str,
    *,
    alias: str | None = None,
    tail: tuple[ImportItem, ...] | None = None,
    hidden: tuple[ImportItem, ...] = (),
) -> ImportDecl:
    return ImportDecl(
        module_path=tuple(path.split("/")),
        wildcard=False,
        alias=alias,
        tail=tail,
        hidden=hidden,
        span=dummy_span(),
        node_id=_node_id(),
    )


def _item(name: str, rename: str | None = None) -> ImportItem:
    return ImportItem(name, rename, dummy_span(), _node_id())


def _module(path: str) -> ModuleId:
    return ModuleId.from_path(path)


def _exports(path: str, *names: str) -> dict[NameAtom, QName]:
    module = _module(path)
    return {name: (module, name) for name in names}


def _build(decls: list[ImportDecl], exports: dict[ModuleId, dict[NameAtom, QName]]) -> ImportEnv:
    return build_import_env(
        tuple(decls),
        {decl.node_id: SingleTarget(_module("/".join(decl.module_path))) for decl in decls},
        exports,
        {module: {} for module in exports},
    )


def test_alias_route_without_members_is_not_a_use_target() -> None:
    module = _module("tools/text")
    env = ImportEnv(
        contributions={module: ModuleContribution(module, {}, False, frozenset({"text"}))},
        unqualified={},
    )

    assert qualifier_members(env, ("text",), anchored=False) == ()


def test_plain_import_contributes_the_full_qualified_surface_without_bare_names() -> None:
    decl = _decl("tools/text")
    module = _module("tools/text")

    env = _build([decl], {module: _exports("tools/text", "trim", "split")})

    assert env.unqualified == {}
    assert set(qualifier_member_ways(env, ("text",), "trim")) == {(module, "trim")}
    assert set(qualifier_member_ways(env, ("text",), "split")) == {(module, "split")}


def test_wildcard_tail_exposes_a_member_path_beneath_its_bare_owner() -> None:
    decl = _decl("tools/geo", tail=())
    module = _module("tools/geo")
    point = (module, ("Geo", "Point"))

    env = _build([decl], {module: {"Geo": (module, "Geo"), ("Geo", "Point"): point}})

    assert env.unqualified[("Geo", "Point")] == frozenset({point})


def test_positive_tail_injects_bare_names_without_narrowing_qualified_access() -> None:
    decl = _decl("tools/text", tail=(_item("trim"),))
    module = _module("tools/text")

    env = _build([decl], {module: _exports("tools/text", "trim", "split")})

    assert env.unqualified == {"trim": frozenset({(module, "trim")})}
    assert set(qualifier_member_ways(env, ("text",), "split")) == {(module, "split")}


def test_tail_rename_is_additive_for_bare_spelling() -> None:
    decl = _decl("tools/text", tail=(_item("trim", "clean"),))
    module = _module("tools/text")

    env = _build([decl], {module: _exports("tools/text", "trim", "split")})

    assert env.unqualified == {
        "clean": frozenset({(module, "trim")}),
        "trim": frozenset({(module, "trim")}),
    }
    assert set(qualifier_member_ways(env, ("text",), "trim")) == {(module, "trim")}


def test_shared_route_resolves_duplicate_contributions_to_the_same_origin() -> None:
    left = _module("pkg/left")
    right = _module("pkg/right")
    origin = _module("core")
    qname = (origin, "shared")
    env = ImportEnv(
        contributions={
            left: ModuleContribution(
                left,
                {"shared": qname},
                False,
                frozenset({"Facade"}),
                routes={
                    "Facade": RouteSurface(
                        members={"shared": qname}, member_ways={"shared": frozenset()}
                    )
                },
            ),
            right: ModuleContribution(
                right,
                {"shared": qname},
                False,
                frozenset({"Facade"}),
                routes={
                    "Facade": RouteSurface(
                        members={"shared": qname}, member_ways={"shared": frozenset()}
                    )
                },
            ),
        },
        unqualified={},
    )

    assert set(qualifier_member_ways(env, ("Facade",), "shared")) == {qname}


def test_suffix_route_keeps_a_hidden_export_and_records_the_hiding_per_declaration() -> None:
    left_decl = _decl("one/config", hidden=(_item("shared"),))
    right_decl = _decl("two/config")
    left = _module("one/config")
    right = _module("two/config")

    env = _build(
        [left_decl, right_decl],
        {
            left: _exports("one/config", "shared"),
            right: _exports("two/config", "shared"),
        },
    )

    reached = {
        qname: {way.node_id for way in ways}
        for qname, ways in qualifier_member_ways(env, ("config",), "shared").items()
    }
    assert reached == {
        (left, "shared"): {left_decl.node_id},
        (right, "shared"): {right_decl.node_id},
    }
    assert {item.declaration for item in env.decl_hiding[left_decl.node_id]} == {(left, "shared")}
    assert right_decl.node_id not in env.decl_hiding


def test_wildcard_tail_distributes_hiding_to_routes_and_bare_names() -> None:
    decl = _decl("tools/text", tail=(), hidden=(_item("debug"),))
    module = _module("tools/text")

    env = _build([decl], {module: _exports("tools/text", "trim", "debug")})

    assert set(env.unqualified) == {"trim", "debug"}
    assert set(qualifier_member_ways(env, ("text",), "trim")) == {(module, "trim")}
    assert set(qualifier_member_ways(env, ("text",), "debug")) == {(module, "debug")}
    assert {item.declaration for item in env.decl_hiding[decl.node_id]} == {(module, "debug")}


def test_repeated_imports_keep_hidden_exports_and_each_declarations_hiding() -> None:
    first = _decl("tools/text", hidden=(_item("trim"),))
    second = _decl("tools/text", hidden=(_item("split"),))
    module = _module("tools/text")

    env = _build([first, second], {module: _exports("tools/text", "trim", "split")})

    assert set(env.contributions[module].members) == {"trim", "split"}
    assert {item.declaration[1] for item in env.decl_hiding[first.node_id]} == {"trim"}
    assert {item.declaration[1] for item in env.decl_hiding[second.node_id]} == {"split"}


def test_hiding_and_tail_atoms_must_name_public_members() -> None:
    tail = _decl("tools/text", tail=(_item("unknown"),))
    hidden = _decl("tools/text", hidden=(_item("unknown"),))
    module = _module("tools/text")

    for decl in (tail, hidden):
        with pytest.raises(AglScopeError):
            validate_import_items(
                (decl,),
                {decl.node_id: SingleTarget(module)},
                {module: _exports("tools/text", "trim")},
                {module: {}},
                (),
            )
