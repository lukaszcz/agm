"""Unit tests for import contributions and qualified routes."""

from __future__ import annotations

import pytest

from agm.agl.modules.ids import ModuleId
from agm.agl.scope.imports import (
    ImportEnv,
    NameAtom,
    QName,
    QualResolutionFound,
    QualResolutionMissingMember,
    SingleTarget,
    build_import_env,
    resolve_qualified,
)
from agm.agl.scope.symbols import AglScopeError
from agm.agl.syntax.nodes import ImportDecl, ImportItem
from agm.agl.syntax.spans import UNKNOWN_SOURCE, SourceSpan


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
    tail: tuple[ImportItem, ...] | None = None,
    hidden: tuple[ImportItem, ...] = (),
) -> ImportDecl:
    return ImportDecl(
        module_path=tuple(path.split("/")),
        wildcard=False,
        alias=alias,
        tail=tail,
        hidden=hidden,
        span=_span(),
        node_id=_node_id(),
    )


def _item(name: str, rename: str | None = None) -> ImportItem:
    return ImportItem(name, rename, _span(), _node_id())


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
    )


def test_plain_import_contributes_the_full_qualified_surface_without_bare_names() -> None:
    decl = _decl("tools/text")
    module = _module("tools/text")

    env = _build([decl], {module: _exports("tools/text", "trim", "split")})

    assert env.unqualified == {}
    assert resolve_qualified(env, ("text",), "trim") == QualResolutionFound(
        module, (module, "trim")
    )
    assert resolve_qualified(env, ("text",), "split") == QualResolutionFound(
        module, (module, "split")
    )


def test_positive_tail_injects_bare_names_without_narrowing_qualified_access() -> None:
    decl = _decl("tools/text", tail=(_item("trim"),))
    module = _module("tools/text")

    env = _build([decl], {module: _exports("tools/text", "trim", "split")})

    assert env.unqualified == {"trim": frozenset({(module, "trim")})}
    assert resolve_qualified(env, ("text",), "split") == QualResolutionFound(
        module, (module, "split")
    )


def test_tail_rename_is_additive_for_bare_spelling() -> None:
    decl = _decl("tools/text", tail=(_item("trim", "clean"),))
    module = _module("tools/text")

    env = _build([decl], {module: _exports("tools/text", "trim", "split")})

    assert env.unqualified == {
        "clean": frozenset({(module, "trim")}),
        "trim": frozenset({(module, "trim")}),
    }
    assert resolve_qualified(env, ("text",), "trim") == QualResolutionFound(
        module, (module, "trim")
    )


def test_plain_hiding_repairs_a_shared_suffix_route() -> None:
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

    assert resolve_qualified(env, ("config",), "shared") == QualResolutionFound(
        right, (right, "shared")
    )


def test_wildcard_tail_distributes_hiding_to_routes_and_bare_names() -> None:
    decl = _decl("tools/text", tail=(), hidden=(_item("debug"),))
    module = _module("tools/text")

    env = _build([decl], {module: _exports("tools/text", "trim", "debug")})

    assert env.unqualified == {"trim": frozenset({(module, "trim")})}
    assert resolve_qualified(env, ("text",), "trim") == QualResolutionFound(
        module, (module, "trim")
    )
    assert isinstance(resolve_qualified(env, ("text",), "debug"), QualResolutionMissingMember)


def test_repeated_imports_union_each_declarations_unhidden_routes() -> None:
    first = _decl("tools/text", hidden=(_item("trim"),))
    second = _decl("tools/text", hidden=(_item("split"),))
    module = _module("tools/text")

    env = _build([first, second], {module: _exports("tools/text", "trim", "split")})

    assert set(env.contributions[module].members) == {"trim", "split"}


def test_hiding_and_tail_atoms_must_name_public_members() -> None:
    tail = _decl("tools/text", tail=(_item("unknown"),))
    hidden = _decl("tools/text", hidden=(_item("unknown"),))
    module = _module("tools/text")

    for decl in (tail, hidden):
        with pytest.raises(AglScopeError):
            _build([decl], {module: _exports("tools/text", "trim")})
