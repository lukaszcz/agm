"""Unit tests for contribution-based slash import resolution."""

from __future__ import annotations

from agm.agl.modules.ids import ModuleId
from agm.agl.scope.imports import (
    QualResolutionAmbiguous,
    QualResolutionFound,
    QualResolutionMissingMember,
    QualResolutionUnknownQualifier,
    SingleTarget,
    WildcardTarget,
    build_import_env,
    resolve_qualified,
)
from agm.agl.syntax.nodes import ImportDecl, ImportItem
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
    )


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
):
    return build_import_env(tuple(decls), targets, exports)


def test_plain_import_contributes_only_qualified_members() -> None:
    decl = _decl("tools/text")
    module = _module("tools/text")

    env = _build(
        [decl],
        {decl.node_id: SingleTarget(module)},
        {module: _exports("tools/text", "trim")},
    )

    assert env.unqualified == {}
    assert resolve_qualified(env, ("text",), "trim") == QualResolutionFound(
        module, (module, "trim")
    )


def test_open_import_injects_selected_members() -> None:
    decl = _decl("tools/text", is_open=True, mode=ImportMode.HIDING, items=(_item("debug"),))
    module = _module("tools/text")

    env = _build(
        [decl],
        {decl.node_id: SingleTarget(module)},
        {module: _exports("tools/text", "trim", "debug")},
    )

    assert env.unqualified == {"trim": frozenset({(module, "trim")})}
    assert isinstance(resolve_qualified(env, ("text",), "debug"), QualResolutionMissingMember)


def test_using_injects_only_its_renamed_members() -> None:
    decl = _decl(
        "tools/text",
        mode=ImportMode.USING,
        items=(_item("trim", "clean"),),
    )
    module = _module("tools/text")

    env = _build(
        [decl],
        {decl.node_id: SingleTarget(module)},
        {module: _exports("tools/text", "trim", "split")},
    )

    assert env.unqualified == {"clean": frozenset({(module, "trim")})}
    assert resolve_qualified(env, ("text",), "clean") == QualResolutionFound(
        module, (module, "trim")
    )
    assert isinstance(resolve_qualified(env, ("text",), "trim"), QualResolutionMissingMember)


def test_alias_is_a_route_but_not_a_suffix_or_anchor() -> None:
    decl = _decl("std/config", alias="settings")
    module = _module("std/config")
    env = _build(
        [decl],
        {decl.node_id: SingleTarget(module)},
        {module: _exports("std/config", "timeout")},
    )

    assert resolve_qualified(env, ("settings",), "timeout") == QualResolutionFound(
        module, (module, "timeout")
    )
    assert isinstance(
        resolve_qualified(env, ("config",), "timeout"), QualResolutionUnknownQualifier
    )
    assert isinstance(
        resolve_qualified(env, ("std", "config"), "timeout", anchored=True),
        QualResolutionUnknownQualifier,
    )


def test_suffix_resolution_filters_members_before_reporting_ambiguity() -> None:
    left_decl = _decl("one/config")
    right_decl = _decl("two/config")
    left = _module("one/config")
    right = _module("two/config")
    env = _build(
        [left_decl, right_decl],
        {
            left_decl.node_id: SingleTarget(left),
            right_decl.node_id: SingleTarget(right),
        },
        {
            left: _exports("one/config", "timeout", "shared"),
            right: _exports("two/config", "retries", "shared"),
        },
    )

    assert resolve_qualified(env, ("config",), "timeout") == QualResolutionFound(
        left, (left, "timeout")
    )
    assert resolve_qualified(env, ("config",), "shared") == QualResolutionAmbiguous(
        ("config",), "shared", (left, right)
    )


def test_anchored_route_requires_the_exact_plain_path() -> None:
    base_decl = _decl("std/config")
    nested_decl = _decl("extra/std/config")
    base = _module("std/config")
    nested = _module("extra/std/config")
    env = _build(
        [base_decl, nested_decl],
        {
            base_decl.node_id: SingleTarget(base),
            nested_decl.node_id: SingleTarget(nested),
        },
        {
            base: _exports("std/config", "timeout"),
            nested: _exports("extra/std/config", "timeout"),
        },
    )

    assert resolve_qualified(env, ("std", "config"), "timeout", anchored=True) == (
        QualResolutionFound(base, (base, "timeout"))
    )


def test_wildcard_distributes_contributions_and_open_names() -> None:
    decl = _decl("plugins", wildcard=True, is_open=True)
    alpha = _module("plugins/alpha")
    beta = _module("plugins/beta")
    env = _build(
        [decl],
        {decl.node_id: WildcardTarget(frozenset({alpha, beta}))},
        {
            alpha: _exports("plugins/alpha", "alpha"),
            beta: _exports("plugins/beta", "beta"),
        },
    )

    assert env.unqualified == {
        "alpha": frozenset({(alpha, "alpha")}),
        "beta": frozenset({(beta, "beta")}),
    }
    assert resolve_qualified(env, ("alpha",), "alpha") == QualResolutionFound(
        alpha, (alpha, "alpha")
    )


def test_repeated_contributions_union_members_and_open_names() -> None:
    plain = _decl("tools/text", mode=ImportMode.HIDING, items=(_item("debug"),))
    selected = _decl("tools/text", mode=ImportMode.USING, items=(_item("debug"),))
    module = _module("tools/text")
    env = _build(
        [plain, selected],
        {plain.node_id: SingleTarget(module), selected.node_id: SingleTarget(module)},
        {module: _exports("tools/text", "trim", "debug")},
    )

    assert set(env.contributions[module].members) == {"trim", "debug"}
    assert env.unqualified == {"debug": frozenset({(module, "debug")})}
