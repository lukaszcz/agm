"""Unit tests for the pure zone-binding algorithm in ``semantics.arguments``.

These tests drive ``bind_arguments`` directly with plain strings and ints as
items, so they exercise the algorithm without any AST, span, or typechecker
dependency. A ``str`` item is bare-name-capable (like a ``VarRef``); an ``int``
item is opaque (like any other expression). ``tests/test_agl_arguments.py``
covers the AST-facing wrapper built on top of this.
"""

from __future__ import annotations

import pytest

from agm.agl.semantics.arguments import (
    ArgumentBindingError,
    ArgumentBindingErrorKind,
    BindParam,
    ParamZone,
    bind_arguments,
    positional_field_names,
)

POSITIONAL_ONLY = ParamZone.POSITIONAL_ONLY
STANDARD = ParamZone.STANDARD
NAMED_ONLY = ParamZone.NAMED_ONLY

Item = str | int


def _bare_name(item: Item) -> str | None:
    return item if isinstance(item, str) else None


def _bind(
    params: list[BindParam],
    positional: list[Item],
    named: list[tuple[str, Item]],
) -> tuple[Item | None, ...]:
    return bind_arguments(params, positional, named, bare_name=_bare_name)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_all_positional_standard() -> None:
    params = [BindParam("x", STANDARD, False), BindParam("y", STANDARD, False)]
    result = _bind(params, [1, 2], [])
    assert result == (1, 2)


def test_named_arg_for_standard() -> None:
    params = [BindParam("x", STANDARD, False), BindParam("y", STANDARD, False)]
    result = _bind(params, [1], [("y", 2)])
    assert result == (1, 2)


def test_defaults_fill_missing() -> None:
    params = [
        BindParam("x", STANDARD, False),
        BindParam("y", STANDARD, True),
        BindParam("z", STANDARD, True),
    ]
    result = _bind(params, [1], [])
    assert result == (1, None, None)


def test_positional_only_filled_positionally() -> None:
    params = [BindParam("x", POSITIONAL_ONLY, False), BindParam("y", STANDARD, False)]
    result = _bind(params, [42, 7], [])
    assert result == (42, 7)


def test_named_only_via_named_arg() -> None:
    params = [BindParam("x", STANDARD, False), BindParam("z", NAMED_ONLY, False)]
    result = _bind(params, [1], [("z", 99)])
    assert result == (1, 99)


def test_named_only_shorthand_bare_positional() -> None:
    """A bare-name positional in named-only territory is shorthand for name=name."""
    params = [BindParam("x", STANDARD, False), BindParam("z", NAMED_ONLY, False)]
    result = _bind(params, [1, "z"], [])
    assert result == (1, "z")


def test_positional_skips_leading_named_only() -> None:
    """A positional arg binds the trailing standard field even with a leading
    named-only field (e.g. an exception's inherited ``message`` field)."""
    params = [BindParam("message", NAMED_ONLY, False), BindParam("code", STANDARD, False)]
    result = _bind(params, [7], [("message", "boom")])
    assert result == ("boom", 7)


def test_mixed_zones_all_positional() -> None:
    params = [
        BindParam("a", POSITIONAL_ONLY, False),
        BindParam("b", STANDARD, False),
        BindParam("c", NAMED_ONLY, True),
    ]
    result = _bind(params, [10, 20], [])
    assert result == (10, 20, None)


def test_all_named_only_via_named_args() -> None:
    params = [BindParam("x", NAMED_ONLY, False), BindParam("y", NAMED_ONLY, False)]
    result = _bind(params, [], [("x", 1), ("y", 2)])
    assert result == (1, 2)


def test_named_only_default_unfilled() -> None:
    params = [BindParam("x", NAMED_ONLY, True), BindParam("y", NAMED_ONLY, False)]
    result = _bind(params, [], [("y", "hello")])
    assert result == (None, "hello")


def test_no_args_all_defaults() -> None:
    params = [BindParam("a", STANDARD, True), BindParam("b", NAMED_ONLY, True)]
    result = _bind(params, [], [])
    assert result == (None, None)


def test_empty_params_empty_args() -> None:
    assert _bind([], [], []) == ()


def test_named_accepts_an_iterable_consumed_once() -> None:
    """``named`` need only be iterable once, e.g. a ``dict.items()`` view."""
    params = [BindParam("x", STANDARD, False), BindParam("y", NAMED_ONLY, False)]
    result = bind_arguments(params, [1], {"y": 2}.items(), bare_name=_bare_name)
    assert result == (1, 2)


def test_bare_name_defaults_to_no_bare_items() -> None:
    """Omitting ``bare_name`` treats every positional item as non-bare."""
    params = [BindParam("x", STANDARD, False)]
    result = bind_arguments(params, [1], [])
    assert result == (1,)


# ---------------------------------------------------------------------------
# Error-branch tests: every ArgumentBindingErrorKind with its positional/named payload
# ---------------------------------------------------------------------------


def test_too_many_positional_all_standard() -> None:
    """Opaque extra positional past all-standard params (no named-only) → the extra
    arg's positional index, no name."""
    params = [BindParam("x", STANDARD, False)]
    with pytest.raises(ArgumentBindingError) as exc_info:
        _bind(params, [1, 2], [])
    err = exc_info.value
    assert err.kind is ArgumentBindingErrorKind.TOO_MANY_POSITIONAL
    assert err.positional_index == 1
    assert err.named_index is None
    assert err.name is None


def test_too_many_positional_no_named_only_zero_params() -> None:
    with pytest.raises(ArgumentBindingError) as exc_info:
        _bind([], [1], [])
    err = exc_info.value
    assert err.kind is ArgumentBindingErrorKind.TOO_MANY_POSITIONAL
    assert err.positional_index == 0
    assert err.named_index is None


def test_positional_in_named_only_territory() -> None:
    """Opaque positional arg once positional-capable params are exhausted, with a
    named-only param present → distinct from the no-named-only overflow case."""
    params = [BindParam("x", STANDARD, False), BindParam("z", NAMED_ONLY, False)]
    with pytest.raises(ArgumentBindingError) as exc_info:
        _bind(params, [1, 2], [])
    err = exc_info.value
    assert err.kind is ArgumentBindingErrorKind.POSITIONAL_IN_NAMED_ONLY
    assert err.positional_index == 1
    assert err.named_index is None
    assert err.name is None


def test_positional_in_named_only_territory_no_positional_capable_params() -> None:
    """No positional-capable params at all but a named-only one exists → an opaque
    positional still reports as landing in named-only territory."""
    params = [BindParam("x", NAMED_ONLY, False)]
    with pytest.raises(ArgumentBindingError) as exc_info:
        _bind(params, [42], [])
    err = exc_info.value
    assert err.kind is ArgumentBindingErrorKind.POSITIONAL_IN_NAMED_ONLY
    assert err.positional_index == 0
    assert err.named_index is None


def test_pos_only_by_name_rejected() -> None:
    params = [BindParam("x", POSITIONAL_ONLY, False), BindParam("y", STANDARD, False)]
    with pytest.raises(ArgumentBindingError) as exc_info:
        _bind(params, [1], [("x", 99)])
    err = exc_info.value
    assert err.kind is ArgumentBindingErrorKind.POSITIONAL_ONLY_BY_NAME
    assert err.name == "x"
    assert err.named_index == 0
    assert err.positional_index is None


def test_unknown_named_arg() -> None:
    params = [BindParam("x", STANDARD, False)]
    with pytest.raises(ArgumentBindingError) as exc_info:
        _bind(params, [1], [("oops", 2)])
    err = exc_info.value
    assert err.kind is ArgumentBindingErrorKind.UNKNOWN_NAME
    assert err.name == "oops"
    assert err.named_index == 0
    assert err.positional_index is None


def test_bare_shorthand_unknown_named_only() -> None:
    """Bare positional shorthand with no matching named-only param → unknown, with
    both the name and the offending positional index."""
    params = [BindParam("x", STANDARD, False), BindParam("z", NAMED_ONLY, False)]
    with pytest.raises(ArgumentBindingError) as exc_info:
        _bind(params, [1, "w"], [])
    err = exc_info.value
    assert err.kind is ArgumentBindingErrorKind.UNKNOWN_NAME
    assert err.name == "w"
    assert err.positional_index == 1
    assert err.named_index is None


def test_duplicate_positional_and_named() -> None:
    params = [BindParam("x", STANDARD, False), BindParam("y", STANDARD, True)]
    with pytest.raises(ArgumentBindingError) as exc_info:
        _bind(params, [1], [("x", 2)])
    err = exc_info.value
    assert err.kind is ArgumentBindingErrorKind.DUPLICATE
    assert err.name == "x"
    assert err.named_index == 0
    assert err.positional_index is None


def test_duplicate_shorthand_and_named() -> None:
    """Bare-name shorthand fills a named-only param, then a named arg fills it too."""
    params = [BindParam("x", STANDARD, False), BindParam("z", NAMED_ONLY, False)]
    with pytest.raises(ArgumentBindingError) as exc_info:
        _bind(params, [1, "z"], [("z", "also")])
    err = exc_info.value
    assert err.kind is ArgumentBindingErrorKind.DUPLICATE
    assert err.name == "z"
    assert err.named_index == 0
    assert err.positional_index is None


def test_duplicate_bare_shorthands_for_same_named_only() -> None:
    """Two bare-name shorthands targeting the same named-only slot → duplicate, with
    the index of the second (offending) positional arg."""
    params = [BindParam("z", NAMED_ONLY, False)]
    with pytest.raises(ArgumentBindingError) as exc_info:
        _bind(params, ["z", "z"], [])
    err = exc_info.value
    assert err.kind is ArgumentBindingErrorKind.DUPLICATE
    assert err.name == "z"
    assert err.positional_index == 1
    assert err.named_index is None


def test_missing_required_standard() -> None:
    params = [BindParam("x", STANDARD, False), BindParam("y", STANDARD, False)]
    with pytest.raises(ArgumentBindingError) as exc_info:
        _bind(params, [1], [])
    err = exc_info.value
    assert err.kind is ArgumentBindingErrorKind.MISSING_REQUIRED
    assert err.name == "y"
    assert err.positional_index is None
    assert err.named_index is None


def test_missing_required_named_only() -> None:
    params = [BindParam("z", NAMED_ONLY, False)]
    with pytest.raises(ArgumentBindingError) as exc_info:
        _bind(params, [], [])
    err = exc_info.value
    assert err.kind is ArgumentBindingErrorKind.MISSING_REQUIRED
    assert err.name == "z"


def test_missing_required_pos_only() -> None:
    params = [BindParam("x", POSITIONAL_ONLY, False)]
    with pytest.raises(ArgumentBindingError) as exc_info:
        _bind(params, [], [])
    err = exc_info.value
    assert err.kind is ArgumentBindingErrorKind.MISSING_REQUIRED
    assert err.name == "x"


# ---------------------------------------------------------------------------
# positional_field_names
# ---------------------------------------------------------------------------


def test_positional_field_names_named_only_between_standard_and_positional_only() -> None:
    """A named-only field between a standard field and a later positional-only
    field stays out; the standard field before it is still included."""
    fields = [
        ("a", STANDARD),
        ("tag", NAMED_ONLY),
        ("b", POSITIONAL_ONLY),
    ]
    assert positional_field_names(fields) == ("a", "b")


def test_positional_field_names_trailing_standard_excluded() -> None:
    """A standard field after the last positional-only field renders named, not
    positionally."""
    fields = [
        ("a", POSITIONAL_ONLY),
        ("b", STANDARD),
    ]
    assert positional_field_names(fields) == ("a",)


def test_positional_field_names_no_positional_only_is_empty() -> None:
    fields = [("a", STANDARD), ("b", NAMED_ONLY)]
    assert positional_field_names(fields) == ()


def test_positional_field_names_all_positional_only() -> None:
    fields = [("a", POSITIONAL_ONLY), ("b", POSITIONAL_ONLY)]
    assert positional_field_names(fields) == ("a", "b")
