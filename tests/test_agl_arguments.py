"""Unit tests for the shared ``bind_arguments`` routine in typecheck.arguments.

These tests drive ``bind_arguments`` directly with a minimal item type,
covering every error branch and the main success paths.  They are independent
of the checker and lowerer.
"""

from __future__ import annotations

import pytest

from agm.agl.syntax.nodes import ParamKind
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck.arguments import BindParam, BoundName, bind_arguments
from agm.agl.typecheck.env import AglTypeError

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

POSITIONAL_ONLY = ParamKind.POSITIONAL_ONLY
STANDARD = ParamKind.STANDARD
NAMED_ONLY = ParamKind.NAMED_ONLY

_SPAN = SourceSpan(1, 1, 1, 10, 0, 10)
_CALL_SPAN = SourceSpan(2, 1, 2, 20, 11, 30)


class _Item:
    """Minimal item type for testing: either a bare-name item or an opaque item."""

    def __init__(self, name: str | None, label: str = "") -> None:
        # name: if set, this is a "bare name" item (VarRef equivalent)
        # label: display string for repr
        self.name = name
        self.label = label or (name if name else "<expr>")

    def __repr__(self) -> str:
        return f"_Item({self.label!r})"


def _item(label: str) -> _Item:
    """Create a non-bare (opaque) item."""
    return _Item(name=None, label=label)


def _bare(name: str) -> _Item:
    """Create a bare-name item (equivalent to VarRef)."""
    return _Item(name=name)


def _bare_name(item: _Item) -> str | None:
    return item.name


def _span_of(_item: _Item) -> SourceSpan:
    return _SPAN


def _bind(
    params: list[BindParam],
    positional: list[_Item],
    named: list[tuple[str, _Item]],
    context_desc: str = "call to 'f'",
) -> tuple[_Item | None, ...]:
    """Helper to call bind_arguments with a simple named arg format."""
    bound_named = [BoundName(name=n, value=v, span=_SPAN) for n, v in named]
    return bind_arguments(
        params,
        positional,
        bound_named,
        bare_name=_bare_name,
        span_of=_span_of,
        call_span=_CALL_SPAN,
        context_desc=context_desc,
    )


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_all_positional_standard() -> None:
    """All-standard params, all positional args."""
    params = [
        BindParam("x", STANDARD, False),
        BindParam("y", STANDARD, False),
    ]
    a, b = _bind(params, [_item("1"), _item("2")], [])
    assert a is not None and a.label == "1"
    assert b is not None and b.label == "2"


def test_named_arg_for_standard() -> None:
    """Named arg fills a standard param."""
    params = [
        BindParam("x", STANDARD, False),
        BindParam("y", STANDARD, False),
    ]
    pos = _item("v_x")
    named_y = _item("v_y")
    result = _bind(params, [pos], [("y", named_y)])
    assert result[0] is pos
    assert result[1] is named_y


def test_defaults_fill_missing() -> None:
    """Params with defaults yield None when not supplied."""
    params = [
        BindParam("x", STANDARD, False),
        BindParam("y", STANDARD, True),
        BindParam("z", STANDARD, True),
    ]
    result = _bind(params, [_item("a")], [])
    assert result[0] is not None
    assert result[1] is None
    assert result[2] is None


def test_positional_only_filled_positionally() -> None:
    """Positional-only param filled by a positional arg."""
    params = [
        BindParam("x", POSITIONAL_ONLY, False),
        BindParam("y", STANDARD, False),
    ]
    x_val = _item("42")
    y_val = _item("7")
    result = _bind(params, [x_val, y_val], [])
    assert result[0] is x_val
    assert result[1] is y_val


def test_named_only_via_named_arg() -> None:
    """Named-only param filled by a named arg."""
    params = [
        BindParam("x", STANDARD, False),
        BindParam("z", NAMED_ONLY, False),
    ]
    x_val = _item("1")
    z_val = _item("99")
    result = _bind(params, [x_val], [("z", z_val)])
    assert result[0] is x_val
    assert result[1] is z_val


def test_named_only_shorthand_bare_positional() -> None:
    """Bare-name positional in named-only territory → shorthand z=z."""
    params = [
        BindParam("x", STANDARD, False),
        BindParam("z", NAMED_ONLY, False),
    ]
    x_val = _item("1")
    z_val = _bare("z")  # bare name = shorthand
    result = _bind(params, [x_val, z_val], [])
    assert result[0] is x_val
    assert result[1] is z_val  # the bare item itself, not a wrapper


def test_mixed_zones_all_positional() -> None:
    """pos-only + standard + named-only, fill pos-only and std positionally."""
    params = [
        BindParam("a", POSITIONAL_ONLY, False),
        BindParam("b", STANDARD, False),
        BindParam("c", NAMED_ONLY, True),
    ]
    a_val = _item("10")
    b_val = _item("20")
    result = _bind(params, [a_val, b_val], [])
    assert result[0] is a_val
    assert result[1] is b_val
    assert result[2] is None  # default


def test_all_named_only_via_named_args() -> None:
    """All named-only params, supplied as named args."""
    params = [
        BindParam("x", NAMED_ONLY, False),
        BindParam("y", NAMED_ONLY, False),
    ]
    xv = _item("a")
    yv = _item("b")
    result = _bind(params, [], [("x", xv), ("y", yv)])
    assert result[0] is xv
    assert result[1] is yv


def test_named_only_default_unfilled() -> None:
    """Named-only param with default: not supplied → None (use default)."""
    params = [
        BindParam("x", NAMED_ONLY, True),
        BindParam("y", NAMED_ONLY, False),
    ]
    yv = _item("hello")
    result = _bind(params, [], [("y", yv)])
    assert result[0] is None
    assert result[1] is yv


def test_generic_call_no_args_all_defaults() -> None:
    """All params have defaults; no args → all None."""
    params = [
        BindParam("a", STANDARD, True),
        BindParam("b", NAMED_ONLY, True),
    ]
    result = _bind(params, [], [])
    assert all(v is None for v in result)


def test_empty_params_empty_args() -> None:
    """Zero-param, zero-arg call is valid."""
    result = _bind([], [], [])
    assert result == ()


# ---------------------------------------------------------------------------
# Error-branch tests
# ---------------------------------------------------------------------------


def test_too_many_positional_all_standard() -> None:
    """Opaque extra positional arg past all-standard params (no named-only) → too-many error."""
    params = [BindParam("x", STANDARD, False)]
    with pytest.raises(AglTypeError, match="[Tt]oo many positional"):
        _bind(params, [_item("1"), _item("2")], [])


def test_too_many_positional_with_named_only() -> None:
    """Two positional args but only one pos-capable param + one named-only → error for second."""
    params = [
        BindParam("x", STANDARD, False),
        BindParam("z", NAMED_ONLY, False),
    ]
    # Second positional is opaque (not bare) → "positional in named-only position"
    with pytest.raises(AglTypeError, match="named-only position"):
        _bind(params, [_item("1"), _item("2+3")], [])


def test_too_many_positional_all_named_only() -> None:
    """No pos-capable params but there IS a named-only: opaque positional → named-only-position."""
    params = [BindParam("x", NAMED_ONLY, False)]
    with pytest.raises(AglTypeError, match="named-only position"):
        _bind(params, [_item("42")], [])


def test_too_many_positional_no_named_only_zero_params() -> None:
    """No params at all; opaque positional → too-many-positional error (no named-only context)."""
    with pytest.raises(AglTypeError, match="[Tt]oo many positional"):
        _bind([], [_item("1")], [])


def test_pos_only_by_name_rejected() -> None:
    """Named arg targeting a positional-only param is an error."""
    params = [
        BindParam("x", POSITIONAL_ONLY, False),
        BindParam("y", STANDARD, False),
    ]
    with pytest.raises(AglTypeError, match="positional-only"):
        _bind(params, [_item("1")], [("x", _item("99"))])


def test_unknown_named_arg() -> None:
    """Named arg that doesn't match any param name → error."""
    params = [BindParam("x", STANDARD, False)]
    with pytest.raises(AglTypeError, match="Unknown"):
        _bind(params, [_item("1")], [("oops", _item("2"))])


def test_duplicate_positional_and_named() -> None:
    """Param filled positionally then again by name → duplicate error."""
    params = [
        BindParam("x", STANDARD, False),
        BindParam("y", STANDARD, True),
    ]
    with pytest.raises(AglTypeError, match="Duplicate"):
        _bind(params, [_item("1")], [("x", _item("again"))])


def test_duplicate_shorthand_and_named() -> None:
    """Bare-name shorthand fills named-only param, then named arg also fills it → duplicate."""
    params = [
        BindParam("x", STANDARD, False),
        BindParam("z", NAMED_ONLY, False),
    ]
    with pytest.raises(AglTypeError, match="Duplicate"):
        _bind(params, [_item("1"), _bare("z")], [("z", _item("also"))])


def test_duplicate_bare_shorthands_for_same_named_only() -> None:
    """Two bare-name shorthands targeting the same named-only slot → duplicate."""
    params = [BindParam("z", NAMED_ONLY, False)]
    # Both positional args are bare "z" shorthands trying to fill the same slot.
    with pytest.raises(AglTypeError, match="Duplicate"):
        _bind(params, [_bare("z"), _bare("z")], [])


def test_missing_required_standard() -> None:
    """Required standard param not supplied → missing-required error."""
    params = [
        BindParam("x", STANDARD, False),
        BindParam("y", STANDARD, False),
    ]
    with pytest.raises(AglTypeError, match="Missing"):
        _bind(params, [_item("1")], [])


def test_missing_required_named_only() -> None:
    """Required named-only param not supplied → missing error."""
    params = [BindParam("z", NAMED_ONLY, False)]
    with pytest.raises(AglTypeError, match="Missing"):
        _bind(params, [], [])


def test_missing_required_pos_only() -> None:
    """Required positional-only param not supplied → missing error."""
    params = [BindParam("x", POSITIONAL_ONLY, False)]
    with pytest.raises(AglTypeError, match="Missing"):
        _bind(params, [], [])


def test_bare_shorthand_unknown_named_only() -> None:
    """Bare name shorthand for a named-only slot but no param with that name → unknown error."""
    params = [
        BindParam("x", STANDARD, False),
        BindParam("z", NAMED_ONLY, False),
    ]
    # Positional-capable slots exhausted, then bare "w" lands in named-only territory
    # but there's no param named "w".
    with pytest.raises(AglTypeError, match="Unknown"):
        _bind(params, [_item("1"), _bare("w")], [])


def test_bare_shorthand_already_filled() -> None:
    """Bare name shorthand lands on a named-only param already filled by named arg → duplicate."""
    params = [
        BindParam("x", STANDARD, False),
        BindParam("z", NAMED_ONLY, False),
    ]
    # z is filled by named arg first (z=something), then bare "z" as positional
    # But note: positional args are processed BEFORE named args in bind_arguments.
    # So the bare "z" is processed first and fills z, then the named arg "z=..." is a duplicate.
    with pytest.raises(AglTypeError, match="Duplicate"):
        _bind(params, [_item("1"), _bare("z")], [("z", _item("also"))])
