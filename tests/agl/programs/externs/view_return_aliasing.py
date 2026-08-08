from collections.abc import MutableSequence

from agl import array


def _require_view(xs: MutableSequence[int]) -> None:
    assert not isinstance(xs, list)
    assert isinstance(xs, MutableSequence)


def return_received(xs: MutableSequence[int]) -> MutableSequence[int]:
    _require_view(xs)
    return xs


def return_snapshot(xs: MutableSequence[int]) -> MutableSequence[int]:
    _require_view(xs)
    return array(list(xs))
