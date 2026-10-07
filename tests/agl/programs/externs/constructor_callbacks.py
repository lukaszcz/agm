from collections.abc import Callable, Iterable

from agl import array


def apply_each(f: Callable[[object], object], values: Iterable[object]) -> object:
    return array([f(value) for value in values])
