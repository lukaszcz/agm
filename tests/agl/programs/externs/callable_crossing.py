from collections.abc import Callable

_stashed: Callable[[int], int] | None = None


def apply(f: Callable[[int], int], value: int) -> int:
    return f(value)


def combine(f: Callable[[int, int], int], left: int, right: int) -> int:
    return f(left, right)


def twice(f: Callable[[object], object], value: object) -> object:
    return f(f(value))


def stash(f: Callable[[int], int]) -> None:
    global _stashed
    _stashed = f


def use_stashed(value: int) -> int:
    assert _stashed is not None
    return _stashed(value)
