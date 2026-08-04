from collections.abc import MutableSequence

_stashed: MutableSequence[int] | None = None


class StaleViewError(RuntimeError):
    pass


def stash(xs: MutableSequence[int]) -> None:
    global _stashed
    _stashed = xs


def mutate_stashed() -> None:
    assert _stashed is not None
    try:
        _stashed.append(9)
    except RuntimeError as error:
        raise StaleViewError() from error
    raise AssertionError("stashed view remained usable")


def stash_and_use(xs: MutableSequence[int]) -> None:
    global _stashed
    _stashed = xs
    _stashed.append(7)
