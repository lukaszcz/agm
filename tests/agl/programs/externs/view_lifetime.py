from collections.abc import MutableSequence

_stashed: MutableSequence[int] | None = None


def stash(xs: MutableSequence[int]) -> None:
    global _stashed
    _stashed = xs


def mutate_stashed() -> None:
    assert _stashed is not None
    _stashed.append(9)


def stash_and_use(xs: MutableSequence[int]) -> None:
    global _stashed
    _stashed = xs
    _stashed.append(7)
