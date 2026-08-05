from agl import Counter


def increment(counter: Counter, amount: int) -> int:
    """Receive the AgL Counter receiver before the declared method argument."""
    value = counter.value
    assert isinstance(value, int)
    return value + amount
