def increment(counter: dict[str, object], amount: int) -> int:
    """Receive the AgL Counter receiver before the declared method argument."""
    value = counter["value"]
    assert isinstance(value, int)
    return value + amount
