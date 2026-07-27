def count(xs: list[object]) -> int:
    """Count the elements of `xs` (never reached for the cyclic argument)."""
    return len(xs)


def identity(x: object) -> object:
    """Return `x` unchanged, but repr it first (forces the sealed handle to
    render its wrapped AgL value, which detects a reference cycle)."""
    repr(x)
    return x
