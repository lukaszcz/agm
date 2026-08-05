from collections.abc import MutableSequence


def reject_other_type_variable(xs: MutableSequence[object], other: MutableSequence[object]) -> None:
    xs[0] = other[0]


def mutate(xs: MutableSequence[object], operation: str) -> None:
    if operation == "reverse":
        xs.reverse()
    elif operation == "rotate":
        last = xs[-1]
        for index in range(len(xs) - 1, 0, -1):
            xs[index] = xs[index - 1]
        xs[0] = last
    elif operation == "duplicate":
        xs[0] = xs[-1]
    elif operation == "reject":
        xs[0] = "replacement"
    elif operation == "sort":
        xs.sort()
