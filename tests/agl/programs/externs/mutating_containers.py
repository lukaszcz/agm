from collections.abc import MutableMapping, MutableSequence

from agl import Batch, array
from agl import dict as agl_dict


def mutate_array(xs: MutableSequence[int], operation: str) -> None:
    if operation == "append":
        xs.append(4)
    elif operation == "insert":
        xs.insert(1, 0)
    elif operation == "sort":
        xs.sort()
    elif operation == "clear":
        xs.clear()


def mutate_dict(values: dict[str, int], operation: str) -> None:
    if operation == "update":
        values.update({"two": 2})
    elif operation == "delete":
        del values["one"]


def mutate_nested(batch: Batch, xs: MutableSequence[int]) -> None:
    assert batch.values == xs
    xs.append(3)


def mutate_aliases(a: MutableSequence[int], b: MutableSequence[int]) -> None:
    assert a == b
    a.append(3)


def mutate_copy(xs: MutableSequence[int]) -> None:
    assert not isinstance(xs, list)
    xs.append(3)


def snapshot_not_list(xs: MutableSequence[int]) -> list[int]:
    assert not isinstance(xs, list)
    assert isinstance(xs, MutableSequence)
    return array(list(xs))


def snapshot_not_dict(d: MutableMapping[str, int]) -> dict[str, int]:
    assert isinstance(d, MutableMapping)
    assert type(d) is not dict
    return agl_dict({key: value for key, value in d.items()})
