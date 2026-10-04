from decimal import Decimal

import agl
from agl import Color, Point, Problem


def probe(x: object) -> str:
    keys = list(x)
    first = keys[0]
    return f"{len(x)} {len(keys)} {first in x} {x[first]}"


def ident(x: object) -> object:
    return x


def mutate_generic(x: object) -> None:
    first = next(iter(x))
    x[first] = x[first]
    del x[first]
    if isinstance(first, int):
        x[10] = "z"
    else:
        x[Point(x=9, y=9)] = 9


def mutate_ints(d: dict[int, str]) -> None:
    d[3] = "c"
    del d[1]
    assert 2 in d and 1 not in d and 99 not in d
    assert list(d) == [2, 3]


def mutate_points(d: dict[Point, int]) -> None:
    d[Point(x=3, y=4)] = 20
    d[Point(x=5, y=6)] = 30
    del d[Point(x=1, y=2)]
    assert Point(x=1, y=2) not in d


def mutate_colors(d: dict[Color, int]) -> None:
    d[Color.Blue()] = 2
    assert Color.Red() in d


def probe_json(d: dict[object, int]) -> str:
    return f"{d[agl.json(True)]} {d[agl.json(1)]} {agl.json('a') in d}"


def mutate_problems(d: dict[Problem, int]) -> None:
    assert Problem(message="m", code=1) in d and Problem(message="m", code=2) not in d
    assert [k.code for k in d] == [1]
    d[Problem(message="m", code=2)] = 2
    d[Problem(message="m", code=1)] = 10


def fill_json(d: dict[object, int]) -> None:
    d[agl.json("a")] = 1


def make_problems() -> dict[Problem, int]:
    return agl.dict({Problem(message="m", code=7): 3})


def make_ints() -> dict[int, str]:
    return agl.dict({1: "a", 2: "b"})


def make_points() -> dict[Point, int]:
    return agl.dict({Point(x=1, y=2): 5})


def make_json() -> dict[object, int]:
    return agl.dict({agl.json("k"): 1, agl.json([1, 2]): 2})


def make_decimals() -> dict[Decimal, str]:
    return agl.dict({Decimal("1.50"): "x"})
