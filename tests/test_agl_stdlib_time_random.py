"""Companion contracts for the ``std/time`` and ``std/random`` modules."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from threading import Barrier
from types import ModuleType, SimpleNamespace
from typing import Protocol, cast

import pytest

from agm.agl.eval.ir_interpreter import IrInterpreter
from agm.agl.ir.ids import NominalId
from agm.agl.ir.program import NominalDescriptor, NominalKind
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.runtime.boundary import AglArrayView, AglException
from agm.agl.runtime.externs import ExternRegistry
from agm.agl.semantics.values import ArrayValue, IntValue, TextValue
from tests._agl_helpers import option_nominal_descriptors
from tests.agl.ir_harness import extern_caps, lower_ir

_STDLIB_ROOT = Path(__file__).resolve().parents[1] / "stdlib"
_TIME_MODULE = ModuleId(("std", "time"))
_RANDOM_MODULE = ModuleId(("std", "random"))
_TIME_PARSE_ERROR = NominalId(9_600_001)
_INDEX_ERROR = NominalId(9_600_002)
_OPTION = NominalId(9_600_003)
_OPTION_NONE = NominalId(9_600_004)
_OPTION_SOME = NominalId(9_600_005)


class _TimeCompanion(Protocol):
    def parse_iso(self, raw: str) -> Decimal: ...

    def format_iso(self, epoch: Decimal) -> str: ...

    def parse(self, raw: str, fmt: str) -> Decimal: ...

    def format(self, epoch: Decimal, fmt: str) -> str: ...


class _RandomCompanion(Protocol):
    def seed(self, value: int) -> None: ...

    def below(self, upper: int) -> int: ...

    def between(self, lower: int, upper: int) -> int: ...

    def uniform(self) -> Decimal: ...

    def choice(self, values: object) -> object: ...

    def shuffle_in_place(self, values: object) -> None: ...


class _StateCompanion(Protocol):
    barrier: Barrier | None
    results: dict[int, tuple[int, int]]


def _companion(
    module_id: ModuleId, name: str, registry: ExternRegistry | None = None
) -> ModuleType:
    active_registry = registry if registry is not None else ExternRegistry()
    active_registry.set_nominals(
        {
            _TIME_PARSE_ERROR: NominalDescriptor(
                nominal=_TIME_PARSE_ERROR,
                module_id=_TIME_MODULE,
                scope_path=(),
                declared_name="TimeParseError",
                kind=NominalKind.EXCEPTION,
                fields=("message", "raw"),
            ),
            _INDEX_ERROR: NominalDescriptor(
                nominal=_INDEX_ERROR,
                module_id=ModuleId(("std", "errors")),
                scope_path=(),
                declared_name="IndexError",
                kind=NominalKind.EXCEPTION,
                fields=("message", "index", "length"),
            ),
            **option_nominal_descriptors(_OPTION, _OPTION_NONE, _OPTION_SOME),
        }
    )
    return active_registry.load_companion(module_id, _STDLIB_ROOT / "std" / f"{name}.py")


def test_time_round_trips_fixed_utc_iso_and_strptime_values() -> None:
    companion = cast(_TimeCompanion, _companion(_TIME_MODULE, "time"))
    epoch = Decimal("1704164645.123456")

    assert companion.format_iso(epoch) == "2024-01-02T03:04:05.123456+00:00"
    assert companion.parse_iso("2024-01-02T03:04:05.123456Z") == epoch
    assert companion.parse_iso("2024-01-02T04:34:05.123456+01:30") == epoch
    assert companion.parse_iso("2024-01-02T03:04:05.123456") == epoch
    assert companion.format(epoch, "%Y/%m/%d %H:%M:%S.%f %z") == "2024/01/02 03:04:05.123456 +0000"
    assert companion.parse("2024/01/02 03:04:05.123456 +0000", "%Y/%m/%d %H:%M:%S.%f %z") == epoch
    assert companion.parse("2024/01/02 04:34:05.123456 +0130", "%Y/%m/%d %H:%M:%S.%f %z") == epoch
    assert companion.parse("2024/01/02 03:04:05.123456", "%Y/%m/%d %H:%M:%S.%f") == epoch


def test_time_parse_errors_are_typed_and_retain_the_raw_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    companion = cast(_TimeCompanion, _companion(_TIME_MODULE, "time"))

    for parse, args in ((companion.parse_iso, ("bad",)), (companion.parse, ("bad", "%Y-%m-%d"))):
        with pytest.raises(AglException) as exc_info:
            parse(*args)
        assert exc_info.value.value.nominal == _TIME_PARSE_ERROR
        assert exc_info.value.value.fields["raw"] == TextValue("bad")

    def overflow(*_: object) -> object:
        raise OverflowError

    for name, args in (("parse_iso", ("out of range",)), ("parse", ("out of range", "%Y-%m-%d"))):
        datetime_method = "fromisoformat" if name == "parse_iso" else "strptime"
        monkeypatch.setattr(companion, "datetime", SimpleNamespace(**{datetime_method: overflow}))
        with pytest.raises(AglException) as exc_info:
            getattr(companion, name)(*args)
        assert exc_info.value.value.nominal == _TIME_PARSE_ERROR
        assert exc_info.value.value.fields["raw"] == TextValue("out of range")


def test_random_seed_reproduces_bounded_sequences_and_shuffle_preserves_a_live_multiset() -> None:
    companion = cast(_RandomCompanion, _companion(_RANDOM_MODULE, "random"))
    values = ArrayValue([IntValue(1), IntValue(2), IntValue(2), IntValue(3), IntValue(4)])
    view = AglArrayView(values)
    companion.seed(21)
    sequence = (companion.below(100), companion.between(-10, 10), companion.uniform())
    companion.seed(21)
    assert (companion.below(100), companion.between(-10, 10), companion.uniform()) == sequence
    assert 0 <= sequence[0] < 100
    assert -10 <= sequence[1] <= 10
    assert Decimal(0) <= sequence[2] < Decimal(1)

    companion.seed(21)
    companion.shuffle_in_place(view)
    assert sorted(value.value for value in values.elements) == [1, 2, 2, 3, 4]


def test_random_choice_raises_typed_index_error_for_an_empty_array() -> None:
    companion = cast(_RandomCompanion, _companion(_RANDOM_MODULE, "random"))

    with pytest.raises(AglException) as exc_info:
        companion.choice([])

    assert exc_info.value.value.nominal == _INDEX_ERROR
    assert exc_info.value.value.fields["index"] == IntValue(0)
    assert exc_info.value.value.fields["length"] == IntValue(0)


def test_random_choice_returns_an_array_element() -> None:
    companion = cast(_RandomCompanion, _companion(_RANDOM_MODULE, "random"))
    companion.seed(9)

    assert companion.choice(["one", "two", "three"]) in {"one", "two", "three"}


def test_random_state_is_isolated_between_concurrent_real_interpreters(tmp_path: Path) -> None:
    entry_path = tmp_path / "entry.agl"
    entry_path.write_text(
        "import std/random\n"
        "extern def checkpoint() -> unit\n"
        "extern def report(seed: int, first: int, second: int) -> unit\n"
        "program def main(run_seed: int) -> unit =\n"
        "  random::seed(run_seed)\n"
        "  checkpoint()\n"
        "  let first = random::below(1000000)\n"
        "  let second = random::below(1000000)\n"
        "  report(run_seed, first, second)\n"
    )
    entry_path.with_suffix(".py").write_text(
        "from threading import Barrier\n"
        "barrier = None\n"
        "results = {}\n"
        "def checkpoint():\n"
        "    if barrier is not None:\n"
        "        barrier.wait(timeout=5)\n"
        "def report(seed, first, second):\n"
        "    results[seed] = (first, second)\n"
    )
    executable = lower_ir(entry_path.read_text(), caps=extern_caps(), origin_path=entry_path)
    registry = ExternRegistry()
    registry.set_nominals(executable.nominals)
    registry.load_companion(_RANDOM_MODULE, _STDLIB_ROOT / "std" / "random.py")
    state_companion = cast(
        _StateCompanion, registry.load_companion(ENTRY_ID, entry_path.with_suffix(".py"))
    )
    program_symbol = next(iter(executable.program_functions))

    def run(seed: int) -> tuple[int, int]:
        IrInterpreter(executable, extern_registry=registry).run(
            program_symbol=program_symbol, arguments=(IntValue(seed),)
        )
        return state_companion.results.pop(seed)

    expected = {seed: run(seed) for seed in (713, 91)}
    state_companion.barrier = Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        actual = dict(zip((713, 91), executor.map(run, (713, 91))))

    assert actual == expected
    assert all(0 <= value < 1_000_000 for sequence in actual.values() for value in sequence)
