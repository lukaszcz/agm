"""Visibility and decimal-context contracts for the ``std/math`` module."""

from __future__ import annotations

import decimal
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.modules.ids import ModuleId
from agm.agl.runtime.externs import ExternRegistry
from agm.agl.scope import AglScopeError
from agm.agl.typecheck import AglTypeError
from tests.agl.module_graph import resolve_and_check_inline_entry

_STDLIB_ROOT = Path(__file__).resolve().parents[1] / "packages" / "stdlib"
_MATH_MODULE = ModuleId(("std", "math"))


class _MathCompanion(Protocol):
    def ceil(self, value: decimal.Decimal) -> int: ...

    def floor(self, value: decimal.Decimal) -> int: ...

    def pow(self, value: decimal.Decimal, exponent: int) -> decimal.Decimal: ...

    def round(self, value: decimal.Decimal, digits: int) -> decimal.Decimal: ...

    def sqrt(self, value: decimal.Decimal) -> decimal.Decimal: ...


def _math_companion() -> _MathCompanion:
    registry = ExternRegistry()
    module: ModuleType = registry.load_companion(_MATH_MODULE, _STDLIB_ROOT / "src" / "math.py")
    return cast(_MathCompanion, module)


def test_scalar_methods_are_ambient_but_free_functions_require_an_import() -> None:
    resolve_and_check_inline_entry(
        "let _: int = (-3).abs()\nlet _: decimal = 1.25.round(1)\n", HostCapabilities()
    )

    with pytest.raises(AglScopeError):
        resolve_and_check_inline_entry("sum([1, 2])\n", HostCapabilities())

    resolve_and_check_inline_entry(
        "import std/math\nlet _: int = math::sum([1, 2])\nlet _: decimal = math::pi + math::e\n",
        HostCapabilities(),
    )


def test_scalar_method_types_keep_numeric_families_separate() -> None:
    resolve_and_check_inline_entry(
        "import std/array\n"
        "let _: array[int] = [2, 1].sort(fn(left: int, right: int) => left.compare(right))\n"
        "let _: array[decimal] = [2.0, 1.0].sort(\n"
        "  fn(left: decimal, right: decimal) => left.compare(right)\n)\n",
        HostCapabilities(),
    )

    with pytest.raises(AglTypeError):
        resolve_and_check_inline_entry("1.to-decimal().pow(2.0)\n", HostCapabilities())


def test_decimal_companion_uses_agl_context_for_rounding_and_roots() -> None:
    companion = _math_companion()

    with decimal.localcontext() as context:
        context.prec = 3
        context.rounding = decimal.ROUND_UP
        assert companion.floor(decimal.Decimal("-3.1")) == -4
        assert companion.ceil(decimal.Decimal("-3.1")) == -3
        assert companion.round(decimal.Decimal("2.345"), 2) == decimal.Decimal("2.34")
        assert companion.sqrt(decimal.Decimal(2)) == decimal.Decimal(
            "1.414213562373095048801688724"
        )
        assert companion.pow(decimal.Decimal("1.25"), 3) == decimal.Decimal("1.953125")
