"""Decimal operations for ``std/math``."""

from __future__ import annotations

import decimal
import functools
from collections.abc import Callable

from agm.agl.eval._decimal import AGL_DECIMAL_CONTEXT


def _in_agl_context[**P, R](operation: Callable[P, R]) -> Callable[P, R]:
    """Run *operation* under AgL's decimal context: its precision, traps, and rounding."""

    @functools.wraps(operation)
    def run(*args: P.args, **kwargs: P.kwargs) -> R:
        with decimal.localcontext(AGL_DECIMAL_CONTEXT):
            return operation(*args, **kwargs)

    return run


def int_pow(value: int, exponent: int) -> int:
    return value**exponent


@_in_agl_context
def _to_integral(value: decimal.Decimal, rounding: str) -> int:
    return int(value.to_integral_value(rounding=rounding))


def floor(value: decimal.Decimal) -> int:
    return _to_integral(value, decimal.ROUND_FLOOR)


def ceil(value: decimal.Decimal) -> int:
    return _to_integral(value, decimal.ROUND_CEILING)


@_in_agl_context
def round(value: decimal.Decimal, digits: int) -> decimal.Decimal:
    return value.quantize(decimal.Decimal(1).scaleb(-digits))


@_in_agl_context
def sqrt(value: decimal.Decimal) -> decimal.Decimal:
    return value.sqrt()


@_in_agl_context
def pow(value: decimal.Decimal, exponent: int) -> decimal.Decimal:
    return value**exponent


__all__ = ["ceil", "floor", "int_pow", "pow", "round", "sqrt"]
