"""Decimal operations for ``std/math``."""

from __future__ import annotations

import decimal

from agm.agl.eval._decimal import AGL_DECIMAL_CONTEXT


def int_pow(value: int, exponent: int) -> int:
    return value**exponent


def floor(value: decimal.Decimal) -> int:
    with decimal.localcontext(AGL_DECIMAL_CONTEXT):
        return int(value.to_integral_value(rounding=decimal.ROUND_FLOOR))


def ceil(value: decimal.Decimal) -> int:
    with decimal.localcontext(AGL_DECIMAL_CONTEXT):
        return int(value.to_integral_value(rounding=decimal.ROUND_CEILING))


def round(value: decimal.Decimal, digits: int) -> decimal.Decimal:
    with decimal.localcontext(AGL_DECIMAL_CONTEXT):
        return value.quantize(decimal.Decimal(1).scaleb(-digits))


def sqrt(value: decimal.Decimal) -> decimal.Decimal:
    with decimal.localcontext(AGL_DECIMAL_CONTEXT):
        return value.sqrt()


def pow(value: decimal.Decimal, exponent: int) -> decimal.Decimal:
    with decimal.localcontext(AGL_DECIMAL_CONTEXT):
        return value**exponent


__all__ = ["ceil", "floor", "int_pow", "pow", "round", "sqrt"]
