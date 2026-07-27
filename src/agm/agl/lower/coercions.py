"""Coercion compiler for the AgL lowering phase.

``compile_coercion(source, target)`` is the ONLY place that reads checker
``Type`` objects to produce a ``Coercion`` descriptor.  Once this function
returns, the coercion is fully pre-resolved; the evaluator only switches on
the returned ``Coercion`` union and never sniffs value types at runtime.

An implicit coercion never rebuilds a data structure — array, dict, record,
and enum values are never recursed into. The whole rule:

  1. equal types → None (identity)
  2. target is decimal, source is int → IntToDecimal (widen)
  3. target is json, source is a scalar JSON-shaped type
     (``text``/``json``/``bool``/``int``/``decimal``) → ToJson
  4. otherwise → None (no implicit coercion; converting an array, dict,
     record, enum, or exception into json requires an explicit ``as json``
     cast)
"""

from __future__ import annotations

from agm.agl.ir.operations import Coercion, IntToDecimal, ToJson
from agm.agl.semantics.types import DecimalType, IntType, JsonType, Type, is_scalar_json_shaped

__all__ = ["compile_coercion"]


def compile_coercion(source: Type, target: Type) -> Coercion | None:
    """Compile an implicit coercion from *source* to *target*.

    Returns a ``Coercion`` descriptor to be wrapped in an ``IrCoerce`` node, or
    ``None`` when no coercion node is needed (identity, or no implicit
    coercion exists for this pair).
    """
    if source == target:
        return None
    if isinstance(target, DecimalType) and isinstance(source, IntType):
        return IntToDecimal()
    if isinstance(target, JsonType) and is_scalar_json_shaped(source):
        return ToJson()
    return None
