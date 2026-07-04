"""The extern (Python FFI) boundary: sealed handles, value walkers, registry.

This is the eval-free runtime service backing ``extern def``:

- :class:`SealedHandle` — the opaque Python-side wrapper for an AgL value at
  a type-variable position crossing the boundary.
- :func:`encode_boundary_value` / :func:`decode_boundary_value` — the
  ``BoundarySchema``-driven walkers that convert a value in each direction,
  deep-copying so neither side can observe the other's mutations.
- :class:`ExternRegistry` — imports companion Python modules, resolves their
  callables, and is the single chokepoint that turns every runtime failure
  crossing the boundary (a raising callable, a return-contract violation, an
  argument-conversion failure) into a catchable ``ExternError``.

Companion import and callable resolution are two separate steps
(:meth:`ExternRegistry.load_companion` then :meth:`ExternRegistry.resolve`)
so a module's companion is imported exactly once even though it may declare
several externs.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import assert_never, cast

from agm.agl.diagnostics import AglError
from agm.agl.ir.contracts import (
    BoundaryDict,
    BoundaryEnum,
    BoundaryException,
    BoundaryList,
    BoundaryRecord,
    BoundaryScalar,
    BoundarySchema,
    BoundarySealVar,
    BoundaryUnit,
    ExternContract,
    ScalarKind,
)
from agm.agl.modules.ids import ModuleId
from agm.agl.runtime.render import render_value
from agm.agl.runtime.serialize import value_to_json_obj
from agm.agl.semantics.exceptions import AglRaise, make_builtin_exception
from agm.agl.semantics.values import (
    BoolValue,
    DecimalValue,
    DictValue,
    EnumValue,
    ExceptionValue,
    IntValue,
    JsonValue,
    ListValue,
    RecordValue,
    TextValue,
    UnitValue,
    Value,
)

# ---------------------------------------------------------------------------
# Internal exceptions
# ---------------------------------------------------------------------------


class BoundaryViolation(Exception):
    """Internal signal for a contract violation while crossing the boundary.

    Raised by :func:`encode_boundary_value` / :func:`decode_boundary_value` on
    any structural mismatch (wrong type, wrong variant, missing/extra/
    misnamed fields, a missing or mismatched seal, ...).  Caught by
    :meth:`ExternRegistry.invoke`, which reports it as ``ExternError`` with an
    empty ``python_type``.
    """


class ExternImportError(AglError):
    """A companion module raised while its top-level code was executed.

    Raised by :meth:`ExternRegistry.load_companion`; the pipeline converts it
    into a load-time diagnostic naming the AgL module.
    """

    def __init__(self, module_id: ModuleId, message: str) -> None:
        super().__init__(f"module {module_id.dotted()!r}: {message}")
        self.module_id = module_id


class ExternResolutionError(AglError):
    """A companion has no callable attribute matching a declared extern name.

    Raised by :meth:`ExternRegistry.resolve`; the pipeline converts it into a
    load-time diagnostic naming the AgL module and the extern function.
    """

    def __init__(self, module_id: ModuleId, name: str) -> None:
        super().__init__(
            f"module {module_id.dotted()!r} extern {name!r}: companion has no "
            "callable attribute of that name"
        )
        self.module_id = module_id
        self.name = name


# ---------------------------------------------------------------------------
# SealedHandle
# ---------------------------------------------------------------------------


class SealedHandle:
    """Opaque wrapper for an AgL value at a sealed type-variable position.

    A Python companion may rearrange, count, and compare handles it receives,
    but cannot inspect or forge them: the wrapped value and seal token are
    private, and there is no other public surface.

    ``__eq__``/``__hash__`` delegate to the wrapped value's own equality and
    hash (never equal to a non-handle), so handles compose correctly in
    Python sets and dicts.  ``__repr__`` shows the rendered AgL value as a
    debugging aid.
    """

    __slots__ = ("_seal", "_value")

    def __init__(self, value: Value, seal: object) -> None:
        self._value = value
        self._seal = seal

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SealedHandle):
            return False
        return self._value == other._value

    def __hash__(self) -> int:
        return hash(self._value)

    def __repr__(self) -> str:
        return render_value(self._value)


def _pytype(obj: object) -> str:
    """Human-readable Python type name for a boundary-violation message."""
    return type(obj).__name__


# ---------------------------------------------------------------------------
# Encode: AgL Value -> Python argument
# ---------------------------------------------------------------------------


def encode_boundary_value(
    schema: BoundarySchema, value: Value, seals: Mapping[str, object]
) -> object:
    """Encode one AgL value crossing an extern boundary as a Python argument.

    Walks *schema* (compiled from the extern's declared parameter/result
    type) and *value* in lockstep, producing a fresh Python object per the
    AgL-to-Python type mapping.  Containers and nominal values are rebuilt
    from scratch and ``json`` leaves are deep-copied, so mutating what Python
    receives never affects the wrapped AgL value.  A type-variable leaf seals
    the value in a :class:`SealedHandle` carrying this call's token for that
    variable (from *seals*, minted once per :meth:`ExternRegistry.invoke`
    call).  A *value* whose runtime shape does not match *schema* raises
    :class:`BoundaryViolation` (an argument-conversion failure at the call
    site, reported by :meth:`ExternRegistry.invoke` as ``ExternError``).
    """
    match schema:
        case BoundaryScalar(kind=kind):
            return _encode_scalar(kind, value)
        case BoundaryUnit():
            if not isinstance(value, UnitValue):
                raise BoundaryViolation(f"expected unit, got {_agltype(value)}")
            return None
        case BoundaryList(element=elem_schema):
            if not isinstance(value, ListValue):
                raise BoundaryViolation(f"expected a list value, got {_agltype(value)}")
            return [encode_boundary_value(elem_schema, e, seals) for e in value.elements]
        case BoundaryDict(value=val_schema):
            if not isinstance(value, DictValue):
                raise BoundaryViolation(f"expected a dict value, got {_agltype(value)}")
            return {
                k: encode_boundary_value(val_schema, v, seals)
                for k, v in value.entries.items()
            }
        case BoundaryRecord(display_name=display_name, fields=fields):
            if not isinstance(value, RecordValue):
                raise BoundaryViolation(
                    f"expected record {display_name!r}, got {_agltype(value)}"
                )
            return {
                fname: encode_boundary_value(fschema, value.fields[fname], seals)
                for fname, fschema in fields
            }
        case BoundaryEnum(display_name=display_name, variants=variants):
            if not isinstance(value, EnumValue):
                raise BoundaryViolation(f"expected enum {display_name!r}, got {_agltype(value)}")
            variant = next((v for v in variants if v.name == value.variant), None)
            if variant is None:
                raise BoundaryViolation(
                    f"enum {display_name!r}: unknown variant {value.variant!r}"
                )
            result: dict[str, object] = {"$case": value.variant}
            result.update(
                {
                    fname: encode_boundary_value(fschema, value.fields[fname], seals)
                    for fname, fschema in variant.fields
                }
            )
            return result
        case BoundaryException(display_name=display_name, fields=fields):
            if not isinstance(value, ExceptionValue):
                raise BoundaryViolation(
                    f"expected exception {display_name!r}, got {_agltype(value)}"
                )
            return {
                fname: encode_boundary_value(fschema, value.fields[fname], seals)
                for fname, fschema in fields
            }
        case BoundarySealVar(var=var):
            return SealedHandle(value, seals[var])
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _agltype(value: Value) -> str:
    """Human-readable AgL value-kind name for a boundary-violation message."""
    return type(value).__name__


def _encode_scalar(kind: ScalarKind, value: Value) -> object:
    """Encode one scalar (or opaque json) leaf as its Python argument."""
    match kind:
        case ScalarKind.TEXT:
            if not isinstance(value, TextValue):
                raise BoundaryViolation(f"expected text, got {_agltype(value)}")
            return value.value
        case ScalarKind.INT:
            if not isinstance(value, IntValue):
                raise BoundaryViolation(f"expected int, got {_agltype(value)}")
            return value.value
        case ScalarKind.DECIMAL:
            if not isinstance(value, DecimalValue):
                raise BoundaryViolation(f"expected decimal, got {_agltype(value)}")
            return value.value
        case ScalarKind.BOOL:
            if not isinstance(value, BoolValue):
                raise BoundaryViolation(f"expected bool, got {_agltype(value)}")
            return value.value
        case ScalarKind.JSON:
            if not isinstance(value, JsonValue):
                raise BoundaryViolation(f"expected json, got {_agltype(value)}")
            return copy.deepcopy(value_to_json_obj(value))
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


# ---------------------------------------------------------------------------
# Decode: Python return value -> AgL Value (strict)
# ---------------------------------------------------------------------------


def decode_boundary_value(
    schema: BoundarySchema, obj: object, seals: Mapping[str, object]
) -> Value:
    """Strictly decode one Python return value against *schema*.

    Mirrors :func:`encode_boundary_value`'s recursion in the opposite
    direction, with exactly these tolerances: a Python ``int`` widens to a
    declared ``decimal``; ``bool`` is rejected where ``int``/``decimal`` is
    declared (``bool`` is an ``int`` subclass); ``float`` is never accepted
    anywhere; every nominal shape (record/exception field set, enum
    ``$case`` and its fields) is matched exactly (missing, extra, or
    misnamed fields and unknown variants are rejected).  A type-variable
    leaf requires a :class:`SealedHandle` carrying this call's token for
    that variable — a stale or cross-variable handle is rejected, and so is
    a raw forged value.  Raises :class:`BoundaryViolation` on any mismatch.
    """
    match schema:
        case BoundaryScalar(kind=kind):
            return _decode_scalar(kind, obj)
        case BoundaryUnit():
            if obj is not None:
                raise BoundaryViolation(f"expected unit (None), got {_pytype(obj)}")
            return UnitValue()
        case BoundaryList(element=elem_schema):
            if not isinstance(obj, list):
                raise BoundaryViolation(f"expected a list, got {_pytype(obj)}")
            items: list[object] = obj
            return ListValue(
                tuple(decode_boundary_value(elem_schema, e, seals) for e in items)
            )
        case BoundaryDict(value=val_schema):
            if not isinstance(obj, dict):
                raise BoundaryViolation(f"expected a dict, got {_pytype(obj)}")
            mapping: dict[object, object] = obj
            entries: dict[str, Value] = {}
            for k, v in mapping.items():
                if not isinstance(k, str):
                    raise BoundaryViolation(f"dict key must be str, got {_pytype(k)}")
                entries[k] = decode_boundary_value(val_schema, v, seals)
            return DictValue(entries=entries)
        case BoundaryRecord(nominal=nominal, display_name=display_name, fields=fields):
            obj_fields = _expect_object(obj, display_name)
            _check_exact_fields(display_name, {fname for fname, _ in fields}, obj_fields)
            record_fields = {
                fname: decode_boundary_value(fschema, obj_fields[fname], seals)
                for fname, fschema in fields
            }
            return RecordValue(nominal=nominal, display_name=display_name, fields=record_fields)
        case BoundaryEnum(nominal=nominal, display_name=display_name, variants=variants):
            obj_fields = _expect_object(obj, display_name)
            case_val = obj_fields.get("$case")
            if not isinstance(case_val, str):
                raise BoundaryViolation(
                    f"enum {display_name!r}: object must have a string '$case' field"
                )
            variant = next((v for v in variants if v.name == case_val), None)
            if variant is None:
                raise BoundaryViolation(f"enum {display_name!r}: unknown variant {case_val!r}")
            expected = {fname for fname, _ in variant.fields} | {"$case"}
            _check_exact_fields(f"{display_name}.{case_val}", expected, obj_fields)
            payload = {
                fname: decode_boundary_value(fschema, obj_fields[fname], seals)
                for fname, fschema in variant.fields
            }
            return EnumValue(
                nominal=nominal, display_name=display_name, variant=case_val, fields=payload
            )
        case BoundaryException(nominal=nominal, display_name=display_name, fields=fields):
            obj_fields = _expect_object(obj, display_name)
            _check_exact_fields(display_name, {fname for fname, _ in fields}, obj_fields)
            exc_fields = {
                fname: decode_boundary_value(fschema, obj_fields[fname], seals)
                for fname, fschema in fields
            }
            return ExceptionValue(nominal=nominal, display_name=display_name, fields=exc_fields)
        case BoundarySealVar(var=var):
            if not isinstance(obj, SealedHandle):
                raise BoundaryViolation(
                    f"expected a sealed handle for type variable {var!r}, got {_pytype(obj)}"
                )
            if obj._seal is not seals.get(var):
                raise BoundaryViolation(
                    f"handle does not carry this call's seal for type variable {var!r}"
                )
            return obj._value
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _expect_object(obj: object, display_name: str) -> dict[str, object]:
    """Return *obj* as a ``str``-keyed dict, or raise ``BoundaryViolation``."""
    if not isinstance(obj, dict):
        raise BoundaryViolation(f"expected an object for {display_name!r}, got {_pytype(obj)}")
    mapping: dict[object, object] = obj
    result: dict[str, object] = {}
    for k, v in mapping.items():
        if not isinstance(k, str):
            raise BoundaryViolation(f"{display_name!r}: object key must be str, got {_pytype(k)}")
        result[k] = v
    return result


def _check_exact_fields(
    display_name: str, expected: set[str], obj_fields: Mapping[str, object]
) -> None:
    """Raise ``BoundaryViolation`` unless *obj_fields* has exactly *expected* keys."""
    actual = set(obj_fields)
    if actual != expected:
        raise BoundaryViolation(
            f"{display_name!r}: field mismatch (expected {sorted(expected)}, "
            f"got {sorted(actual)})"
        )


def _decode_scalar(kind: ScalarKind, obj: object) -> Value:
    """Strictly decode a Python scalar (or opaque json) into the matching leaf value."""
    match kind:
        case ScalarKind.TEXT:
            if isinstance(obj, str):
                return TextValue(obj)
            raise BoundaryViolation(f"expected text (str), got {_pytype(obj)}")
        case ScalarKind.INT:
            if isinstance(obj, bool):
                raise BoundaryViolation("expected int, got bool")
            if isinstance(obj, int):
                return IntValue(obj)
            raise BoundaryViolation(f"expected int, got {_pytype(obj)}")
        case ScalarKind.DECIMAL:
            if isinstance(obj, bool):
                raise BoundaryViolation("expected decimal, got bool")
            if isinstance(obj, Decimal):
                return DecimalValue(obj)
            if isinstance(obj, int):
                return DecimalValue(Decimal(obj))
            raise BoundaryViolation(f"expected decimal, got {_pytype(obj)}")
        case ScalarKind.BOOL:
            if isinstance(obj, bool):
                return BoolValue(obj)
            raise BoundaryViolation(f"expected bool, got {_pytype(obj)}")
        case ScalarKind.JSON:
            if not _is_json_shaped(obj):
                raise BoundaryViolation(f"expected a JSON-shaped value, got {_pytype(obj)}")
            return JsonValue(copy.deepcopy(obj))
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _is_json_shaped(obj: object) -> bool:
    """Return whether *obj* lies in the closed JSON-shape domain.

    ``dict``/``list``/``str``/``int``/:class:`~decimal.Decimal`/``bool``/
    ``None`` recursively; anything else (a ``float``, a :class:`SealedHandle`,
    an arbitrary object) is rejected.
    """
    if obj is None or isinstance(obj, (bool, str, int, Decimal)):
        return True
    if isinstance(obj, list):
        items: list[object] = obj
        return all(_is_json_shaped(e) for e in items)
    if isinstance(obj, dict):
        mapping: dict[object, object] = obj
        return all(isinstance(k, str) and _is_json_shaped(v) for k, v in mapping.items())
    return False


# ---------------------------------------------------------------------------
# ExternRegistry
# ---------------------------------------------------------------------------


class ExternRegistry:
    """Imports extern companions, resolves their callables, and invokes them.

    Two-step resolution mirrors the way the pipeline discovers externs by
    module: :meth:`load_companion` imports a module's companion exactly once
    (cached per canonical path, so re-importing the same file — even for a
    different module id sharing it — is a no-op); :meth:`resolve` then looks
    up one already-loaded companion's callable by name.  :meth:`invoke` is
    the single chokepoint that turns every runtime failure crossing the
    boundary into a catchable ``ExternError``, mirroring
    ``AgentRegistry.dispatch``.
    """

    def __init__(self) -> None:
        self._by_path: dict[Path, ModuleType] = {}
        self._by_module: dict[ModuleId, ModuleType] = {}

    def load_companion(self, module_id: ModuleId, companion_path: Path) -> ModuleType:
        """Import *companion_path* for *module_id*, executing it at most once.

        Registered in ``sys.modules`` under a synthetic name for the duration
        of the import only (no ``sys.path`` manipulation — the companion may
        still import installed packages absolutely).  A companion already
        imported for a different module id under the same canonical path is
        reused without re-running its top-level code.
        """
        canonical = companion_path.resolve()
        cached = self._by_path.get(canonical)
        if cached is not None:
            self._by_module[module_id] = cached
            return cached

        synthetic_name = (
            f"agm_agl_extern_companion__{module_id.dotted().replace('.', '_')}"
            f"__{len(self._by_path)}"
        )
        spec = importlib.util.spec_from_file_location(synthetic_name, canonical)
        # A ``.py``-suffixed location always resolves to a source-file loader
        # (verified to exist by the loader before this is ever called); this
        # can only be ``None`` for a suffix no loader recognizes.
        assert spec is not None and spec.loader is not None, (
            f"cannot build an import spec for companion {canonical}"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[synthetic_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise ExternImportError(
                module_id, f"companion {canonical} failed to import: {exc}"
            ) from exc
        finally:
            del sys.modules[synthetic_name]

        self._by_path[canonical] = module
        self._by_module[module_id] = module
        return module

    def resolve(self, module_id: ModuleId, name: str) -> Callable[..., object]:
        """Return *module_id*'s companion callable named *name*.

        :meth:`load_companion` must have been called for *module_id* first.
        :raises ExternResolutionError: when the companion has no attribute
            named *name*, or that attribute is not callable.
        """
        module = self._by_module.get(module_id)
        assert module is not None, (
            f"module {module_id.dotted()!r} has no loaded companion; "
            "load_companion must be called before resolve"
        )
        if not hasattr(module, name):
            raise ExternResolutionError(module_id, name)
        value: object = cast(object, getattr(module, name))
        if not callable(value):
            raise ExternResolutionError(module_id, name)
        return value

    def invoke(
        self,
        function_name: str,
        contract: ExternContract,
        fn: Callable[..., object],
        args: Sequence[Value],
        trace_id: str,
    ) -> Value:
        """Cross the boundary for one extern call: encode, call, decode.

        Mints a fresh seal token per declared type variable for this call,
        encodes *args* positionally per *contract*, calls *fn*, and strictly
        decodes its result.  All three runtime failure classes — *fn*
        raising, a return-contract violation, and an argument-conversion
        failure — become ``AglRaise(ExternError)`` here, the single
        chokepoint mirroring ``AgentRegistry.dispatch``.  ``python_type`` is
        the raising Python exception's class name, or empty for a contract
        violation.
        """
        seals: dict[str, object] = {var: object() for var in contract.type_params}

        try:
            encoded_args = [
                encode_boundary_value(param.schema, arg, seals)
                for param, arg in zip(contract.params, args, strict=True)
            ]
        except BoundaryViolation as exc:
            raise _extern_error(
                function_name, f"argument conversion failed: {exc}", trace_id, python_type=""
            ) from exc

        try:
            result = fn(*encoded_args)
        except Exception as exc:
            raise _extern_error(
                function_name,
                str(exc) or type(exc).__name__,
                trace_id,
                python_type=type(exc).__name__,
            ) from exc

        try:
            return decode_boundary_value(contract.result, result, seals)
        except BoundaryViolation as exc:
            raise _extern_error(
                function_name, f"return value violates contract: {exc}", trace_id, python_type=""
            ) from exc


def _extern_error(function_name: str, message: str, trace_id: str, *, python_type: str) -> AglRaise:
    """Build the ``AglRaise(ExternError)`` carrier shared by every invoke failure."""
    return AglRaise(
        make_builtin_exception(
            "ExternError",
            message,
            trace_id=trace_id,
            function=TextValue(function_name),
            python_type=TextValue(python_type),
        )
    )
