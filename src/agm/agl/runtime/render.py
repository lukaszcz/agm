"""AgL-native value rendering for string interpolation, print, and REPL echo.

Values render in the AgL syntax used to define them by default.  JSON output
is an explicit opt-in via ``as json``.  Two boolean axes drive the leaf cases:

- **top-level vs nested** — the caller passes the value at top level
  (``top_level=True``); every recursive child call uses ``top_level=False``.
- **interpolation vs REPL echo** — controls only the top-level ``text`` case:
  interpolation (``render_value``) leaves ``text`` verbatim; REPL echo
  (``render_value_repl``) quotes it as an AgL string literal.

Per-kind rules:

- ``text`` (top-level, interpolation)  → verbatim, no quotes
- ``text`` (top-level, REPL echo)      → quoted AgL string literal via ``_quote_text``
- ``text`` (nested, any mode)          → quoted AgL string literal via ``_quote_text``
- ``int`` / ``decimal`` / ``bool``     → ``_scalar_text`` at any depth
- ``unit``                             → ``()``
- ``agent``                            → ``<agent NAME>``
- ``function`` (``Closure``)           → ``<function/N -> T>``
- ``json`` (top-level)                 → pretty JSON, 2-space indent
- ``json`` (nested)                    → compact JSON, single-line
- ``list``                             → ``[e1, e2, ...]``, children nested
- ``dict``                             → ``{"k1": v1, ...}``, keys always quoted
- record                               → ``TypeName(f1: v1, ...)`` declaration order
- enum   → ``TypeName.Variant(f1: v1, ...)``; nullary variant → ``TypeName.Variant``
- exception                            → ``TypeName(f1: v1, ...)`` all fields incl. ``trace_id``

A ``TypeLookup`` is required for nominal values (record, enum, exception) to
emit fields in declaration order.  The caller must supply it; absence is an
internal invariant error.  ``TypeEnvironment`` satisfies the protocol structurally.
"""

from __future__ import annotations

from typing import Protocol

from agm.agl.eval.values import (
    AgentValue,
    BoolValue,
    Closure,
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
from agm.agl.runtime.serialize import dumps_exact, value_to_json_obj
from agm.agl.typecheck.types import EnumType, ExceptionType, RecordType, Type

# ---------------------------------------------------------------------------
# TypeLookup protocol
# ---------------------------------------------------------------------------


class TypeLookup(Protocol):
    """Read-only protocol for resolving a type name to its semantic ``Type``.

    ``TypeEnvironment`` satisfies this protocol structurally.  The REPL
    exposes a read-only facade backed by its persistent environment so
    presentation code cannot mutate session typing state.
    """

    def get_type(self, name: str) -> Type | None: ...


# ---------------------------------------------------------------------------
# Escape mapping for _quote_text
# ---------------------------------------------------------------------------

# JSON escape set extended with ``$`` so ``${`` cannot be read as interpolation
# inside a quoted string literal rendered into output.
_TEXT_ESCAPES: dict[str, str] = {
    '"': '\\"',
    "\\": "\\\\",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
    "\b": "\\b",
    "\f": "\\f",
    "$": "\\$",
}


def _quote_text(s: str) -> str:
    """Return *s* as a double-quoted AgL string literal surface form.

    Applies the JSON escape set plus ``\\$`` (so ``${`` cannot read as
    interpolation) and ``\\uXXXX`` for remaining control characters.  Used
    for both nested ``text`` values and the top-level REPL-echo case so the
    two never diverge.
    """
    out: list[str] = ['"']
    for ch in s:
        esc = _TEXT_ESCAPES.get(ch)
        if esc is not None:
            out.append(esc)
        elif ch < " ":
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _scalar_text(value: IntValue | DecimalValue | BoolValue) -> str:
    """Render an int, decimal, or bool value as a plain text string."""
    if isinstance(value, IntValue):
        return str(value.value)
    if isinstance(value, DecimalValue):
        # Use normalize() to drop trailing zeros (e.g. "1.50" → "1.5"),
        # but keep at least one decimal digit.
        d = value.value.normalize()
        # Avoid scientific notation (e.g. "1E+2" → "100").
        return format(d, "f")
    # BoolValue
    return "true" if value.value else "false"


def _closure_surface(closure: Closure) -> str:
    """Return the human-readable surface form for a ``Closure`` value.

    Uses only the fields the ``Closure`` carries: the arity (from ``params``)
    and the declared return type (from ``return_type``).  The form is
    ``"<function/N -> T>"`` where N is the parameter count and T is the return
    type's canonical representation.

    This surface form is produced ONLY by ``render_value`` for REPL echo and
    ``:bindings``/``:params`` display — it is never reachable from ``print``,
    template interpolation, or ``exec``, which the type checker statically
    prevents (design D9).
    """
    arity = len(closure.params)
    return f"<function/{arity} -> {closure.return_type!r}>"


# ---------------------------------------------------------------------------
# Declaration-order field helper (D7/D11)
# ---------------------------------------------------------------------------


def _ordered_fields(
    value: RecordValue | EnumValue | ExceptionValue,
    type_lookup: TypeLookup,
) -> list[tuple[str, Value]]:
    """Return (field_name, value) pairs in declared order for a nominal value.

    Resolves ``value.type_name`` via ``type_lookup``, validates the nominal
    kind and (for enums) the variant, then checks that the declared and runtime
    field-name sets match exactly.  Returns the runtime values in declared order.

    Raises ``RuntimeError`` for any invariant violation — these cannot be
    produced by a valid AgL program and must never be hidden by fallback logic.
    """
    type_name = value.type_name
    resolved = type_lookup.get_type(type_name)
    if resolved is None:
        raise RuntimeError(
            f"render: unknown type '{type_name}' — type_lookup returned None"
        )

    if isinstance(value, RecordValue):
        if not isinstance(resolved, RecordType):
            raise RuntimeError(
                f"render: type '{type_name}' resolved to {type(resolved).__name__}, "
                f"expected RecordType for RecordValue"
            )
        declared_fields = resolved.fields
        runtime_fields = value.fields
    elif isinstance(value, EnumValue):
        if not isinstance(resolved, EnumType):
            raise RuntimeError(
                f"render: type '{type_name}' resolved to {type(resolved).__name__}, "
                f"expected EnumType for EnumValue"
            )
        variant = value.variant
        if variant not in resolved.variants:
            raise RuntimeError(
                f"render: unknown variant '{variant}' in enum '{type_name}'"
            )
        declared_fields = resolved.variants[variant]
        runtime_fields = value.fields
    else:
        # ExceptionValue
        if not isinstance(resolved, ExceptionType):
            raise RuntimeError(
                f"render: type '{type_name}' resolved to {type(resolved).__name__}, "
                f"expected ExceptionType for ExceptionValue"
            )
        declared_fields = resolved.fields
        runtime_fields = value.fields

    declared_names = set(declared_fields.keys())
    runtime_names = set(runtime_fields.keys())
    if declared_names != runtime_names:
        missing = declared_names - runtime_names
        extra = runtime_names - declared_names
        parts: list[str] = []
        if missing:
            parts.append(f"missing runtime fields: {sorted(missing)}")
        if extra:
            parts.append(f"extra runtime fields: {sorted(extra)}")
        raise RuntimeError(
            f"render: field-set mismatch for '{type_name}': " + "; ".join(parts)
        )

    return [(name, runtime_fields[name]) for name in declared_fields]


# ---------------------------------------------------------------------------
# Core recursive renderer
# ---------------------------------------------------------------------------


def _render(
    value: Value,
    *,
    top_level: bool,
    repl: bool,
    type_lookup: TypeLookup | None,
) -> str:
    """Recursive AgL-native renderer.

    ``top_level=True`` for the outermost call; ``False`` for all children.
    ``repl=True`` enables REPL-echo quoting for a top-level ``text`` value.
    ``type_lookup`` is mandatory when a nominal value is encountered.
    """
    if isinstance(value, TextValue):
        if top_level and not repl:
            # Interpolation context: verbatim.
            return value.value
        # REPL echo (top-level) or nested (any mode): quoted.
        return _quote_text(value.value)

    if isinstance(value, UnitValue):
        return "()"

    if isinstance(value, (IntValue, DecimalValue, BoolValue)):
        return _scalar_text(value)

    if isinstance(value, AgentValue):
        return f"<agent {value.name}>"

    if isinstance(value, Closure):
        return _closure_surface(value)

    if isinstance(value, JsonValue):
        if top_level:
            return dumps_exact(value_to_json_obj(value), indent=2)
        return dumps_exact(value_to_json_obj(value), indent=None)

    if isinstance(value, ListValue):
        if not value.elements:
            return "[]"
        items = [_render(e, top_level=False, repl=repl, type_lookup=type_lookup)
                 for e in value.elements]
        return "[" + ", ".join(items) + "]"

    if isinstance(value, DictValue):
        if not value.entries:
            return "{}"
        items = [
            f"{_quote_text(k)}: {_render(v, top_level=False, repl=repl, type_lookup=type_lookup)}"
            for k, v in value.entries.items()
        ]
        return "{" + ", ".join(items) + "}"

    if isinstance(value, (RecordValue, EnumValue, ExceptionValue)):
        if type_lookup is None:
            raise RuntimeError(
                f"render: type_lookup is required to render nominal value "
                f"'{value.type_name}' in declaration order"
            )
        prefix = (
            f"{value.type_name}.{value.variant}"
            if isinstance(value, EnumValue)
            else value.type_name
        )
        pairs = _ordered_fields(value, type_lookup)
        if not pairs:
            return prefix if isinstance(value, EnumValue) else f"{prefix}()"
        field_parts = [
            f"{name}: {_render(v, top_level=False, repl=repl, type_lookup=type_lookup)}"
            for name, v in pairs
        ]
        return f"{prefix}(" + ", ".join(field_parts) + ")"

    # Exhaustiveness: the Value union is closed; all cases covered above.
    raise RuntimeError(f"render: unhandled value type {type(value).__name__}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_value(value: Value, type_lookup: TypeLookup | None = None) -> str:
    """Render *value* for interpolation, ``print``, or ``as text``.

    Top-level ``text`` is verbatim (no quotes).  All other rendering follows
    the AgL-native rules: scalars as plain text, ``list``/``dict`` in AgL
    bracket/brace form, record/enum/exception as ``TypeName(field: value, ...)``.
    ``json`` values render as pretty-printed JSON (2-space indent) at top level
    and compact single-line JSON when nested.

    A ``TypeLookup`` is optional for scalar and structural container values but
    mandatory when a nominal value (record, enum, exception) is encountered;
    absence at that point is an internal invariant error.
    """
    return _render(value, top_level=True, repl=False, type_lookup=type_lookup)


def render_value_repl(value: Value, type_lookup: TypeLookup | None = None) -> str:
    """Render *value* for REPL echo (``agl>`` prompt and ``:bindings`` / ``:params``).

    Identical to :func:`render_value` except that a top-level ``text`` value is
    shown as a quoted AgL string literal so the REPL echo of ``"aaa"`` reads
    ``"aaa"``.  Text nested inside structured values is also quoted.  Template
    interpolation (``print`` / ``prompt`` / ``exec``) always uses
    :func:`render_value` verbatim.
    """
    return _render(value, top_level=True, repl=True, type_lookup=type_lookup)
