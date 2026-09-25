"""Single value-model home for the AgL semantics layer.

This module is the **single source of truth** for every runtime value type in
the AgL execution pipeline: leaf primitive value tags, container and nominal
types, IR closures, and the per-invocation frame model (``Cell``, ``Slot``,
``Frame``).

There is exactly one ``Value`` union covering all leaf primitive, container,
nominal, and callable value kinds.

Design constraints
------------------
- Imports ONLY from the Python standard library and ``agm.agl.ir.ids``.
  Must NOT import from ``eval``, ``syntax``, ``scope``, ``typecheck``,
  ``runtime``, or ``modules`` — not even under ``TYPE_CHECKING``.
- All value types are frozen dataclasses with ``__slots__`` for memory
  efficiency.
- ``Cell`` is the sole mutable type (not frozen) — it is a mutable box for
  ``var`` bindings.
"""

from __future__ import annotations

import decimal
from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

from agm.agl.ir.ids import ContractId, FunctionId, NominalId, SymbolId
from agm.util.decimal import int_in_range

# ---------------------------------------------------------------------------
# JSON-tree comparison helpers
# ---------------------------------------------------------------------------


def _json_eq(left: object, right: object) -> bool:
    """Compare two JSON-shaped trees with bool-guarded numeric equivalence.

    Mirrors the semantics in the interpreter: JSON numbers compare numerically
    (``1 == 1.0``), but ``bool`` is a distinct JSON kind and never compares
    equal to a number (no Python ``True == 1`` conflation).  Containers recurse
    structurally; ``text`` and ``null`` compare exactly.
    """
    # bool first: Python treats bool as a subclass of int, so ``True == 1``.
    # Guard: a bool only equals another bool of the same value.
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, decimal.Decimal)) and isinstance(right, (int, decimal.Decimal)):
        return decimal.Decimal(left) == decimal.Decimal(right)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        return all(_json_eq(left[i], right[i]) for i in range(len(left)))
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return False
        return all(_json_eq(left[k], right[k]) for k in left)
    return left == right


def _json_hash(obj: object) -> int:
    """Stable hash for a JSON-shaped tree.

    Must be consistent with ``_json_eq``: objects that compare equal must hash
    equal.  Because ``_json_eq`` treats numeric int/Decimal equivalently, we
    normalise numbers to ``Decimal`` before hashing.  Lists and dicts recurse;
    bools are guarded so ``True`` never hashes the same as ``1``.
    """
    if isinstance(obj, bool):
        # Hash True/False distinctly from integers.
        return hash(("__bool__", obj))
    if isinstance(obj, (int, decimal.Decimal)):
        # Normalise to Decimal so 1 and Decimal("1") hash the same.
        return hash(decimal.Decimal(obj))
    if isinstance(obj, list):
        return hash(tuple(_json_hash(e) for e in obj))
    if isinstance(obj, dict):
        return hash(frozenset((_json_hash(k), _json_hash(v)) for k, v in obj.items()))
    return hash(obj)


# ---------------------------------------------------------------------------
# Primitive value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextValue:
    """A ``text`` value: a plain Python ``str``."""

    value: str


@dataclass(frozen=True, slots=True)
class IntValue:
    """An ``int`` value: an arbitrary-precision Python ``int``."""

    value: int


@dataclass(frozen=True, slots=True)
class DecimalValue:
    """A ``decimal`` value: an exact ``decimal.Decimal``."""

    value: decimal.Decimal


@dataclass(frozen=True, slots=True)
class BoolValue:
    """A ``bool`` value."""

    value: bool


@dataclass(frozen=True, slots=True, eq=False)
class JsonValue:
    """A ``json`` value: any JSON-shaped Python object (the dynamic boundary).

    The ``raw`` field is ``object`` to allow ``None``, dicts, lists, strings,
    ints, floats, and bools — exactly what ``json.loads`` can return.  All
    operations on the payload use ``isinstance`` guards, never bare ``Any``
    access.

    ``__eq__`` delegates to ``_json_eq`` so that JSON bool/number conflation is
    prevented inside containers (e.g. ``JsonValue([True]) != JsonValue([1])``),
    consistent with the top-level ``json = json`` comparison semantics.
    ``__hash__`` is consistent with ``__eq__`` via ``_json_hash``.
    """

    raw: object

    def __eq__(self, other: object) -> bool:
        if isinstance(other, JsonValue):
            return _json_eq(self.raw, other.raw)
        return NotImplemented

    def __hash__(self) -> int:
        return _json_hash(self.raw)


# ---------------------------------------------------------------------------
# Unit and agent handle value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UnitValue:
    """The ``unit`` value: a singleton with no data. Always renders ``()``."""


UNIT_VALUE: UnitValue = UnitValue()


# ---------------------------------------------------------------------------
# Callable value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConstructorValue:
    """A first-class constructor used as a callable value — opaque.

    Carries the record identity needed to build a value at the call site.
    Field order and types (and concreteness) come from the call site's checked
    result type; type arguments are erased — never represented at runtime. It
    is not renderable or comparable by the language.

    ``nominal`` is the opaque ``NominalId`` of the record type; its display
    spelling is looked up from the program's descriptor table when needed.
    """

    nominal: NominalId


# ---------------------------------------------------------------------------
# Container value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class ArrayValue:
    """An ``array[T]`` value: a mutable reference to a list of ``Value`` items.

    ``frozen=True`` prevents rebinding the ``elements`` attribute to a new
    list; the payload list itself is mutated in place by indexed assignment,
    and that mutation is observed by every binding, field, capture, or
    iterator that holds a reference to this ``ArrayValue``. Unhashable: the
    payload is mutable, so a stable hash is impossible.

    Equality delegates to :func:`values_equal`, which is cycle-safe and
    co-inductive — reference semantics makes a cyclic array constructible, and
    ``==`` on one must terminate rather than recurse forever.
    """

    elements: list[Value]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ArrayValue):
            return values_equal(self, other)
        return NotImplemented


@dataclass(frozen=True, slots=True, eq=False, init=False)
class DictValue:
    """A ``dict[K, V]`` value: a mutable reference to entries, storage hidden behind an API.

    Two representations: a ``text``-keyed dict stores ``dict[str, Value]``
    directly, with no per-key wrapper or token — the hot path, and the
    default an empty dict starts in. A non-``text`` first insert fixes the
    dict to token-keyed storage instead: ``dict[Hashable, tuple[Value,
    Value]]`` keyed by :func:`key_token`, holding ``(original key, value)``.
    So an empty dict is undetermined (``text``-keyed until proven otherwise)
    and stays that way if every entry is later removed. Updating a key equal
    to one already stored (e.g. ``1.5`` over stored ``1.50``) replaces only
    the value — the originally inserted key is kept, so iteration is stable.

    No code outside this class touches ``_text``/``_tokens``; every bulk
    operation (equality, copying, text-keyed iteration) is a method here so
    callers never need representation-specific storage access.

    Mutation is observed by every binding, field, capture, or iterator that
    holds a reference to this ``DictValue``, as for :class:`ArrayValue`.
    Unhashable: the payload is mutable, so a stable hash is impossible.
    Equality delegates to :func:`values_equal` (cycle-safe, co-inductive).
    """

    _text: dict[str, Value]
    _tokens: dict[Hashable, tuple[Value, Value]] | None

    def __init__(self, entries: dict[str, Value] | None = None) -> None:
        """Construct a dict value, taking ownership of *entries* (no copy).

        *entries* seeds a ``text``-keyed dict directly, for the common case
        of a literal or decoded ``text``-keyed dict; the caller must not
        keep mutating it afterwards, since this ``DictValue`` now owns it in
        place. Omitted or empty, the dict is undetermined — exactly the same
        state either way, since there is no key yet to fix a representation
        from.
        """
        text: dict[str, Value] = {} if entries is None else entries
        object.__setattr__(self, "_text", text)
        object.__setattr__(self, "_tokens", None)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DictValue):
            return values_equal(self, other)
        return NotImplemented

    def __repr__(self) -> str:
        body = ", ".join(f"{key!r}: {value!r}" for key, value in self.items())
        return f"DictValue({{{body}}})"

    def __len__(self) -> int:
        return len(self._text) if self._tokens is None else len(self._tokens)

    def lookup(self, key: Value) -> Value | None:
        """Return the value stored under *key*, or ``None`` if absent."""
        if self._tokens is None:
            if not isinstance(key, TextValue):
                return None
            return self._text.get(key.value)
        pair = self._tokens.get(key_token(key))
        return None if pair is None else pair[1]

    def insert(self, key: Value, value: Value) -> bool:
        """Store *value* under *key*; fixes the representation on the first insert.

        Returns whether *key* was new (``False`` if it already existed — the
        stored key value is then left unchanged, only its value replaced).
        """
        if self._tokens is None and isinstance(key, TextValue):
            is_new = key.value not in self._text
            self._text[key.value] = value
            return is_new
        tokens = self._tokens
        if tokens is None:
            tokens = {}
            object.__setattr__(self, "_tokens", tokens)
        token = key_token(key)
        existing = tokens.get(token)
        if existing is None:
            tokens[token] = (key, value)
            return True
        tokens[token] = (existing[0], value)
        return False

    def update_existing(self, key: Value, value: Value) -> bool:
        """Replace the value stored under *key* only if it already exists; report whether it did."""
        if self._tokens is None:
            if not isinstance(key, TextValue) or key.value not in self._text:
                return False
            self._text[key.value] = value
            return True
        token = key_token(key)
        existing = self._tokens.get(token)
        if existing is None:
            return False
        self._tokens[token] = (existing[0], value)
        return True

    def remove(self, key: Value) -> Value | None:
        """Remove *key* and return its value, or ``None`` if absent."""
        if self._tokens is None:
            if not isinstance(key, TextValue):
                return None
            return self._text.pop(key.value, None)
        pair = self._tokens.pop(key_token(key), None)
        return None if pair is None else pair[1]

    def pop_last(self) -> tuple[Value, Value] | None:
        """Remove and return the most-recently-inserted ``(key, value)`` pair, or ``None`` if empty.

        O(1): pops directly from the storage dict's own insertion order.
        """
        if self._tokens is None:
            if not self._text:
                return None
            key_str, value = self._text.popitem()
            return TextValue(key_str), value
        if not self._tokens:
            return None
        _token, pair = self._tokens.popitem()
        return pair

    def clear(self) -> None:
        """Remove every entry, keeping the fixed representation (if any)."""
        if self._tokens is None:
            self._text.clear()
        else:
            self._tokens.clear()

    def items(self) -> Iterator[tuple[Value, Value]]:
        """Iterate ``(key, value)`` pairs in insertion order, with original key values."""
        if self._tokens is None:
            for key_str, value in self._text.items():
                yield TextValue(key_str), value
        else:
            yield from self._tokens.values()

    def keys(self) -> Iterator[Value]:
        """Iterate original key values in insertion order."""
        if self._tokens is None:
            return (TextValue(key_str) for key_str in self._text)
        return (key for key, _value in self._tokens.values())

    def values(self) -> Iterator[Value]:
        """Iterate values in insertion order."""
        if self._tokens is None:
            return iter(self._text.values())
        return (value for _key, value in self._tokens.values())

    def text_items(self) -> Iterator[tuple[str, Value]]:
        """Iterate ``(str, Value)`` pairs straight from ``text`` storage, unwrapped.

        For callers statically restricted to ``text``-keyed dicts (JSON and
        display encoders, the environment table, ``AglDictView``'s public
        surface): they never see a token-keyed dict, so this skips the
        per-key ``TextValue`` wrap/unwrap :meth:`items` pays.
        """
        return iter(self._text.items())

    def fill_from(
        self, source: "DictValue", transform: Callable[[Value], Value] | None = None
    ) -> None:
        """Fill this (empty) dict from *source*, adopting its representation.

        Copies storage directly — a bulk ``dict.update`` when *transform* is
        ``None``, a bulk comprehension applying *transform* to each value
        otherwise — so this pays no per-key ``insert``/``key_token`` call.
        Keys are immutable and shared either way; only values are
        transformed. *source*'s representation (including an
        emptied-but-token-determined dict) is preserved.
        """
        if source._tokens is None:
            if transform is None:
                self._text.update(source._text)
            else:
                for key_str, value in source._text.items():
                    self._text[key_str] = transform(value)
            return
        tokens: dict[Hashable, tuple[Value, Value]]
        if transform is None:
            tokens = dict(source._tokens)
        else:
            tokens = {
                token: (key, transform(value)) for token, (key, value) in source._tokens.items()
            }
        object.__setattr__(self, "_tokens", tokens)

    def aligned_values(self, other: "DictValue") -> list[tuple[Value, Value]] | None:
        """Pair this dict's values with *other*'s by key, or ``None`` if their key sets differ.

        When both dicts share a representation, compares key sets directly
        against storage (``_text.keys()``/``_tokens.keys()``) at C speed —
        the common case. A representation mismatch (one ``text``-keyed, the
        other token-keyed — e.g. one is undetermined-empty while the other
        holds a ``text`` key under a token fixed by an earlier, since-removed
        non-``text`` key) falls back to a per-key ``lookup`` on *other*.
        """
        if len(self) != len(other):
            return None
        if self._tokens is None and other._tokens is None:
            if self._text.keys() != other._text.keys():
                return None
            return [(value, other._text[key_str]) for key_str, value in self._text.items()]
        if self._tokens is not None and other._tokens is not None:
            if self._tokens.keys() != other._tokens.keys():
                return None
            return [
                (value, other._tokens[token][1]) for token, (_key, value) in self._tokens.items()
            ]
        pairs: list[tuple[Value, Value]] = []
        for key, value in self.items():
            matched = other.lookup(key)
            if matched is None:
                return None
            pairs.append((value, matched))
        return pairs


# ---------------------------------------------------------------------------
# Nominal value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class RecordValue:
    """A record-typed value.

    ``nominal`` is the opaque ``NominalId`` — the identity key; its display
    spelling for rendering and diagnostics is looked up from the program's
    descriptor table, never stored on the value.  ``fields`` holds the
    record's field values, in declaration order (every construction path —
    constructor call, ``with`` update — fills it in that order; :func:`key_token`
    relies on this to token identically regardless of construction path).

    Equality is by ``(nominal, fields)``. Unhashable: ``fields`` may hold a
    mutable array or dict, so a stable hash is impossible. Delegates to
    :func:`values_equal` (cycle-safe, co-inductive) — see :class:`ArrayValue`.
    """

    nominal: NominalId
    fields: dict[str, Value] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RecordValue):
            return values_equal(self, other)
        return NotImplemented


@dataclass(frozen=True, slots=True, eq=False)
class ExceptionValue:
    """A built-in AgL exception value.

    ``nominal`` is the opaque ``NominalId`` — the identity key.  A built-in
    exception a program declares nothing of its own for uses its reserved
    identity (see ``ir.reserved_nominals``) — the shipped standard library's
    own identity; one the program redeclares as its own ``builtin exception``
    uses that declaration's identity instead. Its display spelling (e.g.
    ``"AgentParseError"``) is looked up from the program's descriptor table,
    never stored on the value.
    ``fields`` maps the exception's declared field names to their values, in
    declaration order (see :class:`RecordValue`; the same ordering
    guarantee holds here). The ``"message"`` field is always present (base
    ``Exception`` contract).

    Equality is by ``(nominal, fields)``. Unhashable: ``fields`` may hold a
    mutable array or dict, so a stable hash is impossible. Delegates to
    :func:`values_equal` (cycle-safe, co-inductive) — see :class:`ArrayValue`.
    """

    nominal: NominalId
    fields: dict[str, Value] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ExceptionValue):
            return values_equal(self, other)
        return NotImplemented


# ---------------------------------------------------------------------------
# Cycle-safe structural equality
# ---------------------------------------------------------------------------


#: The value kinds whose equality is structural — the only ones that recurse.
#: Everything else (scalars, ``json``, agents, constructors, closures,
#: iterators, ``unit``) is compared by its own ``__eq__``, so the common
#: scalar comparison leaves :func:`values_equal` after a single check.
_STRUCTURAL_KINDS = (ArrayValue, DictValue, RecordValue, ExceptionValue)


def key_token(value: Value) -> Hashable:
    """Canonical hashable token for *value*, used as a :class:`DictValue` non-``text`` key.

    Values equal under AgL equality share a token, distinct values don't:
    every scalar/``json`` ``Value`` is its own token, since their frozen
    dataclass (or, for ``json``, ``_json_eq``/``_json_hash``-backed) equality
    and hashing already follow AgL equality — distinct classes never compare
    equal (``bool`` != ``int``), ``decimal`` widens (``1.5`` == ``1.50``, same
    hash), and ``json`` bools tag apart from numbers. A record, enum member,
    or exception tokens as its nominal identity paired with its fields'
    tokens, in declaration order (see :class:`RecordValue`).
    """
    if isinstance(value, (RecordValue, ExceptionValue)):
        return (
            value.nominal,
            tuple(key_token(field_value) for field_value in value.fields.values()),
        )
    return value


def value_equal(left: Value, right: Value) -> bool:
    """Return AgL equality for one value position.

    ``int`` and ``decimal`` widen when compared directly. Structural equality
    remains responsible for recursive positions, where values retain their
    exact element types and therefore do not widen. AgL source never reaches
    this branch with an out-of-range int operand -- the lowerer's
    ``IntToDecimal`` coercion widens (and range-checks) any mixed int/decimal
    equality operand at compile time -- but an FFI companion can still
    produce an ``IntValue`` of arbitrary magnitude uncoerced (e.g. indexing a
    live array view), so an out-of-range int is decided ``False`` via the
    cheap ``int_in_range`` check rather than constructing ``Decimal(int)`` for
    a magnitude that can never equal any valid decimal anyway.
    """
    if isinstance(left, IntValue) and isinstance(right, DecimalValue):
        if not int_in_range(left.value):
            return False
        return decimal.Decimal(left.value) == right.value
    if isinstance(left, DecimalValue) and isinstance(right, IntValue):
        if not int_in_range(right.value):
            return False
        return left.value == decimal.Decimal(right.value)
    return values_equal(left, right)


def values_equal(a: Value, b: Value, _seen: "set[tuple[int, int]] | None" = None) -> bool:
    """Structural equality between two values — cycle-safe and co-inductive.

    Every container/nominal ``__eq__`` above delegates here rather than
    recursing through Python's own ``list``/``dict`` equality, which cannot
    thread a memo. This function recurses through itself instead, carrying
    ``_seen`` — a set of ``(id(a), id(b))`` pairs taken to be equal.  A pair
    re-entered while still being compared is co-inductively assumed equal
    (the standard construction for equality on a possibly-cyclic graph):
    comparing two cyclic structures terminates and answers the co-inductive
    relation. This must never raise.

    Every structural kind extends ``_seen``, including the nominal ones.
    Mutable record fields can close a cycle directly, so nominal pairs need
    the same co-inductive entry as arrays and dicts to terminate. The monotone
    memo also keeps a shared subterm from being re-compared once per path that
    reaches it.

    Every other value kind (scalars, ``json``, agents, constructors,
    closures) falls through to its own ``__eq__`` unchanged — this function
    only replaces the *container* recursion, never leaf comparison, so
    equality on acyclic values is byte-for-byte the relation it always was
    (in particular, no int/decimal widening happens *inside* a container).

    The identity check is the first thing this function does, for every
    ``Value`` kind (there is no NaN-like value in AgL where ``a is b`` would
    not imply equal): without it, a nominal value's ``__eq__`` recurses into
    its fields even when compared to itself.
    """
    if a is b:
        return True
    if not isinstance(a, _STRUCTURAL_KINDS):
        return a == b
    if isinstance(a, ArrayValue):
        if not isinstance(b, ArrayValue) or len(a.elements) != len(b.elements):
            return False
        return _children_equal(a, b, _seen, zip(a.elements, b.elements))
    if isinstance(a, DictValue):
        if not isinstance(b, DictValue):
            return False
        pairs = a.aligned_values(b)
        if pairs is None:
            return False
        return _children_equal(a, b, _seen, pairs)
    if isinstance(a, RecordValue):
        if not isinstance(b, RecordValue) or a.nominal != b.nominal:
            return False
        return _fields_equal(a, b, a.fields, b.fields, _seen)
    if not isinstance(b, ExceptionValue) or a.nominal != b.nominal:
        return False
    return _fields_equal(a, b, a.fields, b.fields, _seen)


def _children_equal(
    a: Value,
    b: Value,
    seen: "set[tuple[int, int]] | None",
    pairs: "Iterable[tuple[Value, Value]]",
    /,
) -> bool:
    """Compare the aligned children of *a* and *b*, memoizing the ``(a, b)`` pair.

    *seen* is monotone: a pair is recorded before its children are compared
    (so a cyclic re-entry terminates by co-inductive assumption) and is never
    withdrawn (so a subterm reachable by several paths — a diamond — is
    compared once rather than once per path; withdrawing it on the way out
    makes equality exponential in the depth of a shared structure).

    Retaining a pair whose comparison ultimately FAILS is sound because a
    single ``False`` anywhere aborts the whole comparison: every recursive
    call sits inside an ``all(...)``, so the first mismatch short-circuits
    every enclosing walk and the root returns ``False`` without consulting
    *seen* again. *pairs* is a plain iterable — every call site already
    builds it lazily (``zip`` or a generator expression), so there is nothing
    for a thunk to defer.
    """
    key = (id(a), id(b))
    if seen is None:
        seen = set()
    elif key in seen:
        return True
    seen.add(key)
    return all(values_equal(x, y, seen) for x, y in pairs)


def _fields_equal(
    a: Value,
    b: Value,
    a_fields: dict[str, Value],
    b_fields: dict[str, Value],
    seen: "set[tuple[int, int]] | None",
) -> bool:
    """Compare two nominal field maps, memoizing the owning ``(a, b)`` pair."""
    return a_fields.keys() == b_fields.keys() and _children_equal(
        a, b, seen, ((v, b_fields[k]) for k, v in a_fields.items())
    )


@dataclass(frozen=True, slots=True)
class IrClosureValue:
    """An IR closure: function_id plus its captured environment.

    Its signature labels and arity are looked up from the program's
    ``FunctionDescriptor`` table by ``function_id`` when rendering, never
    stored on the value.
    """

    function_id: FunctionId
    captures: tuple[tuple[SymbolId, Slot], ...]

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


# A first-class function value: a closure or a constructor.
FunctionValue: TypeAlias = IrClosureValue | ConstructorValue


@dataclass(slots=True, eq=False)
class IteratorValue:
    """Internal loop iterator cursor.

    Holds the source sequence by reference, plus its entry length and a
    position index. Array element and structural mutations are therefore
    visible at not-yet-reached live indices, while the entry length prevents
    appends from extending the loop indefinitely. A shorter live sequence
    exhausts the cursor early. Dict keys are materialized once; text retains
    its string and wraps each code point only when consumed. Mutable in place
    so ``IrIterNext`` can advance without rebuilding the object.

    Never rendered, hashed for equality, serialized, or returned to user
    code — it is an evaluator-internal value only.
    """

    elements: "Sequence[Value] | str"
    pos: int = 0
    entry_length: int = field(init=False)

    def __post_init__(self) -> None:
        self.entry_length = len(self.elements)


@dataclass(frozen=True, slots=True, eq=False)
class ContractValue:
    """A type-directed extern occurrence's target contract, bound for the companion.

    Opaque non-data: never rendered, compared, serialized, or crossed except
    as a leading extern argument.
    """

    contract_id: ContractId


# ---------------------------------------------------------------------------
# Broad runtime value union
# ---------------------------------------------------------------------------

Value: TypeAlias = (
    TextValue
    | IntValue
    | DecimalValue
    | BoolValue
    | JsonValue
    | ArrayValue
    | DictValue
    | RecordValue
    | ExceptionValue
    | UnitValue
    | ConstructorValue
    | IrClosureValue
    | IteratorValue
    | ContractValue
)

# Every value a program can observe: an iterator is internal to loop lowering.
ObservableValue: TypeAlias = (
    TextValue
    | IntValue
    | DecimalValue
    | BoolValue
    | JsonValue
    | ArrayValue
    | DictValue
    | RecordValue
    | ExceptionValue
    | UnitValue
    | ConstructorValue
    | IrClosureValue
    | ContractValue
)

# ---------------------------------------------------------------------------
# Frame and cell model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Cell:
    """A mutable box wrapping a ``Value``.

    Used as the slot for ``var`` (mutable) bindings in the per-invocation
    frame.  The cell itself is mutable (not frozen) so that ``IrAssign`` can
    update the contained value in place.

    Closures capture a ``var`` by capturing the ``Cell`` reference; the cell
    is allocated fresh each time ``IrBind`` executes for a ``var`` symbol.
    """

    value: Value


#: A slot in the runtime frame is either a ``Value`` (for ``let`` bindings)
#: or a ``Cell`` (for ``var`` bindings).  Discriminate with ``isinstance``.
Slot = Value | Cell

#: Runtime frame type: maps each bound ``SymbolId`` to its slot.
Frame = dict[SymbolId, Slot]

__all__ = [
    "UNIT_VALUE",
    "ArrayValue",
    "BoolValue",
    "Cell",
    "ConstructorValue",
    "ContractValue",
    "DecimalValue",
    "DictValue",
    "ExceptionValue",
    "Frame",
    "FunctionValue",
    "IntValue",
    "IrClosureValue",
    "IteratorValue",
    "JsonValue",
    "NominalId",
    "RecordValue",
    "Slot",
    "TextValue",
    "UnitValue",
    "Value",
    "_json_eq",
    "_json_hash",
    "key_token",
    "value_equal",
    "values_equal",
]
