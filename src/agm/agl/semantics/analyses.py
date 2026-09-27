"""Declaration-level whole-type analyses over a shared ``TypeTable``.

Nominal types are handles (``semantics.types``) backed by ``TypeDef``
templates in a ``TypeTable`` (``semantics.type_table``); a declaration may
reference itself or another declaration directly or indirectly, so any
whole-type question ("does every value of this type terminate?", "can a
non-data value hide inside this type?") must be answered over the finite
*declaration graph* rather than by walking an individual type tree, which may
be cyclic. Inhabitation, non-data reachability, and hashability below share
that shape: start from a conservative default, and grow a set of facts to a
least fixpoint by re-examining declarations' own field/variant templates
until nothing changes. Because there are finitely
many declarations, this always terminates, and because each fact only ever
flips from "not yet established" to "established" (never back), the fixpoint
is independent of iteration order.

Inhabitation
------------
A declaration is inhabited iff it has at least one finite value. Recursion
through an ``array``/``dict`` field is always fine (the empty collection is a
value regardless of the element type); recursion through a record/exception
field or every variant of an enum is fine only if some path bottoms out
without needing another value of the same (or a mutually recursive)
declaration. Every enum member is a record declaration checked on its own,
and a generic reference counts only which of its arguments are inhabited.
:func:`compute_uninhabited` returns the declarations that never reach that
bottom, solving one declaration-reference SCC at a time, lowest first.

Non-data reachability, hashability, JSON convertibility, and extern crossability
---------------------------------------------------------------------------------
These four facts are one shared declaration-level fixpoint
(:func:`compute_declaration_flags` — see its own docstring for the growth
rule), one instance per :class:`~agm.agl.semantics.type_table.DataProperty`,
each with its own :class:`~agm.agl.semantics.type_table.LeafPolicy`
(:data:`~agm.agl.semantics.type_table.LEAF_POLICIES`). ``EQ``'s policy flags a
declaration that unconditionally reaches ``unit`` or a function type — the
basis of ``=``/``!=`` (:meth:`~agm.agl.semantics.type_table.TypeTable.nominal_satisfies`).
``HASHABLE``'s policy flags one that is not deeply immutable data: it has an
``array``/``dict``/function/``unit``/``var`` field, transitively (see
``semantics.type_table.satisfies``). ``JSON_CONVERTIBLE`` is ``EQ`` plus a
non-``Hashable``-keyed ``dict`` anywhere (any ``Hashable`` key has a JSON wire
form — stringified onto an object key or, failing that, an entries array;
:meth:`~agm.agl.semantics.type_table.TypeTable.nominal_is_json_convertible`)
— a ``dict`` key position filled by one of the declaration's OWN type
parameters, directly or through another declaration's own key parameter, is
deferred rather than flagged (see ``DeclarationFlags.key_params``), since its
wire form depends on the argument a later concrete reference supplies.
``EXTERN_CROSSABLE`` is the same key-position deferral, but its own key rule
is ``text``-only (the only key type with an FFI wire form), and a function
leaf is itself data-crossable (an extern parameter may be a callback), so it
recurses into function types instead of flagging them
(:func:`~agm.agl.semantics.type_table.is_extern_crossable`).

Finiteness (instantiation-closure) capability
----------------------------------------------
A generic recursive declaration may reference itself (or a mutually
recursive peer) at a DIFFERENT argument, not just the same one — e.g.
``Perfect[T]`` referencing ``Perfect[Pair[T, T]]``. Constructing, matching,
and rendering such a type works fine (every actual VALUE is still a finite
tree), but its **instantiation closure** — the set of concrete
``(declaration, args)`` pairs reachable by repeatedly expanding fields
starting from one concrete instantiation — can be infinite: ``Perfect[int]``
reaches ``Perfect[Pair[int, int]]``, ``Perfect[Pair[Pair[int, int], Pair[int,
int]]]``, … forever. A type with an infinite closure has no finite JSON
schema, which matters at schema-producing boundaries such as agent/exec
outputs, casts from JSON/text, and external params.

:func:`compute_finite_closure` decides, once per table build, which
declarations have a finite closure. Unlike inhabitation and non-data
reachability, this is not a monotone fixpoint grown fact-by-fact — it is a
one-shot graph classification: build the declaration reference graph (which
declaration reference templates mention which other declarations, and with
which argument templates), find its strongly-connected components (SCCs),
and within each SCC build a small parameter-dependency graph (which of a
referencing declaration's OWN parameters feed which of the referenced
declaration's parameters, and whether that feed is "growing" — the source
parameter occurs as a proper subterm of the argument template, under a
array/dict/function/nominal-argument constructor, rather than being passed
through unchanged). An SCC's closure is infinite iff that small graph has a
cycle containing at least one growing edge; every declaration in such an SCC
is infinite, everything else is finite. See :func:`compute_finite_closure`
for the full definition and
:meth:`~agm.agl.semantics.type_table.TypeTable.has_finite_schema` for the
per-concrete-type reachability query built on top of it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from typing import assert_never

from agm.agl.semantics.type_table import (
    DeclId,
    LeafPolicy,
    TypeDef,
    TypeDefKind,
    TypeTable,
    decl_id_sort_key,
)
from agm.agl.semantics.types import (
    ArrayType,
    BoolType,
    BottomType,
    DecimalType,
    DictType,
    EnumType,
    ExceptionType,
    FunctionType,
    InferenceVarType,
    IntType,
    JsonType,
    RecordType,
    TextType,
    Type,
    TypeVarType,
    UnitType,
    substitute,
)
from agm.util.graph import sccs

InhabitationKey = tuple[DeclId, tuple[bool, ...]]
"""A declaration applied to arguments, reduced to whether each argument is inhabited."""


# ---------------------------------------------------------------------------
# Inhabitation
# ---------------------------------------------------------------------------


def compute_uninhabited(table: TypeTable) -> frozenset[DeclId]:
    """Return every registered declaration identity that has no finite value.

    Every declaration is checked on its own, inline enum members included: a
    member is a record, so an inhabited sibling never excuses it.

    A type parameter stands for one value of its argument, so a generic
    reference's answer depends only on WHICH of its arguments are inhabited.
    The least fixpoint therefore runs over :data:`InhabitationKey` s: each
    declaration's own key has every parameter inhabited (a free type variable
    is), and evaluating a body demands the key of every reference in it —
    ``Box[Bad]`` demands ``Box`` with an uninhabited parameter, so a generic
    wrapper never hides unguarded recursion, while ``Box[Box[int]]`` demands
    the same key as ``Box[int]``. A parameter that never reaches a field
    outside an ``array``/``dict``/function cannot change the answer, so it is
    fixed to inhabited in every key. There are finitely many keys, so the
    fixpoint terminates.
    """
    solver = _InhabitationSolver(table)
    return frozenset(
        decl_id for decl_id, typedef in table.defs.items() if not solver.solved(decl_id, typedef)
    )


class _InhabitationSolver:
    """Least fixpoint over :data:`InhabitationKey` s, grown as bodies demand keys.

    Keys are evaluated lowest declaration-reference SCC first, so a key read
    from a lower SCC is final unless it was demanded just now: that read
    answers ``None`` (unknown), which suspends the reader until the lower key
    settles instead of demanding keys built from a provisional answer.
    Within an SCC, a key is re-evaluated only when a key it read turns
    inhabited.
    """

    __slots__ = (
        "_table",
        "_defs",
        "_relevant",
        "_rank",
        "_inhabited",
        "_demanded",
        "_readers",
        "_queue",
        "_queued",
        "_order",
        "_current",
        "_current_rank",
    )

    _current: InhabitationKey
    _current_rank: int

    def __init__(self, table: TypeTable) -> None:
        defs = table.defs
        self._table = table
        self._defs = defs
        self._relevant = _compute_relevant_params(defs, through_containers=False)
        components = sccs(
            _inhabitation_references(table), key=lambda decl_id: decl_id_sort_key(defs, decl_id)
        )
        self._rank = {decl_id: rank for rank, comp in enumerate(components) for decl_id in comp}
        self._inhabited: set[InhabitationKey] = set()
        self._demanded: set[InhabitationKey] = set()
        # Key -> same-SCC keys whose evaluation read it while not yet inhabited.
        self._readers: dict[InhabitationKey, set[InhabitationKey]] = {}
        self._queue: list[tuple[int, int, InhabitationKey]] = []
        self._queued: set[InhabitationKey] = set()
        self._order = count()
        for decl_id, typedef in defs.items():
            self._demand(_own_key(decl_id, typedef))
        self._solve()

    def solved(self, decl_id: DeclId, typedef: TypeDef) -> bool:
        """Return whether *typedef* is inhabited with its parameters free."""
        return _own_key(decl_id, typedef) in self._inhabited

    def _demand(self, key: InhabitationKey) -> None:
        self._demanded.add(key)
        self._enqueue(key)

    def _enqueue(self, key: InhabitationKey) -> None:
        if key not in self._queued:
            self._queued.add(key)
            heappush(self._queue, (self._rank[key[0]], next(self._order), key))

    def _solve(self) -> None:
        while self._queue:
            rank, _order, key = heappop(self._queue)
            self._queued.discard(key)
            self._current, self._current_rank = key, rank
            result = self._body_inhabited(key)
            if result is None:
                self._enqueue(key)
            elif result:
                self._inhabited.add(key)
                for reader in self._readers.pop(key, ()):
                    if reader not in self._inhabited:
                        self._enqueue(reader)

    def _read(self, key: InhabitationKey) -> bool | None:
        """Return whether *key* is inhabited so far, or ``None`` while a lower SCC settles it."""
        if key in self._inhabited:
            return True
        if key not in self._demanded:
            self._demand(key)
        if self._rank[key[0]] < self._current_rank:
            return None if key in self._queued else False
        self._readers.setdefault(key, set()).add(self._current)
        return False

    def _body_inhabited(self, key: InhabitationKey) -> bool | None:
        decl_id, params = key
        typedef = self._defs[decl_id]
        env = dict(zip(typedef.type_params, params, strict=True))
        if typedef.kind == "enum":
            return _any_inhabited(self._type_inhabited(member, env) for member in typedef.members)
        if typedef.kind == "exception":
            return self._exception_inhabited(decl_id, typedef)
        return _all_inhabited(self._type_inhabited(t, env) for _fname, t in typedef.fields)

    def _exception_inhabited(self, decl_id: DeclId, typedef: TypeDef) -> bool | None:
        if typedef.abstract:
            return _any_inhabited(
                self._read((child_id, ())) for child_id in self._table.exception_children(decl_id)
            )
        chain = [decl_id]
        base = typedef.base
        while base is not None:
            if base in chain:
                return False
            chain.append(base)
            base = self._defs[base].base
        return _all_inhabited(
            self._type_inhabited(t, {})
            for ancestor in chain
            for _fname, t in self._defs[ancestor].fields
        )

    def _type_inhabited(self, t: Type, env: Mapping[str, bool]) -> bool | None:
        match t:
            case TypeVarType():
                # A variable the declaration does not bind is free, hence inhabited.
                return env.get(t.name, True)
            case InferenceVarType():
                return True
            case RecordType() | EnumType():
                relevant = self._relevant[t.decl_id]
                args: list[bool] = []
                for pname, arg in zip(self._defs[t.decl_id].type_params, t.type_args, strict=True):
                    inhabited = self._type_inhabited(arg, env) if pname in relevant else True
                    if inhabited is None:
                        return None
                    args.append(inhabited)
                return self._read((t.decl_id, tuple(args)))
            case ExceptionType():
                return self._read((t.decl_id, ()))
            case ArrayType() | DictType():
                # The empty collection is always a value, regardless of the
                # element/value type — this is exactly what "guards" recursion.
                return True
            case FunctionType():
                # Function values are opaque for inhabitation; their parameter and
                # result types do not require values to exist at this site.
                return True
            case (
                TextType()
                | JsonType()
                | BoolType()
                | IntType()
                | DecimalType()
                | UnitType()
                | BottomType()
            ):
                return True
            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)


def _all_inhabited(answers: Iterable[bool | None]) -> bool | None:
    """Conjoin *answers*: one uninhabited one decides, else an unknown one does."""
    unknown = False
    for answer in answers:
        if answer is False:
            return False
        unknown = unknown or answer is None
    return None if unknown else True


def _any_inhabited(answers: Iterable[bool | None]) -> bool | None:
    """Disjoin *answers*: one inhabited one decides, else an unknown one does."""
    unknown = False
    for answer in answers:
        if answer:
            return True
        unknown = unknown or answer is None
    return None if unknown else False


def _inhabitation_references(table: TypeTable) -> dict[DeclId, tuple[DeclId, ...]]:
    """Return every declaration's references its inhabitation can read, over-approximated."""
    references: dict[DeclId, tuple[DeclId, ...]] = {}
    for decl_id, typedef in table.defs.items():
        templates = [t for _fname, t in typedef.fields]
        templates.extend(typedef.members)
        found = [ref.decl_id for t in templates for ref in nominal_references(t)]
        if typedef.base is not None:
            found.append(typedef.base)
        if typedef.abstract:
            found.extend(table.exception_children(decl_id))
        references[decl_id] = tuple(found)
    return references


def _own_key(decl_id: DeclId, typedef: TypeDef) -> InhabitationKey:
    """Return *typedef*'s key with every parameter free, hence inhabited."""
    return (decl_id, (True,) * len(typedef.type_params))


def uninhabitable_message(kind: TypeDefKind, name: str) -> str:
    """Return the diagnostic text for an uninhabitable declaration named *name*."""
    label = {"record": "Record", "enum": "Enum", "exception": "Exception"}[kind]
    return (
        f"{label} type '{name}' is uninhabitable: every value of '{name}' would be "
        "infinite. Recursion must be guarded by an enum base-case variant or "
        "an array/dict field."
    )


# ---------------------------------------------------------------------------
# Shared declaration-level "bad" fixpoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeclarationFlags:
    """Whole-table "bad" fixpoint result (see :func:`compute_declaration_flags`).

    ``flagged`` — declarations struck by the policy's evidence.
    ``relevant_params`` — for every declaration, the subset of its own type
    parameters whose instantiation can affect a concrete reference's answer:
    a parameter is relevant if it appears directly in a field (including
    nested in ``array``/``dict``/function-parameter/result position), or is
    passed to another reference's parameter that is ITSELF relevant for that
    reference's declaration — transitively. Unused ("phantom") parameters are
    therefore never relevant, matching the substitute-then-walk semantics
    this replaces: instantiating a phantom parameter with bad evidence cannot
    flag a declaration because the field template never actually mentions it.
    ``key_params`` — for every declaration, the subset of its own type
    parameters that REACH a ``dict``-key POSITION: a direct ``dict`` key, or
    the argument at ANOTHER declaration's own key parameter (transitively) —
    e.g. ``Box[K]`` with field ``d: dict[K, int]`` has key parameter ``K``,
    so ``Outer[T]`` with field ``b: Box[Wrap[T]]`` has key parameter ``T``
    too, reached through ``Wrap[T]`` filling ``Box``'s key parameter. Only a
    parameter the position can actually reach (via ``relevant_params``)
    counts — never a phantom (unused) one. Only meaningful for a policy with
    ``dict_key_ok`` set. A key position whose
    template is bad even assuming every one of the declaration's own type
    parameters satisfies the policy's key rule flags the declaration
    OUTRIGHT (e.g. ``Box[K]`` with ``d: dict[array[int], int]``, or
    ``Outer[T]`` with ``b: Box[Wrap[array[int]]]`` when the position
    evaluates ``Wrap[array[int]]`` rather than a bare parameter); only a key
    position that PASSES under that assumption contributes its own type
    parameters here, deferring the real check to whatever concrete argument
    a later reference supplies for them
    (:meth:`~agm.agl.semantics.type_table.TypeTable._nominal_satisfies_property`).
    """

    flagged: frozenset[DeclId]
    relevant_params: Mapping[DeclId, frozenset[str]]
    key_params: Mapping[DeclId, frozenset[str]]


def compute_declaration_flags(table: TypeTable, policy: LeafPolicy) -> DeclarationFlags:
    """Compute the shared declaration-level "bad" fixpoint over *table*.

    Least fixpoint: every declaration starts unflagged, and flagging only
    grows — never retracts — as evidence accumulates: a bad leaf anywhere in
    the declaration's own field/variant templates (per *policy*, at any
    depth, via :func:`field_templates`), a bad ``var`` field, a reference to
    an already-flagged declaration, or (seeded up front, when
    ``policy.non_data_bad``) host-minted origin
    (:meth:`~agm.agl.semantics.type_table.TypeTable.host_minted_declaration_ids`)
    — a host-minted declaration is an opaque non-data leaf, so it is bad
    exactly where a function/``unit`` leaf is bad, and crossable otherwise
    (e.g. ``EXTERN_CROSSABLE``, which recurses into a function's own
    parameter/result types instead of flagging it outright).
    A bare type variable is never itself bad — deferred to the concrete
    instantiation — which is what lets a self-referential declaration (a
    reference to its own, still-unflagged, identity) go unflagged by the
    cycle alone.

    Exception ``extends``: ``field_flagged`` is the exact-value/inherited-field
    fact — an inherited field flows base -> child. ``flagged`` additionally
    includes affected descendants, so a flagged descendant also flags every
    catchable ancestor (a value statically typed as the ancestor may hold the
    descendant at runtime), without flowing back down to siblings.

    A concrete handle's answer is then: its declaration's flag, OR its
    declaration reaches bad evidence for some ``type_args[i]`` whose
    parameter is in ``relevant_params`` — reproducing the substitute-then-walk
    answer exactly, without ever expanding an instantiation.
    """
    defs = table.defs
    field_flagged = set(table.host_minted_declaration_ids()) if policy.non_data_bad else set()
    flagged = set(field_flagged)
    relevant = _compute_relevant_params(defs, through_containers=True)
    key_params: dict[DeclId, set[str]] = {decl_id: set() for decl_id in defs}
    changed = True
    while changed:
        changed = False
        for decl_id, typedef in defs.items():
            own_params = frozenset(typedef.type_params)
            templates = tuple(t for _fname, t in field_templates(typedef, defs))
            own_var_field_bad = policy.var_fields_bad and (
                bool(typedef.mutable_fields)
                or (
                    typedef.kind == "enum"
                    and any(defs[member.decl_id].mutable_fields for member in typedef.members)
                )
            )
            template_bad = own_var_field_bad or any(
                _template_is_flagged(
                    t, policy, flagged, relevant, key_params, own_params, defs, table
                )
                for t in templates
            )
            inherited_field_bad = (
                typedef.kind == "exception"
                and typedef.base is not None
                and typedef.base in field_flagged
            )
            exact_bad = decl_id in field_flagged or template_bad or inherited_field_bad
            if exact_bad and decl_id not in field_flagged:
                field_flagged.add(decl_id)
                changed = True

            descendant_bad = typedef.kind == "exception" and any(
                child_id in flagged for child_id in table.exception_children(decl_id)
            )
            new_bad = exact_bad or descendant_bad
            if new_bad and decl_id not in flagged:
                flagged.add(decl_id)
                changed = True

            if policy.dict_key_ok is not None:
                gained_keys: set[str] = set()
                for t in templates:
                    gained_keys |= _template_key_params(
                        t, policy, own_params, key_params, relevant, defs, table
                    )
                if not gained_keys <= key_params[decl_id]:
                    key_params[decl_id] |= gained_keys
                    changed = True
    return DeclarationFlags(
        flagged=frozenset(flagged),
        relevant_params={decl_id: frozenset(params) for decl_id, params in relevant.items()},
        key_params={decl_id: frozenset(params) for decl_id, params in key_params.items()},
    )


def field_templates(typedef: TypeDef, defs: Mapping[DeclId, TypeDef]) -> list[tuple[str, Type]]:
    """Return every ``(name, type template)`` in *typedef*'s own body, flattened.

    Unlike inhabitation (which checks each member on its own, since only ONE
    member needs to be inhabited), both the non-data-reachability and
    reference-edge fixpoints look at every field of every variant flat: a
    function/unit anywhere is reachable from some
    value of the declaration, and a reference to another declaration matters,
    regardless of which variant carries it. Enum member fields are first
    specialized through their member handles, so referenced generic records
    retain the arguments applied by the enum. Names are carried alongside so
    a use-site diagnostic can point at the field responsible.
    """
    if typedef.kind == "enum":
        templates: list[tuple[str, Type]] = []
        for member in typedef.members:
            # A member handle always names a registered declaration: ``defs``
            # retains every declaration the table has ever registered,
            # superseded or not.
            member_def = defs[member.decl_id]
            substitutions = dict(zip(member_def.type_params, member.type_args, strict=True))
            templates.extend(
                (name, substitute(field_type, substitutions))
                for name, field_type in member_def.fields
            )
        return templates
    return list(typedef.fields)


def _key_position_ok(
    k: Type, policy: LeafPolicy, own_params: frozenset[str], table: TypeTable
) -> bool:
    """Whether *k*, filling a ``dict``-key position, passes *policy*'s own key rule.

    Assumes *own_params* (the enclosing declaration's own type parameters, in
    scope for the whole walk of one declaration's templates) satisfy the
    rule — deferred, like any bare type variable, to whatever concrete
    argument a later reference supplies for them (:func:`_template_key_params`,
    :attr:`DeclarationFlags.key_params`). Vacuously ``True`` when *policy* has
    no key rule at all (``EQ``/``HASHABLE``, whose ``dict`` fields are
    unconditionally fine/bad regardless of key shape).
    """
    return policy.dict_key_ok is None or policy.dict_key_ok(k, table, own_params)


def _template_is_flagged(
    t: Type,
    policy: LeafPolicy,
    flagged: set[DeclId],
    relevant: Mapping[DeclId, set[str]],
    key_params: Mapping[DeclId, set[str]],
    own_params: frozenset[str],
    defs: Mapping[DeclId, TypeDef],
    table: TypeTable,
) -> bool:
    """Return whether *t* is, or transitively reaches, bad evidence under *policy*.

    A ``dict``-key position — a direct key, or a type argument filling
    another declaration's own key parameter (``key_params``, transitively) —
    is checked against *policy*'s own key rule via :func:`_key_position_ok`,
    which defers a key built from *own_params* rather than flagging it
    outright; see :attr:`DeclarationFlags.key_params`.
    """

    def walk(u: Type) -> bool:
        return _template_is_flagged(
            u, policy, flagged, relevant, key_params, own_params, defs, table
        )

    match t:
        case UnitType():
            return policy.non_data_bad
        case FunctionType():
            if policy.non_data_bad:
                return True
            return any(walk(p) for p in t.params) or walk(t.result)
        case ArrayType():
            if not policy.recurse_containers:
                return True
            return walk(t.elem)
        case DictType():
            if not policy.recurse_containers:
                return True
            if not _key_position_ok(t.key, policy, own_params, table):
                return True
            return walk(t.key) or walk(t.value)
        case ExceptionType():
            return t.decl_id in flagged
        case RecordType() | EnumType():
            decl_id = t.decl_id
            if decl_id in flagged:
                return True
            target = defs[decl_id]
            own_relevant = relevant[decl_id]
            target_key = key_params[decl_id]
            for pname, arg in zip(target.type_params, t.type_args):
                if pname in target_key and not _key_position_ok(arg, policy, own_params, table):
                    return True
            return any(
                walk(arg)
                for pname, arg in zip(target.type_params, t.type_args)
                if pname in own_relevant
            )
        case (
            TextType()
            | JsonType()
            | BoolType()
            | IntType()
            | DecimalType()
            | BottomType()
            | TypeVarType()
            | InferenceVarType()
        ):
            return False
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _template_relevant_params(
    t: Type,
    own_params: frozenset[str],
    relevant: Mapping[DeclId, set[str]],
    defs: Mapping[DeclId, TypeDef],
    *,
    through_containers: bool,
) -> set[str]:
    """Return *own_params* occurring in *t* where they can reach a field.

    A reference passes an occurrence on only at a relevant parameter of its
    target; ``array``/``dict``/function shapes pass it on only when
    *through_containers*.
    """

    def walk(u: Type) -> set[str]:
        return _template_relevant_params(
            u, own_params, relevant, defs, through_containers=through_containers
        )

    match t:
        case TypeVarType():
            return {t.name} if t.name in own_params else set()
        case InferenceVarType():
            return set()
        case ArrayType() | DictType() | FunctionType() if not through_containers:
            return set()
        case ArrayType():
            return walk(t.elem)
        case DictType():
            return walk(t.key) | walk(t.value)
        case FunctionType():
            result: set[str] = set()
            for p in t.params:
                result |= walk(p)
            return result | walk(t.result)
        case RecordType() | EnumType():
            own_relevant = relevant[t.decl_id]
            result = set()
            for pname, arg in zip(defs[t.decl_id].type_params, t.type_args):
                if pname in own_relevant:
                    result |= walk(arg)
            return result
        case (
            ExceptionType()
            | UnitType()
            | TextType()
            | JsonType()
            | BoolType()
            | IntType()
            | DecimalType()
            | BottomType()
            | InferenceVarType()
        ):
            return set()
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _template_key_params(
    t: Type,
    policy: LeafPolicy,
    own_params: frozenset[str],
    key_params: Mapping[DeclId, set[str]],
    relevant: Mapping[DeclId, set[str]],
    defs: Mapping[DeclId, TypeDef],
    table: TypeTable,
) -> set[str]:
    """Return the subset of *own_params* occurring in *t* in a ``dict``-key position that DEFERS.

    A direct ``dict`` key, and a type argument filling another declaration's
    own key parameter (``key_params`` — the same growing fixpoint fact this
    computes, consulted for whatever has already stabilized for the
    REFERENCED declaration), transitively. A position only contributes the
    own parameters it can actually REACH — a bare parameter, or (via
    :func:`_template_relevant_params`) a parameter reaching a field through a
    nominal argument's own relevant positions, never a phantom one — and only
    when it PASSES :func:`_key_position_ok`; a position that fails flags the
    whole declaration outright instead (:func:`_template_is_flagged`), so
    there is nothing left to defer there. Recurses into every constructor to
    find a key position nested arbitrarily deep; a ``dict``'s own key never
    recurses further (``dict`` is excluded from a key's own shape by
    ``Hashable`` itself).
    """
    match t:
        case TypeVarType() | InferenceVarType():
            return set()
        case ArrayType():
            return _template_key_params(
                t.elem, policy, own_params, key_params, relevant, defs, table
            )
        case DictType():
            direct: set[str] = (
                _template_relevant_params(
                    t.key, own_params, relevant, defs, through_containers=True
                )
                if _key_position_ok(t.key, policy, own_params, table)
                else set()
            )
            return direct | _template_key_params(
                t.value, policy, own_params, key_params, relevant, defs, table
            )
        case FunctionType():
            result: set[str] = set()
            for p in t.params:
                result |= _template_key_params(
                    p, policy, own_params, key_params, relevant, defs, table
                )
            result |= _template_key_params(
                t.result, policy, own_params, key_params, relevant, defs, table
            )
            return result
        case RecordType() | EnumType():
            decl_id = t.decl_id
            target = defs[decl_id]
            target_key = key_params[decl_id]
            result = set()
            for pname, arg in zip(target.type_params, t.type_args):
                if pname in target_key and _key_position_ok(arg, policy, own_params, table):
                    result |= _template_relevant_params(
                        arg, own_params, relevant, defs, through_containers=True
                    )
                result |= _template_key_params(
                    arg, policy, own_params, key_params, relevant, defs, table
                )
            return result
        case (
            ExceptionType()
            | UnitType()
            | TextType()
            | JsonType()
            | BoolType()
            | IntType()
            | DecimalType()
            | BottomType()
        ):
            return set()
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


# ---------------------------------------------------------------------------
# Finiteness (instantiation-closure) analysis
# ---------------------------------------------------------------------------

# A parameter of a declaration, identified by the declaration's identity and
# the parameter's own name — a node in the small per-SCC parameter-dependency
# graph.
ParamKey = tuple[DeclId, str]


@dataclass(frozen=True, slots=True)
class _RefEdge:
    """One nominal reference found in a declaration's own body.

    ``target`` is the referenced declaration; ``arg_templates`` are the
    reference's argument templates, positionally aligned with the target's
    OWN type parameters (empty for a reference to a non-generic declaration,
    e.g. an exception or its ``extends`` base).
    """

    target: DeclId
    arg_templates: tuple[Type, ...]


@dataclass(frozen=True, slots=True)
class FiniteClosure:
    """Whole-table finiteness fixpoint result (see :func:`compute_finite_closure`).

    ``infinite`` — declarations whose instantiation closure is infinite (a
    growing polymorphic-recursion cycle).  Every other registered declaration
    has a finite closure.
    ``successors`` — the schema-relevant declaration reference graph (which
    declarations a declaration's body mentions through fields and non-phantom
    type arguments), used by
    :meth:`~agm.agl.semantics.type_table.TypeTable.has_finite_schema` to
    extend a concrete type's own reachable declarations without re-deriving
    the reference graph.
    ``relevant_params`` — for each declaration, the subset of its own type
    parameters whose concrete instantiation can affect the reachable schema.
    Phantom parameters are intentionally absent.
    """

    infinite: frozenset[DeclId]
    successors: Mapping[DeclId, frozenset[DeclId]]
    relevant_params: Mapping[DeclId, frozenset[str]]


def compute_finite_closure(table: TypeTable) -> FiniteClosure:
    """Compute the declaration-level finiteness fixpoint over *table*.

    Per the module docstring: build the declaration reference graph, find its
    SCCs, and within each SCC build the parameter-dependency graph (edge
    ``q -> p`` whenever a reference from a declaration A to a declaration B —
    both in the SCC — has, in its argument template for B's parameter ``p``,
    an occurrence of A's parameter ``q``; the edge is growing when ``q``
    occurs as a proper subterm rather than being the WHOLE argument
    template). An SCC's closure is infinite iff that parameter graph has a
    cycle containing at least one growing edge.

    Permutation cycles (``Swap[B, A]`` referenced from ``Swap[A, B]``'s body)
    and argument-constant references (``R[int]`` referenced from ``R[T]``'s
    body) never contribute a growing edge, so they stay finite. A
    non-generic declaration contributes no parameter nodes at all, so a
    purely-structural recursive cycle (``Tree``, mutually recursive
    records/enums, recursive exceptions) is always finite — matching the
    inhabitation-checked recursion that is already unconditionally legal.
    """
    defs = table.defs
    relevant = _compute_relevant_params(defs, through_containers=True)
    edges = _reference_edges(defs, relevant)
    successors: dict[DeclId, frozenset[DeclId]] = {
        decl_id: frozenset(edge.target for edge in refs) for decl_id, refs in edges.items()
    }
    adjacency: dict[DeclId, tuple[DeclId, ...]] = {
        decl_id: tuple(targets) for decl_id, targets in successors.items()
    }
    components = sccs(adjacency, key=lambda decl_id: decl_id_sort_key(defs, decl_id))
    infinite: set[DeclId] = set()
    for component in components:
        members = frozenset(component)
        if _scc_has_growing_cycle(members, edges, defs, relevant):
            infinite.update(members)
    return FiniteClosure(
        infinite=frozenset(infinite),
        successors=successors,
        relevant_params={decl_id: frozenset(params) for decl_id, params in relevant.items()},
    )


def nominal_references(t: Type) -> Iterator[RecordType | EnumType | ExceptionType]:
    """Yield every nominal reference occurring anywhere in *t*, including nested.

    Recurses into ``array``/``dict``/function shapes and, for a nominal
    reference itself, into its OWN argument templates too — a reference's
    arguments may themselves nest further nominal references (e.g.
    ``Perfect[Wrapper[T]]``). ``t`` is always a finite tree (nominal
    references are handles, not expanded bodies), so this always terminates.
    """
    match t:
        case RecordType() | EnumType():
            yield t
            for arg in t.type_args:
                yield from nominal_references(arg)
        case ExceptionType():
            yield t
        case ArrayType(elem=elem):
            yield from nominal_references(elem)
        case DictType(key=key, value=value):
            yield from nominal_references(key)
            yield from nominal_references(value)
        case FunctionType(params=params, result=result):
            for p in params:
                yield from nominal_references(p)
            yield from nominal_references(result)
        case (
            TextType()
            | JsonType()
            | BoolType()
            | IntType()
            | DecimalType()
            | UnitType()
            | BottomType()
            | TypeVarType()
            | InferenceVarType()
        ):
            return
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def nominal_references_for_schema(
    t: Type,
    defs: Mapping[DeclId, TypeDef],
    relevant_params: Mapping[DeclId, frozenset[str]],
) -> Iterator[RecordType | EnumType | ExceptionType]:
    """Yield nominal references that can affect *t*'s finite schema.

    A record/enum handle is always relevant itself, but its type arguments are
    only relevant when the corresponding declaration parameter is used by the
    declaration's schema. This keeps phantom arguments from pulling unrelated
    infinite declarations into a schema boundary.
    """
    match t:
        case RecordType() | EnumType():
            yield t
            typedef = defs[t.decl_id]
            relevant = relevant_params[t.decl_id]
            for pname, arg in zip(typedef.type_params, t.type_args):
                if pname in relevant:
                    yield from nominal_references_for_schema(arg, defs, relevant_params)
        case ExceptionType():
            yield t
        case ArrayType(elem=elem):
            yield from nominal_references_for_schema(elem, defs, relevant_params)
        case DictType(key=key, value=value):
            yield from nominal_references_for_schema(key, defs, relevant_params)
            yield from nominal_references_for_schema(value, defs, relevant_params)
        case FunctionType(params=params, result=result):
            for p in params:
                yield from nominal_references_for_schema(p, defs, relevant_params)
            yield from nominal_references_for_schema(result, defs, relevant_params)
        case (
            TextType()
            | JsonType()
            | BoolType()
            | IntType()
            | DecimalType()
            | UnitType()
            | BottomType()
            | TypeVarType()
            | InferenceVarType()
        ):
            return
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _compute_relevant_params(
    defs: Mapping[DeclId, TypeDef], *, through_containers: bool
) -> dict[DeclId, set[str]]:
    """Return every declaration's parameters that can reach one of its fields.

    With *through_containers* these are the parameters whose instantiation can
    affect schema reachability; without it, inhabitation.
    """
    relevant: dict[DeclId, set[str]] = {decl_id: set() for decl_id in defs}
    changed = True
    while changed:
        changed = False
        for decl_id, typedef in defs.items():
            own_params = frozenset(typedef.type_params)
            gained: set[str] = set()
            for _fname, template in field_templates(typedef, defs):
                gained |= _template_relevant_params(
                    template, own_params, relevant, defs, through_containers=through_containers
                )
            if not gained <= relevant[decl_id]:
                relevant[decl_id] |= gained
                changed = True
    return relevant


def _reference_edges(
    defs: Mapping[DeclId, TypeDef],
    relevant_params: Mapping[DeclId, set[str]],
) -> dict[DeclId, tuple[_RefEdge, ...]]:
    """Return every declaration's schema-relevant outgoing reference edge."""
    frozen_relevant = {decl_id: frozenset(params) for decl_id, params in relevant_params.items()}
    result: dict[DeclId, tuple[_RefEdge, ...]] = {}
    for decl_id, typedef in defs.items():
        found: list[_RefEdge] = []
        for _fname, template in field_templates(typedef, defs):
            for ref in nominal_references_for_schema(template, defs, frozen_relevant):
                arg_templates = ref.type_args if isinstance(ref, (RecordType, EnumType)) else ()
                found.append(_RefEdge(target=ref.decl_id, arg_templates=arg_templates))
        if typedef.kind == "exception" and typedef.base is not None:
            found.append(_RefEdge(target=typedef.base, arg_templates=()))
        result[decl_id] = tuple(found)
    return result


def _scc_has_growing_cycle(
    members: frozenset[DeclId],
    edges: Mapping[DeclId, tuple[_RefEdge, ...]],
    defs: Mapping[DeclId, TypeDef],
    relevant_params: Mapping[DeclId, set[str]],
) -> bool:
    """Return ``True`` if *members*'s parameter-dependency graph has a growing cycle."""
    adjacency: dict[ParamKey, list[ParamKey]] = {}
    growing_edges: set[tuple[ParamKey, ParamKey]] = set()
    for source_id in members:
        source_def = defs[source_id]
        for ref_edge in edges[source_id]:
            target_id = ref_edge.target
            if target_id not in members:
                continue
            target_def = defs[target_id]
            target_relevant = relevant_params[target_id]
            for param_name, arg_template in zip(target_def.type_params, ref_edge.arg_templates):
                if param_name not in target_relevant:
                    continue
                occurrences = _param_occurrences(
                    arg_template,
                    growing=False,
                    defs=defs,
                    relevant_params=relevant_params,
                )
                for source_param, growing in occurrences.items():
                    if source_param not in source_def.type_params:
                        continue
                    src: ParamKey = (source_id, source_param)
                    dst: ParamKey = (target_id, param_name)
                    adjacency.setdefault(src, []).append(dst)
                    if growing:
                        growing_edges.add((src, dst))
    if not adjacency:
        return False
    param_components = sccs(adjacency, key=lambda n: (*decl_id_sort_key(defs, n[0]), n[1]))
    for component in param_components:
        # A growing self-loop within a singleton SCC ``{node}`` is the pair
        # ``(node, node)``; the same membership test catches it, so singletons
        # need no special case.
        comp_set = frozenset(component)
        if any(src in comp_set and dst in comp_set for src, dst in growing_edges):
            return True
    return False


def _param_occurrences(
    t: Type,
    *,
    growing: bool,
    defs: Mapping[DeclId, TypeDef],
    relevant_params: Mapping[DeclId, set[str]],
) -> dict[str, bool]:
    """Return type-variable occurrences in *t* that affect schema identity.

    An occurrence is growing when it is a PROPER SUBTERM of the top-level
    template passed in — i.e. anywhere except when ``t`` itself, at the top
    level, IS the bare type variable. Recursive descent into a nominal
    reference follows only schema-relevant parameters of that referenced
    declaration, so phantom arguments do not create spurious growth edges.
    When a variable occurs more than once, growing wins (only one growing path
    is needed to make the whole reference growing).
    """
    match t:
        case TypeVarType(name=name):
            return {name: growing}
        case InferenceVarType():
            return {}
        case ArrayType(elem=elem):
            return _param_occurrences(
                elem, growing=True, defs=defs, relevant_params=relevant_params
            )
        case DictType(key=key, value=value):
            return _merge_growing(
                _param_occurrences(key, growing=True, defs=defs, relevant_params=relevant_params),
                _param_occurrences(value, growing=True, defs=defs, relevant_params=relevant_params),
            )
        case FunctionType(params=params, result=result):
            merged: dict[str, bool] = {}
            for p in params:
                merged = _merge_growing(
                    merged,
                    _param_occurrences(p, growing=True, defs=defs, relevant_params=relevant_params),
                )
            merged = _merge_growing(
                merged,
                _param_occurrences(
                    result, growing=True, defs=defs, relevant_params=relevant_params
                ),
            )
            return merged
        case RecordType() | EnumType():
            target = defs[t.decl_id]
            target_relevant = relevant_params[t.decl_id]
            relevant_args = tuple(
                arg
                for pname, arg in zip(target.type_params, t.type_args)
                if pname in target_relevant
            )
            merged = {}
            for arg in relevant_args:
                merged = _merge_growing(
                    merged,
                    _param_occurrences(
                        arg, growing=True, defs=defs, relevant_params=relevant_params
                    ),
                )
            return merged
        case (
            ExceptionType()
            | UnitType()
            | TextType()
            | JsonType()
            | BoolType()
            | IntType()
            | DecimalType()
            | BottomType()
            | InferenceVarType()
        ):
            return {}
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _merge_growing(a: dict[str, bool], b: dict[str, bool]) -> dict[str, bool]:
    result = dict(a)
    for name, growing in b.items():
        result[name] = result.get(name, False) or growing
    return result
