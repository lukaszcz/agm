"""Declaration-level whole-type analyses over a shared ``TypeTable``.

Nominal types are handles (``semantics.types``) backed by ``TypeDef``
templates in a ``TypeTable`` (``semantics.type_table``); a declaration may
reference itself or another declaration directly or indirectly, so any
whole-type question ("does every value of this type terminate?", "does this
type support ``=``?") must be answered over the finite *declaration graph*
rather than by walking an individual type tree, which may be cyclic.  Both
analyses below share that shape: start from a conservative default, and grow
a set of facts to a least fixpoint by repeatedly re-examining every
declaration's own field/variant templates until nothing changes. Because
there are finitely many declarations, this always terminates, and because
each fact only ever flips from "not yet established" to "established" (never
back), the fixpoint is independent of iteration order.

Inhabitation
------------
A declaration is inhabited iff it has at least one finite value. Recursion
through a ``list``/``dict`` field is always fine (the empty collection is a
value regardless of the element type); recursion through a record/exception
field or every variant of an enum is fine only if some path bottoms out
without needing another value of the same (or a mutually recursive)
declaration. :func:`compute_uninhabited` returns the declarations that never
reach that bottom.

Equality capability
--------------------
``=``/``!=`` are undefined for function, agent, and unit values (and for
anything that transitively contains one). :func:`compute_equality_capabilities`
replaces a walk of each concrete instantiation's substituted fields (which
cannot terminate once field types may reference cyclic declarations) with two
per-declaration fixpoint facts: whether the declaration's body is
unconditionally non-comparable, and which of its own type parameters actually
affect comparability of a concrete instantiation ("equality-relevant"
parameters) — see :func:`compute_equality_capabilities` for the full
definition and :meth:`~agm.agl.semantics.type_table.TypeTable.has_no_value_equality`
for how a concrete handle's answer is derived from them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import assert_never

from agm.agl.modules.ids import ModuleId
from agm.agl.semantics.type_table import TypeDef, TypeDefKind, TypeTable
from agm.agl.semantics.types import (
    AgentType,
    BoolType,
    BottomType,
    DecimalType,
    DictType,
    EnumType,
    ExceptionType,
    FunctionType,
    IntType,
    JsonType,
    ListType,
    RecordType,
    TextType,
    Type,
    TypeVarType,
    UnitType,
)

# A declaration's identity in the shared type table.
DeclKey = tuple[ModuleId, str]


# ---------------------------------------------------------------------------
# Inhabitation
# ---------------------------------------------------------------------------


def compute_uninhabited(table: TypeTable) -> frozenset[DeclKey]:
    """Return every registered declaration key that has no finite value.

    Least fixpoint over the whole table: every declaration starts
    uninhabited, and is promoted to inhabited as soon as its own body
    (record/exception: every field; enum: every field of some variant) is
    provably inhabited given the CURRENT set of known-inhabited declarations.
    Iterates to a fixpoint (bounded by the number of declarations) before
    returning the keys that never got promoted.

    A handle's ``type_args`` play no part here: a type variable counts as
    inhabited regardless of what it is eventually instantiated with, so a
    generic declaration's inhabitation is a property of its template alone,
    independent of any particular instantiation. This is exact for the
    common case (a type parameter used directly, or nested in containers,
    behaves the same for inhabitation purposes no matter the argument) and,
    for a self-referential argument (a declaration applying a generic type to
    itself), is a deliberate, documented simplification: it only ever ERRS
    toward accepting a declaration, never toward rejecting an inhabited one,
    so it never produces a false "uninhabitable" diagnostic.
    """
    defs = _all_defs(table)
    inhabited: set[DeclKey] = set()
    changed = True
    while changed:
        changed = False
        for key, typedef in defs.items():
            if key in inhabited:
                continue
            if _decl_inhabited(typedef, inhabited):
                inhabited.add(key)
                changed = True
    return frozenset(defs) - inhabited


def uninhabitable_message(kind: TypeDefKind, name: str) -> str:
    """Return the diagnostic text for an uninhabitable declaration named *name*."""
    label = {"record": "Record", "enum": "Enum", "exception": "Exception"}[kind]
    return (
        f"{label} type '{name}' is uninhabitable: every value of '{name}' would be "
        "infinite. Recursion must be guarded by an enum base-case variant or a "
        "list/dict field."
    )


def _decl_inhabited(typedef: TypeDef, inhabited: set[DeclKey]) -> bool:
    if typedef.kind == "enum":
        return any(
            all(_template_inhabited(t, inhabited) for _fname, t in vfields)
            for _vname, vfields in typedef.variants
        )
    own_ok = all(_template_inhabited(t, inhabited) for _fname, t in typedef.fields)
    if typedef.kind == "exception" and typedef.base is not None:
        # Model the ``extends`` link as a conjunct rather than flattening the
        # base's fields in: equivalent (the base's own conjunct already
        # accounts for ITS base, transitively), and it makes an ``extends``
        # cycle fall out of the SAME fixpoint as ordinary field recursion —
        # neither side of the cycle ever gets independent evidence, so both
        # stay uninhabited rather than needing a separate cycle check.
        return own_ok and typedef.base in inhabited
    return own_ok


def _template_inhabited(t: Type, inhabited: set[DeclKey]) -> bool:
    match t:
        case RecordType() | EnumType() | ExceptionType():
            return (t.module_id, t.name) in inhabited
        case ListType() | DictType():
            # The empty collection is always a value, regardless of the
            # element/value type — this is exactly what "guards" recursion.
            return True
        case (
            TextType()
            | JsonType()
            | BoolType()
            | IntType()
            | DecimalType()
            | UnitType()
            | AgentType()
            | FunctionType()
            | BottomType()
            | TypeVarType()
        ):
            return True
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


# ---------------------------------------------------------------------------
# Equality capability flags
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EqualityCapabilities:
    """Whole-table equality-capability fixpoint result.

    ``no_equality`` — declarations that are unconditionally non-comparable
    (their body contains a function/agent/unit type, or reaches a
    declaration that does, outside of a type-variable position).
    ``relevant_params`` — for every declaration, the subset of its own type
    parameters whose instantiation can affect comparability (see
    :func:`compute_equality_capabilities`).
    """

    no_equality: frozenset[DeclKey]
    relevant_params: Mapping[DeclKey, frozenset[str]]


def compute_equality_capabilities(table: TypeTable) -> EqualityCapabilities:
    """Compute the declaration-level equality-capability fixpoint over *table*.

    Two facts are grown together to a least fixpoint, per declaration:

    - ``no_equality`` (an unconditional, argument-independent fact): true iff
      some field/variant-field template (for an exception, its OWN fields;
      inheritance is folded in as an extra disjunct on the base's already-
      computed flag, exactly like :func:`compute_uninhabited` models
      ``extends`` as a conjunct) contains a function/agent/unit type, or
      references a declaration whose own ``no_equality`` is already true —
      at any depth, but never through a bare type-variable position (a
      parameter standing for "whatever the caller instantiates" is not
      itself a problem).
    - ``relevant_params``: the subset of a declaration's OWN type parameters
      whose concrete instantiation can flip a reference to it from
      comparable to not. A parameter is relevant if it appears directly in a
      field (including nested in ``list``/``dict``/function-parameter/
      result position), or is passed to another reference's parameter that
      is ITSELF relevant for that reference's declaration — transitively.
      Unused ("phantom") parameters are therefore never relevant, matching
      the substitute-then-walk semantics this replaces: instantiating a
      phantom parameter with a non-comparable type cannot poison equality
      because the field template never actually mentions it.

    A concrete handle's answer
    (:meth:`~agm.agl.semantics.type_table.TypeTable.has_no_value_equality`) is
    then: its declaration's ``no_equality`` flag, OR its declaration is
    non-comparable for some ``type_args[i]`` whose parameter is in
    ``relevant_params`` — reproducing today's substitute-then-walk answer
    exactly, without ever expanding an instantiation.
    """
    defs = _all_defs(table)
    no_eq: set[DeclKey] = set()
    relevant: dict[DeclKey, set[str]] = {key: set() for key in defs}
    changed = True
    while changed:
        changed = False
        for key, typedef in defs.items():
            own_params = frozenset(typedef.type_params)
            templates = tuple(_own_equality_templates(typedef))
            bad = key in no_eq or any(
                _template_no_eq(t, no_eq, relevant, defs) for t in templates
            )
            if typedef.kind == "exception" and typedef.base is not None:
                bad = bad or typedef.base in no_eq
            if bad and key not in no_eq:
                no_eq.add(key)
                changed = True
            gained: set[str] = set()
            for t in templates:
                gained |= _template_relevant_params(t, own_params, relevant, defs)
            if not gained <= relevant[key]:
                relevant[key] |= gained
                changed = True
    return EqualityCapabilities(
        no_equality=frozenset(no_eq),
        relevant_params={key: frozenset(params) for key, params in relevant.items()},
    )


def _own_equality_templates(typedef: TypeDef) -> list[Type]:
    """Return every field-type template in *typedef*'s own body, flattened.

    Unlike :func:`_decl_inhabited`'s enum handling (which groups fields by
    variant, since only ONE variant needs to be fully inhabited), equality
    looks at every field of every variant flat: ANY function/agent/unit
    anywhere in the declaration makes every value of it non-comparable,
    regardless of which variant a given value happens to be.
    """
    if typedef.kind == "enum":
        return [ftype for _vname, vfields in typedef.variants for _fname, ftype in vfields]
    return [ftype for _fname, ftype in typedef.fields]


def _template_no_eq(
    t: Type,
    no_eq: set[DeclKey],
    relevant: Mapping[DeclKey, set[str]],
    defs: Mapping[DeclKey, TypeDef],
) -> bool:
    match t:
        case FunctionType() | AgentType() | UnitType():
            return True
        case ListType():
            return _template_no_eq(t.elem, no_eq, relevant, defs)
        case DictType():
            return _template_no_eq(t.value, no_eq, relevant, defs)
        case ExceptionType():
            return (t.module_id, t.name) in no_eq
        case RecordType() | EnumType():
            key = (t.module_id, t.name)
            if key in no_eq:
                return True
            target = defs.get(key)
            if target is None:
                return False
            own_relevant = relevant.get(key, set())
            return any(
                _template_no_eq(arg, no_eq, relevant, defs)
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
        ):
            return False
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _template_relevant_params(
    t: Type,
    own_params: frozenset[str],
    relevant: Mapping[DeclKey, set[str]],
    defs: Mapping[DeclKey, TypeDef],
) -> set[str]:
    match t:
        case TypeVarType():
            return {t.name} if t.name in own_params else set()
        case ListType():
            return _template_relevant_params(t.elem, own_params, relevant, defs)
        case DictType():
            return _template_relevant_params(t.value, own_params, relevant, defs)
        case FunctionType():
            result: set[str] = set()
            for p in t.params:
                result |= _template_relevant_params(p, own_params, relevant, defs)
            result |= _template_relevant_params(t.result, own_params, relevant, defs)
            return result
        case RecordType() | EnumType():
            key = (t.module_id, t.name)
            target = defs.get(key)
            if target is None:
                return set()
            own_relevant = relevant.get(key, set())
            result = set()
            for pname, arg in zip(target.type_params, t.type_args):
                if pname in own_relevant:
                    result |= _template_relevant_params(arg, own_params, relevant, defs)
            return result
        case (
            ExceptionType()
            | AgentType()
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
# Shared helpers
# ---------------------------------------------------------------------------


def _all_defs(table: TypeTable) -> dict[DeclKey, TypeDef]:
    """Return every registered ``TypeDef`` in *table*, keyed by its identity."""
    return {(typedef.module_id, typedef.name): typedef for typedef in table.entries()}
