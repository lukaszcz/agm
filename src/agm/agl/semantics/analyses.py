"""Declaration-level whole-type analyses over a shared ``TypeTable``.

Nominal types are handles (``semantics.types``) backed by ``TypeDef``
templates in a ``TypeTable`` (``semantics.type_table``); a declaration may
reference itself or another declaration directly or indirectly, so any
whole-type question ("does every value of this type terminate?", "can a
non-data value hide inside this type?") must be answered over the finite
*declaration graph* rather than by walking an individual type tree, which may
be cyclic.  Both analyses below share that shape: start from a conservative
default, and grow
a set of facts to a least fixpoint by repeatedly re-examining every
declaration's own field/variant templates until nothing changes. Because
there are finitely many declarations, this always terminates, and because
each fact only ever flips from "not yet established" to "established" (never
back), the fixpoint is independent of iteration order.

Inhabitation
------------
A declaration is inhabited iff it has at least one finite value. Recursion
through an ``array``/``dict`` field is always fine (the empty collection is a
value regardless of the element type); recursion through a record/exception
field or every variant of an enum is fine only if some path bottoms out
without needing another value of the same (or a mutually recursive)
declaration. :func:`compute_uninhabited` returns the declarations that never
reach that bottom.

Non-data reachability
---------------------
The *non-data* types are exactly ``unit`` and function types. Two
independent language rules turn on
whether one of them is reachable from a type — ``=``/``!=`` are undefined for
such a value (and for anything that transitively contains one), and there is
no JSON representation for one either — so the underlying fact is computed
once, here. :func:`compute_non_data_reachability` replaces a walk of each
concrete instantiation's substituted fields (which cannot terminate once
field types may reference cyclic declarations) with two per-declaration
fixpoint facts: whether the declaration's body unconditionally reaches a
non-data type, and which of its own type parameters actually affect the
answer for a concrete instantiation ("relevant" parameters) — see
:func:`compute_non_data_reachability` for the full definition and
:meth:`~agm.agl.semantics.type_table.TypeTable.nominal_reaches_non_data` for
how a concrete handle's answer is derived from them.

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

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import assert_never

from agm.agl.semantics.type_table import (
    DeclId,
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

TypeEnv = Mapping[str, Type]
InstantiationKey = tuple[DeclId, tuple[Type, ...]]


# ---------------------------------------------------------------------------
# Inhabitation
# ---------------------------------------------------------------------------


def compute_uninhabited(table: TypeTable) -> frozenset[DeclId]:
    """Return every registered declaration identity that has no finite value.

    Least fixpoint over the whole table: every declaration starts
    uninhabited, and is promoted to inhabited as soon as its own body
    (record/exception: every field; enum: every field of some variant) is
    provably inhabited given the CURRENT set of known-inhabited declarations.
    Iterates to a fixpoint (bounded by the number of declarations) before
    returning the keys that never got promoted.

    Generic references are checked at the concrete argument templates used at
    the reference site. A free type variable in a declaration body is treated
    as inhabited, but once a generic wrapper is applied to an uninhabited
    recursive type (for example ``Box[Bad]`` where ``Box[T]`` stores a ``T``),
    the wrapper's body is evaluated with that argument substituted, so it does
    not hide unguarded recursion.
    """
    defs = table.defs
    inhabited: set[DeclId] = set()
    changed = True
    while changed:
        changed = False
        for decl_id, typedef in defs.items():
            if decl_id in inhabited:
                continue
            if _InhabitationSolver(defs, inhabited).decl_inhabited(typedef):
                inhabited.add(decl_id)
                changed = True
    member_ids = {
        member.decl_id
        for typedef in defs.values()
        if typedef.kind == "enum"
        for member in typedef.members
        if isinstance(member, RecordType)
    }
    return frozenset(defs) - inhabited - member_ids


class _InhabitationSolver:
    """One declaration's inhabitation walk over a fixed ``inhabited`` set.

    ``defs`` and ``inhabited`` are constant for the walk's lifetime, so the
    answer for a given (instantiation, recursion stack) pair is stable and is
    memoized. Without that memo a declaration whose fields reference the same
    type more than once is re-walked once per path through the type graph,
    which is exponential in the graph's depth for a diamond-shaped one.
    """

    __slots__ = ("_defs", "_inhabited", "_memo")

    def __init__(self, defs: Mapping[DeclId, TypeDef], inhabited: set[DeclId]) -> None:
        self._defs = defs
        self._inhabited = inhabited
        self._memo: dict[tuple[InstantiationKey, frozenset[InstantiationKey]], bool] = {}

    def decl_inhabited(self, typedef: TypeDef) -> bool:
        """Return whether *typedef*'s own body is inhabited."""
        args = tuple(TypeVarType(param) for param in typedef.type_params)
        return self._body_inhabited(typedef, {}, stack=frozenset({(typedef.decl_node_id, args)}))

    def _body_inhabited(
        self, typedef: TypeDef, env: TypeEnv, *, stack: frozenset[InstantiationKey]
    ) -> bool:
        if typedef.kind == "enum":
            return any(
                self._template_inhabited(member, env, stack=stack) for member in typedef.members
            )
        if typedef.kind == "exception":
            return self._exception_decl_inhabited(typedef, env, stack=stack)
        return all(self._template_inhabited(t, env, stack=stack) for _fname, t in typedef.fields)

    def _exception_decl_inhabited(
        self, typedef: TypeDef, env: TypeEnv, *, stack: frozenset[InstantiationKey]
    ) -> bool:
        decl_id = typedef.decl_node_id
        if typedef.abstract:
            return any(
                child.kind == "exception" and child.base == decl_id and child_id in self._inhabited
                for child_id, child in self._defs.items()
            )
        return self._exception_fields_inhabited(
            typedef, env, stack=stack, extends_stack=frozenset({decl_id})
        )

    def _exception_fields_inhabited(
        self,
        typedef: TypeDef,
        env: TypeEnv,
        *,
        stack: frozenset[InstantiationKey],
        extends_stack: frozenset[DeclId],
    ) -> bool:
        own_ok = all(self._template_inhabited(t, env, stack=stack) for _fname, t in typedef.fields)
        if not own_ok:
            return False
        if typedef.base is None:
            return True
        if typedef.base in extends_stack:
            return False
        base_def = self._defs.get(typedef.base)
        if base_def is None or base_def.kind != "exception":
            return False
        return self._exception_fields_inhabited(
            base_def,
            env,
            stack=stack,
            extends_stack=extends_stack | frozenset({typedef.base}),
        )

    def _template_inhabited(
        self, t: Type, env: TypeEnv, *, stack: frozenset[InstantiationKey]
    ) -> bool:
        match t:
            case TypeVarType():
                replacement = env.get(t.name)
                if replacement is None or replacement == t:
                    return True
                return self._template_inhabited(replacement, env, stack=stack)
            case InferenceVarType():
                return True
            case RecordType() | EnumType():
                decl_id = t.decl_id
                if any(stack_id == decl_id for stack_id, _args in stack):
                    return False
                target = self._defs.get(decl_id)
                if target is None:
                    return decl_id in self._inhabited
                args = tuple(substitute(arg, env) for arg in t.type_args)
                instantiation = (decl_id, args)
                memo_key = (instantiation, stack)
                cached = self._memo.get(memo_key)
                if cached is not None:
                    return cached
                result = self._body_inhabited(
                    target,
                    dict(zip(target.type_params, args)),
                    stack=stack | frozenset({instantiation}),
                )
                self._memo[memo_key] = result
                return result
            case ExceptionType():
                decl_id = t.decl_id
                if any(stack_id == decl_id for stack_id, _args in stack):
                    return False
                return decl_id in self._inhabited
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


def uninhabitable_message(kind: TypeDefKind, name: str) -> str:
    """Return the diagnostic text for an uninhabitable declaration named *name*."""
    label = {"record": "Record", "enum": "Enum", "exception": "Exception"}[kind]
    return (
        f"{label} type '{name}' is uninhabitable: every value of '{name}' would be "
        "infinite. Recursion must be guarded by an enum base-case variant or "
        "an array/dict field."
    )


# ---------------------------------------------------------------------------
# Non-data reachability flags
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NonDataReachability:
    """Whole-table non-data-reachability fixpoint result.

    ``reaches_non_data`` — declarations that unconditionally reach a non-data
    type (their body contains a function/unit type, or reaches a
    declaration that does, outside of a type-variable position).
    ``relevant_params`` — for every declaration, the subset of its own type
    parameters whose instantiation can affect that answer (see
    :func:`compute_non_data_reachability`).
    """

    reaches_non_data: frozenset[DeclId]
    relevant_params: Mapping[DeclId, frozenset[str]]


def compute_non_data_reachability(table: TypeTable) -> NonDataReachability:
    """Compute the declaration-level non-data-reachability fixpoint over *table*.

    Two facts are grown together to a least fixpoint, per declaration:

    - ``reaches_non_data`` (an unconditional, argument-independent fact): true
      iff some field/variant-field template contains a function/unit
      type, or references a declaration whose own ``reaches_non_data`` is
      already true — at any depth, but never through a bare type-variable
      position (a parameter standing for "whatever the caller instantiates" is
      not itself a problem). For exceptions this also accounts for subtyping:
      inherited field problems flow from base to child, while an affected
      child also affects each catchable ancestor, because a value statically
      typed as that ancestor may hold the child at runtime.
    - ``relevant_params``: the subset of a declaration's OWN type parameters
      whose concrete instantiation can flip a reference to it from
      non-data-free to not. A parameter is relevant if it appears directly in
      a field (including nested in ``array``/``dict``/function-parameter/
      result position), or is passed to another reference's parameter that
      is ITSELF relevant for that reference's declaration — transitively.
      Unused ("phantom") parameters are therefore never relevant, matching
      the substitute-then-walk semantics this replaces: instantiating a
      phantom parameter with a non-data type cannot poison a declaration
      because the field template never actually mentions it.

    A concrete handle's answer
    (:meth:`~agm.agl.semantics.type_table.TypeTable.nominal_reaches_non_data`)
    is then: its declaration's ``reaches_non_data`` flag, OR its declaration
    reaches a non-data type for some ``type_args[i]`` whose parameter is in
    ``relevant_params`` — reproducing the substitute-then-walk answer exactly,
    without ever expanding an instantiation.
    """
    defs = table.defs
    exception_children: dict[DeclId, set[DeclId]] = {decl_id: set() for decl_id in defs}
    for decl_id, typedef in defs.items():
        if typedef.kind == "exception" and typedef.base in exception_children:
            exception_children[typedef.base].add(decl_id)

    # ``field_non_data`` is the exact-value/inherited-field fact for
    # exceptions. ``non_data`` additionally includes affected descendants,
    # which should poison ancestor catch/base types but must not flow back
    # down to siblings.
    field_non_data: set[DeclId] = set()
    non_data: set[DeclId] = set()
    relevant: dict[DeclId, set[str]] = {decl_id: set() for decl_id in defs}
    changed = True
    while changed:
        changed = False
        for decl_id, typedef in defs.items():
            own_params = frozenset(typedef.type_params)
            templates = tuple(t for _fname, t in field_templates(typedef, defs))
            template_bad = any(
                _template_reaches_non_data(t, non_data, relevant, defs) for t in templates
            )
            inherited_field_bad = (
                typedef.kind == "exception"
                and typedef.base is not None
                and typedef.base in field_non_data
            )
            exact_bad = decl_id in field_non_data or template_bad or inherited_field_bad
            if exact_bad and decl_id not in field_non_data:
                field_non_data.add(decl_id)
                changed = True

            descendant_bad = typedef.kind == "exception" and any(
                child_id in non_data for child_id in exception_children[decl_id]
            )
            bad = exact_bad or descendant_bad
            if bad and decl_id not in non_data:
                non_data.add(decl_id)
                changed = True

            gained: set[str] = set()
            for t in templates:
                gained |= _template_relevant_params(t, own_params, relevant, defs)
            if not gained <= relevant[decl_id]:
                relevant[decl_id] |= gained
                changed = True
    return NonDataReachability(
        reaches_non_data=frozenset(non_data),
        relevant_params={decl_id: frozenset(params) for decl_id, params in relevant.items()},
    )


def field_templates(typedef: TypeDef, defs: Mapping[DeclId, TypeDef]) -> list[tuple[str, Type]]:
    """Return every ``(name, type template)`` in *typedef*'s own body, flattened.

    Unlike :func:`_decl_inhabited`'s enum handling (which groups fields by
    variant, since only ONE variant needs to be fully inhabited), both the
    non-data-reachability and reference-edge fixpoints look at every field of
    every variant flat: a function/unit anywhere is reachable from some
    value of the declaration, and a reference to another declaration matters,
    regardless of which variant carries it. Enum member fields are first
    specialized through their member handles, so referenced generic records
    retain the arguments applied by the enum. Names are carried alongside so
    a use-site diagnostic can point at the field responsible.
    """
    if typedef.kind == "enum":
        templates: list[tuple[str, Type]] = []
        for member in typedef.members:
            member_def = defs.get(member.decl_id)
            if member_def is None:
                continue
            substitutions = dict(zip(member_def.type_params, member.type_args, strict=True))
            templates.extend(
                (name, substitute(field_type, substitutions))
                for name, field_type in member_def.fields
            )
        return templates
    return list(typedef.fields)


def _template_reaches_non_data(
    t: Type,
    non_data: set[DeclId],
    relevant: Mapping[DeclId, set[str]],
    defs: Mapping[DeclId, TypeDef],
) -> bool:
    match t:
        case FunctionType() | UnitType():
            return True
        case ArrayType():
            return _template_reaches_non_data(t.elem, non_data, relevant, defs)
        case DictType():
            return _template_reaches_non_data(t.value, non_data, relevant, defs)
        case ExceptionType():
            return t.decl_id in non_data
        case RecordType() | EnumType():
            decl_id = t.decl_id
            if decl_id in non_data:
                return True
            target = defs.get(decl_id)
            if target is None:
                return False
            own_relevant = relevant.get(decl_id, set())
            return any(
                _template_reaches_non_data(arg, non_data, relevant, defs)
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
) -> set[str]:
    match t:
        case TypeVarType():
            return {t.name} if t.name in own_params else set()
        case InferenceVarType():
            return set()
        case ArrayType():
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
            decl_id = t.decl_id
            target = defs.get(decl_id)
            if target is None:
                return set()
            own_relevant = relevant.get(decl_id, set())
            result = set()
            for pname, arg in zip(target.type_params, t.type_args):
                if pname in own_relevant:
                    result |= _template_relevant_params(arg, own_params, relevant, defs)
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
    relevant = _compute_schema_relevant_params(defs)
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
        case DictType(value=value):
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
            typedef = defs.get(t.decl_id)
            if typedef is None:
                return
            relevant = relevant_params.get(t.decl_id, frozenset())
            for pname, arg in zip(typedef.type_params, t.type_args):
                if pname in relevant:
                    yield from nominal_references_for_schema(arg, defs, relevant_params)
        case ExceptionType():
            yield t
        case ArrayType(elem=elem):
            yield from nominal_references_for_schema(elem, defs, relevant_params)
        case DictType(value=value):
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


def _compute_schema_relevant_params(
    defs: Mapping[DeclId, TypeDef],
) -> dict[DeclId, set[str]]:
    """Return params whose instantiation can affect schema reachability."""
    relevant: dict[DeclId, set[str]] = {decl_id: set() for decl_id in defs}
    changed = True
    while changed:
        changed = False
        for decl_id, typedef in defs.items():
            own_params = frozenset(typedef.type_params)
            gained: set[str] = set()
            for _fname, template in field_templates(typedef, defs):
                gained |= _template_relevant_params(template, own_params, relevant, defs)
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
        # A member of an SCC is normally a registered declaration, but a
        # dangling reference (a field naming a declaration that was never
        # registered — an internal-invariant violation, defensively handled
        # the same way as the non-data-reachability fixpoint) can surface here
        # as its own singleton SCC; treat it as contributing no edges rather
        # than crashing.
        source_def = defs.get(source_id)
        if source_def is None:
            continue
        for ref_edge in edges.get(source_id, ()):
            target_id = ref_edge.target
            if target_id not in members:
                continue
            target_def = defs.get(target_id)
            if target_def is None:  # pragma: no cover
                # Unreachable by construction: a dangling (never-registered)
                # target has no outgoing edges of its own, so it can only
                # ever form its own singleton SCC — never share "members"
                # with a distinct source_id that has an edge into it. Kept
                # as a defensive guard, matching the dangling-source check
                # above, in case that invariant ever stops holding.
                continue
            target_relevant = relevant_params.get(target_id, set())
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
        case DictType(value=value):
            return _param_occurrences(
                value, growing=True, defs=defs, relevant_params=relevant_params
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
            target = defs.get(t.decl_id)
            if target is None:
                relevant_args = t.type_args
            else:
                target_relevant = relevant_params.get(t.decl_id, set())
                relevant_args = tuple(
                    arg
                    for pname, arg in zip(target.type_params, t.type_args)
                    if pname in target_relevant
                )
                if len(t.type_args) > len(target.type_params):
                    relevant_args += t.type_args[len(target.type_params) :]
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
