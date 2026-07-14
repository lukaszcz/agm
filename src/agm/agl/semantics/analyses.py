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
declarations have a finite closure. Unlike inhabitation and equality
capability, this is not a monotone fixpoint grown fact-by-fact — it is a
one-shot graph classification: build the declaration reference graph (which
declaration reference templates mention which other declarations, and with
which argument templates), find its strongly-connected components (SCCs),
and within each SCC build a small parameter-dependency graph (which of a
referencing declaration's OWN parameters feed which of the referenced
declaration's parameters, and whether that feed is "growing" — the source
parameter occurs as a proper subterm of the argument template, under a
list/dict/function/nominal-argument constructor, rather than being passed
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

from agm.agl.modules.ids import ModuleId
from agm.agl.semantics.type_table import TypeDef, TypeDefKind, TypeTable, decl_key_sort_key
from agm.agl.semantics.types import (
    AgentType,
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
    ListType,
    RecordType,
    TextType,
    Type,
    TypeVarType,
    UnitType,
    substitute,
)
from agm.util.graph import sccs

# A declaration's identity in the shared type table.
DeclKey = tuple[ModuleId, str]
TypeEnv = Mapping[str, Type]
InstantiationKey = tuple[DeclKey, tuple[Type, ...]]


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

    Generic references are checked at the concrete argument templates used at
    the reference site. A free type variable in a declaration body is treated
    as inhabited, but once a generic wrapper is applied to an uninhabited
    recursive type (for example ``Box[Bad]`` where ``Box[T]`` stores a ``T``),
    the wrapper's body is evaluated with that argument substituted, so it does
    not hide unguarded recursion.
    """
    defs = _all_defs(table)
    inhabited: set[DeclKey] = set()
    changed = True
    while changed:
        changed = False
        for key, typedef in defs.items():
            if key in inhabited:
                continue
            if _decl_inhabited(typedef, inhabited, defs):
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


def _decl_inhabited(
    typedef: TypeDef,
    inhabited: set[DeclKey],
    defs: Mapping[DeclKey, TypeDef],
) -> bool:
    args = tuple(TypeVarType(param) for param in typedef.type_params)
    return _body_inhabited(
        typedef,
        {},
        inhabited,
        defs,
        stack=frozenset({((typedef.module_id, typedef.name), args)}),
    )


def _body_inhabited(
    typedef: TypeDef,
    env: TypeEnv,
    inhabited: set[DeclKey],
    defs: Mapping[DeclKey, TypeDef],
    *,
    stack: frozenset[InstantiationKey],
) -> bool:
    if typedef.kind == "enum":
        return any(
            all(
                _template_inhabited(t, env, inhabited, defs, stack=stack)
                for _fname, t in vfields
            )
            for _vname, vfields in typedef.variants
        )
    if typedef.kind == "exception":
        return _exception_decl_inhabited(typedef, env, inhabited, defs, stack=stack)
    return all(
        _template_inhabited(t, env, inhabited, defs, stack=stack)
        for _fname, t in typedef.fields
    )


def _exception_decl_inhabited(
    typedef: TypeDef,
    env: TypeEnv,
    inhabited: set[DeclKey],
    defs: Mapping[DeclKey, TypeDef],
    *,
    stack: frozenset[InstantiationKey],
) -> bool:
    key = (typedef.module_id, typedef.name)
    if typedef.abstract:
        return any(
            child.kind == "exception" and child.base == key and child_key in inhabited
            for child_key, child in defs.items()
        )
    return _exception_fields_inhabited(
        typedef,
        env,
        inhabited,
        defs,
        stack=stack,
        extends_stack=frozenset({key}),
    )


def _exception_fields_inhabited(
    typedef: TypeDef,
    env: TypeEnv,
    inhabited: set[DeclKey],
    defs: Mapping[DeclKey, TypeDef],
    *,
    stack: frozenset[InstantiationKey],
    extends_stack: frozenset[DeclKey],
) -> bool:
    own_ok = all(
        _template_inhabited(t, env, inhabited, defs, stack=stack)
        for _fname, t in typedef.fields
    )
    if not own_ok:
        return False
    if typedef.base is None:
        return True
    if typedef.base in extends_stack:
        return False
    base_def = defs.get(typedef.base)
    if base_def is None or base_def.kind != "exception":
        return False
    return _exception_fields_inhabited(
        base_def,
        env,
        inhabited,
        defs,
        stack=stack,
        extends_stack=extends_stack | frozenset({typedef.base}),
    )


def _template_inhabited(
    t: Type,
    env: TypeEnv,
    inhabited: set[DeclKey],
    defs: Mapping[DeclKey, TypeDef],
    *,
    stack: frozenset[InstantiationKey],
) -> bool:
    match t:
        case TypeVarType():
            replacement = env.get(t.name)
            if replacement is None or replacement == t:
                return True
            return _template_inhabited(replacement, env, inhabited, defs, stack=stack)
        case InferenceVarType():
            return True
        case RecordType() | EnumType():
            key = (t.module_id, t.name)
            if any(stack_key == key for stack_key, _args in stack):
                return False
            target = defs.get(key)
            if target is None:
                return key in inhabited
            args = tuple(substitute(arg, env) for arg in t.type_args)
            instantiation = (key, args)
            target_env = dict(zip(target.type_params, args))
            return _body_inhabited(
                target,
                target_env,
                inhabited,
                defs,
                stack=stack | frozenset({instantiation}),
            )
        case ExceptionType():
            key = (t.module_id, t.name)
            if any(stack_key == key for stack_key, _args in stack):
                return False
            return key in inhabited
        case ListType() | DictType():
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
            | AgentType()
            | BottomType()
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
      some field/variant-field template contains a function/agent/unit type,
      or references a declaration whose own ``no_equality`` is already true —
      at any depth, but never through a bare type-variable position (a
      parameter standing for "whatever the caller instantiates" is not
      itself a problem). For exceptions this also accounts for subtyping:
      inherited field problems flow from base to child, while a child with no
      equality also makes each catchable ancestor non-comparable because a
      value statically typed as that ancestor may hold the child at runtime.
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
    exception_children: dict[DeclKey, set[DeclKey]] = {key: set() for key in defs}
    for key, typedef in defs.items():
        if typedef.kind == "exception" and typedef.base in exception_children:
            exception_children[typedef.base].add(key)

    # ``field_no_eq`` is the exact-value/inherited-field fact for exceptions.
    # ``no_eq`` additionally includes non-comparable descendants, which should
    # poison ancestor catch/base types but must not flow back down to siblings.
    field_no_eq: set[DeclKey] = set()
    no_eq: set[DeclKey] = set()
    relevant: dict[DeclKey, set[str]] = {key: set() for key in defs}
    changed = True
    while changed:
        changed = False
        for key, typedef in defs.items():
            own_params = frozenset(typedef.type_params)
            templates = tuple(_own_field_templates(typedef))
            template_bad = any(_template_no_eq(t, no_eq, relevant, defs) for t in templates)
            inherited_field_bad = (
                typedef.kind == "exception"
                and typedef.base is not None
                and typedef.base in field_no_eq
            )
            exact_bad = key in field_no_eq or template_bad or inherited_field_bad
            if exact_bad and key not in field_no_eq:
                field_no_eq.add(key)
                changed = True

            descendant_bad = typedef.kind == "exception" and any(
                child_key in no_eq for child_key in exception_children[key]
            )
            bad = exact_bad or descendant_bad
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


def _own_field_templates(typedef: TypeDef) -> list[Type]:
    """Return every field-type template in *typedef*'s own body, flattened.

    Unlike :func:`_decl_inhabited`'s enum handling (which groups fields by
    variant, since only ONE variant needs to be fully inhabited), both the
    equality-capability and reference-edge fixpoints look at every field of
    every variant flat: a function/agent/unit anywhere makes every value
    non-comparable, and a reference to another declaration matters, regardless
    of which variant carries it.
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
            | InferenceVarType()
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
        case InferenceVarType():
            return set()
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
            | InferenceVarType()
        ):
            return set()
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


# ---------------------------------------------------------------------------
# Finiteness (instantiation-closure) analysis
# ---------------------------------------------------------------------------

# A parameter of a declaration, identified by the declaration's key and the
# parameter's own name — a node in the small per-SCC parameter-dependency
# graph.
ParamKey = tuple[DeclKey, str]


@dataclass(frozen=True, slots=True)
class _RefEdge:
    """One nominal reference found in a declaration's own body.

    ``target`` is the referenced declaration; ``arg_templates`` are the
    reference's argument templates, positionally aligned with the target's
    OWN type parameters (empty for a reference to a non-generic declaration,
    e.g. an exception or its ``extends`` base).
    """

    target: DeclKey
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

    infinite: frozenset[DeclKey]
    successors: Mapping[DeclKey, frozenset[DeclKey]]
    relevant_params: Mapping[DeclKey, frozenset[str]]


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
    defs = _all_defs(table)
    relevant = _compute_schema_relevant_params(defs)
    edges = _reference_edges(defs, relevant)
    successors: dict[DeclKey, frozenset[DeclKey]] = {
        key: frozenset(edge.target for edge in refs) for key, refs in edges.items()
    }
    adjacency: dict[DeclKey, tuple[DeclKey, ...]] = {
        key: tuple(targets) for key, targets in successors.items()
    }
    components = sccs(adjacency, key=decl_key_sort_key)
    infinite: set[DeclKey] = set()
    for component in components:
        members = frozenset(component)
        if _scc_has_growing_cycle(members, edges, defs, relevant):
            infinite.update(members)
    return FiniteClosure(
        infinite=frozenset(infinite),
        successors=successors,
        relevant_params={key: frozenset(params) for key, params in relevant.items()},
    )


def nominal_references(t: Type) -> Iterator[RecordType | EnumType | ExceptionType]:
    """Yield every nominal reference occurring anywhere in *t*, including nested.

    Recurses into ``list``/``dict``/function shapes and, for a nominal
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
        case ListType(elem=elem):
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
            | AgentType()
            | BottomType()
            | TypeVarType()
            | InferenceVarType()
        ):
            return
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def nominal_references_for_schema(
    t: Type,
    defs: Mapping[DeclKey, TypeDef],
    relevant_params: Mapping[DeclKey, frozenset[str]],
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
            key = (t.module_id, t.name)
            typedef = defs.get(key)
            if typedef is None:
                return
            relevant = relevant_params.get(key, frozenset())
            for pname, arg in zip(typedef.type_params, t.type_args):
                if pname in relevant:
                    yield from nominal_references_for_schema(arg, defs, relevant_params)
        case ExceptionType():
            yield t
        case ListType(elem=elem):
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
            | AgentType()
            | BottomType()
            | TypeVarType()
            | InferenceVarType()
        ):
            return
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _compute_schema_relevant_params(
    defs: Mapping[DeclKey, TypeDef],
) -> dict[DeclKey, set[str]]:
    """Return params whose instantiation can affect schema reachability."""
    relevant: dict[DeclKey, set[str]] = {key: set() for key in defs}
    changed = True
    while changed:
        changed = False
        for key, typedef in defs.items():
            own_params = frozenset(typedef.type_params)
            gained: set[str] = set()
            for template in _own_field_templates(typedef):
                gained |= _template_relevant_params(template, own_params, relevant, defs)
            if not gained <= relevant[key]:
                relevant[key] |= gained
                changed = True
    return relevant


def _reference_edges(
    defs: Mapping[DeclKey, TypeDef],
    relevant_params: Mapping[DeclKey, set[str]],
) -> dict[DeclKey, tuple[_RefEdge, ...]]:
    """Return every declaration's schema-relevant outgoing reference edge."""
    frozen_relevant = {key: frozenset(params) for key, params in relevant_params.items()}
    result: dict[DeclKey, tuple[_RefEdge, ...]] = {}
    for key, typedef in defs.items():
        found: list[_RefEdge] = []
        for template in _own_field_templates(typedef):
            for ref in nominal_references_for_schema(template, defs, frozen_relevant):
                target = (ref.module_id, ref.name)
                arg_templates = ref.type_args if isinstance(ref, (RecordType, EnumType)) else ()
                found.append(_RefEdge(target=target, arg_templates=arg_templates))
        if typedef.kind == "exception" and typedef.base is not None:
            found.append(_RefEdge(target=typedef.base, arg_templates=()))
        result[key] = tuple(found)
    return result


def _scc_has_growing_cycle(
    members: frozenset[DeclKey],
    edges: Mapping[DeclKey, tuple[_RefEdge, ...]],
    defs: Mapping[DeclKey, TypeDef],
    relevant_params: Mapping[DeclKey, set[str]],
) -> bool:
    """Return ``True`` if *members*'s parameter-dependency graph has a growing cycle."""
    adjacency: dict[ParamKey, list[ParamKey]] = {}
    growing_edges: set[tuple[ParamKey, ParamKey]] = set()
    for source_key in members:
        # A member of an SCC is normally a registered declaration, but a
        # dangling reference (a field naming a declaration that was never
        # registered — an internal-invariant violation, defensively handled
        # the same way as the equality-capability fixpoint) can surface here
        # as its own singleton SCC; treat it as contributing no edges rather
        # than crashing.
        source_def = defs.get(source_key)
        if source_def is None:
            continue
        for ref_edge in edges.get(source_key, ()):
            target_key = ref_edge.target
            if target_key not in members:
                continue
            target_def = defs.get(target_key)
            if target_def is None:  # pragma: no cover
                # Unreachable by construction: a dangling (never-registered)
                # target has no outgoing edges of its own, so it can only
                # ever form its own singleton SCC — never share "members"
                # with a distinct source_key that has an edge into it. Kept
                # as a defensive guard, matching the dangling-source check
                # above, in case that invariant ever stops holding.
                continue
            target_relevant = relevant_params.get(target_key, set())
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
                    src: ParamKey = (source_key, source_param)
                    dst: ParamKey = (target_key, param_name)
                    adjacency.setdefault(src, []).append(dst)
                    if growing:
                        growing_edges.add((src, dst))
    if not adjacency:
        return False
    param_components = sccs(adjacency, key=lambda n: (n[0][0].segments, n[0][1], n[1]))
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
    defs: Mapping[DeclKey, TypeDef],
    relevant_params: Mapping[DeclKey, set[str]],
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
        case ListType(elem=elem):
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
                    _param_occurrences(
                        p, growing=True, defs=defs, relevant_params=relevant_params
                    ),
                )
            merged = _merge_growing(
                merged,
                _param_occurrences(
                    result, growing=True, defs=defs, relevant_params=relevant_params
                ),
            )
            return merged
        case RecordType() | EnumType():
            key = (t.module_id, t.name)
            target = defs.get(key)
            if target is None:
                relevant_args = t.type_args
            else:
                target_relevant = relevant_params.get(key, set())
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
            | AgentType()
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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _all_defs(table: TypeTable) -> dict[DeclKey, TypeDef]:
    """Return every registered ``TypeDef`` in *table*, keyed by its identity."""
    return {(typedef.module_id, typedef.name): typedef for typedef in table.entries()}
