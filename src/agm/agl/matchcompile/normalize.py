"""Normalize checked AgL patterns into canonical pattern-matrix rows."""

from __future__ import annotations

import decimal
import weakref
from dataclasses import replace
from typing import Never, NoReturn, assert_never

from agm.agl.modules.ids import ModuleId
from agm.agl.scope.symbols import ConstructorRef
from agm.agl.semantics.type_table import TypeDef, TypeTable
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
)
from agm.agl.syntax.nodes import (
    AsPattern,
    BoolLit,
    Case,
    ConstructorPattern,
    DecimalLit,
    IntLit,
    LetDecl,
    LiteralPattern,
    NullLit,
    Pattern,
    StringLit,
    VarPattern,
    WildcardPattern,
)
from agm.agl.typecheck.env import CheckedModule

from .model import (
    BinderProvenance,
    BoolConstructor,
    CaseSite,
    ClosedSignature,
    Constructor,
    ConstructorCell,
    ConstructorField,
    LetBindingAction,
    LetSite,
    LiteralConstructor,
    LiteralKind,
    MatchCaseContext,
    MatrixRow,
    NominalConstructor,
    NormalizedMatchSite,
    Occurrence,
    OccurrenceId,
    OmittedFieldProvenance,
    OpenSignature,
    PatternCell,
    RootOccurrenceProvenance,
    Signature,
    SourceAction,
    SourcePatternProvenance,
    WildcardCell,
)

CheckedPatternOwner = CheckedModule


class MatchCompileInvariantError(RuntimeError):
    """A checked-program invariant required by match compilation was violated."""


def _unsupported(description: str, node: Never) -> NoReturn:
    """Reject a value outside the closed union this dispatch is total over.

    ``assert_never`` keeps the dispatch statically exhaustive; the raise turns a
    checked-output value that escaped the union into a compiler invariant error
    rather than a bare ``AssertionError``.
    """
    try:
        assert_never(node)
    except AssertionError as exc:
        raise MatchCompileInvariantError(
            f"unsupported {description} {type(node).__name__}"
        ) from exc


def resolve_bare_enum_constructors(
    checked: CheckedPatternOwner,
) -> frozenset[tuple[ModuleId, str, str]]:
    """Collect enum constructors whose unqualified call forms are visible.

    The witness renderer may use an explicit call form for field-bearing
    variants, so its visibility set is broader than the nullary-only bare-name
    pattern rule. Ordinary value bindings do not hide these pattern forms.

    A candidate only qualifies when its owner path is a registered enum's own
    declaration path, so the key's middle component really is an enum name. A
    record declared in a named scope never qualifies: its owner path is its
    declaration's enclosing scope, even when that scope shares a name with an
    unrelated enum.
    """
    enum_paths = {
        (typedef.module_id, (*typedef.scope_path, typedef.name))
        for typedef in checked.type_env.type_table.entries()
        if typedef.kind == "enum"
    }
    return frozenset(
        (candidate.owner_module_id, candidate.owner_path[-1], candidate.owner_name)
        for candidates in checked.resolved.constructor_candidates.values()
        for candidate in candidates
        if (candidate.owner_module_id, candidate.owner_path) in enum_paths
    )


def enum_constructor(enum_type: EnumType, variant: str, table: TypeTable) -> NominalConstructor:
    try:
        variants = table.enum_member_names(enum_type)
    except (KeyError, AssertionError) as exc:
        raise MatchCompileInvariantError(
            f"cannot resolve enum signature for checked type {enum_type!r}"
        ) from exc
    member = variants.get(variant)
    if member is None:
        raise MatchCompileInvariantError(
            f"checked enum pattern names unknown variant {enum_type!r}::{variant}"
        )
    return record_constructor(member, table)


def record_constructor(record_type: RecordType, table: TypeTable) -> NominalConstructor:
    try:
        fields = table.record_fields(record_type)
    except (KeyError, AssertionError) as exc:
        raise MatchCompileInvariantError(
            f"cannot resolve record signature for checked type {record_type!r}"
        ) from exc
    return NominalConstructor(
        record_type=record_type,
        fields=tuple(ConstructorField(name, field_type) for name, field_type in fields.items()),
    )


def constructor_inhabits_type(
    constructor: Constructor, subject_type: Type, table: TypeTable
) -> bool:
    """Return whether a constructor denotes any runtime value of ``subject_type``.

    This dispatch is deliberately total over both current closed unions.  In
    particular, runtime numeric equality permits an integral decimal pattern
    to match an integer, but no integer value can equal a fractional or
    non-finite decimal.  AgL decimal values are finite exact decimals.
    """
    match constructor:
        case BoolConstructor() | NominalConstructor() | LiteralConstructor():
            pass
        case _ as unsupported_constructor:
            _unsupported("constructor", unsupported_constructor)

    match subject_type:
        case BoolType():
            return isinstance(constructor, BoolConstructor)
        case EnumType() as enum_type:
            try:
                members = table.enum_members(enum_type)
            except (KeyError, AssertionError) as exc:
                raise MatchCompileInvariantError(
                    f"cannot resolve enum signature for checked type {enum_type!r}"
                ) from exc
            return (
                isinstance(constructor, NominalConstructor) and constructor.record_type in members
            )
        case RecordType() as record_type:
            return (
                isinstance(constructor, NominalConstructor)
                and constructor.record_type == record_type
            )
        case IntType():
            return (
                isinstance(constructor, LiteralConstructor)
                and constructor.kind is LiteralKind.NUMERIC
                and isinstance(constructor.value, decimal.Decimal)
                and constructor.value.is_finite()
                and constructor.value == constructor.value.to_integral_value()
            )
        case DecimalType():
            return (
                isinstance(constructor, LiteralConstructor)
                and constructor.kind is LiteralKind.NUMERIC
                and isinstance(constructor.value, decimal.Decimal)
                and constructor.value.is_finite()
            )
        case TextType():
            return (
                isinstance(constructor, LiteralConstructor) and constructor.kind is LiteralKind.TEXT
            )
        case JsonType():
            return (
                isinstance(constructor, LiteralConstructor) and constructor.kind is LiteralKind.NULL
            )
        case InferenceVarType():
            raise MatchCompileInvariantError("flexible inference type escaped checked output")
        case (
            TypeVarType()
            | ArrayType()
            | DictType()
            | ExceptionType()
            | UnitType()
            | FunctionType()
            | BottomType()
        ):
            return False
        case _ as unsupported_type:
            _unsupported("semantic type", unsupported_type)


def pattern_cell_inhabits_type(cell: PatternCell, subject_type: Type, table: TypeTable) -> bool:
    """Return whether a canonical cell can match a value of ``subject_type``."""
    if isinstance(subject_type, BottomType):
        return False
    if isinstance(cell, WildcardCell):
        return True
    if not constructor_inhabits_type(cell.constructor, subject_type, table):
        return False
    if isinstance(cell.constructor, NominalConstructor):
        return all(
            pattern_cell_inhabits_type(argument, field.type, table)
            for field, argument in zip(cell.constructor.fields, cell.arguments, strict=True)
        )
    return True


_NOMINAL_SIGNATURES: weakref.WeakKeyDictionary[
    TypeTable, dict[tuple[EnumType | RecordType, TypeDef], ClosedSignature]
] = weakref.WeakKeyDictionary()


def _build_enum_signature(enum_type: EnumType, table: TypeTable) -> ClosedSignature:
    try:
        variant_names = tuple(table.enum_member_names(enum_type))
    except (KeyError, AssertionError) as exc:
        raise MatchCompileInvariantError(
            f"cannot resolve enum signature for checked type {enum_type!r}"
        ) from exc
    return ClosedSignature(
        tuple(enum_constructor(enum_type, name, table) for name in variant_names)
    )


def _build_record_signature(record_type: RecordType, table: TypeTable) -> ClosedSignature:
    return ClosedSignature((record_constructor(record_type, table),))


def _nominal_signature(nominal_type: EnumType | RecordType, table: TypeTable) -> ClosedSignature:
    """Return a declaration-sensitive closed signature for a nominal type.

    Memoized per ``(handle, its declaration)``: pairing the handle with the
    ``TypeDef`` it NAMES — looked up by the handle's own declaration identity,
    never by its bare name — is what makes a redeclaration of exactly that
    declaration invalidate the entry, without an unrelated declaration
    sharing the name affecting it either way. A handle naming no registered
    declaration has nothing to key an entry on, so it is built uncached.
    """

    def build() -> ClosedSignature:
        if isinstance(nominal_type, EnumType):
            return _build_enum_signature(nominal_type, table)
        return _build_record_signature(nominal_type, table)

    typedef = table.get_by_id(nominal_type.decl_id)
    if typedef is None:
        return build()
    cache = _NOMINAL_SIGNATURES.get(table)
    if cache is None:
        cache = {}
        _NOMINAL_SIGNATURES[table] = cache
    key = (nominal_type, typedef)
    signature = cache.get(key)
    if signature is None:
        signature = build()
        cache[key] = signature
    return signature


def signature_for_type(subject_type: Type, table: TypeTable) -> Signature:
    """Return the complete constructor signature for every current semantic type.

    The explicit closed dispatch is intentional: adding a semantic ``Type``
    without classifying its matching domain is a compiler error, not an implicit
    fallback to an open domain.
    """
    match subject_type:
        case BoolType():
            return ClosedSignature((BoolConstructor(False), BoolConstructor(True)))
        case EnumType() | RecordType() as nominal_type:
            return _nominal_signature(nominal_type, table)
        case InferenceVarType():
            raise MatchCompileInvariantError("flexible inference type escaped checked output")
        case BottomType():
            return ClosedSignature(())
        case (
            TextType()
            | JsonType()
            | IntType()
            | DecimalType()
            | TypeVarType()
            | ArrayType()
            | DictType()
            | ExceptionType()
            | UnitType()
            | FunctionType()
        ):
            return OpenSignature()
        case _ as unreachable:
            _unsupported("semantic type", unreachable)


def _canonical_literal(pattern: LiteralPattern, subject_type: Type) -> Constructor:
    literal = pattern.literal
    if isinstance(subject_type, BoolType) and isinstance(literal, BoolLit):
        return BoolConstructor(literal.value)
    if isinstance(subject_type, (IntType, DecimalType)) and isinstance(
        literal, (IntLit, DecimalLit)
    ):
        return LiteralConstructor(LiteralKind.NUMERIC, decimal.Decimal(literal.value))
    if isinstance(subject_type, TextType) and isinstance(literal, StringLit):
        return LiteralConstructor(LiteralKind.TEXT, literal.value)
    if isinstance(subject_type, JsonType) and isinstance(literal, NullLit):
        return LiteralConstructor(LiteralKind.NULL, None)
    raise MatchCompileInvariantError(
        "checked literal pattern is incompatible with its occurrence type: "
        f"{type(literal).__name__} against {subject_type!r}"
    )


def _add_as_binder(cell: PatternCell, binder: BinderProvenance) -> PatternCell:
    """Attach an as-pattern binder to the occurrence represented by *cell*."""
    return replace(cell, binders=(*cell.binders, binder))


def _canonical_enum_pattern_variant(
    source_name: str,
    node_id: int,
    constructor_ref: ConstructorRef,
    selected_owner: int | None,
    subject_type: EnumType,
    checked: CheckedPatternOwner,
) -> str:
    """Return the member-record name behind a canonical or aliased pattern spelling."""
    try:
        members = checked.type_env.type_table.enum_member_names(subject_type)
    except (KeyError, AssertionError) as exc:
        raise MatchCompileInvariantError("cannot resolve enum signature") from exc
    recorded_spelling = checked.resolved.pattern_constructor_spellings.get(node_id)
    if recorded_spelling is not None and recorded_spelling != source_name:
        raise MatchCompileInvariantError("invalid final constructor classification")
    candidates = checked.resolved.pattern_constructor_candidates.get(node_id, ())
    candidate_matches_owner = any(
        candidate.owner_decl_node_id == selected_owner for candidate in candidates
    )
    if selected_owner is None or (
        candidates
        and (constructor_ref.owner_decl_node_id != selected_owner or candidate_matches_owner)
        and constructor_ref not in candidates
    ):
        raise MatchCompileInvariantError("invalid final constructor classification")
    member = next((member for member in members.values() if member.decl_id == selected_owner), None)
    if member is None:
        raise MatchCompileInvariantError("invalid final constructor classification")
    return member.name


def normalize_pattern(
    pattern: Pattern,
    subject_type: Type,
    checked: CheckedPatternOwner,
) -> PatternCell:
    """Normalize one checked pattern against its checked occurrence type."""
    provenance = SourcePatternProvenance(pattern.node_id, pattern.span)
    match pattern:
        case WildcardPattern():
            return WildcardCell(provenance=provenance)
        case AsPattern(pattern=inner, node_id=node_id, name=name):
            return _add_as_binder(
                normalize_pattern(inner, subject_type, checked),
                BinderProvenance(node_id=node_id, name=name, span=pattern.span),
            )
        case VarPattern(node_id=node_id, name=name):
            classifications = checked.pattern_classifications
            if node_id in classifications and classifications[node_id] is None:
                return WildcardCell(
                    provenance=provenance,
                    binders=(BinderProvenance(node_id=node_id, name=name, span=pattern.span),),
                )
            constructor_ref = classifications.get(node_id)
            if constructor_ref is None:
                raise MatchCompileInvariantError(
                    "missing final constructor classification for bare pattern"
                )
            if isinstance(subject_type, EnumType):
                canonical_variant = _canonical_enum_pattern_variant(
                    name,
                    node_id,
                    constructor_ref,
                    constructor_ref.owner_decl_node_id,
                    subject_type,
                    checked,
                )
                constructor = enum_constructor(
                    subject_type,
                    canonical_variant,
                    checked.type_env.type_table,
                )
                members = checked.type_env.type_table.enum_members(subject_type)
                assert constructor.record_type in members
                assert constructor.record_type.decl_id == constructor_ref.owner_decl_node_id
                assert constructor.arity == 0
                return ConstructorCell(constructor, (), provenance)
            if not isinstance(subject_type, RecordType):
                raise MatchCompileInvariantError(
                    "final bare constructor has a non-enum or record checked type"
                )
            assert constructor_ref.owner_decl_node_id == subject_type.decl_id
            constructor = record_constructor(subject_type, checked.type_env.type_table)
            assert constructor.arity == 0
            return ConstructorCell(constructor, (), provenance)
        case LiteralPattern():
            return ConstructorCell(
                constructor=_canonical_literal(pattern, subject_type),
                arguments=(),
                provenance=provenance,
            )
        case ConstructorPattern():
            constructor_ref = checked.pattern_constructor_ref_for(pattern.node_id)
            if constructor_ref is None:
                raise MatchCompileInvariantError(
                    "missing final constructor classification for applied pattern"
                )
            if not isinstance(subject_type, (EnumType, RecordType)):
                raise MatchCompileInvariantError(
                    "checked constructor pattern has a non-enum or non-record occurrence type"
                )
            # Compare checker-published nominal identity; normalization does not
            # re-select the constructor from scope candidates.
            selected_owner = checked.pattern_constructor_owner_for(pattern.node_id)
            applied_variant = (
                _canonical_enum_pattern_variant(
                    pattern.name,
                    pattern.node_id,
                    constructor_ref,
                    None if selected_owner is None else selected_owner.value,
                    subject_type,
                    checked,
                )
                if isinstance(subject_type, EnumType)
                else None
            )
            if isinstance(subject_type, EnumType):
                assert applied_variant is not None
                expected_owner = checked.type_env.type_table.enum_member_names(subject_type)[
                    applied_variant
                ].decl_id
            else:
                expected_owner = subject_type.decl_id
            if selected_owner is None or selected_owner.value != expected_owner:
                raise MatchCompileInvariantError(
                    "invalid final constructor classification: published nominal owner disagrees "
                    "with the checked occurrence type"
                )
            if isinstance(subject_type, EnumType):
                assert applied_variant is not None
                nominal_constructor = enum_constructor(
                    subject_type, applied_variant, checked.type_env.type_table
                )
            else:
                nominal_constructor = record_constructor(subject_type, checked.type_env.type_table)
            supplied_pairs = checked.argument_bindings.constructor_patterns.get(pattern.node_id)
            if supplied_pairs is None:
                raise MatchCompileInvariantError(
                    f"missing checked argument bindings for pattern node {pattern.node_id}"
                )
            supplied = dict(supplied_pairs)
            if len(supplied) != len(supplied_pairs):
                raise MatchCompileInvariantError(
                    f"duplicate checked field binding for pattern node {pattern.node_id}"
                )
            declared_names = {field.name for field in nominal_constructor.fields}
            unknown = supplied.keys() - declared_names
            if unknown:
                raise MatchCompileInvariantError(
                    f"checked pattern node {pattern.node_id} binds unknown fields "
                    f"{sorted(unknown)!r}"
                )
            arguments: list[PatternCell] = []
            for field in nominal_constructor.fields:
                child = supplied.get(field.name)
                if child is None:
                    arguments.append(
                        WildcardCell(
                            provenance=OmittedFieldProvenance(
                                field_name=field.name,
                                span=pattern.span,
                            ),
                        )
                    )
                else:
                    arguments.append(normalize_pattern(child, field.type, checked))
            return ConstructorCell(nominal_constructor, tuple(arguments), provenance)
        case _ as unreachable:
            _unsupported("source pattern", unreachable)


def match_case_context(checked: CheckedPatternOwner) -> MatchCaseContext:
    """Resolve the checked qualification metadata shared by one owner's match sites."""
    return MatchCaseContext(
        module_id=checked.module_id,
        enum_owner_forms=checked.type_env.enum_owner_forms(),
        blocked_enum_variants=checked.type_env.blocked_enum_variants(),
        bare_enum_constructors=resolve_bare_enum_constructors(checked),
        owner_program=checked.resolved.program,
    )


def normalize_case(
    case: Case,
    checked: CheckedPatternOwner,
    *,
    case_context: MatchCaseContext | None = None,
) -> NormalizedMatchSite:
    """Normalize one checked source case into a source-priority one-column matrix.

    An owner-wide caller passes *case_context* so all of its match sites share
    one resolution of the checked qualification metadata; direct callers let it
    default to resolving from *checked*.
    """
    try:
        subject_type = checked.node_types[case.subject.node_id]
    except KeyError as exc:
        raise MatchCompileInvariantError(
            f"missing checked subject type for case node {case.node_id}"
        ) from exc
    root = Occurrence(
        id=OccurrenceId(0),
        creation_order=0,
        type=subject_type,
        provenance=RootOccurrenceProvenance(
            site_node_id=case.node_id,
            span=case.subject.span,
        ),
    )
    rows: list[MatrixRow] = []
    for index, branch in enumerate(case.branches):
        cell = normalize_pattern(branch.pattern, subject_type, checked)
        if pattern_cell_inhabits_type(cell, subject_type, checked.type_env.type_table):
            rows.append(
                MatrixRow(
                    cells=(cell,),
                    action_id=branch.node_id,
                    source_index=index,
                    source_pattern_id=branch.pattern.node_id,
                )
            )
    actions = tuple(
        SourceAction(
            action_id=branch.node_id,
            source_index=index,
            body_node_id=branch.body.node_id,
            pattern_span=branch.pattern.span,
        )
        for index, branch in enumerate(case.branches)
    )
    return NormalizedMatchSite(
        site_node_id=case.node_id,
        source=CaseSite(actions=actions),
        span=case.span,
        root=root,
        occurrences=(root,),
        rows=tuple(rows),
        type_table=checked.type_env.type_table,
        case_context=case_context if case_context is not None else match_case_context(checked),
    )


def normalize_let(
    let: LetDecl,
    checked: CheckedPatternOwner,
    *,
    case_context: MatchCaseContext | None = None,
) -> NormalizedMatchSite:
    """Normalize one checked ``let`` as its complete one-row match matrix.

    A let's initializer remains source-owned: this artifact records its identity
    but neither captures it nor represents its continuation.
    """
    try:
        matched_type = checked.let_matched_types[let.node_id]
    except KeyError as exc:
        raise MatchCompileInvariantError(
            f"missing checked matched type for let node {let.node_id}"
        ) from exc
    root = Occurrence(
        id=OccurrenceId(0),
        creation_order=0,
        type=matched_type,
        provenance=RootOccurrenceProvenance(
            site_node_id=let.node_id,
            span=let.value.span,
        ),
    )
    cell = normalize_pattern(let.pattern, matched_type, checked)
    action = LetBindingAction(action_id=let.node_id, source_index=0)
    return NormalizedMatchSite(
        site_node_id=let.node_id,
        source=LetSite(action=action),
        span=let.span,
        root=root,
        occurrences=(root,),
        rows=(
            MatrixRow(
                cells=(cell,),
                action_id=action.action_id,
                source_index=action.source_index,
                source_pattern_id=let.pattern.node_id,
            ),
        ),
        type_table=checked.type_env.type_table,
        case_context=case_context if case_context is not None else match_case_context(checked),
    )


__all__ = [
    "CheckedPatternOwner",
    "MatchCompileInvariantError",
    "constructor_inhabits_type",
    "enum_constructor",
    "match_case_context",
    "normalize_case",
    "normalize_let",
    "normalize_pattern",
    "pattern_cell_inhabits_type",
    "record_constructor",
    "resolve_bare_enum_constructors",
    "signature_for_type",
]
