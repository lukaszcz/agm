"""AgL value syntax -> JSON-native decoding: the host value-decode boundary.

Every host-supplied text value (program-parameter tokens, ``@opt-env``
variables, TOML config strings, engine-setting literals) is either strict
JSON or one AgL value-syntax literal (``agm.agl.value_syntax``).
:func:`value_node_to_json` converts an already-read
:class:`~agm.agl.value_syntax.nodes.ValueNode` into the JSON-native Python
object (``None``/``bool``/``int``/``Decimal``/``str``/``list``/``dict``) the
existing canonical decode boundary already expects, type-directed by a
:class:`~agm.agl.ir.contracts.DecodeSchema` -- no evaluation, no scope. The
result flows through the SAME normalize + JSON-Schema-validate +
``decode_value`` path as a JSON-supplied value; this module never builds a
``Value`` itself.

:func:`host_text_to_json` is the single host-text dispatch built on top of
it: a ``text`` target is taken verbatim, the standard ``Agent`` enum reads
its own text conventions (shorthand, a tagged JSON object, an ``Agent``
member constructor call, or -- optionally -- a verbatim command), and every
other target reads strict JSON, falling back to value syntax.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import assert_never, cast

from agm.agent.spec import AgentCommand
from agm.agent.values import agent_spec_shape, parse_agent_shorthand
from agm.agl.ir.contracts import (
    ArrayDecode,
    DecodeSchema,
    DictDecode,
    EnumDecode,
    FieldDecode,
    RecordDecode,
    RefDecode,
    ScalarDecode,
    ScalarKind,
    VariantDecode,
)
from agm.agl.runtime.convert import (
    StrictJsonParseError,
    parse_json_strict,
    resolve_decode_ref,
)
from agm.agl.semantics.arguments import (
    ArgumentBindingError,
    ArgumentBindingErrorKind,
    BindParam,
    bind_arguments,
)
from agm.agl.value_syntax.errors import ValueSyntaxError
from agm.agl.value_syntax.nodes import (
    ArrayNode,
    BoolNode,
    CtorNode,
    DecimalNode,
    DictNode,
    IntNode,
    NullNode,
    TextNode,
    ValueArg,
    ValueNode,
)
from agm.agl.value_syntax.reader import read_ctor_head, read_value

__all__ = [
    "ValueDecodeError",
    "host_text_to_json",
    "option_some_field_schema",
    "option_some_json_name",
    "value_node_to_json",
]

type DefsMap = Mapping[str, DecodeSchema]

_EMPTY_DEFS: DefsMap = MappingProxyType({})


class ValueDecodeError(ValueError):
    """One value-syntax reading or type-directed decoding failure.

    *offset*, when given, is the source offset the failure is anchored at;
    it is embedded in the message text rather than kept as a separate field,
    since no caller needs the raw position.
    """

    def __init__(self, message: str, offset: int | None = None) -> None:
        text = message if offset is None else f"{message} (at offset {offset})"
        super().__init__(text)


def _resolve(schema: DecodeSchema, defs: DefsMap) -> DecodeSchema:
    """Resolve a possible ``RefDecode`` root to its non-reference body."""
    if not isinstance(schema, RefDecode):
        return schema
    return resolve_decode_ref(schema.key, defs)


def _node_kind(node: ValueNode) -> str:
    """A short, human-facing description of *node*'s own shape, for error messages."""
    match node:
        case NullNode():
            return "null"
        case BoolNode():
            return "a boolean"
        case IntNode():
            return "an integer"
        case DecimalNode():
            return "a decimal"
        case TextNode():
            return "text"
        case ArrayNode():
            return "an array"
        case DictNode():
            return "a dict"
        case CtorNode(name=name):
            return f"constructor {name!r}"
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def value_node_to_json(
    node: ValueNode, schema: DecodeSchema, defs: DefsMap = _EMPTY_DEFS
) -> object:
    """Convert one value-syntax *node* into a JSON-native object per *schema*.

    Type-directed, no evaluation: a text/int/decimal/bool slot expects the
    matching literal (an int widens into a decimal slot); a ``json`` slot
    accepts any non-constructor literal, recursively, heterogeneous; an
    array/dict slot expects the matching bracketed literal; a record/enum
    slot expects a constructor naming the type (or its ``@name`` alias),
    whose arguments bind through the shared zone binder. *defs* resolves a
    ``RefDecode`` node exactly like :func:`~agm.agl.runtime.convert.decode_value`.

    :raises ValueDecodeError: on any type/shape mismatch.
    """
    resolved = _resolve(schema, defs)
    if isinstance(resolved, ScalarDecode):
        return _convert_scalar(node, resolved.kind, defs)
    if isinstance(resolved, ArrayDecode):
        if not isinstance(node, ArrayNode):
            raise ValueDecodeError(f"expected an array, got {_node_kind(node)}", node.start)
        return [value_node_to_json(item, resolved.elem, defs) for item in node.items]
    if isinstance(resolved, DictDecode):
        if not isinstance(node, DictNode):
            raise ValueDecodeError(f"expected a dict, got {_node_kind(node)}", node.start)
        return _convert_dict_entries(node, lambda v: value_node_to_json(v, resolved.value, defs))
    if isinstance(resolved, RecordDecode):
        return _convert_record(node, resolved, defs)
    if isinstance(resolved, EnumDecode):
        return _convert_enum(node, resolved, defs)
    raise AssertionError(  # pragma: no cover -- _resolve never returns a RefDecode
        f"value_decode: unresolved schema {resolved!r}"
    )


def _convert_scalar(node: ValueNode, kind: ScalarKind, defs: DefsMap) -> object:
    match kind:
        case ScalarKind.JSON:
            return _convert_json(node, defs)
        case ScalarKind.TEXT:
            if isinstance(node, TextNode):
                return node.value
        case ScalarKind.INT:
            if isinstance(node, IntNode):
                return node.value
        case ScalarKind.DECIMAL:
            if isinstance(node, IntNode | DecimalNode):
                return node.value
        case ScalarKind.BOOL:
            if isinstance(node, BoolNode):
                return node.value
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)
    expected = kind.name.lower()
    raise ValueDecodeError(f"expected {expected}, got {_node_kind(node)}", node.start)


def _convert_json(node: ValueNode, defs: DefsMap) -> object:
    """Convert *node* into plain heterogeneous JSON data; a constructor is an error."""
    if isinstance(node, NullNode):
        return None
    if isinstance(node, BoolNode):
        return node.value
    if isinstance(node, IntNode | DecimalNode):
        return node.value
    if isinstance(node, TextNode):
        return node.value
    if isinstance(node, ArrayNode):
        return [_convert_json(item, defs) for item in node.items]
    if isinstance(node, DictNode):
        return _convert_dict_entries(node, lambda v: _convert_json(v, defs))
    raise ValueDecodeError("a constructor is not valid inside json", node.start)


def _convert_dict_entries(
    node: DictNode, convert: Callable[[ValueNode], object]
) -> dict[str, object]:
    """Build a dict from *node*'s entries via *convert*, rejecting a duplicate key."""
    result: dict[str, object] = {}
    for entry in node.entries:
        if entry.key in result:
            raise ValueDecodeError(f"duplicate dict key {entry.key!r}", entry.start)
        result[entry.key] = convert(entry.value)
    return result


def _enclosing_name(display_name: str) -> str | None:
    """Return the ONE name immediately enclosing a declaration's own name.

    The second-to-last ``::``-segment of *display_name* -- an inline
    member's enum, or the innermost scope of a scoped record -- or ``None``
    for a top-level standalone declaration (no ``::`` in its display name).
    """
    segments = display_name.split("::")
    return segments[-2] if len(segments) > 1 else None


def _convert_record(node: ValueNode, schema: RecordDecode, defs: DefsMap) -> dict[str, object]:
    if not isinstance(node, CtorNode):
        raise ValueDecodeError(
            f"expected {schema.display_name}, got {_node_kind(node)}", node.start
        )
    if node.name != schema.name and node.name != schema.alias:
        raise ValueDecodeError(
            f"expected constructor {schema.display_name!r}, got {node.name!r}", node.start
        )
    if node.qualifier is not None and node.qualifier != _enclosing_name(schema.display_name):
        raise ValueDecodeError(
            f"{node.qualifier}::{node.name} does not name {schema.display_name!r}", node.start
        )
    return _convert_ctor_args(node, schema.fields, schema.display_name, defs)


def _match_variant(node: CtorNode, schema: EnumDecode) -> VariantDecode | None:
    """Return the enum member *node* names, or ``None`` when it names none.

    A qualifier is accepted when it is either the enum's own declared name
    (naming the member through the enum) or the one name immediately
    enclosing the member's own declaration -- its owning enum for an inline
    member, or the scope of a referenced (non-inline) member's own
    declaration.
    """
    variant = next((v for v in schema.variants if node.name in (v.name, v.alias)), None)
    if variant is None:
        return None
    if node.qualifier is not None and node.qualifier not in (
        schema.name,
        _enclosing_name(variant.display_name),
    ):
        return None
    return variant


def _convert_enum(node: ValueNode, schema: EnumDecode, defs: DefsMap) -> dict[str, object]:
    if not isinstance(node, CtorNode):
        raise ValueDecodeError(
            f"expected {schema.display_name}, got {_node_kind(node)}", node.start
        )
    variant = _match_variant(node, schema)
    if variant is None:
        raise ValueDecodeError(
            f"{node.name!r} does not name a member of {schema.display_name!r}", node.start
        )
    payload = _convert_ctor_args(node, variant.fields, variant.display_name, defs)
    return {"$case": variant.json_name, **payload}


def _convert_ctor_args(
    node: CtorNode, fields: "tuple[FieldDecode, ...]", type_label: str, defs: DefsMap
) -> dict[str, object]:
    """Bind and convert one constructor call's arguments against *fields*."""
    if node.args is None:
        if fields:
            names = ", ".join(f.name for f in fields)
            raise ValueDecodeError(f"{type_label} requires arguments: {names}", node.start)
        return {}
    alias_map: dict[str, str] = {}
    for field in fields:
        alias_map[field.name] = field.name
        if field.alias is not None:
            alias_map[field.alias] = field.name
    bind_params = [BindParam(name=f.name, kind=f.zone, has_default=False) for f in fields]
    positional = [arg for arg in node.args if arg.name is None]
    named: list[tuple[str, ValueArg]] = []
    for arg in node.args:
        if arg.name is None:
            continue
        declared = alias_map.get(arg.name)
        if declared is None:
            raise ValueDecodeError(f"{type_label} has no field {arg.name!r}", arg.start)
        named.append((declared, arg))
    try:
        bound = bind_arguments(bind_params, positional, named)
    except ArgumentBindingError as exc:
        raise _binding_error(exc, node, positional, named, type_label) from exc
    result: dict[str, object] = {}
    for field, bound_arg in zip(fields, bound, strict=True):
        assert bound_arg is not None, (
            "has_default=False: bind_arguments never defers a required field"
        )
        result[field.json_name] = value_node_to_json(bound_arg.value, field.schema, defs)
    return result


def _binding_error(
    exc: ArgumentBindingError,
    node: CtorNode,
    positional: "list[ValueArg]",
    named: "list[tuple[str, ValueArg]]",
    type_label: str,
) -> ValueDecodeError:
    """Translate one zone-binding violation from a constructor call into a domain error."""
    if exc.positional_index is not None:
        offset = positional[exc.positional_index].start
    elif exc.named_index is not None:
        offset = named[exc.named_index][1].start
    else:
        offset = node.start
    match exc.kind:
        case ArgumentBindingErrorKind.MISSING_REQUIRED:
            message = f"{type_label} is missing field {exc.name!r}"
        case ArgumentBindingErrorKind.DUPLICATE:
            message = f"{type_label} field {exc.name!r} supplied more than once"
        case ArgumentBindingErrorKind.POSITIONAL_ONLY_BY_NAME:
            message = f"{type_label} field {exc.name!r} cannot be supplied by name"
        case ArgumentBindingErrorKind.TOO_MANY_POSITIONAL:
            message = f"{type_label} given too many arguments"
        case ArgumentBindingErrorKind.POSITIONAL_IN_NAMED_ONLY:
            message = f"{type_label} arguments must be supplied by name"
        case ArgumentBindingErrorKind.UNKNOWN_NAME:  # pragma: no cover
            # Unreachable: names are resolved against the field alias map before binding.
            message = f"{type_label} has no field {exc.name!r}"
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)
    return ValueDecodeError(message, offset)


def _some_variant(schema: DecodeSchema, defs: DefsMap) -> VariantDecode:
    """Return the standard ``Option[T]``'s ``Some`` variant from its enum schema.

    *schema* is always the standard ``Option`` enum schema -- callers never
    pass anything else -- so its shape is looked up directly rather than
    re-checked here.
    """
    resolved = cast(EnumDecode, _resolve(schema, defs))
    return next(v for v in resolved.variants if v.name == "Some")


def option_some_field_schema(schema: DecodeSchema, defs: DefsMap = _EMPTY_DEFS) -> DecodeSchema:
    """Return the standard ``Option[T]``'s ``Some`` variant's own ``value`` field schema.

    Used to decode an ``Option[T]`` "Some" raw payload against ``T`` before it
    is wrapped back into the enum's own JSON shape.
    """
    variant = _some_variant(schema, defs)
    return next(f for f in variant.fields if f.name == "value").schema


def option_some_json_name(schema: DecodeSchema, defs: DefsMap = _EMPTY_DEFS) -> str:
    """Return the standard ``Option[T]``'s ``Some`` variant's own JSON ``$case`` tag."""
    return _some_variant(schema, defs).json_name


def host_text_to_json(
    text: str, schema: DecodeSchema, defs: DefsMap = _EMPTY_DEFS, *, agent_command_fallback: bool
) -> object:
    """Decode one host-supplied text token into a JSON-native object per *schema*.

    A ``text`` target is taken verbatim. The standard ``Agent`` enum reads
    compact shorthand, a tagged JSON object, or an ``Agent`` member
    constructor call, falling back -- when *agent_command_fallback* -- to a
    verbatim command; without the fallback, text matching none of those is a
    :class:`ValueDecodeError`. Every other target reads strict JSON, falling
    back to AgL value syntax; a failure of both reports both reasons, JSON
    and value syntax alike, since either could be what the writer intended.
    """
    resolved = _resolve(schema, defs)
    if isinstance(resolved, ScalarDecode) and resolved.kind is ScalarKind.TEXT:
        return text
    if isinstance(resolved, EnumDecode) and resolved.host_agent:
        return _decode_agent_text(
            text, resolved, defs, agent_command_fallback=agent_command_fallback
        )
    try:
        return parse_json_strict(text)
    except StrictJsonParseError as exc:
        json_error = exc.message
    try:
        node = read_value(text)
    except ValueSyntaxError as exc:
        raise ValueDecodeError(f"{exc.message}; as JSON: {json_error}", exc.start) from exc
    return value_node_to_json(node, resolved, defs)


def _agent_ctor_probe(text: str, schema: EnumDecode) -> bool:
    """Return whether *text* lexically opens an Agent member constructor call.

    Delegates the qualifier/name/``(`` lexing to
    :func:`~agm.agl.value_syntax.reader.read_ctor_head`, so this can never
    diverge from how the reader itself treats the same text (whitespace
    around ``::``, before ``(``, ...). Only the head name is checked here,
    against *schema*'s member names and ``@name`` aliases; the qualifier
    itself is left to ``_convert_enum``/``_match_variant`` to accept or
    reject once committed.
    """
    head = read_ctor_head(text)
    if head is None:
        return False
    _qualifier, name = head
    return any(name in (v.name, v.alias) for v in schema.variants)


def _decode_agent_text(
    text: str, schema: EnumDecode, defs: DefsMap, *, agent_command_fallback: bool
) -> object:
    """Decode one host Agent text token: shorthand, JSON, a member call, or a command.

    Text that lexically opens a member call (:func:`_agent_ctor_probe`)
    commits to that reading: a read or bind failure inside the call --
    including a wrong qualifier, which ``_convert_enum`` already rejects --
    propagates as a :class:`ValueDecodeError` rather than silently falling
    back to a verbatim command.
    """
    shorthand = parse_agent_shorthand(text)
    if shorthand is not None:
        return agent_spec_shape(shorthand)
    try:
        parsed = parse_json_strict(text)
    except StrictJsonParseError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    if _agent_ctor_probe(text, schema):
        try:
            node = read_value(text)
        except ValueSyntaxError as exc:
            raise ValueDecodeError(exc.message, exc.start) from exc
        return _convert_enum(node, schema, defs)
    if agent_command_fallback:
        return agent_spec_shape(AgentCommand(text))
    raise ValueDecodeError(f"cannot read {text!r} as an Agent value")
