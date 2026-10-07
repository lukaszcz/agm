"""AgL value syntax -> JSON-native decoding: the host value-decode boundary.

Every host-supplied text value (program-parameter tokens, ``@opt-env``
variables, TOML config strings, engine-setting literals) is either strict
JSON that decodes into its slot or one AgL value-syntax literal (``agm.agl.value_syntax``).
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
its own text conventions (a tagged JSON object, an ``Agent`` member
constructor call, shorthand, or -- optionally -- a verbatim command), and
every other target reads strict JSON when that decodes into the slot, else
value syntax.
:func:`host_param_text_to_json` is its host parameter/config form, and
:func:`host_data_to_json` reads native config data by applying that form to
every string nested in it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Literal, assert_never, cast

from agm.agent.spec import AgentCommand
from agm.agent.values import AgentShorthandError, agent_spec_shape, parse_agent_shorthand
from agm.agl.ir.contracts import (
    ArrayDecode,
    DecodeSchema,
    DictDecode,
    DictKeyForm,
    EnumDecode,
    FieldDecode,
    RecordDecode,
    ScalarDecode,
    ScalarKind,
    VariantDecode,
    is_plain_enum,
)
from agm.agl.ir.ids import NominalId
from agm.agl.runtime.convert import (
    StrictJsonParseError,
    decode_value,
    parse_json_strict,
    resolve_decode,
)
from agm.agl.runtime.serialize import JsonShaped, stringified_key_text
from agm.agl.semantics.arguments import (
    ArgumentBindingError,
    ArgumentBindingErrorKind,
    BindParam,
    bind_arguments,
)
from agm.agl.semantics.values import UNIT_VALUE, Value
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
    "host_data_to_json",
    "host_param_text_to_json",
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
    whose arguments bind through the shared zone binder. A dict inside a
    ``json`` slot takes text keys only. *defs* resolves a
    ``RefDecode`` node exactly like :func:`~agm.agl.runtime.convert.decode_value`.

    :raises ValueDecodeError: on any type/shape mismatch.
    """
    resolved = resolve_decode(schema, defs)
    match resolved:
        case ScalarDecode(kind=kind):
            return _convert_scalar(node, kind)
        case ArrayDecode(elem=elem):
            if not isinstance(node, ArrayNode):
                raise ValueDecodeError(f"expected an array, got {_node_kind(node)}", node.start)
            return [value_node_to_json(item, elem, defs) for item in node.items]
        case DictDecode(key_form=key_form, key=key_schema, value=value_schema):
            if not isinstance(node, DictNode):
                raise ValueDecodeError(f"expected a dict, got {_node_kind(node)}", node.start)
            return _convert_dict(node, key_form, key_schema, value_schema, defs)
        case RecordDecode():
            return _convert_record(node, resolved, defs)
        case EnumDecode():
            return _convert_enum(node, resolved, defs)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _convert_scalar(node: ValueNode, kind: ScalarKind) -> object:
    match kind:
        case ScalarKind.JSON:
            return _convert_json(node)
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


def _convert_json(node: ValueNode) -> object:
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
        return [_convert_json(item) for item in node.items]
    if isinstance(node, DictNode):
        return _object_of(
            (cast(str, _convert_scalar(e.key, ScalarKind.TEXT)), _convert_json(e.value), e.start)
            for e in node.entries
        )
    raise ValueDecodeError("a constructor is not valid inside json", node.start)


def _convert_dict(
    node: DictNode,
    key_form: DictKeyForm,
    key_schema: DecodeSchema,
    value_schema: DecodeSchema,
    defs: DefsMap,
) -> object:
    """Convert a dict *node* into the JSON wire shape *key_form* names."""
    return _dict_wire(
        key_form,
        (
            (
                value_node_to_json(entry.key, key_schema, defs),
                value_node_to_json(entry.value, value_schema, defs),
                entry.start,
            )
            for entry in node.entries
        ),
    )


def _dict_wire(key_form: DictKeyForm, pairs: Iterable[tuple[object, object, int | None]]) -> object:
    """Lay converted ``(key, value, offset)`` *pairs* out in the wire shape *key_form* names."""
    match key_form:
        case DictKeyForm.OBJECT_TEXT:
            return _object_of((cast(str, key), value, at) for key, value, at in pairs)
        case DictKeyForm.OBJECT_STRINGIFIED:
            return _object_of(
                (stringified_key_text(cast(JsonShaped, key)), value, at) for key, value, at in pairs
            )
        case DictKeyForm.ENTRIES:
            return [{"key": key, "value": value} for key, value, _ in pairs]
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _object_of(pairs: Iterable[tuple[str, object, int | None]]) -> dict[str, object]:
    """Build an object from ``(wire key, value, offset)`` *pairs*, rejecting a repeated key."""
    result: dict[str, object] = {}
    for key, value, at in pairs:
        if key in result:
            raise ValueDecodeError(f"duplicate dict key {key!r}", at)
        result[key] = value
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


def _convert_enum(node: ValueNode, schema: EnumDecode, defs: DefsMap) -> object:
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
    if is_plain_enum(schema):
        return variant.json_name
    return {"$case": variant.json_name, **payload}


def _convert_ctor_args(
    node: CtorNode, fields: "tuple[FieldDecode, ...]", type_label: str, defs: DefsMap
) -> dict[str, object]:
    """Bind and convert one constructor call's arguments against *fields*.

    A bare constructor (``node.args is None``) is legal when every field
    carries a declared default: it reads exactly like an empty argument list.
    An omitted argument for a defaulted field is simply left out of the
    returned dict -- the same shape a JSON source that omits the key produces
    -- so it fills through ``runtime.convert.decode_value``'s own
    ``default_resolver`` at the value boundary.
    """
    if node.args is None:
        missing = [f.name for f in fields if f.default_index is None]
        if missing:
            names = ", ".join(missing)
            raise ValueDecodeError(f"{type_label} requires arguments: {names}", node.start)
        return {}
    alias_map: dict[str, str] = {}
    for field in fields:
        alias_map[field.name] = field.name
        if field.alias is not None:
            alias_map[field.alias] = field.name
    bind_params = [
        BindParam(name=f.name, kind=f.zone, has_default=f.default_index is not None) for f in fields
    ]
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
        if bound_arg is None:
            continue
        result[field.json_name] = value_node_to_json(bound_arg.value, field.schema, defs)
    return result


#: `_convert_ctor_args` checks a named argument against the field alias map
#: before calling `bind_arguments`, so UNKNOWN_NAME never reaches `_binding_error`.
type _CtorBindingErrorKind = Literal[
    ArgumentBindingErrorKind.MISSING_REQUIRED,
    ArgumentBindingErrorKind.DUPLICATE,
    ArgumentBindingErrorKind.POSITIONAL_ONLY_BY_NAME,
    ArgumentBindingErrorKind.TOO_MANY_POSITIONAL,
    ArgumentBindingErrorKind.POSITIONAL_IN_NAMED_ONLY,
]


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
    # Every named argument's name is checked against the field alias map
    # before `bind_arguments` runs, so UNKNOWN_NAME never reaches here.
    kind = cast(_CtorBindingErrorKind, exc.kind)
    match kind:
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
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)
    return ValueDecodeError(message, offset)


def _some_variant(schema: DecodeSchema, defs: DefsMap) -> VariantDecode:
    """Return the standard ``Option[T]``'s ``Some`` variant from its enum schema.

    *schema* is always the standard ``Option`` enum schema -- callers never
    pass anything else -- so its shape is looked up directly rather than
    re-checked here.
    """
    resolved = cast(EnumDecode, resolve_decode(schema, defs))
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
    a tagged JSON object, an ``Agent`` member constructor call, or compact
    shorthand, falling back -- when *agent_command_fallback* -- to a
    verbatim command; without the fallback, text matching none of those is a
    :class:`ValueDecodeError`. Malformed native shorthand (see
    :func:`parse_agent_shorthand`) is an error even with the fallback.
    Every other target reads strict JSON when it parses AND decodes against
    *schema* (:func:`_fits_schema`), else AgL value syntax -- so JSON text that
    is not a wire value of the slot (``{}`` for an entries-form dict) still
    gets its value-syntax reading, at every depth of nested host strings. Text
    that is not JSON and is no value either reports both reasons; JSON that
    fits nowhere is returned as parsed, for the caller's own decode to reject
    with its precise message.
    """
    resolved = resolve_decode(schema, defs)
    if isinstance(resolved, ScalarDecode) and resolved.kind is ScalarKind.TEXT:
        return text
    if isinstance(resolved, EnumDecode) and resolved.host_agent:
        return _decode_agent_text(
            text, resolved, defs, agent_command_fallback=agent_command_fallback
        )
    json_error: str | None
    try:
        parsed = parse_json_strict(text)
    except StrictJsonParseError as exc:
        json_error = exc.message
    else:
        json_error = None
        if _fits_schema(parsed, resolved, defs):
            return parsed
    try:
        return value_node_to_json(read_value(text), resolved, defs)
    except (ValueSyntaxError, ValueDecodeError) as exc:
        if json_error is None:
            return parsed
        if isinstance(exc, ValueDecodeError):
            raise
        raise ValueDecodeError(f"{exc.message}; as JSON: {json_error}", exc.start) from exc


def _shape_only_default(_nominal: NominalId, _index: int) -> Value:
    """Stand in for a defaulted field's value: a wire-shape check never needs the real one."""
    return UNIT_VALUE


def _fits_schema(parsed: object, schema: DecodeSchema, defs: DefsMap) -> bool:
    """Return whether JSON *parsed* decodes into *schema* (defaults checked by shape only)."""
    try:
        decode_value(schema, parsed, defs, default_resolver=_shape_only_default)
    except ValueError:
        return False
    return True


def host_param_text_to_json(text: str, schema: DecodeSchema, defs: DefsMap = _EMPTY_DEFS) -> object:
    """Decode one host parameter or config string into a JSON-native object per *schema*.

    :func:`host_text_to_json` with the ``Agent`` command fallback, plus one
    reading of its own: a plain enum also takes a member's bare JSON name.
    """
    resolved = resolve_decode(schema, defs)
    if (
        isinstance(resolved, EnumDecode)
        and is_plain_enum(resolved)
        and any(text == variant.json_name for variant in resolved.variants)
    ):
        return text
    return host_text_to_json(text, resolved, defs, agent_command_fallback=True)


def host_data_to_json(value: object, schema: DecodeSchema, defs: DefsMap = _EMPTY_DEFS) -> object:
    """Read every string nested in JSON-native host config data as its slot's host text.

    A native config array or table is JSON data, except that a string inside
    it reads exactly as a top-level config string of that slot's type does
    (:func:`host_param_text_to_json`). A ``json`` slot keeps its data as is,
    and data matching no slot is returned unchanged for validation to reject.
    """
    resolved = resolve_decode(schema, defs)
    if isinstance(resolved, ScalarDecode) and resolved.kind is ScalarKind.JSON:
        return value
    if isinstance(value, str):
        return host_param_text_to_json(value, resolved, defs)
    if isinstance(value, list) and isinstance(resolved, ArrayDecode):
        return [host_data_to_json(item, resolved.elem, defs) for item in value]
    if isinstance(value, list) and isinstance(resolved, DictDecode):
        return [_host_entry_to_json(item, resolved, defs) for item in value]
    if not isinstance(value, dict):
        return value
    return _host_table_to_json(value, resolved, defs)


def _host_entry_to_json(item: object, schema: DictDecode, defs: DefsMap) -> object:
    """Read a native ``{key, value}`` entry table through *schema*'s key and value slots."""
    if not isinstance(item, dict) or "key" not in item or "value" not in item:
        return item
    entry: Mapping[str, object] = item
    return {
        **entry,
        "key": host_data_to_json(item["key"], schema.key, defs),
        "value": host_data_to_json(item["value"], schema.value, defs),
    }


def _host_table_to_json(data: Mapping[str, object], schema: DecodeSchema, defs: DefsMap) -> object:
    """Read each entry of the table *data* as host data for the slot *schema* gives its key.

    A table in a dict slot is laid out in the key's wire form, each table key
    read as a native string of the key slot is.
    """
    match schema:
        case DictDecode(key_form=key_form, key=key_schema, value=item_schema):
            return _dict_wire(
                key_form,
                (
                    (
                        host_data_to_json(key, key_schema, defs),
                        host_data_to_json(item, item_schema, defs),
                        None,
                    )
                    for key, item in data.items()
                ),
            )
        case RecordDecode(fields=record_fields):
            fields = record_fields
        case EnumDecode(variants=variants):
            tag = data.get("$case")
            fields = next((v.fields for v in variants if v.json_name == tag), ())
        case _:
            return data
    slots = {field.json_name: field.schema for field in fields}
    return {
        key: host_data_to_json(item, slots[key], defs) if key in slots else item
        for key, item in data.items()
    }


def _agent_ctor_probe(text: str, schema: EnumDecode) -> bool:
    """Return whether *text* lexically is an Agent member constructor call or bare name.

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
    """Decode one host Agent text token: JSON, a member call, shorthand, or a command.

    Text that lexically opens a member call (:func:`_agent_ctor_probe`)
    commits to that reading: a read or bind failure inside the call --
    including a wrong qualifier, which ``_convert_enum`` already rejects --
    propagates as a :class:`ValueDecodeError` rather than silently falling
    back to a verbatim command. Shorthand is tried after both, so a ``/``
    inside a JSON object or a call is never read as a provider/model split;
    malformed native shorthand (see :func:`parse_agent_shorthand`) likewise
    commits and errors.

    Whitespace-only text is always an error, before any other reading is
    tried: an empty command is never a meaningful ``AgentCommand`` fallback.
    """
    if not text.strip():
        raise ValueDecodeError(f"expected a non-empty Agent value, got {text!r}")
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
    try:
        shorthand = parse_agent_shorthand(text)
    except AgentShorthandError as exc:
        raise ValueDecodeError(str(exc)) from exc
    if shorthand is not None:
        return agent_spec_shape(shorthand)
    if agent_command_fallback:
        return agent_spec_shape(AgentCommand(text))
    raise ValueDecodeError(f"cannot read {text!r} as an Agent value")
