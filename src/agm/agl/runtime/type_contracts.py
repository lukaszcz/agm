"""``agl.TypeContract``: a type-directed extern's target type as its companion sees it."""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import NamedTuple, cast

from agm.agl.ir.contracts import ContractRequest, TypeNode, TypeNodeRef, TypeTreeEntry
from agm.agl.ir.ids import NominalId


class TypeContractField(NamedTuple):
    """One record or member field: declared name, ``@doc``, and type."""

    name: str
    doc: str | None
    contract: TypeContract


class TypeContract:
    """Immutable description of one target type, possibly cyclic for a recursive target.

    ``kind`` is ``text``/``int``/``decimal``/``bool``/``json``/``array``/``dict``/
    ``record``/``enum``/``member``. ``nominal`` is the synthesized class of a
    record, enum, or member (``None`` otherwise); a companion constructs a
    record or member by calling it with declared field names. ``fields`` maps
    JSON names to fields and ``members`` maps JSON tags to member contracts,
    both in declaration order; ``items``/``values`` are an array's elements and
    a dict's values. Only the host constructs one.
    """

    __slots__ = (
        "kind",
        "label",
        "doc",
        "nominal",
        "fields",
        "members",
        "items",
        "values",
        "_fragment",
        "_defs",
    )
    kind: str
    label: str
    doc: str | None
    nominal: type[object] | None
    fields: Mapping[str, TypeContractField]
    members: Mapping[str, TypeContract]
    items: TypeContract | None
    values: TypeContract | None
    _fragment: str
    _defs: str | None

    def __new__(cls) -> TypeContract:
        raise TypeError("TypeContract is built by the host")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("TypeContract is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("TypeContract is immutable")

    @property
    def schema(self) -> dict[str, object]:
        """A fresh copy of this type's self-contained JSON Schema."""
        schema = _load_object(self._fragment)
        if self._defs is not None:
            schema["$defs"] = _load_object(self._defs)
        return schema

    def __repr__(self) -> str:
        return f"TypeContract({self.kind}, {self.label!r})"


def build_type_contract(
    request: ContractRequest, classes: Mapping[NominalId, type[object]]
) -> TypeContract:
    """Build *request*'s ``TypeContract`` graph, one object per ``$defs`` key.

    *classes* supplies each nominal's synthesized class. Every schema carries
    the request's root ``$defs`` its ``$ref``s resolve against. A root naming a
    ``$defs`` key is that key's shared object when their labels agree, and
    otherwise a distinct object carrying the target's own label.
    """
    tree = request.type_tree
    assert tree is not None and request.json_schema is not None
    root_defs = _load_object(request.json_schema).get("$defs")
    defs = None if root_defs is None else json.dumps(root_defs)
    bodies = dict(tree.defs)
    by_key: dict[str, TypeContract] = {}

    def resolve(entry: TypeTreeEntry) -> TypeContract:
        if isinstance(entry, TypeNode):
            return build(entry, entry.label, object.__new__(TypeContract))
        cached = by_key.get(entry.key)
        if cached is None:
            cached = object.__new__(TypeContract)
            by_key[entry.key] = cached
            body = bodies[entry.key]
            build(body, body.label, cached)
        return cached

    def build(node: TypeNode, label: str, contract: TypeContract) -> TypeContract:
        attrs: dict[str, object] = {
            "kind": node.kind.value,
            "label": label,
            "doc": node.doc,
            "nominal": None if node.nominal is None else classes[node.nominal],
            "fields": MappingProxyType(
                {
                    field.json_name: TypeContractField(field.name, field.doc, resolve(field.node))
                    for field in node.fields
                }
            ),
            "members": MappingProxyType({tag: resolve(member) for tag, member in node.members}),
            "items": None if node.items is None else resolve(node.items),
            "values": None if node.values is None else resolve(node.values),
            "_fragment": node.schema,
            "_defs": defs,
        }
        for name, value in attrs.items():
            object.__setattr__(contract, name, value)
        return contract

    root = tree.root
    label = request.target_type_label
    if isinstance(root, TypeNodeRef):
        body = bodies[root.key]
        if body.label == label:
            return resolve(root)
        root = body
    return build(root, label, object.__new__(TypeContract))


def _load_object(text: str) -> dict[str, object]:
    """Parse a JSON Schema object."""
    return cast("dict[str, object]", json.loads(text))
