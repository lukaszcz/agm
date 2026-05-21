"""Shared TOML parsing and manipulation helpers."""

from __future__ import annotations

from pathlib import Path

import tomlkit
from tomlkit.items import Item, Table
from tomlkit.toml_document import TOMLDocument

TomlDict = dict[str, object]


def toml_dict(value: object) -> TomlDict:
    """Coerce *value* to a ``TomlDict``, returning an empty dict for non-dicts."""

    if isinstance(value, dict):
        return dict(value)
    return {}


def load_toml_file(path: Path) -> TomlDict:
    """Parse a TOML file and return its contents as a ``TomlDict``."""

    with path.open("r", encoding="utf-8") as handle:
        doc = tomlkit.load(handle)
    return toml_dict(doc.unwrap())


def load_toml_doc(path: Path) -> TOMLDocument:
    """Parse a TOML file and return a round-trippable ``TOMLDocument``."""

    with path.open("r", encoding="utf-8") as handle:
        return tomlkit.load(handle)


def _get_or_create_table(doc: TOMLDocument, table_name: str) -> Table:
    """Return the ``[table_name]`` table from *doc*, creating it if absent."""

    if table_name in doc:
        existing: Item = doc[table_name]
        if isinstance(existing, Table):
            return existing
    new_table = tomlkit.table()
    doc.add(table_name, new_table)
    return new_table


def set_toml_table_value(doc: TOMLDocument, table_name: str, key: str, value: str) -> None:
    """Set *key* = *value* inside ``[*table_name*]``, creating the table if absent."""

    table = _get_or_create_table(doc, table_name)
    table[key] = value


def dumps_toml(doc: TOMLDocument) -> str:
    """Serialize a ``TOMLDocument`` back to a TOML string."""

    return tomlkit.dumps(doc)
