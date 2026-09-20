"""Shared TOML parsing and manipulation helpers.

:func:`parse_toml_doc` is the one place a TOML document from outside is read,
so config, manifests and dependency files all get the same lone-surrogate
rejection.
"""

from __future__ import annotations

import re
from pathlib import Path

import tomlkit
from tomlkit.exceptions import InvalidUnicodeValueError
from tomlkit.items import Item, Table
from tomlkit.toml_document import TOMLDocument

from agm.util.unicode import holds_surrogate

TomlDict = dict[str, object]

# A TOML ``\U0000Dxxx`` escape: the 8-digit spelling of a lone surrogate that
# tomlkit accepts even though it rejects the 4-digit ``\uDxxx`` form.
_TOML_SURROGATE_ESCAPE = re.compile(r"\\U0000[dD][89a-fA-F]")


def _toml_linecol(text: str, index: int) -> tuple[int, int]:
    """Return the 1-based (line, col) of *index* in *text*, tomlkit's own convention."""
    line = text.count("\n", 0, index) + 1
    col = index - text.rfind("\n", 0, index)
    return line, col


def parse_toml_doc(text: str) -> TOMLDocument:
    """Parse *text* as TOML, rejecting a lone surrogate spelled as ``\\U0000Dxxx``.

    tomlkit already rejects the 4-digit ``\\uD800`` escape but decodes the
    8-digit ``\\U0000D800`` spelling of the same code point, so this raises
    the same :exc:`~tomlkit.exceptions.InvalidUnicodeValueError` for that form.
    """
    doc = tomlkit.parse(text)
    match = _TOML_SURROGATE_ESCAPE.search(text)
    if match is not None and holds_surrogate(doc.unwrap()):
        line, col = _toml_linecol(text, match.start())
        raise InvalidUnicodeValueError(line, col)
    return doc


def toml_dict(value: object) -> TomlDict:
    """Coerce *value* to a ``TomlDict``, returning an empty dict for non-dicts."""

    if isinstance(value, dict):
        return dict(value)
    return {}


def load_toml_file(path: Path) -> TomlDict:
    """Parse a TOML file and return its contents as a ``TomlDict``."""

    with path.open("r", encoding="utf-8") as handle:
        doc = parse_toml_doc(handle.read())
    return toml_dict(doc.unwrap())


def load_toml_doc(path: Path) -> TOMLDocument:
    """Parse a TOML file and return a round-trippable ``TOMLDocument``."""

    with path.open("r", encoding="utf-8") as handle:
        return parse_toml_doc(handle.read())


def _get_or_create_table(doc: TOMLDocument, table_name: str) -> Table:
    """Return the ``[table_name]`` table from *doc*, creating it if absent."""

    if table_name in doc:
        existing: Item = doc[table_name]
        if isinstance(existing, Table):
            return existing
    new_table = tomlkit.table()
    doc.add(table_name, new_table)
    return new_table


def set_toml_table_value(doc: TOMLDocument, table_name: str, key: str, value: str | bool) -> None:
    """Set *key* = *value* inside ``[*table_name*]``, creating the table if absent."""

    table = _get_or_create_table(doc, table_name)
    table[key] = value


def empty_toml_doc() -> TOMLDocument:
    """Create and return an empty ``TOMLDocument``."""

    return tomlkit.document()


def dumps_toml(doc: TOMLDocument) -> str:
    """Serialize a ``TOMLDocument`` back to a TOML string."""

    return tomlkit.dumps(doc)
