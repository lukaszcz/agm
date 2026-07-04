"""REPL-oriented static type display helpers.

The semantic ``Type.__repr__`` implementations are intentionally compact and
canonical because they are used throughout diagnostics, schemas, and ordinary
binding echoes.  Type-focused REPL surfaces (``:type`` and bare type entries)
can afford a richer view: for nominal records and enums they print the relevant
field / constructor declarations rather than only the nominal name.
"""

from __future__ import annotations

from agm.agl.semantics.types import EnumType, RecordType, Type


def format_type_for_repl(typ: Type) -> str:
    """Return the type-focused REPL display for ``typ``.

    Primitive, container, function, agent, unit, and exception types keep their
    canonical compact representation.  Records and enums expand to AgL-like
    declarations so a user can see the available fields or constructors at the
    prompt without finding the original declaration.
    """
    if isinstance(typ, RecordType):
        return _format_record_type(typ)
    if isinstance(typ, EnumType):
        return _format_enum_type(typ)
    return repr(typ)


def format_type_echo_for_repl(typ: Type) -> str:
    """Format a bare type-entry echo for the REPL.

    Single-line type displays keep the historical ``<type: T>`` form.  Expanded
    record / enum declarations are wrapped on their own lines so the declaration
    indentation stays readable.
    """
    rendered = format_type_for_repl(typ)
    if "\n" in rendered:
        return f"<type:\n{rendered}\n>"
    return f"<type: {rendered}>"


def _format_record_type(typ: RecordType) -> str:
    name = repr(typ)
    if not typ.fields:
        return f"record {name}()"
    lines = [f"record {name}"]
    lines.extend(
        f"  {field_name}: {field_type!r}"
        for field_name, field_type in typ.fields.items()
    )
    return "\n".join(lines)


def _format_enum_type(typ: EnumType) -> str:
    lines = [f"enum {typ!r}"]
    for variant_name, fields in typ.variants.items():
        if fields:
            field_list = ", ".join(
                f"{field_name}: {field_type!r}" for field_name, field_type in fields.items()
            )
            lines.append(f"  | {variant_name}({field_list})")
        else:
            lines.append(f"  | {variant_name}")
    return "\n".join(lines)
