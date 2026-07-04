"""REPL-oriented static type display helpers.

The semantic ``Type.__repr__`` implementations are intentionally compact and
canonical because they are used throughout diagnostics, schemas, and ordinary
binding echoes.  Type-focused REPL surfaces (``:type`` and bare type entries)
can afford a richer view: for nominal records and enums they print the relevant
field / constructor declarations rather than only the nominal name.
"""

from __future__ import annotations

from agm.agl.semantics.types import EnumType, RecordType, Type
from agm.agl.typecheck.env import GenericTypeDef


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
    return format_type_text_echo_for_repl(format_type_for_repl(typ))


def format_type_text_echo_for_repl(rendered: str) -> str:
    """Wrap pre-rendered type display text in the REPL ``<type…>`` echo form."""
    if "\n" in rendered:
        return f"<type:\n{rendered}\n>"
    return f"<type: {rendered}>"


def format_generic_type_def_for_repl(name: str, gdef: GenericTypeDef) -> str:
    """Return a declaration-like display for an unapplied generic type definition."""
    display_name = f"{name}[{', '.join(gdef.type_params)}]"
    template = gdef.template
    if isinstance(template, RecordType):
        return _format_record_type_with_name(template, display_name)
    return _format_enum_type_with_name(template, display_name)


def _format_record_type(typ: RecordType) -> str:
    return _format_record_type_with_name(typ, repr(typ))


def _format_record_type_with_name(typ: RecordType, name: str) -> str:
    if not typ.fields:
        return f"record {name}()"
    lines = [f"record {name}"]
    lines.extend(
        f"  {field_name}: {field_type!r}"
        for field_name, field_type in typ.fields.items()
    )
    return "\n".join(lines)


def _format_enum_type(typ: EnumType) -> str:
    return _format_enum_type_with_name(typ, repr(typ))


def _format_enum_type_with_name(typ: EnumType, name: str) -> str:
    lines = [f"enum {name}"]
    for variant_name, fields in typ.variants.items():
        if fields:
            field_list = ", ".join(
                f"{field_name}: {field_type!r}" for field_name, field_type in fields.items()
            )
            lines.append(f"  | {variant_name}({field_list})")
        else:
            lines.append(f"  | {variant_name}")
    return "\n".join(lines)
