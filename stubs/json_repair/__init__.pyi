"""Minimal type stubs for json-repair.

Only the ``repair_json`` entry point is used by agm — we declare it to
return ``str`` (the non-``return_objects`` / non-``logging`` overload) so
that the surrounding code is fully typed without ``Any`` contamination.
"""

def repair_json(
    json_str: str = ...,
    *,
    return_objects: bool = ...,
    skip_json_loads: bool = ...,
    logging: bool = ...,
    ensure_ascii: bool = ...,
) -> str: ...
