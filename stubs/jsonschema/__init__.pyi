"""Minimal type stubs for jsonschema.

Only the ``validate`` function and ``ValidationError`` exception are used
by agm.  The stubs are intentionally minimal: we only type what we need
and avoid exposing ``Any`` to our strictly-typed codebase.
"""

class ValidationError(Exception):
    message: str
    def __init__(self, message: str, **kwargs: object) -> None: ...

def validate(instance: object, schema: object) -> None: ...
