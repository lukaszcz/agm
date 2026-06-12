"""Minimal type stubs for jsonschema.

Only the pieces used by agm are typed: ``ValidationError`` (with the
attributes the codec inspects) and the ``Draft202012Validator`` class used to
collect all validation errors via ``iter_errors``.  The stubs are intentionally
minimal: we only type what we need and avoid exposing ``Any`` to our
strictly-typed codebase.
"""

from collections.abc import Iterator
from collections.abc import Sequence

class ValidationError(Exception):
    message: str
    validator: str
    validator_value: object
    instance: object
    path: Sequence[object]
    json_path: str
    def __init__(self, message: str, **kwargs: object) -> None: ...

class Draft202012Validator:
    def __init__(self, schema: object) -> None: ...
    def iter_errors(self, instance: object) -> Iterator[ValidationError]: ...
