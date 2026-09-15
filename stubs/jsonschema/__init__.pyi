"""Minimal type stubs for jsonschema.

Only the pieces used by agm are typed: ``ValidationError`` (with the
attributes the codec inspects), the ``Draft202012Validator`` class used to
collect all validation errors via ``iter_errors``, and ``TypeChecker``
(``Draft202012Validator.TYPE_CHECKER``), used to redefine the ``integer``
type check to also accept an integral ``Decimal``.  The stubs are
intentionally minimal: we only type what we need and avoid exposing ``Any``
to our strictly-typed codebase.
"""

from collections.abc import Callable, Iterator, Sequence

class ValidationError(Exception):
    message: str
    validator: str
    validator_value: object
    instance: object
    path: Sequence[object]
    absolute_path: Sequence[object]
    context: list["ValidationError"]
    json_path: str
    def __init__(self, message: str, **kwargs: object) -> None: ...

class TypeChecker:
    def is_type(self, instance: object, type: str) -> bool: ...
    def redefine(
        self, type: str, fn: Callable[["TypeChecker", object], bool]
    ) -> "TypeChecker": ...

class Draft202012Validator:
    TYPE_CHECKER: TypeChecker
    def __init__(self, schema: object) -> None: ...
    def iter_errors(self, instance: object) -> Iterator[ValidationError]: ...
