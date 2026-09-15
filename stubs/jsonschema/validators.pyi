"""Minimal type stub for ``jsonschema.validators``.

Only ``extend`` is typed: AgL uses it once, to build the Draft 2020-12
validator class with a Decimal-aware ``integer`` type check (see
``agm.agl.runtime.convert.AglValidator``). The built class is not a subclass
of ``Draft202012Validator``, so ``extend`` returns ``type[Validator]``, the
protocol both satisfy.
"""

from jsonschema import Draft202012Validator, TypeChecker
from jsonschema.protocols import Validator

def extend(
    validator: type[Draft202012Validator],
    validators: object = ...,
    version: object = ...,
    type_checker: TypeChecker | None = ...,
    format_checker: object = ...,
) -> type[Validator]: ...
