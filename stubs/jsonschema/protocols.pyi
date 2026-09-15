"""Minimal type stub for ``jsonschema.protocols``.

Only ``Validator`` is typed: the protocol every jsonschema validator class
(including one built by ``jsonschema.validators.extend``) satisfies. Typed
with only the constructor and ``iter_errors``, the pieces agm uses (see
``agm.agl.runtime.convert.AglValidator``).
"""

from collections.abc import Iterator
from typing import Protocol

from jsonschema import ValidationError

class Validator(Protocol):
    def __init__(self, schema: object) -> None: ...
    def iter_errors(self, instance: object) -> Iterator[ValidationError]: ...
