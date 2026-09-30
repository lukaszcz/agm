"""Decoding of an AgL ``ParsePolicy`` value into an attempt count."""

from __future__ import annotations

from agm.agl.ir.builtin_nominals import BuiltinNominals
from agm.agl.semantics.values import IntValue, RecordValue

__all__ = ["decode_max_attempts"]


def decode_max_attempts(value: RecordValue, nominals: BuiltinNominals) -> int:
    """Return the attempts *value* allows: 1 for ``Abort``, ``1 + n`` for ``Retry(n)``.

    Members resolve through the program's live ``ParsePolicy`` declaration, the
    one a written policy is typed against. Raises ``ValueError`` for a negative ``n``.
    """
    if value.nominal != nominals.resolve_member("ParsePolicy", "Retry").nominal:
        return 1
    retries = value.fields["n"]
    assert isinstance(retries, IntValue)
    if retries.value < 0:
        raise ValueError(f"Retry(n = {retries.value}) must not be negative")
    return 1 + retries.value
