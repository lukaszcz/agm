"""The single error raised by AgL value-syntax lexing and reading."""

from __future__ import annotations


class ValueSyntaxError(Exception):
    """A malformed literal or value; offsets index the scanned source."""

    def __init__(self, message: str, start: int, end: int) -> None:
        super().__init__(message)
        self.message = message
        self.start = start
        self.end = end
