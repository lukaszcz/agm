"""Stubs for lark.exceptions."""

from __future__ import annotations

from lark.lexer import Token
from lark import Tree

class LarkError(Exception): ...

class UnexpectedInput(LarkError):
    line: int
    column: int
    pos_in_stream: int

class UnexpectedToken(UnexpectedInput):
    token: Token
    expected: set[str]

class UnexpectedCharacters(UnexpectedInput):
    char: str

class UnexpectedEOF(UnexpectedInput):
    expected: set[str]

class VisitError(LarkError):
    rule: str
    obj: Tree | Token
    orig_exc: Exception

    def __init__(self, rule: str, obj: Tree | Token, orig_exc: Exception) -> None: ...
