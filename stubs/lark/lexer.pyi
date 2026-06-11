"""Minimal stubs for lark.lexer used by the AgL custom lexer."""

from __future__ import annotations

from typing import Iterator

class Token(str):
    type: str
    start_pos: int | None
    value: str
    line: int | None
    column: int | None
    end_line: int | None
    end_column: int | None
    end_pos: int | None

    def __new__(
        cls,
        type: str,
        value: str,
        start_pos: int | None = ...,
        line: int | None = ...,
        column: int | None = ...,
        end_line: int | None = ...,
        end_column: int | None = ...,
        end_pos: int | None = ...,
    ) -> Token: ...

class TextSlice:
    text: str
    start: int
    end: int

    def __init__(self, text: str, start: int, end: int | None = ...) -> None: ...
    def is_complete_text(self) -> bool: ...

class LexerState:
    text: str | TextSlice

    def __init__(self, text: str | TextSlice) -> None: ...

class Lexer:
    def __init__(self, lexer_conf: object) -> None: ...
    def lex(self, lexer_state: LexerState, parser_state: object) -> Iterator[Token]: ...
