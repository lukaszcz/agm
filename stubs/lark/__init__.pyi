"""Minimal lark stubs for strict mypy."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from lark.lexer import Token
from lark.tree import Meta

_T = TypeVar("_T")
_TC = TypeVar("_TC", bound=type)

# The module-level logger instance.
logger: logging.Logger

class Tree:
    data: str
    children: list[Token | Tree]
    meta: Meta

    def __init__(self, data: str, children: list[Token | Tree]) -> None: ...
    """Position metadata attached to a Tree node by propagate_positions=True."""

    line: int
    column: int
    end_line: int
    end_column: int
    start_pos: int
    end_pos: int
    empty: bool

class GrammarError(Exception): ...

class UnexpectedInput(Exception):
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

class Transformer:
    """Base class for Lark tree transformers."""

    def transform(self, tree: Tree) -> object: ...

# v_args decorator: enables meta=True to pass Tree.meta as first arg.
# When decorating a class, returns the class unchanged (identity on the type).
def v_args(
    *,
    meta: bool = ...,
    inline: bool = ...,
    tree: bool = ...,
    wrapper: Callable[..., Any] | None = ...,
) -> Callable[[_TC], _TC]: ...

class Lark:
    def __init__(
        self,
        grammar: str,
        *,
        parser: str = ...,
        lexer: Any = ...,
        propagate_positions: bool = ...,
        maybe_placeholders: bool = ...,
        start: str = ...,
        debug: bool = ...,
        keep_all_tokens: bool = ...,
    ) -> None: ...

    def parse(self, text: str, start: str | None = ...) -> Tree: ...

    def lex(self, text: str) -> Iterator[Token]: ...
