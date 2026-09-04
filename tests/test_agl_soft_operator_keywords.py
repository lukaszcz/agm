"""Operator words are soft keywords: operators in operator position, names elsewhere.

`or`, `and`, `not`, `is`, `in`, `to`, `downto`, `step` and `with` spell AgL's
built-in operators, but nothing about an operator requires its spelling to be
reserved everywhere. After a `.` or a `::`, in a field declaration, or in a
named-argument label, only a name can appear, so the lexer leaves these words
as plain names there. `by` carries no syntactic role at all: the range stride
is spelled `step`.
"""

from __future__ import annotations

import pytest

from agm.agl import PipelineDriver
from agm.agl.lexer import tokenize
from tests._agl_helpers import run_inline_command

OPERATOR_WORDS = ("or", "and", "not", "is", "in", "to", "downto", "step", "with")


def _non_layout_tokens(source: str) -> list[tuple[str, str]]:
    return [
        (token.type, str(token)) for token in tokenize(source) if not token.type.startswith("_")
    ]


@pytest.mark.parametrize("word", OPERATOR_WORDS)
def test_operator_word_after_a_dot_is_a_name(word: str) -> None:
    """A member name is the only thing that can follow `.`, so no operator is meant."""
    assert ("NAME", word) in _non_layout_tokens(f"value.{word}")


@pytest.mark.parametrize("word", OPERATOR_WORDS)
def test_operator_word_after_a_qualifier_is_a_name(word: str) -> None:
    """`Type::word` declares or references a member, never applies an operator."""
    assert ("NAME", word) in _non_layout_tokens(f"def Box::{word}(self) -> int = 1")


@pytest.mark.parametrize("word", OPERATOR_WORDS)
def test_operator_word_as_a_named_argument_label_is_a_name(word: str) -> None:
    """A label is followed by `=`, which no operand can start."""
    assert ("NAME", word) in _non_layout_tokens(f"f({word} = 1)")


@pytest.mark.parametrize("word", OPERATOR_WORDS)
def test_operator_word_as_a_field_declaration_is_a_name(word: str) -> None:
    """A field declaration is followed by `:`, which no operand can start."""
    assert ("NAME", word) in _non_layout_tokens(f"record R\n  {word}: int\n")


def test_infix_words_after_an_operand_are_operators() -> None:
    """Following an operand is what makes an infix word an operator."""
    tokens = _non_layout_tokens("a or b and c in d")

    assert ("OR", "or") in tokens
    assert ("AND", "and") in tokens
    assert ("IN", "in") in tokens


def test_not_before_an_operand_is_an_operator() -> None:
    """`not` is prefix, so it is an operator exactly where an operand may start."""
    tokens = _non_layout_tokens("x is not Ready and not ready")

    assert [token for token in tokens if token[1] == "not"] == [("NOT", "not"), ("NOT", "not")]


def test_operator_words_name_record_fields_and_arguments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A record can declare, construct and read fields named for every operator word."""
    fields = "\n".join(f"  {word}: int" for word in OPERATOR_WORDS)
    arguments = ", ".join(f"{word} = {index}" for index, word in enumerate(OPERATOR_WORDS))
    reads = "\n".join(f"  print value.{word}" for word in OPERATOR_WORDS)
    source = (
        f"record R\n{fields}\n\n"
        f"program def main() -> unit =\n  let value = R({arguments})\n{reads}\n"
    )
    result = run_inline_command(PipelineDriver(), source)

    assert list(result.diagnostics) == []
    assert result.error is None
    assert capsys.readouterr().out == "".join(f"{index}\n" for index in range(len(OPERATOR_WORDS)))


def test_operator_words_name_methods(capsys: pytest.CaptureFixture[str]) -> None:
    """Methods named for operator words declare and dispatch like any other."""
    result = run_inline_command(
        PipelineDriver(),
        """
record Flag
  set: bool

def Flag::or(self, other: bool) -> bool = self.set or other
def Flag::not(self) -> bool = not self.set

program def main() -> unit =
  let flag = Flag(set = false)
  print flag.or(true)
  print flag.not()
""",
    )

    assert list(result.diagnostics) == []
    assert result.error is None
    assert capsys.readouterr().out == "true\ntrue\n"


def test_operators_still_apply_in_operator_position(capsys: pytest.CaptureFixture[str]) -> None:
    """Demotion changes nothing about how the operators themselves behave."""
    result = run_inline_command(
        PipelineDriver(),
        """
record Point
  x: int
  y: int

program def main() -> unit =
  print(true or false)
  print(true and not false)
  print(2 in [1, 2, 3])
  let point = Point(x = 1, y = 2)
  let moved = point with x = 5
  print moved
""",
    )

    assert list(result.diagnostics) == []
    assert result.error is None
    assert capsys.readouterr().out == "true\ntrue\ntrue\nPoint(x = 5, y = 2)\n"


def test_range_stride_is_spelled_step(capsys: pytest.CaptureFixture[str]) -> None:
    """`step` gives a range its stride."""
    result = run_inline_command(
        PipelineDriver(),
        """
program def main() -> unit =
  for i in 1 to 5 step 2 do
    print i
  done
  for i in 5 downto 1 step 2 do
    print i
  done
""",
    )

    assert list(result.diagnostics) == []
    assert result.error is None
    assert capsys.readouterr().out == "1\n3\n5\n5\n3\n1\n"


def test_by_is_an_ordinary_name(capsys: pytest.CaptureFixture[str]) -> None:
    """`by` carries no syntactic role, so it is available as a plain identifier."""
    result = run_inline_command(
        PipelineDriver(),
        """
def by(value: int) -> int = value * 2

program def main() -> unit =
  let downto = 3
  print by(downto)
""",
    )

    assert list(result.diagnostics) == []
    assert result.error is None
    assert capsys.readouterr().out == "6\n"
