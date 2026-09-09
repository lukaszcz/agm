"""Allocation regressions for text workflows, independent of machine speed."""

import tracemalloc
from collections.abc import Callable

import pytest

from agm.agl.eval.ir_interpreter import IrInterpreter
from agm.agl.lexer import tokenize
from agm.agl.semantics.values import TextValue
from tests.agl.ir_harness import lower_inline_ir


def _peak_allocation[T](operation: Callable[[], T]) -> tuple[T, int]:
    # Prime coverage instrumentation before measuring workflow allocations.
    operation()
    tracemalloc.start()
    try:
        result = operation()
        return result, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_break_from_large_text_does_not_materialize_characters() -> None:
    source = 'var first = ""\nfor ch in "' + "a" * 50_000 + '" do\n  first := ch\n  break\ndone\n'
    program = lower_inline_ir(source, default_stdlib=False)
    interpreter = IrInterpreter(program)
    result, peak = _peak_allocation(
        lambda: interpreter.run(program_symbol=program.synthetic_main_symbol)
    )
    assert result["first"] == TextValue("a")
    # Generous room for the evaluator's frames, but not 50,000 character values.
    assert peak < 256_000


@pytest.mark.parametrize("character", ["a", "界", "😀"])
def test_large_triple_template_uses_compact_dedent_storage(character: str) -> None:
    body = character * 30_000
    source = '"""\n    ' + body + '\n    %{value}\n    end\n    """'
    tokens, peak = _peak_allocation(lambda: list(tokenize(source)))
    assert [str(token) for token in tokens if token.type == "STRING_FRAGMENT"] == [
        body + "\n",
        "\nend",
    ]
    # Allow source/output copies and scanner buffers, but not per-character maps.
    assert peak < 32 * len(source)
