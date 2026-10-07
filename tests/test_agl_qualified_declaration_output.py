"""Runtime discriminators for qualified declarations with identical static types."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl import PipelineDriver
from tests._agl_helpers import agl_roots, repl_session, run_inline_code
from tests.agl.qualifier_support import all_groupings, eval_grouped_final

_HEADER = (
    "import palette",
    "scope palette\n  let value: int = 22\n  def read() -> int = 22\nend palette",
)
_PROBE = """\
print palette::value
print /palette::value
print ::palette::value
print(palette::read())
print(/palette::read())
let own = ::palette::read
let imported = /palette::read
print(own())
print(imported())
print(palette::only-imported())
"""
_EXPECTED = "22\n11\n22\n22\n11\n22\n11\n33\n"


@pytest.mark.parametrize(
    ("mode", "sizes"),
    [
        pytest.param("file", (), id="file"),
        pytest.param("inline", (), id="inline"),
        *(
            pytest.param("repl", sizes, id="repl-" + ".".join(map(str, sizes)))
            for sizes in all_groupings(len(_HEADER) + 1)
        ),
    ],
)
def test_qualified_same_typed_declarations_select_distinct_values(
    mode: str,
    sizes: tuple[int, ...],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "palette.agl").write_text(
        "let value: int = 11\ndef read() -> int = 11\ndef only-imported() -> int = 33\n",
        encoding="utf-8",
    )
    if mode == "repl":
        session = repl_session(cwd=tmp_path)
        try:
            result = eval_grouped_final(session, (*_HEADER, _PROBE), sizes)
            assert result.ok, (result.diagnostics, result.error)
        finally:
            session.close()
    else:
        runtime = PipelineDriver(resolve_agent_spec=None, get_sandbox_context=None)
        source = "\n".join(_HEADER) + "\n"
        if mode == "file":
            source += "program def main() -> unit =\n" + "".join(
                "  " + line + "\n" for line in _PROBE.splitlines()
            )
            entry = tmp_path / "entry.agl"
            entry.write_text(source, encoding="utf-8")
            outcome = runtime.run(entry.read_text(encoding="utf-8"), entry_path=entry)
        else:
            outcome = run_inline_code(runtime, source + _PROBE, roots=agl_roots(tmp_path))
        assert outcome.ok, (outcome.diagnostics, outcome.error)
    assert capsys.readouterr().out == _EXPECTED
