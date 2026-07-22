"""Static-compile guard for every AgL snippet in the reference docs.

Every ```agl fenced block under ``docs/agl/reference/*.md`` is discovered and
run through AgL's full static pipeline — the same passes ``agm exec --dry-run``
performs (lex → parse → scope → typecheck → matchcompile → lower), with no
agent ever executed. This keeps the documentation's examples from silently
rotting as the language evolves.

A block's expectation is declared by an HTML comment on the line immediately
preceding its opening fence:

* ``<!-- agl-check: fragment -->`` — an illustrative fragment or deliberately
  incomplete example (references undefined names, imports absent modules, …).
  It must be rejected as a standalone program.
* ``<!-- agl-check: error -->`` — a deliberately-rejected program that
  demonstrates a static error. It MUST fail the static pipeline (the exact
  message is not asserted).

A block with no marker is the default and MUST statically compile, so a newly
added doc example is checked automatically unless it is explicitly opted out.

"Statically compiles" means the pipeline reached a lowered program. A program
that lowers but would fail only at run time — for example a required ``param``
left unbound here — still counts as compiling, exactly as ``--dry-run`` treats
it.
"""

from __future__ import annotations

import io
import re
import textwrap
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import pytest

from agm.agl import PipelineDriver
from agm.agl.runtime.agents import AgentRequest

_REFERENCE_DIR = Path(__file__).resolve().parents[1] / "docs" / "agl" / "reference"
_FENCE = re.compile(r"^[ \t]*```agl\s*$")
_MARKER = re.compile(r"<!--\s*agl-check:\s*(?P<kind>skip|fragment|error)\s*-->")


@dataclass(frozen=True)
class Snippet:
    """One ```agl fenced block extracted from a reference document."""

    path: Path
    line: int  # 1-based source line of the opening fence
    marker: str | None  # "fragment", "error", or None (must-compile)
    source: str

    @property
    def id(self) -> str:
        return f"{self.path.name}:{self.line}"


def _discover_snippets() -> list[Snippet]:
    """Collect every ```agl block and its compilation expectation."""
    snippets: list[Snippet] = []
    for path in sorted(_REFERENCE_DIR.glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        index = 0
        while index < len(lines):
            if _FENCE.match(lines[index]) is None:
                index += 1
                continue
            fence_line = index + 1
            body: list[str] = []
            index += 1
            while index < len(lines) and lines[index].strip() != "```":
                body.append(lines[index])
                index += 1
            marker: str | None = None
            if fence_line - 2 >= 0:
                match = _MARKER.search(lines[fence_line - 2])
                if match is not None:
                    marker = match.group("kind")
            snippets.append(
                Snippet(
                    path=path,
                    line=fence_line,
                    marker=marker,
                    source=textwrap.dedent("\n".join(body)),
                )
            )
            index += 1
    return snippets


_SNIPPETS = _discover_snippets()


def test_reference_docs_do_not_skip_snippets() -> None:
    """Every reference snippet must have an executable test expectation."""
    skipped = [snippet.id for snippet in _SNIPPETS if snippet.marker == "skip"]
    assert not skipped, f"reference snippets must not opt out of testing: {skipped}"


def _unused_agent(request: AgentRequest) -> str:
    """A default agent that is never invoked (the pipeline stops at --dry-run).

    Registering one mirrors ``agm exec``'s default runner floor: it satisfies
    each program's declared agents and the built-in ``ask`` so a block that
    calls an agent still reaches lowering without any agent running.
    """
    return ""


def _statically_compiles(source: str) -> tuple[bool, list[str]]:
    """Return whether *source* reaches a lowered program, plus any diagnostics.

    Runs the full static pipeline without executing anything, as
    ``agm exec --dry-run`` does. A lowered program (``executable is not None``)
    means every static pass succeeded; post-lowering, run-time-only failures
    such as an unbound ``param`` are ignored.
    """
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        driver = PipelineDriver(default_agent=_unused_agent)
        prepared = driver.prepare_program(source)
        preflight = driver.preflight_params(prepared, param_values={})
    diagnostics = [diag.message for diag in preflight.result.diagnostics]
    return preflight.executable is not None, diagnostics


@pytest.mark.parametrize("snippet", _SNIPPETS, ids=[snippet.id for snippet in _SNIPPETS])
def test_reference_doc_snippet(snippet: Snippet) -> None:
    compiles, diagnostics = _statically_compiles(snippet.source)
    if snippet.marker in {"error", "fragment"}:
        assert not compiles, (
            f"{snippet.id}: marked 'agl-check: {snippet.marker}' but it statically compiled"
        )
    else:
        assert compiles, (
            f"{snippet.id}: expected a static compile but got diagnostics: {diagnostics}"
        )
