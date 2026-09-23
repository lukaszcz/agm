"""How the host recognizes the standard declaration of a built-in name.

A ``builtin`` declaration in *any* standard-library module is the standard
declaration of its name; the prelude module is only the auto-imported entry
point into that surface.  These tests run a standard library whose built-ins
are spread over several ``std/*`` modules, so nothing may depend on them all
living in ``std/prelude``.
"""

from __future__ import annotations

import unittest.mock
from pathlib import Path

import pytest

from agm.agl import PipelineDriver
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import RunResult
from agm.agl.runtime.request import AgentRequest
from tests._agl_helpers import run_inline_command
from tests._process_helpers import FakeShell

_MODULES: dict[str, str] = {
    "prelude.agl": (
        "import std/config\n"
        "\n"
        "export std/errors\n"
        "export std/option\n"
        "export std/io\n"
        "export std/exec\n"
        "export std/agent\n"
        "export std/session\n"
        "export std/optional\n"
        "export std/sandbox\n"
    ),
    "errors.agl": (
        "builtin\n"
        "exception Exception\n"
        "  @arg-named message: text\n"
        "\n"
        "builtin\n"
        "exception KeyError extends Exception\n"
        "  key: text\n"
        "\n"
        "builtin\n"
        "exception IndexError extends Exception\n"
        "  index: int\n"
        "  length: int\n"
    ),
    "option.agl": ("builtin\nenum Option[T] =\n  | None\n  | Some(value: T)\n"),
    "optional.agl": (
        "import std/option::{Option}\n"
        "\n"
        "builtin\n"
        "enum Optional[T]\n"
        "  | Option::Some[T]\n"
        "  | Option::None\n"
        "  | Default\n"
    ),
    "config.agl": (
        "import std/agent::{Agent, AgentSandbox}\n"
        "import std/option::{Option}\n"
        "\n"
        'builtin var default-agent: Agent = AgentClaude("sonnet", "medium")\n'
        "builtin var default-sandbox: AgentSandbox = Disabled\n"
        "builtin var strict-json: bool = false\n"
        "builtin var timeout: Option[text] = None\n"
        "builtin var trace: bool = false\n"
        "builtin var trace-file: Option[text] = None\n"
    ),
    "io.agl": ("builtin def print[T](value: T) -> unit\nbuiltin def render[T](value: T) -> text\n"),
    "exec.agl": (
        "import std/errors::{Exception}\n"
        "\n"
        "builtin\n"
        "record ExecResult\n"
        "  stdout: text\n"
        "  exit-code: int\n"
        "  stderr: text\n"
        "  timed-out: bool\n"
        "\n"
        "builtin\n"
        "exception ExecError extends Exception\n"
        "  command: text\n"
        "  exit-code: int\n"
        "  stdout: text\n"
        "  stderr: text\n"
        "  timed-out: bool\n"
        "\n"
        "builtin def exec(command: text) -> ExecResult\n"
    ),
    "agent.agl": (
        "import std/errors::{Exception}\n"
        "import std/option::{Option}\n"
        "import std/sandbox::Sandbox\n"
        "\n"
        "builtin\n"
        "enum Agent\n"
        "  | AgentCommand(command: text)\n"
        "  | AgentClaude(model: text, thinking: text)\n"
        "  | AgentCodex(model: text, thinking: text)\n"
        "  | AgentPi(provider: text, model: text, thinking: text)\n"
        "\n"
        "builtin\n"
        "enum AgentSandbox\n"
        "  | Disabled\n"
        "  | Native\n"
        "  | std/sandbox::Sandbox\n"
        "\n"
        "builtin\n"
        "record AgentRequest\n"
        "  agent: Agent\n"
        "  prompt: text\n"
        "  target-type: Option[text]\n"
        "  format-instructions: Option[text]\n"
        "  json-schema: Option[json]\n"
        "  attempt: int\n"
        "  previous-error: Option[text]\n"
        "  metadata: json\n"
        "  sandbox: AgentSandbox\n"
        "\n"
        "builtin\n"
        "exception AgentCallError extends Exception\n"
        "  agent: Agent\n"
        "  cause: text\n"
        "  metadata: json\n"
        "\n"
        "builtin\n"
        "exception AgentParseError extends Exception\n"
        "  agent: Agent\n"
        "  target-type: text\n"
        "  expected-schema: json\n"
        "  raw: text\n"
        "  normalized-raw: text\n"
        "  validation-errors: json\n"
        "  attempts: int\n"
        "  metadata: json\n"
        "\n"
        "builtin def ask(prompt: text) -> text\n"
    ),
    "sandbox.agl": (
        "import std/option::{Option}\n"
        "import std/optional::{Optional}\n"
        "\n"
        "builtin record Sandbox(\n"
        "  memory: Optional[text] = Default,\n"
        "  swap: Optional[text] = Default,\n"
        "  settings: Option[text] = None,\n"
        "  patch: bool = true,\n"
        ")\n"
    ),
    "session.agl": (
        "import std/agent::{Agent}\n"
        "import std/errors::{Exception}\n"
        "import std/option::{Option}\n"
        "\n"
        "builtin\n"
        "enum SessionTransport\n"
        "  | Cli\n"
        "  | Rpc\n"
        "\n"
        "builtin record Session(id: text, agent: Agent, transport: SessionTransport)\n"
        "\n"
        "builtin record SessionStats(\n"
        "  input-tokens: int,\n"
        "  output-tokens: int,\n"
        "  cost: decimal,\n"
        "  context-percent: decimal,\n"
        ")\n"
        "\n"
        "builtin exception SessionError extends Exception(operation: text)\n"
        "\n"
        "builtin def Session::open(\n"
        "  agent: Agent,\n"
        "  transport: Option[SessionTransport] = Option[SessionTransport]::None,\n"
        '  name: text = "",\n'
        ") -> Session\n"
        "builtin def Session::default() -> Session\n"
        "builtin def Session::close(self) -> unit\n"
    ),
}


@pytest.fixture(name="split_stdlib")
def _split_stdlib(tmp_path: Path) -> RootSet:
    """Write a standard library whose built-ins live outside ``std/prelude``."""
    std_dir = tmp_path / "split_stdlib" / "src"
    std_dir.mkdir(parents=True)
    for name, source in _MODULES.items():
        (std_dir / name).write_text(source, encoding="utf-8")
    root = std_dir.parent
    return RootSet(roots=frozenset(), stdlib_roots=frozenset({root}))


def _run(source: str, roots: RootSet, **options: object) -> RunResult:
    return run_inline_command(PipelineDriver(**options), source, roots=roots)


def test_exceptions_declared_outside_the_prelude_are_raisable_and_catchable(
    split_stdlib: RootSet, capsys: pytest.CaptureFixture[str]
) -> None:
    result = _run(
        'let caught = try\n  raise KeyError(key = "k", message = "boom")\ncatch KeyError as e =>\n'
        "  e.key\nprint(caught)\n"
        'let indexed = try\n  raise IndexError(index = 3, length = 1, message = "oops")\n'
        "catch IndexError as e =>\n  e.index\nprint(indexed)\n",
        split_stdlib,
    )
    assert list(result.diagnostics) == [], " | ".join(d.message for d in result.diagnostics)
    assert result.error is None
    assert capsys.readouterr().out == "k\n3\n"


def test_a_host_raised_exception_matches_its_standard_library_declaration(
    split_stdlib: RootSet, capsys: pytest.CaptureFixture[str]
) -> None:
    """The host mints ``IndexError`` as the ``std/errors`` declaration itself."""
    result = _run(
        "let items = [1]\n"
        "let caught = try\n  items[5]\ncatch IndexError as e =>\n  e.length\nprint(caught)\n",
        split_stdlib,
    )
    assert list(result.diagnostics) == []
    assert result.error is None
    assert capsys.readouterr().out == "1\n"


def test_an_uncaught_standard_exception_still_spells_its_bare_name(
    split_stdlib: RootSet,
) -> None:
    result = _run("let items: array[int] = []\nprint(items[0])\n", split_stdlib)
    assert list(result.diagnostics) == []
    assert result.error is not None
    assert result.error.type_name == "IndexError"


def test_exec_dispatches_to_the_declaration_in_its_own_standard_module(
    split_stdlib: RootSet, capsys: pytest.CaptureFixture[str]
) -> None:
    shell = FakeShell([{"command": "echo hi", "stdout": "hi\n"}])
    with unittest.mock.patch("agm.core.process.run_capture_result", side_effect=shell):
        result = _run(
            'let r = exec("echo hi")\nprint(r.stdout)\nprint(r.exit-code)\n', split_stdlib
        )
    assert list(result.diagnostics) == []
    assert result.error is None
    assert capsys.readouterr().out == "hi\n0\n"


def test_a_failing_exec_raises_the_catchable_standard_exec_error(
    split_stdlib: RootSet, capsys: pytest.CaptureFixture[str]
) -> None:
    shell = FakeShell([{"command": "false", "returncode": 1}])
    with unittest.mock.patch("agm.core.process.run_capture_result", side_effect=shell):
        result = _run(
            'let code = try\n  let out: text = exec("false")\n  "no"\n'
            "catch ExecError as e =>\n  render(e.exit-code)\nprint(code)\n",
            split_stdlib,
        )
    assert list(result.diagnostics) == []
    assert result.error is None
    assert capsys.readouterr().out == "1\n"


def test_session_statics_resolve_against_a_session_declared_outside_the_prelude(
    split_stdlib: RootSet, capsys: pytest.CaptureFixture[str]
) -> None:
    result = _run(
        "def opener(a: Agent) -> Session = Session::open(a)\n"
        "def defaulted() -> Session = Session::default()\n"
        'print("ok")\n',
        split_stdlib,
    )
    assert list(result.diagnostics) == [], " | ".join(d.message for d in result.diagnostics)
    assert result.error is None
    assert capsys.readouterr().out == "ok\n"


def test_ask_dispatches_through_a_standard_module_declaration(
    split_stdlib: RootSet, capsys: pytest.CaptureFixture[str]
) -> None:
    prompts: list[str] = []

    def dispatch(request: AgentRequest) -> str:
        prompts.append(request.prompt)
        return "answered"

    result = _run(
        'let reply = ask("question")\nprint(reply)\n',
        split_stdlib,
        agent_dispatcher=dispatch,
    )
    assert list(result.diagnostics) == []
    assert result.error is None
    assert prompts == ["question"]
    assert capsys.readouterr().out == "answered\n"


def test_an_entry_declaration_overrides_the_standard_library_one(
    split_stdlib: RootSet, capsys: pytest.CaptureFixture[str]
) -> None:
    """A program's own ``builtin`` declaration still wins over the std one.

    One bare builtin name may be declared once per program, so an override of
    a name the standard library already declares is written at its own scope
    path. The host raises the index failure below, so it can only be caught
    here if the entry's declaration — not ``std/errors``'s — is the one minted.
    """
    result = _run(
        "scope Mine\n"
        "  builtin\n"
        "  exception IndexError extends Exception\n"
        "    index: int\n"
        "    length: int\n"
        "\n"
        "  def probe(items: array[int]) -> int =\n"
        "    try\n"
        "      items[5]\n"
        "    catch IndexError as e =>\n"
        "      e.index\n"
        "end Mine\n"
        "\n"
        "print(Mine::probe([7]))\n",
        split_stdlib,
    )
    assert list(result.diagnostics) == [], " | ".join(d.message for d in result.diagnostics)
    assert result.error is None
    assert capsys.readouterr().out == "5\n"


def test_reserved_fallbacks_still_serve_a_program_with_no_standard_library(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without any standard library the host's own identities answer instead."""
    result = run_inline_command(
        PipelineDriver(),
        "builtin def print[T](value: T) -> unit\n"
        'let caught = try\n  raise KeyError(key = "k", message = "boom")\n'
        "catch KeyError as e =>\n  e.key\nprint(caught)\n",
        default_stdlib=False,
    )
    assert list(result.diagnostics) == [], " | ".join(d.message for d in result.diagnostics)
    assert result.error is None
    assert capsys.readouterr().out == "k\n"


def test_an_uncaught_reserved_exception_spells_its_bare_name() -> None:
    result = run_inline_command(
        PipelineDriver(),
        'raise KeyError(key = "k", message = "boom")\n',
        default_stdlib=False,
    )
    assert list(result.diagnostics) == []
    assert result.error is not None
    assert result.error.type_name == "KeyError"
