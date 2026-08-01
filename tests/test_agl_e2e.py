"""End-to-end behavior tests for AgL (the `agm exec` workflow DSL).

This suite is the TDD specification for the AgL implementation. Every
tests/agl/programs/**/*.agl file is a complete AgL program executed under
each scenario in its sidecar `<name>.scenarios.json`; every
tests/agl/rejections/**/*.agl file is an invalid program that the static
pipeline must reject before executing anything. The data format is documented
in tests/agl/README.md.

Public contract exercised here (AgL implementation requirements , the AgL DSL design ):

    from agm.agl import PipelineDriver

    runtime = PipelineDriver(
        default_strict_json=False,  # lenient JSON recovery is the default
        default_agent=fn,           # the built-in `ask` agent (a host callable;
                                    # `ask` cannot be registered by name)
    )
    runtime.register_agent(name, fn)   # fn(request) -> str; request.prompt is the
                                       # rendered user prompt
    result = runtime.run(source, param_values={...})

RunResult surface asserted:

    result.ok           True iff static checks and param validation passed and
                        no uncaught AgL exception was raised
    result.diagnostics  pre-execution failures (static errors, param
                        validation), each with `.message: str` and
                        `.line: int` (1-based source line)
    result.error        the uncaught AgL exception or None, exposing
                        `.type_name: str` and `.fields` — a mapping of the
                        exception's declared fields (including "message") to
                        JSON-shaped Python values

`print` writes to the process stdout (captured with capsys).
"""

from __future__ import annotations

import json
import unittest.mock
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tests._process_helpers import FakeShell

AGL_DIR = Path(__file__).parent / "agl"
PROGRAMS_DIR = AGL_DIR / "programs"
REJECTIONS_DIR = AGL_DIR / "rejections"
REPO_STDLIB_ROOT = Path(__file__).resolve().parents[1] / "stdlib"
EXTERNS_PROGRAMS_DIR = PROGRAMS_DIR / "externs"


def _load_json(path: Path) -> Any:
    # parse_float=Decimal: AgL has no binary floats; scenario
    # params and expected exception fields must round-trip decimals exactly.
    return json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)


@dataclass
class ScriptedAgent:
    """Replays a scenario's scripted responses and records rendered prompts.

    ``schemas`` records, alongside each call's ``prompt``, the structured JSON
    Schema from the output contract for that same call. It is a SEPARATE list
    from ``prompts`` so existing ``equals``/``contains`` assertions against the
    literal user prompt are unaffected; a scenario that wants to assert on the
    schema instead uses ``schema_contains`` or ``schema_paths``.
    """

    name: str
    responses: list[str]
    repeat_last: bool = False
    prompts: list[str] = field(default_factory=list)
    schemas: list[Any] = field(default_factory=list)
    overflowed: bool = False

    def __call__(self, request: Any) -> str:
        self.prompts.append(request.prompt)
        contract = request.output_contract
        self.schemas.append(contract.json_schema if contract is not None else None)
        index = len(self.prompts) - 1
        if index < len(self.responses):
            return self.responses[index]
        if self.repeat_last and self.responses:
            return self.responses[-1]
        self.overflowed = True
        return ""


def _agent_from_spec(name: str, spec: Any) -> ScriptedAgent:
    if isinstance(spec, list):
        return ScriptedAgent(name=name, responses=[str(r) for r in spec])
    return ScriptedAgent(
        name=name,
        responses=[str(r) for r in spec["responses"]],
        repeat_last=bool(spec.get("repeat_last", False)),
    )


def _run_program(
    source: str, scenario: dict[str, Any], program: Path
) -> tuple[Any, dict[str, ScriptedAgent], FakeShell]:
    from agm.agl import PipelineDriver

    agents = {
        name: _agent_from_spec(name, spec) for name, spec in scenario.get("agents", {}).items()
    }
    shell = FakeShell(scenario.get("shell", []))
    kwargs: dict[str, Any] = {}
    runtime_cfg = scenario.get("runtime", {})
    if "default_call_depth_limit" in runtime_cfg:
        kwargs["default_call_depth_limit"] = runtime_cfg["default_call_depth_limit"]
    if "default_strict_json" in runtime_cfg:
        kwargs["default_strict_json"] = runtime_cfg["default_strict_json"]
    if "ask" in agents:
        kwargs["default_agent"] = agents["ask"]
    runtime = PipelineDriver(**kwargs)

    def register_agents(declarations: tuple[Any, ...]) -> None:
        for name, agent in agents.items():
            if name == "ask":
                continue
            matches = [declaration for declaration in declarations if declaration.name == name]
            if len(matches) == 1:
                runtime.register_scoped_agent(matches[0].scope_path, name, agent)
            else:
                runtime.register_agent(name, agent)

    module_roots = scenario.get("module_roots", [])
    default_stdlib = not scenario.get("no_stdlib", False)
    with unittest.mock.patch("agm.core.process.run_capture_result", side_effect=shell):
        if module_roots:
            from agm.agl.modules.roots import RootSet

            roots = RootSet(
                roots=frozenset(
                    {
                        *((AGL_DIR / str(root)).resolve() for root in module_roots),
                        REPO_STDLIB_ROOT,
                    }
                )
            )
            prepared = PipelineDriver.prepare_program(
                source, entry_path=None, roots=roots, default_stdlib=default_stdlib
            )
            register_agents(prepared.declared_agents)
            result = runtime.run_prepared(prepared, param_values=scenario.get("params", {}))
        elif program.is_relative_to(EXTERNS_PROGRAMS_DIR):
            # Three branches, in order: `module_roots` above builds a graph from
            # explicit roots (multi-module fixtures); this branch also runs
            # through the graph, via `entry_path`, because `extern def` requires
            # a real file-backed origin so companion resolution can find the
            # sibling `.py`; every other fixture falls through to the plain,
            # non-graph `runtime.run` path in the `else` branch below.
            from agm.agl.modules.roots import RootSet

            roots = RootSet(roots=frozenset({program.parent.resolve(), REPO_STDLIB_ROOT}))
            prepared = PipelineDriver.prepare_program(
                source, entry_path=program, roots=roots, default_stdlib=default_stdlib
            )
            register_agents(prepared.declared_agents)
            result = runtime.run_prepared(prepared, param_values=scenario.get("params", {}))
        else:
            register_agents(runtime.declared_agents(source, default_stdlib=default_stdlib))
            result = runtime.run(
                source, param_values=scenario.get("params", {}), default_stdlib=default_stdlib
            )
    return result, agents, shell


def _assert_host_error(result: Any, agents: dict[str, ScriptedAgent], spec: dict[str, Any]) -> None:
    assert not result.ok, "expected the run to fail param validation"
    assert result.error is None, "param validation failure is not an AgL exception"
    messages = " | ".join(d.message for d in result.diagnostics)
    for needle in spec.get("message_contains", []):
        assert needle.lower() in messages.lower(), (
            f"no diagnostic mentions {needle!r}; diagnostics: {messages!r}"
        )
    for agent in agents.values():
        assert agent.prompts == [], (
            f"agent {agent.name!r} was called despite param validation failing"
        )


def _assert_outcome(result: Any, expect: dict[str, Any]) -> None:
    diags = " | ".join(d.message for d in result.diagnostics)
    assert list(result.diagnostics) == [], f"unexpected static diagnostics: {diags}"
    if "raises" in expect:
        spec = expect["raises"]
        assert result.error is not None, f"expected uncaught {spec['type']}, got none"
        assert result.error.type_name == spec["type"]
        for key, value in spec.get("fields", {}).items():
            actual = result.error.fields[key]
            assert actual == value, f"{spec['type']}.{key}: expected {value!r}, got {actual!r}"
        message = str(result.error.fields.get("message", ""))
        for needle in spec.get("message_contains", []):
            assert needle in message, f"{needle!r} not in message {message!r}"
        assert not result.ok
    else:
        if result.error is not None:
            raise AssertionError(
                f"unexpected uncaught {result.error.type_name}: {result.error.fields!r}"
            )
        assert result.ok


def _assert_output(out: str, expect: dict[str, Any]) -> None:
    if "stdout" in expect:
        assert out == expect["stdout"]
    for needle in expect.get("stdout_contains", []):
        assert needle in out, f"{needle!r} not in stdout {out!r}"
    for needle in expect.get("stdout_not_contains", []):
        assert needle not in out, f"{needle!r} unexpectedly in stdout {out!r}"


def _schema_contains(schema: Any, needle: str) -> bool:
    if isinstance(schema, dict):
        return any(
            str(key) == needle or _schema_contains(value, needle) for key, value in schema.items()
        )
    if isinstance(schema, list):
        return any(_schema_contains(item, needle) for item in schema)
    return schema == needle


def _assert_schema_paths(schema: Any, assertions: list[dict[str, Any]]) -> None:
    """Assert exact JSON Schema values at dictionary-key paths."""
    for assertion in assertions:
        actual = schema
        path = assertion["path"]
        for key in path:
            assert isinstance(actual, dict), f"{path!r} is not a dictionary path in {schema!r}"
            assert key in actual, f"{path!r} is absent from schema {schema!r}"
            actual = actual[key]
        assert actual == assertion["equals"], (
            f"schema at {path!r}: expected {assertion['equals']!r}, got {actual!r}"
        )


def test_schema_paths_assert_exact_nested_values() -> None:
    schema = {
        "$ref": "#/$defs/Workflow_Task",
        "$defs": {
            "Workflow_Task": {
                "properties": {"children": {"items": {"$ref": "#/$defs/Workflow_Task"}}}
            }
        },
    }

    _assert_schema_paths(
        schema,
        [
            {"path": ["$ref"], "equals": "#/$defs/Workflow_Task"},
            {
                "path": [
                    "$defs",
                    "Workflow_Task",
                    "properties",
                    "children",
                    "items",
                    "$ref",
                ],
                "equals": "#/$defs/Workflow_Task",
            },
        ],
    )


def _assert_calls(agents: dict[str, ScriptedAgent], expect: dict[str, Any]) -> None:
    for name, agent in agents.items():
        assert not agent.overflowed, f"agent {name!r} was called more times than scripted"
    expected_calls = expect.get("calls", {})
    assert set(expected_calls) == set(agents), (
        "every scripted agent must have an exact call-count assertion; "
        f"expected entries for {sorted(agents)}, got {sorted(expected_calls)}"
    )
    for name, agent in agents.items():
        count = expected_calls[name]
        actual = len(agent.prompts)
        assert actual == count, f"agent {name!r}: expected {count} calls, got {actual}"
    for spec in expect.get("prompts", []):
        prompts = agents[spec["agent"]].prompts
        call = spec["call"]
        assert call < len(prompts), (
            f"agent {spec['agent']!r} made only {len(prompts)} calls, no call {call}"
        )
        prompt = prompts[call]
        if "equals" in spec:
            assert prompt == spec["equals"]
        for needle in spec.get("contains", []):
            assert needle in prompt, f"{needle!r} not in prompt {prompt!r}"
        for needle in spec.get("not_contains", []):
            assert needle not in prompt, f"{needle!r} unexpectedly in prompt {prompt!r}"
        schema = agents[spec["agent"]].schemas[call]
        for needle in spec.get("schema_contains", []):
            assert _schema_contains(schema, needle), f"{needle!r} not in schema {schema!r}"
        _assert_schema_paths(schema, spec.get("schema_paths", []))


def _scenario_params() -> list[Any]:
    params: list[Any] = []
    for program in sorted(PROGRAMS_DIR.rglob("*.agl")):
        sidecar = program.with_name(program.stem + ".scenarios.json")
        rel = program.relative_to(PROGRAMS_DIR).with_suffix("")
        for scenario in _load_json(sidecar)["scenarios"]:
            params.append(pytest.param(program, scenario, id=f"{rel}::{scenario['name']}"))
    return params


def _rejection_params() -> list[Any]:
    return [
        pytest.param(program, id=str(program.relative_to(REJECTIONS_DIR).with_suffix("")))
        for program in sorted(REJECTIONS_DIR.rglob("*.agl"))
    ]


@pytest.mark.parametrize(("program", "scenario"), _scenario_params())
def test_program_scenario(
    program: Path, scenario: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    result, agents, shell = _run_program(program.read_text(encoding="utf-8"), scenario, program)
    out = capsys.readouterr().out
    expect = scenario["expect"]
    if "host_error" in expect:
        _assert_host_error(result, agents, expect["host_error"])
        shell.assert_complete()
        return
    _assert_outcome(result, expect)
    _assert_output(out, expect)
    _assert_calls(agents, expect)
    shell.assert_complete()


@pytest.mark.parametrize("program", _rejection_params())
def test_static_rejection(program: Path) -> None:
    from agm.agl import PipelineDriver

    expect = _load_json(program.with_name(program.stem + ".expect.json"))["diagnostic"]
    result = PipelineDriver().run(program.read_text(encoding="utf-8"), param_values={})
    assert not result.ok, "expected the program to be rejected statically"
    assert result.error is None, "static rejection must happen before execution"
    diagnostics = list(result.diagnostics)
    assert diagnostics, "expected at least one diagnostic"
    if "line" in expect:
        lines = [d.line for d in diagnostics]
        assert expect["line"] in lines, f"no diagnostic on line {expect['line']}; lines: {lines}"
    joined = " | ".join(d.message for d in diagnostics)
    for needle in expect.get("message_contains", []):
        assert needle.lower() in joined.lower(), (
            f"no diagnostic mentions {needle!r}; diagnostics: {joined!r}"
        )


def _scoped_stdlib_root(tmp_path: Path) -> Path:
    """Build a throwaway module root with ``std/core.agl`` wrapped in ``scope Std``.

    Never the installed/repo stdlib root — the standard ``module_roots``
    scenario mechanism always adds ``REPO_STDLIB_ROOT`` too, which would
    collide with this substitute ``std/core``, so callers build their own
    ``RootSet`` from the returned root instead.
    """
    core_source = (REPO_STDLIB_ROOT / "std" / "core.agl").read_text(encoding="utf-8")
    scoped_stdlib_root = tmp_path / "scoped_stdlib"
    (scoped_stdlib_root / "std").mkdir(parents=True)
    (scoped_stdlib_root / "std" / "core.agl").write_text(
        f"scope Std\n{core_source}end Std\n", encoding="utf-8"
    )
    return scoped_stdlib_root


def test_scoped_stdlib_arrangement_runs_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A whole stdlib module wrapped in a named scope region works end to end.

    Wraps the real ``stdlib/std/core.agl`` verbatim in a ``scope Std ... end
    Std`` region under a throwaway module root (never the installed/repo
    stdlib — the standard ``module_roots`` scenario mechanism always adds
    ``REPO_STDLIB_ROOT`` too, which would collide with this substitute
    ``std/core``, so this test builds its own ``RootSet`` instead), then runs
    a program against it through parsing, scope resolution, typechecking,
    lowering, and evaluation. Exercises a scoped ``builtin def`` (``exec``)
    dispatching to a scoped ``builtin record`` (``ExecResult``) at a real host
    boundary, and a fully scoped exception hierarchy (``Exception``/
    ``RangeError`` both declared inside the region) raised and left uncaught.
    """
    from agm.agl import PipelineDriver
    from agm.agl.modules.roots import RootSet

    scoped_stdlib_root = _scoped_stdlib_root(tmp_path)

    program = (
        'let r = Std::exec("echo hi")\nprint(r.stdout)\nraise Std::RangeError(message = "boom")\n'
    )

    shell = FakeShell([{"command": "echo hi", "stdout": "hi\n"}])
    runtime = PipelineDriver()
    with unittest.mock.patch("agm.core.process.run_capture_result", side_effect=shell):
        result = runtime.run(program, roots=RootSet(roots=frozenset({scoped_stdlib_root})))
    shell.assert_complete()

    assert list(result.diagnostics) == [], (
        f"unexpected static diagnostics: {' | '.join(d.message for d in result.diagnostics)}"
    )
    assert capsys.readouterr().out == "hi\n"
    assert result.error is not None, "expected the uncaught scoped RangeError"
    assert result.error.type_name == "Std::RangeError"
    assert result.error.fields["message"] == "boom"


def test_scoped_stdlib_arrangement_structured_exec_result_is_the_scoped_nominal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A structured ``exec`` result is tagged with its own scoped nominal.

    The result is annotated ``Std::ExecResult`` explicitly, so the static
    target type is the scoped nominal by the annotation rather than by
    ``exec``'s own default. The field access below then only succeeds if the
    host-minted value carries that very same scoped nominal — an exact
    nominal projection rejects any other identity.
    """
    from agm.agl import PipelineDriver
    from agm.agl.modules.roots import RootSet

    scoped_stdlib_root = _scoped_stdlib_root(tmp_path)

    program = 'let r: Std::ExecResult = Std::exec("echo hi")\nprint(r.stdout)\n'

    shell = FakeShell([{"command": "echo hi", "stdout": "hi\n"}])
    runtime = PipelineDriver()
    with unittest.mock.patch("agm.core.process.run_capture_result", side_effect=shell):
        result = runtime.run(program, roots=RootSet(roots=frozenset({scoped_stdlib_root})))
    shell.assert_complete()

    assert list(result.diagnostics) == [], (
        f"unexpected static diagnostics: {' | '.join(d.message for d in result.diagnostics)}"
    )
    assert result.error is None, f"unexpected error: {result.error}"
    assert capsys.readouterr().out == "hi\n"


def test_scoped_stdlib_arrangement_uncaught_host_raised_exec_error_reports_scoped_spelling(
    tmp_path: Path,
) -> None:
    """An uncaught host-raised exception reports its declared, scoped spelling.

    A failed (non-structured, ``text``-typed) ``Std::exec`` call raises
    ``ExecError`` at the host boundary; left uncaught, it now reports
    ``Std::ExecError`` — the outside-the-region half of the catchability fix
    observable without a qualified ``catch`` (the grammar admits none)."""
    from agm.agl import PipelineDriver
    from agm.agl.modules.roots import RootSet

    scoped_stdlib_root = _scoped_stdlib_root(tmp_path)

    program = 'let out: text = Std::exec("false")\nprint(out)\n'

    shell = FakeShell([{"command": "false", "returncode": 1}])
    runtime = PipelineDriver()
    with unittest.mock.patch("agm.core.process.run_capture_result", side_effect=shell):
        result = runtime.run(program, roots=RootSet(roots=frozenset({scoped_stdlib_root})))
    shell.assert_complete()

    assert list(result.diagnostics) == [], (
        f"unexpected static diagnostics: {' | '.join(d.message for d in result.diagnostics)}"
    )
    assert result.error is not None, "expected the uncaught scoped ExecError"
    assert result.error.type_name == "Std::ExecError"


def test_scoped_builtin_hierarchy_declared_in_the_entry_module_catches_a_host_raise(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A host-raised exception is catchable by its declared, scoped spelling.

    ``catch`` accepts only an unqualified NAME (the grammar admits no
    qualified ``A::B`` there), so the catching code lives inside the same
    ``scope Host`` region that declares the hierarchy, where the bare name
    resolves to the region's own declaration — the same arrangement
    ``scoped_builtin_stdlib_arrangement.agl``'s own ``report``/``RangeError``
    catch already uses. This program declares the whole exec/exception
    surface itself and loads without the standard library (``default_stdlib
    =False``), so its own ``scope Host`` module lowers normally (unlike the
    real ``std/core`` module, which is excluded from per-module function
    lowering). A failed (``text``-typed, non-structured) ``exec`` call
    raises ``ExecError`` at the host boundary; before per-path identity was
    restored the host would have minted a path-free nominal while the type
    kept its declared path, so this bare ``catch`` could never match. Now
    both sides agree, and the program handles it.
    """
    from agm.agl import PipelineDriver

    program = (
        "scope Host\n"
        "builtin\n"
        "exception Exception\n"
        "  *\n"
        "  message: text\n"
        "  trace_id: text\n"
        "builtin\n"
        "exception ExecError extends Exception\n"
        "  *\n"
        "  command: text\n"
        "  exit_code: int\n"
        "  stdout: text\n"
        "  stderr: text\n"
        "  timed_out: bool\n"
        "builtin record ExecResult\n"
        "  stdout: text\n"
        "  exit_code: int\n"
        "  stderr: text\n"
        "  timed_out: bool\n"
        "builtin def exec(command: text) -> ExecResult\n"
        "def run(cmd: text) -> text =\n"
        "  try\n"
        "    let out: text = exec(cmd)\n"
        "    out\n"
        "  catch ExecError as e =>\n"
        '    "caught"\n'
        "end Host\n"
        'print(Host::run("false"))\n'
    )

    shell = FakeShell([{"command": "false", "returncode": 1}])
    runtime = PipelineDriver()
    with unittest.mock.patch("agm.core.process.run_capture_result", side_effect=shell):
        result = runtime.run(program, default_stdlib=False)
    shell.assert_complete()

    assert list(result.diagnostics) == [], (
        f"unexpected static diagnostics: {' | '.join(d.message for d in result.diagnostics)}"
    )
    assert result.error is None, f"expected the raised ExecError to be caught: {result.error}"
    assert capsys.readouterr().out == "caught\n"
