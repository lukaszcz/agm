"""IR evaluation tests for agent dispatch (ask / ask-request).

Each test evaluates an AgL program through the IR pipeline with scripted agent responses
and asserts the produced values and stdout.
"""

from __future__ import annotations

import json as _json
from typing import TYPE_CHECKING

import pytest

from agm.agl.semantics.values import (
    ArrayValue,
    BoolValue,
    ExceptionValue,
    IntValue,
    JsonValue,
    RecordValue,
    TextValue,
)
from tests._agl_helpers import run_inline_command
from tests.agl.ir_harness import (
    agent_caps,
    evaluate_ir,
    evaluate_ir_raises_with_agents,
    evaluate_ir_with_agents,
    lower_inline_ir,
)

if TYPE_CHECKING:
    from agm.agl.syntax.nodes import Call
    from agm.agl.syntax.spans import SourceSpan


# ---------------------------------------------------------------------------
# builtin Agent methods
# ---------------------------------------------------------------------------


def test_agent_ask_method_accepts_type_args_and_named_defaults() -> None:
    """Agent::ask is a normal selected builtin method with the receiver as agent."""
    source = """\
let worker: Agent = AgentCommand("worker")
let answer: int = worker.ask::[int]("How many?", strict-json = true)
answer
"""
    ir = evaluate_ir_with_agents(source, scripts={"worker": ["42"]})

    assert ir["answer"] == IntValue(42)


def test_agent_ask_request_method_uses_its_receiver() -> None:
    """Agent::ask-request constructs a request without dispatching."""
    source = """\
let worker: Agent = AgentCommand("worker")
let request = worker.ask-request("Draft it.")
request
"""
    ir = evaluate_ir_with_agents(source, scripts={"worker": []})

    request = ir["request"]
    assert isinstance(request, RecordValue)
    assert request.fields["agent"] == RecordValue(
        nominal=request.fields["agent"].nominal,
        display_name=f"{'Agent'}::{'AgentCommand'}",
        fields={"command": TextValue("worker")},
    )


def test_agent_ask_method_can_be_called_through_a_bound_value() -> None:
    source = """\
let worker: Agent = AgentCommand("worker")
let query: text -> text = worker.ask
let answer = query("Question")
answer
"""

    ir = evaluate_ir_with_agents(source, scripts={"worker": ["answer"]})

    assert ir["answer"] == TextValue("answer")


def test_agent_ask_method_can_be_called_through_an_unbound_value() -> None:
    source = """\
let query: (Agent, text) -> text = Agent::ask
let worker: Agent = AgentCommand("worker")
let answer = query(worker, "Question")
answer
"""

    ir = evaluate_ir_with_agents(source, scripts={"worker": ["answer"]})

    assert ir["answer"] == TextValue("answer")


def test_agent_ask_method_import_alias_preserves_unbound_value_semantics() -> None:
    source = """\
import std/agent::{Agent::ask as query}
let worker: Agent = AgentCommand("worker")
let answer: text = query(worker, "Question")
answer
"""

    ir = evaluate_ir_with_agents(source, scripts={"worker": ["answer"]})

    assert ir["answer"] == TextValue("answer")


def test_agent_ask_method_can_be_called_directly_as_an_unbound_value() -> None:
    source = """\
let worker: Agent = AgentCommand("worker")
let answer: text = Agent::ask(worker, "Question")
answer
"""

    ir = evaluate_ir_with_agents(source, scripts={"worker": ["answer"]})

    assert ir["answer"] == TextValue("answer")


def test_agent_ask_method_direct_unbound_call_retains_named_options() -> None:
    source = """\
let worker: Agent = AgentCommand("worker")
let answer = Agent::ask::[text](worker, "Question", format = "text")
answer
"""

    ir = evaluate_ir_with_agents(source, scripts={"worker": ["answer"]})

    assert ir["answer"] == TextValue("answer")


def test_agent_ask_method_direct_unbound_call_accepts_explicit_specialization() -> None:
    source = """\
let worker: Agent = AgentCommand("worker")
let answer = Agent::ask::[int](worker, "Question")
answer
"""

    ir = evaluate_ir_with_agents(source, scripts={"worker": ["42"]})

    assert ir["answer"] == IntValue(42)


def test_agent_ask_method_partial_application_uses_its_bound_value() -> None:
    source = """\
let worker: Agent = AgentCommand("worker")
let query: text -> int = worker.ask(?)
let answer = query("Question")
answer
"""

    ir = evaluate_ir_with_agents(source, scripts={"worker": ["42"]})

    assert ir["answer"] == IntValue(42)


def test_agent_ask_request_method_value_defaults_to_text() -> None:
    source = """\
let worker: Agent = AgentCommand("worker")
let make-request = worker.ask-request
let request = make-request("Describe this")
request
"""

    ir = evaluate_ir(source)

    request = ir["request"]
    assert isinstance(request, RecordValue)
    target = request.fields["target-type"]
    assert isinstance(target, RecordValue)
    assert target.display_name.rsplit("::", maxsplit=1)[-1] == "Some"
    assert target.fields["value"] == TextValue("text")


def test_agent_ask_request_method_value_preserves_explicit_specialization() -> None:
    source = """\
let worker: Agent = AgentCommand("worker")
let make-request = worker.ask-request::[int]
let request = make-request("Count this")
request
"""

    ir = evaluate_ir(source)

    request = ir["request"]
    assert isinstance(request, RecordValue)
    target = request.fields["target-type"]
    assert isinstance(target, RecordValue)
    assert target.display_name.rsplit("::", maxsplit=1)[-1] == "Some"
    assert target.fields["value"] == TextValue("int")


def test_user_declared_method_named_ask_dispatches_as_an_ordinary_method() -> None:
    """A field access naming ``ask`` is only a *speculative* builtin route.

    The receiver here is a user record, not ``Agent``, so the selected method
    is the ordinary one declared below; it must run as a plain method call
    rather than dispatching through agent machinery (which would need a
    scripted agent response to avoid erroring).
    """
    source = """\
record Greeter
  name: text

def Greeter::ask(self, prompt: text) -> text = "%{self.name}: %{prompt}"

let g = Greeter(name = "g")
let reply = g.ask("hi")
reply
"""
    ir = evaluate_ir(source)

    assert ir["reply"] == TextValue("g: hi")


def test_user_declared_method_named_ask_works_as_a_value_and_partially_applied() -> None:
    """An ordinary ``ask``/``ask-request`` method keeps its first-class forms.

    Both allocate a bound closure over the receiver, the shape a builtin method
    never has, so the speculative builtin route must not capture either one.
    """
    source = """\
record Greeter
  name: text

def Greeter::ask[T](self, prompt: T) -> T = prompt
def Greeter::ask-request(self, prompt: text) -> text = "%{self.name}: %{prompt}"

let g = Greeter(name = "g")
let bound = g.ask-request
let partial = g.ask::[text](?)
let greeting = bound("hi")
let echoed = partial("there")
()
"""
    ir = evaluate_ir(source)

    assert ir["greeting"] == TextValue("g: hi")
    assert ir["echoed"] == TextValue("there")


def test_first_class_ask_request_preserves_explicit_output_specialization() -> None:
    source = """\
let make-request = ask-request::[int]
let request = make-request("Count this")
request
"""

    ir = evaluate_ir(source)

    request = ir["request"]
    assert isinstance(request, RecordValue)
    target = request.fields["target-type"]
    assert isinstance(target, RecordValue)
    assert target.display_name.rsplit("::", maxsplit=1)[-1] == "Some"
    assert target.fields["value"] == TextValue("int")


# ---------------------------------------------------------------------------
# simple text ask
# ---------------------------------------------------------------------------


def test_text_ask_basic() -> None:
    """Text-codec ask: passthrough, no JSON involved."""
    source = """\
let summarizer = AgentCommand("summarizer")
let summary: text = ask("Summarise it.", agent = summarizer)
summary
"""
    ir = evaluate_ir_with_agents(
        source,
        scripts={"summarizer": ["This is a summary."]},
    )
    assert ir["summary"] == TextValue("This is a summary.")


# ---------------------------------------------------------------------------
# JSON int ask
# ---------------------------------------------------------------------------


def test_json_int_ask() -> None:
    """JSON-int ask: agent returns a bare integer."""
    source = """\
let counter = AgentCommand("counter")
let n: int = ask("How many?", agent = counter)
n
"""
    ir = evaluate_ir_with_agents(
        source,
        scripts={"counter": ["42"]},
    )
    assert ir["n"] == IntValue(42)


# ---------------------------------------------------------------------------
# JSON record ask
# ---------------------------------------------------------------------------


def test_json_record_ask() -> None:
    """JSON-record ask: agent returns a JSON object matching a record type."""
    source = """\
record Point
  x: int
  y: int

let locator = AgentCommand("locator")
let pt: Point = ask("Find the point.", agent = locator)
pt
"""
    ir = evaluate_ir_with_agents(
        source,
        scripts={"locator": ['{"x": 3, "y": 7}']},
    )
    assert isinstance(ir["pt"], RecordValue)
    assert ir["pt"].fields["x"] == IntValue(3)
    assert ir["pt"].fields["y"] == IntValue(7)


# ---------------------------------------------------------------------------
# JSON array ask
# ---------------------------------------------------------------------------


def test_json_array_ask() -> None:
    """JSON-array ask: agent returns a JSON array."""
    source = """\
let lister = AgentCommand("lister")
let items: array[text] = ask("List items.", agent = lister)
items
"""
    ir = evaluate_ir_with_agents(
        source,
        scripts={"lister": ['["alpha", "beta", "gamma"]']},
    )
    assert isinstance(ir["items"], ArrayValue)
    assert ir["items"].elements == [TextValue("alpha"), TextValue("beta"), TextValue("gamma")]
    assert isinstance(ir["items"], ArrayValue)
    assert ir["items"].elements == [TextValue("alpha"), TextValue("beta"), TextValue("gamma")]


# ---------------------------------------------------------------------------
# JSON enum ask
# ---------------------------------------------------------------------------


def test_json_enum_ask() -> None:
    """JSON-enum ask: agent returns a discriminated enum value."""
    source = """\
enum Status
  | Ok
  | Err(msg: text)

let checker = AgentCommand("checker")
let status: Status = ask("Check it.", agent = checker)
status
"""
    ir = evaluate_ir_with_agents(
        source,
        scripts={"checker": ['{"$case": "Ok"}']},
    )
    assert isinstance(ir["status"], RecordValue)
    assert ir["status"].display_name == "Status::Ok"


# ---------------------------------------------------------------------------
# lenient JSON recovery (fence stripping)
# ---------------------------------------------------------------------------


def test_lenient_json_fence_stripping() -> None:
    """Lenient mode: agent wraps JSON in a markdown fence — still parsed."""
    source = """\
let answerer = AgentCommand("answerer")
let n: int = ask("Give me a number.", agent = answerer)
n
"""
    fenced = "```json\n17\n```"
    ir = evaluate_ir_with_agents(
        source,
        scripts={"answerer": [fenced]},
    )
    assert ir["n"] == IntValue(17)


# ---------------------------------------------------------------------------
# retry success (on-parse-error: Retry(n: 1))
# ---------------------------------------------------------------------------


def test_retry_success_second_attempt() -> None:
    """Retry policy: first response is invalid JSON, second is valid."""
    source = """\
let parser = AgentCommand("parser")
let n: int = ask("Parse this.", agent = parser, on-parse-error = Retry(n = 1))
n
"""
    ir = evaluate_ir_with_agents(
        source,
        scripts={"parser": ["not json at all", "99"]},
    )
    assert ir["n"] == IntValue(99)


# ---------------------------------------------------------------------------
# retry exhausted raises AgentParseError
# ---------------------------------------------------------------------------


def test_retry_exhausted_raises() -> None:
    """Retry policy: all attempts fail → AgentParseError raised."""
    source = """\
let parser = AgentCommand("parser")
let n: int = ask("Parse this.", agent = parser, on-parse-error = Retry(n = 1))
n
"""
    ir_exc = evaluate_ir_raises_with_agents(
        source,
        scripts={"parser": ["bad1", "bad2"]},
    )
    assert isinstance(ir_exc, ExceptionValue)
    assert ir_exc.display_name == "AgentParseError"


# ---------------------------------------------------------------------------
# strict JSON mode
# ---------------------------------------------------------------------------


def test_strict_json_mode() -> None:
    """strict_json: true — bare JSON without fences, no repair."""
    source = """\
let strict-agent = AgentCommand("strict_agent")
let b: bool = ask("True or false?", agent = strict-agent, strict-json = true)
b
"""
    ir = evaluate_ir_with_agents(
        source,
        scripts={"strict_agent": ["true"]},
    )
    assert ir["b"] == BoolValue(True)


# ---------------------------------------------------------------------------
# unit-typed ask (no output parsing)
# ---------------------------------------------------------------------------


def test_unit_typed_ask() -> None:
    """A bare ask runs once and discards its successful output."""
    from agm.agl import PipelineDriver

    calls: list[object] = []

    def notify(request: object) -> str:
        calls.append(request)
        return "acknowledged"

    runtime = PipelineDriver(agent_dispatcher=notify)
    result = run_inline_command(runtime, 'ask("Notify!")\n()')
    assert result.ok
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# ask inside a function
# ---------------------------------------------------------------------------


def test_ask_inside_function() -> None:
    """A root function uses an agent supplied by an immutable root binding."""
    source = """\
let namer: Agent = AgentCommand("namer")
def get-name(prompt: text) -> text = ask(prompt, agent = namer)
let name: text = get-name("What is the name?")
name
"""
    ir = evaluate_ir_with_agents(
        source,
        scripts={"namer": ["Alice"]},
    )
    assert ir["name"] == TextValue("Alice")


# ---------------------------------------------------------------------------
# multiple agents
# ---------------------------------------------------------------------------


def test_multiple_agents() -> None:
    """Multiple named agents: each call is routed to the correct agent."""
    source = """\
let first = AgentCommand("first")
let second = AgentCommand("second")
let a: text = ask("First.", agent = first)
let b: text = ask("Second.", agent = second)
b
"""
    ir = evaluate_ir_with_agents(
        source,
        scripts={
            "first": ["hello"],
            "second": ["world"],
        },
    )
    assert ir["a"] == TextValue("hello")
    assert ir["b"] == TextValue("world")


# ---------------------------------------------------------------------------
# JSON schema validation error (wrong type)
# ---------------------------------------------------------------------------


def test_schema_validation_failure_wrong_type() -> None:
    """Agent returns invalid JSON (fails schema validation) → AgentParseError."""
    source = """\
let validator = AgentCommand("validator")
let n: int = ask("Give int.", agent = validator)
n
"""
    # Agent returns a string, not an integer — schema validation fails.
    ir_exc = evaluate_ir_raises_with_agents(
        source,
        scripts={"validator": ['"not an int"']},
    )
    assert isinstance(ir_exc, ExceptionValue)
    assert ir_exc.display_name == "AgentParseError"


# ---------------------------------------------------------------------------
# strict JSON: invalid JSON raises AgentParseError
# ---------------------------------------------------------------------------


def test_strict_json_invalid_raises() -> None:
    """strict_json=true with fenced JSON: strict mode does not strip fences."""
    source = """\
let strict-agent = AgentCommand("strict_agent")
let n: int = ask("Give int.", agent = strict-agent, strict-json = true)
n
"""
    # Fenced JSON fails in strict mode (strict does not strip fences).
    ir_exc = evaluate_ir_raises_with_agents(
        source,
        scripts={"strict_agent": ["```json\n42\n```"]},
    )
    assert isinstance(ir_exc, ExceptionValue)
    assert ir_exc.display_name == "AgentParseError"


# ---------------------------------------------------------------------------
# default agent (ask without agent: named arg)
# ---------------------------------------------------------------------------


def test_default_agent_ask() -> None:
    """ask() with no agent: named arg uses the default agent."""
    source = """\
let result: text = ask("Hello default.")
result
"""
    ir = evaluate_ir_with_agents(
        source,
        scripts={},
        default_responses=["default response"],
    )
    assert ir["result"] == TextValue("default response")


# ---------------------------------------------------------------------------
# ask-request builds an AgentRequest record
# ---------------------------------------------------------------------------


def test_ask_request_builds_record() -> None:
    """ask-request: no agent dispatch, returns an AgentRequest-shaped record."""
    source = """\
let dummy = AgentCommand("dummy")
let req = ask-request("My prompt.", agent = dummy)
let prompt-text: text = req.prompt
prompt-text
"""
    # ask-request does not call the agent — no scripted responses needed.
    ir = evaluate_ir_with_agents(
        source,
        scripts={"dummy": []},
    )
    assert ir["prompt-text"] == TextValue("My prompt.")

    req = ir["req"]
    assert isinstance(req, RecordValue)
    program = lower_inline_ir(source, caps=agent_caps())
    assert req.nominal == program.builtin_nominals.nominal("AgentRequest")
    assert isinstance(req.fields["agent"], RecordValue)
    assert req.fields["agent"].display_name.rsplit("::", maxsplit=1)[-1] == "AgentCommand"
    assert isinstance(req.fields["target-type"], RecordValue)
    assert req.fields["target-type"].display_name.rsplit("::", maxsplit=1)[-1] == "Some"
    assert req.fields["target-type"].fields["value"] == TextValue("text")
    assert isinstance(req.fields["format-instructions"], RecordValue)
    assert req.fields["format-instructions"].display_name.rsplit("::", maxsplit=1)[-1] == "None"
    assert isinstance(req.fields["json-schema"], RecordValue)
    assert req.fields["json-schema"].display_name.rsplit("::", maxsplit=1)[-1] == "None"


def test_ask_request_carries_the_requested_output_contract() -> None:
    """A typed ask-request records the contract its ask would have dispatched."""
    source = """\
record Answer
  value: int

let worker = AgentCommand("worker")
let req = ask-request::[Answer]("How many?", agent = worker, strict-json = true)
req
"""
    ir = evaluate_ir_with_agents(source, scripts={"worker": []})

    req = ir["req"]
    assert isinstance(req, RecordValue)
    target = req.fields["target-type"]
    assert isinstance(target, RecordValue)
    assert target.display_name == "Option::Some"
    assert target.fields["value"] == TextValue("Answer")
    schema = req.fields["json-schema"]
    assert isinstance(schema, RecordValue)
    assert schema.display_name == "Option::Some"
    assert isinstance(schema.fields["value"], JsonValue)
    assert "value" in _json.dumps(schema.fields["value"].raw)
    instructions = req.fields["format-instructions"]
    assert isinstance(instructions, RecordValue)
    assert instructions.display_name == "Option::Some"
    metadata = req.fields["metadata"]
    assert isinstance(metadata, JsonValue)
    assert isinstance(metadata.raw, dict)
    assert metadata.raw["codec_name"] == "json"
    assert metadata.raw["strict_json"] is True


def test_ask_request_records_its_retry_policy() -> None:
    """``on-parse-error`` shapes the attempt budget recorded on the request."""
    source = """\
let worker = AgentCommand("worker")
let req = ask-request::[int]("How many?", agent = worker, on-parse-error = Retry(n = 2))
req
"""
    ir = evaluate_ir_with_agents(source, scripts={"worker": []})

    req = ir["req"]
    assert isinstance(req, RecordValue)
    metadata = req.fields["metadata"]
    assert isinstance(metadata, JsonValue)
    assert isinstance(metadata.raw, dict)
    assert metadata.raw["max_attempts"] == 3


def test_ask_request_juxtaposed_with_type_args() -> None:
    """``ask-request::[T] prompt`` is the juxtaposed form of the typed call."""
    source = """\
let req = ask-request::[int] "How many?"
req
"""
    ir = evaluate_ir_with_agents(source, scripts={}, default_responses=[])

    req = ir["req"]
    assert isinstance(req, RecordValue)
    target = req.fields["target-type"]
    assert isinstance(target, RecordValue)
    assert target.fields["value"] == TextValue("int")


def test_ask_juxtaposed_with_type_args_parses_its_output() -> None:
    """``ask::[T] prompt`` dispatches and parses like ``ask::[T](prompt)``."""
    source = """\
let worker = AgentCommand("worker")
let answer: int = worker.ask::[int] "How many?"
answer
"""
    ir = evaluate_ir_with_agents(source, scripts={"worker": ["42"]})

    assert ir["answer"] == IntValue(42)


# ---------------------------------------------------------------------------
# retry with schema validation error (covers result.errors branch)
# ---------------------------------------------------------------------------


def test_retry_with_schema_validation_error_then_success() -> None:
    """Retry: first response fails schema, second is valid."""
    source = """\
let fixer = AgentCommand("fixer")
let n: int = ask("Give int.", agent = fixer, on-parse-error = Retry(n = 1))
n
"""
    # First response: string (wrong type) → schema error; second: valid int.
    ir = evaluate_ir_with_agents(
        source,
        scripts={"fixer": ['"oops"', "42"]},
    )
    assert ir["n"] == IntValue(42)


# ---------------------------------------------------------------------------
# enum bad_case validation failure (covers _classify_enum_failure_typeless)
# ---------------------------------------------------------------------------


def test_enum_bad_case_raises_agent_parse_error() -> None:
    """Enum ask: agent returns unknown $case → AgentParseError."""
    import json

    from agm.agl.ir.contracts import (
        ContractRequest,
        EnumDecode,
        ScalarDecode,
        ScalarKind,
        VariantDecode,
    )
    from agm.agl.ir.ids import NominalId
    from agm.agl.runtime.codec import _parse_contract_output

    nominal = NominalId(1)
    decode = EnumDecode(
        nominal=nominal,
        display_name="Status",
        variants=(
            VariantDecode(name="Ok", nominal=NominalId(999), display_name="Ok", fields=()),
            VariantDecode(
                name="Err",
                nominal=NominalId(999),
                display_name="Err",
                fields=(("msg", ScalarDecode(ScalarKind.TEXT)),),
            ),
        ),
    )
    schema = {
        "oneOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["$case"],
                "properties": {"$case": {"const": "Ok"}},
            },
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["$case", "msg"],
                "properties": {"$case": {"const": "Err"}, "msg": {"type": "string"}},
            },
        ]
    }
    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema=json.dumps(schema, sort_keys=True),
        decode=decode,
        target_type_label="Status",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    # Unknown $case.
    result = _parse_contract_output('{"$case": "Unknown"}', contract, effective_strict=False)
    assert not result.ok
    assert len(result.errors) == 1
    assert result.errors[0].category == "bad_case"

    # Missing $case (not a dict at all first, then missing).
    result2 = _parse_contract_output('{"value": "hello"}', contract, effective_strict=False)
    assert not result2.ok
    assert len(result2.errors) == 1
    assert result2.errors[0].category == "bad_case"

    # Missing 'msg' field for Err variant.
    result3 = _parse_contract_output('{"$case": "Err"}', contract, effective_strict=False)
    assert not result3.ok
    assert len(result3.errors) == 1
    assert result3.errors[0].category == "missing_field"


# ---------------------------------------------------------------------------
# enum with retry success (covers _find_enum_decode_at_path)
# ---------------------------------------------------------------------------


def test_enum_retry_then_success() -> None:
    """Enum ask: first response has bad case, second is valid."""
    source = """\
enum Status
  | Ok
  | Err(msg: text)

let checker = AgentCommand("checker")
let s: Status = ask("Status?", agent = checker, on-parse-error = Retry(n = 1))
s
"""
    ir = evaluate_ir_with_agents(
        source,
        scripts={"checker": ['{"$case": "Bad"}', '{"$case": "Ok"}']},
    )
    assert isinstance(ir["s"], RecordValue)
    assert ir["s"].display_name == "Status::Ok"


# ---------------------------------------------------------------------------
# validate.py defensive checks (hand-built invalid IR)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("request_only", (False, True))
def test_validate_ir_ask_missing_contract(request_only: bool) -> None:
    """validate_ir: an ask node referencing a missing contract_id → InvalidIrError."""

    from agm.agl.ir.ids import ContractId, Location, SourceId
    from agm.agl.ir.nodes import IrAsk, IrAskRequest, IrConstText
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import InvalidIrError, validate_ir
    from agm.agl.modules.ids import ENTRY_ID

    src_id = SourceId(0)
    dummy_loc = Location(source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0)
    node_type: type[IrAsk] | type[IrAskRequest] = IrAskRequest if request_only else IrAsk
    ask_node = node_type(
        location=dummy_loc,
        agent=IrConstText(location=dummy_loc, value="ask"),
        prompt=IrConstText(location=dummy_loc, value="test"),
        contract_id=ContractId(999),
        max_attempts=1,
    )
    prog = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=(ask_node,))},
        symbols={},
        nominals={},
        sources={src_id: SourceFile(display_name="<test>", normalized_text="test")},
        contracts={},  # No contracts!
    )
    with pytest.raises(InvalidIrError, match="contract_id"):
        validate_ir(prog, deep=True)


@pytest.mark.parametrize("request_only", (False, True))
def test_validate_ir_ask_max_attempts_zero(request_only: bool) -> None:
    """validate_ir: an ask node with max_attempts=0 → InvalidIrError."""

    from agm.agl.ir.contracts import ContractRequest
    from agm.agl.ir.ids import ContractId, Location, SourceId
    from agm.agl.ir.nodes import IrAsk, IrAskRequest, IrConstText
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import InvalidIrError, validate_ir
    from agm.agl.modules.ids import ENTRY_ID

    src_id = SourceId(0)
    dummy_loc = Location(source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0)
    cid = ContractId(0)
    req = ContractRequest(
        codec_name="text",
        strict_json=None,
        json_schema=None,
        decode=None,
        target_type_label="text",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    node_type: type[IrAsk] | type[IrAskRequest] = IrAskRequest if request_only else IrAsk
    ask_node = node_type(
        location=dummy_loc,
        agent=IrConstText(location=dummy_loc, value="ask"),
        prompt=IrConstText(location=dummy_loc, value="test"),
        contract_id=cid,
        max_attempts=0,  # invalid!
    )
    prog = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=(ask_node,))},
        symbols={},
        nominals={},
        sources={src_id: SourceFile(display_name="<test>", normalized_text="test")},
        contracts={cid: req},
    )
    with pytest.raises(InvalidIrError, match="max_attempts"):
        validate_ir(prog, deep=True)


def test_validate_contract_request_json_missing_schema() -> None:
    """_validate_contract_request: json codec but no schema → InvalidIrError."""

    from agm.agl.ir.contracts import ContractRequest
    from agm.agl.ir.ids import ContractId, Location, SourceId
    from agm.agl.ir.nodes import IrConstUnit
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import InvalidIrError, validate_ir
    from agm.agl.modules.ids import ENTRY_ID

    src_id = SourceId(0)
    dummy_loc = Location(source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0)
    cid = ContractId(0)
    bad_req = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema=None,  # missing!
        decode=None,
        target_type_label="int",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    prog = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={
            ENTRY_ID: ExecutableModule(
                module_id=ENTRY_ID,
                initializers=(IrConstUnit(location=dummy_loc),),
            )
        },
        symbols={},
        nominals={},
        sources={src_id: SourceFile(display_name="<test>", normalized_text="test")},
        contracts={cid: bad_req},
    )
    with pytest.raises(InvalidIrError, match="json_schema"):
        validate_ir(prog, deep=True)


# ---------------------------------------------------------------------------
# ask-request via lowerer BuiltinKind.ASK_REQUEST path
# ---------------------------------------------------------------------------


def test_ask_request_builds_a_text_request_record() -> None:
    """ask-request builds its fixed text-contract AgentRequest record."""
    source = """\
let worker = AgentCommand("worker")
let req = ask-request("Give me a number.", agent = worker)
let prompt-text: text = req.prompt
prompt-text
"""
    ir = evaluate_ir_with_agents(
        source,
        scripts={"worker": []},
    )
    assert ir["prompt-text"] == TextValue("Give me a number.")


# ---------------------------------------------------------------------------
# _parse_contract_output unit tests for uncovered branches
# ---------------------------------------------------------------------------


def test_parse_contract_output_json_schema_none() -> None:
    """_parse_contract_output: json codec but json_schema is None → failure."""
    from agm.agl.ir.contracts import ContractRequest
    from agm.agl.runtime.codec import _parse_contract_output

    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema=None,
        decode=None,
        target_type_label="int",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    result = _parse_contract_output("42", contract, effective_strict=False)
    assert not result.ok
    assert "json_schema" in result.error_msg


def test_parse_contract_output_ambiguous_multi_value() -> None:
    """_parse_contract_output: lenient mode with ambiguous multi-value → failure."""
    from agm.agl.ir.contracts import ContractRequest, ScalarDecode, ScalarKind
    from agm.agl.runtime.codec import _parse_contract_output

    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema=_json.dumps({"type": "integer"}),
        decode=ScalarDecode(ScalarKind.INT),
        target_type_label="int",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    # Two bare JSON values → _extract_json_text returns _AMBIGUOUS_MULTI_VALUE.
    result = _parse_contract_output("1 2", contract, effective_strict=False)
    assert not result.ok
    assert "Ambiguous" in result.error_msg or "multiple" in result.error_msg


def test_parse_contract_output_lenient_no_json_found() -> None:
    """_parse_contract_output: lenient mode with no JSON at all → failure."""
    from agm.agl.ir.contracts import ContractRequest, ScalarDecode, ScalarKind
    from agm.agl.runtime.codec import _parse_contract_output

    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema=_json.dumps({"type": "integer"}),
        decode=ScalarDecode(ScalarKind.INT),
        target_type_label="int",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    result = _parse_contract_output("no json here at all!@#$%", contract, effective_strict=False)
    assert not result.ok


def test_parse_contract_output_schema_not_dict() -> None:
    """_parse_contract_output: json_schema that parses to a non-dict → failure."""
    from agm.agl.ir.contracts import ContractRequest, ScalarDecode, ScalarKind
    from agm.agl.runtime.codec import _parse_contract_output

    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema="[]",  # a list, not a dict
        decode=ScalarDecode(ScalarKind.INT),
        target_type_label="int",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    result = _parse_contract_output("42", contract, effective_strict=False)
    assert not result.ok
    assert "not a JSON object" in result.error_msg


def test_parse_contract_output_decode_none() -> None:
    """_parse_contract_output: decode=None with valid schema → failure."""
    from agm.agl.ir.contracts import ContractRequest
    from agm.agl.runtime.codec import _parse_contract_output

    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema=_json.dumps({"type": "integer"}),
        decode=None,  # deliberately None
        target_type_label="int",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    result = _parse_contract_output("42", contract, effective_strict=False)
    assert not result.ok
    assert "decode" in result.error_msg


def test_parse_contract_output_strict_parse_failure() -> None:
    """_parse_contract_output: strict mode with invalid JSON → failure."""
    from agm.agl.ir.contracts import ContractRequest, ScalarDecode, ScalarKind
    from agm.agl.runtime.codec import _parse_contract_output

    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema=_json.dumps({"type": "integer"}),
        decode=ScalarDecode(ScalarKind.INT),
        target_type_label="int",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    result = _parse_contract_output("not json!", contract, effective_strict=True)
    assert not result.ok
    assert "Strict JSON" in result.error_msg


def test_parse_agent_output_required_field_error() -> None:
    """parse_agent_output: missing required field on record → missing_field error."""
    from agm.agl.ir.contracts import (
        ContractRequest,
        RecordDecode,
        ScalarDecode,
        ScalarKind,
    )
    from agm.agl.ir.ids import NominalId as IrNominalId
    from agm.agl.runtime.codec import _parse_contract_output

    nom = IrNominalId(1)
    schema = _json.dumps(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["x", "y"],
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
        }
    )
    decode = RecordDecode(
        nominal=nom,
        display_name="Point",
        fields=(("x", ScalarDecode(ScalarKind.INT)), ("y", ScalarDecode(ScalarKind.INT))),
    )
    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema=schema,
        decode=decode,
        target_type_label="Point",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    # Missing 'y' field.
    result = _parse_contract_output('{"x": 1}', contract, effective_strict=False)
    assert not result.ok
    assert len(result.errors) >= 1
    assert any(e.category == "missing_field" for e in result.errors)


def test_parse_agent_output_additional_properties_error() -> None:
    """parse_agent_output: extra field on record → unknown_field error."""
    from agm.agl.ir.contracts import ContractRequest, RecordDecode, ScalarDecode, ScalarKind
    from agm.agl.ir.ids import NominalId
    from agm.agl.runtime.codec import _parse_contract_output

    nom = NominalId(1)
    schema = _json.dumps(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["x"],
            "properties": {"x": {"type": "integer"}},
        }
    )
    decode = RecordDecode(
        nominal=nom,
        display_name="Point",
        fields=(("x", ScalarDecode(ScalarKind.INT)),),
    )
    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema=schema,
        decode=decode,
        target_type_label="Point",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    result = _parse_contract_output('{"x": 1, "extra": true}', contract, effective_strict=False)
    assert not result.ok
    assert any(e.category == "unknown_field" for e in result.errors)


def test_parse_agent_output_wrong_type_error() -> None:
    """parse_agent_output: wrong JSON type (string vs integer) → wrong_type error."""
    from agm.agl.ir.contracts import ContractRequest, ScalarDecode, ScalarKind
    from agm.agl.runtime.codec import _parse_contract_output

    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema=_json.dumps({"type": "integer"}),
        decode=ScalarDecode(ScalarKind.INT),
        target_type_label="int",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    result = _parse_contract_output('"not an int"', contract, effective_strict=False)
    assert not result.ok
    assert any(e.category == "wrong_type" for e in result.errors)


def test_parse_agent_output_unknown_validator_fallback() -> None:
    """parse_agent_output: schema with minimum validator → wrong_type fallback."""
    from agm.agl.ir.contracts import ContractRequest, ScalarDecode, ScalarKind
    from agm.agl.runtime.codec import _parse_contract_output

    # Use a "minimum" constraint that fails — falls through to default ValidationError.
    schema = _json.dumps({"type": "integer", "minimum": 100})
    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema=schema,
        decode=ScalarDecode(ScalarKind.INT),
        target_type_label="int",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    result = _parse_contract_output("5", contract, effective_strict=False)
    assert not result.ok
    # The error should be categorized (wrong_type is our fallback for unknown validators)
    assert len(result.errors) >= 1


def test_make_validation_error_non_error_object() -> None:
    """_make_validation_error: non-ValidationError argument → wrong_type with str()."""
    from agm.agl.ir.contracts import ScalarDecode, ScalarKind
    from agm.agl.runtime.codec import _make_validation_error

    decode = ScalarDecode(ScalarKind.INT)
    ve = _make_validation_error("plain string error", decode)
    assert ve.category == "wrong_type"
    assert "plain string error" in ve.message


def test_enum_instance_not_dict_bad_case() -> None:
    """_classify_enum_failure: non-dict instance → bad_case error."""
    from agm.agl.ir.contracts import (
        ContractRequest,
        EnumDecode,
        VariantDecode,
    )
    from agm.agl.ir.ids import NominalId
    from agm.agl.runtime.codec import _parse_contract_output

    nominal = NominalId(1)
    decode = EnumDecode(
        nominal=nominal,
        display_name="Flag",
        variants=(
            VariantDecode(name="On", nominal=NominalId(999), display_name="On", fields=()),
            VariantDecode(name="Off", nominal=NominalId(999), display_name="Off", fields=()),
        ),
    )
    schema = _json.dumps(
        {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["$case"],
                    "properties": {"$case": {"const": "On"}},
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["$case"],
                    "properties": {"$case": {"const": "Off"}},
                },
            ]
        }
    )
    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema=schema,
        decode=decode,
        target_type_label="Flag",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    # Pass a non-dict (string) → instance not dict path.
    result = _parse_contract_output('"not-a-dict"', contract, effective_strict=False)
    assert not result.ok
    assert any(e.category == "bad_case" for e in result.errors)


def test_enum_no_case_tag_bad_case() -> None:
    """_classify_enum_failure: dict missing $case → bad_case error."""
    from agm.agl.ir.contracts import (
        ContractRequest,
        EnumDecode,
        VariantDecode,
    )
    from agm.agl.ir.ids import NominalId

    nominal = NominalId(1)
    decode = EnumDecode(
        nominal=nominal,
        display_name="Flag",
        variants=(
            VariantDecode(name="On", nominal=NominalId(999), display_name="On", fields=()),
            VariantDecode(name="Off", nominal=NominalId(999), display_name="Off", fields=()),
        ),
    )
    schema = _json.dumps(
        {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["$case"],
                    "properties": {"$case": {"const": "On"}},
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["$case"],
                    "properties": {"$case": {"const": "Off"}},
                },
            ]
        }
    )
    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema=schema,
        decode=decode,
        target_type_label="Flag",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    from agm.agl.runtime.codec import _parse_contract_output

    # Missing $case key entirely.
    result = _parse_contract_output('{"value": 42}', contract, effective_strict=False)
    assert not result.ok
    assert any(e.category == "bad_case" for e in result.errors)


def test_enum_bad_case_no_decode_schema() -> None:
    """decode=None in ContractRequest → failure (no decode schema check before validation)."""
    from agm.agl.ir.contracts import ContractRequest
    from agm.agl.runtime.codec import _parse_contract_output

    schema = _json.dumps(
        {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["$case"],
                    "properties": {"$case": {"const": "On"}},
                },
            ]
        }
    )
    # decode=None: _find_enum_decode_at_path returns None.
    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema=schema,
        decode=None,
        target_type_label="Flag",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    result = _parse_contract_output('{"$case": "Unknown"}', contract, effective_strict=False)
    assert not result.ok  # decode=None → failure before schema validation


def test_find_enum_decode_at_path_through_array() -> None:
    """_find_enum_decode_at_path: navigate through ArrayDecode to find EnumDecode."""
    from agm.agl.ir.contracts import (
        ArrayDecode,
        ContractRequest,
        EnumDecode,
        VariantDecode,
    )
    from agm.agl.ir.ids import NominalId
    from agm.agl.runtime.codec import _find_enum_decode_at_path

    nominal = NominalId(1)
    enum_dec = EnumDecode(
        nominal=nominal,
        display_name="Status",
        variants=(VariantDecode(name="Ok", nominal=NominalId(999), display_name="Ok", fields=()),),
    )
    array_dec = ArrayDecode(elem=enum_dec)
    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema="{}",
        decode=array_dec,
        target_type_label="array[Status]",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    # Navigate into element 0 of array.
    result = _find_enum_decode_at_path(contract.decode, [0])
    assert isinstance(result, EnumDecode)
    assert result.display_name == "Status"


def test_find_enum_decode_at_path_through_dict() -> None:
    """_find_enum_decode_at_path: navigate through DictDecode to find EnumDecode."""
    from agm.agl.ir.contracts import (
        ContractRequest,
        DictDecode,
        EnumDecode,
        VariantDecode,
    )
    from agm.agl.ir.ids import NominalId
    from agm.agl.runtime.codec import _find_enum_decode_at_path

    nominal = NominalId(1)
    enum_dec = EnumDecode(
        nominal=nominal,
        display_name="Status",
        variants=(VariantDecode(name="Ok", nominal=NominalId(999), display_name="Ok", fields=()),),
    )
    dict_dec = DictDecode(value=enum_dec)
    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema="{}",
        decode=dict_dec,
        target_type_label="dict[Status]",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    result = _find_enum_decode_at_path(contract.decode, ["somekey"])
    assert isinstance(result, EnumDecode)


def test_find_enum_decode_at_path_through_record() -> None:
    """_find_enum_decode_at_path: navigate through RecordDecode fields to find EnumDecode."""
    from agm.agl.ir.contracts import (
        ContractRequest,
        EnumDecode,
        RecordDecode,
        ScalarDecode,
        ScalarKind,
        VariantDecode,
    )
    from agm.agl.ir.ids import NominalId
    from agm.agl.runtime.codec import _find_enum_decode_at_path

    nominal = NominalId(1)
    enum_dec = EnumDecode(
        nominal=nominal,
        display_name="Status",
        variants=(VariantDecode(name="Ok", nominal=NominalId(999), display_name="Ok", fields=()),),
    )
    rec_nominal = NominalId(2)
    rec_dec = RecordDecode(
        nominal=rec_nominal,
        display_name="Wrapper",
        fields=(("status", enum_dec), ("n", ScalarDecode(ScalarKind.INT))),
    )
    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema="{}",
        decode=rec_dec,
        target_type_label="Wrapper",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    # Navigate into record field "status".
    result = _find_enum_decode_at_path(contract.decode, ["status"])
    assert isinstance(result, EnumDecode)

    # Non-existent field → None.
    result2 = _find_enum_decode_at_path(contract.decode, ["missing"])
    assert result2 is None

    # Non-string path element in record → None.
    result3 = _find_enum_decode_at_path(contract.decode, [0])
    assert result3 is None


def test_find_enum_decode_at_path_enum_at_top_navigated_into() -> None:
    """_find_enum_decode_at_path: enum at top level navigated deeper → None."""
    from agm.agl.ir.contracts import (
        ContractRequest,
        EnumDecode,
        VariantDecode,
    )
    from agm.agl.ir.ids import NominalId
    from agm.agl.runtime.codec import _find_enum_decode_at_path

    nominal = NominalId(1)
    enum_dec = EnumDecode(
        nominal=nominal,
        display_name="Status",
        variants=(VariantDecode(name="Ok", nominal=NominalId(999), display_name="Ok", fields=()),),
    )
    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema="{}",
        decode=enum_dec,
        target_type_label="Status",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    # Path goes inside the enum (invalid) → None.
    result = _find_enum_decode_at_path(contract.decode, ["something"])
    assert result is None


def test_find_enum_decode_at_path_scalar_navigated_into() -> None:
    """_find_enum_decode_at_path: scalar navigated into → None (else branch)."""
    from agm.agl.ir.contracts import (
        ContractRequest,
        ScalarDecode,
        ScalarKind,
    )
    from agm.agl.runtime.codec import _find_enum_decode_at_path

    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema="{}",
        decode=ScalarDecode(ScalarKind.INT),
        target_type_label="int",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    # Scalar can't be navigated into.
    result = _find_enum_decode_at_path(contract.decode, ["key"])
    assert result is None


def test_find_enum_decode_at_path_end_at_scalar() -> None:
    """_find_enum_decode_at_path: path ends at scalar → None (not EnumDecode)."""
    from agm.agl.ir.contracts import (
        ContractRequest,
        RecordDecode,
        ScalarDecode,
        ScalarKind,
    )
    from agm.agl.ir.ids import NominalId
    from agm.agl.runtime.codec import _find_enum_decode_at_path

    nom = NominalId(1)
    rec_dec = RecordDecode(
        nominal=nom,
        display_name="Point",
        fields=(("x", ScalarDecode(ScalarKind.INT)),),
    )
    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema="{}",
        decode=rec_dec,
        target_type_label="Point",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    # Navigate to "x" which is a ScalarDecode, not EnumDecode → None.
    result = _find_enum_decode_at_path(contract.decode, ["x"])
    assert result is None


def test_enum_known_case_with_additional_props_error() -> None:
    """_classify_enum_failure: known case but extra field → unknown_field."""
    from agm.agl.ir.contracts import (
        ContractRequest,
        EnumDecode,
        ScalarDecode,
        ScalarKind,
        VariantDecode,
    )
    from agm.agl.ir.ids import NominalId

    nominal = NominalId(1)
    decode = EnumDecode(
        nominal=nominal,
        display_name="Status",
        variants=(
            VariantDecode(name="Ok", nominal=NominalId(999), display_name="Ok", fields=()),
            VariantDecode(
                name="Err",
                nominal=NominalId(999),
                display_name="Err",
                fields=(("msg", ScalarDecode(ScalarKind.TEXT)),),
            ),
        ),
    )
    schema = _json.dumps(
        {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["$case"],
                    "properties": {"$case": {"const": "Ok"}},
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["$case", "msg"],
                    "properties": {"$case": {"const": "Err"}, "msg": {"type": "string"}},
                },
            ]
        }
    )
    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema=schema,
        decode=decode,
        target_type_label="Status",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    from agm.agl.runtime.codec import _parse_contract_output

    # Ok with extra field → additionalProperties error.
    result = _parse_contract_output('{"$case": "Ok", "extra": 1}', contract, effective_strict=False)
    assert not result.ok
    assert any(e.category == "unknown_field" for e in result.errors)


def test_validate_contract_request_json_missing_decode() -> None:
    """validate.py: json codec with json_schema set but decode=None → InvalidIrError."""

    from agm.agl.ir.contracts import ContractRequest
    from agm.agl.ir.ids import ContractId, Location, SourceId
    from agm.agl.ir.nodes import IrConstUnit
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import InvalidIrError, validate_ir
    from agm.agl.modules.ids import ENTRY_ID

    src_id = SourceId(0)
    dummy_loc = Location(source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0)
    cid = ContractId(0)
    bad_req = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema=_json.dumps({"type": "integer"}),  # schema is present!
        decode=None,  # but decode is missing!
        target_type_label="int",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    prog = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={
            ENTRY_ID: ExecutableModule(
                module_id=ENTRY_ID,
                initializers=(IrConstUnit(location=dummy_loc),),
            )
        },
        symbols={},
        nominals={},
        sources={src_id: SourceFile(display_name="<test>", normalized_text="test")},
        contracts={cid: bad_req},
    )
    with pytest.raises(InvalidIrError, match="decode"):
        validate_ir(prog, deep=True)


def test_validate_contract_request_json_decode_check_nominals() -> None:
    """validate.py: json codec with decode→_check_decode_nominals is called (808→exit path)."""
    from agm.agl.ir.contracts import ContractRequest, ScalarDecode, ScalarKind
    from agm.agl.ir.ids import ContractId, Location, SourceId
    from agm.agl.ir.nodes import IrConstUnit
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import validate_ir
    from agm.agl.modules.ids import ENTRY_ID

    src_id = SourceId(0)
    dummy_loc = Location(source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0)
    cid = ContractId(0)
    req = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema=_json.dumps({"type": "integer"}),
        decode=ScalarDecode(ScalarKind.INT),
        target_type_label="int",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    prog = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={
            ENTRY_ID: ExecutableModule(
                module_id=ENTRY_ID,
                initializers=(IrConstUnit(location=dummy_loc),),
            )
        },
        symbols={},
        nominals={},
        sources={src_id: SourceFile(display_name="<test>", normalized_text="test")},
        contracts={cid: req},
    )
    # Should pass validation without error (covers 808→_check_decode_nominals).
    validate_ir(prog, deep=True)


def test_validate_contract_request_recursive_decode_defs() -> None:
    """validate.py accepts a ContractRequest whose decode is a recursive RefDecode + defs table."""
    from agm.agl.ir.contracts import (
        ContractRequest,
        EnumDecode,
        RefDecode,
        ScalarDecode,
        ScalarKind,
        VariantDecode,
    )
    from agm.agl.ir.ids import ContractId, Location, NominalId, SourceId
    from agm.agl.ir.nodes import IrConstUnit
    from agm.agl.ir.program import (
        ExecutableModule,
        ExecutableProgram,
        NominalDescriptor,
        NominalKind,
        SourceFile,
        VariantDescriptor,
    )
    from agm.agl.ir.validate import validate_ir
    from agm.agl.modules.ids import ENTRY_ID

    src_id = SourceId(0)
    dummy_loc = Location(source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0)
    tree_nominal = NominalId(10)
    tree_body = EnumDecode(
        nominal=tree_nominal,
        display_name="Tree",
        variants=(
            VariantDecode("Leaf", NominalId(11), "Tree::Leaf", ()),
            VariantDecode(
                "Node",
                NominalId(12),
                "Tree::Node",
                (
                    ("value", ScalarDecode(ScalarKind.INT)),
                    ("left", RefDecode("Tree")),
                    ("right", RefDecode("Tree")),
                ),
            ),
        ),
    )
    cid = ContractId(0)
    req = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema=_json.dumps({"$ref": "#/$defs/Tree"}),
        decode=RefDecode("Tree"),
        target_type_label="Tree",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
        defs=(("Tree", tree_body),),
    )
    prog = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={
            ENTRY_ID: ExecutableModule(
                module_id=ENTRY_ID,
                initializers=(IrConstUnit(location=dummy_loc),),
            )
        },
        symbols={},
        nominals={
            tree_nominal: NominalDescriptor(
                nominal=tree_nominal,
                module_id=ENTRY_ID,
                scope_path=(),
                declared_name="Tree",
                kind=NominalKind.ENUM,
                variants=(
                    VariantDescriptor("Leaf", (), NominalId(11)),
                    VariantDescriptor("Node", ("value", "left", "right"), NominalId(12)),
                ),
            ),
            NominalId(11): NominalDescriptor(
                NominalId(11), ENTRY_ID, ("Tree",), "Leaf", NominalKind.RECORD
            ),
            NominalId(12): NominalDescriptor(
                NominalId(12),
                ENTRY_ID,
                ("Tree",),
                "Node",
                NominalKind.RECORD,
                ("value", "left", "right"),
            ),
        },
        sources={src_id: SourceFile(display_name="<test>", normalized_text="test")},
        contracts={cid: req},
    )
    validate_ir(prog, deep=True)  # no exception


def test_validate_contract_request_recursive_decode_unknown_defs_key() -> None:
    """An unresolvable RefDecode key in a ContractRequest's decode → InvalidIrError."""
    from agm.agl.ir.contracts import ContractRequest, RefDecode
    from agm.agl.ir.ids import ContractId, Location, SourceId
    from agm.agl.ir.nodes import IrConstUnit
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import InvalidIrError, validate_ir
    from agm.agl.modules.ids import ENTRY_ID

    src_id = SourceId(0)
    dummy_loc = Location(source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0)
    cid = ContractId(0)
    req = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema=_json.dumps({"$ref": "#/$defs/Tree"}),
        decode=RefDecode("Tree"),
        target_type_label="Tree",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
        defs=(),  # missing the "Tree" entry
    )
    prog = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={
            ENTRY_ID: ExecutableModule(
                module_id=ENTRY_ID,
                initializers=(IrConstUnit(location=dummy_loc),),
            )
        },
        symbols={},
        nominals={},
        sources={src_id: SourceFile(display_name="<test>", normalized_text="test")},
        contracts={cid: req},
    )
    with pytest.raises(InvalidIrError, match=r"unknown \$defs key"):
        validate_ir(prog, deep=True)


# ---------------------------------------------------------------------------
# ir_interpreter.py uncovered paths
# ---------------------------------------------------------------------------


def test_ir_ask_request_text_contract() -> None:
    """IrAskRequest builds an AgentRequest with its fixed text contract."""
    source = """\
let a = AgentCommand("a")
let req = ask-request("Do it.", agent = a)
let prompt-text: text = req.prompt
prompt-text
"""
    from tests.agl.ir_harness import evaluate_ir_with_agents

    ir = evaluate_ir_with_agents(
        source,
        scripts={"a": []},
    )
    assert ir["prompt-text"] == TextValue("Do it.")


def test_lower_on_parse_error_abort_gives_one_attempt() -> None:
    """_extract_max_attempts: Abort policy → 1 attempt."""
    source = """\
let a = AgentCommand("a")
let n: int = ask("?", agent = a, on-parse-error = Abort)
n
"""
    from tests.agl.ir_harness import evaluate_ir_with_agents

    ir = evaluate_ir_with_agents(
        source,
        scripts={"a": ["5"]},
    )
    assert ir["n"] == IntValue(5)


# ---------------------------------------------------------------------------
# additional coverage tests
# ---------------------------------------------------------------------------


def test_parse_lenient_json_parse_failure_after_repair() -> None:
    """_parse_contract_output lenient: json_text found but json.loads fails → failure."""
    from agm.agl.ir.contracts import ContractRequest, ScalarDecode, ScalarKind
    from agm.agl.runtime.codec import _parse_contract_output

    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema=_json.dumps({"type": "integer"}),
        decode=ScalarDecode(ScalarKind.INT),
        target_type_label="int",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    # Strict mode parse failure.
    result = _parse_contract_output("this is not json at all!!", contract, effective_strict=True)
    assert not result.ok
    assert "Strict JSON" in result.error_msg


def test_parse_value_conversion_failure() -> None:
    """_parse_contract_output: schema valid but decode raises ValueError → failure."""
    from agm.agl.ir.contracts import ContractRequest, ScalarDecode, ScalarKind
    from agm.agl.runtime.codec import _parse_contract_output

    schema = _json.dumps({"type": "string"})
    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema=schema,
        decode=ScalarDecode(ScalarKind.INT),  # INT decode on a string value
        target_type_label="int",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    result = _parse_contract_output('"not-a-number"', contract, effective_strict=False)
    # Either schema validation fails (wrong_type) or decode fails (value conversion)
    assert not result.ok


def test_validate_ir_ask_deep_valid_contract() -> None:
    """validate_ir: IrAsk with valid contract passes deep validation (656->exit path)."""
    from agm.agl.ir.contracts import ContractRequest
    from agm.agl.ir.ids import ContractId, Location, SourceId
    from agm.agl.ir.nodes import IrAsk, IrConstText
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import validate_ir
    from agm.agl.modules.ids import ENTRY_ID

    src_id = SourceId(0)
    dummy_loc = Location(source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0)
    cid = ContractId(0)
    req = ContractRequest(
        codec_name="text",
        strict_json=None,
        json_schema=None,
        decode=None,
        target_type_label="text",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    node = IrAsk(
        location=dummy_loc,
        agent=IrConstText(location=dummy_loc, value="ask"),
        prompt=IrConstText(location=dummy_loc, value="test"),
        contract_id=cid,
        max_attempts=1,
    )
    prog = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=(node,))},
        symbols={},
        nominals={},
        sources={src_id: SourceFile(display_name="<test>", normalized_text="test")},
        contracts={cid: req},
    )
    # Should pass without error (covers the deep valid path 656->exit).
    validate_ir(prog, deep=True)


def test_validate_ir_ask_request_deep_valid_contract() -> None:
    """validate_ir: a well-formed IrAskRequest passes deep validation."""
    from agm.agl.ir.contracts import ContractRequest
    from agm.agl.ir.ids import ContractId, Location, SourceId
    from agm.agl.ir.nodes import IrAskRequest, IrConstText
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import validate_ir
    from agm.agl.modules.ids import ENTRY_ID

    src_id = SourceId(0)
    dummy_loc = Location(source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0)
    cid = ContractId(0)
    node = IrAskRequest(
        location=dummy_loc,
        agent=IrConstText(location=dummy_loc, value="ask"),
        prompt=IrConstText(location=dummy_loc, value="test"),
        contract_id=cid,
        max_attempts=1,
    )
    prog = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=(node,))},
        symbols={},
        nominals={},
        sources={src_id: SourceFile(display_name="<test>", normalized_text="test")},
        contracts={
            cid: ContractRequest(
                codec_name="text",
                strict_json=None,
                json_schema=None,
                decode=None,
                target_type_label="text",
                structured_exec=False,
                format_instructions="",
                is_unit=False,
            )
        },
    )
    validate_ir(prog, deep=True)


def test_ir_ask_request_has_a_text_target() -> None:
    """ask-request always reports the fixed text target on its request record."""
    source = """\
let a = AgentCommand("a")
let req = ask-request("Do it.", agent = a)
let target = req.target-type
target
"""
    from tests.agl.ir_harness import evaluate_ir_with_agents

    ir = evaluate_ir_with_agents(
        source,
        scripts={"a": []},
    )
    assert isinstance(ir["target"], RecordValue)
    assert ir["target"].display_name == "Option::Some"
    assert ir["target"].fields["value"] == TextValue("text")


def test_lower_on_parse_error_self_qualified_retry() -> None:
    """Self-qualified Retry parse policy produces the correct attempt count."""
    source = """\
let a = AgentCommand("a")
let n: int = ask("?", agent = a, on-parse-error = ::Retry(n = 2))
n
"""
    from tests.agl.ir_harness import evaluate_ir_with_agents

    # First 2 responses are bad JSON, 3rd is valid.
    ir = evaluate_ir_with_agents(
        source,
        scripts={"a": ["bad", "bad", "7"]},
    )
    assert ir["n"] == IntValue(7)


@pytest.mark.parametrize("request_only", (False, True))
def test_validate_ir_ask_shallow_does_not_check_contracts(request_only: bool) -> None:
    """validate_ir: shallow (deep=False) validation skips an ask node's contract checks."""
    from agm.agl.ir.ids import ContractId, Location, SourceId
    from agm.agl.ir.nodes import IrAsk, IrAskRequest, IrConstText
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import validate_ir
    from agm.agl.modules.ids import ENTRY_ID

    src_id = SourceId(0)
    dummy_loc = Location(source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0)
    bad_cid = ContractId(999)
    node_type: type[IrAsk] | type[IrAskRequest] = IrAskRequest if request_only else IrAsk
    node = node_type(
        location=dummy_loc,
        agent=IrConstText(location=dummy_loc, value="ask"),
        prompt=IrConstText(location=dummy_loc, value="test"),
        contract_id=bad_cid,
        max_attempts=1,
    )
    prog = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=(node,))},
        symbols={},
        nominals={},
        sources={src_id: SourceFile(display_name="<test>", normalized_text="test")},
        contracts={},  # Missing contract, but deep=False skips checks.
    )
    # deep=False → skips contract_id check (covers 656->exit branch).
    validate_ir(prog, deep=False)


def test_validate_contract_request_json_is_unit_decode_none() -> None:
    """validate.py: json codec with is_unit=True and decode=None → no error (808->exit path)."""
    from agm.agl.ir.contracts import ContractRequest
    from agm.agl.ir.ids import ContractId, Location, SourceId
    from agm.agl.ir.nodes import IrConstUnit
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import validate_ir
    from agm.agl.modules.ids import ENTRY_ID

    src_id = SourceId(0)
    dummy_loc = Location(source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0)
    cid = ContractId(0)
    # json codec, is_unit=True, decode=None → 800 skipped, 804 skipped, 808 decode is None
    req = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema=None,
        decode=None,
        target_type_label="unit",
        structured_exec=False,
        format_instructions="",
        is_unit=True,  # is_unit bypasses json_schema/decode required checks
    )
    prog = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={
            ENTRY_ID: ExecutableModule(
                module_id=ENTRY_ID,
                initializers=(IrConstUnit(location=dummy_loc),),
            )
        },
        symbols={},
        nominals={},
        sources={src_id: SourceFile(display_name="<test>", normalized_text="test")},
        contracts={cid: req},
    )
    # Should pass validation (decode is None, is_unit=True → 808->exit path).
    validate_ir(prog, deep=True)


def _make_span() -> "SourceSpan":
    """Return a minimal SourceSpan for use in hand-built AST nodes."""
    from agm.agl.syntax.spans import SourceSpan

    return SourceSpan(
        start_line=1, start_col=1, end_line=1, end_col=2, start_offset=0, end_offset=1
    )


def _extract_max_attempts_for_test(
    outer_call: "Call",
) -> int:
    """Call _Lowerer._extract_max_attempts via object.__new__ to avoid full initialisation."""
    from agm.agl.lower.lowerer import _Lowerer

    fake: _Lowerer = object.__new__(_Lowerer)
    return fake._extract_max_attempts(outer_call)


def test_lower_extract_max_attempts_field_access_retry() -> None:
    """_extract_max_attempts: FieldAccess callee → callee.field = 'Retry'."""
    from agm.agl.syntax.nodes import Call, FieldAccess, IntLit, NamedArg, VarRef

    span = _make_span()
    # FieldAccess callee: like writing `somemod.Retry`.
    field_callee = FieldAccess(
        obj=VarRef(name="somemod", span=span, node_id=1),
        field="Retry",
        span=span,
        node_id=2,
    )
    n_arg = NamedArg(name="n", value=IntLit(value=3, span=span, node_id=5), span=span, node_id=6)
    inner_call = Call(
        callee=field_callee,
        args=(),
        named_args=(n_arg,),
        span=span,
        node_id=3,
    )
    named_arg = NamedArg(name="on-parse-error", value=inner_call, span=span, node_id=7)
    outer_call = Call(
        callee=VarRef(name="ask", span=span, node_id=8),
        args=(),
        named_args=(named_arg,),
        span=span,
        node_id=4,
    )
    result = _extract_max_attempts_for_test(outer_call)
    # Retry(n: 3) → 1 + 3 = 4.
    assert result == 4


def test_lower_extract_max_attempts_unknown_callee() -> None:
    """_extract_max_attempts: non-VarRef/FieldAccess callee → callee_name=None → 1 attempt."""
    from agm.agl.syntax.nodes import Call, IntLit, NamedArg, VarRef

    span = _make_span()
    # Use IntLit as callee (not VarRef or FieldAccess) → else branch → callee_name=None.
    # IntLit is a valid Expr, so this is type-clean; callee type is the Expr union.
    weird_callee = IntLit(value=0, span=span, node_id=10)
    inner_call = Call(
        callee=weird_callee,
        args=(),
        named_args=(),
        span=span,
        node_id=11,
    )
    named_arg = NamedArg(name="on-parse-error", value=inner_call, span=span, node_id=14)
    outer_call = Call(
        callee=VarRef(name="ask", span=span, node_id=12),
        args=(),
        named_args=(named_arg,),
        span=span,
        node_id=13,
    )
    result = _extract_max_attempts_for_test(outer_call)
    # callee_name=None → not "Retry" → returns 1.
    assert result == 1


def test_lower_extract_max_attempts_field_access_non_retry() -> None:
    """_extract_max_attempts: FieldAccess callee with non-Retry field → 1 attempt."""
    from agm.agl.syntax.nodes import Call, FieldAccess, NamedArg, VarRef

    span = _make_span()
    # FieldAccess with non-Retry field name.
    field_callee = FieldAccess(
        obj=VarRef(name="somemod", span=span, node_id=20),
        field="Abort",  # not "Retry"!
        span=span,
        node_id=21,
    )
    inner_call = Call(
        callee=field_callee,
        args=(),
        named_args=(),
        span=span,
        node_id=22,
    )
    named_arg = NamedArg(name="on-parse-error", value=inner_call, span=span, node_id=25)
    outer_call = Call(
        callee=VarRef(name="ask", span=span, node_id=23),
        args=(),
        named_args=(named_arg,),
        span=span,
        node_id=24,
    )
    result = _extract_max_attempts_for_test(outer_call)
    # callee_name="Abort" (from FieldAccess.field), not "Retry" → returns 1.
    assert result == 1


def test_enum_required_field_loop_partial_coverage() -> None:
    """_classify_enum_failure: known case with missing field → missing_field (loop covers all)."""
    from agm.agl.ir.contracts import (
        ContractRequest,
        EnumDecode,
        ScalarDecode,
        ScalarKind,
        VariantDecode,
    )
    from agm.agl.ir.ids import NominalId

    nominal = NominalId(1)
    decode = EnumDecode(
        nominal=nominal,
        display_name="Pair",
        variants=(
            VariantDecode(
                name="Both",
                nominal=NominalId(999),
                display_name="Both",
                fields=(("a", ScalarDecode(ScalarKind.INT)), ("b", ScalarDecode(ScalarKind.INT))),
            ),
        ),
    )
    # Schema requiring both 'a' and 'b'.
    schema = _json.dumps(
        {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["$case", "a", "b"],
                    "properties": {
                        "$case": {"const": "Both"},
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                }
            ]
        }
    )
    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema=schema,
        decode=decode,
        target_type_label="Pair",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    from agm.agl.runtime.codec import _parse_contract_output

    # Both 'a' present but 'b' is missing → required error with 2 items in list.
    result = _parse_contract_output('{"$case": "Both", "a": 1}', contract, effective_strict=False)
    assert not result.ok
    # Should find 'b' as missing (loop iterates past '$case' and 'a').
    assert any(e.category in ("missing_field", "bad_case") for e in result.errors)


def test_parse_lenient_extracted_json_fails_decode() -> None:
    """_parse_contract_output lenient: extracted JSON text fails json.loads."""
    from unittest.mock import patch

    from agm.agl.ir.contracts import ContractRequest, ScalarDecode, ScalarKind
    from agm.agl.runtime.codec import _parse_contract_output

    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema=_json.dumps({"type": "integer"}),
        decode=ScalarDecode(ScalarKind.INT),
        target_type_label="int",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    # Patch _extract_json_text in codec to return a non-ambiguous broken string.
    with patch("agm.agl.runtime.codec._extract_json_text", return_value="{broken: !}"):
        result = _parse_contract_output("irrelevant", contract, effective_strict=False)
    assert not result.ok
    assert "JSON parse failed" in result.error_msg


def test_make_validation_error_required_non_array() -> None:
    """_make_validation_error: required validator with non-array required → field=None."""
    from unittest.mock import MagicMock

    from jsonschema import ValidationError as JsError

    from agm.agl.ir.contracts import ScalarDecode, ScalarKind
    from agm.agl.runtime.codec import _make_validation_error

    decode = ScalarDecode(ScalarKind.INT)
    fake_error = MagicMock(spec=JsError)
    fake_error.validator = "required"
    fake_error.validator_value = "not-a-list"  # not a list → name stays None
    fake_error.instance = {"some": "dict"}
    fake_error.message = "something required"
    fake_error.path = []

    ve = _make_validation_error(fake_error, decode)
    assert ve.category == "missing_field"
    assert ve.field is None  # name was never set (loop never ran)


def test_make_validation_error_required_all_present() -> None:
    """_make_validation_error: required validator where all fields ARE present → field=None."""
    from unittest.mock import MagicMock

    from jsonschema import ValidationError as JsError

    from agm.agl.ir.contracts import ScalarDecode, ScalarKind
    from agm.agl.runtime.codec import _make_validation_error

    decode = ScalarDecode(ScalarKind.INT)
    # Mock a required error where required=["x"] but instance already has "x".
    fake_error = MagicMock(spec=JsError)
    fake_error.validator = "required"
    fake_error.validator_value = ["x"]
    fake_error.instance = {"x": 1}  # x IS present → loop completes without break
    fake_error.message = "required property"
    fake_error.path = []

    ve = _make_validation_error(fake_error, decode)
    assert ve.category == "missing_field"
    assert ve.field is None  # missing was never set (break never happened)


def test_classify_enum_sub_error_type_only_fallback() -> None:
    """_classify_enum_failure: known case, type-mismatch payload → defensive bad_case fallback."""
    from agm.agl.ir.contracts import (
        ContractRequest,
        EnumDecode,
        ScalarDecode,
        ScalarKind,
        VariantDecode,
    )
    from agm.agl.ir.ids import NominalId

    nominal = NominalId(1)
    decode = EnumDecode(
        nominal=nominal,
        display_name="Status",
        variants=(
            VariantDecode(name="Ok", nominal=NominalId(999), display_name="Ok", fields=()),
            VariantDecode(
                name="Err",
                nominal=NominalId(999),
                display_name="Err",
                fields=(("msg", ScalarDecode(ScalarKind.TEXT)),),
            ),
        ),
    )
    # Schema WITHOUT additionalProperties: False → sub-errors will be 'const' and 'type' only.
    schema = _json.dumps(
        {
            "oneOf": [
                {"type": "object", "required": ["$case"], "properties": {"$case": {"const": "Ok"}}},
                {
                    "type": "object",
                    "required": ["$case", "msg"],
                    "properties": {"$case": {"const": "Err"}, "msg": {"type": "string"}},
                },
            ]
        }
    )
    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema=schema,
        decode=decode,
        target_type_label="Status",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    from agm.agl.runtime.codec import _parse_contract_output

    # Known case "Err" but wrong type for "msg" (integer): all fields present, no extra fields
    # → _classify_enum_failure falls through to the defensive bad_case fallback.
    raw = _json.dumps({"$case": "Err", "msg": 42})
    result = _parse_contract_output(raw, contract, effective_strict=False)
    assert not result.ok
    assert any(e.category == "bad_case" for e in result.errors)


def test_classify_enum_failure_nullary_case_all_fields_present() -> None:
    """_classify_enum_failure: known nullary case with no missing/extra fields → fallback."""
    from unittest.mock import MagicMock

    from jsonschema import ValidationError as JsError

    from agm.agl.ir.contracts import EnumDecode, VariantDecode
    from agm.agl.ir.ids import NominalId
    from agm.agl.runtime.codec import _classify_enum_failure

    nominal = NominalId(1)
    decode = EnumDecode(
        nominal=nominal,
        display_name="Status",
        variants=(
            VariantDecode(name="Err", nominal=NominalId(999), display_name="Err", fields=()),
        ),
    )
    # instance has only "$case" → no missing or extra fields in the nullary "Err" variant.
    main_error = MagicMock(spec=JsError)
    main_error.validator = "oneOf"
    main_error.instance = {"$case": "Err"}
    main_error.absolute_path = []
    main_error.path = []

    ve = _classify_enum_failure(main_error, "$", decode)
    # All variant fields accounted for → defensive fallback returns bad_case.
    assert ve.category == "bad_case"


def test_classify_enum_failure_known_case_all_payload_present() -> None:
    """_classify_enum_failure: known case with all payload fields present → defensive fallback."""
    from unittest.mock import MagicMock

    from jsonschema import ValidationError as JsError

    from agm.agl.ir.contracts import EnumDecode, ScalarDecode, ScalarKind, VariantDecode
    from agm.agl.ir.ids import NominalId
    from agm.agl.runtime.codec import _classify_enum_failure

    nominal = NominalId(1)
    decode = EnumDecode(
        nominal=nominal,
        display_name="Status",
        variants=(
            VariantDecode(name="Ok", nominal=NominalId(999), display_name="Ok", fields=()),
            VariantDecode(
                name="Err",
                nominal=NominalId(999),
                display_name="Err",
                fields=(("msg", ScalarDecode(ScalarKind.TEXT)),),
            ),
        ),
    )
    # "msg" IS present — no missing, no extra → defensive fallback.
    main_error = MagicMock(spec=JsError)
    main_error.validator = "oneOf"
    main_error.instance = {"$case": "Err", "msg": "hello"}
    main_error.absolute_path = []
    main_error.path = []

    ve = _classify_enum_failure(main_error, "$", decode)
    assert ve.category == "bad_case"


# ---------------------------------------------------------------------------
# Enum oneOf validation-error message tests
#
# These tests verify that when ask() exhausts all attempts due to enum errors,
# the IR pipeline produces AgentParseError with the expected validation_errors
# (category, message, field) content.
# ---------------------------------------------------------------------------


def test_enum_unknown_case_exhausted() -> None:
    """Enum ask exhausted: unknown $case → AgentParseError with expected validation_errors.

    Verifies the exact message content for the unknown-$case shape.
    """
    source = """\
enum Status
  | Ok
  | Err(msg: text)

let checker = AgentCommand("checker")
let status: Status = ask("Check.", agent = checker)
status
"""
    ir_exc = evaluate_ir_raises_with_agents(
        source,
        scripts={"checker": ['{"$case": "Bogus"}']},
    )
    assert ir_exc.display_name == "AgentParseError"
    # validation_errors is stored as a JsonValue(raw=[{...}]).
    from agm.agl.semantics.values import JsonValue

    errors_val = ir_exc.fields.get("validation-errors")
    assert isinstance(errors_val, JsonValue)
    assert isinstance(errors_val.raw, list)
    assert len(errors_val.raw) >= 1
    first_err = errors_val.raw[0]
    assert isinstance(first_err, dict)
    msg = first_err.get("message", "")
    # The message identifies the bad case and lists the valid variants.
    assert 'Unknown "$case"' in msg
    assert "Bogus" in msg
    assert "Status" in msg
    assert "Ok" in msg
    assert "Err" in msg


def test_enum_missing_field_exhausted() -> None:
    """Enum ask exhausted: known $case but missing field → AgentParseError.

    Verifies the expected validation_errors message and field content for the missing-field shape.
    """
    source = """\
enum Status
  | Ok
  | Err(msg: text)

let checker = AgentCommand("checker")
let status: Status = ask("Check.", agent = checker)
status
"""
    ir_exc = evaluate_ir_raises_with_agents(
        source,
        scripts={"checker": ['{"$case": "Err"}']},
    )
    assert ir_exc.display_name == "AgentParseError"

    from agm.agl.semantics.values import JsonValue

    errors_val = ir_exc.fields.get("validation-errors")
    assert isinstance(errors_val, JsonValue)
    assert isinstance(errors_val.raw, list)
    assert len(errors_val.raw) >= 1
    first_err = errors_val.raw[0]
    assert isinstance(first_err, dict)
    msg = first_err.get("message", "")
    # IR reference message: "Enum variant 'Err' is missing field 'msg'."
    assert "Err" in msg
    assert "msg" in msg
    assert "missing" in msg.lower()
    # field attribute must name the missing field.
    assert first_err.get("field") == "msg"

    # Verify the same validation_errors fields are consistent.
    ir_errors_val = ir_exc.fields.get("validation-errors")
    assert isinstance(ir_errors_val, JsonValue)
    ir_first = ir_errors_val.raw[0]
    assert isinstance(ir_first, dict)
    assert ir_first.get("message") == msg, (
        f"Message mismatch:\n  reference: {ir_first.get('message')!r}\n  actual: {msg!r}"
    )
    assert ir_first.get("field") == first_err.get("field")


def test_enum_unexpected_field_exhausted() -> None:
    """Enum ask exhausted: known $case but unexpected field → AgentParseError.

    Verifies the expected validation_errors message and field content for the
    unexpected-field shape.  The IR sets field=key for unknown_field records (not None).
    """
    source = """\
enum Status
  | Ok
  | Err(msg: text)

let checker = AgentCommand("checker")
let status: Status = ask("Check.", agent = checker)
status
"""
    ir_exc = evaluate_ir_raises_with_agents(
        source,
        scripts={"checker": ['{"$case": "Ok", "extra_field": 42}']},
    )
    assert ir_exc.display_name == "AgentParseError"

    from agm.agl.semantics.values import JsonValue

    errors_val = ir_exc.fields.get("validation-errors")
    assert isinstance(errors_val, JsonValue)
    assert isinstance(errors_val.raw, list)
    assert len(errors_val.raw) >= 1
    first_err = errors_val.raw[0]
    assert isinstance(first_err, dict)
    msg = first_err.get("message", "")
    # IR reference message: "Enum variant 'Ok' has an unexpected field 'extra_field'."
    assert "Ok" in msg
    assert "extra_field" in msg
    assert "unexpected" in msg.lower()
    # IR reference sets field=key (the unexpected field name).
    assert first_err.get("field") == "extra_field"

    # Verify the same validation_errors fields are consistent.
    ir_errors_val = ir_exc.fields.get("validation-errors")
    assert isinstance(ir_errors_val, JsonValue)
    ir_first = ir_errors_val.raw[0]
    assert isinstance(ir_first, dict)
    assert ir_first.get("message") == msg, (
        f"Message mismatch:\n  reference: {ir_first.get('message')!r}\n  actual: {msg!r}"
    )
    assert ir_first.get("field") == first_err.get("field")


@pytest.mark.parametrize("request_only", (False, True))
def test_ir_ask_request_rejects_a_non_agent_value(request_only: bool) -> None:
    """Malformed IR cannot expose an AgentRequest whose agent is not an Agent value."""
    from agm.agl.eval.ir_interpreter import IrInterpreter
    from agm.agl.ir.contracts import ContractRequest
    from agm.agl.ir.ids import ContractId, Location, SourceId
    from agm.agl.ir.nodes import IrAsk, IrAskRequest, IrConstInt, IrConstText
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.modules.ids import ENTRY_ID

    source_id = SourceId(0)
    location = Location(
        source_id=source_id,
        start_offset=0,
        end_offset=1,
        start_line=1,
        start_col=0,
    )
    contract_id = ContractId(0)
    node: IrAsk | IrAskRequest
    node_type: type[IrAsk] | type[IrAskRequest] = IrAskRequest if request_only else IrAsk
    node = node_type(
        location=location,
        agent=IrConstInt(location=location, value=1),
        prompt=IrConstText(location=location, value="prompt"),
        contract_id=contract_id,
        max_attempts=1,
    )
    contracts: dict[ContractId, ContractRequest] = {
        contract_id: ContractRequest(
            codec_name="text",
            strict_json=None,
            json_schema=None,
            decode=None,
            target_type_label="text",
            structured_exec=False,
            format_instructions="",
            is_unit=False,
        )
    }
    program = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={
            ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=(node,)),
        },
        symbols={},
        nominals={},
        sources={source_id: SourceFile(display_name="<test>", normalized_text="test")},
        contracts=contracts,
    )

    with pytest.raises(TypeError, match="Agent member record"):
        IrInterpreter(program).run()
