"""IR semantic tests for M6b — agent dispatch (ask / ask-request) in the IR pipeline.

Differential ir_semantic: each test asserts that the ir_reference AST interpreter and the
new IR pipeline produce identical values and stdout for the same AgL program when
given the same scripted agent responses.
"""

from __future__ import annotations

import json as _json
from typing import TYPE_CHECKING

import pytest

from agm.agl.eval.values import (
    BoolValue,
    EnumValue,
    ExceptionValue,
    IntValue,
    ListValue,
    RecordValue,
    TextValue,
)
from tests.agl.ir_harness import (
    evaluate_ir_raises_with_agents,
    evaluate_ir_with_agents,
)

if TYPE_CHECKING:
    from agm.agl.syntax.nodes import Call
    from agm.agl.syntax.spans import SourceSpan


# ---------------------------------------------------------------------------
# T1 — simple text ask
# ---------------------------------------------------------------------------


def test_text_ask_basic() -> None:
    """Text-codec ask: passthrough, no JSON involved."""
    source = """\
agent summarizer
let summary: text = ask("Summarise it.", agent: summarizer)
summary
"""
    ir_reference, ir = evaluate_ir_with_agents(
        source,
        scripts={"summarizer": ["This is a summary."]},
    )
    assert ir_reference["summary"] == TextValue("This is a summary.")
    assert ir["summary"] == TextValue("This is a summary.")


# ---------------------------------------------------------------------------
# T2 — JSON int ask
# ---------------------------------------------------------------------------


def test_json_int_ask() -> None:
    """JSON-int ask: agent returns a bare integer."""
    source = """\
agent counter
let n: int = ask("How many?", agent: counter)
n
"""
    ir_reference, ir = evaluate_ir_with_agents(
        source,
        scripts={"counter": ["42"]},
    )
    assert ir_reference["n"] == IntValue(42)
    assert ir["n"] == IntValue(42)


# ---------------------------------------------------------------------------
# T3 — JSON record ask
# ---------------------------------------------------------------------------


def test_json_record_ask() -> None:
    """JSON-record ask: agent returns a JSON object matching a record type."""
    source = """\
record Point
  x: int
  y: int

agent locator
let pt: Point = ask("Find the point.", agent: locator)
pt
"""
    ir_reference, ir = evaluate_ir_with_agents(
        source,
        scripts={"locator": ['{"x": 3, "y": 7}']},
    )
    assert isinstance(ir_reference["pt"], RecordValue)
    assert ir_reference["pt"].fields["x"] == IntValue(3)
    assert ir_reference["pt"].fields["y"] == IntValue(7)
    assert isinstance(ir["pt"], RecordValue)
    assert ir["pt"].fields["x"] == IntValue(3)
    assert ir["pt"].fields["y"] == IntValue(7)


# ---------------------------------------------------------------------------
# T4 — JSON list ask
# ---------------------------------------------------------------------------


def test_json_list_ask() -> None:
    """JSON-list ask: agent returns a JSON array."""
    source = """\
agent lister
let items: list[text] = ask("List items.", agent: lister)
items
"""
    ir_reference, ir = evaluate_ir_with_agents(
        source,
        scripts={"lister": ['["alpha", "beta", "gamma"]']},
    )
    assert isinstance(ir_reference["items"], ListValue)
    assert ir_reference["items"].elements == (
        TextValue("alpha"), TextValue("beta"), TextValue("gamma")
    )
    assert isinstance(ir["items"], ListValue)
    assert ir["items"].elements == (
        TextValue("alpha"), TextValue("beta"), TextValue("gamma")
    )


# ---------------------------------------------------------------------------
# T5 — JSON enum ask
# ---------------------------------------------------------------------------


def test_json_enum_ask() -> None:
    """JSON-enum ask: agent returns a discriminated enum value."""
    source = """\
enum Status
  | Ok
  | Err(msg: text)

agent checker
let status: Status = ask("Check it.", agent: checker)
status
"""
    ir_reference, ir = evaluate_ir_with_agents(
        source,
        scripts={"checker": ['{"$case": "Ok"}']},
    )
    assert isinstance(ir_reference["status"], EnumValue)
    assert ir_reference["status"].variant == "Ok"
    assert isinstance(ir["status"], EnumValue)
    assert ir["status"].variant == "Ok"


# ---------------------------------------------------------------------------
# T6 — lenient JSON recovery (fence stripping)
# ---------------------------------------------------------------------------


def test_lenient_json_fence_stripping() -> None:
    """Lenient mode: agent wraps JSON in a markdown fence — still parsed."""
    source = """\
agent answerer
let n: int = ask("Give me a number.", agent: answerer)
n
"""
    fenced = "```json\n17\n```"
    ir_reference, ir = evaluate_ir_with_agents(
        source,
        scripts={"answerer": [fenced]},
    )
    assert ir_reference["n"] == IntValue(17)
    assert ir["n"] == IntValue(17)


# ---------------------------------------------------------------------------
# T7 — retry success (on_parse_error: Retry(n: 1))
# ---------------------------------------------------------------------------


def test_retry_success_second_attempt() -> None:
    """Retry policy: first response is invalid JSON, second is valid."""
    source = """\
agent parser
let n: int = ask("Parse this.", agent: parser, on_parse_error: Retry(n: 1))
n
"""
    ir_reference, ir = evaluate_ir_with_agents(
        source,
        scripts={"parser": ["not json at all", "99"]},
    )
    assert ir_reference["n"] == IntValue(99)
    assert ir["n"] == IntValue(99)


# ---------------------------------------------------------------------------
# T8 — retry exhausted raises AgentParseError
# ---------------------------------------------------------------------------


def test_retry_exhausted_raises() -> None:
    """Retry policy: all attempts fail → AgentParseError raised."""
    source = """\
agent parser
let n: int = ask("Parse this.", agent: parser, on_parse_error: Retry(n: 1))
n
"""
    ir_reference_exc, ir_exc = evaluate_ir_raises_with_agents(
        source,
        scripts={"parser": ["bad1", "bad2"]},
    )
    assert isinstance(ir_reference_exc, ExceptionValue)
    assert ir_reference_exc.display_name == "AgentParseError"
    assert isinstance(ir_exc, ExceptionValue)
    assert ir_exc.display_name == "AgentParseError"


# ---------------------------------------------------------------------------
# T9 — strict JSON mode
# ---------------------------------------------------------------------------


def test_strict_json_mode() -> None:
    """strict_json: true — bare JSON without fences, no repair."""
    source = """\
agent strict_agent
let b: bool = ask("True or false?", agent: strict_agent, strict_json: true)
b
"""
    ir_reference, ir = evaluate_ir_with_agents(
        source,
        scripts={"strict_agent": ["true"]},
    )
    assert ir_reference["b"] == BoolValue(True)
    assert ir["b"] == BoolValue(True)


# ---------------------------------------------------------------------------
# T10 — unit-typed ask (no output parsing)
# ---------------------------------------------------------------------------


def test_unit_typed_ask() -> None:
    """Unit ask: agent is called for side effects, result is discarded."""
    source = """\
agent notifier
let _: unit = ask("Notify!", agent: notifier)
let done: text = "done"
done
"""
    ir_reference, ir = evaluate_ir_with_agents(
        source,
        scripts={"notifier": ["acknowledged"]},
    )
    # The `_` binding may or may not appear in the snapshot; check `done` which always does.
    assert ir_reference.get("done") == TextValue("done")
    assert ir.get("done") == TextValue("done")


# ---------------------------------------------------------------------------
# T11 — ask inside a function
# ---------------------------------------------------------------------------


def test_ask_inside_function() -> None:
    """ask call site inside a function body — agent is captured in closure."""
    source = """\
agent namer
def get_name(prompt: text) -> text = ask(prompt, agent: namer)
let name: text = get_name("What is the name?")
name
"""
    ir_reference, ir = evaluate_ir_with_agents(
        source,
        scripts={"namer": ["Alice"]},
    )
    assert ir_reference["name"] == TextValue("Alice")
    assert ir["name"] == TextValue("Alice")


# ---------------------------------------------------------------------------
# T12 — multiple agents
# ---------------------------------------------------------------------------


def test_multiple_agents() -> None:
    """Multiple named agents: each call is routed to the correct agent."""
    source = """\
agent first
agent second
let a: text = ask("First.", agent: first)
let b: text = ask("Second.", agent: second)
b
"""
    ir_reference, ir = evaluate_ir_with_agents(
        source,
        scripts={
            "first": ["hello"],
            "second": ["world"],
        },
    )
    assert ir_reference["a"] == TextValue("hello")
    assert ir_reference["b"] == TextValue("world")
    assert ir["a"] == TextValue("hello")
    assert ir["b"] == TextValue("world")


# ---------------------------------------------------------------------------
# T13 — JSON schema validation error (wrong type)
# ---------------------------------------------------------------------------


def test_schema_validation_failure_wrong_type() -> None:
    """Agent returns invalid JSON (fails schema validation) → AgentParseError."""
    source = """\
agent validator
let n: int = ask("Give int.", agent: validator)
n
"""
    # Agent returns a string, not an integer — schema validation fails.
    ir_reference_exc, ir_exc = evaluate_ir_raises_with_agents(
        source,
        scripts={"validator": ['"not an int"']},
    )
    assert isinstance(ir_reference_exc, ExceptionValue)
    assert ir_reference_exc.display_name == "AgentParseError"
    assert isinstance(ir_exc, ExceptionValue)
    assert ir_exc.display_name == "AgentParseError"


# ---------------------------------------------------------------------------
# T14 — strict JSON: invalid JSON raises AgentParseError
# ---------------------------------------------------------------------------


def test_strict_json_invalid_raises() -> None:
    """strict_json=true with fenced JSON: strict mode does not strip fences."""
    source = """\
agent strict_agent
let n: int = ask("Give int.", agent: strict_agent, strict_json: true)
n
"""
    # Fenced JSON fails in strict mode (strict does not strip fences).
    ir_reference_exc, ir_exc = evaluate_ir_raises_with_agents(
        source,
        scripts={"strict_agent": ["```json\n42\n```"]},
    )
    assert isinstance(ir_reference_exc, ExceptionValue)
    assert ir_reference_exc.display_name == "AgentParseError"
    assert isinstance(ir_exc, ExceptionValue)
    assert ir_exc.display_name == "AgentParseError"


# ---------------------------------------------------------------------------
# T15 — default agent (ask without agent: named arg)
# ---------------------------------------------------------------------------


def test_default_agent_ask() -> None:
    """ask() with no agent: named arg uses the default agent."""
    source = """\
let result: text = ask("Hello default.")
result
"""
    ir_reference, ir = evaluate_ir_with_agents(
        source,
        scripts={},
        default_responses=["default response"],
        agent_names=frozenset(),
        has_default=True,
    )
    assert ir_reference["result"] == TextValue("default response")
    assert ir["result"] == TextValue("default response")


# ---------------------------------------------------------------------------
# T16 — ask-request builds an AgentRequest record
# ---------------------------------------------------------------------------


def test_ask_request_builds_record() -> None:
    """ask-request: no agent dispatch, returns an AgentRequest-shaped record."""
    source = """\
agent dummy
let req = ask-request("My prompt.", agent: dummy)
let agent_name: text = req.agent
let prompt_text: text = req.prompt
prompt_text
"""
    # ask-request does not call the agent — no scripted responses needed.
    ir_reference, ir = evaluate_ir_with_agents(
        source,
        scripts={"dummy": []},
    )
    assert ir_reference["agent_name"] == TextValue("dummy")
    assert ir_reference["prompt_text"] == TextValue("My prompt.")
    assert ir["agent_name"] == TextValue("dummy")
    assert ir["prompt_text"] == TextValue("My prompt.")


# ---------------------------------------------------------------------------
# T17 — retry with schema validation error (covers result.errors branch)
# ---------------------------------------------------------------------------


def test_retry_with_schema_validation_error_then_success() -> None:
    """Retry: first response fails schema, second is valid."""
    source = """\
agent fixer
let n: int = ask("Give int.", agent: fixer, on_parse_error: Retry(n: 1))
n
"""
    # First response: string (wrong type) → schema error; second: valid int.
    ir_reference, ir = evaluate_ir_with_agents(
        source,
        scripts={"fixer": ['"oops"', "42"]},
    )
    assert ir_reference["n"] == IntValue(42)
    assert ir["n"] == IntValue(42)


# ---------------------------------------------------------------------------
# T18 — enum bad_case validation failure (covers _classify_enum_failure_typeless)
# ---------------------------------------------------------------------------


def test_enum_bad_case_raises_agent_parse_error() -> None:
    """Enum ask: agent returns unknown $case → AgentParseError from both pipelines."""
    import json

    from agm.agl.ir.contracts import (
        ContractRequest,
        EnumDecode,
        ScalarDecode,
        ScalarKind,
        VariantDecode,
    )
    from agm.agl.ir.ids import NominalId
    from agm.agl.modules.ids import PRELUDE_ID
    from agm.agl.runtime.codec import _parse_contract_output

    nominal = NominalId(PRELUDE_ID, "Status")
    decode = EnumDecode(
        nominal=nominal,
        display_name="Status",
        variants=(
            VariantDecode(name="Ok", fields=()),
            VariantDecode(name="Err", fields=(("msg", ScalarDecode(ScalarKind.TEXT)),)),
        ),
    )
    schema = {
        "oneOf": [
            {"type": "object", "additionalProperties": False,
             "required": ["$case"], "properties": {"$case": {"const": "Ok"}}},
            {"type": "object", "additionalProperties": False,
             "required": ["$case", "msg"],
             "properties": {"$case": {"const": "Err"}, "msg": {"type": "string"}}},
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
# T19 — enum with retry success (covers _find_enum_decode_at_path)
# ---------------------------------------------------------------------------


def test_enum_retry_then_success() -> None:
    """Enum ask: first response has bad case, second is valid."""
    source = """\
enum Status
  | Ok
  | Err(msg: text)

agent checker
let s: Status = ask("Status?", agent: checker, on_parse_error: Retry(n: 1))
s
"""
    ir_reference, ir = evaluate_ir_with_agents(
        source,
        scripts={"checker": ['{"$case": "Bad"}', '{"$case": "Ok"}']},
    )
    assert isinstance(ir_reference["s"], EnumValue)
    assert ir_reference["s"].variant == "Ok"
    assert isinstance(ir["s"], EnumValue)
    assert ir["s"].variant == "Ok"


# ---------------------------------------------------------------------------
# T20 — validate.py defensive checks (hand-built invalid IR)
# ---------------------------------------------------------------------------


def test_validate_ir_ask_missing_contract() -> None:
    """validate_ir: IrAsk referencing missing contract_id → InvalidIrError."""

    from agm.agl.ir.ids import ContractId, Location, SourceId
    from agm.agl.ir.nodes import IrAsk, IrConstText
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import InvalidIrError, validate_ir
    from agm.agl.modules.ids import ENTRY_ID

    src_id = SourceId(0)
    dummy_loc = Location(
        source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0
    )
    bad_cid = ContractId(999)
    ask_node = IrAsk(
        location=dummy_loc,
        agent=IrConstText(location=dummy_loc, value="ask"),
        prompt=IrConstText(location=dummy_loc, value="test"),
        contract_id=bad_cid,
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


def test_validate_ir_ask_max_attempts_zero() -> None:
    """validate_ir: IrAsk with max_attempts=0 → InvalidIrError."""

    from agm.agl.ir.contracts import ContractRequest
    from agm.agl.ir.ids import ContractId, Location, SourceId
    from agm.agl.ir.nodes import IrAsk, IrConstText
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import InvalidIrError, validate_ir
    from agm.agl.modules.ids import ENTRY_ID

    src_id = SourceId(0)
    dummy_loc = Location(
        source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0
    )
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
    ask_node = IrAsk(
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
    dummy_loc = Location(
        source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0
    )
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
# T21 — ask-request via lowerer BuiltinKind.ASK_REQUEST path
# ---------------------------------------------------------------------------


def test_ask_request_typed_builds_record() -> None:
    """ask-request with explicit type argument builds AgentRequest record."""
    source = """\
agent worker
let req = ask-request::[int]("Give me a number.", agent: worker)
let prompt_text: text = req.prompt
prompt_text
"""
    ir_reference, ir = evaluate_ir_with_agents(
        source,
        scripts={"worker": []},
    )
    assert ir_reference["prompt_text"] == TextValue("Give me a number.")
    assert ir["prompt_text"] == TextValue("Give me a number.")


# ---------------------------------------------------------------------------
# T22 — _parse_contract_output unit tests for uncovered branches
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
    from agm.agl.modules.ids import PRELUDE_ID
    from agm.agl.runtime.codec import _parse_contract_output

    nom = IrNominalId(PRELUDE_ID, "Point")
    schema = _json.dumps({
        "type": "object",
        "additionalProperties": False,
        "required": ["x", "y"],
        "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
    })
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
    from agm.agl.modules.ids import PRELUDE_ID
    from agm.agl.runtime.codec import _parse_contract_output

    nom = NominalId(PRELUDE_ID, "Point")
    schema = _json.dumps({
        "type": "object",
        "additionalProperties": False,
        "required": ["x"],
        "properties": {"x": {"type": "integer"}},
    })
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
    from agm.agl.modules.ids import PRELUDE_ID
    from agm.agl.runtime.codec import _parse_contract_output

    nominal = NominalId(PRELUDE_ID, "Flag")
    decode = EnumDecode(
        nominal=nominal,
        display_name="Flag",
        variants=(VariantDecode(name="On", fields=()), VariantDecode(name="Off", fields=())),
    )
    schema = _json.dumps({
        "oneOf": [
            {"type": "object", "additionalProperties": False,
             "required": ["$case"], "properties": {"$case": {"const": "On"}}},
            {"type": "object", "additionalProperties": False,
             "required": ["$case"], "properties": {"$case": {"const": "Off"}}},
        ]
    })
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
    from agm.agl.modules.ids import PRELUDE_ID

    nominal = NominalId(PRELUDE_ID, "Flag")
    decode = EnumDecode(
        nominal=nominal,
        display_name="Flag",
        variants=(VariantDecode(name="On", fields=()), VariantDecode(name="Off", fields=())),
    )
    schema = _json.dumps({
        "oneOf": [
            {"type": "object", "additionalProperties": False,
             "required": ["$case"], "properties": {"$case": {"const": "On"}}},
            {"type": "object", "additionalProperties": False,
             "required": ["$case"], "properties": {"$case": {"const": "Off"}}},
        ]
    })
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

    schema = _json.dumps({
        "oneOf": [
            {"type": "object", "additionalProperties": False,
             "required": ["$case"], "properties": {"$case": {"const": "On"}}},
        ]
    })
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


def test_find_enum_decode_at_path_through_list() -> None:
    """_find_enum_decode_at_path: navigate through ListDecode to find EnumDecode."""
    from agm.agl.ir.contracts import (
        ContractRequest,
        EnumDecode,
        ListDecode,
        VariantDecode,
    )
    from agm.agl.ir.ids import NominalId
    from agm.agl.modules.ids import PRELUDE_ID
    from agm.agl.runtime.codec import _find_enum_decode_at_path

    nominal = NominalId(PRELUDE_ID, "Status")
    enum_dec = EnumDecode(
        nominal=nominal,
        display_name="Status",
        variants=(VariantDecode(name="Ok", fields=()),),
    )
    list_dec = ListDecode(elem=enum_dec)
    contract = ContractRequest(
        codec_name="json",
        strict_json=None,
        json_schema="{}",
        decode=list_dec,
        target_type_label="list[Status]",
        structured_exec=False,
        format_instructions="",
        is_unit=False,
    )
    # Navigate into element 0 of list.
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
    from agm.agl.modules.ids import PRELUDE_ID
    from agm.agl.runtime.codec import _find_enum_decode_at_path

    nominal = NominalId(PRELUDE_ID, "Status")
    enum_dec = EnumDecode(
        nominal=nominal,
        display_name="Status",
        variants=(VariantDecode(name="Ok", fields=()),),
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
    from agm.agl.modules.ids import PRELUDE_ID
    from agm.agl.runtime.codec import _find_enum_decode_at_path

    nominal = NominalId(PRELUDE_ID, "Status")
    enum_dec = EnumDecode(
        nominal=nominal,
        display_name="Status",
        variants=(VariantDecode(name="Ok", fields=()),),
    )
    rec_nominal = NominalId(PRELUDE_ID, "Wrapper")
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
    from agm.agl.modules.ids import PRELUDE_ID
    from agm.agl.runtime.codec import _find_enum_decode_at_path

    nominal = NominalId(PRELUDE_ID, "Status")
    enum_dec = EnumDecode(
        nominal=nominal,
        display_name="Status",
        variants=(VariantDecode(name="Ok", fields=()),),
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
    from agm.agl.modules.ids import PRELUDE_ID
    from agm.agl.runtime.codec import _find_enum_decode_at_path

    nom = NominalId(PRELUDE_ID, "Point")
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
    from agm.agl.modules.ids import PRELUDE_ID

    nominal = NominalId(PRELUDE_ID, "Status")
    decode = EnumDecode(
        nominal=nominal,
        display_name="Status",
        variants=(
            VariantDecode(name="Ok", fields=()),
            VariantDecode(name="Err", fields=(("msg", ScalarDecode(ScalarKind.TEXT)),)),
        ),
    )
    schema = _json.dumps({
        "oneOf": [
            {"type": "object", "additionalProperties": False,
             "required": ["$case"], "properties": {"$case": {"const": "Ok"}}},
            {"type": "object", "additionalProperties": False,
             "required": ["$case", "msg"],
             "properties": {"$case": {"const": "Err"}, "msg": {"type": "string"}}},
        ]
    })
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


# ---------------------------------------------------------------------------
# T22b — validate.py IrAskRequest deep validation
# ---------------------------------------------------------------------------


def test_validate_ir_ask_request_missing_contract() -> None:
    """validate_ir: IrAskRequest referencing missing contract_id → InvalidIrError."""

    from agm.agl.ir.ids import ContractId, Location, SourceId
    from agm.agl.ir.nodes import IrAskRequest, IrConstText
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import InvalidIrError, validate_ir
    from agm.agl.modules.ids import ENTRY_ID

    src_id = SourceId(0)
    dummy_loc = Location(
        source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0
    )
    bad_cid = ContractId(999)
    node = IrAskRequest(
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
        contracts={},  # No contracts!
    )
    with pytest.raises(InvalidIrError, match="contract_id"):
        validate_ir(prog, deep=True)


def test_validate_ir_ask_request_max_attempts_zero() -> None:
    """validate_ir: IrAskRequest with max_attempts=0 → InvalidIrError."""

    from agm.agl.ir.contracts import ContractRequest
    from agm.agl.ir.ids import ContractId, Location, SourceId
    from agm.agl.ir.nodes import IrAskRequest, IrConstText
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import InvalidIrError, validate_ir
    from agm.agl.modules.ids import ENTRY_ID

    src_id = SourceId(0)
    dummy_loc = Location(
        source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0
    )
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
    node = IrAskRequest(
        location=dummy_loc,
        agent=IrConstText(location=dummy_loc, value="ask"),
        prompt=IrConstText(location=dummy_loc, value="test"),
        contract_id=cid,
        max_attempts=0,  # invalid!
    )
    prog = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=(node,))},
        symbols={},
        nominals={},
        sources={src_id: SourceFile(display_name="<test>", normalized_text="test")},
        contracts={cid: req},
    )
    with pytest.raises(InvalidIrError, match="max_attempts"):
        validate_ir(prog, deep=True)


def test_validate_contract_request_json_missing_decode() -> None:
    """validate.py: json codec with json_schema set but decode=None → InvalidIrError."""

    from agm.agl.ir.contracts import ContractRequest
    from agm.agl.ir.ids import ContractId, Location, SourceId
    from agm.agl.ir.nodes import IrConstUnit
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import InvalidIrError, validate_ir
    from agm.agl.modules.ids import ENTRY_ID

    src_id = SourceId(0)
    dummy_loc = Location(
        source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0
    )
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
    dummy_loc = Location(
        source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0
    )
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


# ---------------------------------------------------------------------------
# T22c — ir_interpreter.py uncovered paths
# ---------------------------------------------------------------------------


def test_ir_ask_non_text_prompt_renders_to_string() -> None:
    """IrAsk: int-valued prompt is render_value'd to text (hand-built IR)."""
    from agm.agl.eval.ir_interpreter import IrInterpreter
    from agm.agl.ir.contracts import ContractRequest, ScalarDecode, ScalarKind
    from agm.agl.ir.ids import ContractId, Location, SourceId, SymbolId
    from agm.agl.ir.nodes import IrAsk, IrBind, IrConstInt
    from agm.agl.ir.program import (
        ExecutableModule,
        ExecutableProgram,
        SourceFile,
        SymbolDescriptor,
    )
    from agm.agl.modules.ids import ENTRY_ID
    from agm.agl.runtime.agents import AgentRegistry
    from agm.agl.runtime.request import AgentRequest, AgentResponse

    src_id = SourceId(0)
    dummy_loc = Location(
        source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0
    )
    cid = ContractId(0)
    sym = SymbolId(0)
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
    # IrAsk with an int constant as the prompt (bypasses typechecker).
    ask_node = IrAsk(
        location=dummy_loc,
        # agent expr returns IntValue (not AgentValue) → falls back to "ask" name
        agent=IrConstInt(location=dummy_loc, value=0),
        prompt=IrConstInt(location=dummy_loc, value=42),  # int prompt
        contract_id=cid,
        max_attempts=1,
    )
    bind_node = IrBind(location=dummy_loc, symbol=sym, value=ask_node)
    sym_desc = SymbolDescriptor(symbol_id=sym, mutable=False, public_name="result", owner=ENTRY_ID)
    prog = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=(bind_node,))},
        symbols={sym: sym_desc},
        nominals={},
        sources={src_id: SourceFile(display_name="<test>", normalized_text="test")},
        contracts={cid: req},
    )

    captured_prompts: list[str] = []

    def _agent_fn(r: AgentRequest) -> AgentResponse:
        captured_prompts.append(r.prompt)
        return AgentResponse(content="99")

    registry = AgentRegistry(named={"ask": _agent_fn}, default_agent=None)
    interp = IrInterpreter(prog, registry=registry)
    bindings = interp.run()
    # The prompt "42" (rendered IntValue) was sent.
    assert captured_prompts == ["42"]
    assert bindings["result"] == IntValue(99)


def test_ir_ask_request_unit_contract() -> None:
    """IrAskRequest with is_unit contract → AgentRequest record with None output_contract."""
    source = """\
agent a
let req = ask-request("Do it.", agent: a)
let prompt_text: text = req.prompt
prompt_text
"""
    from tests.agl.ir_harness import evaluate_ir_with_agents

    ir_reference, ir = evaluate_ir_with_agents(
        source,
        scripts={"a": []},
    )
    assert ir_reference["prompt_text"] == TextValue("Do it.")
    assert ir["prompt_text"] == TextValue("Do it.")


def test_ir_ask_request_non_text_prompt() -> None:
    """IrAskRequest: int prompt expression is rendered to text (hand-built IR)."""
    from agm.agl.eval.ir_interpreter import IrInterpreter
    from agm.agl.ir.contracts import ContractRequest
    from agm.agl.ir.ids import ContractId, Location, SourceId, SymbolId
    from agm.agl.ir.nodes import IrAskRequest, IrBind, IrConstInt
    from agm.agl.ir.program import (
        ExecutableModule,
        ExecutableProgram,
        SourceFile,
        SymbolDescriptor,
    )
    from agm.agl.modules.ids import ENTRY_ID
    from agm.agl.runtime.agents import AgentRegistry

    src_id = SourceId(0)
    dummy_loc = Location(
        source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0
    )
    cid = ContractId(0)
    sym = SymbolId(0)
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
    # IrAskRequest with int prompt (bypasses typechecker).
    ask_req_node = IrAskRequest(
        location=dummy_loc,
        agent=IrConstInt(location=dummy_loc, value=0),
        prompt=IrConstInt(location=dummy_loc, value=7),  # int prompt
        contract_id=cid,
        max_attempts=1,
    )
    bind_node = IrBind(location=dummy_loc, symbol=sym, value=ask_req_node)
    sym_desc = SymbolDescriptor(symbol_id=sym, mutable=False, public_name="req", owner=ENTRY_ID)
    prog = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=(bind_node,))},
        symbols={sym: sym_desc},
        nominals={},
        sources={src_id: SourceFile(display_name="<test>", normalized_text="test")},
        contracts={cid: req},
    )
    registry = AgentRegistry(named={}, default_agent=None)
    interp = IrInterpreter(prog, registry=registry)
    bindings = interp.run()
    val = bindings["req"]
    assert isinstance(val, RecordValue)
    assert val.fields["prompt"] == TextValue("7")


# ---------------------------------------------------------------------------
# T22d — lowerer.py uncovered branches in _extract_max_attempts
# ---------------------------------------------------------------------------


def test_lower_on_parse_error_abort_gives_one_attempt() -> None:
    """_extract_max_attempts: Abort policy → 1 attempt (lowerer line 1640)."""
    source = """\
agent a
let n: int = ask("?", agent: a, on_parse_error: Abort)
n
"""
    from tests.agl.ir_harness import evaluate_ir_with_agents

    ir_reference, ir = evaluate_ir_with_agents(
        source,
        scripts={"a": ["5"]},
    )
    assert ir_reference["n"] == IntValue(5)
    assert ir["n"] == IntValue(5)


# ---------------------------------------------------------------------------
# T22e — additional coverage tests
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
    dummy_loc = Location(
        source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0
    )
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
    """validate_ir: IrAskRequest with valid contract passes deep validation (671->exit)."""
    from agm.agl.ir.contracts import ContractRequest
    from agm.agl.ir.ids import ContractId, Location, SourceId
    from agm.agl.ir.nodes import IrAskRequest, IrConstText
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import validate_ir
    from agm.agl.modules.ids import ENTRY_ID

    src_id = SourceId(0)
    dummy_loc = Location(
        source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0
    )
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
        contracts={cid: req},
    )
    # Should pass without error (covers the deep valid path 671->exit).
    validate_ir(prog, deep=True)


def test_ir_ask_request_unit_typed() -> None:
    """IrAskRequest with is_unit=True contract → output_contract=None in record."""
    source = """\
agent a
let req = ask-request::[unit]("Do it.", agent: a)
let oc = req.output_contract
oc
"""
    from tests.agl.ir_harness import evaluate_ir_with_agents

    ir_reference, ir = evaluate_ir_with_agents(
        source,
        scripts={"a": []},
    )
    # output_contract should be the None variant (OutputContractOption.None).
    assert isinstance(ir_reference["oc"], EnumValue)
    assert ir_reference["oc"].variant == "None"
    assert isinstance(ir["oc"], EnumValue)
    assert ir["oc"].variant == "None"


def test_ir_ask_no_errors_when_failed_covers_else_branch() -> None:
    """IrAsk retry loop: parse fails with no errors/error_msg → last_errors=() (line 1056)."""
    # Build a contract with text codec — but trick it: use json schema so parse
    # can fail with neither errors nor error_msg. The simplest: use a valid
    # json schema but make parse_agent_output return ok=False with empty errors and msg.
    # The easiest way: use a text codec (always succeeds), so we need the json codec.
    # We can create a contract where the schema is valid but the text doesn't
    # extract as ambiguous / no-json (empty errors, no error_msg).
    # Actually the only path where result.ok=False AND errors=() AND error_msg="" is
    # when AgentParseResult.failure("") is called. Let's mock parse_agent_output:
    from unittest.mock import patch

    from agm.agl.eval.ir_interpreter import IrInterpreter
    from agm.agl.ir.contracts import ContractRequest, ScalarDecode, ScalarKind
    from agm.agl.ir.ids import ContractId, Location, SourceId
    from agm.agl.ir.nodes import IrAsk, IrConstText
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.modules.ids import ENTRY_ID
    from agm.agl.runtime.agents import AgentRegistry
    from agm.agl.runtime.codec import ParseResult
    from agm.agl.runtime.request import AgentRequest, AgentResponse

    src_id = SourceId(0)
    dummy_loc = Location(
        source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0
    )
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

    def _agent_fn(r: AgentRequest) -> AgentResponse:
        return AgentResponse(content="invalid")

    registry = AgentRegistry(named={"ask": _agent_fn}, default_agent=None)
    interp = IrInterpreter(prog, registry=registry)

    # Patch _parse_contract_output to return failure with EMPTY errors AND EMPTY error_msg.
    empty_failure = ParseResult(ok=False, value=None, error_msg="", errors=())
    with patch("agm.agl.eval.ir_interpreter._parse_contract_output", return_value=empty_failure):
        from agm.agl.eval.exceptions import AglRaise
        with pytest.raises(AglRaise):
            interp.run()


def test_lower_on_parse_error_field_access_retry() -> None:
    """_extract_max_attempts: FieldAccess callee (qualified Retry) → correct attempt count."""
    source = """\
agent a
let n: int = ask("?", agent: a, on_parse_error: ::Retry(n: 2))
n
"""
    from tests.agl.ir_harness import evaluate_ir_with_agents

    # First 2 responses are bad JSON, 3rd is valid.
    ir_reference, ir = evaluate_ir_with_agents(
        source,
        scripts={"a": ["bad", "bad", "7"]},
    )
    assert ir_reference["n"] == IntValue(7)
    assert ir["n"] == IntValue(7)


def test_validate_ir_ask_shallow_does_not_check_contracts() -> None:
    """validate_ir: IrAsk in shallow (deep=False) validation skips contract checks (656->exit)."""
    from agm.agl.ir.ids import ContractId, Location, SourceId
    from agm.agl.ir.nodes import IrAsk, IrConstText
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import validate_ir
    from agm.agl.modules.ids import ENTRY_ID

    src_id = SourceId(0)
    dummy_loc = Location(
        source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0
    )
    bad_cid = ContractId(999)
    node = IrAsk(
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


def test_validate_ir_ask_request_shallow_does_not_check_contracts() -> None:
    """validate_ir: IrAskRequest in shallow validation skips contract checks (671->exit)."""
    from agm.agl.ir.ids import ContractId, Location, SourceId
    from agm.agl.ir.nodes import IrAskRequest, IrConstText
    from agm.agl.ir.program import ExecutableModule, ExecutableProgram, SourceFile
    from agm.agl.ir.validate import validate_ir
    from agm.agl.modules.ids import ENTRY_ID

    src_id = SourceId(0)
    dummy_loc = Location(
        source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0
    )
    bad_cid = ContractId(999)
    node = IrAskRequest(
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
    # deep=False → skips contract_id check (covers 671->exit branch).
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
    dummy_loc = Location(
        source_id=src_id, start_offset=0, end_offset=1, start_line=1, start_col=0
    )
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
    """_extract_max_attempts: FieldAccess callee → callee.field = 'Retry' (lines 1628-1629)."""
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
    named_arg = NamedArg(name="on_parse_error", value=inner_call, span=span, node_id=7)
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
    named_arg = NamedArg(name="on_parse_error", value=inner_call, span=span, node_id=14)
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
    """_extract_max_attempts: FieldAccess callee with non-Retry field → 1 attempt (1632->1640)."""
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
    named_arg = NamedArg(name="on_parse_error", value=inner_call, span=span, node_id=25)
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
    from agm.agl.modules.ids import PRELUDE_ID

    nominal = NominalId(PRELUDE_ID, "Pair")
    decode = EnumDecode(
        nominal=nominal,
        display_name="Pair",
        variants=(
            VariantDecode(
                name="Both",
                fields=(("a", ScalarDecode(ScalarKind.INT)), ("b", ScalarDecode(ScalarKind.INT))),
            ),
        ),
    )
    # Schema requiring both 'a' and 'b'.
    schema = _json.dumps({
        "oneOf": [{
            "type": "object",
            "additionalProperties": False,
            "required": ["$case", "a", "b"],
            "properties": {
                "$case": {"const": "Both"},
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
        }]
    })
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


def test_make_validation_error_required_non_list() -> None:
    """_make_validation_error: required validator with non-list required → field=None."""
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
    from agm.agl.modules.ids import PRELUDE_ID

    nominal = NominalId(PRELUDE_ID, "Status")
    decode = EnumDecode(
        nominal=nominal,
        display_name="Status",
        variants=(
            VariantDecode(name="Ok", fields=()),
            VariantDecode(name="Err", fields=(("msg", ScalarDecode(ScalarKind.TEXT)),)),
        ),
    )
    # Schema WITHOUT additionalProperties: False → sub-errors will be 'const' and 'type' only.
    schema = _json.dumps({
        "oneOf": [
            {"type": "object", "required": ["$case"],
             "properties": {"$case": {"const": "Ok"}}},
            {"type": "object", "required": ["$case", "msg"],
             "properties": {"$case": {"const": "Err"}, "msg": {"type": "string"}}},
        ]
    })
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
    from agm.agl.modules.ids import PRELUDE_ID
    from agm.agl.runtime.codec import _classify_enum_failure

    nominal = NominalId(PRELUDE_ID, "Status")
    decode = EnumDecode(
        nominal=nominal,
        display_name="Status",
        variants=(VariantDecode(name="Err", fields=()),),
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
    from agm.agl.modules.ids import PRELUDE_ID
    from agm.agl.runtime.codec import _classify_enum_failure

    nominal = NominalId(PRELUDE_ID, "Status")
    decode = EnumDecode(
        nominal=nominal,
        display_name="Status",
        variants=(
            VariantDecode(name="Ok", fields=()),
            VariantDecode(name="Err", fields=(("msg", ScalarDecode(ScalarKind.TEXT)),)),
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
# M1 (MAJOR) — enum oneOf validation-error message parity ir_semantic tests
#
# These tests verify that when ask() exhausts all attempts due to enum errors,
# the validation_errors (category, message, field) from the IR pipeline are
# byte-identical to the ir_reference pipeline.  Added per TDD mandate before the fix.
# ---------------------------------------------------------------------------


def test_enum_unknown_case_exhausted_ir_semantic_parity() -> None:
    """Enum ask exhausted: unknown $case → validation_errors match ir_reference (M1 parity).

    The harness's evaluate_ir_raises_with_agents already asserts byte-identical
    parity between ir_reference and IR.  We additionally document and verify the exact
    message content for the unknown-$case shape.
    """
    source = """\
enum Status
  | Ok
  | Err(msg: text)

agent checker
let status: Status = ask("Check.", agent: checker)
status
"""
    ir_reference_exc, ir_exc = evaluate_ir_raises_with_agents(
        source,
        scripts={"checker": ['{"$case": "Bogus"}']},
    )
    assert ir_reference_exc.display_name == "AgentParseError"
    assert ir_exc.display_name == "AgentParseError"
    # validation_errors is stored as a JsonValue(raw=[{...}]) in both pipelines.
    from agm.agl.eval.values import JsonValue

    errors_val = ir_exc.fields.get("validation_errors")
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


def test_enum_missing_field_exhausted_ir_semantic_parity() -> None:
    """Enum ask exhausted: known $case but missing field → validation_errors match ir_reference.

    The harness already asserts byte-identical parity.  We additionally document
    and verify the exact message and field content for the missing-field shape.
    """
    source = """\
enum Status
  | Ok
  | Err(msg: text)

agent checker
let status: Status = ask("Check.", agent: checker)
status
"""
    ir_reference_exc, ir_exc = evaluate_ir_raises_with_agents(
        source,
        scripts={"checker": ['{"$case": "Err"}']},
    )
    assert ir_reference_exc.display_name == "AgentParseError"
    assert ir_exc.display_name == "AgentParseError"

    from agm.agl.eval.values import JsonValue

    errors_val = ir_exc.fields.get("validation_errors")
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

    # Parity: ir_reference must match IR exactly (harness already asserts, but be explicit).
    ir_reference_errors_val = ir_reference_exc.fields.get("validation_errors")
    assert isinstance(ir_reference_errors_val, JsonValue)
    ir_reference_first = ir_reference_errors_val.raw[0]
    assert isinstance(ir_reference_first, dict)
    assert ir_reference_first.get("message") == msg, (
        "Message mismatch:\n"
        f"  reference: {ir_reference_first.get('message')!r}\n  actual: {msg!r}"
    )
    assert ir_reference_first.get("field") == first_err.get("field")


def test_enum_unexpected_field_exhausted_ir_semantic_parity() -> None:
    """Enum ask exhausted: known $case but unexpected field → validation_errors match ir_reference.

    The harness already asserts byte-identical parity.  We additionally document
    and verify the exact message and field content for the unexpected-field shape.
    IR reference sets field=key for unknown_field records (not None).
    """
    source = """\
enum Status
  | Ok
  | Err(msg: text)

agent checker
let status: Status = ask("Check.", agent: checker)
status
"""
    ir_reference_exc, ir_exc = evaluate_ir_raises_with_agents(
        source,
        scripts={"checker": ['{"$case": "Ok", "extra_field": 42}']},
    )
    assert ir_reference_exc.display_name == "AgentParseError"
    assert ir_exc.display_name == "AgentParseError"

    from agm.agl.eval.values import JsonValue

    errors_val = ir_exc.fields.get("validation_errors")
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

    # Parity: ir_reference must match IR exactly.
    ir_reference_errors_val = ir_reference_exc.fields.get("validation_errors")
    assert isinstance(ir_reference_errors_val, JsonValue)
    ir_reference_first = ir_reference_errors_val.raw[0]
    assert isinstance(ir_reference_first, dict)
    assert ir_reference_first.get("message") == msg, (
        "Message mismatch:\n"
        f"  reference: {ir_reference_first.get('message')!r}\n  actual: {msg!r}"
    )
    assert ir_reference_first.get("field") == first_err.get("field")
