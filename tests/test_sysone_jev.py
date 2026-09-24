"""The ``sysone`` package: manifest, the ``jev`` module surface, its test transport, and
the companion's requests, settings, pooled clients, error mapping, and trace records."""

from __future__ import annotations

import json
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import pytest
import typesafe_sdk

from agm.agl.pipeline import PipelineDriver, RunResult
from agm.agl.runtime.arguments import OptionSome
from agm.agl.runtime.externs import ExternRegistry
from agm.cli_support.program_options import (
    EXEC_RESERVED_FLAGS,
    ProgramCommand,
    build_program_command,
)
from agm.core.parse import parse_timeout
from tests._agl_helpers import prepare_inline_command, run_inline_command
from tests._jev_helpers import (
    JEV_MODULE,
    SYSONE_ROOT,
    TEST_API_KEY,
    JevMount,
    JevTransport,
    install_jev_transport,
    jev_roots,
    load_jev_companion,
    mount_jev,
)
from tests._package_helpers import package_info

_NOUL_BODY = {
    "state": "I was charged twice.",
    "model": "jev-latest",
    "questions": {"billing": {"type": "noul", "instructions": "Is this about billing?"}},
}
_NOUL_EXCHANGE = {
    "expect": {"body": _NOUL_BODY},
    "json": {
        "model": "jev-1",
        "answers": {"billing": {"type": "noul", "noul": 0.9}},
        "usage": {"input_tokens": 12, "output_tokens": 1},
    },
    "headers": {"x-typesafe-request-id": "req-1"},
}


def test_manifest_declares_the_sdk_as_a_python_requirement() -> None:
    manifest = package_info(SYSONE_ROOT).manifest

    assert manifest.name == "sysone"
    assert manifest.python_dependencies == ("typesafe-sdk>=0.7,<1",)


def test_jev_surface_type_checks_through_an_import() -> None:
    source = """\
import sysone/jev

enum Team = Billing | Technical
enum Level = Low | High

record Triage
  @doc("Is it urgent?")
  urgent: bool
  team: jev::Choice[Team]
  level: jev::Score[Level]

jev::model := Some("jev-latest")
jev::noul-threshold := 0.7
let criteria = jev::NoulCriteria("spam" as json, "not spam" as json)
let questions: dict[text, jev::Question] = {
  "spam": jev::Question::NoulQuestion("Is it spam?" as json, Some(criteria)),
  "tone": jev::Question::ChoiceQuestion("Tone?" as json, {"calm": null}),
  "urgency": jev::Question::ScoreQuestion(null, ["low" as json, "high" as json]),
}
let noul = jev::Noul(0.8)
let default-value: bool = noul.value()
let explicit-value: bool = noul.value(0.9)
let billing: Team = Team::Billing
let high: Level = Level::High
let choice: jev::Choice[Team] = jev::Choice(billing, 0.9, {"Billing": 0.9})
let score: jev::Score[Level] = jev::Score(high, 0.9, 0.8, [0.1, 0.9])
let answer: jev::Answer = jev::Answer::NoulAnswer(0.5)
let response = jev::Response("jev-1", {"a": answer}, jev::Usage(None, Some(1)), None)
let auth = jev::JevAuthError(request-id = None, status = 401, body = null, message = "m")
let limited = jev::JevRateLimitError(request-id = None, status = 429, body = null,
  retry-after-ms = Some(10), message = "m")
let timed-out = jev::JevTimeoutError(request-id = None, timeout = "10s", message = "m")
let untargeted = jev::JevTargetError(request-id = None, target = "text", message = "m")

def probe(state: json) -> unit =
  let raw: jev::Response = jev::system-one(state, questions)
  let raw-result: Result[jev::Response, jev::JevError] = jev::try-system-one(state, questions)
  let noul: jev::Noul = jev::ask-noul("Spam?", state, Some(criteria))
  let noul-result: Result[jev::Noul, jev::JevError] = jev::try-ask-noul("Spam?", state)
  let team: jev::Choice[Team] = jev::ask-choice("Team?", state, timeout = Some("5s"))
  let level = jev::ask-score::[Level]("Level?", state, model = Some("jev-1"))
  let urgent: bool = jev::ask("Urgent?", state)
  let bare-team = jev::ask::[Team]("Team?", state)
  let triage: Triage = jev::ask-many(state)
  let keyed = jev::system-one(state, questions, api-key = Some("k"), base-url = Some("u"),
    max-retries = 0)
  let keyed-result: Result[jev::Response, jev::JevError] = jev::try-system-one(state,
    questions, api-key = Some("k"), base-url = None, max-retries = 1)
  let keyed-noul = jev::try-ask-noul("Spam?", state, max-retries = 0)
  let strict: bool = jev::ask("Urgent?", state, noul-threshold = 0.9, api-key = Some("k"))
  let strict-triage: Triage = jev::ask-many(state, base-url = Some("u"), noul-threshold = 0.6)
  let keyed-choice = jev::ask-choice::[Team]("Team?", state, max-retries = 3)
  let keyed-score = jev::ask-score::[Level]("Level?", state, api-key = None)
  ()
"""
    result = PipelineDriver(get_sandbox_context=None).check_prepared(
        prepare_inline_command(source, roots=jev_roots())
    )

    assert result.ok, result.diagnostics


def test_registry_returns_the_loaded_jev_companion() -> None:
    registry = ExternRegistry()
    companion = load_jev_companion(
        PipelineDriver(extern_registry=registry, get_sandbox_context=None), registry
    )

    assert callable(companion.open_client)
    assert callable(companion.system_one)


def test_scripted_transport_answers_through_the_swapped_client_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, transport = _scripted_client(monkeypatch, [_NOUL_EXCHANGE])

    response = _ask_billing(client)

    assert response.nouls["billing"].noul == 0.9
    assert response.raw_http_response.headers["x-typesafe-request-id"] == "req-1"
    assert transport.requests[0].headers["authorization"] == f"Bearer {TEST_API_KEY}"
    transport.assert_complete()


def test_scripted_transport_reports_unsent_outcomes(monkeypatch: pytest.MonkeyPatch) -> None:
    mount = mount_jev(monkeypatch, [_NOUL_EXCHANGE])

    with pytest.raises(AssertionError):
        mount.transport.assert_complete()


def _scripted_client(
    monkeypatch: pytest.MonkeyPatch, outcomes: list[dict[str, object]]
) -> tuple[typesafe_sdk.TypeSafeClient, JevTransport]:
    _, transport, companion = mount_jev(monkeypatch, outcomes)
    settings = companion.Settings(api_key=None, base_url=None, max_retries=0)
    return companion.open_client(settings), transport


def _ask_billing(
    client: typesafe_sdk.TypeSafeClient, timeout: float | None = None
) -> typesafe_sdk.SystemOneResponse:
    return client.system_one(
        state="I was charged twice.",
        questions={"billing": typesafe_sdk.Noul(instructions="Is this about billing?")},
        timeout=timeout,
    )


def test_scripted_expectation_checks_url_headers_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = {
        **_NOUL_EXCHANGE,
        "expect": {
            "body": _NOUL_BODY,
            "url": "https://api.typesafe.ai/v1/systemone",
            "headers": {"Authorization": f"Bearer {TEST_API_KEY}", "x-absent": None},
            "timeout": 5.0,
        },
    }
    client, transport = _scripted_client(monkeypatch, [exchange])

    _ask_billing(client, timeout=5.0)

    transport.assert_complete()


@pytest.mark.parametrize(
    "expect",
    (
        {"body": {**_NOUL_BODY, "model": "other"}},
        {"url": "https://elsewhere.invalid/v1/systemone"},
        {"headers": {"authorization": "Bearer other"}},
        {"timeout": 1.0},
    ),
)
def test_scripted_expectation_mismatch_fails_the_test(
    monkeypatch: pytest.MonkeyPatch, expect: dict[str, object]
) -> None:
    client, _ = _scripted_client(monkeypatch, [{**_NOUL_EXCHANGE, "expect": expect}])

    with pytest.raises(pytest.fail.Exception):
        _ask_billing(client, timeout=5.0)


def test_unscripted_request_fails_the_test(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _scripted_client(monkeypatch, [])

    with pytest.raises(pytest.fail.Exception):
        _ask_billing(client)


@pytest.mark.parametrize(
    ("failure", "error"),
    (
        ("connection", typesafe_sdk.TypeSafeAPIConnectionError),
        ("timeout", typesafe_sdk.TypeSafeAPITimeoutError),
    ),
)
def test_scripted_transport_failure_reaches_the_sdk(
    monkeypatch: pytest.MonkeyPatch, failure: str, error: type[Exception]
) -> None:
    client, transport = _scripted_client(monkeypatch, [{"fail": failure}])

    with pytest.raises(error):
        _ask_billing(client)
    transport.assert_complete()


def test_scripted_timeout_reports_the_request_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _scripted_client(monkeypatch, [{"fail": "timeout"}])

    with pytest.raises(typesafe_sdk.TypeSafeAPITimeoutError) as raised:
        _ask_billing(client, timeout=7.0)
    assert raised.value.timeout == 7.0


def test_transport_installs_into_a_callers_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ExternRegistry()
    driver = PipelineDriver(extern_registry=registry, get_sandbox_context=None)

    transport = install_jev_transport(monkeypatch, driver, registry, [_NOUL_EXCHANGE])
    companion = registry.loaded_companion(JEV_MODULE)
    assert companion is not None
    client = companion.open_client(companion.Settings(api_key=None, base_url=None, max_retries=0))

    assert _ask_billing(client).nouls["billing"].noul == 0.9
    transport.assert_complete()


# ---------------------------------------------------------------------------
# Behavior: requests through the companion
# ---------------------------------------------------------------------------

_PROGRAM_HEADER = """\
import sysone/jev
import sysone/jev::{JevError, JevApiError, JevAuthError, JevRequestError, JevRateLimitError}
import sysone/jev::{JevServerError, JevConnectionError, JevTimeoutError, JevResponseError}
import sysone/jev::{JevTargetError}
"""
_NO_RETRIES = {"sysone/jev::max-retries": 0}
_BILLING_QUESTION = {"billing": {"type": "noul", "instructions": "Is this about billing?"}}
_BILLING_CALL = """\
let billing: dict[text, jev::Question] = {
  "billing": jev::Question::NoulQuestion("Is this about billing?" as json, None),
}
"""


_STATE = "I was charged twice."


def _body(
    questions: dict[str, object] = _BILLING_QUESTION,
    state: object = _STATE,
    model: str = "jev-latest",
) -> dict[str, object]:
    return {"state": state, "model": model, "questions": questions}


def _noul(noul: float) -> dict[str, object]:
    return {"type": "noul", "noul": noul}


def _answered(
    answers: dict[str, object],
    questions: dict[str, object] | None = None,
    *,
    state: object = _STATE,
    request_id: str = "req-1",
) -> dict[str, object]:
    """One scripted exchange answering *answers*; given *questions*, it expects them about
    *state*."""
    exchange: dict[str, object] = {
        "json": {
            "model": "jev-1",
            "answers": answers,
            "usage": {"input_tokens": 12, "output_tokens": 1},
        },
        "headers": {"x-typesafe-request-id": request_id},
    }
    if questions is not None:
        exchange["expect"] = {"body": _body(questions, state=state)}
    return exchange


_BILLING_ANSWER = {"billing": _noul(0.9)}


def _program(body: str, declarations: str = "") -> str:
    """A program whose ``main`` runs *body* after module-level *declarations*."""
    main = "program def main() -> unit =\n" + textwrap.indent(body, "  ")
    return "\n".join((_PROGRAM_HEADER, declarations, main))


def _run_jev(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: Sequence[Mapping[str, object]],
    body: str,
    *,
    module_params: Mapping[str, object] | None = None,
    declarations: str = "",
    **run_kwargs: object,
) -> tuple[RunResult, JevMount]:
    """Run *body* as ``main``'s body after *declarations* against scripted *outcomes*,
    retries off by default."""
    mount = mount_jev(monkeypatch, outcomes)
    result = run_inline_command(
        mount.driver,
        _program(body, declarations),
        roots=jev_roots(),
        module_params={**_NO_RETRIES, **(module_params or {})},
        **run_kwargs,
    )
    return result, mount


_NONE = {"$case": "None"}


def _some(value: object) -> dict[str, object]:
    """A raised exception's ``Option`` field holding *value*, as ``RunError.fields`` shows it."""
    return {"$case": "Some", "value": value}


def _printed(capsys: pytest.CaptureFixture[str]) -> list[str]:
    return capsys.readouterr().out.splitlines()


def test_system_one_sends_every_question_kind_and_maps_every_answer_kind(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    questions = {
        "spam": {
            "type": "noul",
            "instructions": {"task": "Is it spam?", "weight": 0.5},
            "criteria": {"true": "Unsolicited", "false": None},
        },
        "bare": {"type": "noul"},
        "tone": {
            "type": "choice",
            "instructions": "What is the tone?",
            "criteria": {"calm": None, "angry": "An upset message"},
        },
        "urgency": {
            "type": "score",
            "criteria": ["Can wait", {"level": "today", "rank": 2}],
        },
    }
    outcome = {
        "expect": {"body": _body(questions, state={"message": "Hi", "amount": 12.5})},
        "json": {
            "model": "jev-1",
            "answers": {
                "spam": {"type": "noul", "noul": 0.25},
                "bare": {"type": "noul", "noul": 1},
                "tone": {
                    "type": "choice",
                    "choice": "angry",
                    "confidence": 0.8,
                    "probabilities": {"calm": 0.2, "angry": 0.8},
                },
                "urgency": {
                    "type": "score",
                    "score": 0.75,
                    "confidence": 0.5,
                    "legend": {"1": {"level": "today", "rank": 2}, "0": "Can wait"},
                    "probabilities": {"1": 0.75, "0": 0.25},
                },
            },
            "usage": {"input_tokens": 120, "output_tokens": 12},
        },
        "headers": {"x-typesafe-request-id": "req-7"},
    }
    body = """\
let state = {"message": "Hi" as json, "amount": 12.5 as json} as json
let questions: dict[text, jev::Question] = {
  "spam": jev::Question::NoulQuestion(
    {"task": "Is it spam?" as json, "weight": 0.5 as json} as json,
    Some(jev::NoulCriteria("Unsolicited" as json, null))),
  "bare": jev::Question::NoulQuestion(null, None),
  "tone": jev::Question::ChoiceQuestion("What is the tone?" as json,
    {"calm": null, "angry": "An upset message" as json}),
  "urgency": jev::Question::ScoreQuestion(null,
    ["Can wait" as json, {"level": "today" as json, "rank": 2 as json} as json]),
}
let response = jev::system-one(state, questions)
let answers: dict[text, jev::Answer] = {
  "spam": jev::Answer::NoulAnswer(0.25),
  "bare": jev::Answer::NoulAnswer(1),
  "tone": jev::Answer::ChoiceAnswer("angry", 0.8, {"calm": 0.2, "angry": 0.8}),
  "urgency": jev::Answer::ScoreAnswer(0.75, 0.5, [0.25, 0.75],
    ["Can wait" as json, {"level": "today" as json, "rank": 2 as json} as json]),
}
print(response == jev::Response("jev-1", answers, jev::Usage(Some(120), Some(12)), Some("req-7")))
"""
    result, mount = _run_jev(monkeypatch, [outcome], body)

    assert result.ok, result.error
    mount.transport.assert_complete()
    assert _printed(capsys) == ["true"]


def test_system_one_maps_unreported_usage_and_request_id_to_none(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    outcome = {
        "expect": {"body": _body()},
        "json": {
            "model": "jev-2",
            "answers": {"billing": {"type": "noul", "noul": 0.5}},
            "usage": {},
        },
    }
    body = (
        _BILLING_CALL
        + """\
let response = jev::system-one("I was charged twice." as json, billing)
let answers: dict[text, jev::Answer] = {"billing": jev::Answer::NoulAnswer(0.5)}
print(response == jev::Response("jev-2", answers, jev::Usage(None, None), None))
"""
    )
    result, mount = _run_jev(monkeypatch, [outcome], body)

    assert result.ok, result.error
    mount.transport.assert_complete()
    assert _printed(capsys) == ["true"]


@pytest.mark.parametrize(
    ("criteria_source", "criteria_wire", "noul", "values"),
    (
        ("", None, 0.9, ["true", "true", "false"]),
        (
            ', Some(jev::NoulCriteria({"kind": "billing"} as json, "Anything else" as json))',
            {"true": {"kind": "billing"}, "false": "Anything else"},
            0.3,
            ["false", "true", "false"],
        ),
    ),
)
def test_ask_noul_asks_one_noul_question_and_returns_its_probability(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    criteria_source: str,
    criteria_wire: dict[str, object] | None,
    noul: float,
    values: list[str],
) -> None:
    question: dict[str, object] = {"type": "noul", "instructions": "Is this about billing?"}
    if criteria_wire is not None:
        question["criteria"] = criteria_wire
    outcome = {
        "expect": {"body": _body({"question": question})},
        "json": {
            "model": "jev-1",
            "answers": {"question": {"type": "noul", "noul": noul}},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    }
    body = f"""\
let state = "I was charged twice." as json
let answer = jev::ask-noul("Is this about billing?", state{criteria_source})
print(answer == jev::Noul({noul}))
print(answer.value())
print(answer.value(0.2))
jev::noul-threshold := 0.95
print(answer.value())
"""
    result, mount = _run_jev(monkeypatch, [outcome], body)

    assert result.ok, result.error
    mount.transport.assert_complete()
    assert _printed(capsys) == ["true", values[0], "true", values[2]]


@pytest.mark.parametrize(
    "answers",
    (
        # Missing: the answer is under another name.
        {"other": {"type": "noul", "noul": 0.5}},
        # Mistyped: a choice answer to the noul question.
        {
            "question": {
                "type": "choice",
                "choice": "yes",
                "confidence": 0.9,
                "probabilities": {"yes": 0.9, "no": 0.1},
            }
        },
    ),
)
def test_a_missing_or_mistyped_answer_raises_a_traced_jev_response_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, answers: dict[str, object]
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    outcome = {
        "json": {"model": "jev-1", "answers": answers, "usage": {}},
        "headers": {"x-typesafe-request-id": "req-3"},
    }
    body = 'let _ = jev::ask-noul("Billing?", null)\n'
    result, mount = _run_jev(monkeypatch, [outcome], body, trace_file=trace_path)

    assert result.error is not None
    assert result.error.type_name == "JevResponseError"
    assert result.error.fields["field-path"] == "answers.question"
    assert result.error.fields["request-id"] == _some("req-3")
    mount.transport.assert_complete()
    failures = [r for r in _trace_records(trace_path) if r["kind"] == "jev_failure"]
    assert [(r["error_type"], r["status"], r["request_id"]) for r in failures] == [
        ("JevResponseError", None, "req-3")
    ]


# ---------------------------------------------------------------------------
# Settings, per-call overrides, and the SDK's environment defaults
# ---------------------------------------------------------------------------


def test_settings_reach_the_request_and_per_call_values_override_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def exchange(model: str, url: str, key: str, timeout: float) -> dict[str, object]:
        return {
            **_answered(_BILLING_ANSWER),
            "expect": {
                "body": _body(model=model),
                "url": f"{url}/v1/systemone",
                "headers": {"authorization": f"Bearer {key}"},
                "timeout": timeout,
            },
        }

    default_url = "https://api.typesafe.ai"
    outcomes = [
        exchange("jev-latest", default_url, TEST_API_KEY, 10.0),
        exchange("jev-2", "https://jev.example", "source-key", 120.0),
        exchange("jev-3", "https://jev.example", "source-key", 5.0),
        exchange("jev-2", "https://jev.example", "source-key", 1.5),
        exchange("jev-2", "https://call.example", "call-key", 120.0),
    ]
    body = (
        _BILLING_CALL
        + """\
let state = "I was charged twice." as json
let _ = jev::system-one(state, billing)
jev::model := Some("jev-2")
jev::timeout := Some("2m")
jev::base-url := Some("https://jev.example")
jev::api-key := Some("source-key")
let _ = jev::system-one(state, billing)
let _ = jev::system-one(state, billing, model = Some("jev-3"), timeout = Some("5s"))
let _ = jev::try-system-one(state, billing, timeout = Some("1.5"))
let _ = jev::system-one(state, billing, api-key = Some("call-key"),
  base-url = Some("https://call.example"))
"""
    )
    result, mount = _run_jev(monkeypatch, outcomes, body)

    assert result.ok, result.error
    mount.transport.assert_complete()


def test_module_parameters_seed_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = [
        {
            **_answered(_BILLING_ANSWER),
            "expect": {
                "body": _body(model="jev-seeded"),
                "url": "https://seeded.example/v1/systemone",
                "timeout": 30.0,
            },
        }
    ]
    params = {
        "sysone/jev::model": OptionSome("jev-seeded"),
        "sysone/jev::base-url": OptionSome("https://seeded.example"),
        "sysone/jev::timeout": OptionSome("30s"),
    }
    body = _BILLING_CALL + 'let _ = jev::system-one("I was charged twice." as json, billing)\n'
    result, mount = _run_jev(monkeypatch, outcomes, body, module_params=params)

    assert result.ok, result.error
    mount.transport.assert_complete()


def test_none_settings_leave_the_sdk_environment_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = [
        {
            **_answered(_BILLING_ANSWER),
            "expect": {
                "body": _body(model="jev-env"),
                "url": "https://env.example/v1/systemone",
                "headers": {"authorization": "Bearer env-default-key"},
            },
        }
    ]
    mount = mount_jev(monkeypatch, outcomes)
    monkeypatch.setenv("TYPESAFE_API_KEY", "env-default-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://env.example")
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "jev-env")
    source = _program(
        _BILLING_CALL + 'let _ = jev::system-one("I was charged twice." as json, billing)\n'
    )

    result = run_inline_command(mount.driver, source, roots=jev_roots(), module_params=_NO_RETRIES)

    assert result.ok, result.error
    mount.transport.assert_complete()


def _cli_module_params(driver: PipelineDriver, source: str, tokens: list[str]) -> dict[str, object]:
    """Module parameter values ``agm exec`` would parse from *tokens* for *source*'s program."""
    discovery = driver.discover_programs(prepare_inline_command(source, roots=jev_roots()))
    (program,) = [program for program in discovery.programs if program.module.is_entry]
    params = tuple(discovery.params_for(program))
    command = build_program_command(program, EXEC_RESERVED_FLAGS, params)
    assert isinstance(command, ProgramCommand)
    paths = {param.key: param.declaration_path for param in params}
    return {paths[key]: value for key, value in command.parse(tokens).params.items()}


@pytest.mark.parametrize(
    ("tokens", "key"),
    ((["--jev-max-retries", "0"], "env-key"), (["--jev-api-key", "cli-key"], "cli-key")),
)
def test_the_api_key_option_falls_back_to_its_environment_variable(
    monkeypatch: pytest.MonkeyPatch, tokens: list[str], key: str
) -> None:
    outcomes = [
        {**_answered(_BILLING_ANSWER), "expect": {"headers": {"authorization": f"Bearer {key}"}}}
    ]
    mount = mount_jev(monkeypatch, outcomes)
    source = _program(
        _BILLING_CALL + 'let _ = jev::system-one("I was charged twice." as json, billing)\n'
    )
    monkeypatch.setenv("TYPESAFE_API_KEY", "env-key")
    module_params = _cli_module_params(mount.driver, source, tokens)
    # Only the parsed setting may carry the key now, not the SDK's own lookup.
    monkeypatch.delenv("TYPESAFE_API_KEY")

    result = run_inline_command(
        mount.driver, source, roots=jev_roots(), module_params={**_NO_RETRIES, **module_params}
    )

    assert result.ok, result.error
    mount.transport.assert_complete()


@pytest.mark.parametrize(
    "body",
    (
        _BILLING_CALL + 'let _ = jev::system-one(null, billing, timeout = Some("soon"))\n',
        'jev::timeout := Some("10 minutes")\nlet _ = jev::ask-noul("Billing?", null)\n',
        'let _ = jev::try-ask-noul("Billing?", null, timeout = Some("-1s"))\n',
        'let _ = jev::try-ask-noul("Billing?", null, timeout = Some("0"))\n',
        'let _ = jev::ask-noul("Billing?", null, timeout = Some("99999999999999999999"))\n',
    ),
)
def test_a_timeout_that_is_not_a_positive_duration_raises_type_error_before_any_request(
    monkeypatch: pytest.MonkeyPatch, body: str
) -> None:
    result, mount = _run_jev(monkeypatch, [], body)

    assert result.error is not None
    assert result.error.type_name == "TypeError"
    assert mount.transport.requests == []


_RETRYABLE_503 = {"status": 503, "headers": {"retry-after-ms": "0"}}


@pytest.mark.parametrize(
    ("retries", "outcomes", "type_name"),
    (
        (1, [_RETRYABLE_503, _answered(_BILLING_ANSWER)], None),
        (0, [_RETRYABLE_503], "JevServerError"),
    ),
)
def test_max_retries_reaches_the_sdk_retry_policy(
    monkeypatch: pytest.MonkeyPatch,
    retries: int,
    outcomes: list[dict[str, object]],
    type_name: str | None,
) -> None:
    # `retry-after-ms: 0` makes the SDK retry without waiting.
    body = _BILLING_CALL + f"let _ = jev::system-one(null, billing, max-retries = {retries})\n"
    result, mount = _run_jev(monkeypatch, outcomes, body)

    mount.transport.assert_complete()
    if type_name is None:
        assert result.ok, result.error
    else:
        assert result.error is not None
        assert result.error.type_name == type_name


# ---------------------------------------------------------------------------
# The pooled client
# ---------------------------------------------------------------------------


class _ClientLog(NamedTuple):
    opened: list[object]
    closed: list[object]


def _log_clients(monkeypatch: pytest.MonkeyPatch, companion: ModuleType) -> _ClientLog:
    """Record the settings of every client *companion* opens and every client it closes."""
    log = _ClientLog([], [])
    scripted_open = companion.open_client

    def logged_open(settings: object) -> object:
        client = scripted_open(settings)
        log.opened.append(settings)
        real_close = client.close

        def logged_close() -> None:
            log.closed.append(settings)
            real_close()

        monkeypatch.setattr(client, "close", logged_close)
        return client

    monkeypatch.setattr(companion, "open_client", logged_open)
    return log


def test_one_client_is_pooled_per_settings_and_closed_when_the_run_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = [
        _answered({name: _noul(0.9)}) for name in ["billing"] * 2 + ["question"] + ["billing"] * 4
    ]
    body = (
        _BILLING_CALL
        + """\
let state = "I was charged twice." as json
let _ = jev::system-one(state, billing)
let _ = jev::system-one(state, billing, model = Some("jev-9"), timeout = Some("3s"))
let _ = jev::ask-noul("Billing?", state)
jev::api-key := Some("other-key")
let _ = jev::system-one(state, billing)
jev::api-key := None
let _ = jev::try-system-one(state, billing)
let _ = jev::system-one(state, billing, max-retries = 1)
let _ = jev::system-one(state, billing, base-url = Some("https://api.typesafe.ai"))
"""
    )
    mount = mount_jev(monkeypatch, outcomes)
    log = _log_clients(monkeypatch, mount.companion)
    settings = mount.companion.Settings

    result = run_inline_command(
        mount.driver, _program(body), roots=jev_roots(), module_params=_NO_RETRIES
    )

    assert result.ok, result.error
    mount.transport.assert_complete()
    assert log.opened == [
        settings(api_key=None, base_url=None, max_retries=0),
        settings(api_key="other-key", base_url=None, max_retries=0),
        settings(api_key=None, base_url=None, max_retries=1),
        settings(api_key=None, base_url="https://api.typesafe.ai", max_retries=0),
    ]
    assert sorted(map(repr, log.closed)) == sorted(map(repr, log.opened))


def test_pooled_clients_close_when_the_run_fails_and_each_run_opens_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mount = mount_jev(monkeypatch, [_answered(_BILLING_ANSWER), {"status": 401}])
    log = _log_clients(monkeypatch, mount.companion)
    source = _program(
        _BILLING_CALL + 'let _ = jev::system-one("I was charged twice." as json, billing)\n'
    )

    first = run_inline_command(mount.driver, source, roots=jev_roots(), module_params=_NO_RETRIES)
    second = run_inline_command(mount.driver, source, roots=jev_roots(), module_params=_NO_RETRIES)

    assert first.ok, first.error
    assert second.error is not None
    assert second.error.type_name == "JevAuthError"
    assert len(log.opened) == 2
    assert len(log.closed) == 2


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------

_ERROR_BODY = {"error": "scripted failure detail"}


@pytest.mark.parametrize(
    ("status", "headers", "type_name", "fields"),
    (
        (401, {}, "JevAuthError", {}),
        (403, {}, "JevAuthError", {}),
        (400, {}, "JevRequestError", {}),
        (404, {}, "JevRequestError", {}),
        (422, {}, "JevRequestError", {}),
        (429, {"retry-after-ms": "1500"}, "JevRateLimitError", {"retry-after-ms": _some(1500)}),
        (429, {}, "JevRateLimitError", {"retry-after-ms": _NONE}),
        (500, {}, "JevServerError", {}),
        (529, {}, "JevServerError", {}),
        (409, {}, "JevApiError", {}),
    ),
)
def test_an_unsuccessful_response_raises_its_jev_api_error(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    headers: dict[str, str],
    type_name: str,
    fields: dict[str, object],
) -> None:
    scripted = {
        "status": status,
        "json": _ERROR_BODY,
        "headers": {"x-typesafe-request-id": "req-9", **headers},
    }
    body = _BILLING_CALL + 'let _ = jev::system-one("I was charged twice." as json, billing)\n'
    result, mount = _run_jev(monkeypatch, [scripted], body)

    assert result.error is not None
    assert result.error.type_name == type_name
    assert result.error.fields["status"] == status
    assert result.error.fields["request-id"] == _some("req-9")
    assert result.error.fields["body"] == _ERROR_BODY
    assert "scripted failure detail" in str(result.error.fields["message"])
    for name, value in fields.items():
        assert result.error.fields[name] == value
    mount.transport.assert_complete()


def test_an_error_body_json_cannot_hold_is_null(monkeypatch: pytest.MonkeyPatch) -> None:
    scripted = {"status": 400, "content": '{"error": NaN, "limit": Infinity}'}
    body = _BILLING_CALL + "let _ = jev::system-one(null, billing)\n"
    result, mount = _run_jev(monkeypatch, [scripted], body)

    assert result.error is not None
    assert result.error.type_name == "JevRequestError"
    assert result.error.fields["body"] is None
    mount.transport.assert_complete()


@pytest.mark.parametrize(
    ("outcome", "type_name", "field"),
    (
        ({"fail": "connection"}, "JevConnectionError", "cause"),
        ({"fail": "timeout"}, "JevTimeoutError", "timeout"),
    ),
)
def test_a_request_without_a_response_raises_its_jev_error(
    monkeypatch: pytest.MonkeyPatch, outcome: dict[str, object], type_name: str, field: str
) -> None:
    body = _BILLING_CALL + 'let _ = jev::system-one(null, billing, timeout = Some("7s"))\n'
    result, mount = _run_jev(monkeypatch, [outcome], body)

    assert result.error is not None
    assert result.error.type_name == type_name
    assert result.error.fields["request-id"] == _NONE
    assert result.error.fields[field]
    mount.transport.assert_complete()


def test_a_timeout_error_reports_the_request_timeout_as_duration_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _BILLING_CALL + 'let _ = jev::system-one(null, billing, timeout = Some("2m"))\n'
    result, _ = _run_jev(monkeypatch, [{"fail": "timeout"}], body)

    assert result.error is not None
    assert parse_timeout(str(result.error.fields["timeout"])) == 120.0


@pytest.mark.parametrize(
    ("response", "field_path"),
    (
        ({"model": "jev-1", "answers": {}}, "usage"),
        (
            {"model": "jev-1", "answers": {"billing": {"type": "noul"}}, "usage": {}},
            "answers.billing.noul",
        ),
    ),
)
def test_a_structurally_invalid_success_body_raises_jev_response_error(
    monkeypatch: pytest.MonkeyPatch, response: dict[str, object], field_path: str
) -> None:
    outcome = {"json": response, "headers": {"x-typesafe-request-id": "req-4"}}
    body = _BILLING_CALL + "let _ = jev::system-one(null, billing)\n"
    result, mount = _run_jev(monkeypatch, [outcome], body)

    assert result.error is not None
    assert result.error.type_name == "JevResponseError"
    assert result.error.fields["field-path"] == field_path
    assert result.error.fields["request-id"] == _some("req-4")
    mount.transport.assert_complete()


@pytest.mark.parametrize(
    ("body", "api_key_env"),
    (
        # No API key anywhere: the SDK refuses to build the client.
        ('let _ = jev::ask-noul("Billing?", null)\n', None),
        # No questions: the SDK refuses the request.
        ("let _ = jev::system-one(null, {})\n", TEST_API_KEY),
    ),
)
def test_an_sdk_error_of_no_specific_class_raises_plain_jev_error(
    monkeypatch: pytest.MonkeyPatch, body: str, api_key_env: str | None
) -> None:
    mount = mount_jev(monkeypatch, [])
    if api_key_env is None:
        monkeypatch.delenv("TYPESAFE_API_KEY")

    result = run_inline_command(
        mount.driver, _program(body), roots=jev_roots(), module_params=_NO_RETRIES
    )

    assert result.error is not None
    assert result.error.type_name == "JevError"
    assert result.error.fields["request-id"] == _NONE
    assert mount.transport.requests == []


_CATCHES = (
    ({"status": 401}, "JevAuthError"),
    ({"status": 422}, "JevRequestError"),
    ({"status": 429}, "JevRateLimitError"),
    ({"status": 503}, "JevServerError"),
    ({"status": 418}, "JevApiError"),
    ({"fail": "connection"}, "JevConnectionError"),
    ({"fail": "timeout"}, "JevTimeoutError"),
    ({"json": {"model": "jev-1"}}, "JevResponseError"),
)


@pytest.mark.parametrize(("outcome", "type_name"), _CATCHES)
def test_each_jev_error_is_caught_by_its_type_and_by_jev_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    outcome: dict[str, object],
    type_name: str,
) -> None:
    body = (
        _BILLING_CALL
        + f"""\
let exact = try
  let _ = jev::system-one(null, billing)
  "none"
catch {type_name} as e => "exact"
print(exact)
let root = try
  let _ = jev::ask-noul("Billing?", null)
  "none"
catch JevError as e => "root"
print(root)
"""
    )
    result, mount = _run_jev(monkeypatch, [outcome, outcome], body)

    assert result.ok, result.error
    mount.transport.assert_complete()
    assert _printed(capsys) == ["exact", "root"]


@pytest.mark.parametrize(("outcome", "type_name"), _CATCHES)
def test_the_try_twins_return_err_with_the_jev_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    outcome: dict[str, object],
    type_name: str,
) -> None:
    body = (
        _BILLING_CALL
        + f"""\
let raw = jev::try-system-one(null, billing)
print(raw.err().unwrap() is {type_name})
let noul = jev::try-ask-noul("Billing?", null)
print(noul.err().unwrap() is {type_name})
"""
    )
    result, mount = _run_jev(monkeypatch, [outcome, outcome], body)

    assert result.ok, result.error
    mount.transport.assert_complete()
    assert _printed(capsys) == ["true", "true"]


def test_the_try_twins_return_ok_with_the_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    noul_outcome = {
        "json": {
            "model": "jev-1",
            "answers": {"question": {"type": "noul", "noul": 0.6}},
            "usage": {},
        }
    }
    body = (
        _BILLING_CALL
        + """\
let raw = jev::try-system-one("I was charged twice." as json, billing, model = Some("jev-5"))
let answers: dict[text, jev::Answer] = {"billing": jev::Answer::NoulAnswer(0.9)}
let usage = jev::Usage(Some(12), Some(1))
print(raw.unwrap() == jev::Response("jev-1", answers, usage, Some("req-1")))
let noul = jev::try-ask-noul("Billing?", null, max-retries = 3)
print(noul.unwrap() == jev::Noul(0.6))
"""
    )
    outcomes = [
        {**_answered(_BILLING_ANSWER), "expect": {"body": _body(model="jev-5")}},
        noul_outcome,
    ]
    result, mount = _run_jev(monkeypatch, outcomes, body)

    assert result.ok, result.error
    mount.transport.assert_complete()
    assert _printed(capsys) == ["true", "true"]


# ---------------------------------------------------------------------------
# Trace records
# ---------------------------------------------------------------------------

_SECRET_KEY = "sk-trace-secret-0123"


def _trace_records(path: Path) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_requests_responses_and_failures_are_traced_without_the_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    outcomes = [
        {
            **_answered({"billing": _noul(0.5)}, request_id="req-11"),
            "expect": {"body": _body(model="jev-8")},
        },
        {"status": 429, "headers": {"x-typesafe-request-id": "req-12"}},
        {"fail": "connection"},
    ]
    body = (
        _BILLING_CALL
        + f"""\
jev::api-key := Some("{_SECRET_KEY}")
let state = "I was charged twice." as json
let _ = jev::system-one(state, billing, model = Some("jev-8"))
let _ = jev::try-system-one(state, billing)
let _ = jev::try-ask-noul("Billing?", state)
"""
    )
    result, mount = _run_jev(monkeypatch, outcomes, body, trace_file=trace_path)

    assert result.ok, result.error
    mount.transport.assert_complete()
    assert _SECRET_KEY not in trace_path.read_text(encoding="utf-8")
    records = [r for r in _trace_records(trace_path) if str(r["kind"]).startswith("jev_")]
    assert [r["kind"] for r in records] == [
        "jev_request",
        "jev_response",
        "jev_request",
        "jev_failure",
        "jev_request",
        "jev_failure",
    ]
    assert {r["origin"] for r in records} == {"sysone/jev"}
    first_request, response, second_request, rate_limited, third_request, disconnected = records
    assert first_request["model"] == "jev-8"
    assert first_request["state"] == "I was charged twice."
    assert first_request["questions"] == _BILLING_QUESTION
    assert second_request["model"] is None
    assert third_request["questions"] == {"question": {"type": "noul", "instructions": "Billing?"}}
    assert response["model"] == "jev-1"
    assert response["answers"] == {"billing": {"type": "noul", "noul": 0.5}}
    assert response["usage"] == {"input_tokens": 12, "output_tokens": 1}
    assert response["request_id"] == "req-11"
    assert rate_limited["error_type"] == "JevRateLimitError"
    assert rate_limited["status"] == 429
    assert rate_limited["request_id"] == "req-12"
    assert disconnected["error_type"] == "JevConnectionError"
    assert disconnected["status"] is None
    assert disconnected["request_id"] is None


# ---------------------------------------------------------------------------
# Type-directed questions: `ask`, `ask-choice`, `ask-score`, `ask-many`
# ---------------------------------------------------------------------------

_TARGETS = """\
enum Team
  | @doc("Invoices and refunds.") Billing
  | Technical
  | @json-name("acct") Account

enum Level
  | @doc("Can wait.") Low
  | Medium
  | @json-name("hi") @doc("Drop everything.") High

record Triage
  @doc("Is it urgent?")
  urgent: bool
  spam: jev::Noul
  @json-name("owner")
  team: Team
  routed: jev::Choice[Team]
  @doc("How severe is it?") @json-name("sev")
  severity: jev::Score[Level]

enum Shape
  | Square(side: int)
  | Circle

enum Solo = Only

enum Duo = Left | Right

enum Ten = T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10

enum Wide = W1 | W2 | W3 | W4 | W5 | W6 | W7 | W8 | W9 | W10 | W11

enum Tree
  | Leaf
  | Node(children: array[Tree])

record Chain
  finished: bool
  next: Option[Chain]

record Ticket
  urgent: bool
  note: text

record Blank
"""
_TEAM_OPTIONS = {"Billing": "Invoices and refunds.", "Technical": None, "acct": None}
_LEVELS = ["Can wait.", "Medium", "Drop everything."]


def _question(
    kind: str, instructions: str | None = "Which?", **criteria: object
) -> dict[str, object]:
    question: dict[str, object] = {"type": kind, **criteria}
    if instructions is not None:
        question["instructions"] = instructions
    return question


_NOUL_Q = _question("noul")
_CHOICE_Q = _question("choice", criteria=_TEAM_OPTIONS)
_SCORE_Q = _question("score", criteria=_LEVELS)


def _choice(choice: str, probabilities: dict[str, float]) -> dict[str, object]:
    return {
        "type": "choice",
        "choice": choice,
        "confidence": 0.7,
        "probabilities": probabilities,
    }


def _score(
    score: float, probabilities: dict[str, float], levels: Sequence[str] = _LEVELS
) -> dict[str, object]:
    legend = {str(index): level for index, level in enumerate(levels)}
    return {
        "type": "score",
        "score": score,
        "confidence": 0.4,
        "legend": legend,
        "probabilities": probabilities,
    }


def _single(question: dict[str, object], answer: dict[str, object]) -> dict[str, object]:
    return _answered({"question": answer}, {"question": question})


_EVEN_TIE = {"0": 0.2, "1": 0.4, "2": 0.4}
_ACCOUNT_ODDS = {"Billing": 0.1, "Technical": 0.2, "acct": 0.7}
_BILLING_ODDS = {"Billing": 0.7, "Technical": 0.2, "acct": 0.1}
_TECHNICAL_ODDS = {"Billing": 0.2, "Technical": 0.7, "acct": 0.1}


@pytest.mark.parametrize(
    ("target", "question", "answer", "expected"),
    (
        ("bool", _NOUL_Q, _noul(0.5), "true"),
        ("bool", _NOUL_Q, _noul(0.49), "false"),
        ("jev::Noul", _NOUL_Q, _noul(0.3), "jev::Noul(0.3)"),
        ("Team", _CHOICE_Q, _choice("acct", _ACCOUNT_ODDS), "Team::Account"),
        ("Team", _CHOICE_Q, _choice("Billing", _BILLING_ODDS), "Team::Billing"),
        (
            "jev::Choice[Team]",
            _CHOICE_Q,
            _choice("Technical", {"Billing": 0.1, "Technical": 0.7, "acct": 0.2}),
            "jev::Choice(Team::Technical as Team, 0.7,"
            ' {"Billing": 0.1, "Technical": 0.7, "acct": 0.2})',
        ),
        # A tie between the two most probable levels selects the lower one.
        (
            "jev::Score[Level]",
            _SCORE_Q,
            _score(1.2, _EVEN_TIE),
            "jev::Score(Level::Medium as Level, 1.2, 0.4, [0.2, 0.4, 0.4])",
        ),
        (
            "jev::Score[Level]",
            _SCORE_Q,
            _score(1.9, {"2": 0.9, "0": 0.05, "1": 0.05}),
            "jev::Score(Level::High as Level, 1.9, 0.4, [0.05, 0.05, 0.9])",
        ),
    ),
)
def test_ask_asks_the_question_its_target_selects_and_builds_the_target(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    target: str,
    question: dict[str, object],
    answer: dict[str, object],
    expected: str,
) -> None:
    body = f"""\
let state = "{_STATE}" as json
let explicit = jev::ask::[{target}]("Which?", state)
let contextual: {target} = jev::ask("Which?", state)
let expected: {target} = {expected}
print(explicit == expected)
print(contextual == expected)
"""
    outcomes = [_single(question, answer)] * 2
    result, mount = _run_jev(monkeypatch, outcomes, body, declarations=_TARGETS)

    assert result.ok, result.error
    mount.transport.assert_complete()
    assert _printed(capsys) == ["true", "true"]


def test_ask_choice_and_ask_score_wrap_their_member_type(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    choice = _choice("acct", {"Billing": 0.1, "Technical": 0.1, "acct": 0.8})
    score = _score(0.3, {"0": 0.7, "1": 0.3, "2": 0.0})
    outcomes = [
        _single(_CHOICE_Q, choice),
        _single(_CHOICE_Q, choice),
        _single(_SCORE_Q, score),
        _single(_SCORE_Q, score),
    ]
    body = f"""\
let state = "{_STATE}" as json
let probabilities = {{"Billing": 0.1, "Technical": 0.1, "acct": 0.8}}
let chosen: jev::Choice[Team] = jev::Choice(Team::Account as Team, 0.7, probabilities)
print(jev::ask-choice::[Team]("Which?", state) == chosen)
let contextual-choice: jev::Choice[Team] = jev::ask-choice("Which?", state)
print(contextual-choice == chosen)
let scored: jev::Score[Level] = jev::Score(Level::Low as Level, 0.3, 0.4, [0.7, 0.3, 0.0])
print(jev::ask-score::[Level]("Which?", state) == scored)
let contextual-score: jev::Score[Level] = jev::ask-score("Which?", state)
print(contextual-score == scored)
"""
    result, mount = _run_jev(monkeypatch, outcomes, body, declarations=_TARGETS)

    assert result.ok, result.error
    mount.transport.assert_complete()
    assert _printed(capsys) == ["true"] * 4


_TRIAGE_QUESTIONS = {
    "urgent": _question("noul", "Is it urgent?"),
    "spam": _question("noul", "spam"),
    "owner": _question("choice", "team", criteria=_TEAM_OPTIONS),
    "routed": _question("choice", "routed", criteria=_TEAM_OPTIONS),
    "sev": _question("score", "How severe is it?", criteria=_LEVELS),
}


@pytest.mark.parametrize(
    ("answers", "expected"),
    (
        (
            {
                "urgent": _noul(0.8),
                "spam": _noul(0.1),
                "owner": _choice("Billing", _BILLING_ODDS),
                "routed": _choice("acct", _ACCOUNT_ODDS),
                "sev": _score(1.2, _EVEN_TIE),
            },
            "Triage(true, jev::Noul(0.1), Team::Billing,"
            ' jev::Choice(Team::Account as Team, 0.7, {"Billing": 0.1, "Technical": 0.2,'
            ' "acct": 0.7}),'
            " jev::Score(Level::Medium as Level, 1.2, 0.4, [0.2, 0.4, 0.4]))",
        ),
        (
            {
                "sev": _score(0.0, {"0": 1.0, "1": 0.0, "2": 0.0}),
                "routed": _choice("Technical", _TECHNICAL_ODDS),
                "owner": _choice("acct", _ACCOUNT_ODDS),
                "spam": _noul(0.9),
                "urgent": _noul(0.2),
            },
            "Triage(false, jev::Noul(0.9), Team::Account,"
            ' jev::Choice(Team::Technical as Team, 0.7, {"Billing": 0.2, "Technical": 0.7,'
            ' "acct": 0.1}),'
            " jev::Score(Level::Low as Level, 0.0, 0.4, [1.0, 0.0, 0.0]))",
        ),
    ),
)
def test_ask_many_asks_one_question_per_field_in_one_request(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    answers: dict[str, object],
    expected: str,
) -> None:
    body = f"""\
let state = {{"ticket": "{_STATE}" as json}} as json
print(jev::ask-many::[Triage](state) == {expected})
let contextual: Triage = jev::ask-many(state)
print(contextual == {expected})
"""
    outcomes = [_answered(answers, _TRIAGE_QUESTIONS, state={"ticket": _STATE})] * 2
    result, mount = _run_jev(monkeypatch, outcomes, body, declarations=_TARGETS)

    assert result.ok, result.error
    mount.transport.assert_complete()
    assert _printed(capsys) == ["true", "true"]


def test_the_noul_threshold_setting_decides_bool_targets(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    outcomes = [_single(_NOUL_Q, _noul(0.7))] * 3 + [
        _answered({"urgent": _noul(0.7)}, {"urgent": _question("noul", "urgent")})
    ] * 2
    body = f"""\
let state = "{_STATE}" as json
let default: bool = jev::ask("Which?", state)
jev::noul-threshold := 0.8
let raised: bool = jev::ask("Which?", state)
let per-call: bool = jev::ask("Which?", state, noul-threshold = 0.7)
print([default, raised, per-call])
let many = jev::ask-many::[Urgency](state)
let many-per-call = jev::ask-many::[Urgency](state, noul-threshold = 0.6)
print([many.urgent, many-per-call.urgent])
"""
    declarations = _TARGETS + "record Urgency\n  urgent: bool\n"
    result, mount = _run_jev(monkeypatch, outcomes, body, declarations=declarations)

    assert result.ok, result.error
    mount.transport.assert_complete()
    assert _printed(capsys) == ["[true, false, true]", "[false, true]"]


@pytest.mark.parametrize(
    ("call", "target"),
    (
        ('jev::ask::[text]("Which?", null)', "text"),
        ('jev::ask::[decimal]("Which?", null)', "decimal"),
        ('jev::ask::[array[bool]]("Which?", null)', "array[bool]"),
        ('jev::ask::[Ticket]("Which?", null)', "Ticket"),
        ('jev::ask::[Option[Team]]("Which?", null)', "std/option::Option[Team]"),
        ('jev::ask::[Shape]("Which?", null)', "Shape"),
        ('jev::ask::[Solo]("Which?", null)', "Solo"),
        ('jev::ask::[Tree]("Which?", null)', "Tree"),
        ('jev::ask::[Chain]("Which?", null)', "Chain"),
        ('jev::ask::[jev::Choice[Shape]]("Which?", null)', "sysone/jev::Choice[Shape]"),
        ('jev::ask::[jev::Score[Wide]]("Which?", null)', "sysone/jev::Score[Wide]"),
        ('jev::ask-choice::[Solo]("Which?", null)', "Solo"),
        ('jev::ask-choice::[text]("Which?", null)', "text"),
        ('jev::ask-score::[Shape]("Which?", null)', "Shape"),
        ('jev::ask-score::[Wide]("Which?", null)', "Wide"),
        ('jev::ask-score::[Solo]("Which?", null)', "Solo"),
        ('jev::ask-score::[text]("Which?", null)', "text"),
        ("jev::ask-many::[Team](null)", "Team"),
        ("jev::ask-many::[bool](null)", "bool"),
        ("jev::ask-many::[Blank](null)", "Blank"),
    ),
)
def test_an_unsupported_target_raises_jev_target_error_before_any_request(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], call: str, target: str
) -> None:
    body = f"""\
let outcome = try
  let _ = {call}
  "none"
catch JevTargetError as e => e.target
print(outcome)
"""
    result, mount = _run_jev(monkeypatch, [], body, declarations=_TARGETS)

    assert result.ok, result.error
    assert mount.transport.requests == []
    assert _printed(capsys) == [target]


@pytest.mark.parametrize(("record", "target"), (("Ticket", "Ticket.note"), ("Chain", "Chain.next")))
def test_an_unsupported_ask_many_field_raises_jev_target_error_locating_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], record: str, target: str
) -> None:
    body = f"""\
let outcome = try
  let _ = jev::ask-many::[{record}](null)
  "none"
catch JevTargetError as e => e.target
print(outcome == "{target}")
"""
    result, mount = _run_jev(monkeypatch, [], body, declarations=_TARGETS)

    assert result.ok, result.error
    assert mount.transport.requests == []
    assert _printed(capsys) == ["true"]


def test_a_jev_target_error_is_traced_without_a_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    body = 'let _ = try jev::ask::[text]("Which?", null) catch JevTargetError as e => ""\n'
    result, mount = _run_jev(monkeypatch, [], body, declarations=_TARGETS, trace_file=trace_path)

    assert result.ok, result.error
    assert mount.transport.requests == []
    records = [r for r in _trace_records(trace_path) if str(r["kind"]).startswith("jev_")]
    assert [(r["kind"], r["error_type"]) for r in records] == [("jev_failure", "JevTargetError")]


def test_the_smallest_choice_and_the_largest_rubric_are_targets(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ten = [f"T{level}" for level in range(1, 11)]
    odds = {str(index): 0.1 for index in range(10)} | {"9": 0.2, "0": 0.0}
    outcomes = [
        _single(
            _question("choice", criteria={"Left": None, "Right": None}),
            _choice("Right", {"Left": 0.3, "Right": 0.7}),
        ),
        _single(_question("score", criteria=ten), _score(6.3, odds, ten)),
    ]
    body = f"""\
let state = "{_STATE}" as json
print(jev::ask::[Duo]("Which?", state))
print(jev::ask-score::[Ten]("Which?", state).level)
"""
    result, mount = _run_jev(monkeypatch, outcomes, body, declarations=_TARGETS)

    assert result.ok, result.error
    mount.transport.assert_complete()
    assert _printed(capsys) == ["Duo::Right", "Ten::T10"]


@pytest.mark.parametrize(
    "outcome",
    (
        {"status": 401},
        _single(_CHOICE_Q, _noul(0.5)),
        _single(_CHOICE_Q, _choice("Unknown", {"Unknown": 1.0})),
    ),
)
def test_a_type_directed_failure_is_caught_as_jev_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    outcome: dict[str, object],
) -> None:
    body = f"""\
let state = "{_STATE}" as json
let team: Team = try jev::ask::[Team]("Which?", state) catch JevError as e => Team::Account
print(team)
let solo: Solo = try jev::ask::[Solo]("Which?", state) catch JevError as e => Solo::Only
print(solo)
"""
    result, mount = _run_jev(monkeypatch, [outcome], body, declarations=_TARGETS)

    assert result.ok, result.error
    mount.transport.assert_complete()
    assert _printed(capsys) == ["Team::Account", "Solo::Only"]


@pytest.mark.parametrize(
    ("target", "question", "answer", "field_path"),
    (
        ("Team", _CHOICE_Q, _choice("Other", {"Other": 1.0}), "answers.question.choice"),
        ("jev::Score[Level]", _SCORE_Q, _score(3.0, {"3": 1.0}), "answers.question.probabilities"),
        ("jev::Score[Level]", _SCORE_Q, _score(0.0, {}), "answers.question.probabilities"),
        # A level without a probability.
        (
            "jev::Score[Level]",
            _SCORE_Q,
            _score(1.0, {"0": 0.5, "2": 0.5}),
            "answers.question.probabilities",
        ),
        # A probability for a level outside the rubric.
        (
            "jev::Score[Level]",
            _SCORE_Q,
            _score(1.0, {**_EVEN_TIE, "3": 0.0}),
            "answers.question.probabilities",
        ),
        # A label without a probability.
        ("Team", _CHOICE_Q, _choice("acct", {"acct": 1.0}), "answers.question.probabilities"),
        # A probability for a label outside the choice.
        (
            "jev::Choice[Team]",
            _CHOICE_Q,
            _choice("acct", {**_ACCOUNT_ODDS, "Other": 0.0}),
            "answers.question.probabilities",
        ),
        ("bool", _NOUL_Q, _choice("Billing", {"Billing": 1.0}), "answers.question"),
    ),
)
def test_an_answer_outside_the_target_raises_jev_response_error(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    question: dict[str, object],
    answer: dict[str, object],
    field_path: str,
) -> None:
    outcome = _answered({"question": answer}, {"question": question}, request_id="req-5")
    body = f'let answer = jev::ask::[{target}]("Which?", "{_STATE}" as json)\n'
    result, mount = _run_jev(monkeypatch, [outcome], body, declarations=_TARGETS)

    assert result.error is not None
    assert result.error.type_name == "JevResponseError"
    assert result.error.fields["field-path"] == field_path
    assert result.error.fields["request-id"] == _some("req-5")
    mount.transport.assert_complete()
