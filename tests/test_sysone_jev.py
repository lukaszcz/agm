"""The ``sysone`` package: manifest, the ``jev`` module surface, and its test transport."""

from __future__ import annotations

import pytest
import typesafe_sdk

from agm.agl.pipeline import PipelineDriver
from agm.agl.runtime.externs import ExternRegistry
from tests._agl_helpers import prepare_inline_command
from tests._jev_helpers import (
    JEV_MODULE,
    SYSONE_ROOT,
    TEST_API_KEY,
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
    result = PipelineDriver().check_prepared(prepare_inline_command(source, roots=jev_roots()))

    assert result.ok, result.diagnostics


def test_registry_returns_the_loaded_jev_companion() -> None:
    registry = ExternRegistry()
    companion = load_jev_companion(PipelineDriver(extern_registry=registry), registry)

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
    driver = PipelineDriver(extern_registry=registry)

    transport = install_jev_transport(monkeypatch, driver, registry, [_NOUL_EXCHANGE])
    companion = registry.loaded_companion(JEV_MODULE)
    assert companion is not None
    client = companion.open_client(companion.Settings(api_key=None, base_url=None, max_retries=0))

    assert _ask_billing(client).nouls["billing"].noul == 0.9
    transport.assert_complete()
