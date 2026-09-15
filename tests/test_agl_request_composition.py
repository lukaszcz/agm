"""Prompt composition for initial agent requests and corrective follow-ups.

Corrective validation feedback is a category-based summary: it must not expose
response-derived validator messages, paths, or keys.
"""

from __future__ import annotations

import pytest

from agm.agent.spec import AgentCommand
from agm.agl.runtime.contract import TypelessOutputContract
from agm.agl.runtime.request import (
    AgentRequest,
    ValidationError,
    compose_corrective_follow_up,
    compose_initial_agent_prompt,
    compose_session_corrective_follow_up,
)


@pytest.mark.parametrize("strict_json", [False, True])
def test_initial_prompt_includes_the_format_contract_for_each_json_mode(strict_json: bool) -> None:
    contract = TypelessOutputContract(
        target_type="int",
        codec_name="json",
        strict_json=strict_json,
        format_instructions="Return an integer as JSON.",
        json_schema=None,
    )

    prompt = compose_initial_agent_prompt(
        AgentRequest(
            agent=AgentCommand(command="runner"),
            prompt="Count.",
            output_contract=contract,
        )
    )

    assert prompt == "Count.\n\nReturn an integer as JSON."


def test_initial_prompt_without_a_format_contract_is_just_the_original_prompt() -> None:
    prompt = compose_initial_agent_prompt(
        AgentRequest(agent=AgentCommand(command="runner"), prompt="Count.")
    )

    assert prompt == "Count."


@pytest.mark.parametrize("strict_json", [False, True])
def test_corrective_follow_up_contains_feedback_without_recomposing_the_initial_prompt(
    strict_json: bool,
) -> None:
    contract = TypelessOutputContract(
        target_type="int",
        codec_name="json",
        strict_json=strict_json,
        format_instructions="Return an integer as JSON.",
        json_schema=None,
    )

    follow_up = compose_corrective_follow_up(
        AgentRequest(
            agent=AgentCommand(command="runner"),
            prompt="ORIGINAL REQUEST",
            attempt=1,
            previous_invalid_output="not-a-number",
            validation_errors=[
                ValidationError(
                    category="unknown_field",
                    message='Unexpected response key "rejected_response_key" at $.nested.',
                    path="$.nested.rejected_response_key",
                    field="rejected_response_key",
                ),
                ValidationError(
                    category="unknown_field",
                    message='Unexpected response key "another_rejected_key" at $.nested.',
                    path="$.nested.another_rejected_key",
                    field="another_rejected_key",
                ),
            ],
            output_contract=contract,
        )
    )

    assert "ORIGINAL REQUEST" not in follow_up
    assert "Return an integer as JSON." not in follow_up
    assert (
        "Validation errors:\n- The response contains fields not permitted by the schema."
        in follow_up
    )
    assert follow_up.count("The response contains fields not permitted by the schema.") == 1
    assert "rejected_response_key" not in follow_up
    assert "another_rejected_key" not in follow_up
    assert "$.nested" not in follow_up
    assert "Previous response:\nnot-a-number" in follow_up
    assert follow_up.endswith("Return only valid JSON matching the schema.")


def test_session_corrective_follow_up_omits_the_original_prompt_and_invalid_output() -> None:
    follow_up = compose_session_corrective_follow_up(
        AgentRequest(
            agent=AgentCommand(command="runner"),
            prompt="ORIGINAL REQUEST",
            attempt=1,
            previous_invalid_output="not-a-number",
            validation_errors=[
                ValidationError(
                    category="wrong_type",
                    message='"untrusted value" is not of type integer at $.payload.',
                    path="$.payload",
                    field="payload",
                )
            ],
        )
    )

    assert follow_up.startswith(
        "Validation errors:\n- The response contains a value with an incorrect type."
    )
    assert "untrusted value" not in follow_up
    assert "$.payload" not in follow_up
    assert "ORIGINAL REQUEST" not in follow_up
    assert "not-a-number" not in follow_up
    assert follow_up.endswith("Return only valid JSON matching the schema.")


def test_corrective_follow_up_without_a_format_contract_keeps_the_json_reminder() -> None:
    follow_up = compose_corrective_follow_up(
        AgentRequest(
            agent=AgentCommand(command="runner"),
            prompt="ORIGINAL REQUEST",
            attempt=1,
        )
    )

    assert "ORIGINAL REQUEST" not in follow_up
    assert (
        "Validation errors:\n- The response does not match the required output format." in follow_up
    )
    assert follow_up.endswith("Return only valid JSON matching the schema.")
