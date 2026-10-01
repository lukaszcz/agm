"""Tests for structured agent progress streams and final response decoding."""

from __future__ import annotations

import json

import pytest

from agm.agent.stream import ClaudeOutputStream, CodexOutputStream, decode_claude_stream_json


def _jsonl(value: object) -> str:
    return json.dumps(value) + "\n"


def test_claude_stream_separates_progress_from_final_response() -> None:
    output: list[tuple[str, str]] = []
    stream = ClaudeOutputStream(lambda phase, text: output.append((phase, text)))
    events = [
        {"type": "stream_event"},
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "name": "Read"},
            },
        },
        {
            "type": "stream_event",
            "event": {"type": "content_block_start", "content_block": {"type": "text"}},
        },
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Inspecting"},
            },
        },
        {"type": "assistant", "message": {"stop_reason": "tool_use"}},
        {"type": "assistant", "message": {"stop_reason": "end_turn"}},
        {
            "type": "stream_event",
            "event": {"type": "content_block_delta", "delta": {"type": "thinking_delta"}},
        },
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "The final answer"},
            },
        },
        {
            "type": "stream_event",
            "event": {"type": "content_block_delta", "delta": {"type": "text_delta"}},
        },
        {"type": "stream_event", "event": {"type": "content_block_stop"}},
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "name": ""},
            },
        },
        {"type": "assistant", "message": {"stop_reason": "end_turn"}},
        {"type": "assistant", "message": {"stop_reason": "tool_use"}},
        {"type": "assistant", "message": "malformed"},
        {"type": "assistant"},
        {
            "type": "user",
            "message": {
                "content": [
                    None,
                    {"type": "other"},
                    {"type": "tool_result", "content": ""},
                    {"type": "tool_result", "content": "file contents"},
                    {
                        "type": "tool_result",
                        "content": [None, {"text": "more output"}, {"text": 2}],
                    },
                ]
            },
        },
        {"type": "user", "message": {"content": "malformed"}},
        {"type": "user"},
        {"type": "other"},
    ]
    encoded = "\n".join(json.dumps(event) for event in events)
    stream.feed("\nnot-json\n[]\n" + encoded[:90])
    stream.feed(encoded[90:] + "\n" + _jsonl({"type": "assistant"}).rstrip())
    stream.finish()
    stream.finish()

    assert output == [
        ("progress", "[Read]\n"),
        ("progress", "Inspecting"),
        ("progress", "file contents"),
        ("progress", "more output"),
    ]


def test_claude_final_response_is_decoded_from_result_event() -> None:
    output = "\n  \n" + _jsonl({"type": "system"}) + _jsonl({"type": "result", "result": "answer"})
    assert decode_claude_stream_json(output) == "answer"


@pytest.mark.parametrize(
    "output",
    ["{", "[]", _jsonl({"type": "result", "result": 2}), _jsonl({"type": "system"})],
)
def test_claude_final_response_rejects_invalid_streams(output: str) -> None:
    with pytest.raises(ValueError):
        decode_claude_stream_json(output)


def test_codex_stream_echoes_tool_progress_but_not_final_message() -> None:
    output: list[tuple[str, str]] = []
    stream = CodexOutputStream(lambda phase, text: output.append((phase, text)))
    events = [
        {"type": "item.started", "item": {"type": "command_execution", "command": "ls"}},
        {
            "type": "item.updated",
            "item": {
                "id": "cmd-1",
                "type": "command_execution",
                "aggregated_output": "one\n",
            },
        },
        {
            "type": "item.updated",
            "item": {
                "id": "cmd-1",
                "type": "command_execution",
                "aggregated_output": "one\ntwo\n",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "cmd-1",
                "type": "command_execution",
                "aggregated_output": "one\ntwo\n",
            },
        },
        {"type": "item.started", "item": {"type": "web_search", "query": "AGM"}},
        {"type": "item.started", "item": {"type": "mcp_tool_call", "name": "lookup"}},
        {
            "type": "item.completed",
            "item": {
                "type": "mcp_tool_call",
                "result": {"content": [{"type": "text", "text": "tool result\n"}]},
            },
        },
        {"type": "item.started", "item": {"type": "web_search"}},
        {"type": "item.started", "item": {"type": "reasoning"}},
        {"type": "item.completed", "item": {"type": "reasoning"}},
        {"type": "item.started", "item": {"type": "command_execution"}},
        {
            "type": "item.updated",
            "item": {"type": "reasoning", "text": "not user-facing"},
        },
        {
            "type": "item.updated",
            "item": {"id": "cmd-1", "type": "command_execution", "aggregated_output": "revised\n"},
        },
        {
            "type": "item.completed",
            "item": {"type": "command_execution", "aggregated_output": "without id\n"},
        },
        {"type": "item.completed", "item": {"type": "mcp_tool_call", "result": None}},
        {
            "type": "item.completed",
            "item": {"type": "mcp_tool_call", "result": {"content": "invalid"}},
        },
        {
            "type": "item.completed",
            "item": {
                "type": "mcp_tool_call",
                "result": {"content": [None, {"type": "image"}, {"type": "text"}]},
            },
        },
        {"type": "item.completed", "item": {"type": "command_execution"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "final"}},
        {"type": "item.started"},
        {"type": "item.completed", "item": {"type": 2}},
        {"type": "other", "item": {"type": "command_execution", "command": "ignored"}},
    ]
    encoded = "\n".join(json.dumps(event) for event in events)
    stream.feed("\ninvalid\n[]\n{}\n" + encoded)
    stream.finish()
    stream.finish()

    assert output == [
        ("progress", "$ ls\n"),
        ("progress", "one\n"),
        ("progress", "two\n"),
        ("progress", "[AGM]\n"),
        ("progress", "[lookup]\n"),
        ("progress", "tool result\n"),
        ("progress", "revised\n"),
        ("progress", "without id\n"),
    ]
