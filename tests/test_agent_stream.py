"""Tests for structured agent progress streams and final response decoding."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from agm.agent.stream import ClaudeOutputStream, CodexOutputStream, decode_claude_stream_json


def _output_collector(output: list[dict[str, str]]) -> Callable[..., None]:
    def collect(
        phase: str,
        text: str,
        *,
        event_type: str | None = None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
    ) -> None:
        item = {"phase": phase, "text": text}
        for key, value in (
            ("type", event_type),
            ("tool_name", tool_name),
            ("tool_call_id", tool_call_id),
        ):
            if value is not None:
                item[key] = value
        output.append(item)

    return collect


def _jsonl(value: object) -> str:
    return json.dumps(value, ensure_ascii=False) + "\n"


def test_claude_stream_separates_progress_from_final_response() -> None:
    output: list[dict[str, str]] = []
    stream = ClaudeOutputStream(_output_collector(output))
    events = [
        {"type": "stream_event"},
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "id": "tool-1", "name": "Read"},
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
        {"type": "stream_event", "event": {"type": "message_delta"}},
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
                    {"type": "tool_result", "tool_use_id": "tool-1", "content": "file contents"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
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
        {
            "phase": "progress",
            "type": "tool_call",
            "tool_name": "Read",
            "tool_call_id": "tool-1",
            "text": "[Read]\n",
        },
        {"phase": "progress", "type": "message", "text": "Inspecting"},
        {
            "phase": "progress",
            "type": "tool_result",
            "tool_call_id": "tool-1",
            "text": "file contents",
        },
        {
            "phase": "progress",
            "type": "tool_result",
            "tool_call_id": "tool-1",
            "text": "more output",
        },
    ]


def test_claude_stream_waits_for_message_delta_before_classifying_text() -> None:
    output: list[dict[str, str]] = []
    stream = ClaudeOutputStream(_output_collector(output))
    events = [
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "I will inspect."},
            },
        },
        {"type": "assistant", "message": {"stop_reason": None}},
        {
            "type": "stream_event",
            "event": {"type": "message_delta", "delta": {"stop_reason": None}},
        },
        {
            "type": "stream_event",
            "event": {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        },
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "The final answer."},
            },
        },
        {"type": "assistant", "message": {"stop_reason": None}},
        {
            "type": "stream_event",
            "event": {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        },
    ]

    stream.feed("".join(_jsonl(event) for event in events))
    stream.finish()

    assert output == [{"phase": "progress", "type": "message", "text": "I will inspect."}]


def test_claude_final_response_is_decoded_from_result_event() -> None:
    output = "\n  \n" + _jsonl({"type": "system"}) + _jsonl({"type": "result", "result": "answer"})
    assert decode_claude_stream_json(output) == "answer"


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_claude_final_response_preserves_unicode_jsonl_separators(separator: str) -> None:
    result = f"left{separator}right"

    assert decode_claude_stream_json(_jsonl({"type": "result", "result": result})) == result


@pytest.mark.parametrize(
    "output",
    ["{", "[]", _jsonl({"type": "result", "result": 2}), _jsonl({"type": "system"})],
)
def test_claude_final_response_rejects_invalid_streams(output: str) -> None:
    with pytest.raises(ValueError):
        decode_claude_stream_json(output)


def test_codex_stream_echoes_tool_progress_but_not_final_message() -> None:
    output: list[dict[str, str]] = []
    stream = CodexOutputStream(_output_collector(output))
    events = [
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": ""},
        },
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "I will inspect."},
        },
        {"type": "item.started", "item": {"type": "agent_message"}},
        {
            "type": "item.started",
            "item": {"id": "cmd-1", "type": "command_execution", "command": "ls"},
        },
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
        {
            "type": "item.started",
            "item": {"id": "search-1", "type": "web_search", "query": "AGM"},
        },
        {
            "type": "item.started",
            "item": {
                "id": "mcp-1",
                "type": "mcp_tool_call",
                "server": "docs",
                "tool": "lookup",
            },
        },
        {"type": "item.started", "item": {"type": "mcp_tool_call", "tool": "search"}},
        {"type": "item.started", "item": {"type": "mcp_tool_call"}},
        {
            "type": "item.completed",
            "item": {
                "id": "mcp-1",
                "type": "mcp_tool_call",
                "server": "docs",
                "tool": "lookup",
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
        {"phase": "progress", "type": "message", "text": "I will inspect."},
        {
            "phase": "progress",
            "type": "tool_call",
            "tool_name": "command_execution",
            "tool_call_id": "cmd-1",
            "text": "$ ls\n",
        },
        {
            "phase": "progress",
            "type": "tool_result",
            "tool_name": "command_execution",
            "tool_call_id": "cmd-1",
            "text": "one\n",
        },
        {
            "phase": "progress",
            "type": "tool_result",
            "tool_name": "command_execution",
            "tool_call_id": "cmd-1",
            "text": "two\n",
        },
        {
            "phase": "progress",
            "type": "tool_call",
            "tool_name": "web_search",
            "tool_call_id": "search-1",
            "text": "[AGM]\n",
        },
        {
            "phase": "progress",
            "type": "tool_call",
            "tool_name": "docs.lookup",
            "tool_call_id": "mcp-1",
            "text": "[docs.lookup]\n",
        },
        {
            "phase": "progress",
            "type": "tool_call",
            "tool_name": "search",
            "text": "[search]\n",
        },
        {
            "phase": "progress",
            "type": "tool_result",
            "tool_name": "docs.lookup",
            "tool_call_id": "mcp-1",
            "text": "tool result\n",
        },
        {
            "phase": "progress",
            "type": "tool_result",
            "tool_name": "command_execution",
            "tool_call_id": "cmd-1",
            "text": "revised\n",
        },
        {
            "phase": "progress",
            "type": "tool_result",
            "tool_name": "command_execution",
            "text": "without id\n",
        },
    ]
