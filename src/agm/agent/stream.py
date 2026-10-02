"""Structured agent output streams and their final-response decoders."""

from __future__ import annotations

import json
from typing import cast

from agm.agent.transport import AgentOutputCallback
from agm.util.unicode import loads_json


class ClaudeOutputStream:
    """Decode Claude's stream-json progress while retaining its final response."""

    def __init__(self, callback: AgentOutputCallback) -> None:
        self._callback = callback
        self._buffer = ""
        self._assistant_text: list[str] = []

    def feed(self, chunk: str) -> None:
        """Consume one process chunk and emit completed progress events."""
        self._buffer += chunk
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._consume_line(line)

    def finish(self) -> None:
        """Consume a final unterminated JSONL record, if present."""
        if self._buffer:
            self._consume_line(self._buffer)
            self._buffer = ""

    def _consume_line(self, line: str) -> None:
        if not line.strip():
            return
        try:
            decoded: object = loads_json(line)
        except json.JSONDecodeError:
            return
        if not isinstance(decoded, dict):
            return
        event = cast(dict[str, object], decoded)
        event_type = event.get("type")
        if event_type == "stream_event":
            self._stream_event(event)
        elif event_type == "assistant":
            self._assistant_message(event)
        elif event_type == "user":
            self._tool_result(event)

    def _stream_event(self, event: dict[str, object]) -> None:
        raw_event = event.get("event")
        if not isinstance(raw_event, dict):
            return
        stream_event = cast(dict[str, object], raw_event)
        stream_event_type = stream_event.get("type")
        if stream_event_type == "content_block_start":
            block = stream_event.get("content_block")
            if isinstance(block, dict) and cast(dict[str, object], block).get("type") == "tool_use":
                block = cast(dict[str, object], block)
                name = block.get("name")
                if isinstance(name, str) and name:
                    call_id = block.get("id")
                    self._callback(
                        "progress",
                        f"[{name}]\n",
                        event_type="tool_call",
                        tool_name=name,
                        tool_call_id=call_id if isinstance(call_id, str) and call_id else None,
                    )
        elif stream_event_type == "content_block_delta":
            delta = stream_event.get("delta")
            if (
                isinstance(delta, dict)
                and cast(dict[str, object], delta).get("type") == "text_delta"
            ):
                delta = cast(dict[str, object], delta)
                text = delta.get("text")
                if isinstance(text, str):
                    self._assistant_text.append(text)
        elif stream_event_type == "message_delta":
            delta = stream_event.get("delta")
            if isinstance(delta, dict):
                stop_reason = cast(dict[str, object], delta).get("stop_reason")
                if stop_reason is not None:
                    self._classify_assistant_text(stop_reason)

    def _assistant_message(self, event: dict[str, object]) -> None:
        message = event.get("message")
        if not isinstance(message, dict):
            return
        message = cast(dict[str, object], message)
        stop_reason = message.get("stop_reason")
        if stop_reason is None:
            return
        self._classify_assistant_text(stop_reason)

    def _classify_assistant_text(self, stop_reason: object) -> None:
        if stop_reason not in ("end_turn", "stop_sequence"):
            text = "".join(self._assistant_text)
            if text:
                self._callback("progress", text, event_type="message")
        self._assistant_text.clear()

    def _tool_result(self, event: dict[str, object]) -> None:
        message = event.get("message")
        if not isinstance(message, dict):
            return
        content = message.get("content")
        if not isinstance(content, list):
            return
        for block in content:
            if not isinstance(block, dict):
                continue
            block = cast(dict[str, object], block)
            if block.get("type") != "tool_result":
                continue
            result = block.get("content")
            call_id = block.get("tool_use_id")
            if isinstance(result, str) and result:
                self._callback(
                    "progress",
                    result,
                    event_type="tool_result",
                    tool_call_id=call_id if isinstance(call_id, str) and call_id else None,
                )
            elif isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        item = cast(dict[str, object], item)
                        if isinstance(item.get("text"), str):
                            self._callback(
                                "progress",
                                cast(str, item["text"]),
                                event_type="tool_result",
                                tool_call_id=(
                                    call_id if isinstance(call_id, str) and call_id else None
                                ),
                            )


def decode_claude_stream_json(output: str) -> str:
    """Extract the final assistant response from Claude's stream-json output."""
    final_response: str | None = None
    for line in output.split("\n"):
        if not line.strip():
            continue
        try:
            decoded: object = loads_json(line)
        except json.JSONDecodeError as exc:
            raise ValueError("Claude returned malformed stream-json") from exc
        if not isinstance(decoded, dict):
            raise ValueError("Claude stream-json event was not an object")
        event = cast(dict[str, object], decoded)
        if event.get("type") == "result":
            result = event.get("result")
            if not isinstance(result, str):
                raise ValueError("Claude final stream-json result was not text")
            final_response = result
    if final_response is None:
        raise ValueError("Claude stream-json output had no final result")
    return final_response


class CodexOutputStream:
    """Read Codex JSONL activity without forwarding its protocol records."""

    def __init__(self, callback: AgentOutputCallback) -> None:
        self._callback = callback
        self._buffer = ""
        self._command_outputs: dict[str, str] = {}
        self._pending_agent_message: str | None = None

    def feed(self, chunk: str) -> None:
        """Consume complete JSONL records from one process chunk."""
        self._buffer += chunk
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._consume_line(line)

    def finish(self) -> None:
        """Consume a final unterminated JSONL record, if present."""
        if self._buffer:
            self._consume_line(self._buffer)
            self._buffer = ""

    def _consume_line(self, line: str) -> None:
        if not line.strip():
            return
        try:
            decoded: object = loads_json(line)
        except json.JSONDecodeError:
            return
        if not isinstance(decoded, dict):
            return
        event = cast(dict[str, object], decoded)
        event_type = event.get("type")
        if not isinstance(event_type, str):
            return
        item = event.get("item")
        if not isinstance(item, dict):
            return
        item = cast(dict[str, object], item)
        item_type = item.get("type")
        if not isinstance(item_type, str):
            return
        if event_type == "item.completed" and item_type == "agent_message":
            self._emit_pending_agent_message()
            text = item.get("text")
            self._pending_agent_message = text if isinstance(text, str) else None
            return
        if event_type in {"item.started", "item.updated", "item.completed"}:
            self._emit_pending_agent_message()
        if item_type == "agent_message":
            return
        if event_type == "item.started":
            self._item_started(item_type, item)
        elif event_type == "item.updated":
            self._item_updated(item_type, item)
        elif event_type == "item.completed":
            self._item_completed(item_type, item)

    def _item_started(self, item_type: str, item: dict[str, object]) -> None:
        call_id = item.get("id")
        tool_call_id = call_id if isinstance(call_id, str) else None
        if item_type == "command_execution":
            command = item.get("command")
            if isinstance(command, str) and command:
                self._callback(
                    "progress",
                    f"$ {command}\n",
                    event_type="tool_call",
                    tool_name=item_type,
                    tool_call_id=tool_call_id,
                )
        elif item_type == "web_search":
            title = item.get("query") or item.get("name")
            if isinstance(title, str) and title:
                self._callback(
                    "progress",
                    f"[{title}]\n",
                    event_type="tool_call",
                    tool_name=item_type,
                    tool_call_id=tool_call_id,
                )
        elif item_type == "mcp_tool_call":
            tool = item.get("tool")
            if isinstance(tool, str) and tool:
                server = item.get("server")
                title = f"{server}.{tool}" if isinstance(server, str) and server else tool
                self._callback(
                    "progress",
                    f"[{title}]\n",
                    event_type="tool_call",
                    tool_name=title,
                    tool_call_id=tool_call_id,
                )

    def _emit_pending_agent_message(self) -> None:
        text = self._pending_agent_message
        self._pending_agent_message = None
        if text:
            self._callback("progress", text, event_type="message")

    def _item_completed(self, item_type: str, item: dict[str, object]) -> None:
        if item_type == "command_execution":
            self._emit_command_output(item)
        elif item_type == "mcp_tool_call":
            self._emit_mcp_result(item)

    def _item_updated(self, item_type: str, item: dict[str, object]) -> None:
        if item_type == "command_execution":
            self._emit_command_output(item)

    def _emit_command_output(self, item: dict[str, object]) -> None:
        output = item.get("aggregated_output")
        if not isinstance(output, str) or not output:
            return
        item_id = item.get("id")
        tool_call_id = item_id if isinstance(item_id, str) else None
        if not isinstance(item_id, str) or not item_id:
            added = output
        else:
            previous = self._command_outputs.get(item_id, "")
            added = output[len(previous) :] if output.startswith(previous) else output
            self._command_outputs[item_id] = output
        if added:
            self._callback(
                "progress",
                added,
                event_type="tool_result",
                tool_name="command_execution",
                tool_call_id=tool_call_id,
            )

    def _emit_mcp_result(self, item: dict[str, object]) -> None:
        result = item.get("result")
        if not isinstance(result, dict):
            return
        server = item.get("server")
        tool = item.get("tool")
        if isinstance(tool, str) and tool:
            tool_name = f"{server}.{tool}" if isinstance(server, str) and server else tool
        else:
            tool_name = None
        item_id = item.get("id")
        tool_call_id = item_id if isinstance(item_id, str) else None
        content = cast(dict[str, object], result).get("content")
        if not isinstance(content, list):
            return
        for block in content:
            if not isinstance(block, dict):
                continue
            block = cast(dict[str, object], block)
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                self._callback(
                    "progress",
                    cast(str, block["text"]),
                    event_type="tool_result",
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                )
