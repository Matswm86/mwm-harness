"""The conversation model: Anthropic-style content blocks.

Everything inside the harness (history, transcript, hooks) speaks this shape:

    {"role": "user" | "assistant", "content": str | [block, ...]}

with blocks ``text``, ``thinking``, ``tool_use`` and ``tool_result``. The
existing Stop gates parse exactly this shape, so it is the storage format.
Conversion to the OpenAI wire format happens here and only for the request
that leaves the process.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from mwm_harness.streaming import AssembledTurn

Block = dict[str, Any]


@dataclass
class Message:
    role: str
    content: str | list[Block]
    # Meta messages reach the model but are not something the person typed
    # (hook feedback, injected context). Gates skip them when looking for the prompt.
    is_meta: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def blocks(self) -> list[Block]:
        if isinstance(self.content, str):
            return [text_block(self.content)] if self.content else []
        return self.content

    def text(self) -> str:
        return "\n".join(
            b.get("text") or "" for b in self.blocks() if b.get("type") == "text"
        ).strip()

    def tool_uses(self) -> list[Block]:
        return [b for b in self.blocks() if b.get("type") == "tool_use"]


def text_block(text: str) -> Block:
    return {"type": "text", "text": text}


def tool_use_block(name: str, tool_input: dict[str, Any], call_id: str = "") -> Block:
    return {
        "type": "tool_use",
        "id": call_id or f"toolu_{uuid.uuid4().hex[:24]}",
        "name": name,
        "input": tool_input,
    }


def tool_result_block(tool_use_id: str, content: str, is_error: bool = False) -> Block:
    block: Block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        block["is_error"] = True
    return block


def assistant_from_turn(turn: AssembledTurn) -> tuple[Message, dict[str, str]]:
    """Build the assistant message for a streamed turn.

    Returns the message and a map ``tool_use_id -> error`` for calls whose
    arguments did not parse. Those calls are kept with an empty input so the
    history stays valid; the loop answers them with an error result instead of
    running anything.
    """
    blocks: list[Block] = []
    errors: dict[str, str] = {}
    if turn.reasoning:
        blocks.append({"type": "thinking", "thinking": turn.reasoning})
    if turn.text:
        blocks.append(text_block(turn.text))
    for call in turn.tool_calls:
        block = tool_use_block(call.name or "unknown_tool", call.arguments or {}, call.id)
        blocks.append(block)
        if call.error:
            errors[block["id"]] = f"{call.error}. Raw arguments: {call.raw_arguments[:500]}"
    return Message("assistant", blocks), errors


def to_openai(
    system: str, messages: list[Message], echo_reasoning: bool = False
) -> list[dict[str, Any]]:
    """Convert block history into OpenAI chat messages.

    Tool results become ``role: tool`` messages and must directly follow the
    assistant message that made the calls, so they are emitted before any text
    that shares their user message. Adjacent plain user messages are merged
    because some endpoints reject two user turns in a row.
    """
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for message in messages:
        if message.role == "assistant":
            out.append(_assistant_to_openai(message, echo_reasoning))
            continue
        texts: list[str] = []
        for block in message.blocks():
            if block.get("type") == "tool_result":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": _result_text(block.get("content")),
                    }
                )
            elif block.get("type") == "text" and block.get("text"):
                texts.append(block["text"])
        if texts:
            joined = "\n\n".join(texts)
            if out and out[-1]["role"] == "user":
                out[-1]["content"] += "\n\n" + joined
            else:
                out.append({"role": "user", "content": joined})
    return out


def _assistant_to_openai(message: Message, echo_reasoning: bool) -> dict[str, Any]:
    entry: dict[str, Any] = {"role": "assistant", "content": message.text() or None}
    calls = [
        {
            "id": block["id"],
            "type": "function",
            "function": {
                "name": block["name"],
                "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
            },
        }
        for block in message.tool_uses()
    ]
    if calls:
        entry["tool_calls"] = calls
    if echo_reasoning:
        reasoning = "".join(
            b.get("thinking") or "" for b in message.blocks() if b.get("type") == "thinking"
        )
        if reasoning:
            entry["reasoning_content"] = reasoning
    return entry


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text") or "" for b in content if isinstance(b, dict))
    return "" if content is None else str(content)


def tools_to_openai(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``specs`` are ``{"name", "description", "input_schema"}`` dicts."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["input_schema"],
            },
        }
        for spec in specs
    ]
