"""Assemble one model turn from OpenAI-compatible streamed chunks.

The endpoint sends a turn as many small JSON chunks. Text arrives as string
deltas, a tool call arrives as fragments: the first fragment for a call carries
its ``id`` and ``name``, later fragments carry only more of the ``arguments``
string, and every fragment says which call it belongs to through ``index``.
With parallel tool calls the fragments of different calls can interleave.
Token usage arrives in a final chunk whose ``choices`` list is empty.

This module is pure: it takes chunk dicts and returns a finished turn, so it is
tested without a network or an API key.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """One finished tool call. ``arguments`` is None when the JSON was malformed."""

    index: int
    id: str
    name: str
    raw_arguments: str
    arguments: dict[str, Any] | None
    error: str | None = None


@dataclass
class AssembledTurn:
    text: str
    reasoning: str
    tool_calls: list[ToolCall]
    finish_reason: str | None
    usage: dict[str, Any] | None

    @property
    def cached_tokens(self) -> int | None:
        """Prompt tokens served from the provider's cache, when it reports them."""
        details = (self.usage or {}).get("prompt_tokens_details") or {}
        return details.get("cached_tokens")


@dataclass
class _PartialCall:
    id: str = ""
    name: str = ""
    arguments: list[str] = field(default_factory=list)


class StreamAssembler:
    """Feed chunks in arrival order, then call ``finish()``."""

    def __init__(self) -> None:
        self._text: list[str] = []
        self._reasoning: list[str] = []
        self._calls: dict[int, _PartialCall] = {}
        self._finish_reason: str | None = None
        self._usage: dict[str, Any] | None = None

    def feed(self, chunk: dict[str, Any]) -> None:
        if chunk.get("usage"):
            self._usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if choice.get("finish_reason"):
                self._finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                self._text.append(delta["content"])
            if delta.get("reasoning_content"):
                self._reasoning.append(delta["reasoning_content"])
            for fragment in delta.get("tool_calls") or []:
                self._feed_fragment(fragment)

    def _feed_fragment(self, fragment: dict[str, Any]) -> None:
        # A provider that omits index can only mean a single call.
        index = fragment.get("index", 0)
        call = self._calls.setdefault(index, _PartialCall())
        if fragment.get("id"):
            call.id = fragment["id"]
        function = fragment.get("function") or {}
        if function.get("name"):
            call.name = function["name"]
        if function.get("arguments"):
            call.arguments.append(function["arguments"])

    def finish(self) -> AssembledTurn:
        calls = [self._finish_call(index, self._calls[index]) for index in sorted(self._calls)]
        return AssembledTurn(
            text="".join(self._text),
            reasoning="".join(self._reasoning),
            tool_calls=calls,
            finish_reason=self._finish_reason,
            usage=self._usage,
        )

    @staticmethod
    def _finish_call(index: int, call: _PartialCall) -> ToolCall:
        raw = "".join(call.arguments)
        arguments: dict[str, Any] | None = None
        error: str | None = None
        if not call.name:
            error = "tool call has no function name"
        try:
            # A call with no arguments is legal and streams as an empty string.
            parsed = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as exc:
            error = f"arguments are not valid JSON: {exc.msg} at char {exc.pos}"
        else:
            if isinstance(parsed, dict):
                arguments = parsed
            else:
                error = f"arguments must be a JSON object, got {type(parsed).__name__}"
        return ToolCall(
            index=index,
            id=call.id,
            name=call.name,
            raw_arguments=raw,
            arguments=arguments,
            error=error,
        )


def parse_sse_line(line: str) -> dict[str, Any] | None:
    """Turn one server-sent-events line into a chunk dict.

    Returns None for blank lines, comments, the ``[DONE]`` marker and any line
    that is not a ``data:`` line.
    """
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return None
    return json.loads(payload)
