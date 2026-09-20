"""Model providers: each yields OpenAI-style stream chunks for one request.

The loop owns the ``StreamAssembler``, not the provider. That way a cancel in
the middle of a stream still leaves the loop holding the text received so far.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

import httpx

from mwm_harness.config import ModelSpec, api_key_for
from mwm_harness.messages import Message, to_openai, tools_to_openai
from mwm_harness.streaming import parse_sse_line


class ProviderError(Exception):
    """The endpoint refused or broke the request. ``status`` is the HTTP code, if any."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class Provider(Protocol):
    def stream(
        self,
        spec: ModelSpec,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]: ...


def build_request(
    spec: ModelSpec, system: str, messages: list[Message], tools: list[dict[str, Any]]
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": spec.id,
        "messages": to_openai(system, messages, spec.echo_reasoning),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        body["tools"] = tools_to_openai(tools)
        if spec.parallel_tool_calls:
            body["parallel_tool_calls"] = True
    if spec.thinking_field:
        body[spec.thinking_field] = True
    body.update(spec.extra_body)
    return body


class OpenAICompatProvider:
    """Any endpoint that speaks ``/chat/completions`` with server-sent events."""

    def __init__(self, secrets: dict[str, str] | None = None, timeout: float = 600.0) -> None:
        self._secrets = secrets
        self._timeout = httpx.Timeout(timeout, connect=20.0)

    async def stream(
        self,
        spec: ModelSpec,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        key = api_key_for(spec, self._secrets)
        url = spec.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        body = build_request(spec, system, messages, tools)
        try:
            async with (
                httpx.AsyncClient(timeout=self._timeout) as client,
                client.stream("POST", url, headers=headers, json=body) as response,
            ):
                if response.status_code != 200:
                    detail = (await response.aread()).decode("utf-8", errors="replace")[:600]
                    raise ProviderError(
                        f"{spec.id}: HTTP {response.status_code}: {detail.replace(key, '<key>')}",
                        response.status_code,
                    )
                async for line in response.aiter_lines():
                    try:
                        chunk = parse_sse_line(line)
                    except json.JSONDecodeError as exc:
                        raise ProviderError(f"{spec.id}: unreadable stream line: {exc}") from exc
                    if chunk is not None:
                        yield chunk
        except httpx.HTTPError as exc:
            raise ProviderError(f"{spec.id}: {type(exc).__name__}: {exc}") from exc


def chunks_for(
    text: str = "",
    tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
    reasoning: str = "",
    usage: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build the chunk list one scripted turn streams. Used by tests and demos."""
    chunks: list[dict[str, Any]] = []
    if reasoning:
        chunks.append({"choices": [{"delta": {"reasoning_content": reasoning}}]})
    for start in range(0, len(text), 20):
        chunks.append({"choices": [{"delta": {"content": text[start : start + 20]}}]})
    for index, (name, arguments) in enumerate(tool_calls or []):
        raw = json.dumps(arguments)
        half = len(raw) // 2
        first = {"index": index, "id": f"call_{index}_{name}", "function": {"name": name}}
        first["function"]["arguments"] = raw[:half]
        rest = {"index": index, "function": {"arguments": raw[half:]}}
        chunks.append({"choices": [{"delta": {"tool_calls": [first]}}]})
        chunks.append({"choices": [{"delta": {"tool_calls": [rest]}}]})
    reason = "tool_calls" if tool_calls else "stop"
    chunks.append({"choices": [{"delta": {}, "finish_reason": reason}]})
    chunks.append(
        {"choices": [], "usage": usage or {"prompt_tokens": 100, "completion_tokens": 10}}
    )
    return chunks


Script = list[dict[str, Any]] | Callable[[list[Message]], list[dict[str, Any]]]


class ScriptedProvider:
    """Replays prepared turns. One script entry is consumed per request."""

    def __init__(self, turns: list[Script]) -> None:
        self._turns = list(turns)
        self.requests: list[dict[str, Any]] = []

    async def stream(
        self,
        spec: ModelSpec,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        self.requests.append(build_request(spec, system, messages, tools))
        if not self._turns:
            raise ProviderError("scripted provider ran out of turns")
        turn = self._turns.pop(0)
        for chunk in turn(messages) if callable(turn) else turn:
            yield chunk
