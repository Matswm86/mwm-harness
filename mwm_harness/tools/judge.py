"""Judge: ask Jev for a typed second opinion. Every call lands in the decision log."""

from __future__ import annotations

import json
from typing import Any

from mwm_harness.config import ConfigError
from mwm_harness.jev import JevClient, JevError, check_questions
from mwm_harness.tools.base import Tool, ToolContext, ToolResult


class Judge(Tool):
    name = "Judge"
    description = (
        "Ask Jev, a fast classifier model, typed questions about a piece of text or data "
        "(`state`). Question types: noul = probability that a yes/no question is true; "
        "choice = one option from `criteria` {option: description}; score = a level on "
        "`criteria` [level descriptions, lowest first]. Jev returns probabilities and a "
        "confidence, no text. Use it as ONE extra vote beside other evidence (classifying "
        "research or knowledge-base items, grading a described setup), never as the only "
        "basis for a decision. `state` leaves this machine: no names, account numbers or "
        "keys. Each call is logged with an id; report the id so the real result can be "
        "recorded later and Jev's hit rate tracked."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "state": {"type": "string", "description": "the text or data to judge"},
            "questions": {
                "type": "object",
                "description": "{name: {type: noul|choice|score, instructions, criteria}}",
            },
            "domain": {"type": "string", "description": "research, brain, trading or other"},
            "label": {"type": "string", "description": "what is judged, e.g. a strategy name"},
        },
        "required": ["state", "questions"],
    }

    def __init__(self, client: JevClient | None = None) -> None:
        self._client = client

    def refusal(self, tool_input: dict[str, Any], ctx: ToolContext) -> str | None:
        return check_questions(tool_input.get("questions"))

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        client = self._client or JevClient()
        try:
            verdict = await client.ask(
                tool_input["state"],
                tool_input["questions"],
                domain=str(tool_input.get("domain") or "other"),
                label=str(tool_input.get("label") or ""),
                caller="harness:Judge",
            )
        except (JevError, ConfigError) as exc:
            return ToolResult(f"Jev gave no verdict ({exc}). Decide without it.", True)
        body = {
            "decision_id": verdict.id,
            "decisions": verdict.decisions(),
            "answers": verdict.answers,
            "latency_ms": verdict.latency_ms,
            "logged_to": str(client.log.path),
        }
        return ToolResult(json.dumps(body, ensure_ascii=False, indent=1))
