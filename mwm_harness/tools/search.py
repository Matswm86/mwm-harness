"""ToolSearch: loads the full definition of a deferred MCP tool when the model needs it.

A small-context model cannot carry the definitions of every MCP tool in each
request (146 tools were about 30,000 tokens on 2026-10-05). The session then
sends only their names, in this tool's description, and the model loads the
ones it wants. A loaded tool stays in the request for the rest of the session.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from mwm_harness.tools.base import Tool, ToolContext, ToolResult


class ToolSearch(Tool):
    name = "ToolSearch"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    '"select:NAME1,NAME2" loads those exact tools; any other text is a '
                    "keyword search over tool names and descriptions"
                ),
            },
            "max_results": {"type": "integer", "description": "keyword search only (default 5)"},
        },
        "required": ["query"],
    }
    read_only = True

    def __init__(self, deferred: Callable[[], dict[str, Tool]], loaded: set[str]) -> None:
        self._deferred = deferred
        self.loaded = loaded

    @property
    def description(self) -> str:  # type: ignore[override]
        names = sorted(self._deferred())
        return (
            "These tools exist but their definitions are not loaded. Before calling one, "
            "load it with this tool; it can be called from the next step on. "
            "Deferred tools:\n" + "\n".join(names)
        )

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        deferred = self._deferred()
        query = str(tool_input["query"]).strip()
        if query.startswith("select:"):
            wanted = [n.strip() for n in query[len("select:") :].split(",") if n.strip()]
            found = [n for n in wanted if n in deferred]
            missing = [n for n in wanted if n not in deferred]
        else:
            limit = int(tool_input.get("max_results") or 5)
            found = _rank(query, deferred)[:limit]
            missing = []
        if not found:
            return ToolResult(f"no deferred tool matches {query!r}", True)
        self.loaded.update(found)
        specs = [deferred[n].spec() for n in found]
        note = f"\nNot found: {', '.join(missing)}" if missing else ""
        return ToolResult(f"Loaded {len(found)} tools:\n{json.dumps(specs, indent=1)}{note}")


def _rank(query: str, tools: dict[str, Tool]) -> list[str]:
    words = [w for w in re.split(r"[^a-z0-9]+", query.lower()) if w]
    scored = []
    for name, tool in tools.items():
        lowered = name.lower()
        text = f"{lowered} {tool.description.lower()}"
        score = sum(3 if w in lowered else 1 for w in words if w in text)
        if score:
            scored.append((-score, name))
    return [name for _, name in sorted(scored)]
