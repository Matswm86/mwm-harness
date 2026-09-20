"""TodoWrite: the task list the front-ends render live."""

from __future__ import annotations

from typing import Any

from mwm_harness.tools.base import Tool, ToolContext, ToolResult

STATUSES = ("pending", "in_progress", "completed")


class TodoWrite(Tool):
    name = "TodoWrite"
    description = (
        "Replace the session task list. Send the full list every time. Each item has "
        "content, status (pending, in_progress, completed) and activeForm. "
        "Keep exactly one item in_progress while working."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {"type": "string", "enum": list(STATUSES)},
                        "activeForm": {"type": "string"},
                    },
                    "required": ["content", "status"],
                },
            }
        },
        "required": ["todos"],
    }
    read_only = True

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        todos = tool_input["todos"]
        for item in todos:
            if not isinstance(item, dict) or not item.get("content"):
                return ToolResult("every todo needs a content string", True)
            if item.get("status") not in STATUSES:
                return ToolResult(f"status must be one of {STATUSES}", True)
        ctx.todos[:] = todos
        active = sum(1 for item in todos if item["status"] == "in_progress")
        done = sum(1 for item in todos if item["status"] == "completed")
        return ToolResult(f"task list saved: {len(todos)} items, {done} done, {active} in progress")
