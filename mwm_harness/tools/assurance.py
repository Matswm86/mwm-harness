"""AssuranceReport: the model fills the three cells the assurance card reserves for it.

The card (plan item 22) is filled from code evidence; a model may not assert PASS on safe,
correct or reliable. It MAY: fill a performant or scalable row N/A with a reason, and name the
operation row's paths (was the real app run, how do you deploy, roll back, observe). Every row
it fills is labelled model-written wherever it renders, and ``Session.assurance_handler``
re-checks the same rule server-side.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from mwm_harness.tools.base import Tool, ToolContext, ToolResult


class AssuranceReport(Tool):
    name = "AssuranceReport"
    description = (
        "Fill the model-written rows of this turn's assurance card: performant and scalable "
        "may be set to na with a reason (nothing hot changed / no scaling target), and "
        "fit for operation must be set with named paths: how the real app was run, the deploy "
        "path, the rollback path, and where logs and alerts live. Status pass is allowed on "
        "these rows only with a reason an auditor can check. You cannot fill safe, correct or "
        "reliable: those come from scans, tests and flags, not from your word. Call it once "
        "after finishing an editing turn when the card says the operation row is empty."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "quality": {"type": "string"},
                        "status": {"type": "string", "enum": ["na", "pass"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["quality", "status", "reason"],
                },
            }
        },
        "required": ["rows"],
    }
    read_only = True  # it only fills card rows in this session's memory; nothing touches disk

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        handler: Callable[[list[dict[str, Any]]], Awaitable[str]] | None = getattr(
            ctx, "assurance_handler", None
        )
        if handler is None:
            return ToolResult("this session has no assurance card to fill", True)
        rows = tool_input.get("rows")
        if not isinstance(rows, list) or not rows:
            return ToolResult("rows must be a non-empty list", True)
        try:
            report = await handler(rows)
        except ValueError as exc:
            return ToolResult(str(exc), True)
        return ToolResult(report)
