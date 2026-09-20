"""ExitPlanMode: the model hands its plan to the person and waits for the answer."""

from __future__ import annotations

from typing import Any

from mwm_harness.tools.base import Tool, ToolContext, ToolResult


class ExitPlanMode(Tool):
    name = "ExitPlanMode"
    description = (
        "In plan mode, present the finished plan (markdown) to the person for approval. "
        "Approved: plan mode ends and editing tools work again. Rejected: stay in plan mode "
        "and revise. Only call this when the plan is complete."
    )
    input_schema = {
        "type": "object",
        "properties": {"plan": {"type": "string", "description": "The plan, in markdown"}},
        "required": ["plan"],
    }
    read_only = True  # the approval happens inside the plan handler

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        plan = str(tool_input["plan"]).strip()
        if not plan:
            return ToolResult("the plan is empty", True)
        if ctx.plan_handler is None:
            return ToolResult("this session has no way to show a plan", True)
        if await ctx.plan_handler(plan):
            return ToolResult("The person approved the plan. Plan mode is off; carry it out.")
        return ToolResult(
            "The person rejected the plan. Plan mode stays on: ask what to change, or revise.", True
        )
