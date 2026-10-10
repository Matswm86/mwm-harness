"""The Bash tool."""

from __future__ import annotations

from typing import Any

from mwm_harness.tools.base import Tool, ToolContext, ToolResult

DEFAULT_TIMEOUT_MS = 120_000
MAX_TIMEOUT_MS = 600_000


class Bash(Tool):
    name = "Bash"
    description = (
        "Run a bash command in the project directory and return its combined output. "
        "timeout is in milliseconds (default 120000, max 600000). Use Grep and Glob "
        "for searching instead of shell grep and find. Commands run without network "
        "access (no internet, no local servers); set network: true for a command that "
        "needs it (pip install, git push, curl), and the person is asked first."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "description": {"type": "string", "description": "What the command does"},
            "timeout": {"type": "integer"},
            "network": {
                "type": "boolean",
                "description": "true if the command needs the internet or a local server",
            },
        },
        "required": ["command"],
    }

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        timeout_ms = min(int(tool_input.get("timeout") or DEFAULT_TIMEOUT_MS), MAX_TIMEOUT_MS)
        network = bool(tool_input.get("network"))
        result = await ctx.sandbox.run(tool_input["command"], ctx.cwd, timeout_ms / 1000, network)
        output = result.output.rstrip()
        if result.timed_out:
            note = f"[killed after {timeout_ms} ms timeout]"
            return ToolResult(ctx.cap(f"{output}\n{note}" if output else note), True)
        if result.exit_code != 0:
            return ToolResult(ctx.cap(f"{output}\n[exit code {result.exit_code}]".lstrip()), True)
        return ToolResult(ctx.cap(output) if output else "(no output)")
