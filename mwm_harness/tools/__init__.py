"""Built-in tools. Names match Claude Code's so existing hook matchers keep working."""

from mwm_harness.tools.base import Tool, ToolContext, ToolResult
from mwm_harness.tools.files import Edit, Glob, Grep, Read, Write
from mwm_harness.tools.shell import Bash
from mwm_harness.tools.todo import TodoWrite
from mwm_harness.tools.web import WebFetch, WebSearch


def default_tools(web_allow_private: bool = False) -> dict[str, Tool]:
    tools = [
        Bash(),
        Read(),
        Write(),
        Edit(),
        Glob(),
        Grep(),
        TodoWrite(),
        WebFetch(web_allow_private),
        WebSearch(),
    ]
    return {tool.name: tool for tool in tools}


__all__ = ["Tool", "ToolContext", "ToolResult", "default_tools"]
