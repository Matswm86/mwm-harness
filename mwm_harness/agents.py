"""Subagents: ``agents/<name>.md`` files and the ``Task`` tool that runs one.

An agent file is frontmatter (``name``, ``description``, optional ``tools`` and
``model``) followed by the agent's system prompt. A subagent is a child session:
its own history and transcript, the agent's prompt, a filtered tool set, the
same permission rules and approver as the parent, and no ``Task`` tool of its
own. Only its final answer returns to the parent's context.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from mwm_harness.config import ModelSpec, workspace_root
from mwm_harness.skills import split_frontmatter
from mwm_harness.tools.base import Tool, ToolContext, ToolResult

# Claude Code agent files name Claude tiers; the settings map them to configured models.
INHERIT = ("", "inherit")
# Agents that ship with the harness (the cross-family critic).
BUILTIN_AGENTS = Path(__file__).parent / "builtin_agents"


@dataclass
class Agent:
    name: str
    description: str
    prompt: str
    path: Path
    tools: list[str] = field(default_factory=list)  # empty = every tool of the parent
    model: str = ""

    def allows(self, tool_name: str) -> bool:
        if tool_name == "Task":
            return False  # subagents do not start subagents
        if not self.tools or "*" in self.tools:
            return True
        return any(fnmatchcase(tool_name, pattern) for pattern in self.tools)


def agent_roots(cwd: Path) -> list[Path]:
    return [
        cwd / ".claude" / "agents",
        workspace_root() / ".claude" / "agents",
        Path.home() / ".claude" / "agents",
        BUILTIN_AGENTS,  # last, so an agent file of the same name shadows a built-in
    ]


def load_agents(roots: list[Path]) -> dict[str, Agent]:
    """The first root that defines a name wins."""
    agents: dict[str, Agent] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.md")):
            try:
                fields, body = split_frontmatter(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                continue
            name = fields.get("name") or path.stem
            if not body.strip() or name in agents:
                continue
            tools = [t.strip() for t in fields.get("tools", "").strip("[]").split(",") if t.strip()]
            agents[name] = Agent(
                name,
                fields.get("description", ""),
                body.strip(),
                path,
                tools,
                fields.get("model", "").strip(),
            )
    return agents


def resolve_model(
    agent: Agent, parent: ModelSpec, models: dict[str, ModelSpec], mapping: dict[str, str]
) -> tuple[ModelSpec, str]:
    """The model an agent runs on, and a note when its wish could not be met."""
    wanted = agent.model
    if wanted in INHERIT:
        return parent, ""
    target = mapping.get(wanted, wanted)
    if target in models:
        return models[target], ""
    return parent, f"agent {agent.name} asks for model {wanted!r}, which is not configured"


Runner = Callable[[str, str], Awaitable[ToolResult]]


class TaskTool(Tool):
    name = "Task"
    input_schema = {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "3 to 5 words naming the task"},
            "prompt": {"type": "string", "description": "Everything the agent needs to know"},
            "subagent_type": {"type": "string", "description": "Agent name from the list"},
        },
        "required": ["prompt", "subagent_type"],
    }
    read_only = True  # the agent's own tool calls pass the permission rules one by one

    def __init__(self, agents: dict[str, Agent], runner: Runner) -> None:
        self.agents = agents
        self.runner = runner
        rows = [f"- {a.name}: {a.description[:220]}" for a in agents.values()]
        self.description = (  # type: ignore[misc]
            "Hand a self-contained task to a subagent. It starts with no knowledge of this "
            "conversation, works with its own tools and returns one final report. Agents:\n"
            + "\n".join(rows)
        )

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        name = str(tool_input["subagent_type"])
        if name not in self.agents:
            return ToolResult(f"unknown agent {name}; available: {', '.join(self.agents)}", True)
        prompt = str(tool_input["prompt"]).strip()
        if not prompt:
            return ToolResult("the prompt is empty", True)
        return await self.runner(name, prompt)
