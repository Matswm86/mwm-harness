"""Load the agent, skill and command files installed on this machine and report what resolves.

For every agent: the model it asks for and the tool names in its list that no
harness tool or configured MCP server provides. No model is called.

    .venv/bin/python spike/real_agents_check.py
"""

from __future__ import annotations

import asyncio
import sys
from collections import Counter
from fnmatch import fnmatchcase
from pathlib import Path

from mwm_harness.agents import agent_roots, load_agents
from mwm_harness.config import load_settings
from mwm_harness.mcp_client import McpManager, default_mcp_files, load_server_configs
from mwm_harness.skills import command_roots, load_commands, load_skills, skill_roots
from mwm_harness.tools import default_tools


async def main() -> int:
    cwd = Path.cwd()
    settings = load_settings()
    agents = load_agents(agent_roots(cwd))
    skills = load_skills(skill_roots(cwd, settings.skill_dirs))
    commands = load_commands(command_roots(cwd))
    manager = McpManager(load_server_configs(default_mcp_files(cwd)))
    names = set(default_tools()) | {"Skill", "Task"} | set(await manager.discover())
    await manager.close()

    print(
        f"{len(agents)} agents, {len(skills)} skills, {len(commands)} command files, "
        f"{len(names)} tool names (MCP included)"
    )
    print("models asked for:", dict(Counter(a.model or "inherit" for a in agents.values())))
    unmapped = {a.model for a in agents.values() if a.model and a.model != "inherit"}
    unmapped -= set(settings.agent_models)
    if unmapped:
        print(
            f"no agent_models entry for: {sorted(unmapped)} (those agents run on the parent model)"
        )
    missing = Counter()
    for agent in agents.values():
        for pattern in agent.tools:
            if pattern != "*" and not any(fnmatchcase(name, pattern) for name in names):
                missing[pattern] += 1
    for pattern, count in missing.most_common():
        print(f"  tool not available here: {pattern} (listed by {count} agents)")
    return 0 if agents else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
