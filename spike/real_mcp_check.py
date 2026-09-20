"""Phase 4 closing check against the MCP servers configured on this machine.

1. Every server in the workspace ``.mcp.json`` is started and asked for its tools
   (the cache is bypassed).
2. A scripted model calls ``search_knowledge`` through a real ``Session``, so the
   call takes the same path a live model's call would: schema check, permission
   rules, the tool, the transcript.

No model endpoint and no API key are used. Exit 0 = both steps passed.

    .venv/bin/python spike/real_mcp_check.py ["search text"]
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from pathlib import Path

from mwm_harness.config import ModelSpec, Settings
from mwm_harness.loop import Session
from mwm_harness.mcp_client import McpManager, default_mcp_files, load_server_configs
from mwm_harness.providers import ScriptedProvider, chunks_for

MODEL = ModelSpec(id="scripted", base_url="http://unused.invalid/v1", key_env="UNUSED")
LONGEST_TOOL_NAME = 64  # the OpenAI wire format allows no more


class ApproveAndReport:
    """Stands in for the person at the prompt, and shows that the prompt was reached."""

    async def ask(self, tool_name: str, tool_input: dict, reason: str) -> bool:
        print(f"\napproval asked for {tool_name} ({reason}): answered yes")
        return True


async def main(query: str) -> int:
    configs = load_server_configs(default_mcp_files(Path.cwd()))
    if not configs:
        print("no MCP servers configured")
        return 1
    with tempfile.TemporaryDirectory() as folder:
        scratch = Path(folder)
        manager = McpManager(configs, log_dir=scratch / "logs", cache_dir=scratch / "cache")
        started = time.monotonic()
        tools = await manager.discover(refresh=True)
        print(
            f"listed {len(tools)} tools from {len(configs)} servers in "
            f"{time.monotonic() - started:.1f} s"
        )
        for row in manager.status():
            print(row)
        failed = [name for name, server in manager.servers.items() if server.error]
        empty = [name for name in configs if not manager.listings.get(name)]
        too_long = [name for name in tools if len(name) > LONGEST_TOOL_NAME]
        if too_long:
            print(f"tool names over {LONGEST_TOOL_NAME} characters: {too_long}")

        target = next((name for name in tools if name.endswith("__search_knowledge")), "")
        answered = False
        if target:
            turns = [chunks_for(tool_calls=[(target, {"query": query})]), chunks_for("done")]
            session = Session(
                cwd=scratch,
                model=MODEL,
                provider=ScriptedProvider(turns),
                settings=Settings(sandbox="off"),
                approver=ApproveAndReport(),
                hook_settings=[],
                sessions_dir=scratch / "sessions",
                system_prompt="check",
                mcp=manager,
                skills={},
                commands={},
                agents={},
            )
            await session.start()
            started = time.monotonic()
            await session.send("search")
            results = [
                block
                for message in session.messages
                for block in message.blocks()
                if block.get("type") == "tool_result"
            ]
            await session.close()
            if results:
                content = str(results[0]["content"])
                answered = not results[0].get("is_error") and len(content) > 50
                print(
                    f"\n{target}({query!r}) answered in {time.monotonic() - started:.1f} s, "
                    f"{len(content):,} characters, error={bool(results[0].get('is_error'))}"
                )
                print(content[:700])
        else:
            await manager.close()
            print("no search_knowledge tool found")

    print(
        f"\nservers failed: {failed or 'none'} | servers with no tools: {empty or 'none'} | "
        f"search_knowledge answered: {answered}"
    )
    return 0 if not failed and not empty and not too_long and answered else 1


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "opening range breakout stop placement"
    sys.exit(asyncio.run(main(text)))
