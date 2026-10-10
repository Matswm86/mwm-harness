"""Handoffs rail panel: the newest rolling handoffs, memory search and read-only file view.

Handoffs are ``memory/handoff-*.md`` files under the workspace. Search goes to the ``memory-archive``
MCP server (one shared service on the laptop, listed in the workspace ``.mcp.json``), which ranks dated
sections and returns them as JSON. Nothing here writes to memory.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import workspace_root
from .mcp_client import McpError, McpServer, default_mcp_files, load_server_configs

SERVER = "memory-archive"
HEADING = re.compile(r"^## (.+)$", re.MULTILINE)
DATE = re.compile(r"20\d\d-\d\d-\d\d")


class OutsideMemory(Exception):
    pass


def memory_dir() -> Path:
    return workspace_root() / "memory"


def _frontmatter_value(text: str, key: str) -> str:
    m = re.search(rf"^\s*{key}:\s*(.+)$", text[:1500], re.MULTILINE)
    return m.group(1).strip().strip("'\"") if m else ""


def list_handoffs(limit: int = 40, root: Path | None = None) -> list[dict[str, Any]]:
    """Newest-modified handoffs first, each with its top section heading (rolling handoffs put the newest entry on top)."""
    root = root or memory_dir()
    files = sorted(root.glob("handoff-*.md"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    out = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        top = HEADING.search(text)
        heading = top.group(1).strip() if top else ""
        date = DATE.search(heading)
        out.append(
            {
                "path": path.name,
                "name": path.stem.removeprefix("handoff-"),
                "description": _frontmatter_value(text, "description"),
                "latest": heading,
                "date": date.group(0) if date else "",
                "mtime": path.stat().st_mtime,
            }
        )
    return out


def read_memory(relative: str, root: Path | None = None) -> dict[str, Any]:
    root = (root or memory_dir()).resolve()
    target = (root / relative.removeprefix("memory/")).resolve()
    if root not in target.parents or target.suffix != ".md":
        raise OutsideMemory(relative)
    if not target.is_file():
        return {"path": f"memory/{relative}", "error": "no such file"}
    return {
        "path": f"memory/{target.relative_to(root)}",
        "content": target.read_text(encoding="utf-8", errors="replace"),
    }


class MemorySearch:
    """Lazy client for the memory-archive MCP server."""

    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd
        self.server: McpServer | None = None

    async def search(self, query: str, k: int = 8) -> list[dict[str, Any]]:
        if self.server is None:
            configs = load_server_configs(default_mcp_files(self.cwd))
            if SERVER not in configs:
                raise McpError(f"{SERVER} is not in .mcp.json")
            self.server = McpServer(configs[SERVER])
        result = await self.server.call_tool(
            "search_memory", {"query": query, "k": k, "as_json": True}
        )
        text = "".join(
            b.get("text", "") for b in result.get("content", []) if b.get("type") == "text"
        )
        if result.get("isError"):
            raise McpError(text or "search_memory failed")
        return json.loads(text)

    async def close(self) -> None:
        if self.server is not None:
            await self.server.close()
            self.server = None
