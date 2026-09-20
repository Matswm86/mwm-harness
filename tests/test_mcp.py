"""The MCP client against a fake stdio server and a mocked HTTP endpoint."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import pytest
from mwm_harness.mcp_client import (
    McpError,
    McpManager,
    McpServer,
    ServerConfig,
    load_server_configs,
    render_result,
)
from mwm_harness.sandbox import Sandbox
from mwm_harness.tools.base import ToolContext

FAKE = Path(__file__).parent / "fake_mcp_server.py"


def run(coro):
    return asyncio.run(coro)


def stdio_config(timeout: float = 20.0) -> ServerConfig:
    return ServerConfig("fake", "stdio", command=sys.executable, args=[str(FAKE)], timeout=timeout)


def context(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, scratch=tmp_path / "scratch", sandbox=Sandbox("off", []))


def test_load_server_configs_reads_both_transports_and_timeout(tmp_path):
    first = tmp_path / ".mcp.json"
    first.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "brain": {"type": "http", "url": "http://127.0.0.1:1/mcp"},
                    "video": {"command": "python3", "args": ["s.py"], "timeout": 600000},
                }
            }
        )
    )
    second = tmp_path / "other.json"
    second.write_text(json.dumps({"mcpServers": {"brain": {"command": "shadowed"}}}))
    configs = load_server_configs([first, second, tmp_path / "missing.json"])
    assert configs["brain"].transport == "http"
    assert configs["video"].transport == "stdio"
    assert configs["video"].timeout == 600.0
    assert configs["video"].cwd == tmp_path


def test_stdio_lists_paginated_tools_and_calls_one(tmp_path):
    async def scenario():
        manager = McpManager({"fake": stdio_config()}, cache_dir=tmp_path / "cache")
        try:
            tools = await manager.discover()
            assert sorted(tools) == [
                "mcp__fake__delete_everything",
                "mcp__fake__echo",
                "mcp__fake__fail",
                "mcp__fake__slow",
            ]
            assert tools["mcp__fake__echo"].read_only is True
            assert tools["mcp__fake__delete_everything"].read_only is False
            result = await tools["mcp__fake__echo"].run({"text": "hei"}, context(tmp_path))
            assert (result.content, result.is_error) == ("echo: hei", False)
            failed = await tools["mcp__fake__fail"].run({}, context(tmp_path))
            assert (failed.content, failed.is_error) == ("it broke", True)
            assert manager.servers["fake"].instructions == "fake instructions"
        finally:
            await manager.close()

    run(scenario())


def test_listing_comes_from_cache_without_starting_the_server(tmp_path):
    async def scenario():
        first = McpManager({"fake": stdio_config()}, cache_dir=tmp_path / "cache")
        await first.discover()
        await first.close()

        second = McpManager({"fake": stdio_config()}, cache_dir=tmp_path / "cache")
        tools = await second.discover()
        assert len(tools) == 4
        assert "fake" in second.from_cache
        assert second.servers["fake"].running is False
        # The first call starts the process.
        result = await tools["mcp__fake__echo"].run({"text": "x"}, context(tmp_path))
        assert result.content == "echo: x"
        assert second.servers["fake"].running is True
        await second.close()
        assert second.servers["fake"].running is False

    run(scenario())


def test_call_timeout_becomes_a_tool_error(tmp_path):
    async def scenario():
        manager = McpManager({"fake": stdio_config(timeout=0.5)}, cache_dir=tmp_path / "cache")
        try:
            tools = await manager.discover()
            started = time.monotonic()
            result = await tools["mcp__fake__slow"].run({"seconds": 5}, context(tmp_path))
            assert result.is_error
            assert "no answer" in result.content
            assert time.monotonic() - started < 3
        finally:
            await manager.close()

    run(scenario())


def test_unknown_command_is_reported_not_raised(tmp_path):
    async def scenario():
        config = ServerConfig("ghost", "stdio", command="/nonexistent/mcp-server")
        manager = McpManager({"ghost": config}, cache_dir=tmp_path / "cache")
        tools = await manager.discover()
        assert tools == {}
        assert "cannot start" in manager.servers["ghost"].error
        assert "FAILED" in manager.status()[0]
        await manager.close()

    run(scenario())


def test_jsonrpc_error_raises(tmp_path):
    async def scenario():
        server = McpServer(stdio_config())
        try:
            with pytest.raises(McpError, match="unknown tool"):
                await server.call_tool("nope", {})
        finally:
            await server.close()

    run(scenario())


def test_http_json_and_event_stream_answers(tmp_path):
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(
            {"method": body.get("method"), "session": request.headers.get("mcp-session-id")}
        )
        method = body.get("method")
        if method == "initialize":
            result = {"protocolVersion": "2025-06-18", "serverInfo": {"name": "h"}}
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": body["id"], "result": result},
                headers={"Mcp-Session-Id": "abc"},
            )
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "tools/list":
            tools = [{"name": "search_knowledge", "inputSchema": {"type": "object"}}]
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": body["id"], "result": {"tools": tools}}
            )
        answer = {
            "jsonrpc": "2.0",
            "id": body["id"],
            "result": {"content": [{"type": "text", "text": "found it"}]},
        }
        progress = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {}}
        stream = f"event: message\ndata: {json.dumps(progress)}\n\nevent: message\ndata: {json.dumps(answer)}\n\n"
        return httpx.Response(200, text=stream, headers={"Content-Type": "text/event-stream"})

    async def scenario():
        config = ServerConfig("brain", "http", url="http://brain.test/mcp")
        manager = McpManager(
            {"brain": config},
            cache_dir=tmp_path / "cache",
            http_transport=httpx.MockTransport(handler),
        )
        tools = await manager.discover()
        result = await tools["mcp__brain__search_knowledge"].run({"query": "q"}, context(tmp_path))
        assert result.content == "found it"
        await manager.close()

    run(scenario())
    assert seen[0]["session"] is None
    assert all(entry["session"] == "abc" for entry in seen[1:])


def test_http_error_status_is_a_tool_error(tmp_path):
    async def scenario():
        config = ServerConfig("down", "http", url="http://down.test/mcp")
        transport = httpx.MockTransport(lambda request: httpx.Response(502, text="bad gateway"))
        manager = McpManager(
            {"down": config}, cache_dir=tmp_path / "cache", http_transport=transport
        )
        assert await manager.discover() == {}
        assert "HTTP 502" in manager.servers["down"].error
        await manager.close()

    run(scenario())


def test_render_result_saves_images_and_falls_back_to_structured(tmp_path):
    text = render_result(
        {"content": [{"type": "image", "mimeType": "image/png", "data": "aGVsbG8="}]}, tmp_path
    )
    assert "image/png, 5 bytes" in text
    saved = list(tmp_path.glob("mcp-*.png"))
    assert saved and saved[0].read_bytes() == b"hello"
    assert '"a": 1' in render_result({"structuredContent": {"a": 1}}, tmp_path)
    assert render_result({}, tmp_path) == "(empty result)"


def test_session_runs_an_mcp_tool_behind_the_permission_rules(make_session, tmp_path):
    from conftest import FixedApprover
    from mwm_harness.providers import chunks_for

    approver = FixedApprover(False)
    turns = [
        chunks_for(
            tool_calls=[
                ("mcp__fake__echo", {"text": "hei"}),
                ("mcp__fake__delete_everything", {}),
                ("mcp__fake__slow", {"seconds": 0}),
            ]
        ),
        chunks_for("done"),
    ]
    session, _, provider = make_session(turns, approver=approver)
    session.mcp = McpManager({"fake": stdio_config()}, cache_dir=tmp_path / "cache")
    session.permissions.allow_patterns = ("mcp__fake__sl*",)

    async def scenario():
        await session.start()
        ended = await session.send("go")
        await session.close()
        return ended

    assert run(scenario()).reason == "done"
    offered = [spec["function"]["name"] for spec in provider.requests[0]["tools"]]
    assert "mcp__fake__echo" in offered and "WebFetch" in offered
    results = [
        block
        for message in session.messages
        for block in message.blocks()
        if block.get("type") == "tool_result"
    ]
    assert results[0]["content"] == "echo: hei"  # readOnlyHint: runs unasked
    assert results[1]["is_error"] and "declined" in results[1]["content"]
    assert results[2]["content"] == "woke"  # allowed by an mcp_allow pattern
    assert [name for name, _ in approver.asked] == ["mcp__fake__delete_everything"]
    assert session.mcp.servers["fake"].running is False
